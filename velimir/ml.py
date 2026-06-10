import copy
import logging
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .ml_loader import (
    MeterClassRegistry,
    get_meter_weights,
    get_loader,
)
from .nlp import PartOfSpeech


class SharedEncoder(nn.Module):
    def __init__(self, hidden=128, pos_emb_dim=8):
        super().__init__()

        num_pos_classes = len(PartOfSpeech) + 1

        self.pos_emb = nn.Embedding(num_pos_classes, pos_emb_dim, padding_idx=0)

        self.line_encoder = nn.LSTM(
            input_size=2 + pos_emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
        )
        self.line_attn = nn.Linear(hidden * 2, 1)

        self.stanza_encoder = nn.LSTM(
            input_size=hidden * 2,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
        )

    def forward(self, accent_input, pos_input, line_counts=None):
        total_N, T, _ = accent_input.shape

        syllable_mask = (accent_input != -1).any(dim=-1)
        lengths = syllable_mask.sum(dim=1).to(dtype=torch.int64, device="cpu")

        x = accent_input.masked_fill(~syllable_mask.unsqueeze(-1), 0.0)
        pos_emb = self.pos_emb(pos_input + 1)
        x = torch.cat([x, pos_emb], dim=-1)

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.line_encoder(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)

        scores = self.line_attn(out).squeeze(-1)
        scores = scores.masked_fill(~syllable_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        line_vecs = (out * weights.unsqueeze(-1)).sum(dim=1)

        if line_counts is None:
            stanza_input = line_vecs.unsqueeze(0)
            stanza_out, _ = self.stanza_encoder(stanza_input)
            context = stanza_out.squeeze(0)
        else:
            line_chunks = list(torch.split(line_vecs, line_counts.tolist()))
            max_N = max(line_counts).item()

            padded = pad_sequence(line_chunks, batch_first=True)

            stanza_lengths = line_counts.to(dtype=torch.int64, device="cpu")
            packed = pack_padded_sequence(
                padded, stanza_lengths, batch_first=True, enforce_sorted=False
            )
            stanza_out, _ = self.stanza_encoder(packed)
            stanza_out, _ = pad_packed_sequence(
                stanza_out, batch_first=True, total_length=max_N
            )

            context = torch.cat(
                [stanza_out[i, :lc] for i, lc in enumerate(line_counts)], dim=0
            )

        return out, context


class MeterModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_classes = MeterClassRegistry.num()

        self.encoder = SharedEncoder()
        self.meter_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, accent_input, pos_input, line_counts=None):
        _, context = self.encoder(accent_input, pos_input, line_counts)
        return self.meter_head(context)


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_classes = MeterClassRegistry.num()
        meter_emb_dim = 16

        self.encoder = SharedEncoder()
        self.meter_emb = nn.Embedding(num_classes, meter_emb_dim)
        self.accent_head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden * 2 + meter_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, accent_input, pos_input, meter_target, line_counts=None):
        T = accent_input.shape[1]
        out, context = self.encoder(accent_input, pos_input, line_counts)

        meter_emb = self.meter_emb(meter_target)
        meter_emb_T = meter_emb.unsqueeze(1).expand(-1, T, -1)
        context_T = context.unsqueeze(1).expand(-1, T, -1)

        accent_x = torch.cat([out, context_T, meter_emb_T], dim=-1)
        return self.accent_head(accent_x).squeeze(-1)


def meter_forward_loss(model, batch, line_counts, loss_fn, device):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    pos_input = batch.pos_input.to(device, non_blocking=True)
    meter_target = batch.meter_target.to(device, non_blocking=True)
    line_counts = line_counts.to(device)

    meter_logits = model(accent_input, pos_input, line_counts)
    return loss_fn(meter_logits, meter_target)


def train_meter(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for batch, line_counts in loader:
        optimizer.zero_grad()

        loss = meter_forward_loss(model, batch, line_counts, loss_fn, device)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Meter model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_meter(model, loader, device):
    model.eval()
    total_loss = 0.0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch, line_counts in loader:
            loss = meter_forward_loss(model, batch, line_counts, loss_fn, device)
            total_loss += loss.item()

    return total_loss / len(loader)


def accent_forward_loss(model, batch, line_counts, loss_fn, device):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    pos_input = batch.pos_input.to(device, non_blocking=True)
    meter_target = batch.meter_target.to(device, non_blocking=True)
    accent_target = batch.accent_target.to(device, non_blocking=True)
    line_counts = line_counts.to(device)

    accent_logits = model(accent_input, pos_input, meter_target, line_counts)

    accent_mask = accent_target != -1
    return loss_fn(accent_logits[accent_mask], accent_target[accent_mask])


def train_accent(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    loss_fn = nn.BCEWithLogitsLoss()

    for batch, line_counts in loader:
        optimizer.zero_grad()

        loss = accent_forward_loss(model, batch, line_counts, loss_fn, device)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Accent model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_accent(model, loader, device):
    model.eval()
    total_loss = 0.0

    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch, line_counts in loader:
            loss = accent_forward_loss(model, batch, line_counts, loss_fn, device)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model, train_func, eval_func, scheduler, max_epochs, patience):
    best_validation_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_func()
        validation_loss = eval_func()
        scheduler.step(validation_loss)

        logging.info(
            f"Epoch {epoch} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )

        if validation_loss + 1e-5 < best_validation_loss:
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_loss = validation_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered at epoch %d", epoch)
                break

    return best_state_dict


def train_models(
    train_set,
    validation_set,
    max_epochs=100,
    patience=6,
    batch_size=512,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for training", device)

    train_loader = get_loader(
        train_set,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    validation_loader = get_loader(
        validation_set,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    accent_model = AccentModel().to(device)
    meter_model = MeterModel().to(device)

    accent_optimizer = torch.optim.Adam(accent_model.parameters(), lr=3e-4)
    meter_optimizer = torch.optim.Adam(meter_model.parameters(), lr=3e-4)

    accent_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        accent_optimizer,
        patience=2,
    )
    meter_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meter_optimizer,
        patience=2,
    )

    logging.info("Training accent model")
    accent_state_dict = train_model(
        accent_model,
        partial(train_accent, accent_model, train_loader, accent_optimizer, device),
        partial(eval_accent, accent_model, validation_loader, device),
        scheduler=accent_scheduler,
        max_epochs=max_epochs,
        patience=patience,
    )

    logging.info("Training meter model")
    meter_state_dict = train_model(
        meter_model,
        partial(train_meter, meter_model, train_loader, meter_optimizer, device),
        partial(eval_meter, meter_model, validation_loader, device),
        scheduler=meter_scheduler,
        max_epochs=max_epochs,
        patience=patience,
    )

    return accent_state_dict, meter_state_dict
