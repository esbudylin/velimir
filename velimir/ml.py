import logging
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ml_loader import (
    MeterClassRegistry,
    get_loader,
    get_meter_weights,
    get_refiner_loader,
)
from .nlp import PartOfSpeech


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_meter_classes = MeterClassRegistry.num()
        meter_emb_dim = 16
        num_pos_classes = len(PartOfSpeech) + 1
        pos_emb_dim = 8

        self.meter_emb = nn.Embedding(num_meter_classes, meter_emb_dim)
        self.pos_emb = nn.Embedding(num_pos_classes, pos_emb_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            input_size=3 + meter_emb_dim + pos_emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, accent_input, meter_class, pos_input):
        """
        accent_input: (B, T, 3) with -1 padding
        meter_class: (B,)
        pos_input: (B, T) with -1 padding
        """

        mask = (accent_input != -1).any(dim=-1)  # (B, T)
        lengths = mask.sum(dim=1).to(dtype=torch.int64, device="cpu")

        _, T, _ = accent_input.shape

        accent_input = accent_input.masked_fill(~mask.unsqueeze(-1), 0.0)

        meter_emb = self.meter_emb(meter_class)  # (B, D)
        meter_emb = meter_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

        pos_emb = self.pos_emb(pos_input + 1)  # (B, T, D_pos)

        x = torch.cat([accent_input, meter_emb, pos_emb], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)

        logits = self.head(out).squeeze(-1)

        return logits


def train_accent(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        loss = accent_forward_loss(model, batch, device)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Accent model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def accent_forward_loss(model, batch, device):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    meter_class = batch.meter_class.to(device, non_blocking=True)
    pos_input = batch.part_of_speech_input.to(device, non_blocking=True)
    y = batch.poetic_accents.to(device, non_blocking=True)

    mask = y != -1

    logits = model(accent_input, meter_class, pos_input)

    loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])

    return loss


def eval_accent(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss = accent_forward_loss(model, batch, device)

            total_loss += loss.item()

    return total_loss / len(loader)


class MeterModel(nn.Module):
    def __init__(self):
        super().__init__()

        pos_emb_dim = 8
        num_pos_classes = len(PartOfSpeech) + 1

        hidden = 128

        num_classes = MeterClassRegistry.num()

        self.pos_emb = nn.Embedding(num_pos_classes, pos_emb_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            input_size=3 + pos_emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, accent_input, pos_input):
        mask = (accent_input != -1).any(dim=-1)

        _, T, _ = accent_input.shape

        lengths = mask.sum(dim=1).to(dtype=torch.int64, device="cpu")
        x = accent_input.masked_fill(~mask.unsqueeze(-1), 0.0)

        pos_emb = self.pos_emb(pos_input + 1)
        x = torch.cat([x, pos_emb], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)

        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)

        return self.fc(pooled)


def train_meter(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for batch in loader:
        optimizer.zero_grad()

        loss = meter_forward_loss(model, batch, loss_fn, device)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Meter model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def meter_forward_loss(model, batch, loss_fn, device):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    pos_input = batch.part_of_speech_input.to(device, non_blocking=True)
    meter_target = batch.meter_class.to(device, non_blocking=True)

    logits = model(accent_input, pos_input)

    loss = loss_fn(logits, meter_target)

    return loss


def eval_meter(model, loader, device):
    model.eval()
    total_loss = 0.0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in loader:
            loss = meter_forward_loss(model, batch, loss_fn, device)

            total_loss += loss.item()

    return total_loss / len(loader)


class StanzaRefiner(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_classes = MeterClassRegistry.num()

        self.line_encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.line_attn = nn.Linear(hidden * 2, 1)

        self.stanza_encoder = nn.LSTM(
            input_size=hidden * 2 + num_classes,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, accent_input, meter_logits):
        N, T, _ = accent_input.shape

        syllable_mask = (accent_input != -1).any(dim=-1)
        lengths = syllable_mask.sum(dim=1).to(dtype=torch.int64, device="cpu")

        x = accent_input.masked_fill(~syllable_mask.unsqueeze(-1), 0.0)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.line_encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)

        scores = self.line_attn(out).squeeze(-1)
        scores = scores.masked_fill(~syllable_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        line_enc = (out * weights.unsqueeze(-1)).sum(dim=1)

        x = torch.cat([line_enc, meter_logits], dim=-1).unsqueeze(0)

        out, _ = self.stanza_encoder(x)
        out = out.squeeze(0)

        return self.head(out)


def train_refiner(model, loader, optimizer, device, meter_model):
    model.train()
    total_loss = 0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for batch in loader:
        optimizer.zero_grad()

        loss = refiner_forward_loss(model, batch, loss_fn, device, meter_model)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Refiner model: skipping invalid stanza")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def refiner_forward_loss(model, batch, loss_fn, device, meter_model):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    meter_target = batch.meter_target.to(device, non_blocking=True)
    pos_input = batch.pos_input.to(device, non_blocking=True)

    with torch.no_grad():
        meter_logits = meter_model(accent_input, pos_input)

    refined = model(accent_input, meter_logits)

    loss = loss_fn(refined, meter_target)

    return loss


def eval_refiner(model, loader, device, meter_model):
    model.eval()
    total_loss = 0.0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in loader:
            loss = refiner_forward_loss(model, batch, loss_fn, device, meter_model)
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
    patience=3,
    batch_size=2048,
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

    accent_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(accent_optimizer)
    meter_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meter_optimizer)

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


def train_refiner_model(
    train_chunks,
    val_chunks,
    meter_state_dict,
    max_epochs=100,
    patience=3,
    num_workers=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meter_model = MeterModel().to(device)
    meter_model.load_state_dict(meter_state_dict)
    meter_model.eval()
    for param in meter_model.parameters():
        param.requires_grad = False

    model = StanzaRefiner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_loader = get_refiner_loader(
        train_chunks,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = get_refiner_loader(
        val_chunks,
        num_workers=num_workers,
        pin_memory=False,
    )

    state_dict = train_model(
        model,
        partial(train_refiner, model, train_loader, optimizer, device, meter_model),
        partial(eval_refiner, model, val_loader, device, meter_model),
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
    )

    return state_dict
