import copy
import logging
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .ml_loader import (
    MeterClassRegistry,
    get_meter_weights,
    get_unified_loader,
)
from .nlp import PartOfSpeech


class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_classes = MeterClassRegistry.num()
        meter_emb_dim = 16
        num_pos_classes = len(PartOfSpeech) + 1
        pos_emb_dim = 8

        self.pos_emb = nn.Embedding(num_pos_classes, pos_emb_dim, padding_idx=0)
        self.meter_emb = nn.Embedding(num_classes, meter_emb_dim)

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

        self.meter_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

        self.accent_head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden * 2 + meter_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
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

        meter_logits = self.meter_head(context)

        meter_probs = torch.softmax(meter_logits, dim=-1)
        meter_emb = meter_probs @ self.meter_emb.weight
        meter_emb_T = meter_emb.unsqueeze(1).expand(-1, T, -1)
        context_T = context.unsqueeze(1).expand(-1, T, -1)

        accent_x = torch.cat([out, context_T, meter_emb_T], dim=-1)
        accent_logits = self.accent_head(accent_x).squeeze(-1)

        return meter_logits, accent_logits


def unified_forward_loss(
    model, batch, line_counts, loss_fns, device, accent_loss_weight
):
    accent_input = batch.accent_input.to(device, non_blocking=True)
    pos_input = batch.pos_input.to(device, non_blocking=True)
    meter_target = batch.meter_target.to(device, non_blocking=True)
    accent_target = batch.accent_target.to(device, non_blocking=True)
    line_counts = line_counts.to(device)

    meter_logits, accent_logits = model(accent_input, pos_input, line_counts)

    meter_loss = loss_fns[0](meter_logits, meter_target)

    accent_mask = accent_target != -1
    accent_loss = loss_fns[1](accent_logits[accent_mask], accent_target[accent_mask])

    return meter_loss + accent_loss_weight * accent_loss, meter_loss, accent_loss


def train_unified(model, loader, optimizer, device, accent_loss_weight):
    model.train()
    total_loss = 0
    total_meter_loss = 0
    total_accent_loss = 0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    meter_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    accent_loss_fn = nn.BCEWithLogitsLoss()
    loss_fns = (meter_loss_fn, accent_loss_fn)

    for batch, line_counts in loader:
        optimizer.zero_grad()

        loss, meter_loss, accent_loss = unified_forward_loss(
            model, batch, line_counts, loss_fns, device, accent_loss_weight
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Unified model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()
        total_meter_loss += meter_loss.item()
        total_accent_loss += accent_loss.item()

    n = len(loader)
    logging.info(
        "training: meter_loss=%.4f accent_loss=%.4f",
        total_meter_loss / n,
        total_accent_loss / n,
    )

    return total_loss / n


def eval_unified(model, loader, device, accent_loss_weight):
    model.eval()
    total_loss = 0.0
    total_meter_loss = 0.0
    total_accent_loss = 0.0

    class_weights = get_meter_weights().to(device, non_blocking=True)
    meter_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    accent_loss_fn = nn.BCEWithLogitsLoss()
    loss_fns = (meter_loss_fn, accent_loss_fn)

    with torch.no_grad():
        for batch, line_counts in loader:
            loss, meter_loss, accent_loss = unified_forward_loss(
                model, batch, line_counts, loss_fns, device, accent_loss_weight
            )
            total_loss += loss.item()
            total_meter_loss += meter_loss.item()
            total_accent_loss += accent_loss.item()

    n = len(loader)
    logging.info(
        "validation: meter_loss=%.4f accent_loss=%.4f",
        total_meter_loss / n,
        total_accent_loss / n,
    )

    return total_loss / n


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


def train_unified_model(
    train_chunks,
    val_chunks,
    max_epochs=100,
    patience=6,
    accent_loss_weight=7.0,
    batch_size=512,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for training", device)

    train_loader = get_unified_loader(
        train_chunks,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    val_loader = get_unified_loader(
        val_chunks,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    model = UnifiedModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    logging.info("Training unified model")
    state_dict = train_model(
        model,
        partial(
            train_unified, model, train_loader, optimizer, device, accent_loss_weight
        ),
        partial(eval_unified, model, val_loader, device, accent_loss_weight),
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
    )

    return state_dict
