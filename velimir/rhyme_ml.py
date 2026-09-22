import logging
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .ml import train_model
from .rhyme_ml_loader import (
    EMBEDDING_SIZE,
    PAD_ID,
    RhymePairBatch,
    RhymePairRow,
    get_rhyme_pair_loader,
)


class LineEndingEncoder(nn.Module):
    def __init__(self, emb_dim=32, hidden=128, num_layers=2, dropout=0.1):
        super().__init__()

        self.emb = nn.Embedding(EMBEDDING_SIZE, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=emb_dim + 1,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, ids, stress):
        mask = ids != PAD_ID
        lengths = mask.sum(dim=1).to(dtype=torch.int64, device="cpu")

        x = torch.cat([self.emb(ids), stress.unsqueeze(-1)], dim=-1)

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.shape[1])

        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)

        return (out * weights.unsqueeze(-1)).sum(dim=1)


class RhymePairModel(nn.Module):
    def __init__(self, emb_dim=32, hidden=128):
        super().__init__()

        self.encoder = LineEndingEncoder(emb_dim=emb_dim, hidden=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, ids_a, stress_a, ids_b, stress_b):
        emb_a = self.encoder(ids_a, stress_a)
        emb_b = self.encoder(ids_b, stress_b)

        features = torch.cat(
            [emb_a, emb_b, (emb_a - emb_b).abs(), emb_a * emb_b],
            dim=-1,
        )

        return self.head(features).squeeze(-1)


def get_pos_weight(rows: list[RhymePairRow]) -> torch.Tensor:
    positives = sum(row.label for row in rows)
    negatives = len(rows) - positives

    return torch.tensor([negatives / positives], dtype=torch.float32)


def rhyme_forward_loss(model, batch: RhymePairBatch, loss_fn, device):
    logits = model(
        batch.ids_a.to(device, non_blocking=True),
        batch.stress_a.to(device, non_blocking=True),
        batch.ids_b.to(device, non_blocking=True),
        batch.stress_b.to(device, non_blocking=True),
    )

    return loss_fn(logits, batch.labels.to(device, non_blocking=True))


def train_rhyme(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        loss = rhyme_forward_loss(model, batch, loss_fn, device)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Rhyme model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_rhyme(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss = rhyme_forward_loss(model, batch, loss_fn, device)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_rhyme_model(
    train_rows: list[RhymePairRow],
    val_rows: list[RhymePairRow],
    seed: int,
    max_epochs: int = 100,
    patience: int = 6,
    batch_size: int = 1024,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for rhyme training", device)

    train_loader = get_rhyme_pair_loader(
        train_rows,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    val_loader = get_rhyme_pair_loader(
        val_rows,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    torch.manual_seed(seed)
    model = RhymePairModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    pos_weight = get_pos_weight(train_rows).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logging.info("Training rhyme pair model")
    state_dict, val_loss, epochs = train_model(
        model,
        partial(train_rhyme, model, train_loader, optimizer, device, loss_fn),
        partial(eval_rhyme, model, val_loader, device, loss_fn),
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
    )

    return state_dict, val_loss, epochs
