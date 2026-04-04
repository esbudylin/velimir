import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ml_loader import get_loader


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128

        self.encoder = nn.LSTM(
            input_size=3,
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

    def forward(self, x):
        h, _ = self.encoder(x)

        logits = self.head(h).squeeze(-1)

        return logits


def train_accent(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        # move tensors to GPU
        x = batch.accent_input.to(device, non_blocking=True)
        y = batch.poetic_accents.to(device, non_blocking=True)
        mask = y != -1

        optimizer.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Accent model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


class MeterModel(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 1
        syllable_distances_size = 4
        meta_size = 6

        hidden_size = 64

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2 + syllable_distances_size, meta_size)

    def forward(self, poetic_accents, syllable_distances):
        mask = (poetic_accents != -1).squeeze(-1)  # (B, T)

        poetic_accents = poetic_accents.masked_fill(~mask.unsqueeze(-1), 0.0)

        out, _ = self.encoder(poetic_accents)

        out = out * mask.unsqueeze(-1)

        pooled = out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = torch.cat([pooled, syllable_distances], dim=1)

        return self.fc(pooled)


def train_meter(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        poetic_accents = batch.poetic_accents.to(device, non_blocking=True)
        # add feature dimension
        poetic_accents = poetic_accents.unsqueeze(-1)

        meter_target = batch.meta.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(
            poetic_accents,
            batch.syllable_distances.to(device, non_blocking=True),
        )  # (batch, meta_size)

        meter_part = logits[:, :3]
        caesura_part = logits[:, 3:5]
        unstable_part = logits[:, 5]

        meter_loss = F.mse_loss(
            meter_part,
            meter_target[:, :3],
        )
        caesura_loss = F.mse_loss(
            caesura_part,
            meter_target[:, 3:5],
        )
        unstable_loss = F.binary_cross_entropy_with_logits(
            unstable_part,
            meter_target[:, 5],
        )

        loss = meter_loss + caesura_loss + unstable_loss

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error("Meter model: skipping invalid batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_models(
    poems,
    epochs=9,
    batch_size=32,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for training", device)

    loader = get_loader(
        poems,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
    )

    accent_model = AccentModel().to(device)
    meter_model = MeterModel().to(device)

    accent_optimizer = torch.optim.Adam(accent_model.parameters(), lr=2e-4)
    meter_optimizer = torch.optim.Adam(meter_model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        accent_loss = train_accent(accent_model, loader, accent_optimizer, device)
        meter_loss = train_meter(meter_model, loader, meter_optimizer, device)
        logging.info(
            f"Epoch {epoch} accent_loss={accent_loss:.4f} meter_loss={meter_loss:.4f}"
        )

    return accent_model, meter_model
