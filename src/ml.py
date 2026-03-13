import logging
import random
from typing import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.models import OutputPoem


@dataclass(slots=True)
class Sample:
    x: torch.Tensor
    poetic_accents: torch.Tensor
    meter: int
    unstable: int


@dataclass(slots=True)
class CollatedSample:
    x: torch.Tensor
    poetic_accents: torch.Tensor
    meter: torch.Tensor
    unstable: torch.Tensor
    mask: torch.Tensor


@dataclass(slots=True)
class ForwardedMeter:
    meter: torch.Tensor
    unstable: torch.Tensor


class PoetryDataset(Dataset):
    def __init__(self, poems):
        self.poems = poems
        self.index = []

        for pi, poem_data in enumerate(poems):
            _, lines = poem_data
            for li in range(len(lines)):
                self.index.append((pi, li))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pi, li = self.index[i]

        poem = OutputPoem.decode(self.poems[pi])
        line = poem.lines[li]
        masks = line.syllable_masks

        ling = torch.tensor(masks.linguistic_accent_mask, dtype=torch.float32)
        last = torch.tensor(masks.last_in_word_mask, dtype=torch.float32)
        poetic = torch.tensor(masks.poetic_accent_mask, dtype=torch.float32)

        x = torch.stack([ling, last], dim=1)

        meter = line.meters[0]
        meter_type = meter.meter
        unstable = int(meter.unstable)

        return Sample(
            x=x,
            poetic_accents=poetic,
            meter=meter_type,
            unstable=unstable,
        )


def collate(batch):
    x = [b.x for b in batch]
    poetic = [b.poetic_accents for b in batch]
    # caesura = [b["caesura"] for b in batch]

    meter = torch.tensor([b.meter for b in batch])
    unstable = torch.tensor([b.unstable for b in batch])

    lengths = torch.tensor([len(v) for v in x])

    x = pad_sequence(x, batch_first=True)
    poetic = pad_sequence(poetic, batch_first=True)
    # caesura = pad_sequence(caesura, batch_first=True)

    mask = torch.arange(x.size(1))[None, :] < lengths[:, None]

    return CollatedSample(
        x=x,
        poetic_accents=poetic,
        meter=meter,
        unstable=unstable,
        # "caesura": caesura,
        mask=mask,
    )


class AccentModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()

        input_size = 2

        self.encoder = nn.LSTM(
            input_size,
            hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.encoder(x)

        logits = self.head(h).squeeze(-1)

        return logits


class MeterModel(nn.Module):
    def __init__(self, hidden=128, num_meter=12):
        super().__init__()

        input_size = 2

        self.encoder = nn.LSTM(
            input_size,
            hidden,
            batch_first=True,
            bidirectional=True,
        )

        enc = hidden * 2

        self.meter_head = nn.Linear(enc, num_meter)
        self.unstable_head = nn.Linear(enc, 1)

        self.caesura_head = nn.Linear(enc, 1)

    def forward(self, x):
        h, _ = self.encoder(x)

        pooled = h.mean(dim=1)

        meter = self.meter_head(pooled)
        unstable = self.unstable_head(pooled).squeeze(-1)

        # caesura = self.caesura_head(h).squeeze(-1)

        return ForwardedMeter(meter=meter, unstable=unstable)


def train_accent(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for batch in loader:
        x = batch.x.to(device)
        y = batch.poetic_accents.to(device)
        mask = batch.mask.to(device)

        logits = model(x)

        loss = F.binary_cross_entropy_with_logits(
            logits[mask],
            y[mask],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_meter(accent_model, meter_model, loader, optimizer, device):
    meter_model.train()
    accent_model.eval()

    total_loss = 0

    for batch in loader:
        x = batch.x.to(device)

        meter_target = batch.meter.to(device)
        unstable_target = batch.unstable.to(device).float()
        # caesura_target = batch["caesura"].to(device)

        with torch.no_grad():
            accent_logits = accent_model(x)
            poetic = torch.sigmoid(accent_logits)

        meter_input = torch.stack(
            [
                poetic,
                x[:, :, 1],  # last_in_word
            ],
            dim=2,
        )

        pred = meter_model(meter_input)

        meter_loss = F.cross_entropy(
            pred.meter,
            meter_target,
        )

        unstable_loss = F.binary_cross_entropy_with_logits(
            pred.unstable,
            unstable_target,
        )

        # caesura_loss = F.binary_cross_entropy_with_logits(
        #     pred["caesura"][mask],
        #     caesura_target[mask],
        # )

        loss = meter_loss + unstable_loss  # + caesura_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def split_poems(
    poems: Iterator,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list]:
    poems_l = list(poems)

    random.seed(seed)
    random.shuffle(poems_l)

    split = int(len(poems_l) * (1 - test_ratio))

    train_poems = poems_l[:split]
    test_poems = poems_l[split:]

    return train_poems, test_poems


def validate_models(
    accent_model,
    meter_model,
    poems: list[OutputPoem],
    batch_size: int = 32,
):

    device = next(accent_model.parameters()).device

    dataset = PoetryDataset(poems)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    accent_model.eval()
    meter_model.eval()

    accent_correct = 0
    accent_total = 0

    meter_correct = 0
    meter_total = 0

    unstable_correct = 0
    unstable_total = 0

    # caesura_correct = 0
    # caesura_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            mask = batch.mask.to(device)

            poetic_target = batch.poetic_accents.to(device)
            meter_target = batch.meter.to(device)
            unstable_target = batch.unstable.to(device)
            # caesura_target = batch["caesura"].to(device)

            # Accent prediction
            accent_logits = accent_model(x)
            accent_pred = (torch.sigmoid(accent_logits) > 0.5).float()

            accent_correct += (accent_pred[mask] == poetic_target[mask]).sum().item()
            accent_total += mask.sum().item()

            # Meter model input
            meter_input = torch.stack(
                [
                    accent_pred,
                    x[:, :, 1],
                ],
                dim=2,
            )

            pred = meter_model(meter_input)

            meter_pred = pred.meter.argmax(dim=1)
            unstable_pred = (torch.sigmoid(pred.unstable) > 0.5).long()
            # caesura_pred = (torch.sigmoid(pred["caesura"]) > 0.5).float()

            meter_correct += (meter_pred == meter_target).sum().item()
            meter_total += meter_target.size(0)

            unstable_correct += (unstable_pred == unstable_target).sum().item()
            unstable_total += unstable_target.size(0)

            # caesura_correct += (caesura_pred[mask] == caesura_target[mask]).sum().item()
            # caesura_total += mask.sum().item()

    return {
        "accent_accuracy": accent_correct / accent_total,
        "meter_accuracy": meter_correct / meter_total,
        "unstable_accuracy": unstable_correct / unstable_total,
        # "caesura_accuracy": caesura_correct / caesura_total,
    }


def train_models(
    poems: list,
    epochs: int = 10,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoetryDataset(poems)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    accent_model = AccentModel().to(device)
    meter_model = MeterModel().to(device)

    accent_optimizer = torch.optim.Adam(accent_model.parameters(), lr=1e-3)
    meter_optimizer = torch.optim.Adam(meter_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        accent_loss = train_accent(
            accent_model,
            loader,
            accent_optimizer,
            device,
        )

        meter_loss = train_meter(
            accent_model,
            meter_model,
            loader,
            meter_optimizer,
            device,
        )

        logging.info(
            f"epoch {epoch} accent_loss={accent_loss:.4f} meter_loss={meter_loss:.4f}"
        )

    return accent_model, meter_model
