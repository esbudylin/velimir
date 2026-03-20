import logging
import random
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
    meta: torch.Tensor  # meter, caesuras, unstable


class PoetryDataset(Dataset):
    def __init__(self, poems: list):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []
        for poem_data in poems:
            poem = OutputPoem.decode(poem_data)
            for line in poem.lines:
                masks = line.syllable_masks
                x = torch.stack(
                    [
                        torch.tensor(masks.linguistic_accent_mask, dtype=torch.float32),
                        torch.tensor(masks.last_in_word_mask, dtype=torch.float32),
                    ],
                    dim=1,
                )
                poetic = torch.tensor(masks.poetic_accent_mask, dtype=torch.float32)

                meter = self._fixed_size_tensor(
                    [int(m.meter) for m in line.meters],
                    size=3,
                )

                caesura = self._fixed_size_tensor(
                    line.caesura,
                    size=2,
                )

                unstable_flag = any(m.unstable for m in line.meters)
                if unstable_flag:
                    unstable = torch.tensor([1], dtype=torch.float32)
                else:
                    unstable = torch.tensor([0], dtype=torch.float32)

                meta = torch.cat([meter, caesura, unstable])

                self.samples.append(Sample(x=x, poetic_accents=poetic, meta=meta))

        logging.info("Dataset loading finished")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def _fixed_size_tensor(values, size, dtype=torch.float32, padding_value=-1):
        values = list(values)
        if len(values) < size:
            values += [padding_value] * (size - len(values))
        else:
            values = values[:size]
        return torch.tensor(values, dtype=dtype)


def collate(batch: list[Sample]):
    x = [b.x for b in batch]
    poetic = [b.poetic_accents for b in batch]
    meta = [b.meta for b in batch]

    x = pad_sequence(x, batch_first=True)
    poetic = pad_sequence(poetic, batch_first=True)
    meta = torch.stack(meta)

    return Sample(x=x, poetic_accents=poetic, meta=meta)


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128

        self.encoder = nn.LSTM(
            input_size=2,
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
        x = batch.x.to(device, non_blocking=True)
        y = batch.poetic_accents.to(device, non_blocking=True)
        mask = y != -1

        optimizer.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


class MeterModel(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 1
        hidden_size = 64

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 6)

    def forward(self, x):
        out, _ = self.encoder(x)
        pooled = out.mean(dim=1)
        output = self.fc(pooled)
        return output


def train_meter(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch.poetic_accents.to(device, non_blocking=True)
        # add feature dimension
        x = x.unsqueeze(-1)

        meter_target = batch.meta.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(x)  # (batch, meta_size)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def split_poems(
    poems,
    test_ratio: float = 0.05,
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

    total_acccent_correct = 0
    accent_total = 0

    meter_correct = 0
    meter_total = 0

    unstable_correct = 0
    unstable_total = 0

    caesura_correct = 0
    caesura_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)

            poetic_target = batch.poetic_accents.to(device)
            meta_target = batch.meta.to(device)  # (B, 6)

            # =====================
            # Accent
            # =====================
            accent_logits = accent_model(x)
            accent_pred = (torch.sigmoid(accent_logits) > 0.5).float()

            mask = poetic_target != -1
            total_acccent_correct += (
                (accent_pred[mask] == poetic_target[mask]).sum().item()
            )
            accent_total += mask.sum().item()

            # =====================
            # Meter input
            # =====================
            accent_pred = accent_pred.masked_fill(~mask, -1).unsqueeze(-1)
            meter_pred = meter_model(accent_pred)

            pred_meter = meter_pred[:, :3]
            pred_caesura = meter_pred[:, 3:5]
            pred_unstable = meter_pred[:, 5]

            target_meter = meta_target[:, :3]
            target_caesura = meta_target[:, 3:5]
            target_unstable = meta_target[:, 5]

            # =====================
            # Meter
            # =====================
            meter_correct += (pred_meter.round() == target_meter).sum().item()
            meter_total += torch.numel(target_meter)

            # =====================
            # Caesura (integer positions)
            # =====================
            caesura_correct += (pred_caesura.round() == target_caesura).sum().item()
            caesura_total += torch.numel(target_caesura)

            # =====================
            # Unstable (binary)
            # =====================
            unstable_pred = (torch.sigmoid(pred_unstable) > 0.5).float()
            unstable_correct += (unstable_pred == target_unstable).sum().item()
            unstable_total += torch.numel(target_unstable)

    return {
        "accent_accuracy": total_acccent_correct / accent_total if accent_total else 0,
        "meter_accuracy": meter_correct / meter_total if meter_total else 0,
        "caesura_accuracy": caesura_correct / caesura_total if caesura_total else 0,
        "unstable_accuracy": unstable_correct / unstable_total if unstable_total else 0,
    }


def train_models(
    poems,
    epochs=6,
    batch_size=32,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for training", device)

    dataset = PoetryDataset(poems)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    accent_model = AccentModel().to(device)
    meter_model = MeterModel().to(device)

    accent_optimizer = torch.optim.Adam(accent_model.parameters(), lr=2e-4)
    meter_optimizer = torch.optim.Adam(meter_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        accent_loss = train_accent(accent_model, loader, accent_optimizer, device)
        meter_loss = train_meter(meter_model, loader, meter_optimizer, device)
        logging.info(
            f"Epoch {epoch} accent_loss={accent_loss:.4f} meter_loss={meter_loss:.4f}"
        )

    return accent_model, meter_model
