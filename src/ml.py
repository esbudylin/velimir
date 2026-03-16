import random
import logging
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.models import OutputPoem, Line


@dataclass(slots=True)
class Sample:
    x: torch.Tensor
    poetic_accents: torch.Tensor
    meter: torch.Tensor
    unstable: torch.Tensor


class PoetryDataset(Dataset):
    def __init__(self, poems: list):
        self.chunks: list[list[Line]] = []
        for poem_data in poems:
            poem = OutputPoem.decode(poem_data)
            lines_it = iter(poem.lines)

            # Делим большие стихотворения на части
            while chunk := list(islice(lines_it, 8)):
                self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        x_line_tensors = []
        poetic_acc_tensors = []
        meters = []
        unstable = []

        for line in chunk:
            masks = line.syllable_masks

            x_line_tensors.append(
                torch.stack(
                    [
                        torch.tensor(
                            masks.linguistic_accent_mask,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            masks.last_in_word_mask,
                            dtype=torch.float32,
                        ),
                    ],
                    dim=1,
                )
            )

            poetic_acc_tensors.append(
                torch.as_tensor(masks.poetic_accent_mask, dtype=torch.float32)
            )

            # TODO: handle caesura here
            m = line.meters[0]
            meters.append(m.meter)
            unstable.append(m.unstable)

        return Sample(
            x=pad_sequence(
                x_line_tensors,
                padding_value=-1,
                batch_first=True,
            ),
            poetic_accents=pad_sequence(
                poetic_acc_tensors,
                padding_value=-1,
                batch_first=True,
            ),
            meter=torch.tensor(meters),
            unstable=torch.tensor(unstable),
        )


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 32

        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
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

    def forward(self, x):
        h, _ = self.encoder(x)
        pooled = h.mean(dim=1)
        meter = self.meter_head(pooled)
        unstable = self.unstable_head(pooled).squeeze(-1)
        return ForwardedMeter(meter=meter, unstable=unstable)


def train_accent(model, dataset, n_epoch, device):
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=lambda a: a,
    )

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for iter in range(1, n_epoch + 1):
        current_loss = 0

        for batch in loader:
            batch_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                x = sample.x.to(device, non_blocking=True)
                y = sample.poetic_accents.to(device, non_blocking=True)
                mask = y != -1
                output = model(x)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output[mask],
                    y[mask],
                )
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        logging.info(
            f"Accent Model: epoch {iter} ({iter / n_epoch:.0%}): \t average batch loss = {current_loss / len(loader)}"
        )


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
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoetryDataset(poems)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda a: a,
    )

    accent_model.eval()
    meter_model.eval()

    total_acccent_correct = 0
    accent_total = 0

    for batch in loader:
        for sample in batch:
            x = sample.x.to(device)

            poetic_target = sample.poetic_accents.to(device)
            # meter_target = sample.meter.to(device)
            # unstable_target = batch.unstable.to(device)
            # caesura_target = batch["caesura"].to(device)

            # Accent prediction
            accent_logits = accent_model(x)
            accent_pred = (torch.sigmoid(accent_logits) > 0.5).float()

            accent_false = ((accent_pred == 1) & (poetic_target == 0)).sum().item()
            accent_correct = ((poetic_target == 1) & (accent_pred == 1)).sum().item()

            total_acccent_correct += accent_correct - accent_false
            accent_total += (poetic_target > 0).sum().item()

    return {
        "accent_accuracy": total_acccent_correct / accent_total,
        # "meter_accuracy": meter_correct / meter_total,
        # "unstable_accuracy": unstable_correct / unstable_total,
        # "caesura_accuracy": caesura_correct / caesura_total,
    }


def train_models(poems, epochs=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s for training", device)

    dataset = PoetryDataset(poems)

    accent_model = AccentModel().to(device)

    train_accent(accent_model, dataset, epochs, device)

    accent_model.eval()

    return accent_model, accent_model
