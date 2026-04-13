import logging
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ml_loader import MeterClassRegistry, get_loader


class AccentModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128
        num_meter_classes = MeterClassRegistry.num()
        meter_emb_dim = 16

        self.meter_emb = nn.Embedding(num_meter_classes, meter_emb_dim)

        self.encoder = nn.LSTM(
            input_size=3 + meter_emb_dim,
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

    def forward(self, accent_input, meter_class):
        """
        accent_input: (B, T, 3) with -1 padding
        meter_class: (B,)
        """

        mask = (accent_input != -1).all(dim=-1)  # (B, T)
        lengths = mask.sum(dim=1).cpu()

        _, T, _ = accent_input.shape

        meter_emb = self.meter_emb(meter_class)  # (B, D)
        meter_emb = meter_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

        x = torch.cat([accent_input, meter_emb], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

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
    y = batch.poetic_accents.to(device, non_blocking=True)

    mask = y != -1

    logits = model(accent_input, meter_class)

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

        input_size = 3

        hidden = 128

        num_classes = MeterClassRegistry.num()

        self.encoder = nn.LSTM(
            input_size=input_size,
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

    def forward(self, accent_input):
        mask = (accent_input != -1).any(dim=-1)

        lengths = mask.sum(dim=1).cpu()
        x = accent_input.masked_fill(~mask.unsqueeze(-1), 0.0)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)

        return self.fc(pooled)


def train_meter(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    class_weights = MeterClassRegistry.get_weights().to(device, non_blocking=True)
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
    meter_target = batch.meter_class.to(device, non_blocking=True)

    logits = model(accent_input)

    loss = loss_fn(logits, meter_target)

    return loss


def eval_meter(model, loader, device):
    model.eval()
    total_loss = 0.0

    class_weights = MeterClassRegistry.get_weights().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in loader:
            loss = meter_forward_loss(model, batch, loss_fn, device)

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
