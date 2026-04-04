import logging
from collections import Counter

import torch

from .ml_loader import get_loader
from .domain_models import Poem, MeterType


def update_meter_confustion(meter_confusion, pred_meter, target_meter):
    for i in range(pred_meter.size(0)):
        pred_tuple = tuple(pred_meter[i].tolist())
        target_tuple = tuple(target_meter[i].tolist())

        if pred_tuple != target_tuple:
            meter_confusion[(target_tuple, pred_tuple)] += 1


def log_meter_confusion(meter_confusion):
    logging.info("==== Meter Error Analysis ====")

    if meter_confusion:
        logging.info("Top meter confusions:")

        def meters_to_str(li):
            res = []
            for m in li:
                if m != -1:
                    res.append(MeterType(m).to_str())
                else:
                    res.append("0")

            return "".join(res)

        for (target, pred), count in meter_confusion.most_common(10):
            logging.info(
                "Target %s → Pred %s | count=%d",
                meters_to_str(target),
                meters_to_str(pred),
                count,
            )
    else:
        logging.info("No meter errors")


def validate_models(
    accent_model,
    meter_model,
    poems: list[Poem],
    batch_size: int = 16,
):
    device = next(accent_model.parameters()).device

    loader = get_loader(poems, batch_size=batch_size, shuffle=False)

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

    meter_confusion = Counter()

    with torch.no_grad():
        for batch in loader:
            accent_input = batch.accent_input.to(device)

            poetic_target = batch.poetic_accents.to(device)
            meta_target = batch.meta.to(device)  # (B, 6)

            # =====================
            # Accent
            # =====================
            accent_logits = accent_model(accent_input)
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
            meter_pred = meter_model(accent_pred, batch.syllable_distances.to(device))

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
            update_meter_confustion(
                meter_confusion,
                pred_meter=pred_meter.round(),
                target_meter=target_meter,
            )

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

    log_meter_confusion(meter_confusion)

    return {
        "accent_accuracy": total_acccent_correct / accent_total if accent_total else 0,
        "meter_accuracy": meter_correct / meter_total if meter_total else 0,
        "caesura_accuracy": caesura_correct / caesura_total if caesura_total else 0,
        "unstable_accuracy": unstable_correct / unstable_total if unstable_total else 0,
    }
