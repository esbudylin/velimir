import os
import sqlite3
from collections import deque

import torch

from .domain_models import MeterType, Poem
from .ml_loader import get_loader
from .settings import PREDICTION_DB_PATH

error_db_schema = """
CREATE TABLE IF NOT EXISTS predictions (
    poem_path TEXT,
    line_idx INTEGER,

    -- Accent (sequence)
    accent_pred TEXT,
    accent_target TEXT,

    -- Meter (fixed size)
    meter_pred TEXT,
    meter_target TEXT,

    -- Caesura
    caesura_pred TEXT,
    caesura_target TEXT,

    -- Unstable
    unstable_pred INTEGER,
    unstable_target INTEGER,

    UNIQUE(poem_path, line_idx) ON CONFLICT FAIL
);
"""


def init_db():
    if os.path.exists(PREDICTION_DB_PATH):
        os.remove(PREDICTION_DB_PATH)

    conn = sqlite3.connect(PREDICTION_DB_PATH)
    conn.execute(error_db_schema)
    conn.commit()
    return conn


def rhythm_to_str(t):
    return "".join(str(int(x)) if x != -1 else "" for x in t.tolist())


def meters_to_str(li):
    res = []
    for m in li:
        if m != -1:
            try:
                res.append(MeterType(m).to_str())
            except ValueError:
                res.append("?")
        else:
            res.append("0")

    return "".join(res)


def caesura_to_str(t):
    return "".join(str(int(x)) if x != -1 else "-" for x in t.tolist())


def validate_models(
    accent_model,
    meter_model,
    poems: list,
    batch_size: int = 16,
):
    device = next(accent_model.parameters()).device

    loader = get_loader(poems, batch_size=batch_size, shuffle=False)

    accent_model.eval()
    meter_model.eval()

    conn = init_db()
    cursor = conn.cursor()

    total_acccent_correct = 0
    accent_total = 0

    meter_correct = 0
    meter_total = 0

    unstable_correct = 0
    unstable_total = 0

    caesura_correct = 0
    caesura_total = 0

    poem_counts = deque(
        (
            {
                "path": poem.path,
                "lines_count": len(poem.lines),
                "lines_consumed": 0,
            }
            for poem in map(Poem.decode, poems)
        )
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
            accent_pred_masked = accent_pred.masked_fill(~mask, -1)
            meter_pred = meter_model(accent_pred_masked.unsqueeze(-1))

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

            # =====================
            # DB logging
            # =====================
            rows = []
            batch_size_actual = pred_meter.size(0)

            for line_idx in range(batch_size_actual):
                current_poem = poem_counts[0]

                accent_p = rhythm_to_str(accent_pred_masked[line_idx])
                accent_t = rhythm_to_str(poetic_target[line_idx])

                meter_p = meters_to_str(pred_meter[line_idx].round().tolist())
                meter_t = meters_to_str(target_meter[line_idx].tolist())

                caesura_p = caesura_to_str(pred_caesura[line_idx].round())
                caesura_t = caesura_to_str(target_caesura[line_idx])

                unstable_p = int(unstable_pred[line_idx].item())
                unstable_t = int(target_unstable[line_idx].item())

                rows.append(
                    (
                        current_poem["path"],
                        current_poem["lines_consumed"],
                        accent_p,
                        accent_t,
                        meter_p,
                        meter_t,
                        caesura_p,
                        caesura_t,
                        unstable_p,
                        unstable_t,
                    )
                )

                current_poem["lines_consumed"] += 1

                if current_poem["lines_consumed"] == current_poem["lines_count"]:
                    poem_counts.popleft()

            cursor.executemany(
                """
                INSERT INTO predictions (
                    poem_path, line_idx,
                    accent_pred, accent_target,
                    meter_pred, meter_target,
                    caesura_pred, caesura_target,
                    unstable_pred, unstable_target
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                rows,
            )

    conn.commit()
    conn.close()

    return {
        "accent_accuracy": total_acccent_correct / accent_total if accent_total else 0,
        "meter_accuracy": meter_correct / meter_total if meter_total else 0,
        "caesura_accuracy": caesura_correct / caesura_total if caesura_total else 0,
        "unstable_accuracy": unstable_correct / unstable_total if unstable_total else 0,
    }
