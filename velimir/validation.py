import os
import sqlite3

import torch

from .domain_models import MeterClass
from .ml_loader import MeterClassRegistry, RawSample, get_loader
from .settings import PREDICTION_DB_PATH

error_db_schema = """
CREATE TABLE predictions (
    poem_path TEXT,
    line_idx INTEGER,

    -- Accent (sequence)
    accent_pred TEXT,
    accent_target TEXT,

    meter_class_pred INTEGER,
    meter_class_target INTEGER,

    -- Meter formula and caesura are converted from meter class
    meter_pred TEXT,
    meter_target TEXT,

    caesura_pred TEXT,
    caesura_target TEXT,

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


def meters_to_str(mc: MeterClass):
    acc = []

    for m, u in zip(mc.meter_types, mc.unstable):
        mstr = m.to_str()
        if u:
            mstr += "*"
        acc.append(mstr)

    return "~".join(acc)


def caesura_to_str(li):
    return ",".join(str(round(x, 3)) for x in li)


def validate_models(
    accent_model,
    meter_model,
    raw_samples: list[RawSample],
    batch_size: int = 16,
):
    device = next(accent_model.parameters()).device

    loader = get_loader(raw_samples, batch_size=batch_size, shuffle=False)

    accent_model.eval()
    meter_model.eval()

    conn = init_db()
    cursor = conn.cursor()

    total_acccent_correct = 0
    accent_total = 0

    meter_correct = 0
    meter_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            accent_input = batch.accent_input.to(device)

            poetic_target = batch.poetic_accents.to(device)
            meter_target = batch.meter_class.to(device)

            # =====================
            # Meter
            # =====================
            meter_pred = torch.argmax(meter_model(accent_input), dim=1)
            meter_correct += (meter_pred.round() == meter_target).sum().item()
            meter_total += torch.numel(meter_target)

            # =====================
            # Accent
            # =====================
            accent_logits = accent_model(accent_input, meter_pred)
            accent_pred = (torch.sigmoid(accent_logits) > 0.5).float()

            mask = poetic_target != -1
            total_acccent_correct += (
                (accent_pred[mask] == poetic_target[mask]).sum().item()
            )
            accent_total += mask.sum().item()

            accent_pred_masked = accent_pred.masked_fill(~mask, -1)

            # =====================
            # DB logging
            # =====================
            rows = []
            batch_size_actual = meter_pred.size(0)

            for line_idx in range(batch_size_actual):
                current_sample = raw_samples[batch_idx * batch_size + line_idx]

                accent_p = rhythm_to_str(accent_pred_masked[line_idx])
                accent_t = rhythm_to_str(poetic_target[line_idx])

                meter_class_p_i = meter_pred[line_idx].round().item()
                meter_class_t_i = meter_target[line_idx].item()

                meter_class_p = MeterClassRegistry.int_to_mc(meter_class_p_i)
                meter_class_t = MeterClassRegistry.int_to_mc(meter_class_t_i)

                meter_p = meters_to_str(meter_class_p)
                meter_t = meters_to_str(meter_class_t)

                caesura_p = caesura_to_str(meter_class_p.caesura)
                caesura_t = caesura_to_str(meter_class_t.caesura)

                rows.append(
                    (
                        current_sample.poem_path,
                        current_sample.line_idx,
                        accent_p,
                        accent_t,
                        meter_class_p_i,
                        meter_class_t_i,
                        meter_p,
                        meter_t,
                        caesura_p,
                        caesura_t,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO predictions (
                    poem_path, line_idx,
                    accent_pred, accent_target,
                    meter_class_pred, meter_class_target,
                    meter_pred, meter_target,
                    caesura_pred, caesura_target
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
    }
