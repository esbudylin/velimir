import os
import sqlite3

import torch
from torch.nn.utils.rnn import pad_sequence

from .domain_models import MeterClass
from .ml_loader import (
    MeterClassRegistry,
    RawSample,
    make_accent_input,
    get_loader,
)
from .settings import PREDICTION_DB_PATH

predictions_schema = """
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

refiner_predictions_schema = """
CREATE TABLE refiner_predictions (
    poem_path TEXT,
    line_idx INTEGER,

    accent_pred TEXT,
    accent_target TEXT,

    meter_class_pred INTEGER,
    meter_class_target INTEGER,

    meter_pred TEXT,
    meter_target TEXT,

    caesura_pred TEXT,
    caesura_target TEXT,

    UNIQUE(poem_path, line_idx) ON CONFLICT FAIL
);
"""


def init_db(path=None):
    if path is None:
        path = PREDICTION_DB_PATH

    if path != ":memory:" and os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    conn.execute(predictions_schema)
    conn.execute(refiner_predictions_schema)
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
    return ",".join(str(x) for x in li)


def make_row(rs, accent_pred_str, accent_target_str, meter_pred_int, meter_target_int):
    mc_pred = MeterClassRegistry.int_to_mc(meter_pred_int)
    mc_target = MeterClassRegistry.int_to_mc(meter_target_int)

    return (
        rs.poem_path,
        rs.line_idx,
        accent_pred_str,
        accent_target_str,
        meter_pred_int,
        meter_target_int,
        meters_to_str(mc_pred),
        meters_to_str(mc_target),
        caesura_to_str(mc_pred.caesura),
        caesura_to_str(mc_target.caesura),
    )


def evaluate_accent_pass(
    accent_model,
    device: torch.device,
    samples: list[RawSample],
    meter_preds: torch.Tensor,
    batch_size: int,
):
    loader = get_loader(samples, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    offset = 0

    rhythms_pred: list[str] = []
    rhythms_target: list[str] = []
    meter_pred_ints: list[int] = []

    with torch.no_grad():
        for batch in loader:
            accent_input = batch.accent_input.to(device)
            pos_input = batch.part_of_speech_input.to(device)
            poetic_target = batch.poetic_accents.to(device)

            n = accent_input.size(0)

            batch_meter_preds = meter_preds[offset : offset + n].to(device)

            accent_logits = accent_model(accent_input, batch_meter_preds, pos_input)
            accent_pred = (torch.sigmoid(accent_logits) > 0.5).float()

            mask = poetic_target != -1
            correct += (accent_pred[mask] == poetic_target[mask]).sum().item()
            total += mask.sum().item()

            accent_pred_masked = accent_pred.masked_fill(~mask, -1)

            for i in range(n):
                rhythms_pred.append(rhythm_to_str(accent_pred_masked[i]))
                rhythms_target.append(rhythm_to_str(poetic_target[i]))
                meter_pred_ints.append(batch_meter_preds[i].item())

            offset += n

    accuracy = correct / total if total else 0.0
    return accuracy, rhythms_pred, rhythms_target, meter_pred_ints


def build_accent_rows(
    samples: list[RawSample],
    rhythms_pred: list[str],
    rhythms_target: list[str],
    meter_pred_ints: list[int],
):
    rows = []
    for rs, rp, rt, mi in zip(samples, rhythms_pred, rhythms_target, meter_pred_ints):
        rows.append(
            make_row(rs, rp, rt, mi, rs.meter_class)
        )
    return rows


def write_rows(conn: sqlite3.Connection, table: str, rows: list[tuple]):
    cursor = conn.cursor()
    insert_sql = f"""
        INSERT INTO {table} (
            poem_path, line_idx,
            accent_pred, accent_target,
            meter_class_pred, meter_class_target,
            meter_pred, meter_target,
            caesura_pred, caesura_target
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_sql, rows)


def evaluate_models(
    accent_model,
    meter_model,
    device: torch.device,
    raw_samples: list[RawSample],
    conn: sqlite3.Connection,
    batch_size: int = 16,
):
    loader = get_loader(raw_samples, batch_size=batch_size, shuffle=False)

    meter_preds = torch.full((len(raw_samples),), -1, dtype=torch.long)

    meter_correct = 0
    meter_total = 0
    meter_offset = 0

    with torch.no_grad():
        for batch in loader:
            accent_input = batch.accent_input.to(device)
            pos_input = batch.part_of_speech_input.to(device)
            meter_target = batch.meter_class.to(device)

            n = accent_input.size(0)

            pred = torch.argmax(meter_model(accent_input, pos_input), dim=1)
            meter_preds[meter_offset : meter_offset + n] = pred.cpu()
            meter_correct += (pred == meter_target).sum().item()
            meter_total += n

            meter_offset += n

    accent_accuracy, rhythms_pred, rhythms_target, meter_pred_ints = evaluate_accent_pass(
        accent_model,
        device,
        raw_samples,
        meter_preds,
        batch_size,
    )

    rows = build_accent_rows(raw_samples, rhythms_pred, rhythms_target, meter_pred_ints)
    write_rows(conn, "predictions", rows)

    return {
        "accent_accuracy": accent_accuracy,
        "meter_accuracy": meter_correct / meter_total if meter_total else 0.0,
    }


def evaluate_refiner_models(
    accent_model,
    meter_model,
    refiner_model,
    device: torch.device,
    chunks: list[list[RawSample]],
    conn: sqlite3.Connection,
    batch_size: int = 16,
):
    samples = [rs for chunk in chunks for rs in chunk]

    refined_meter_preds = torch.full((len(samples),), -1, dtype=torch.long)

    meter_correct = 0
    meter_total = 0
    global_offset = 0

    with torch.no_grad():
        for chunk_lines in chunks:
            accent_tensors = [make_accent_input(rs) for rs in chunk_lines]
            pos_tensors = [
                torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)
                for rs in chunk_lines
            ]

            accent_input = pad_sequence(
                accent_tensors, batch_first=True, padding_value=-1
            ).to(device)
            pos_input = pad_sequence(
                pos_tensors, batch_first=True, padding_value=-1
            ).to(device)

            meter_targets = torch.tensor(
                [rs.meter_class for rs in chunk_lines], dtype=torch.long
            ).to(device)

            base_logits = meter_model(accent_input, pos_input)
            refined = refiner_model(accent_input, base_logits)
            preds = torch.argmax(refined, dim=1)

            for local_idx in range(len(chunk_lines)):
                refined_meter_preds[global_offset + local_idx] = preds[local_idx]

            meter_correct += (preds == meter_targets).sum().item()
            meter_total += len(meter_targets)
            global_offset += len(chunk_lines)

    refiner_meter_accuracy = meter_correct / meter_total if meter_total else 0.0

    refiner_accent_accuracy, rhythms_pred, rhythms_target, meter_pred_ints = evaluate_accent_pass(
        accent_model,
        device,
        samples,
        refined_meter_preds,
        batch_size,
    )

    rows = build_accent_rows(samples, rhythms_pred, rhythms_target, meter_pred_ints)
    write_rows(conn, "refiner_predictions", rows)

    return {
        "refiner_meter_accuracy": refiner_meter_accuracy,
        "refiner_accent_accuracy": refiner_accent_accuracy,
    }
