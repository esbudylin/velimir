import os
import sqlite3

import torch
from torch.nn.utils.rnn import pad_sequence

from .domain_models import MeterClass
from .ml_loader import (
    MeterClassRegistry,
    RawSample,
    make_accent_input,
)
from .settings import PREDICTION_DB_PATH

predictions_schema = """
CREATE TABLE predictions (
    poem_path TEXT,
    line_idx INTEGER,
    chunk_idx INTEGER,

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


def init_db(path=None):
    if path is None:
        path = PREDICTION_DB_PATH

    if path != ":memory:" and os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    conn.execute(predictions_schema)
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

    match acc:
        case [only] if mc.caesura:
            return only + "~"
        case _:
            return "~".join(acc)


def caesura_to_str(li):
    return ",".join(str(x) for x in li)


def make_row(rs, accent_pred_str, accent_target_str, meter_pred_int, meter_target_int):
    mc_pred = MeterClassRegistry.int_to_mc(meter_pred_int)
    mc_target = MeterClassRegistry.int_to_mc(meter_target_int)

    return (
        rs.poem_path,
        rs.line_idx,
        rs.chunk_idx,
        accent_pred_str,
        accent_target_str,
        meter_pred_int,
        meter_target_int,
        meters_to_str(mc_pred),
        meters_to_str(mc_target),
        caesura_to_str(mc_pred.caesura),
        caesura_to_str(mc_target.caesura),
    )


def write_rows(conn: sqlite3.Connection, rows: list[tuple]):
    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO predictions (
            poem_path, line_idx, chunk_idx,
            accent_pred, accent_target,
            meter_class_pred, meter_class_target,
            meter_pred, meter_target,
            caesura_pred, caesura_target
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_sql, rows)


def evaluate_models(
    meter_model,
    accent_model,
    device: torch.device,
    test_chunks: list[list[RawSample]],
    conn: sqlite3.Connection,
    batch_size: int = 16,
):
    samples = [rs for chunk in test_chunks for rs in chunk]

    meter_preds = torch.full((len(samples),), -1, dtype=torch.long)
    accent_pred_strs: list[str] = []
    accent_target_strs: list[str] = []

    meter_correct = 0
    meter_total = 0
    accent_correct = 0
    accent_correct_gt = 0
    accent_total = 0
    global_offset = 0

    with torch.no_grad():
        for chunk_lines in test_chunks:
            accent_tensors = [make_accent_input(rs) for rs in chunk_lines]
            pos_tensors = [
                torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)
                for rs in chunk_lines
            ]
            accent_targets = [
                torch.tensor(rs.syllables.poetic_accents, dtype=torch.float32)
                for rs in chunk_lines
            ]

            accent_input = pad_sequence(
                accent_tensors, batch_first=True, padding_value=-1
            ).to(device)
            pos_input = pad_sequence(
                pos_tensors, batch_first=True, padding_value=-1
            ).to(device)
            accent_target = pad_sequence(
                accent_targets, batch_first=True, padding_value=-1
            ).to(device)

            meter_target = torch.tensor(
                [rs.meter_class for rs in chunk_lines], dtype=torch.long
            ).to(device)

            meter_logits = meter_model(accent_input, pos_input)
            pred_meter = torch.argmax(meter_logits, dim=1)

            meter_correct += (pred_meter == meter_target).sum().item()
            meter_total += len(meter_target)

            accent_logits = accent_model(accent_input, pos_input, pred_meter)
            pred_accent = (torch.sigmoid(accent_logits) > 0.5).float()

            accent_logits_gt = accent_model(accent_input, pos_input, meter_target)
            pred_accent_gt = (torch.sigmoid(accent_logits_gt) > 0.5).float()

            out_T = accent_logits.shape[1]
            accent_target_trunc = accent_target[:, :out_T]
            accent_mask = accent_target_trunc != -1
            accent_correct += (
                (pred_accent[accent_mask] == accent_target_trunc[accent_mask])
                .sum()
                .item()
            )
            accent_correct_gt += (
                (pred_accent_gt[accent_mask] == accent_target_trunc[accent_mask])
                .sum()
                .item()
            )
            accent_total += accent_mask.sum().item()

            pred_accent_masked = pred_accent.masked_fill(~accent_mask, -1)

            for local_idx, rs in enumerate(chunk_lines):
                meter_preds[global_offset] = pred_meter[local_idx]
                accent_pred_strs.append(rhythm_to_str(pred_accent_masked[local_idx]))
                accent_target_strs.append(rhythm_to_str(accent_target_trunc[local_idx]))
                global_offset += 1

    meter_accuracy = meter_correct / meter_total if meter_total else 0.0
    accent_accuracy = accent_correct / accent_total if accent_total else 0.0
    accent_accuracy_gt = accent_correct_gt / accent_total if accent_total else 0.0

    rows = []
    for rs, rp, rt, mi in zip(
        samples, accent_pred_strs, accent_target_strs, meter_preds
    ):
        rows.append(make_row(rs, rp, rt, int(mi), rs.meter_class))
    write_rows(conn, rows)

    return {
        "meter_accuracy": meter_accuracy,
        "accent_accuracy": accent_accuracy,
        "accent_accuracy_gt": accent_accuracy_gt,
    }
