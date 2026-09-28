import csv
import sqlite3

import numpy as np
import torch
from scipy.stats import rankdata

from .ml_loader import RhymePairRow


def predict_rhyme_probs(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    probs = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch.ids_a.to(device),
                batch.stress_a.to(device),
                batch.ids_b.to(device),
                batch.stress_b.to(device),
            )

            probs.append(torch.sigmoid(torch.as_tensor(logits)).cpu())
            labels.append(batch.labels)

    return torch.cat(probs), torch.cat(labels)


def confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict:
    preds = preds.astype(bool)
    labels = labels.astype(bool)

    true_positives = int(np.sum(preds & labels))
    true_negatives = int(np.sum(~preds & ~labels))
    false_positives = int(np.sum(preds & ~labels))
    false_negatives = int(np.sum(~preds & labels))

    return {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores)

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rhyme_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> dict:
    metrics = confusion_counts(labels, probs >= threshold)
    metrics["auc"] = roc_auc(labels, probs)

    return metrics


RHYME_ERROR_FIELDS = [
    "error_type",
    "pair_id",
    "poem_path",
    "word_a",
    "word_b",
    "phon_a",
    "accents_a",
    "phon_b",
    "accents_b",
    "label",
    "probability",
]

RHYME_ERROR_QUERY = """
    SELECT po.path, p.word_a, p.word_b,
           p.phon_a, p.accents_a, p.phon_b, p.accents_b
    FROM pairs p
    JOIN poems po ON p.poem_id = po.id
    WHERE p.id = ?
"""


def write_rhyme_errors(
    path: str,
    pairs_db_path: str,
    rows: list[RhymePairRow],
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> int:
    written = 0

    conn = sqlite3.connect(pairs_db_path)

    try:
        with open(path, "w", encoding="utf8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=RHYME_ERROR_FIELDS)
            writer.writeheader()

            for row, label, prob in zip(rows, labels, probs):
                predicted = prob >= threshold

                if predicted == bool(label):
                    continue

                (poem_path, word_a, word_b, phon_a, accents_a, phon_b, accents_b) = (
                    conn.execute(RHYME_ERROR_QUERY, (row.pair_id,)).fetchone()
                )

                writer.writerow(
                    {
                        "error_type": (
                            "false_positive" if predicted else "false_negative"
                        ),
                        "pair_id": row.pair_id,
                        "poem_path": poem_path,
                        "word_a": word_a,
                        "word_b": word_b,
                        "phon_a": phon_a,
                        "accents_a": accents_a,
                        "phon_b": phon_b,
                        "accents_b": accents_b,
                        "label": int(label),
                        "probability": float(prob),
                    }
                )
                written += 1
    finally:
        conn.close()

    return written
