import argparse
import logging

import numpy as np
import torch
from scipy.stats import rankdata

from velimir.logger import LoggingSettings
from velimir.rhyme_identifier import calc_phonetic_rhyme_coef
from velimir.rhyme_ml import RhymePairModel
from velimir.rhyme_ml_loader import (
    RhymePairRow,
    get_rhyme_pair_loader,
    load_rhyme_pairs,
    split_pairs,
)
from velimir.settings import (
    RHYME_MODEL,
    RHYME_PAIRS_DB_PATH,
    RHYME_PAIRS_TEST_DB_PATH,
    RHYME_TEST_MODEL,
)

BASELINE_THRESHOLD = 0.5


def predict_rhyme_probs(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

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

            probs.append(torch.sigmoid(logits).cpu())
            labels.append(batch.labels)

    return torch.cat(probs), torch.cat(labels)


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    preds = preds.astype(bool)
    labels = labels.astype(bool)

    tp = int(np.sum(preds & labels))
    tn = int(np.sum(~preds & ~labels))
    fp = int(np.sum(preds & ~labels))
    fn = int(np.sum(~preds & labels))

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
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


def baseline_scores(rows: list[RhymePairRow]) -> np.ndarray:
    scores = []

    for row in rows:
        try:
            scores.append(calc_phonetic_rhyme_coef(row.phon_a, row.phon_b))
        except Exception:
            scores.append(0.0)

    return np.array(scores, dtype=np.float64)


def report(name: str, labels: np.ndarray, scores: np.ndarray):
    metrics = binary_metrics(labels, scores >= BASELINE_THRESHOLD)

    logging.info(
        "%s: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        roc_auc(labels, scores),
    )


def evaluate(test_run: bool = False):
    db_path = RHYME_PAIRS_TEST_DB_PATH if test_run else RHYME_PAIRS_DB_PATH
    model_path = RHYME_TEST_MODEL if test_run else RHYME_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rhyme_pairs(db_path)

    _, _, test_rows = split_pairs(rows)

    if not test_rows:
        raise ValueError("Test split is empty")

    model = RhymePairModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    loader = get_rhyme_pair_loader(
        test_rows,
        shuffle=False,
        batch_size=1024,
        num_workers=0,
    )

    probs, labels = predict_rhyme_probs(model, loader, device)

    report("model", labels.numpy(), probs.numpy())
    report("baseline", labels.numpy(), baseline_scores(test_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the rhyme pair model.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Evaluate the test model on the test dataset",
    )
    args = parser.parse_args()

    LoggingSettings.setup()

    evaluate(test_run=args.test_run)
