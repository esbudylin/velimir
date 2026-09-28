import logging

import torch

from velimir.logger import LoggingSettings
from velimir.rhyme.evaluation import (
    predict_rhyme_probs,
    rhyme_metrics,
    write_rhyme_errors,
)
from velimir.rhyme.ml import RhymePairModel
from velimir.rhyme.ml_loader import (
    get_rhyme_pair_loader,
    load_rhyme_pairs,
    split_pairs,
)
from velimir.settings import (
    RHYME_ERRORS_CSV,
    RHYME_MODEL,
    RHYME_PAIRS_DB_PATH,
)


def log_metrics(metrics: dict) -> None:
    tp = metrics["true_positives"]
    tn = metrics["true_negatives"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    logging.info("true_positives=%d true_negatives=%d", tp, tn)
    logging.info("false_positives=%d false_negatives=%d", fp, fn)
    logging.info(
        "accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        accuracy,
        precision,
        recall,
        f1,
        metrics["auc"],
    )


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    rows = load_rhyme_pairs(RHYME_PAIRS_DB_PATH)
    _, _, test_rows = split_pairs(rows)

    if not test_rows:
        raise ValueError("Rhyme test split is empty")

    loader = get_rhyme_pair_loader(
        test_rows,
        shuffle=False,
        batch_size=1024,
        num_workers=0,
    )

    model = RhymePairModel().to(device)
    model.load_state_dict(torch.load(RHYME_MODEL, map_location=device))
    model.eval()

    probs, labels = predict_rhyme_probs(model, loader, device)

    labels_np = labels.numpy()
    probs_np = probs.numpy()

    log_metrics(rhyme_metrics(labels_np, probs_np))

    errors = write_rhyme_errors(
        RHYME_ERRORS_CSV,
        RHYME_PAIRS_DB_PATH,
        test_rows,
        labels_np,
        probs_np,
    )
    logging.info("Wrote %d errors to %s", errors, RHYME_ERRORS_CSV)


if __name__ == "__main__":
    LoggingSettings.setup()
    evaluate()
