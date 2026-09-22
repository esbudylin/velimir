import logging

import torch

from velimir.evaluation import (
    evaluate_models,
    init_db,
    predict_rhyme_probs,
    rhyme_metrics,
    write_rhyme_errors,
)
from velimir.io import load_poems_from_msgpack, load_models
from velimir.logger import LoggingSettings
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.onnx import load_onnx_models, load_rhyme_onnx_model
from velimir.rhyme_ml import RhymePairModel
from velimir.rhyme_ml_loader import (
    get_rhyme_pair_loader,
    load_rhyme_pairs,
    split_pairs,
)
from velimir.settings import RHYME_ERRORS_CSV, RHYME_MODEL, RHYME_PAIRS_DB_PATH


def run_evaluation(meter_model, accent_model, device, test_chunks):
    conn = init_db(":memory:")
    results = evaluate_models(meter_model, accent_model, device, test_chunks, conn)
    conn.commit()
    conn.close()
    return results


def verify():
    device = torch.device("cpu")

    meter_pt, accent_pt = load_models(device)
    meter_onnx, accent_onnx = load_onnx_models()

    poems = load_poems_from_msgpack()
    _, _, test_chunks = split_chunks(fetch_raw_samples(poems))

    logging.info("=== PyTorch Evaluation ===")
    meter_pt.eval()
    accent_pt.eval()
    results_pt = run_evaluation(meter_pt, accent_pt, device, test_chunks)
    for k, v in results_pt.items():
        logging.info("%s=%f", k, v)

    logging.info("=== ONNX Evaluation ===")
    results_onnx = run_evaluation(meter_onnx, accent_onnx, device, test_chunks)
    for k, v in results_onnx.items():
        logging.info("%s=%f", k, v)

    logging.info("=== Accuracy Comparison ===")
    for k in results_pt:
        diff = abs(results_pt[k] - results_onnx[k])
        logging.info(
            "%s: pt=%.6f onnx=%.6f diff=%.6f",
            k,
            results_pt[k],
            results_onnx[k],
            diff,
        )


def verify_rhyme():
    device = torch.device("cpu")

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

    model_pt = RhymePairModel().to(device)
    model_pt.load_state_dict(torch.load(RHYME_MODEL, map_location=device))
    model_pt.eval()

    model_onnx = load_rhyme_onnx_model()

    logging.info("=== Rhyme PyTorch Evaluation ===")
    probs_pt, labels = predict_rhyme_probs(model_pt, loader, device)
    results_pt = rhyme_metrics(labels.numpy(), probs_pt.numpy())
    for k, v in results_pt.items():
        logging.info("%s=%f", k, v)

    logging.info("=== Rhyme ONNX Evaluation ===")
    probs_onnx, _ = predict_rhyme_probs(model_onnx, loader, device)
    results_onnx = rhyme_metrics(labels.numpy(), probs_onnx.numpy())
    for k, v in results_onnx.items():
        logging.info("%s=%f", k, v)

    logging.info("=== Rhyme Metrics Comparison ===")
    for k in results_pt:
        diff = abs(results_pt[k] - results_onnx[k])
        logging.info(
            "%s: pt=%.6f onnx=%.6f diff=%.6f",
            k,
            results_pt[k],
            results_onnx[k],
            diff,
        )

    logging.info("=== Rhyme Probability Comparison ===")
    logging.info(
        "max probability diff: %.6g",
        (probs_pt - probs_onnx).abs().max().item(),
    )

    errors = write_rhyme_errors(
        RHYME_ERRORS_CSV,
        RHYME_PAIRS_DB_PATH,
        test_rows,
        labels.numpy(),
        probs_pt.numpy(),
    )
    logging.info("Wrote %d rhyme errors to %s", errors, RHYME_ERRORS_CSV)


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    verify()
    verify_rhyme()
