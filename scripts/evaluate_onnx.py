import logging

import torch

from velimir.evaluation import evaluate_unified, init_db
from velimir.io import load_poems_from_msgpack, load_unified_model
from velimir.logger import LoggingSettings
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.onnx import load_onnx_models


def run_evaluation(model, device, test_chunks):
    conn = init_db(":memory:")
    results = evaluate_unified(model, device, test_chunks, conn)
    conn.commit()
    conn.close()
    return results


def verify():
    device = torch.device("cpu")

    unified_pt = load_unified_model(device)
    unified_onnx = load_onnx_models()

    poems = load_poems_from_msgpack()
    _, _, test_chunks = split_chunks(fetch_raw_samples(poems))

    logging.info("=== PyTorch Evaluation ===")
    unified_pt.eval()
    results_pt = run_evaluation(unified_pt, device, test_chunks)
    for k, v in results_pt.items():
        logging.info("%s=%f", k, v)

    logging.info("=== ONNX Evaluation ===")
    results_onnx = run_evaluation(unified_onnx, device, test_chunks)
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


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    verify()
