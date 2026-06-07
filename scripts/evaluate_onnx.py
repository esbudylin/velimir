import logging

import torch

from velimir.evaluation import evaluate_models, init_db
from velimir.io import load_poems_from_msgpack, load_models
from velimir.logger import LoggingSettings
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.onnx import load_onnx_models


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


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    verify()
