import logging

import torch

from velimir.evaluation import evaluate_models
from velimir.io import load_models, load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.ml_loader import (
    MeterClassRegistry,
    fetch_raw_samples,
    get_loader,
    split_samples,
)
from velimir.onnx import load_onnx_models


def verify_single_batch(accent_pt, meter_pt, accent_onnx, meter_onnx, batch):
    accent_input = batch.accent_input
    pos_input = batch.part_of_speech_input

    with torch.no_grad():
        meter_pt_out = meter_pt(accent_input, pos_input)
        meter_onnx_out = meter_onnx(accent_input, pos_input)
        meter_pred = torch.argmax(meter_pt_out, dim=1)
        accent_pt_out = accent_pt(accent_input, meter_pred, pos_input)
        accent_onnx_out = accent_onnx(accent_input, meter_pred, pos_input)

    meter_diff = (meter_pt_out - meter_onnx_out).abs()
    accent_diff = (accent_pt_out - accent_onnx_out).abs()

    mask = accent_input[:, :, 0] != -1

    logging.info("=== Numerical Verification ===")
    logging.info(
        "Meter logits:  max_diff=%.6f  mean_diff=%.6f",
        meter_diff.max().item(),
        meter_diff.mean().item(),
    )
    logging.info(
        "Accent logits: max_diff=%.6f  mean_diff=%.6f",
        accent_diff.max().item(),
        accent_diff.mean().item(),
    )
    logging.info(
        "Accent logits (non-padded): max_diff=%.6f  mean_diff=%.6f",
        accent_diff[mask].max().item(),
        accent_diff[mask].mean().item(),
    )

    meter_onnx_pred = torch.argmax(meter_onnx_out, dim=1)
    meter_agreement = (meter_pred == meter_onnx_pred).float().mean().item()
    logging.info("Meter prediction agreement on batch: %.4f", meter_agreement)

    accent_pred_pt = (torch.sigmoid(accent_pt_out) > 0.5).float()
    accent_pred_onnx = (torch.sigmoid(accent_onnx_out) > 0.5).float()
    accent_agreement = (
        (accent_pred_pt[mask] == accent_pred_onnx[mask]).float().mean().item()
    )
    logging.info("Accent prediction agreement on batch: %.4f", accent_agreement)


def verify():
    device = torch.device("cpu")

    accent_pt, meter_pt = load_models(device)
    accent_onnx, meter_onnx = load_onnx_models()

    poems = load_poems_from_msgpack()
    _, _, test_set = split_samples(fetch_raw_samples(poems))

    loader = get_loader(test_set, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    verify_single_batch(accent_pt, meter_pt, accent_onnx, meter_onnx, batch)

    logging.info("=== PyTorch Evaluation ===")
    accent_pt.eval()
    meter_pt.eval()
    results_pt = evaluate_models(accent_pt, meter_pt, device, test_set)
    for k, v in results_pt.items():
        logging.info("%s=%f", k, v)

    logging.info("=== ONNX Evaluation ===")
    results_onnx = evaluate_models(accent_onnx, meter_onnx, device, test_set)
    for k, v in results_onnx.items():
        logging.info("%s=%f", k, v)

    logging.info("=== Accuracy Comparison ===")
    for k in results_pt:
        diff = abs(results_pt[k] - results_onnx[k])
        logging.info(
            "%s: pt=%.6f onnx=%.6f diff=%.6f", k, results_pt[k], results_onnx[k], diff
        )


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    verify()
