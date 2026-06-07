import logging

import numpy as np
import torch

from velimir.io import load_models
from velimir.logger import LoggingSettings
from velimir.ml_preprocess import MeterClassRegistry
from velimir.onnx import MAX_SEQ_LEN
from velimir.settings import METER_ONNX_MODEL, ACCENT_ONNX_MODEL


def make_dummy_inputs():
    N = 18
    accent_input = torch.zeros(N, MAX_SEQ_LEN, 2)
    pos_input = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.long)
    meter_target = torch.zeros(N, dtype=torch.long)
    return accent_input, pos_input, meter_target


def export():
    device = torch.device("cpu")

    accent_input, pos_input, meter_target = make_dummy_inputs()

    logging.info(
        "Exporting meter model to %s (N=%d, T=%d)",
        METER_ONNX_MODEL,
        18,
        MAX_SEQ_LEN,
    )

    meter_model, accent_model = load_models(device)
    torch.onnx.export(
        meter_model,
        (accent_input, pos_input),
        METER_ONNX_MODEL,
        input_names=["accent_input", "pos_input"],
        output_names=["meter_logits"],
        dynamo=False,
        dynamic_axes={
            "accent_input": {0: "batch_size"},
            "pos_input": {0: "batch_size"},
            "meter_logits": {0: "batch_size"},
        },
    )

    logging.info("Meter model exported successfully")

    logging.info(
        "Exporting accent model to %s (N=%d, T=%d)",
        ACCENT_ONNX_MODEL,
        18,
        MAX_SEQ_LEN,
    )

    torch.onnx.export(
        accent_model,
        (accent_input, pos_input, meter_target),
        ACCENT_ONNX_MODEL,
        input_names=["accent_input", "pos_input", "meter_target"],
        output_names=["accent_logits"],
        dynamo=False,
        dynamic_axes={
            "accent_input": {0: "batch_size"},
            "pos_input": {0: "batch_size"},
            "meter_target": {0: "batch_size"},
            "accent_logits": {0: "batch_size"},
        },
    )

    logging.info("Accent model exported successfully")


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    export()
