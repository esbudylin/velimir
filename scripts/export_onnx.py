import logging

import torch

from velimir.io import load_unified_model
from velimir.logger import LoggingSettings
from velimir.ml_preprocess import MeterClassRegistry
from velimir.onnx import MAX_SEQ_LEN
from velimir.settings import UNIFIED_ONNX_MODEL


def make_dummy_inputs():
    N = 18
    accent_input = torch.zeros(N, MAX_SEQ_LEN, 2)
    pos_input = torch.zeros(N, MAX_SEQ_LEN, dtype=torch.long)
    return accent_input, pos_input


def export():
    device = torch.device("cpu")

    model = load_unified_model(device)

    dummy_accent, dummy_pos = make_dummy_inputs()

    logging.info(
        "Exporting unified model to %s (N=%d, T=%d)",
        UNIFIED_ONNX_MODEL,
        18,
        MAX_SEQ_LEN,
    )

    torch.onnx.export(
        model,
        (dummy_accent, dummy_pos),
        UNIFIED_ONNX_MODEL,
        input_names=["accent_input", "pos_input"],
        output_names=["meter_logits", "accent_logits"],
        dynamo=False,
        dynamic_axes={
            "accent_input": {0: "batch_size"},
            "pos_input": {0: "batch_size"},
            "meter_logits": {0: "batch_size"},
            "accent_logits": {0: "batch_size"},
        },
    )

    logging.info("Unified model exported successfully")


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    export()
