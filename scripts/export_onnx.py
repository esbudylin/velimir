import logging

import torch

from velimir.io import load_models
from velimir.logger import LoggingSettings
from velimir.ml_preprocess import MeterClassRegistry
from velimir.settings import ACCENT_ONNX_MODEL, METER_ONNX_MODEL

MAX_SEQ_LEN = 128


def make_dummy_accent_inputs():
    B = 2
    accent_input = torch.zeros(B, MAX_SEQ_LEN, 3)
    meter_class = torch.zeros(B, dtype=torch.long)
    pos_input = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    return accent_input, meter_class, pos_input


def make_dummy_meter_inputs():
    B = 2
    accent_input = torch.zeros(B, MAX_SEQ_LEN, 3)
    pos_input = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    return accent_input, pos_input


def export_accent(accent_model):
    dummy_accent_input, dummy_meter_class, dummy_pos_input = make_dummy_accent_inputs()

    logging.info("Exporting accent model to %s (T=%d)", ACCENT_ONNX_MODEL, MAX_SEQ_LEN)

    torch.onnx.export(
        accent_model,
        (dummy_accent_input, dummy_meter_class, dummy_pos_input),
        ACCENT_ONNX_MODEL,
        input_names=["accent_input", "meter_class", "pos_input"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={
            "accent_input": {0: "batch"},
            "meter_class": {0: "batch"},
            "pos_input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    logging.info("Accent model exported successfully")


def export_meter(meter_model):
    dummy_accent_input, dummy_pos_input = make_dummy_meter_inputs()

    logging.info("Exporting meter model to %s (T=%d)", METER_ONNX_MODEL, MAX_SEQ_LEN)

    torch.onnx.export(
        meter_model,
        (dummy_accent_input, dummy_pos_input),
        METER_ONNX_MODEL,
        input_names=["accent_input", "pos_input"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={
            "accent_input": {0: "batch"},
            "pos_input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    logging.info("Meter model exported successfully")


def export_models():
    device = torch.device("cpu")

    accent_model, meter_model = load_models(device)

    export_accent(accent_model)
    export_meter(meter_model)


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    export_models()
