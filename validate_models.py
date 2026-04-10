import logging

import torch

from velimir.io import load_models, load_poems_from_msgpack
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_samples
from velimir.settings import LoggingSettings
from velimir.validation import validate_models


def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model, meter_model = load_models(device)

    poems = load_poems_from_msgpack()
    _, _, test_set = split_samples(fetch_raw_samples(poems))

    validation_results = validate_models(accent_model, meter_model, test_set)

    for k, v in validation_results.items():
        logging.info("%s=%f", k, v)


if __name__ == "__main__":
    logging.basicConfig(**LoggingSettings().model_dump())
    MeterClassRegistry.initialize()

    validate()
