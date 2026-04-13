import logging

import torch

from velimir.io import load_models, load_poems_from_msgpack
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_samples
from velimir.settings import LoggingSettings
from velimir.evaluation import evaluate_models


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model, meter_model = load_models(device)

    poems = load_poems_from_msgpack()
    _, _, test_set = split_samples(fetch_raw_samples(poems))

    evaluation_results = evaluate_models(accent_model, meter_model, test_set)

    for k, v in evaluation_results.items():
        logging.info("%s=%f", k, v)


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    evaluate()
