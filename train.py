import argparse
import logging
from itertools import islice

import torch

from velimir.io import load_poems_from_msgpack
from velimir.ml import train_models
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_samples
from velimir.settings import (
    ACCENT_MODEL,
    ACCENT_TEST_MODEL,
    METER_MODEL,
    METER_TEST_MODEL,
    LoggingSettings,
)


def train(test_run: bool = False):
    logging.info("Loading poems from msgpack")
    training_kwargs = {}

    poems = load_poems_from_msgpack()
    raw_samples = fetch_raw_samples(poems)

    if test_run:
        testing_subset = 1000

        training_kwargs["batch_size"] = 128
        training_kwargs["max_epochs"] = 5

        logging.info(
            "Test run enabled: using a small subset (%d) of lines",
            testing_subset,
        )
        raw_samples = islice(raw_samples, testing_subset)

    training_set, validation_set, _ = split_samples(raw_samples)

    logging.info("Training is starting...")
    accent_state_dict, meter_state_dict = train_models(
        training_set,
        validation_set,
        **training_kwargs,
    )

    logging.info("Saving trained models...")

    if test_run:
        torch.save(accent_state_dict, ACCENT_TEST_MODEL)
        torch.save(meter_state_dict, METER_TEST_MODEL)
    else:
        torch.save(accent_state_dict, ACCENT_MODEL)
        torch.save(meter_state_dict, METER_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train accent/meter models.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    args = parser.parse_args()

    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    train(test_run=args.test_run)
