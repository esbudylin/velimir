import argparse
import logging
from itertools import islice

import torch

from velimir.io import load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.ml import train_models
from velimir.ml_loader import (
    MeterClassRegistry,
    fetch_raw_samples,
    split_chunks,
)
from velimir.settings import (
    METER_MODEL,
    METER_TEST_MODEL,
    ACCENT_MODEL,
    ACCENT_TEST_MODEL,
)


def train(test_run: bool = False):
    logging.info("Loading poems from msgpack")

    training_kwargs = {}

    poems = load_poems_from_msgpack()
    raw_samples = fetch_raw_samples(poems)

    if test_run:
        testing_subset = 1000

        training_kwargs["max_epochs"] = 5
        training_kwargs["batch_size"] = 8

        logging.info(
            "Test run enabled: using a small subset (%d) of lines",
            testing_subset,
        )
        raw_samples = islice(raw_samples, testing_subset)

    train_chunks, val_chunks, _ = split_chunks(raw_samples)

    accent_state_dict, meter_state_dict = train_models(
        train_chunks,
        val_chunks,
        **training_kwargs,
    )

    logging.info("Saving trained models...")

    if test_run:
        torch.save(meter_state_dict, METER_TEST_MODEL)
        torch.save(accent_state_dict, ACCENT_TEST_MODEL)
    else:
        torch.save(meter_state_dict, METER_MODEL)
        torch.save(accent_state_dict, ACCENT_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train meter and accent models.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    args = parser.parse_args()

    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    train(test_run=args.test_run)
