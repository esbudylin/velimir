import argparse
import logging
from itertools import islice

import torch

from velimir.io import load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.ml import train_unified_model
from velimir.ml_loader import (
    MeterClassRegistry,
    fetch_raw_samples,
    split_chunks,
)
from velimir.settings import (
    UNIFIED_MODEL,
    UNIFIED_TEST_MODEL,
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

    raw_samples = list(raw_samples)

    train_chunks, val_chunks, _ = split_chunks(raw_samples)

    logging.info("Training unified model...")
    state_dict = train_unified_model(
        train_chunks,
        val_chunks,
        **training_kwargs,
    )

    logging.info("Saving trained model...")

    if test_run:
        torch.save(state_dict, UNIFIED_TEST_MODEL)
    else:
        torch.save(state_dict, UNIFIED_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unified accent/meter model.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    args = parser.parse_args()

    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    train(test_run=args.test_run)
