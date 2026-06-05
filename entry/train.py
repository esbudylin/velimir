import argparse
import logging
from itertools import islice

import torch

from velimir.io import load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.ml import train_models, train_refiner_model
from velimir.ml_loader import (
    MeterClassRegistry,
    fetch_raw_samples,
    split_lines,
    split_chunks,
)
from velimir.settings import (
    ACCENT_MODEL,
    ACCENT_TEST_MODEL,
    METER_MODEL,
    METER_TEST_MODEL,
    REFINER_MODEL,
    REFINER_TEST_MODEL,
)


def train(test_run: bool = False, refiner_only: bool = False):
    logging.info("Loading poems from msgpack")

    training_kwargs = {}
    refiner_training_kwargs = {}

    poems = load_poems_from_msgpack()
    raw_samples = fetch_raw_samples(poems)

    if test_run:
        testing_subset = 1000

        training_kwargs["batch_size"] = 128
        training_kwargs["max_epochs"] = 5

        refiner_training_kwargs["max_epochs"] = 5

        logging.info(
            "Test run enabled: using a small subset (%d) of lines",
            testing_subset,
        )
        raw_samples = islice(raw_samples, testing_subset)

    raw_samples = list(raw_samples)

    train_chunks, val_chunks, _ = split_chunks(raw_samples)
    training_set, validation_set, _ = split_lines(raw_samples)

    if not refiner_only:
        logging.info("Training is starting...")
        accent_state_dict, meter_state_dict = train_models(
            training_set,
            validation_set,
            **training_kwargs,
        )

        logging.info("Saving accent/meter models...")

        if test_run:
            torch.save(accent_state_dict, ACCENT_TEST_MODEL)
            torch.save(meter_state_dict, METER_TEST_MODEL)
        else:
            torch.save(accent_state_dict, ACCENT_MODEL)
            torch.save(meter_state_dict, METER_MODEL)
    else:
        logging.info("Skipping accent/meter training, loading pre-trained meter model...")
        meter_model_path = METER_TEST_MODEL if test_run else METER_MODEL
        meter_state_dict = torch.load(meter_model_path)

    logging.info("Training refiner model...")
    refiner_state_dict = train_refiner_model(
        train_chunks,
        val_chunks,
        meter_state_dict,
        **refiner_training_kwargs,
    )

    logging.info("Saving refiner model...")

    if test_run:
        torch.save(refiner_state_dict, REFINER_TEST_MODEL)
    else:
        torch.save(refiner_state_dict, REFINER_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train accent/meter models.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    parser.add_argument(
        "--refiner-only",
        action="store_true",
        help="Skip accent/meter training and train only the refiner model",
    )
    args = parser.parse_args()

    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    train(test_run=args.test_run, refiner_only=args.refiner_only)
