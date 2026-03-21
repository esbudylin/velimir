import logging
import argparse
from itertools import islice

import torch

from src.io import load_poems_from_msgpack
from src.ml import split_poems, train_models
from src.model_validation import validate_models
from src.settings import (
    LoggingSettings,
    ACCENT_MODEL,
    METER_MODEL,
    ACCENT_TEST_MODEL,
    METER_TEST_MODEL,
)


def main(test_run: bool = False):
    logging.basicConfig(**LoggingSettings().model_dump())

    logging.info("Loading poems from msgpack")
    poems = load_poems_from_msgpack()

    if test_run:
        testing_subset = 100
        logging.info(
            "Test run enabled: using a small subset (%d) of poems",
            testing_subset,
        )
        poems = islice(poems, testing_subset)

    training_set, test_set = split_poems(poems)

    logging.info("Training is starting...")
    accent_model, meter_model = train_models(training_set)
    validation_results = validate_models(accent_model, meter_model, test_set)

    logging.info("\n".join(f"{k}={v}" for k, v in validation_results.items()))

    logging.info("Saving trained models...")

    if test_run:
        torch.save(accent_model.state_dict(), ACCENT_TEST_MODEL)
        torch.save(meter_model.state_dict(), METER_TEST_MODEL)
    else:
        torch.save(accent_model.state_dict(), ACCENT_MODEL)
        torch.save(meter_model.state_dict(), METER_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train accent/meter models.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
