import argparse
import logging
import random
from itertools import islice

import torch

from velimir.domain_models import Poem
from velimir.io import load_models, load_poems_from_msgpack
from velimir.ml import train_models
from velimir.settings import (
    ACCENT_MODEL,
    ACCENT_TEST_MODEL,
    METER_MODEL,
    METER_TEST_MODEL,
    LoggingSettings,
)
from velimir.validation import validate_models


def split_poems(
    poems,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list, list]:
    poems_l = list(poems)

    rng = random.Random(42)
    rng.shuffle(poems_l)

    split = int(len(poems_l) * (1 - test_ratio))

    train_poems = poems_l[:split]
    test_poems = poems_l[split:]

    return train_poems, test_poems


def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model, meter_model = load_models(device)

    poems = load_poems_from_msgpack()
    _, test_set = split_poems(poems)

    validation_results = validate_models(accent_model, meter_model, test_set)

    for k, v in validation_results.items():
        logging.info("%s=%f", k, v)


def train(test_run: bool = False):
    logging.info("Loading poems from msgpack")
    poems = load_poems_from_msgpack()

    if test_run:
        testing_subset = 100
        logging.info(
            "Test run enabled: using a small subset (%d) of poems",
            testing_subset,
        )
        poems = islice(poems, testing_subset)

    training_set, _ = split_poems(poems)

    logging.info("Training is starting...")
    accent_model, meter_model = train_models(training_set)

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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform model validation on test data",
    )
    args = parser.parse_args()

    logging.basicConfig(**LoggingSettings().model_dump())

    if args.validate:
        validate()
    else:
        train(test_run=args.test_run)
