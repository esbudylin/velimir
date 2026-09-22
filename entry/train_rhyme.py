import argparse
import logging

import torch

from velimir.logger import LoggingSettings
from velimir.rhyme_ml import train_rhyme_model
from velimir.rhyme_ml_loader import load_rhyme_pairs, split_pairs
from velimir.settings import (
    RHYME_MODEL,
    RHYME_PAIRS_DB_PATH,
    RHYME_PAIRS_TEST_DB_PATH,
    RHYME_TEST_MODEL,
)

SEED = 1046019622


def train(test_run: bool = False, **training_kwargs):
    db_path = RHYME_PAIRS_TEST_DB_PATH if test_run else RHYME_PAIRS_DB_PATH
    model_path = RHYME_TEST_MODEL if test_run else RHYME_MODEL

    if test_run:
        training_kwargs.setdefault("max_epochs", 5)
        training_kwargs.setdefault("batch_size", 8)
        training_kwargs.setdefault("num_workers", 0)

    rows = load_rhyme_pairs(db_path)

    train_rows, val_rows, _ = split_pairs(rows)

    state_dict, val_loss, epochs = train_rhyme_model(
        train_rows,
        val_rows,
        seed=SEED,
        **training_kwargs,
    )

    logging.info("Best validation loss %.4f after %d epochs", val_loss, epochs)

    torch.save(state_dict, model_path)
    logging.info("Rhyme model saved to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the rhyme pair model.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run training on a small subset of data for testing purposes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size for training",
    )
    args = parser.parse_args()

    LoggingSettings.setup()

    train(
        test_run=args.test_run,
        **{k: v for k, v in vars(args).items() if k != "test_run"},
    )
