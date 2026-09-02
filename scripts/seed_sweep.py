import argparse
import json
import logging
import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch

from velimir.evaluation import evaluate_models, init_db
from velimir.io import load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.ml import AccentModel, MeterModel, train_models
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks

RUNS = 10
BASE_SEED = 425065

OUTPUT_DIR = Path("data/models/sweep")
TEST_OUTPUT_DIR = Path("data/models/sweep-test")

TEST_SUBSET = 1000


def load_and_eval(accent_path, meter_path, test_chunks, device, batch_size=None):
    accent_model = AccentModel().to(device)
    accent_model.load_state_dict(torch.load(accent_path, map_location=device))
    accent_model.eval()

    meter_model = MeterModel().to(device)
    meter_model.load_state_dict(torch.load(meter_path, map_location=device))
    meter_model.eval()

    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    conn = init_db(":memory:")
    eval_results = evaluate_models(
        meter_model,
        accent_model,
        device,
        test_chunks,
        conn,
        **kwargs,
    )
    conn.close()
    return eval_results


def main(test_run: bool = False):
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    output_dir = TEST_OUTPUT_DIR if test_run else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"

    if results_path.exists():
        with open(results_path) as f:
            results = [json.loads(line) for line in f if line.strip()]
    else:
        results = []

    completed_runs = {r["run"] for r in results}

    logging.info("Loading data...")
    poems = load_poems_from_msgpack()
    raw_samples = fetch_raw_samples(poems)

    if test_run:
        logging.info("Test run: limiting to %d samples", TEST_SUBSET)
        raw_samples = islice(raw_samples, TEST_SUBSET)
        training_kwargs = {"max_epochs": 2, "batch_size": 8}
    else:
        training_kwargs = {}

    train_chunks, val_chunks, test_chunks = split_chunks(raw_samples)

    for run_id in range(RUNS):
        if run_id in completed_runs:
            logging.info("Run %d already complete, skipping", run_id)
            continue

        run_rng = random.Random(BASE_SEED + run_id)
        accent_seed = run_rng.randint(0, 2**31 - 1)
        meter_seed = run_rng.randint(0, 2**31 - 1)

        logging.info(
            "=== Run %d/%d: accent_seed=%d, meter_seed=%d ===",
            run_id + 1,
            RUNS,
            accent_seed,
            meter_seed,
        )

        accent_state_dict, meter_state_dict, metrics = train_models(
            train_chunks,
            val_chunks,
            accent_seed=accent_seed,
            meter_seed=meter_seed,
            **training_kwargs,
        )

        run_dir = output_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(accent_state_dict, run_dir / "accent")
        torch.save(meter_state_dict, run_dir / "meter")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eval_results = load_and_eval(
            run_dir / "accent",
            run_dir / "meter",
            test_chunks,
            device,
            batch_size=training_kwargs.get("batch_size"),
        )

        run_result = {
            "run": run_id,
            "accent_seed": accent_seed,
            "meter_seed": meter_seed,
            **eval_results,
            **asdict(metrics),
        }

        results.append(run_result)

        with open(results_path, "a") as f:
            f.write(json.dumps(run_result) + "\n")

        logging.info(
            "Run %d complete: %s",
            run_id,
            json.dumps(
                {
                    k: round(v, 3) if isinstance(v, float) else v
                    for k, v in run_result.items()
                }
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training N times with different seeds to find optimal seeds"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Use small data subset and few epochs for testing",
    )
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Re-evaluate saved models and update results.jsonl with new metrics",
    )
    args = parser.parse_args()

    main(args.test_run)
