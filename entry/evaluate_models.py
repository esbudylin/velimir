import logging

import torch

from velimir.io import load_models, load_poems_from_msgpack
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.logger import LoggingSettings
from velimir.evaluation import evaluate_models, evaluate_refiner_models, init_db
from velimir.identifier import logic_refine


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    accent_model, meter_model, _refiner_model = load_models(device)
    accent_model.eval()
    meter_model.eval()

    poems = load_poems_from_msgpack()
    _, _, test_chunks = split_chunks(fetch_raw_samples(poems))
    test_set = [rs for chunk in test_chunks for rs in chunk]

    conn = init_db()

    base_results = evaluate_models(
        accent_model,
        meter_model,
        device,
        test_set,
        conn,
    )
    for k, v in base_results.items():
        logging.info("%s=%f", k, v)

    refiner_results = evaluate_refiner_models(
        accent_model,
        meter_model,
        device,
        test_chunks,
        conn,
        refiner=logic_refine,
    )
    for k, v in refiner_results.items():
        logging.info("%s=%f", k, v)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    evaluate()
