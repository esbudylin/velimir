import logging

import torch

from velimir.io import load_poems_from_msgpack, load_models
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.logger import LoggingSettings
from velimir.evaluation import evaluate_models, init_db


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    meter_model, accent_model = load_models(device)

    poems = load_poems_from_msgpack()
    _, _, test_chunks = split_chunks(fetch_raw_samples(poems))

    conn = init_db()

    results = evaluate_models(meter_model, accent_model, device, test_chunks, conn)
    for k, v in results.items():
        logging.info("%s=%f", k, v)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    evaluate()
