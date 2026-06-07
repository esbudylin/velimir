import logging

import torch

from velimir.io import load_poems_from_msgpack, load_unified_model
from velimir.ml_loader import MeterClassRegistry, fetch_raw_samples, split_chunks
from velimir.logger import LoggingSettings
from velimir.evaluation import evaluate_unified, init_db


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = load_unified_model(device)
    model.eval()

    poems = load_poems_from_msgpack()
    _, _, test_chunks = split_chunks(fetch_raw_samples(poems))

    conn = init_db()

    results = evaluate_unified(model, device, test_chunks, conn)
    for k, v in results.items():
        logging.info("%s=%f", k, v)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    LoggingSettings.setup()
    MeterClassRegistry.initialize()

    evaluate()
