import csv
import logging
import os
from dataclasses import asdict, dataclass

DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs"

METADATA_TABLE = os.path.join(DATA_DIRECTORY, "rnc", "tables", "poetic.csv")
TEXTS_DIR = os.path.join(DATA_DIRECTORY, "rnc", "texts")
OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "poems.msgpack")
MODELS_DIRECTORY = os.path.join(DATA_DIRECTORY, "models")

ACCENT_MODEL = os.path.join(MODELS_DIRECTORY, "accent")
METER_MODEL = os.path.join(MODELS_DIRECTORY, "meter")

ACCENT_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "accent-test")
METER_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "meter-test")

ACCENT_DICT_DIR = os.path.join(DATA_DIRECTORY, "accent_dicts")
ACCENT_DICT_PATHS = [
    os.path.join(ACCENT_DICT_DIR, n)
    for n in ["accent.dic", "accent1.dic", "accent2.dic"]
]
PREDICTION_DB_PATH = os.path.join(DATA_DIRECTORY, "predictions.db")
GRAMMAR_DB_PATH = os.path.join(DATA_DIRECTORY, "grammar.db")
GRAMMAR_TEST_DB_PATH = os.path.join(DATA_DIRECTORY, "grammar_test.db")

METER_VOCAB_PATH = os.path.join(DATA_DIRECTORY, "meter_vocab.jsonl")


class InputDialect(csv.unix_dialect):
    delimiter = ";"


@dataclass
class LoggingSettings:
    filename: str = "main.log"
    encoding: str = "utf-8"
    level: int = logging.INFO
    filemode: str = "w"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"

    @classmethod
    def setup(cls):
        config = asdict(cls())

        log_file = os.environ.get("LOG_FILE")
        if log_file is not None:
            config["filename"] = log_file

        log_dir = os.path.dirname(config["filename"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(**config)
