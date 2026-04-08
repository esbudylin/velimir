import csv
import os
import logging

from pydantic_settings import BaseSettings

DATA_DIRECTORY = "data"

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

METER_VOCAB_PATH = os.path.join(DATA_DIRECTORY, "meter_vocab.jsonl")


class InputDialect(csv.unix_dialect):
    delimiter = ";"


class LoggingSettings(BaseSettings):
    filename: str = "main.log"
    encoding: str = "utf-8"
    level: int = logging.INFO
    filemode: str = "w"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"

    class Config:
        env_prefix = "LOGGING_"
