import csv
import os
import logging

from pydantic_settings import BaseSettings

DATA_FOLDER = "data"

METADATA_TABLE = os.path.join(DATA_FOLDER, "rnc", "tables", "poetic.csv")
TEXTS_DIR = os.path.join(DATA_FOLDER, "rnc", "texts")
OUTPUT_FILE = os.path.join(DATA_FOLDER, "poems.msgpack")


class InputDialect(csv.unix_dialect):
    delimiter = ";"


class LoggingSettings(BaseSettings):
    filename: str = "main.log"
    encoding: str = "utf-8"
    level: int = logging.INFO
    filemode: str = "w"

    class Config:
        env_prefix = "LOGGING_"
