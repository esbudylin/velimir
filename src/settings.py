import csv
import os
import logging

from pydantic_settings import BaseSettings

DATA_FOLDER = "data"

METADATA_TABLE = os.path.join(DATA_FOLDER, "tables", "poetic.csv")
TEXSTS_FOLDER = os.path.join(DATA_FOLDER, "texts")

OUTPUT_FOLDER = "output"
OUTPUT_JSON = os.path.join(OUTPUT_FOLDER, "output.json")


class InputDialect(csv.unix_dialect):
    delimiter = ";"


class LoggingSettings(BaseSettings):
    filename: str = os.path.join(OUTPUT_FOLDER, "main.log")
    level: int = logging.INFO
    filemode: str = "w"

    class Config:
        env_prefix = "LOGGING_"
