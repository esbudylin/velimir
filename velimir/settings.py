import csv
import os

DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs"

METADATA_TABLE = os.path.join(DATA_DIRECTORY, "rnc", "tables", "poetic.csv")
TEXTS_DIR = os.path.join(DATA_DIRECTORY, "rnc", "texts")
OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "poems.msgpack")
MODELS_DIRECTORY = os.path.join(DATA_DIRECTORY, "models")

ACCENT_MODEL = os.path.join(MODELS_DIRECTORY, "accent")
METER_MODEL = os.path.join(MODELS_DIRECTORY, "meter")
REFINER_MODEL = os.path.join(MODELS_DIRECTORY, "refiner")

ACCENT_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "accent-test")
METER_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "meter-test")

ACCENT_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "accent.onnx")
METER_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "meter.onnx")
REFINER_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "refiner.onnx")

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
