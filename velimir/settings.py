import csv
import os

DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs"

METADATA_TABLE = os.path.join(DATA_DIRECTORY, "rnc", "tables", "poetic.csv")
TEXTS_DIR = os.path.join(DATA_DIRECTORY, "rnc", "texts")
OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "poems.msgpack")
MODELS_DIRECTORY = os.path.join(DATA_DIRECTORY, "models")

UNIFIED_MODEL = os.path.join(MODELS_DIRECTORY, "unified")
UNIFIED_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "unified-test")
UNIFIED_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "unified.onnx")

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
