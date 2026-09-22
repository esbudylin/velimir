import csv
import os

DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs"

DATASETS_DIRECTORY = os.path.join(DATA_DIRECTORY, "datasets")

METADATA_TABLE = os.path.join(DATA_DIRECTORY, "rnc", "tables", "poetic.csv")
TEXTS_DIR = os.path.join(DATA_DIRECTORY, "rnc", "texts")
OUTPUT_FILE = os.path.join(DATASETS_DIRECTORY, "poems.msgpack")
MODELS_DIRECTORY = os.path.join(DATA_DIRECTORY, "models")

METER_MODEL = os.path.join(MODELS_DIRECTORY, "meter")
METER_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "meter-test")
METER_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "meter.onnx")

ACCENT_MODEL = os.path.join(MODELS_DIRECTORY, "accent")
ACCENT_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "accent-test")
ACCENT_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "accent.onnx")

ACCENT_DICT_DIR = os.path.join(DATA_DIRECTORY, "accent_dicts")
ACCENT_DICT_PATHS = [
    os.path.join(ACCENT_DICT_DIR, n)
    for n in ["accent.dic", "accent1.dic", "accent2.dic"]
]
PREDICTION_DB_PATH = os.path.join(DATA_DIRECTORY, "predictions.db")
GRAMMAR_DB_PATH = os.path.join(DATASETS_DIRECTORY, "grammar.db")
GRAMMAR_TEST_DB_PATH = os.path.join(DATASETS_DIRECTORY, "grammar_test.db")

RHYME_DB_PATH = os.path.join(DATASETS_DIRECTORY, "rhyme.db")
RHYME_TEST_DB_PATH = os.path.join(DATASETS_DIRECTORY, "rhyme_test.db")

RHYME_PAIRS_DB_PATH = os.path.join(DATASETS_DIRECTORY, "rhyme_pairs.db")
RHYME_PAIRS_TEST_DB_PATH = os.path.join(DATASETS_DIRECTORY, "rhyme_pairs_test.db")
RHYME_MODEL = os.path.join(MODELS_DIRECTORY, "rhyme")
RHYME_TEST_MODEL = os.path.join(MODELS_DIRECTORY, "rhyme-test")
RHYME_ONNX_MODEL = os.path.join(MODELS_DIRECTORY, "rhyme.onnx")
RHYME_ERRORS_CSV = os.path.join(DATA_DIRECTORY, "rhyme_errors.csv")

DATASET_FILES = [
    os.path.basename(GRAMMAR_DB_PATH),
    os.path.basename(OUTPUT_FILE),
    os.path.basename(RHYME_DB_PATH),
    os.path.basename(RHYME_PAIRS_DB_PATH),
]

METER_VOCAB_PATH = os.path.join(DATA_DIRECTORY, "meter_vocab.jsonl")


class InputDialect(csv.unix_dialect):
    delimiter = ";"
