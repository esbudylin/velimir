import argparse
import csv
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from itertools import islice
from typing import Iterator

from bitarray import bitarray

from velimir import accentuator, domain_models
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import parse_input_lines
from velimir.phonetics import to_phonetic_repr
from velimir.settings import (
    METADATA_TABLE,
    RHYME_DB_PATH,
    RHYME_PAIRS_DB_PATH,
    RHYME_PAIRS_TEST_DB_PATH,
    InputDialect,
)

POSITIVE_QUERY = """
SELECT path, json_group_array(word), json_group_array(accents)
FROM (
    SELECT DISTINCT p.path        AS path,
                    r.poem_id     AS poem_id,
                    r.seq         AS seq,
                    r.rhyme_group AS rhyme_group,
                    r.word        AS word,
                    r.accents     AS accents
    FROM rhymes r
    JOIN poems p ON r.poem_id = p.ROWID
    WHERE r.rhyme_group <> -1
)
GROUP BY poem_id, seq, rhyme_group
HAVING COUNT(*) > 1
"""

SCHEMA = """
CREATE TABLE poems (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL
);

CREATE TABLE pairs (
    id INTEGER PRIMARY KEY,
    poem_id INTEGER NOT NULL REFERENCES poems(id),
    label INTEGER NOT NULL,

    word_a TEXT NOT NULL,
    word_b TEXT NOT NULL,

    phon_a TEXT NOT NULL,
    accents_a TEXT NOT NULL,
    phon_b TEXT NOT NULL,
    accents_b TEXT NOT NULL
);

CREATE INDEX idx_pairs_poem ON pairs(poem_id);
"""


@dataclass(slots=True)
class EncodedEnding:
    word: str
    phon: str
    accents: str


@dataclass(slots=True)
class RhymePair:
    poem_path: str
    label: bool
    word_a: EncodedEnding
    word_b: EncodedEnding


def encode_ending(word: str, accents) -> EncodedEnding:
    phon = to_phonetic_repr(word, accents)

    return EncodedEnding(
        word=word,
        phon=phon.phonetics,
        accents="".join("1" if s else "0" for s in phon.accents),
    )


def clean_ending(text: str) -> str | None:
    word = accentuator.remove_accent_marks(text).lower()
    cleaned = re.sub(r"[^а-я-\sё]", "", word).strip("- ")

    if not cleaned:
        return None

    return cleaned


def encode_raw_ending(text: str) -> EncodedEnding | None:
    cleaned = clean_ending(text)

    if cleaned is None:
        return None

    accents = accentuator.extract_accent_mask(text)

    return encode_ending(cleaned, accents)


def extract_line_ending(line: domain_models.InputLine) -> str:
    if line.rhyme_zone:
        return line.rhyme_zone

    tokens = line.text.split()

    if not tokens:
        return ""

    selected = [tokens[-1]]

    for i in range(len(tokens) - 2, -1, -1):
        if any(accentuator.extract_accent_mask(" ".join(selected))):
            break

        selected.insert(0, tokens[i])

    return " ".join(selected)


def make_positive_pairs(conn: sqlite3.Connection) -> Iterator[RhymePair]:
    for path, words_json, accents_json in conn.execute(POSITIVE_QUERY):
        endings = []

        for word, accents in zip(json.loads(words_json), json.loads(accents_json)):
            ending = encode_ending(word, bitarray(accents))
            endings.append(ending)

        for i, ending in enumerate(endings):
            if i + 1 == len(endings):
                break

            next_ending = endings[i + 1]

            if ending.word == next_ending.word:
                continue

            yield RhymePair(path, True, ending, next_ending)


def make_negative_pairs(
    rows: Iterator[dict],
) -> Iterator[RhymePair]:
    for row in rows:
        poem = domain_models.InputPoem.from_row(row)

        if poem.rhyme.strip() not in ("0", ""):
            continue

        try:
            xml = read_poem_xml(poem.path)
            lines, _ = parse_input_lines(xml, allow_latin=True)
        except Exception as error:
            delayed_logger.record()
            logging.warning("Can't read poem %s: %s", poem.path, error)
            continue

        for i, line in enumerate(lines):
            if i + 1 == len(lines):
                break

            next_line = lines[i + 1]

            ending_text = extract_line_ending(line)
            next_ending_text = extract_line_ending(next_line)

            if not ending_text or not next_ending_text:
                continue

            ending = encode_raw_ending(ending_text)
            next_ending = encode_raw_ending(next_ending_text)

            if ending is None or next_ending is None:
                continue

            if ending.word == next_ending.word:
                continue

            yield RhymePair(poem.path, False, ending, next_ending)


def insert_pairs(
    conn: sqlite3.Connection,
    pairs: Iterator[RhymePair],
    batch_size: int = 50000,
):
    insert_sql = """
        INSERT INTO pairs (
            poem_id, label,
            word_a, word_b,
            phon_a, accents_a, phon_b, accents_b
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor = conn.cursor()
    poem_ids: dict[str, int] = {}
    total = 0

    def poem_id(path: str) -> int:
        cached = poem_ids.get(path)

        if cached is not None:
            return cached

        cursor.execute("INSERT OR IGNORE INTO poems (path) VALUES (?)", (path,))
        row = cursor.execute("SELECT id FROM poems WHERE path = ?", (path,)).fetchone()

        poem_ids[path] = row[0]

        return row[0]

    while True:
        batch = list(islice(pairs, batch_size))

        if not batch:
            break

        cursor.executemany(
            insert_sql,
            [
                (
                    poem_id(p.poem_path),
                    int(p.label),
                    p.word_a.word,
                    p.word_b.word,
                    p.word_a.phon,
                    p.word_a.accents,
                    p.word_b.phon,
                    p.word_b.accents,
                )
                for p in batch
            ],
        )
        conn.commit()

        total += len(batch)

    return total


def build(test_run: bool = False):
    db_path = RHYME_PAIRS_TEST_DB_PATH if test_run else RHYME_PAIRS_DB_PATH

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()

    rhyme_conn = sqlite3.connect(RHYME_DB_PATH)

    csv_file = open(METADATA_TABLE, "r", encoding="utf8")
    reader = csv.DictReader(csv_file, dialect=InputDialect)

    positives = make_positive_pairs(rhyme_conn)
    negatives = make_negative_pairs(reader)

    if test_run:
        testing_subset = 1000
        positives = islice(positives, testing_subset)
        negatives = islice(negatives, testing_subset)

    positive_count = insert_pairs(conn, positives)

    logging.info("Inserted %d positive pairs", positive_count)

    negative_count = insert_pairs(conn, negatives)
    logging.info("Inserted %d negative pairs", negative_count)

    conn.close()
    logging.info("Rhyme pairs database written to %s", db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a dataset of rhyming and non-rhyming line ending pairs."
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process a small subset and dump to a test database",
    )
    args = parser.parse_args()

    LoggingSettings.setup()

    build(test_run=args.test_run)
