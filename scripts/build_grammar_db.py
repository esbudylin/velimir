import argparse
import csv
import itertools
import logging
import os
import sqlite3
from dataclasses import dataclass
from itertools import count
from typing import Iterator

import msgpack
from bs4 import BeautifulSoup

from velimir import parsers, accentuator
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import delayed_logger
from velimir.nlp import GrammarFeatures, initialize, markup
from velimir.settings import (
    GRAMMAR_DB_PATH,
    GRAMMAR_TEST_DB_PATH,
    METADATA_TABLE,
    InputDialect,
    LoggingSettings,
)


@dataclass
class GrammarSample:
    poem_path: str
    line_idx: int
    features: GrammarFeatures


def clean_line_for_markup(line):
    return parsers.clean_line(accentuator.remove_accent_marks(line))


def extract_grammar_features(nlp, poem_path: str, xml: str) -> Iterator[GrammarSample]:
    soup = BeautifulSoup(xml, "xml")

    line_count = count()

    for verse in soup.find_all("p", class_="verse"):
        if stanza := list(parsers.extract_lines(verse, line_count)):
            features = markup(nlp, [clean_line_for_markup(il.text) for il in stanza])
            yield from (
                GrammarSample(poem_path, il.idx, features[i])
                for i, il in enumerate(stanza)
            )
        else:
            delayed_logger.record()
            logging.warning("Skipping empty stanza")


def transform_data(nlp, csv_reader: csv.DictReader) -> Iterator[GrammarSample]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)

        try:
            yield from extract_grammar_features(nlp, poem.path, xml_str)
        except Exception as error:
            delayed_logger.record()
            logging.exception(error)
            continue


def serialize_features(features: GrammarFeatures):
    pos_codes = [int(pos) for pos in features.part_of_speech]
    deprel_codes = [int(dep) for dep in features.dep_rels]
    return msgpack.packb(pos_codes), msgpack.packb(deprel_codes)


def write_into_sqlite(conn, samples: Iterator[GrammarSample]):
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE poems (
            path TEXT UNIQUE NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE grammar_features (
            poem_id INTEGER NOT NULL REFERENCES poems(rowid),
            line_idx INTEGER NOT NULL,
            part_of_speech BLOB NOT NULL,
            dep_rels BLOB NOT NULL
        )
        """
    )

    conn.commit()

    def batched(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    poem_id_cache: dict[str, int] = {}

    for batch in batched(samples, size=10000):
        insert_buffer = []

        for sample in batch:
            if sample.poem_path not in poem_id_cache:
                result = cursor.execute(
                    "INSERT OR IGNORE INTO poems (path) VALUES (?) RETURNING rowid",
                    (sample.poem_path,),
                )
                row = result.fetchone()
                if row is None:
                    raise ValueError(
                        "Poem %s is already in a db. Missing cache value",
                        sample.poem_path,
                    )

                poem_id_cache[sample.poem_path] = row[0]

            poem_id = poem_id_cache[sample.poem_path]
            pos, deprel = serialize_features(sample.features)
            insert_buffer.append((poem_id, sample.line_idx, pos, deprel))

        cursor.executemany(
            """
                INSERT INTO grammar_features
                    (poem_id, line_idx, part_of_speech, dep_rels)
                VALUES (?, ?, ?, ?)
                """,
            insert_buffer,
        )
        conn.commit()

        logging.info("Line batch recorded to db")


def main(test_run: bool = False):
    db_path = GRAMMAR_TEST_DB_PATH if test_run else GRAMMAR_DB_PATH

    LoggingSettings.setup()

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    nlp = initialize()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        if test_run:
            input_reader = itertools.islice(input_reader, 2)

        transformed_data = transform_data(nlp, input_reader)
        write_into_sqlite(conn, transformed_data)

    conn.close()
    logging.info("Grammar database written to %s", db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build grammar features database.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process only 2 poems and dump to a test database",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
