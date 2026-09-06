import argparse
import csv
import itertools
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterator

from velimir import accentuator, cyrlat
from velimir.creation_date import CreationDate
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import parse_input_lines
from velimir.rhyme import RhymeVisitor, rhyme_grammar
from velimir.settings import (
    METADATA_TABLE,
    RHYME_DB_PATH,
    RHYME_TEST_DB_PATH,
    InputDialect,
)

IRREGULAR_STANZA_MARK = "нарушения строфики"


@dataclass
class RhymeSample:
    rhyme_type: str

    seq: int
    order_in_seq: int

    rhyme_group: int

    word: str
    accents: list[bool]  # позиции ударных слогов рифмованного слова


@dataclass
class PoemSamples:
    author: str
    path: str
    header: str
    creation_date: CreationDate
    samples: list[RhymeSample]


def extract_rhyme_features(
    rhyme_vistor: RhymeVisitor,
    poem: InputPoem,
    xml: str,
) -> Iterator[RhymeSample]:
    if IRREGULAR_STANZA_MARK in poem.extra:
        return

    raw_rhyme = poem.rhyme.strip()

    if not raw_rhyme:
        return

    try:
        rhyme_formula = rhyme_vistor.parse(raw_rhyme)
    except Exception:
        delayed_logger.record()
        logging.warning("Can't parse rhyme formula: %s", poem.rhyme)
        return

    if len(rhyme_formula) > 1:  # в стихотворении несколько типов рифмы
        return

    entry, *_ = rhyme_formula

    type = entry["type"]
    schema = entry.get("schema")

    if not schema:
        return  # TODO: монорим

    input_lines, _ = parse_input_lines(xml, allow_latin=True)

    rhyme_seq_len = len(schema)
    rhyme_seqs = [
        input_lines[i : i + rhyme_seq_len]
        for i in range(0, len(input_lines), rhyme_seq_len)
    ]

    for seq_idx, seq in enumerate(rhyme_seqs):
        for order_in_seq, (rhyme_group, line) in enumerate(zip(schema, seq)):
            accents = accentuator.extract_accent_mask(line.rhyme_zone)

            if not line.rhyme_zone:
                continue

            if cyrlat.detect(line.rhyme_zone) != cyrlat.DetectionResult.CYR:
                logging.info("Skipping non-cyrillic rhyme: %s", line.rhyme_zone)
                continue

            word = accentuator.remove_accent_marks(line.rhyme_zone).lower()
            cleaned_word = re.sub(r"[^а-я-\sё]", "", word).strip("-").strip()

            if not cleaned_word:
                logging.warning(
                    "Possible data loss on input cleaning. Cleaned %s. Input %s",
                    cleaned_word,
                    line.rhyme_zone,
                )
                continue

            yield RhymeSample(
                seq=seq_idx,
                order_in_seq=order_in_seq,
                rhyme_type=type,
                rhyme_group=rhyme_group,
                accents=accents,
                word=cleaned_word,
            )


def transform_data(csv_reader: csv.DictReader) -> Iterator[PoemSamples]:
    rhyme_visitor = RhymeVisitor()
    rhyme_visitor.grammar = rhyme_grammar

    for row in csv_reader:
        poem = InputPoem.from_row(row)

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        try:
            creation_date = CreationDate.extract(poem)
        except ValueError as error:
            logging.warning("Can't parse creation date for %s: %s", poem.path, error)
            continue

        xml_str = read_poem_xml(poem.path)

        try:
            samples = list(extract_rhyme_features(rhyme_visitor, poem, xml_str))
        except Exception as error:
            delayed_logger.record()
            logging.exception(error)
            continue

        yield PoemSamples(
            author=poem.author,
            path=poem.path,
            header=poem.header.strip(),
            creation_date=creation_date,
            samples=samples,
        )


def write_into_sqlite(conn, transformed_data: Iterator[PoemSamples]):
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE authors (
            name TEXT UNIQUE NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE poems (
            path TEXT UNIQUE NOT NULL,
            header TEXT,
            author_id INTEGER NOT NULL REFERENCES authors(rowid)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE rhymes (
            poem_id INTEGER NOT NULL REFERENCES poems(rowid),
            rhyme_type TEXT,

            seq INTEGER,
            order_in_seq INTEGER,
            rhyme_group INTEGER,

            word TEXT,
            accents TEXT, -- binary mask

            UNIQUE(poem_id, seq, order_in_seq) ON CONFLICT FAIL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE creation_dates (
            poem_id INTEGER NOT NULL REFERENCES poems(rowid),
            date_low INTEGER NOT NULL,
            date_high INTEGER,
            is_exact INTEGER NOT NULL,

            UNIQUE(poem_id) ON CONFLICT FAIL
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
    author_id_cache: dict[str, int] = {}

    for batch in batched(transformed_data, size=10000):
        insert_buffer = []

        for poem in batch:
            if poem.author not in author_id_cache:
                result = cursor.execute(
                    "INSERT OR IGNORE INTO authors (name) VALUES (?) RETURNING rowid",
                    (poem.author,),
                )
                row = result.fetchone()
                if row is None:
                    raise ValueError(
                        "Author %s is already in a db. Missing cache value",
                        poem.author,
                    )

                author_id_cache[poem.author] = row[0]

            author_id = author_id_cache[poem.author]

            if poem.path not in poem_id_cache:
                result = cursor.execute(
                    "INSERT OR IGNORE INTO poems (path, header, author_id) VALUES (?, ?, ?) RETURNING rowid",
                    (poem.path, poem.header, author_id),
                )
                row = result.fetchone()
                if row is None:
                    raise ValueError(
                        "Poem %s is already in a db. Missing cache value",
                        poem.path,
                    )

                poem_id_cache[poem.path] = row[0]

                cursor.execute(
                    """
                    INSERT INTO creation_dates
                        (poem_id, date_low, date_high, is_exact)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row[0],
                        poem.creation_date.lower,
                        poem.creation_date.upper,
                        int(poem.creation_date.is_exact),
                    ),
                )

            poem_id = poem_id_cache[poem.path]

            for sample in poem.samples:
                accent_str = "".join(str(int(accent)) for accent in sample.accents)

                insert_buffer.append(
                    (
                        poem_id,
                        sample.rhyme_type,
                        sample.seq,
                        sample.order_in_seq,
                        sample.rhyme_group,
                        sample.word,
                        accent_str,
                    )
                )

        cursor.executemany(
            """
                INSERT INTO rhymes
                    (poem_id, rhyme_type, seq, order_in_seq, rhyme_group, word, accents)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
            insert_buffer,
        )
        conn.commit()

        logging.info("Rhyme batch recorded to db")


def main(test_run: bool = False):
    db_path = RHYME_TEST_DB_PATH if test_run else RHYME_DB_PATH

    LoggingSettings.setup()

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        if test_run:
            input_reader = itertools.islice(input_reader, 2)

        transformed_data = transform_data(input_reader)
        write_into_sqlite(conn, transformed_data)

    conn.close()
    logging.info("Rhyme database written to %s", db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build rhyme features database.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process only 2 poems and dump to a test database",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
