import argparse
import csv
import itertools
import os
import sqlite3
from typing import Iterator

from bs4 import BeautifulSoup
from pymorphy2 import MorphAnalyzer

from velimir import accentuator, io, nlp
from velimir.domain_models import InputLine, InputPoem
from velimir.parsers import (
    clean_line,
    extract_lines,
    parse_line_formula,
)
from velimir.settings import METADATA_TABLE, InputDialect
from velimir.logger import LoggingSettings

morph_analyzer = MorphAnalyzer()

DB_PATH = "data/pos_accent.db"
TEST_DB_PATH = "data/pos_accent_test.db"


def extract_lines_from_csv(csv_reader: csv.DictReader) -> Iterator[InputLine]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        xml_str = io.read_poem_xml(poem.path)
        soup = BeautifulSoup(xml_str, "xml")

        yield from extract_lines(soup)


def extract_pos_accent_pairs(poetic_accents, last_in_word, parts_of_speech):
    current_word_acc = False
    pos_accent_pairs = []

    for has_accent, is_end in zip(poetic_accents, last_in_word):
        current_word_acc = current_word_acc or has_accent
        if is_end:
            pos_accent_pairs.append((next(parts_of_speech), current_word_acc))
            current_word_acc = False

    return pos_accent_pairs


def parse_line(line):
    line_formula = parse_line_formula(line.meter)

    if not line_formula:
        return []

    meter_repr = "~".join(m.meter.to_str() for m in line_formula.meters)

    cleaned_line = clean_line(accentuator.remove_accent_marks(line.text))

    poetic_accents = accentuator.extract_accent_mask(line.text)
    last_in_word = accentuator.extract_word_ending_mask(cleaned_line)

    words = nlp.extract_words_for_morph(cleaned_line)
    parts_of_speech = map(lambda w: nlp.extract_part_of_speech(w).to_str(), words)

    pos_accent_pairs = extract_pos_accent_pairs(
        poetic_accents,
        last_in_word,
        parts_of_speech,
    )

    return [
        dict(
            pos=pos,
            has_accent=has_accent,
            meter=meter_repr,
            word=word.casefold(),
            syllable_count=accentuator.vowel_count(word),
        )
        for word, (pos, has_accent) in zip(words, pos_accent_pairs)
    ]


def write_into_sqlite(cursor, conn, parsed_lines):
    def batched(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    for batch in batched(parsed_lines, size=10000):
        cursor.executemany(
            """
            INSERT INTO pos_accent (pos, has_accent, meter, word, syllable_count)
            VALUES (:pos, :has_accent, :meter, :word, :syllable_count)
            """,
            batch,
        )
        conn.commit()


def main(test_run: bool = False):
    db_path = TEST_DB_PATH if test_run else DB_PATH

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pos_accent (
        pos TEXT,
        has_accent INTEGER,
        meter TEXT,
        word TEXT,
        syllable_count INTEGER
    )
    """)

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        if test_run:
            input_reader = itertools.islice(input_reader, 100)

        lines = extract_lines_from_csv(input_reader)
        parsed_lines = itertools.chain.from_iterable(map(parse_line, lines))
        write_into_sqlite(cursor, conn, parsed_lines)


if __name__ == "__main__":
    LoggingSettings.setup()

    parser = argparse.ArgumentParser(description="Build POS-accent database.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process only 100 poems and dump to a test database",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
