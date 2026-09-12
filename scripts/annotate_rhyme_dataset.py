import argparse
import json
import logging
import os
import sqlite3

from bitarray import bitarray

from velimir.logger import LoggingSettings
from velimir.phonetics import RhymeInput, calc_rhyming_coef
from velimir.settings import (
    RHYME_ANNOTATIONS_DB_PATH,
    RHYME_ANNOTATIONS_TEST_DB_PATH,
    RHYME_DB_PATH,
    RHYME_TEST_DB_PATH,
)

select_query = """
SELECT path,
       json_group_array(word) AS words,
       json_group_array(accents) AS accents
FROM (
    SELECT DISTINCT poems.path AS path,
                    rhymes.poem_id AS poem_id,
                    rhymes.seq AS seq,
                    rhymes.rhyme_group AS rhyme_group,
                    word,
                    accents
    FROM rhymes
    JOIN poems ON rhymes.poem_id == poems.ROWID
    JOIN authors ON poems.author_id == authors.ROWID
    WHERE rhyme_group <> -1
)
GROUP BY poem_id, seq, rhyme_group
HAVING COUNT(*) > 1
"""


def parse_rhymes(words: str, accents: str) -> list[RhymeInput]:
    return [
        RhymeInput(word=word, accents=bitarray(accents_mask))
        for word, accents_mask in zip(json.loads(words), json.loads(accents))
    ]


def annotate(source_conn: sqlite3.Connection, output_conn: sqlite3.Connection):
    output_cursor = output_conn.cursor()

    output_cursor.execute(
        """
        CREATE TABLE rhyme_annotations (
            path TEXT NOT NULL,
            words TEXT NOT NULL,
            rhyming_coef REAL NOT NULL
        )
        """
    )

    insert_buffer = []
    skipped = 0

    for path, words, accents in source_conn.execute(select_query):
        try:
            rhymes = parse_rhymes(words, accents)
            rhyming_index = calc_rhyming_coef(rhymes)
        except Exception as error:
            logging.warning("Can't annotate rhyme group %s: %s", path, error)
            skipped += 1
            continue

        insert_buffer.append((path, words, rhyming_index))

    output_cursor.executemany(
        """
        INSERT INTO rhyme_annotations (path, words, rhyming_coef)
        VALUES (?, ?, ?)
        """,
        insert_buffer,
    )
    output_conn.commit()

    logging.info("Annotated %d rhyme groups, skipped %d", len(insert_buffer), skipped)


def main(test_run: bool = False):
    source_db_path = RHYME_TEST_DB_PATH if test_run else RHYME_DB_PATH
    output_db_path = (
        RHYME_ANNOTATIONS_TEST_DB_PATH if test_run else RHYME_ANNOTATIONS_DB_PATH
    )

    LoggingSettings.setup()

    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    source_conn = sqlite3.connect(source_db_path)
    output_conn = sqlite3.connect(output_db_path)

    try:
        annotate(source_conn, output_conn)
    finally:
        source_conn.close()
        output_conn.close()

    logging.info("Rhyme annotations written to %s", output_db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate rhyme dataset.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Read the test rhyme database and dump to a test annotations database",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
