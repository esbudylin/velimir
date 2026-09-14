import argparse
import json
import logging
import sqlite3

from bitarray import bitarray

from velimir.logger import LoggingSettings
from velimir.phonetics import RhymeInput, calc_rhyming_coef
from velimir.settings import (
    RHYME_DB_PATH,
    RHYME_TEST_DB_PATH,
)

select_query = """
SELECT poem_id,
       path,
       seq,
       rhyme_group,
       json_group_array(word) AS words,
       json_group_array(accents) AS accents
FROM (
    SELECT DISTINCT rhymes.poem_id AS poem_id,
                    rhymes.seq AS seq,
                    rhymes.rhyme_group AS rhyme_group,
                    poems.path AS path,
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


def annotate(conn: sqlite3.Connection):
    cursor = conn.cursor()

    cursor.execute(
        """
        DROP TABLE IF EXISTS rhyme_annotations
        """
    )

    cursor.execute(
        """
        CREATE TABLE rhyme_annotations (
            poem_id INTEGER NOT NULL REFERENCES poems(rowid),
            seq INTEGER NOT NULL,
            rhyme_group INTEGER NOT NULL,
            rhyming_coef REAL NOT NULL,

            UNIQUE(poem_id, seq, rhyme_group) ON CONFLICT FAIL
        )
        """
    )

    insert_buffer = []
    skipped = 0

    for poem_id, path, seq, group, words, accents in conn.execute(select_query):
        try:
            rhymes = parse_rhymes(words, accents)
            rhyming_index = calc_rhyming_coef(rhymes)
        except Exception as error:
            logging.warning("Can't annotate rhyme group %s: %s", path, error)
            skipped += 1
            continue

        insert_buffer.append((poem_id, seq, group, rhyming_index))

    cursor.executemany(
        """
        INSERT INTO rhyme_annotations (poem_id, seq, rhyme_group, rhyming_coef)
        VALUES (?, ?, ?, ?)
        """,
        insert_buffer,
    )

    cursor.execute(
        """
        DROP VIEW IF EXISTS validated_rhyme_groups
        """
    )

    cursor.execute(
        """
        CREATE VIEW validated_rhyme_groups AS
        WITH text_avg_coef AS (
            SELECT poem_id, AVG(rhyming_coef) as avg_coef FROM rhyme_annotations
            GROUP BY(poem_id)
        ),
        rounded AS (
            SELECT
                ra.poem_id,
                seq,
                rhyme_group,
                ROUND(ra.rhyming_coef, 2) AS rhyming_coef,
                ROUND(tac.avg_coef, 2) AS avg_coef
            FROM rhyme_annotations ra
            JOIN text_avg_coef tac ON ra.poem_id = tac.poem_id
        )
        SELECT *
        FROM rounded
        WHERE
            rhyming_coef >= 0.2
            OR (rhyming_coef >= 0.1 AND avg_coef >= 0.45);
        """
    )

    conn.commit()

    logging.info("Annotated %d rhyme groups, skipped %d", len(insert_buffer), skipped)


def main(test_run: bool = False):
    db_path = RHYME_TEST_DB_PATH if test_run else RHYME_DB_PATH
    db_conn = sqlite3.connect(db_path)

    LoggingSettings.setup()

    try:
        annotate(db_conn)
    finally:
        db_conn.close()

    logging.info("Rhyme annotations written to %s", db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate rhyme dataset.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Read the test rhyme database and dump to a test annotations database",
    )
    args = parser.parse_args()

    main(test_run=args.test_run)
