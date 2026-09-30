# Собирает данные о точности акцентуатора,
# основываясь на текстах акцентников, размеченных в корпусе.
# NB: В акцентниках размечены реальные ударения, а не икты.

import argparse
import csv
import logging
import re
import time
from collections import Counter
from typing import Iterator

from bs4 import BeautifulSoup

from velimir.domain_models import InputPoem, SyllableFeatures
from velimir.io import read_poem_xml
from velimir.logger import delayed_logger, LoggingSettings
from velimir.parsers import (
    clean_line,
    extract_lines,
    extract_syllable_features,
    is_vowel,
    remove_accent_marks,
)
from velimir.settings import (
    ACCENT_ERRORS_CSV,
    METADATA_TABLE,
    STRESS_MARK_ORD,
    InputDialect,
)

ACCENT_ERROR_FIELDS = ["path", "corpus_line", "our_line"]


def extract_ak_lines(csv_reader: csv.DictReader) -> Iterator[tuple[str, str]]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        if "Ак" not in poem.meter:
            continue

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)
        soup = BeautifulSoup(xml_str, "xml")

        for line in extract_lines(soup):
            if "Ак" in line.meter:
                yield poem.path, line.text


def put_accents(line: str, mask) -> str:
    result = []
    vowel_pos = 0

    for char in line:
        result.append(char)
        if is_vowel(char):
            if mask[vowel_pos]:
                result.append(chr(STRESS_MARK_ORD))
            vowel_pos += 1

    return "".join(result)


def calc_accent_diff(
    samples: Iterator[tuple[str, str]],
) -> Counter:
    total_lines = 0
    total_words = 0
    total_diff = 0.0

    diffed_words: Counter[str] = Counter()

    with open(ACCENT_ERRORS_CSV, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ACCENT_ERROR_FIELDS)
        writer.writeheader()

        for path, line in samples:
            try:
                sm = extract_syllable_features(line)
            except Exception as e:
                delayed_logger.record()
                logging.error("error while processing line %s", line)
                logging.exception(e)
                continue

            diff_indexes = accent_diff_word_indexes(sm)

            if diff_indexes:
                cleaned = clean_line(remove_accent_marks(line))
                writer.writerow(
                    {
                        "path": path,
                        "corpus_line": line,
                        "our_line": put_accents(cleaned, sm.linguistic_accents),
                    }
                )

            line_stripped = re.sub(r"[^А-яЁё\s-]+", "", line)
            line_words = list(
                filter(
                    lambda w: sum(map(is_vowel, w)),
                    line_stripped.split(),
                )
            )
            for di in diff_indexes:
                diffed_words[line_words[di].lower()] += 1

            word_count = sum(sm.last_in_word)
            if word_count:
                total_diff += len(diff_indexes) / word_count
                total_lines += 1
                total_words += word_count

    avg_diff = total_diff / total_lines if total_lines else 0

    print(f"Total lines {total_lines}")
    print(f"Total words {total_words}")
    print(f"Diff {avg_diff:.4f}")

    return diffed_words


def accent_diff_word_indexes(masks: SyllableFeatures) -> list[int]:
    result = []

    word_ling = []
    word_poet = []
    word_index = 0

    for ling, poet, last in zip(
        masks.linguistic_accents,
        masks.poetic_accents,
        masks.last_in_word,
    ):
        word_ling.append(ling)
        word_poet.append(poet)

        if last:  # конец слова
            # поэтические ударения размечены, и слово не односложное
            if len(word_poet) > 1 and sum(word_poet):
                diff = sum(a != b for a, b in zip(word_ling, word_poet))

                if diff:
                    result.append(word_index)

            word_ling = []
            word_poet = []
            word_index += 1

    return result


def main():
    LoggingSettings.setup()

    start_time = time.time()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)
        ak_lines = list(extract_ak_lines(input_reader))
        accent_diff = calc_accent_diff(ak_lines)

    for word, count in accent_diff.most_common(40):
        logging.info("%s | count=%d", word, count)

    total_time = time.time() - start_time
    print(f"Total time {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the accentuator on accented poems."
    )
    args = parser.parse_args()

    main()
