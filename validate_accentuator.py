# Собирает данные о точности акцентуатора,
# основываясь на текстах акцентников и тактовиков, размеченных в корпусе.
# NB: В акцентниках размечены реальные ударения, а не икты.

import csv
import logging
import re
import time
from collections import Counter
from typing import Iterator

from bs4 import BeautifulSoup

import velimir.accentuator as accentuator
from velimir.domain_models import InputPoem, SyllableMasks
from velimir.io import read_accent_dicts, read_poem_xml
from velimir.logger import delayed_logger
from velimir.parsers import extract_lines, extract_syllable_masks
from velimir.settings import (
    ACCENT_DICT_PATHS,
    METADATA_TABLE,
    InputDialect,
    LoggingSettings,
)


def extract_ak_lines(csv_reader: csv.DictReader) -> Iterator[str]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        if not ("Ак" in poem.meter or "Тк" in poem.meter):
            continue

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)
        soup = BeautifulSoup(xml_str, "xml")

        for line in extract_lines(soup):
            if "Ак" in line.meter or "Тк" in line.meter:
                yield line.text


def calc_accent_diff(lines: Iterator[str], accent_line_fn) -> Counter:
    total_lines = 0
    total_words = 0
    total_diff = 0.0

    diffed_words: Counter[str] = Counter()

    for line in lines:
        try:
            sm = extract_syllable_masks(line)
        except Exception as e:
            delayed_logger.record()
            logging.error("error while processing line %s", line)
            logging.exception(e)
            continue

        diff_indexes = accent_diff_word_indexes(sm)

        line_stripped = re.sub(r"[^А-яЁё\s-]+", "", line)
        line_words = list(
            filter(
                lambda w: sum(map(accentuator.is_vowel, w)),
                line_stripped.split(),
            )
        )
        for di in diff_indexes:
            diffed_words[line_words[di].lower()] += 1

        word_count = sum(sm.last_in_word_mask)
        if word_count:
            total_diff += len(diff_indexes) / word_count
            total_lines += 1
            total_words += word_count

    avg_diff = total_diff / total_lines if total_lines else 0

    print(f"Total lines {total_lines}")
    print(f"Total words {total_words}")
    print(f"Diff {avg_diff:.4f}")

    return diffed_words


def accent_diff_word_indexes(masks: SyllableMasks) -> list[int]:
    result = []

    word_ling = []
    word_poet = []
    word_index = 0

    for ling, poet, last in zip(
        masks.linguistic_accent_mask,
        masks.poetic_accent_mask,
        masks.last_in_word_mask,
    ):
        word_ling.append(ling)
        word_poet.append(poet)

        if last:  # конец слова
            # ударения размечены в обоих вариантах
            if sum(word_ling) and sum(word_poet):
                diff = sum(a != b for a, b in zip(word_ling, word_poet))

                if diff:
                    result.append(word_index)

            word_ling = []
            word_poet = []
            word_index += 1

    return result


def main():
    LoggingSettings.setup()

    accentuator.build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    start_time = time.time()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)
        ak_lines = list(extract_ak_lines(input_reader))
        accent_diff = calc_accent_diff(ak_lines, accentuator.accent_line)

    for word, count in accent_diff.most_common(40):
        logging.info("%s | count=%d", word, count)

    total_time = time.time() - start_time
    print(f"Total time {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
