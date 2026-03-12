# Собирает данные о точности акцентуатора,
# основываясь на текстах акцентников, размеченных в корпусе.
# NB: В акцентниках размечены реальные ударения, а не икты.

import csv
import logging
import time
from typing import Iterator

from src.logger import delayed_logger
from src.models import InputPoem, SyllableMasks
from src.settings import METADATA_TABLE, InputDialect, LoggingSettings
from src.parsers import extract_lines, extract_syllable_masks
from src.io import read_poem_xml


def extract_ak_lines(csv_reader: csv.DictReader) -> Iterator[str]:
    for row in csv_reader:
        poem = InputPoem(**row)

        if "Ак" not in poem.meter:
            continue

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)

        for line in extract_lines(xml_str):
            if "Ак" not in line.meter:
                continue
            yield line.text


def calc_accent_diff(lines: Iterator[str]) -> tuple[int, int, float]:
    total_lines = 0
    total_words = 0
    total_diff = 0

    for line in lines:
        try:
            sm = extract_syllable_masks([], line)
        except Exception as e:
            delayed_logger.record()
            logging.error("error while processing line %s", line)
            logging.error(e)
            continue

        diff_indexes = accent_diff_word_indexes(sm)
        word_count = sum(sm.last_in_word_mask)
        if word_count:
            total_diff += len(diff_indexes) / word_count
            total_lines += 1
            total_words += word_count

    avg_diff = total_diff / total_lines if total_lines else 0

    return total_lines, total_words, avg_diff


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
    logging.basicConfig(**LoggingSettings().model_dump())
    start_time = time.time()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)
        ak_lines = extract_ak_lines(input_reader)
        total_lines, total_words, diff = calc_accent_diff(ak_lines)

    total_time = time.time() - start_time
    print(f"Total lines {total_lines}")
    print(f"Total words {total_words}")
    print(f"Diff {diff:.4f}")

    print(f"Total time {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
