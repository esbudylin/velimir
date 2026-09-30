"""
Формирует датасет для валидации акцентуаторов,
на основе акцентников, размеченных в корпусе.
NB: Икты, размеченные в акцентниках, совпадают с языковыми ударениям
"""

import csv
import logging
from typing import Iterator

from bs4 import BeautifulSoup

from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import (
    clean_line,
    extract_accent_mask,
    extract_lines,
    remove_accent_marks,
)
from velimir.settings import ACCENT_DATASET_CSV, METADATA_TABLE, InputDialect

DATASET_FIELDS = ["path", "text", "mask"]


def mask_to_bits(mask: list[bool]) -> str:
    return "".join("1" if bit else "0" for bit in mask)


def extract_ak_lines(csv_reader: csv.DictReader) -> Iterator[tuple[str, str]]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        if "Ак" not in poem.meter:
            continue

        delayed_logger.create(
            logging.INFO,
            "Transforming poem: %s, meter: %s",
            poem.path,
            poem.formula,
        )

        xml_str = read_poem_xml(poem.path)
        soup = BeautifulSoup(xml_str, "xml")

        for line in extract_lines(soup):
            if "Ак" in line.meter:
                yield poem.path, line.text


def build_dataset(
    samples: Iterator[tuple[str, str]],
    writer: csv.DictWriter,
):
    for path, line in samples:
        cleaned = clean_line(remove_accent_marks(line))

        writer.writerow(
            {
                "path": path,
                "text": cleaned,
                "mask": mask_to_bits(extract_accent_mask(line)),
            }
        )


def main():
    LoggingSettings.setup()

    with open(ACCENT_DATASET_CSV, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DATASET_FIELDS)
        writer.writeheader()

        with open(METADATA_TABLE, "r", encoding="utf8") as metadata_file:
            reader = csv.DictReader(metadata_file, dialect=InputDialect)

            build_dataset(extract_ak_lines(reader), writer)

    logging.info("Dataset written to %s", ACCENT_DATASET_CSV)


if __name__ == "__main__":
    main()
