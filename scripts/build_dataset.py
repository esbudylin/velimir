# Builds training dataset from the corpus files.
import csv
import logging
from typing import Iterator
from dataclasses import replace

from bitarray import bitarray

from velimir.domain_models import InputPoem, MeterType, Poem, Line
from velimir.io import read_poem_xml, save_poems_as_msgpack
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import transform_poem
from velimir.settings import (
    METADATA_TABLE,
    InputDialect,
)

dactyl_hexameter_mask = bitarray("10010010010010010")


def amend_line_meters(lines: list[Line]):
    for line in lines:
        multiple_meters = len(line.meters) > 1
        matches_rhythm = line.syllables.poetic_accents == dactyl_hexameter_mask
        is_hex = line.meters[0].meter == MeterType.HEXAMETER

        if not multiple_meters and matches_rhythm and is_hex:
            delayed_logger.record()
            logging.info("Transforming hexameter into dactyl")
            line.meters = [replace(line.meters[0], meter=MeterType.DACTYL)]


def transform_data(csv_reader: csv.DictReader) -> Iterator[Poem]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)

        try:
            transformed_poem = transform_poem(xml_str)
            amend_line_meters(transformed_poem["lines"])

            yield Poem(path=poem.path, **transformed_poem)
        except Exception as error:
            delayed_logger.record()
            logging.exception(error)
            continue


def main():
    LoggingSettings.setup()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        transformed_data = transform_data(input_reader)

        save_poems_as_msgpack(transformed_data)


if __name__ == "__main__":
    main()
