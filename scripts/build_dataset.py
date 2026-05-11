# Builds training dataset from the corpus files.
import csv
import logging
from typing import Iterator

from velimir.accentuator import build_accent_dict
from velimir.domain_models import InputPoem, Poem
from velimir.io import read_accent_dicts, read_poem_xml, save_poems_as_msgpack
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import transform_poem
from velimir.settings import (
    ACCENT_DICT_PATHS,
    METADATA_TABLE,
    InputDialect,
)


def transform_data(csv_reader: csv.DictReader) -> Iterator[Poem]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)

        try:
            transformed_poem = transform_poem(xml_str)
            yield Poem(path=poem.path, **transformed_poem)
        except Exception as error:
            delayed_logger.record()
            logging.exception(error)
            continue


def main():
    LoggingSettings.setup()

    build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        transformed_data = transform_data(input_reader)

        save_poems_as_msgpack(transformed_data)


if __name__ == "__main__":
    main()
