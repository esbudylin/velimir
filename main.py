import csv
import logging
import os
from typing import Iterator
from itertools import islice

import msgpack

from src.logger import delayed_logger
from src.models import InputPoem, OutputPoem
from src.settings import (
    METADATA_TABLE,
    OUTPUT_FILE,
    TEXTS_DIR,
    InputDialect,
    LoggingSettings,
)
from src.parsers import transform_poem


def read_poem_xml(text_path):
    xml_path = os.path.join(TEXTS_DIR, text_path) + ".xml"

    with open(xml_path, "r", encoding="utf8") as f:
        return f.read()


def transform_data(csv_reader: csv.DictReader) -> Iterator[OutputPoem]:
    for row in csv_reader:
        poem = InputPoem(**row)

        delayed_logger.create(
            logging.INFO, "Transforming poem: %s, meter: %s", poem.path, poem.formula
        )

        xml_str = read_poem_xml(poem.path)

        try:
            transformed_poem = transform_poem(poem, xml_str)
            yield transformed_poem
        except Exception as error:
            delayed_logger.record()
            logging.exception(error)
            continue


def save_data(data: Iterator[OutputPoem]):
    batch_size = 500

    with open(OUTPUT_FILE, "wb") as f:
        while True:
            chunk = list(islice(data, batch_size))
            if not chunk:
                break

            serialized_data = msgpack.packb(
                [poem.encode() for poem in chunk],
                use_bin_type=True,
            )

            f.write(serialized_data)


def main():
    logging.basicConfig(**LoggingSettings().model_dump())

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        transformed_data = transform_data(input_reader)

        save_data(transformed_data)


if __name__ == "__main__":
    main()
