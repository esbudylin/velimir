import os
from itertools import islice
from typing import Iterator

import msgpack

from .domain_models import OutputPoem
from .settings import OUTPUT_FILE, TEXTS_DIR


def read_poem_xml(text_path):
    xml_path = os.path.join(TEXTS_DIR, text_path) + ".xml"

    with open(xml_path, "r", encoding="utf8") as f:
        return f.read()


def save_poems_as_msgpack(data: Iterator[OutputPoem]):
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


def load_poems_from_msgpack() -> Iterator:
    with open(OUTPUT_FILE, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for batch in unpacker:
            for poem_data in batch:
                # We don't serialize data to OutputPoem at this stage,
                # to optimize for memory usage
                yield poem_data


def read_accent_dicts(filenames):
    for filename in filenames:
        with open(filename, encoding="cp1251") as file_read:
            for line in file_read:
                yield line
