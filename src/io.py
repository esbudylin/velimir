import os
from itertools import islice
from typing import Iterator

import msgpack

from src.models import OutputPoem
from src.settings import TEXTS_DIR, OUTPUT_FILE


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

            # TODO: rewrite for a flatten data dump?
            serialized_data = msgpack.packb(
                [poem.encode() for poem in chunk],
                use_bin_type=True,
            )

            f.write(serialized_data)
