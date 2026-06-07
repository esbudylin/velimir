import os
from itertools import islice
from typing import Iterator

from .domain_models import Poem
from .settings import OUTPUT_FILE, TEXTS_DIR, METER_MODEL, ACCENT_MODEL


def read_poem_xml(text_path):
    xml_path = os.path.join(TEXTS_DIR, text_path) + ".xml"

    with open(xml_path, "r", encoding="utf8") as f:
        return f.read()


def save_poems_as_msgpack(data: Iterator[Poem]):
    import msgpack

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


def load_poems_from_msgpack() -> Iterator[Poem]:
    import msgpack

    with open(OUTPUT_FILE, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for batch in unpacker:
            for poem_data in batch:
                yield Poem.decode(poem_data)


def read_accent_dicts(filenames):
    for filename in filenames:
        with open(filename, encoding="utf8") as file_read:
            for line in file_read:
                yield line


def load_models(device):
    import torch

    from .ml import MeterModel, AccentModel

    meter = MeterModel().to(device)
    meter.load_state_dict(torch.load(METER_MODEL, map_location=device))
    meter.eval()

    accent = AccentModel().to(device)
    accent.load_state_dict(torch.load(ACCENT_MODEL, map_location=device))
    accent.eval()

    return meter, accent
