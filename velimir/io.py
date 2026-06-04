import os
from itertools import islice
from typing import Iterator

from .domain_models import Poem
from .settings import OUTPUT_FILE, TEXTS_DIR, ACCENT_MODEL, METER_MODEL, REFINER_MODEL


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


def load_model(accent_type, model_path, device):
    import torch

    model = accent_type().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def load_models(device):
    from .ml import AccentModel, MeterModel, StanzaRefiner

    accent_model = load_model(AccentModel, ACCENT_MODEL, device)
    meter_model = load_model(MeterModel, METER_MODEL, device)
    refiner_model = load_model(StanzaRefiner, REFINER_MODEL, device)

    return accent_model, meter_model, refiner_model
