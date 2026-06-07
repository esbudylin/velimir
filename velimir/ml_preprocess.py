import json

from velimir.domain_models import MeterClass
from velimir.settings import METER_VOCAB_PATH


class MeterClassRegistry:
    _vocab: list[MeterClass] = None
    _mc_to_idx: dict[MeterClass, int] = None

    @classmethod
    def initialize(cls):
        if cls._vocab is not None:
            return

        vocab = []
        counts = []

        with open(METER_VOCAB_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                mc = MeterClass.from_dict(data)
                mc_count = data["count"]
                vocab.append(mc)
                counts.append(mc_count)

        cls._vocab = vocab
        cls._counts = counts
        cls._mc_to_idx = {mc: idx for idx, mc in enumerate(vocab)}

    @classmethod
    def mc_to_int(cls, mc: MeterClass) -> int | None:
        return cls._mc_to_idx.get(mc)

    @classmethod
    def int_to_mc(cls, i: int) -> MeterClass:
        if i < 0:
            raise ValueError("Meter class index cannot be negative")

        return cls._vocab[i]

    @classmethod
    def num(cls) -> int:
        return len(cls._vocab)


def break_into_chunks(
    lines: list[str],
    stanza_breaks: list[int],
):
    target_size = 12

    chunk = []

    for i, start in enumerate(stanza_breaks):
        end = stanza_breaks[i + 1] if i + 1 < len(stanza_breaks) else len(lines)
        stanza = lines[start:end]

        # Split oversized stanzas.
        if len(stanza) > target_size:
            if chunk:
                yield chunk
                chunk = []

            for j in range(0, len(stanza), target_size):
                yield stanza[j : j + target_size]
            continue

        if not chunk:
            chunk = list(stanza)
            continue

        current_size = len(chunk)
        merged_size = current_size + len(stanza)

        # If merging gets us closer to target, do it.
        if abs(merged_size - target_size) <= abs(current_size - target_size):
            chunk.extend(stanza)
        else:
            yield chunk
            chunk = list(stanza)

    if chunk:
        yield chunk
