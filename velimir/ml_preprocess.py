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


def break_into_stanzas(lines: list, stanza_breaks: list[int]):
    for i, start in enumerate(stanza_breaks):
        end = stanza_breaks[i + 1] if i + 1 < len(stanza_breaks) else len(lines)
        yield lines[start:end]


def compute_mean_ling_accents_per_stanza(
    ling_accent_masks,
    stanza_breaks: list[int],
) -> list[list[float]]:
    stanzas = break_into_stanzas(ling_accent_masks, stanza_breaks)

    res = []

    for stanza in stanzas:
        if not stanza:
            continue

        max_len = max(len(line) for line in stanza)

        sums = [0] * max_len
        counts = [0] * max_len

        for line in stanza:
            for i, val in enumerate(line):
                sums[i] += val
                counts[i] += 1

        mean = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_len)]

        res.append(mean)

    return res
