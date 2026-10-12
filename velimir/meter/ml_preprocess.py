import json

from ..domain_models import MeterClass
from ..settings import METER_VOCAB_PATH


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
) -> list[list[str]]:
    target_size = 12
    chunk_min_size = 6
    chunk_max_size = 18

    if len(lines) == 0:
        raise ValueError("Trying to chunk an empty poem")

    if len(stanza_breaks) == 0:
        raise ValueError("Trying to chunk a poem without stanza breaks")

    if len(lines) <= target_size:
        return [lines]

    res = []
    pending_under_min = []

    for i, start in enumerate(stanza_breaks):
        end = stanza_breaks[i + 1] if i + 1 < len(stanza_breaks) else len(lines)
        stanza = lines[start:end]

        if pending_under_min:
            flatten_pending = [line for st in pending_under_min for line in st]

            if len(flatten_pending) >= chunk_min_size:
                res.append(flatten_pending)
            else:
                stanza = flatten_pending + stanza

            pending_under_min = []

        over_max = len(stanza) > chunk_max_size
        under_min = len(stanza) < chunk_min_size

        if not (over_max or under_min):
            res.append(stanza)
        elif over_max:
            chunked = [
                stanza[j : j + target_size] for j in range(0, len(stanza), target_size)
            ]
            if len(stanza) % target_size > chunk_min_size:
                res.extend(chunked)
            else:
                res.extend(chunked[:-1])
                pending_under_min.append(chunked[-1])
        elif under_min:
            pending_under_min.append(stanza)

    if pending_under_min:
        flatten_pending = [line for st in pending_under_min for line in st]

        if len(flatten_pending) >= chunk_min_size:
            res.append(flatten_pending)
        else:
            concat = res[-1] + flatten_pending

            for target in range(chunk_max_size, chunk_min_size - 1, -1):
                if len(concat) % target > chunk_min_size:
                    res.pop()
                    res.extend(
                        concat[j : j + target] for j in range(0, len(concat), target)
                    )
                    break

    if [line for chunk in res for line in chunk] != lines:
        raise ValueError("Input/output mismatch")

    if any(filter(lambda n: n < chunk_min_size, map(len, res))):
        raise ValueError("Trying to return a chunk under min size")

    if any(filter(lambda n: n > chunk_max_size, map(len, res))):
        raise ValueError("Trying to return a chunk that is too big")

    return res
