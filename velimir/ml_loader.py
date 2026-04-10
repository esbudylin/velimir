import json
import logging
import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from velimir.domain_models import MeterClass, Poem, SyllableMasks
from velimir.settings import METER_VOCAB_PATH


def get_loader(poems, **kwargs):
    dataset = PoetryDataset(poems)
    return DataLoader(dataset, collate_fn=collate, **kwargs)


@dataclass(slots=True)
class RawSample:
    stanza_stat: list[float]
    syllable_masks: SyllableMasks
    meter_class: MeterClass


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    poetic_accents: torch.Tensor
    meter_class: torch.Tensor


class PoetryDataset(Dataset):
    def __init__(self, raw_samples: list[RawSample]):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []
        rare_meters_excluded = 0

        for rs in raw_samples:
            meter_class = MeterClassRegistry.mc_to_int(rs.meter_class)

            if meter_class is None:
                # Исключаем редкие типы метров из датасета
                rare_meters_excluded += 1
                continue

            masks = rs.syllable_masks

            meter_class_t = torch.tensor(meter_class, dtype=torch.long)

            accent_input = torch.stack(
                [
                    torch.tensor(rs.stanza_stat, dtype=torch.float32),
                    torch.tensor(masks.linguistic_accent_mask, dtype=torch.float32),
                    torch.tensor(masks.last_in_word_mask, dtype=torch.float32),
                ],
                dim=1,
            )
            poetic = torch.tensor(masks.poetic_accent_mask, dtype=torch.float32)

            self.samples.append(
                Sample(
                    accent_input=accent_input,
                    poetic_accents=poetic,
                    meter_class=meter_class_t,
                )
            )

        logging.info(
            "Dataset loading finished. %d samples created",
            len(self.samples),
        )
        logging.info(
            "%d lines are excluded from dataset as having rare meter types",
            rare_meters_excluded,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MeterClassRegistry:
    _vocab: list[MeterClass] = None
    _mc_to_idx: dict[MeterClass, int] = None
    _counts: list[int] = None
    _weights: torch.Tensor = None

    @classmethod
    def initialize(cls):
        if cls._vocab is not None:
            return  # already initialized

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

    @classmethod
    def get_weights(cls) -> torch.Tensor:
        if cls._weights is not None:
            return cls._weights

        counts = torch.tensor(cls._counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1)

        # sqrt inv
        weights = 1.0 / torch.sqrt(counts)

        weights = weights / weights.sum()

        cls._weights = weights
        return weights


def collate(batch: list[Sample]):
    accent_input = [b.accent_input for b in batch]
    poetic = [b.poetic_accents for b in batch]
    meters = [b.meter_class for b in batch]

    accent_input = pad_sequence(
        accent_input,
        batch_first=True,
        padding_value=-1,
    )
    poetic = pad_sequence(
        poetic,
        batch_first=True,
        padding_value=-1,
    )

    return Sample(
        accent_input=accent_input,
        poetic_accents=poetic,
        meter_class=torch.stack(meters),
    )


def break_into_stanzas(lines: list, stanza_breaks: list[int]):
    for i, start in enumerate(stanza_breaks):
        end = stanza_breaks[i + 1] if i + 1 < len(stanza_breaks) else len(lines)
        yield lines[start:end]


def compute_mean_ling_accents_per_stanza(
    ling_accent_masks,
    stanza_breaks: list[int],
) -> list[float]:
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


def split_samples(
    raw_samples: Iterator[RawSample],
    test_ratio: float = 0.02,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list, list, list]:
    samples_l = list(raw_samples)

    rng = random.Random(seed)
    rng.shuffle(samples_l)

    n = len(samples_l)

    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    train_size = n - test_size - val_size

    train_set = samples_l[:train_size]
    val_set = samples_l[train_size : train_size + val_size]
    test_set = samples_l[train_size + val_size :]

    return train_set, val_set, test_set


def fetch_raw_samples(poems: Iterator[Poem]) -> Iterator[RawSample]:
    for poem in poems:
        stanza_stats = compute_mean_ling_accents_per_stanza(
            [li.syllable_masks.linguistic_accent_mask for li in poem.lines],
            poem.stanza_breaks,
        )
        stanzas = break_into_stanzas(poem.lines, poem.stanza_breaks)

        for current_stanza, stanza in enumerate(stanzas):
            for line in stanza:
                masks = line.syllable_masks

                if not masks.poetic_accent_mask:
                    logging.error("Empty line in text %s. Skipping...", poem.path)
                    continue

                stanza_stat = stanza_stats[current_stanza][: line.length()]

                yield RawSample(
                    syllable_masks=masks,
                    stanza_stat=stanza_stat,
                    meter_class=line.to_meterclass(),
                )
