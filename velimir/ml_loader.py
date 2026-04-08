import json
import logging
import random
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from velimir.domain_models import MeterClass, Poem
from velimir.settings import METER_VOCAB_PATH


def get_loader(poems, **kwargs):
    dataset = PoetryDataset(poems)
    return DataLoader(dataset, collate_fn=collate, **kwargs)


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    poetic_accents: torch.Tensor
    meter_class: torch.Tensor


class PoetryDataset(Dataset):
    def __init__(self, poems: list):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []
        rare_meters_excluded = 0

        for poem_data in poems:
            poem = Poem.decode(poem_data)

            stanza_stats = compute_mean_ling_accents_per_stanza(
                poem.stanza_breaks,
                [li.syllable_masks.linguistic_accent_mask for li in poem.lines],
            )
            current_stanza = 0

            for i, line in enumerate(poem.lines):
                if (
                    len(poem.stanza_breaks) != current_stanza + 1
                    and i == poem.stanza_breaks[current_stanza + 1]
                ):
                    current_stanza += 1

                masks = line.syllable_masks

                if not masks.poetic_accent_mask:
                    logging.error("Empty line in text %s. Skipping...", poem.path)
                    continue

                meter_class = MeterClassRegistry.mc_to_int(line.to_meterclass())

                if meter_class is None:
                    # Исключаем редкие типы метров из датасета
                    rare_meters_excluded += 1
                    continue

                meter_class_t = torch.tensor(meter_class, dtype=torch.long)

                stanza_stat = stanza_stats[current_stanza][: line.length()]

                accent_input = torch.stack(
                    [
                        torch.tensor(stanza_stat, dtype=torch.float32),
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
    def get_weights(cls, mode: str = "sqrt_inv") -> torch.Tensor:
        if cls._weights is not None:
            return cls._weights

        counts = torch.tensor(cls._counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1)

        if mode == "inv":
            weights = 1.0 / counts
        elif mode == "sqrt_inv":
            weights = 1.0 / torch.sqrt(counts)
        elif mode == "log_inv":
            weights = 1.0 / torch.log1p(counts)
        else:
            raise ValueError(f"Unknown weight mode: {mode}")

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


def compute_mean_ling_accents_per_stanza(
    stanza_breaks: list[int],
    ling_accent_masks,
):
    stanzas = []

    for i, start in enumerate(stanza_breaks):
        end = (
            stanza_breaks[i + 1]
            if i + 1 < len(stanza_breaks)
            else len(ling_accent_masks)
        )
        stanzas.append(ling_accent_masks[start:end])

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


def split_poems(
    poems,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list, list]:
    poems_l = list(poems)

    rng = random.Random(seed)
    rng.shuffle(poems_l)

    split = int(len(poems_l) * (1 - test_ratio))

    train_poems = poems_l[:split]
    test_poems = poems_l[split:]

    return train_poems, test_poems
