import logging
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .domain_models import Poem, SyllableDistances


def get_loader(poems, **kwargs):
    dataset = PoetryDataset(poems)
    return DataLoader(dataset, collate_fn=collate, **kwargs)


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    poetic_accents: torch.Tensor
    meta: torch.Tensor  # meter, caesuras, unstable
    syllable_distances: torch.Tensor  # anacrusa, min dist, max dist, mean dist


class PoetryDataset(Dataset):
    def __init__(self, poems: list):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []
        for poem_data in poems:
            poem = Poem.decode(poem_data)

            stanza_stats = compute_mean_ling_accents_per_stanza(poem)
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

                meter = fixed_size_tensor(
                    [int(m.meter) for m in line.meters],
                    size=3,
                )

                caesura = fixed_size_tensor(
                    line.caesura,
                    size=2,
                )

                unstable_flag = any(m.unstable for m in line.meters)
                if unstable_flag:
                    unstable = torch.tensor([1], dtype=torch.float32)
                else:
                    unstable = torch.tensor([0], dtype=torch.float32)

                meta = torch.cat([meter, caesura, unstable])

                syllable_distances = torch.tensor(
                    SyllableDistances(masks.poetic_accent_mask).to_array(),
                    dtype=torch.float32,
                )

                self.samples.append(
                    Sample(
                        accent_input=accent_input,
                        poetic_accents=poetic,
                        meta=meta,
                        syllable_distances=syllable_distances,
                    )
                )

        logging.info("Dataset loading finished")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def fixed_size_tensor(values, size, dtype=torch.float32, padding_value=-1):
    values = list(values)
    if len(values) < size:
        values += [padding_value] * (size - len(values))
    else:
        values = values[:size]
    return torch.tensor(values, dtype=dtype)


def collate(batch: list[Sample]):
    accent_input = [b.accent_input for b in batch]
    poetic = [b.poetic_accents for b in batch]
    meta = [b.meta for b in batch]
    syllable_distances = [b.syllable_distances for b in batch]

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
    meta = torch.stack(meta)
    syllable_distances = torch.stack(syllable_distances)

    return Sample(
        accent_input=accent_input,
        poetic_accents=poetic,
        meta=meta,
        syllable_distances=syllable_distances,
    )


def compute_mean_ling_accents_per_stanza(poem: Poem):
    stanzas = []

    stanza_breaks = poem.stanza_breaks
    lines = poem.lines

    for i, start in enumerate(stanza_breaks):
        end = stanza_breaks[i + 1] if i + 1 < len(stanza_breaks) else len(lines)
        stanzas.append(lines[start:end])

    res = []

    for stanza in stanzas:
        if not stanza:
            continue

        max_len = max(line.length() for line in stanza)

        sums = [0] * max_len
        counts = [0] * max_len

        for line in stanza:
            for i, val in enumerate(line.syllable_masks.linguistic_accent_mask):
                sums[i] += val
                counts[i] += 1

        mean = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_len)]

        res.append(mean)

    return res
