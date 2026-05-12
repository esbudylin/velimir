import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import Iterator

import msgpack
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from velimir.domain_models import MeterClass, Poem, SyllableFeatures
from velimir.nlp import DependencyRelation, GrammarFeatures, PartOfSpeech
from velimir.settings import GRAMMAR_DB_PATH, METER_VOCAB_PATH


def get_loader(poems, **kwargs):
    dataset = PoetryDataset(poems)
    return DataLoader(dataset, collate_fn=collate, **kwargs)


@dataclass(slots=True)
class RawSample:
    poem_path: str
    line_idx: int
    meter_class: int
    stanza_stat: list[float]
    syllables: SyllableFeatures
    grammar: GrammarFeatures


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    part_of_speech_input: torch.Tensor
    deprel_input: torch.Tensor
    poetic_accents: torch.Tensor
    meter_class: torch.Tensor


class PoetryDataset(Dataset):
    def __init__(self, raw_samples: list[RawSample]):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []

        for rs in raw_samples:
            syllables = rs.syllables

            meter_class_t = torch.tensor(rs.meter_class, dtype=torch.long)

            accent_input = torch.stack(
                [
                    torch.tensor(rs.stanza_stat, dtype=torch.float32),
                    torch.tensor(syllables.linguistic_accents, dtype=torch.float32),
                    torch.tensor(syllables.last_in_word, dtype=torch.float32),
                ],
                dim=1,
            )

            poetic = torch.tensor(syllables.poetic_accents, dtype=torch.float32)

            pos = torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)
            deprel = torch.tensor(rs.grammar.dep_rels, dtype=torch.long)

            self.samples.append(
                Sample(
                    accent_input=accent_input,
                    poetic_accents=poetic,
                    meter_class=meter_class_t,
                    part_of_speech_input=pos,
                    deprel_input=deprel,
                )
            )

        logging.info(
            "Dataset loading finished. %d samples created",
            len(self.samples),
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
    pos = [b.part_of_speech_input for b in batch]
    deprel = [b.deprel_input for b in batch]

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
    pos = pad_sequence(
        pos,
        batch_first=True,
        padding_value=-1,
    )
    deprel = pad_sequence(
        deprel,
        batch_first=True,
        padding_value=-1,
    )

    return Sample(
        accent_input=accent_input,
        poetic_accents=poetic,
        meter_class=torch.stack(meters),
        part_of_speech_input=pos,
        deprel_input=deprel,
    )


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


class GrammarDB:
    def __init__(self):
        self.conn = sqlite3.connect(GRAMMAR_DB_PATH)

        logging.info("Building grammar index")

        # Создаём временную таблицу, чтобы набросить на неё индекс. В
        # результате содержание исходного файла БД не изменяется. Это
        # делается для того, чтобы не хранить результат индексации в
        # репозитории

        self.conn.execute(
            """
            CREATE TEMP TABLE grammar_features_tmp
            AS SELECT * FROM grammar_features
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_grammar_features
            ON grammar_features_tmp(poem_id, line_idx)
            """
        )

        logging.info("Index built succesfuly")

    def fetch(self, poem_path: str, line_idx: int) -> GrammarFeatures:
        cursor = self.conn.execute(
            """SELECT gf.part_of_speech, gf.dep_rels
            FROM grammar_features_tmp gf
            JOIN poems p ON gf.poem_id = p.rowid
            WHERE p.path = ? AND gf.line_idx = ?""",
            (poem_path, line_idx),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(
                f"Can't find grammar features for line {line_idx}, poem {poem_path} in the db",
            )

        pos_codes = msgpack.unpackb(row[0])
        deprel_codes = msgpack.unpackb(row[1])

        return GrammarFeatures(
            part_of_speech=[PartOfSpeech(c) for c in pos_codes],
            dep_rels=[DependencyRelation(c) for c in deprel_codes],
        )


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
    logging.info("Loading raw samples")

    grammar_db = GrammarDB()

    rare_meters_excluded = 0

    for poem in poems:
        stanza_stats = compute_mean_ling_accents_per_stanza(
            [li.syllables.linguistic_accents for li in poem.lines],
            poem.stanza_breaks,
        )
        stanzas = break_into_stanzas(poem.lines, poem.stanza_breaks)

        for current_stanza, stanza in enumerate(stanzas):
            for line in stanza:
                syllables = line.syllables
                meter_class = MeterClassRegistry.mc_to_int(line.to_meterclass())

                if meter_class is None:
                    # Исключаем редкие типы метров из датасета
                    rare_meters_excluded += 1
                    continue

                stanza_stat = stanza_stats[current_stanza][: line.length()]

                try:
                    gf = grammar_db.fetch(poem.path, line.idx)
                    gf_expanded = gf.expand(syllables.last_in_word)
                except ValueError as e:
                    logging.error(
                        "Poem %s, Line %d. Failed when processing grammar features: %s",
                        poem.path,
                        line.idx,
                        e,
                    )
                    continue

                yield RawSample(
                    line_idx=line.idx,
                    syllables=syllables,
                    stanza_stat=stanza_stat,
                    meter_class=meter_class,
                    poem_path=poem.path,
                    grammar=gf_expanded,
                )

    grammar_db.conn.close()

    logging.info(
        "%d lines are excluded from dataset as having rare meter types",
        rare_meters_excluded,
    )
