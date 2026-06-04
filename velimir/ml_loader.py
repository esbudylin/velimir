import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import Iterator

import msgpack
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from velimir.domain_models import Poem, SyllableFeatures
from velimir.nlp import GrammarFeatures
from velimir.settings import GRAMMAR_DB_PATH
from velimir.ml_preprocess import (
    MeterClassRegistry,
    break_into_chunks,
    compute_mean_ling_accents_per_stanza,
)


def get_loader(poems, **kwargs):
    dataset = LineDataset(poems)
    return DataLoader(dataset, collate_fn=collate, **kwargs)


@dataclass(slots=True)
class RawSample:
    poem_path: str
    line_idx: int
    chunk_idx: int
    meter_class: int
    stanza_stat: list[float]
    syllables: SyllableFeatures
    grammar: GrammarFeatures


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    part_of_speech_input: torch.Tensor
    poetic_accents: torch.Tensor
    meter_class: torch.Tensor


class LineDataset(Dataset):
    def __init__(self, raw_samples: list[RawSample]):
        logging.info("Loading poetry dataset")

        self.samples: list[Sample] = []

        for rs in raw_samples:
            meter_class_t = torch.tensor(rs.meter_class, dtype=torch.long)

            accent_input = make_accent_input(rs)

            poetic = torch.tensor(rs.syllables.poetic_accents, dtype=torch.float32)

            pos = torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)

            self.samples.append(
                Sample(
                    accent_input=accent_input,
                    poetic_accents=poetic,
                    meter_class=meter_class_t,
                    part_of_speech_input=pos,
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


def collate(batch: list[Sample]):
    accent_input = [b.accent_input for b in batch]
    poetic = [b.poetic_accents for b in batch]
    meters = [b.meter_class for b in batch]
    pos = [b.part_of_speech_input for b in batch]

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

    return Sample(
        accent_input=accent_input,
        poetic_accents=poetic,
        meter_class=torch.stack(meters),
        part_of_speech_input=pos,
    )


@dataclass(slots=True)
class RefinerSample:
    accent_input: torch.Tensor
    meter_target: torch.Tensor
    pos_input: torch.Tensor


class RefinerDataset(Dataset):
    def __init__(self, chunks: list[list[RawSample]]):
        logging.info("Loading refiner dataset")

        self.samples: list[RefinerSample] = []

        for chunk_lines in chunks:
            accent_tensors = []
            pos_tensors = []
            meter_targets = []

            for rs in chunk_lines:
                accent_tensors.append(make_accent_input(rs))
                pos_tensors.append(
                    torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)
                )
                meter_targets.append(rs.meter_class)

            accent_padded = pad_sequence(
                accent_tensors, batch_first=True, padding_value=-1
            )
            pos_padded = pad_sequence(pos_tensors, batch_first=True, padding_value=-1)
            meter_target = torch.tensor(meter_targets, dtype=torch.long)

            self.samples.append(
                RefinerSample(
                    accent_input=accent_padded,
                    pos_input=pos_padded,
                    meter_target=meter_target,
                )
            )

        logging.info(
            "Refiner dataset loading finished. %d chunks created",
            len(self.samples),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_refiner(batch: list[RefinerSample]):
    return batch[0]


def get_refiner_loader(chunks, **kwargs):
    dataset = RefinerDataset(chunks)
    return DataLoader(
        dataset,
        collate_fn=collate_refiner,
        batch_size=1,
        **kwargs,
    )


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

        logging.info("Index built successfully")

    def fetch(self, poem_path: str, line_idx: int) -> GrammarFeatures:
        cursor = self.conn.execute(
            """SELECT gf.features
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

        serialized_features = msgpack.unpackb(row[0])

        return GrammarFeatures.decode(serialized_features)


def get_meter_weights() -> torch.Tensor:
    counts = torch.tensor(MeterClassRegistry._counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.sum()
    return weights


def make_accent_input(rs: RawSample) -> torch.Tensor:
    syllables = rs.syllables
    return torch.stack(
        [
            torch.tensor(rs.stanza_stat, dtype=torch.float32),
            torch.tensor(syllables.linguistic_accents, dtype=torch.float32),
            torch.tensor(syllables.last_in_word, dtype=torch.float32),
        ],
        dim=1,
    )


def split_lines(
    raw_samples: list[RawSample],
    test_ratio: float = 0.02,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[RawSample], list[RawSample], list[RawSample]]:
    train_chunks, val_chunks, test_chunks = split_chunks(
        raw_samples, test_ratio, val_ratio, seed
    )

    train_set = [rs for chunk in train_chunks for rs in chunk]
    val_set = [rs for chunk in val_chunks for rs in chunk]
    test_set = [rs for chunk in test_chunks for rs in chunk]

    return train_set, val_set, test_set


def split_chunks(
    raw_samples: list[RawSample],
    test_ratio: float = 0.02,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[list[RawSample]], list[list[RawSample]], list[list[RawSample]]]:
    all_chunks: list[list[RawSample]] = []
    current_key = None
    current_chunk: list[RawSample] = []

    for rs in raw_samples:
        key = (rs.poem_path, rs.chunk_idx)
        if key != current_key:
            if current_chunk:
                all_chunks.append(current_chunk)
            current_chunk = [rs]
            current_key = key
        else:
            current_chunk.append(rs)

    if current_chunk:
        all_chunks.append(current_chunk)

    rng = random.Random(seed)
    rng.shuffle(all_chunks)

    n = len(all_chunks)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size

    train_chunks = all_chunks[:train_size]
    val_chunks = all_chunks[train_size : train_size + val_size]
    test_chunks = all_chunks[train_size + val_size :]

    return train_chunks, val_chunks, test_chunks


def fetch_raw_samples(poems: Iterator[Poem]) -> Iterator[RawSample]:
    logging.info("Loading raw samples")

    grammar_db = GrammarDB()

    rare_meters_excluded = 0

    for poem in poems:
        stanza_stats = compute_mean_ling_accents_per_stanza(
            [li.syllables.linguistic_accents for li in poem.lines],
            poem.stanza_breaks,
        )
        chunks = break_into_chunks(poem.lines, poem.stanza_breaks)

        for chunk_idx, chunk in enumerate(chunks):
            for line in chunk:
                syllables = line.syllables
                meter_class = MeterClassRegistry.mc_to_int(line.to_meterclass())

                if meter_class is None:
                    # Исключаем редкие типы метров из датасета
                    rare_meters_excluded += 1
                    continue

                stanza_stat = stanza_stats[chunk_idx][: line.length()]

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
                    chunk_idx=chunk_idx,
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
