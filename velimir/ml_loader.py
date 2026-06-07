import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import Iterator

import msgpack
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from velimir.domain_models import Poem, SyllableFeatures
from velimir.nlp import GrammarFeatures
from velimir.settings import GRAMMAR_DB_PATH
from velimir.ml_preprocess import (
    MeterClassRegistry,
    break_into_chunks,
)


@dataclass(slots=True)
class RawSample:
    poem_path: str
    line_idx: int
    chunk_idx: int
    meter_class: int
    syllables: SyllableFeatures
    grammar: GrammarFeatures


def make_accent_input(rs: RawSample) -> torch.Tensor:
    syllables = rs.syllables
    return torch.stack(
        [
            torch.tensor(syllables.linguistic_accents, dtype=torch.float32),
            torch.tensor(syllables.last_in_word, dtype=torch.float32),
        ],
        dim=1,
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


def fetch_raw_samples(poems: Iterator[Poem]) -> Iterator[RawSample]:
    logging.info("Loading raw samples")

    grammar_db = GrammarDB()

    rare_meters_excluded = 0

    for poem in poems:
        chunks = break_into_chunks(poem.lines, poem.stanza_breaks)

        for chunk_idx, chunk in enumerate(chunks):
            for line in chunk:
                syllables = line.syllables
                meter_class = MeterClassRegistry.mc_to_int(line.to_meterclass())

                if meter_class is None:
                    rare_meters_excluded += 1
                    continue

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
                    meter_class=meter_class,
                    poem_path=poem.path,
                    grammar=gf_expanded,
                )

    grammar_db.conn.close()

    logging.info(
        "%d lines are excluded from dataset as having rare meter types",
        rare_meters_excluded,
    )


def split_chunks(
    raw_samples: Iterator[RawSample],
    test_ratio: float = 0.02,
    val_ratio: float = 0.02,
    seed: int = 42,
):
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


@dataclass(slots=True)
class Sample:
    accent_input: torch.Tensor
    pos_input: torch.Tensor
    meter_target: torch.Tensor
    accent_target: torch.Tensor


class SampleDataset(Dataset):
    def __init__(self, chunks: list[list[RawSample]]):
        logging.info("Loading dataset")

        self.samples: list[Sample] = []

        for chunk_lines in chunks:
            accent_tensors = []
            pos_tensors = []
            meter_targets = []
            accent_targets = []

            for rs in chunk_lines:
                accent_tensors.append(make_accent_input(rs))
                pos_tensors.append(
                    torch.tensor(rs.grammar.part_of_speech, dtype=torch.long)
                )
                meter_targets.append(rs.meter_class)
                accent_targets.append(
                    torch.tensor(rs.syllables.poetic_accents, dtype=torch.float32)
                )

            accent_input = pad_sequence(
                accent_tensors, batch_first=True, padding_value=-1
            )
            pos_input = pad_sequence(pos_tensors, batch_first=True, padding_value=-1)
            meter_target = torch.tensor(meter_targets, dtype=torch.long)
            accent_target = pad_sequence(
                accent_targets, batch_first=True, padding_value=-1
            )

            self.samples.append(
                Sample(
                    accent_input=accent_input,
                    pos_input=pos_input,
                    meter_target=meter_target,
                    accent_target=accent_target,
                )
            )

        logging.info(
            "Dataset loading finished. %d chunks created",
            len(self.samples),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_samples(batch: list[Sample]):
    line_counts = torch.tensor(
        [s.accent_input.shape[0] for s in batch], dtype=torch.long
    )
    max_T = max(s.accent_input.shape[1] for s in batch)

    all_accent = []
    all_pos = []
    all_meter = []
    all_accent_target = []

    for s in batch:
        N, T, _ = s.accent_input.shape
        if T < max_T:
            pad_a = F.pad(s.accent_input, (0, 0, 0, max_T - T), value=-1)
            pad_p = F.pad(s.pos_input, (0, max_T - T), value=-1)
            pad_at = F.pad(s.accent_target, (0, max_T - T), value=-1)
        else:
            pad_a = s.accent_input
            pad_p = s.pos_input
            pad_at = s.accent_target

        all_accent.append(pad_a)
        all_pos.append(pad_p)
        all_meter.append(s.meter_target)
        all_accent_target.append(pad_at)

    accent_input = torch.cat(all_accent, dim=0)
    pos_input = torch.cat(all_pos, dim=0)
    meter_target = torch.cat(all_meter, dim=0)
    accent_target = torch.cat(all_accent_target, dim=0)

    sample = Sample(
        accent_input=accent_input,
        pos_input=pos_input,
        meter_target=meter_target,
        accent_target=accent_target,
    )
    return sample, line_counts


def get_loader(chunks, **kwargs):
    dataset = SampleDataset(chunks)
    return DataLoader(dataset, collate_fn=collate_samples, **kwargs)
