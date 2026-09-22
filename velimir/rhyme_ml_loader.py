import logging
import random
import sqlite3
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from velimir.phonetics import PHONETIC_VOCAB

PAD_ID = 0
EMBEDDING_SIZE = len(PHONETIC_VOCAB) + 1


@dataclass(slots=True)
class RhymePairRow:
    label: int
    phon_a: str
    stress_a: str
    phon_b: str
    stress_b: str


@dataclass(slots=True)
class RhymePairSample:
    ids_a: torch.Tensor
    stress_a: torch.Tensor
    ids_b: torch.Tensor
    stress_b: torch.Tensor
    label: float


@dataclass(slots=True)
class RhymePairBatch:
    ids_a: torch.Tensor
    stress_a: torch.Tensor
    ids_b: torch.Tensor
    stress_b: torch.Tensor
    labels: torch.Tensor


def load_rhyme_pairs(db_path: str) -> list[RhymePairRow]:
    logging.info("Loading rhyme pairs from %s", db_path)

    conn = sqlite3.connect(db_path)

    try:
        rows = [
            RhymePairRow(*row)
            for row in conn.execute(
                """
                SELECT label,
                       phon_a, accents_a,
                       phon_b, accents_b
                FROM pairs
                ORDER BY id
                """
            )
        ]
    finally:
        conn.close()

    logging.info("Loaded %d rhyme pairs", len(rows))

    return rows


def encode_phonetic(
    phon: str,
    stress: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        ids = torch.tensor([PHONETIC_VOCAB[ch] for ch in phon], dtype=torch.long)
    except KeyError as error:
        raise ValueError(
            f"Unknown phonetic character {error.args[0]!r} in {phon!r}"
        ) from error

    flags = torch.tensor([s == "1" for s in stress], dtype=torch.float32)

    return ids, flags


class RhymePairDataset(Dataset):
    def __init__(self, rows: list[RhymePairRow]):
        self.samples = [self._encode(row) for row in rows]

    @staticmethod
    def _encode(row: RhymePairRow) -> RhymePairSample:
        ids_a, stress_a = encode_phonetic(row.phon_a, row.stress_a)
        ids_b, stress_b = encode_phonetic(row.phon_b, row.stress_b)

        return RhymePairSample(
            ids_a=ids_a,
            stress_a=stress_a,
            ids_b=ids_b,
            stress_b=stress_b,
            label=float(row.label),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> RhymePairSample:
        return self.samples[idx]


def collate_rhyme_pairs(batch: list[RhymePairSample]) -> RhymePairBatch:
    return RhymePairBatch(
        ids_a=pad_sequence(
            [s.ids_a for s in batch], batch_first=True, padding_value=PAD_ID
        ),
        stress_a=pad_sequence(
            [s.stress_a for s in batch], batch_first=True, padding_value=0.0
        ),
        ids_b=pad_sequence(
            [s.ids_b for s in batch], batch_first=True, padding_value=PAD_ID
        ),
        stress_b=pad_sequence(
            [s.stress_b for s in batch], batch_first=True, padding_value=0.0
        ),
        labels=torch.tensor([s.label for s in batch], dtype=torch.float32),
    )


def split_pairs(
    rows: list[RhymePairRow],
    test_ratio: float = 0.02,
    val_ratio: float = 0.02,
    seed: int = 195968,
):
    rng = random.Random(seed)

    shuffled = list(rows)
    rng.shuffle(shuffled)

    n = len(shuffled)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    test_rows = shuffled[:test_size]
    val_rows = shuffled[test_size : test_size + val_size]
    train_rows = shuffled[test_size + val_size :]

    logging.info(
        "Split %d pairs into train=%d, val=%d, test=%d",
        n,
        len(train_rows),
        len(val_rows),
        len(test_rows),
    )

    return train_rows, val_rows, test_rows


def get_rhyme_pair_loader(
    rows: list[RhymePairRow],
    **kwargs,
) -> DataLoader:
    dataset = RhymePairDataset(rows)
    return DataLoader(dataset, collate_fn=collate_rhyme_pairs, **kwargs)
