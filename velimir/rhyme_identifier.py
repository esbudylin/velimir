from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np
from bitarray import bitarray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from velimir.onnx import MAX_SEQ_LEN, OnnxRhyme
from velimir.phonetics import PHONETIC_VOCAB, PhoneticRepr, to_phonetic_repr
from velimir.rhyme import RhymeFormula, SpecialRhymeEntry

RHYME_BATCH_SIZE = 4096


@dataclass
class RhymeInput:
    word: str
    accents: bitarray


def encode_phonetic(phon: PhoneticRepr) -> tuple[np.ndarray, np.ndarray]:
    ids = [PHONETIC_VOCAB[ch] for ch in phon.phonetics[:MAX_SEQ_LEN]]
    stress = [float(flag) for flag in phon.accents[:MAX_SEQ_LEN]]

    padding = MAX_SEQ_LEN - len(ids)

    return (
        np.array(ids + [0] * padding, dtype=np.int64),
        np.array(stress + [0.0] * padding, dtype=np.float32),
    )


def calc_rhyme_probs(
    pairs: list[tuple[PhoneticRepr, PhoneticRepr]],
    model: OnnxRhyme,
) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float64)

    probs = []

    for start in range(0, len(pairs), RHYME_BATCH_SIZE):
        chunk = pairs[start : start + RHYME_BATCH_SIZE]
        encoded = [(encode_phonetic(a), encode_phonetic(b)) for a, b in chunk]

        logits = model(
            np.stack([a[0] for a, _ in encoded]),
            np.stack([a[1] for a, _ in encoded]),
            np.stack([b[0] for _, b in encoded]),
            np.stack([b[1] for _, b in encoded]),
        )

        probs.append(1.0 / (1.0 + np.exp(-logits)))

    return np.concatenate(probs)


def calc_rhyme_matrix(rhymes: list[RhymeInput], model: OnnxRhyme) -> np.ndarray:
    phonetic_words = [to_phonetic_repr(rhyme.word, rhyme.accents) for rhyme in rhymes]

    size = len(phonetic_words)
    matrix = np.eye(size, dtype=np.float64)

    index_pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]

    if not index_pairs:
        return matrix

    probs = calc_rhyme_probs(
        [(phonetic_words[i], phonetic_words[j]) for i, j in index_pairs],
        model,
    )

    for (i, j), prob in zip(index_pairs, probs):
        matrix[i, j] = prob
        matrix[j, i] = prob

    return matrix


def cluster_rhyme_matrix(
    matrix: np.ndarray,
    *,
    max_distance: float = 0.5,
) -> list[int]:
    size = matrix.shape[0]

    if size < 2:
        return list(range(size))

    distance = 1.0 - matrix

    hierarchy = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(hierarchy, t=max_distance, criterion="distance")

    relabled = {}
    key = count()

    for label in labels:
        if label in relabled:
            continue

        relabled[label] = next(key)

    return [relabled[la] for la in labels]


def extract_rhyme_schema(labels: list[int]) -> list[int]:
    sizes = Counter(labels)

    schema = []

    for label in labels:
        if sizes[label] == 1:
            schema.append(SpecialRhymeEntry.NO_RHYME)
        else:
            schema.append(label)

    return schema


@dataclass
class PatternEntry:
    patterns: list[int]
    diff: int
    repeats: int

    def matches(self, next_pattern, next_diff):
        if self.diff != next_diff:
            return False
        if len(self.patterns) != len(next_pattern):
            return False
        if (
            list(map(lambda a: a + (self.diff * self.repeats), self.patterns))
            != next_pattern
        ):
            return False
        return True


def build_patterns(inp: list[int], acc: list[PatternEntry]) -> list[PatternEntry]:
    max_period_len = 14
    period_diffs = [1, 2]  # 2 - обычная рифма, 1 - цепная

    outs = []

    for period in range(1, max_period_len + 1):
        if period > len(inp):
            break

        for diff in period_diffs:
            last_res = acc[-1] if len(acc) > 0 else None

            new_pattern = inp[:period]

            if last_res and last_res.matches(new_pattern, diff):
                new_acc = acc[:-1] + [
                    PatternEntry(
                        patterns=last_res.patterns,
                        diff=diff,
                        repeats=last_res.repeats + 1,
                    )
                ]

                return build_patterns(inp[period:], new_acc)

            new_acc = acc + [
                PatternEntry(
                    patterns=new_pattern,
                    diff=diff,
                    repeats=1,
                )
            ]

            outs.append(build_patterns(inp[period:], new_acc))

    # TODO handle cases with same len
    return sorted(outs, key=len)[0] if outs else acc


def identify_rhyme_schema(
    rhymes: list[RhymeInput],
    model: OnnxRhyme,
) -> list[RhymeFormula]:
    rhyme_matrix = calc_rhyme_matrix(rhymes, model)
    clusters = cluster_rhyme_matrix(rhyme_matrix)
    poem_schema = extract_rhyme_schema(clusters)

    return build_patterns(poem_schema, [])
