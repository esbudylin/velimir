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


def transform_clusters_into_formula(clusters: list[int]) -> RhymeFormula:
    pass


def identify_rhyme_schema(rhymes: list[RhymeInput], model: OnnxRhyme) -> str:
    m = calc_rhyme_matrix(rhymes, model)
    c = cluster_rhyme_matrix(m)

    return " ".join(map(lambda a: str(int(a)), extract_rhyme_schema(c)))
    # return format_rhyme_schema(extract_rhyme_schema(c))
