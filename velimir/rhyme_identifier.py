from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np
from bitarray import bitarray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from velimir.onnx import MAX_SEQ_LEN, OnnxRhyme
from velimir.phonetics import PHONETIC_VOCAB, PhoneticRepr, to_phonetic_repr
from velimir.rhyme import RhymeFormula, RhymeType, SpecialRhymeEntry

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
    pattern: list[int]
    diff: list[int]
    repeats: int

    def matches(self, next_pattern: list[int]) -> bool:
        if len(self.pattern) < 2:
            return False
        if len(self.pattern) != len(next_pattern):
            return False

        diff = np.asarray(self.diff)

        if not np.any(diff):
            return False

        expected = np.asarray(self.pattern) + diff * self.repeats

        return bool(np.array_equal(expected, np.asarray(next_pattern)))


def decomposition_cost(entries: list[PatternEntry]) -> tuple[int, int, int]:
    singleton_lines = 0
    zero_shifts = 0

    for entry in entries:
        # Паттерн без внутренних повторов (все метки уникальны) не является
        # рифмовкой, поэтому его строки считаем как одиночные
        if entry.repeats == 1 or len(set(entry.pattern)) == len(entry.pattern):
            singleton_lines += len(entry.pattern) * entry.repeats
            continue

        pattern = np.asarray(entry.pattern)
        diff = np.asarray(entry.diff)

        # Понижаем приоритет паттернов с 0 в diff векторе
        # 0 обозначает монотонные и отсутвующие рифмы
        # которые реже встречаются в текстах
        zero_shifts += int(np.count_nonzero((diff == 0) & (pattern >= 0)))

    return (singleton_lines, zero_shifts, len(entries))


def build_patterns(inp: list[int], acc: list[PatternEntry]) -> list[PatternEntry]:
    max_period_len = 14

    outs = []

    for period in range(1, max_period_len + 1):
        if period > len(inp):
            break

        last_res = acc[-1] if len(acc) > 0 else None

        new_pattern = inp[:period]

        if last_res and last_res.matches(new_pattern):
            new_acc = acc[:-1] + [
                PatternEntry(
                    pattern=last_res.pattern,
                    diff=last_res.diff,
                    repeats=last_res.repeats + 1,
                )
            ]

            return build_patterns(inp[period:], new_acc)

        if period >= 2 and 2 * period <= len(inp):
            diff = np.asarray(inp[period : 2 * period]) - np.asarray(new_pattern)

            # 0 - нет рифмы / монотонная рифма
            # 1 - цепная рифма, двойная/тройная...
            # 2 - всё прочее
            if np.all(diff >= 0) and np.all(diff <= 2) and np.any(diff):
                outs.append(
                    build_patterns(
                        inp[period:],
                        acc
                        + [
                            PatternEntry(
                                pattern=new_pattern,
                                diff=diff.tolist(),
                                repeats=1,
                            )
                        ],
                    )
                )

        outs.append(
            build_patterns(
                inp[period:],
                acc
                + [
                    PatternEntry(
                        pattern=new_pattern,
                        diff=[0] * period,
                        repeats=1,
                    )
                ],
            )
        )

    return min(outs, key=decomposition_cost) if outs else acc


def canonicalize_pattern(pattern: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    next_label = 0
    result = []

    for label in pattern:
        if label < 0:
            result.append(label)
            continue

        if label not in mapping:
            mapping[label] = next_label
            next_label += 1

        result.append(mapping[label])

    return result


def classify_rhyme_type(entry: PatternEntry) -> RhymeType:
    pattern = canonicalize_pattern(entry.pattern)
    unique_entries = len(set(pattern))

    match pattern:
        case [0, 1, 0, 1]:
            return RhymeType.CROSS
        case [0, 1, 1, 0]:
            return RhymeType.ENCIRCLING
        case [0, -1, 0, -1]:
            return RhymeType.ODD
        case [-1, 0, -1, 0]:
            return RhymeType.EVEN
        case [0, 0]:
            return RhymeType.PAIRED
        case [0, 0, 0]:
            return RhymeType.TRIPLE
        case [0, 0, 0, 0]:
            return RhymeType.QUADRUPLE
        case [0, 0, 0, 0, 0]:
            return RhymeType.QUINTUPLE

    if all(label == -1 for label in pattern):
        return RhymeType.NONE
    if unique_entries == 2 and len(pattern) == 5:
        return RhymeType.DELAYED
    if unique_entries == 1:
        return RhymeType.MONORHYME
    # Цепная рифмовка: единый сдвиг на 1 и наличие внутренних повторов
    if set(entry.diff) == {1} and len(set(pattern)) < len(pattern):
        return RhymeType.CHAIN

    return RhymeType.COMPLEX


def build_rhyme_formulas(patterns: list[PatternEntry]) -> list[RhymeFormula]:
    return [
        RhymeFormula(classify_rhyme_type(entry), [canonicalize_pattern(entry.pattern)])
        for entry in patterns
        if entry.repeats > 1 or len(patterns) == 1
    ]


def render_rhyme_formulas(formulas: list[RhymeFormula]) -> str:
    return " # ".join(formula.to_str() for formula in formulas)


def identify_rhyme_schema(
    rhymes: list[RhymeInput],
    model: OnnxRhyme,
) -> list[RhymeFormula]:
    rhyme_matrix = calc_rhyme_matrix(rhymes, model)
    clusters = cluster_rhyme_matrix(rhyme_matrix)
    poem_schema = extract_rhyme_schema(clusters)
    patterns = build_patterns(poem_schema, [])

    return build_rhyme_formulas(patterns)
