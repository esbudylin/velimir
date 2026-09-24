from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import count

import numpy as np
from bitarray import bitarray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from velimir.onnx import MAX_SEQ_LEN, OnnxRhyme
from velimir.phonetics import PHONETIC_VOCAB, PhoneticRepr, to_phonetic_repr
from velimir.rhyme import RhymeFormula, RhymeType, SpecialRhymeEntry

RHYME_BATCH_SIZE = 4096

RHYME_COVERAGE_THRESHOLD = 0.25
SPORADIC_COVERAGE_THRESHOLD = 0.05


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
    pattern: np.ndarray
    diff: np.ndarray
    repeats: int

    def matches(self, next_pattern: np.ndarray) -> bool:
        if len(self.pattern) < 2:
            return False
        if len(self.pattern) != len(next_pattern):
            return False

        if not np.any(self.diff):
            return False

        expected = self.pattern + self.diff * self.repeats

        return bool(np.array_equal(expected, next_pattern))

    def _key(self) -> tuple:
        return (
            self.pattern.tobytes(),
            self.diff.tobytes(),
            self.repeats,
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PatternEntry):
            return NotImplemented
        return self._key() == other._key()


@dataclass
class PatternCandidate:
    cost: int
    count: int
    entries: tuple[PatternEntry, ...]


@cache
def entry_cost(entry: PatternEntry) -> int:
    counts = Counter(entry.pattern.tolist())

    # Паттерн без внутренних повторов (все метки уникальны) не является
    # рифмовкой, поэтому его строки считаем как одиночные
    if entry.repeats == 1 or len(counts) == len(entry.pattern):
        return len(entry.pattern) * entry.repeats

    # Уникальная положительная метка не рифмуется, её строки одиночные.
    # Цепная рифмовка (diff == 1) связывает строфы уникальной меткой,
    # поэтому её не считаем одиночной.
    if not np.all(entry.diff == 1):
        return (
            sum(c for label, c in counts.items() if label >= 0 and c == 1)
            * entry.repeats
        )

    return 0


def build_patterns(inp_l: list[int]) -> list[PatternEntry]:
    arr = np.asarray(inp_l)
    size = len(arr)
    max_period_len = 14

    @cache
    def solve(idx: int, last: PatternEntry | None) -> PatternCandidate:
        if idx >= size:
            if last is None:
                return PatternCandidate(0, 0, ())

            return PatternCandidate(entry_cost(last), 1, (last,))

        if last is not None:
            period = len(last.pattern)

            if idx + period <= size and last.matches(arr[idx : idx + period]):
                extended = PatternEntry(last.pattern, last.diff, last.repeats + 1)
                return solve(idx + period, extended)

        def start_entry(new_entry: PatternEntry) -> PatternCandidate:
            result = solve(idx + len(new_entry.pattern), new_entry)

            if last is None:
                return result

            return PatternCandidate(
                entry_cost(last) + result.cost,
                1 + result.count,
                (last,) + result.entries,
            )

        candidates = []

        for period in range(1, max_period_len + 1):
            if period > size - idx:
                continue

            new_pattern = arr[idx : idx + period]

            if period >= 2 and 2 * period <= size - idx:
                diff = arr[idx + period : idx + 2 * period] - new_pattern

                # 0 - нет рифмы / монотонная рифма / рефрен
                # нулевая рифмовка возможна только для строк,
                # ранее отмеченных отрицательными числами
                negative_ok = bool(np.all(new_pattern[diff == 0] < 0))

                # 1 - цепная рифма, двойная/тройная...
                first_type = bool(np.all((diff == 1) | (diff == 0)))

                # 2 - всё прочее
                second_type = bool(np.all((diff == 2) | (diff == 0)))

                if np.any(diff) and negative_ok and (first_type or second_type):
                    candidates.append(start_entry(PatternEntry(new_pattern, diff, 1)))

            candidates.append(
                start_entry(PatternEntry(new_pattern, np.zeros(period), 1))
            )

        return min(candidates, key=lambda res: (res.cost, res.count))

    return list(solve(0, None).entries)


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
    pattern = canonicalize_pattern(entry.pattern.tolist())
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
    if np.all(entry.diff == 1) == 1 and len(set(pattern)) < len(pattern):
        return RhymeType.CHAIN

    return RhymeType.COMPLEX


def build_rhyme_formulas(
    patterns: list[PatternEntry],
    inp_len: int,
) -> list[RhymeFormula]:
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
    patterns = build_patterns(poem_schema)

    return build_rhyme_formulas(patterns, len(rhymes))
