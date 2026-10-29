import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import count, groupby, repeat

import numpy as np
from bitarray import bitarray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..onnx import MAX_SEQ_LEN, OnnxRhyme
from .formula import (
    SCHEMALESS_TYPES,
    RhymeFormula,
    RhymeType,
    SpecialRhymeEntry,
)
from .phonetics import PHONETIC_VOCAB, PhoneticRepr, to_phonetic_repr

RHYME_BATCH_SIZE = 4096

ENTRY_COST_CACHE_SIZE = 8192

RHYME_PATTERN_THRESHOLD = 0.25

RHYME_COVERAGE_THRESHOLD = 0.6
SPORADIC_COVERAGE_THRESHOLD = 0.3


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
    max_distance: float = 0.3,
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

        pattern_canonical = canonicalize_pattern(self.pattern.tolist())
        next_pattern_canonical = canonicalize_pattern(next_pattern.tolist())

        return pattern_canonical == next_pattern_canonical

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
    cost: tuple[int, int, int]
    count: int
    entries: tuple[PatternEntry, ...]


@lru_cache(maxsize=ENTRY_COST_CACHE_SIZE)
def entry_cost(entry: PatternEntry) -> tuple[int, int, int]:
    counts = Counter(entry.pattern.tolist())
    singletones = 0

    # Паттерн без внутренних повторов (все метки уникальны) не является
    # рифмовкой, поэтому его строки считаем как одиночные
    if entry.repeats == 1 or len(counts) == len(entry.pattern):
        singletones = len(entry.pattern) * entry.repeats

    # Уникальная положительная метка не рифмуется, её строки одиночные.
    # Цепная рифмовка (diff == 1) связывает строфы уникальной меткой,
    # поэтому её не считаем одиночной.
    if not np.all(entry.diff == 1):
        non_repeating = sum(c for label, c in counts.items() if label >= 0 and c == 1)
        singletones = non_repeating * entry.repeats

    # Поощраем простые паттерны с наличием рифмовки
    complexity = np.any(entry.diff > 2) or not np.any(entry.diff)

    return singletones, int(complexity), -entry.repeats


def calc_diff(arr1, arr2):
    # Отрицательные числа остаются без изменений
    # Ноль и положительные преобразуются. Например:
    # arr2 - [0, 4, 0, 4]
    # arr1 - [2, 3, 2, 3]
    # =>
    # arr2 - [4, 5, 4, 5]
    # arr1 - [2, 3, 2, 3]

    introduced = set(arr1.tolist())
    freevals = count(np.max(arr1) + 1)
    visited = {}

    canon_arr2 = []
    for n in arr2:
        if n in introduced or n < 0:
            canon_arr2.append(n)
        else:
            if n not in visited:
                visited[n] = next(freevals)
            canon_arr2.append(visited[n])

    return np.array(canon_arr2) - arr1


def is_rhyme_repetition(
    pattern: list[int], discovered_rhyme_periods: set[int]
) -> bool:
    period = len(pattern)

    for rhyme_period in discovered_rhyme_periods:
        if period <= rhyme_period or period % rhyme_period:
            continue

        chunks = [
            canonicalize_pattern(pattern[start : start + rhyme_period])
            for start in range(0, period, rhyme_period)
        ]

        if all(chunk == chunks[0] for chunk in chunks[1:]):
            return True

    return False


def build_patterns(inp_l: list[int]) -> list[PatternEntry]:
    arr = np.asarray(inp_l)
    size = len(arr)
    max_period_len = 18

    @cache
    def solve(idx: int, last: PatternEntry | None) -> PatternCandidate:
        if idx >= size:
            if last is None:
                return PatternCandidate((0, 0, 0), 0, ())

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
                tuple(map(sum, zip(entry_cost(last), result.cost))),
                1 + result.count,
                (last,) + result.entries,
            )

        candidates = []
        discovered_rhyme_periods: set[int] = set()

        for period in range(1, max_period_len + 1):
            if period > size - idx:
                continue

            new_pattern = arr[idx : idx + period]

            if period >= 2 and 2 * period <= size - idx:
                diff = calc_diff(new_pattern, arr[idx + period : idx + 2 * period])

                # 0 - нет рифмы / монотонная рифма / рефрен
                # нулевая рифмовка возможна только для строк,
                # ранее отмеченных отрицательными числами
                negative_ok = np.all(new_pattern[diff == 0] < 0)

                has_rhyme = False

                # 1 - цепная рифма, двойная/тройная...
                # 2 - перекрестные, четные...
                # >2 - скользящие и сложные рифмы
                for diff_val in range(1, max_period_len + 1):
                    if has_rhyme:
                        break
                    has_rhyme = has_rhyme or np.all((diff == diff_val) | (diff == 0))

                if np.any(diff) and negative_ok and has_rhyme:
                    candidates.append(start_entry(PatternEntry(new_pattern, diff, 1)))
                    discovered_rhyme_periods.add(period)

            pattern = new_pattern.tolist()

            if not is_rhyme_repetition(pattern, discovered_rhyme_periods):
                candidates.append(
                    start_entry(PatternEntry(new_pattern, np.zeros(period), 1))
                )

        return min(candidates, key=lambda res: (res.cost, res.count))

    return list(solve(0, None).entries)


def canonicalize_pattern(pattern: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    next_label = count()
    result = []
    tauto_entries = {}

    for label in pattern:
        if len(tauto_entries) > 99:
            raise ValueError

        if SpecialRhymeEntry.TAUTO >= label > SpecialRhymeEntry.MONO:
            if not tauto_entries.get(label):
                offset = 99 - len(tauto_entries)
                tauto_entries[label] = SpecialRhymeEntry.TAUTO - offset
            result.append(tauto_entries[label])
            continue

        if label < 0:
            result.append(label)
            continue

        if label not in mapping:
            mapping[label] = next(next_label)

        result.append(mapping[label])

    return result


def classify_rhyme_type(entry: PatternEntry) -> RhymeType:
    pattern = canonicalize_pattern(entry.pattern.tolist())
    unique_entries = len(set(pattern))
    counts = Counter(entry.pattern.tolist())

    match pattern:
        case [a, -1, b, -1] if a == b != -1:
            return RhymeType.ODD
        case [-1, a, -1, b] if a == b != -1:
            return RhymeType.EVEN
        case [a, b, c, d] if a == c != -1 and b == d != -1:
            return RhymeType.CROSS
        case [a, b, c, d, e, f] if a == c == e != -1 and b == d == f != -1:
            return RhymeType.CROSS
        case [a, b, c, d] if a == d != -1 and b == c != -1:
            return RhymeType.ENCIRCLING
        case [0, 0]:
            return RhymeType.PAIRED
        case [0, 0, 0]:
            return RhymeType.TRIPLE
        case [0, 0, 0, 0]:
            return RhymeType.QUADRUPLE
        case [0, 0, 0, 0, 0]:
            return RhymeType.QUINTUPLE
        case [0, 1, 2, 0, 1, 2] | [0, 1, 2, 3, 0, 1, 2, 3]:
            return RhymeType.SLIDING

    if all(label == -1 for label in pattern):
        return RhymeType.UNKNOWN
    if unique_entries == 2 and len(pattern) == 5:
        return RhymeType.DELAYED
    if (
        np.all(entry.diff == 1) == 1
        and len(set(pattern)) < len(pattern)
        and entry.repeats > 2
    ):
        return RhymeType.CHAIN

    non_repeated_entries = sum(
        c for label, c in counts.items() if label >= 0 and c == 1
    )
    if entry.repeats > 1 and not non_repeated_entries:
        return RhymeType.COMPLEX

    return RhymeType.UNKNOWN


def test_rhyming_percent(rhyming_percent) -> RhymeType:
    if rhyming_percent > RHYME_COVERAGE_THRESHOLD:
        return RhymeType.FREE
    elif rhyming_percent > SPORADIC_COVERAGE_THRESHOLD:
        return RhymeType.SPORADIC
    else:
        return RhymeType.NONE


def segment_formula(rt: RhymeType, formula: list[int]) -> list[list[int]]:
    if rt == RhymeType.SLIDING:
        middle = len(formula) // 2
        return [formula[:middle], formula[middle:]]
    if rt == RhymeType.CHAIN:
        return [formula, [f + 1 for f in formula], [f + 2 for f in formula]]
    if rt == RhymeType.COMPLEX:
        res = [[]]
        unique_in_subformula = set()

        for f in formula:
            if (
                len(unique_in_subformula) == 2 and f not in unique_in_subformula
            ) or len(res[-1]) >= 4:
                res.append([])
                unique_in_subformula = set()

            res[-1].append(f)
            unique_in_subformula.add(f)

        return res

    return [formula]


def build_rhyme_formulas(
    patterns: list[PatternEntry],
    poem_schema: list[int],
) -> list[RhymeFormula]:
    if (
        len(set(poem_schema)) == 1
        and list(filter(lambda p: p > -1, poem_schema))
        and len(poem_schema) > 3
    ):
        return [RhymeFormula(RhymeType.MONORHYME)]

    threshold = len(poem_schema) * RHYME_PATTERN_THRESHOLD

    cleaned = [
        dict(
            rtype=classify_rhyme_type(entry),
            pattern=tuple(canonicalize_pattern(entry.pattern)),
            size=len(entry.pattern) * entry.repeats,
        )
        for entry in patterns
    ]

    grouped = groupby(
        cleaned,
        lambda k: (
            k["rtype"],
            k["pattern"] if k["rtype"] not in SCHEMALESS_TYPES else tuple(),
        ),
    )

    joined = []

    for (rtype, formula), ipatterns in grouped:
        entry_patterns = list(ipatterns)
        patterns_len = sum(p["size"] for p in entry_patterns)

        if patterns_len < threshold:
            continue

        if rtype == RhymeType.UNKNOWN:
            continue

        joined.append((rtype, formula))

    if not joined:
        # Различие между вольной/спорадической/нулевой рифмой
        rhyming = sum(label > -1 for label in poem_schema)
        return [RhymeFormula(test_rhyming_percent(rhyming / len(poem_schema)))]

    res = []

    for (rt, formula), _ in groupby(joined):
        if formula:
            res.append(RhymeFormula(rt, segment_formula(rt, formula)))
        else:
            res.append(RhymeFormula(rt))

    return res


def filter_far_away_rhymes(rhymes: list[int]) -> list[int]:
    """
    splits rhyme groups that join lines that are too far away
    doesn't apply to MONO, REFRAIN, NONE
    """
    max_allowed_distance = 8
    rhyme_group_positions = defaultdict(list)

    for i, rhyme in enumerate(rhymes):
        rhyme_group_positions[rhyme].append(i)

    modified_group_positions = defaultdict(list)

    for rhyme_group, positions in rhyme_group_positions.items():
        if rhyme_group < 0:  # don't touch special rhyme entries
            modified_group_positions[rhyme_group] += positions
            continue

        current_group = 0
        subgroups = defaultdict(list)
        distances = [b - a for a, b in zip(positions, positions[1:])]

        subgroups[current_group].append(positions[0])

        for i, dist in enumerate(distances):
            if dist > max_allowed_distance:
                current_group += 1

            subgroups[current_group].append(positions[i + 1])

        max_group = max((g for g in modified_group_positions if g >= 0), default=0)

        for subgroup in subgroups.values():
            if len(subgroup) == 1:
                modified_group_positions[SpecialRhymeEntry.NO_RHYME] += subgroup
                continue

            max_group += 1
            modified_group_positions[max_group] = subgroup

    result = list(repeat(None, len(rhymes)))

    for group, positions in modified_group_positions.items():
        for pos in positions:
            result[pos] = group

    if None in result:
        raise ValueError

    return result


def identify_special_rhyme_entries(
    rhymes: list[RhymeInput], schema: list[int]
) -> tuple[list[int], list[int]]:
    cleaned_words = []
    for r in rhymes:
        word = r.word.lower()
        word = re.sub(r"[^а-яё]", "", word)

        cleaned_words.append(word)

    # TAUTO - the rhyme based on repetition of the same word
    # REFRAIN - the rhyme based on repetition of the same word at regular intervals
    words_with_positions = defaultdict(list)
    for i, word in enumerate(cleaned_words):
        words_with_positions[word].append(i)

    tauto_candidates = []
    refrain = 0

    for positions in words_with_positions.values():
        if len(positions) == 1:
            continue

        distances = [b - a for a, b in zip(positions, positions[1:])]

        if len(distances) > 1 and len(set(distances)) == 1:
            if refrain > 99:
                raise ValueError

            for pos in positions:
                schema[pos] = SpecialRhymeEntry.REFRAIN - refrain

            refrain += 1
        else:
            tauto_candidates.append(positions)

    # TODO: add step for identifying
    # MONO - repeating rhyme through text

    return schema, tauto_candidates


def process_tauto_candidates(
    poem_schema: list[int], tauto_candidates: list[list[int]]
) -> list[int]:
    # проверяем, есть ли в группах, к которым принадлежат найденные кандидаты
    # слова, которые не относятся к тавторифмам-кандидатам
    # если нет - однозначно классифицируем кандидат как тавторифму
    group_positions = defaultdict(set)
    for pos, group_index in enumerate(poem_schema):
        if group_index >= 0:
            group_positions[group_index].add(pos)

    for candidate, positions in enumerate(tauto_candidates):
        if candidate > 99:
            raise ValueError

        pos_set = set(positions)
        related_group_indexes = {poem_schema[pos] for pos in positions}

        for group_index in related_group_indexes:
            group = group_positions.get(group_index)
            if group is not None and len(group) > 1 and group <= pos_set:
                for pos in group:
                    poem_schema[pos] = SpecialRhymeEntry.TAUTO - (99 - candidate)

    return poem_schema


def build_poem_schema(
    rhymes: list[RhymeInput],
    model: OnnxRhyme,
) -> list[int]:
    rhyme_matrix = calc_rhyme_matrix(rhymes, model)

    labels = cluster_rhyme_matrix(rhyme_matrix)

    poem_schema = extract_rhyme_schema(labels)
    poem_schema, tauto_candidates = identify_special_rhyme_entries(rhymes, poem_schema)
    poem_schema = filter_far_away_rhymes(poem_schema)
    poem_schema = process_tauto_candidates(poem_schema, tauto_candidates)

    return canonicalize_pattern(poem_schema)


def render_rhyme_formulas(formulas: list[RhymeFormula]) -> str:
    return " # ".join(formula.to_str() for formula in formulas)


def identify_rhyme_formula(poem_schema: list[int]) -> list[RhymeFormula]:
    patterns = build_patterns(poem_schema)
    formulas = build_rhyme_formulas(patterns, poem_schema)

    return formulas
