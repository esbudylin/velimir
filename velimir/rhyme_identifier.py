from collections import Counter
from dataclasses import dataclass
from functools import cache
from statistics import mean
from itertools import count

import numpy as np
from bitarray import bitarray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from velimir import accentuator
from velimir.phonetics import is_consonant_pair, to_phonetic_repr
from velimir.rhyme import SpecialRhymeEntry

RHYME_SCHEMA_ALPHABET = "абвгдежзийкл"


@dataclass
class RhymeInput:
    word: str
    accents: bitarray


def trim_phonetic_pair(pair: tuple[str, str]) -> tuple[str, str]:
    pair_with_syllables = [(w, accentuator.vowel_count(w)) for w in pair]

    short, long = sorted(pair_with_syllables, key=lambda p: p[1])
    short_w, short_s = short
    long_w, long_s = long

    if short_s == long_s:
        return pair

    reversed_new_long_w = ""
    found_vowels = 0

    for ch in reversed(long_w):
        if accentuator.is_vowel(ch):
            found_vowels += 1

        if found_vowels > short_s:
            break

        reversed_new_long_w += ch

    return short_w, "".join(reversed(reversed_new_long_w))


def calc_substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if is_consonant_pair(a, b):
        return 0.5
    if a + b in ["оэ", "эо"]:  # возможная пара о - ё
        return 0.5
    return 1.0


# copied from: https://rosettacode.org/wiki/Levenshtein_distance#Python
def levenshtein_distance(str1, str2) -> float:
    m = len(str1)
    n = len(str2)

    d = [[i] for i in range(1, m + 1)]  # d matrix rows
    d.insert(0, list(range(0, n + 1)))  # d matrix columns

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            substitution_cost = calc_substitution_cost(str1[i - 1], str2[j - 1])

            d[i].insert(
                j,
                min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + substitution_cost,
                ),
            )

    return d[-1][-1]


def calc_rhyming_coef(rhymes: list[RhymeInput]) -> float:
    phonetic_words = list(map(lambda r: to_phonetic_repr(r.word, r.accents), rhymes))

    phonetic_pairs = []

    for i, word in enumerate(phonetic_words):
        if i + 1 == len(phonetic_words):
            break

        next_word = phonetic_words[i + 1]
        phonetic_pairs.append((word, next_word))

    phonetic_pairs = list(map(trim_phonetic_pair, phonetic_pairs))

    weighted_distances = map(
        lambda pair: calc_phonetic_rhyme_coef(*pair), phonetic_pairs
    )

    return mean(weighted_distances)


@cache
def calc_phonetic_rhyme_coef(phonetic_a: str, phonetic_b: str) -> float:
    a, b = trim_phonetic_pair((phonetic_a, phonetic_b))

    if not a and not b:
        raise ValueError("Zero size inputs")

    distance = levenshtein_distance(a, b)

    return 1.0 - distance / max(len(a), len(b))


def calc_rhyme_matrix(rhymes: list[RhymeInput]) -> np.ndarray:
    phonetic_words = [to_phonetic_repr(rhyme.word, rhyme.accents) for rhyme in rhymes]

    size = len(phonetic_words)
    matrix = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        matrix[i, i] = 1.0

        for j in range(i + 1, size):
            coef = calc_phonetic_rhyme_coef(phonetic_words[i], phonetic_words[j])
            matrix[i, j] = coef
            matrix[j, i] = coef

    return matrix


def cluster_rhyme_matrix(
    matrix: np.ndarray,
    *,
    max_distance: float = 0.8,
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


def format_rhyme_schema(schema: list[int]) -> str:
    letters = []

    for entry in schema:
        match entry:
            case SpecialRhymeEntry.NO_RHYME:
                letters.append("х")
            case SpecialRhymeEntry.TAUTO:
                letters.append("т")
            case SpecialRhymeEntry.MONO:
                letters.append("м")
            case SpecialRhymeEntry.REFRAIN:
                letters.append("р")
            case _:
                letters.append(RHYME_SCHEMA_ALPHABET[entry])

    return "".join(letters)


def identify_rhyme_schema(rhymes: list[RhymeInput]) -> str:
    m = calc_rhyme_matrix(rhymes)
    c = cluster_rhyme_matrix(m)

    return " ".join(map(lambda a: str(int(a)), extract_rhyme_schema(c)))
    # return format_rhyme_schema(extract_rhyme_schema(c))
