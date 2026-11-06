from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from math import inf

import numpy as np
from velimir.nlp import morph_analyzer
from velimir.parsers import is_vowel

MIN_PARTIAL_LENGTH = 6

NAME_GRAMMEMES = {"Name", "Surn", "Patr"}

SINGLE_LETTER_WORDS = ("к", "в", "я", "а", "с", "и", "о", "б", "у")

known_forms: set[str] = set()
char_probabilities: dict[str, float] = {}


@dataclass
class Acro:
    sequence: str
    forms: list[str]
    statistics: dict
    partial: bool


def normalize_yo(word: str) -> str:
    return word.replace("ё", "е").replace("Ё", "Е")


@lru_cache(maxsize=128)
def simulated_chi2(
    n: int,
    simulations: int,
    seed: int | None,
    chars: tuple[str, ...],
    probs: tuple[float, ...],
) -> np.ndarray:
    probs_array = np.array(probs)
    expected = n * probs_array
    valid = expected > 0

    rng = np.random.default_rng(seed)
    simulated = rng.multinomial(n, probs_array, size=simulations)
    chi2 = np.sum(
        (simulated[:, valid] - expected[valid]) ** 2 / expected[valid],
        axis=1,
    )
    chi2.sort()

    return chi2


def calculate_sequence_statistics(
    sequence: str,
    simulations: int = 100_000,
    seed: int | None = 98514,
) -> dict:
    """
    Test whether a sequence deviates from an independent character
    generation model with known character probabilities.

    Uses Pearson's chi-square statistic with Monte Carlo calibration.
    """

    if not sequence:
        raise ValueError("Sequence cannot be empty")

    if any(p < 0 for p in char_probabilities.values()):
        raise ValueError("Probabilities cannot be negative")

    if not np.isclose(sum(char_probabilities.values()), 1.0):
        raise ValueError("Probabilities must sum to 1")

    if any(char not in char_probabilities for char in sequence):
        raise ValueError("Sequence contains unknown characters")

    chars = list(char_probabilities)
    probs = np.array([char_probabilities[c] for c in chars])

    n = len(sequence)
    observed = np.array([sequence.count(c) for c in chars])
    expected = n * probs

    # Pearson's chi-square statistic.
    # Characters with zero probability must also have zero observations.
    valid = expected > 0

    if np.any(observed[~valid] > 0):
        return {
            "exact_probability": 0.0,
            "chi2": float("inf"),
            "p_value": 1 / (simulations + 1),
        }

    chi2_observed = np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid])

    # Exact probability of the ordered sequence.
    log_probability = sum(np.log(char_probabilities[c]) for c in sequence)
    exact_probability = np.exp(log_probability)

    # Monte Carlo calibration.
    chi2_simulated = simulated_chi2(
        n,
        simulations,
        seed,
        tuple(chars),
        tuple(probs),
    )

    # Include observed statistic using the standard +1 correction.
    count = len(chi2_simulated) - np.searchsorted(
        chi2_simulated, chi2_observed, side="left"
    )
    p_value = (count + 1) / (simulations + 1)

    return {
        "exact_probability": exact_probability,
        "log_probability": log_probability,
        "chi2": chi2_observed,
        "p_value": p_value,
    }


def is_name(parsed) -> bool:
    return any(parse.tag.grammemes & NAME_GRAMMEMES for parse in parsed)


@lru_cache(maxsize=4096)
def parse_word(word: str):
    return tuple(w for w in morph_analyzer.parse(word) if w.is_known)


@lru_cache(maxsize=8192)
def matched_form(word: str) -> str | None:
    word = normalize_yo(word.lower())

    if len(word) == 1 or len(word) == 2:
        return False
        # return word if word in SINGLE_LETTER_WORDS else None

    if not any(map(is_vowel, word)):
        return None

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "ъ" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "й" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "ы" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
    ]

    for form in forms:
        if form in known_forms:
            return form

        parsed = parse_word(form)

        if is_name(parsed):
            return form

        normal_forms = {w.normal_form for w in parsed}

        if known_forms.intersection(normal_forms):
            return form

    return None


def load_dictionary(forms) -> None:
    known_forms.clear()
    known_forms.update(forms)
    matched_form.cache_clear()


def load_char_probabilities(probabilities: dict[str, float]) -> None:
    char_probabilities.clear()
    char_probabilities.update(probabilities)


def find_acro(inp: str):
    if len(inp) == 0:
        raise ValueError

    # build graph
    candidates_by_slice = {}
    vertices = {}

    for i in range(len(inp) + 1):
        vertices[i] = []

        for j in range(i + 1, len(inp) + 1):
            candidate = inp[i:j]
            candidate_matched_form = matched_form(candidate)

            if not candidate_matched_form:
                continue

            candidates_by_slice[(i, j)] = candidate_matched_form

            vertices[i].append(j)

    # calculate distances
    dist = {(i, j): inf for i in vertices for j in vertices}

    for i, neighbors in vertices.items():
        for j in neighbors:
            dist[i, j] = -len(inp[i:j])  # reward longest words

    # apply search
    def dag_critical_path(start):
        single_source = [(inf, None) for v in vertices]
        single_source[start] = (0, None)

        def relax(u, v):
            u_d, u_p = single_source[u]
            new_distance = u_d + dist[u, v]

            if u_d > new_distance:
                single_source[v] = (new_distance, u)

        for i, neighbors in vertices.items():
            if i < start:
                continue

            for j in neighbors:
                relax(i, j)

        return single_source

    # reconstruct path and word sequence
    def reconstruct_path(start, source):
        ps = {p: i for i, (_, p) in enumerate(source) if p is not None}

        if not ps:
            return []

        current = start
        res = [current]

        while current := ps.get(current):
            res.append(current)

        return res

    paths = []

    for start in vertices:
        if not vertices[start]:
            continue

        source = dag_critical_path(start)
        path = reconstruct_path(start, source)

        if not path:
            continue

        paths.append(path)
        # path already takes the whole substring
        if path[-1] - path[0] == len(inp) - start:
            break

    if not paths:
        return []

    paths = sorted(paths, key=lambda p: p[0] - p[-1])
    best_path = paths[0]

    res = []

    for i, j in zip(best_path, best_path[1:]):
        res.append((inp[i:j], candidates_by_slice[i, j], i))

    return res


def detect_acro(inp: str) -> Acro | None:
    variant = find_acro(inp)

    if not variant:
        return None

    forms = [dict_form for _, dict_form, _ in variant if dict_form]
    sequence = "".join(word for word, _, _ in variant)
    partial = variant[0][2] != 0

    # TODO : move to find_acro_variants
    if len(sequence) < 4 or partial and len(sequence) < 7:
        return None

    return Acro(
        sequence=sequence,
        forms=forms,
        statistics=calculate_sequence_statistics(normalize_yo(sequence)),
        partial=partial,
    )
