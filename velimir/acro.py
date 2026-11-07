from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import inf

import numpy as np
from velimir.parsers import is_vowel


SINGLE_LETTER_WORDS = {"к", "в", "я", "а", "с", "и", "о", "б", "у"}

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


@lru_cache(maxsize=8192)
def matched_form(word: str) -> str | None:
    word = normalize_yo(word.lower())

    if len(word) == 1:
        return word if word in SINGLE_LETTER_WORDS else None

    if not any(map(is_vowel, word)):
        return None

    if word.endswith("ъ"):  # редкий случай - дореформенная орфография
        word = word[:-1]

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "ъ" + word[i:] for i in range(1, len(word))),
        *(word[:i] + "й" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "ы" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "е" + word[i + 1 :] for i, c in enumerate(word) if c == "э"),
        *(word[:i] + "э" + word[i + 1 :] for i, c in enumerate(word) if c == "е"),
    ]

    for form in forms:
        if form in known_forms:
            return form

    return None


def load_dictionary(forms) -> None:
    known_forms.clear()
    known_forms.update(forms)
    matched_form.cache_clear()


def load_char_probabilities(probabilities: dict[str, float]) -> None:
    char_probabilities.clear()
    char_probabilities.update(probabilities)


def valid_variant(variant, is_partial):
    total_len = sum(len(w) for w, _, _ in variant)

    if total_len < 4:
        return False

    if is_partial and total_len < 6:
        return False

    # отсекаем варианты, в которых преобладают короткие слова (1-3 символа)
    def count_shorts(threshold):
        return sum(len(word) for word, in_dict, _ in variant if len(word) < threshold)

    if count_shorts(3) / total_len > 0.3:
        return False

    if is_partial and count_shorts(4) / total_len > 0.4:
        return False

    return True


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
            u_d, _ = single_source[u]
            v_d, _ = single_source[v]

            new_distance = u_d + dist[u, v]

            if v_d > new_distance:
                single_source[v] = (new_distance, u)

        for i, neighbors in vertices.items():
            if i < start:
                continue

            for j in neighbors:
                relax(i, j)

        return single_source

    # reconstruct path
    def reconstruct_path(source):
        reachable = [
            (distance, vertex)
            for vertex, (distance, parent) in enumerate(source)
            if parent is not None
        ]

        if not reachable:
            return []

        _, current = min(reachable)

        res = deque([current])

        while (parent := source[current][1]) is not None:
            res.appendleft(parent)
            current = parent

        return list(res)

    paths = []

    for start in vertices:
        if not vertices[start]:
            continue

        source = dag_critical_path(start)
        path = reconstruct_path(source)

        if not path:
            continue

        paths.append(path)
        # path already takes the whole substring
        if path[-1] - path[0] == len(inp) - start:
            break

    if not paths:
        return [], None

    def reconstruct_word_seq(path):
        res = []
        for i, j in zip(path, path[1:]):
            res.append((inp[i:j], candidates_by_slice[i, j], i))

        return res

    paths = sorted(paths, key=lambda p: p[0] - p[-1])

    for path in paths:
        seq = reconstruct_word_seq(path)
        is_partial = path[0] != 0 or path[-1] != len(inp)
        if valid_variant(seq, is_partial):
            return seq, is_partial

    return [], None


def detect_acro(inp: str) -> Acro | None:
    variant, partial = find_acro(inp)

    if not variant:
        return None

    forms = [dict_form for _, dict_form, _ in variant if dict_form]
    sequence = "".join(word for word, _, _ in variant)

    return Acro(
        sequence=sequence,
        forms=forms,
        statistics=calculate_sequence_statistics(normalize_yo(sequence)),
        partial=partial,
    )
