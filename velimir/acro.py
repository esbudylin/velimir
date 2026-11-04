from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

import numpy as np

from velimir.nlp import morph_analyzer
from velimir.parsers import is_vowel

MIN_PARTIAL_LENGTH = 6

NAME_GRAMMEMES = {"Name", "Surn", "Patr"}

SINGLE_LETTER_WORDS = ("к", "в", "я", "а", "с", "и", "о", "б")

known_forms: set[str] = set()
char_probabilities: dict[str, float] = {}


@dataclass
class Acro:
    sequence: str
    forms: list[str]
    coverage: int
    total: int
    statistics: dict

    @property
    def partial(self) -> bool:
        return self.coverage < self.total


def normalize_yo(word: str) -> str:
    return word.replace("ё", "е").replace("Ё", "Е")


@lru_cache(maxsize=4096)
def is_name(word: str) -> bool:
    word = normalize_yo(word.lower())

    return any(
        parse.is_known and parse.tag.grammemes & NAME_GRAMMEMES
        for parse in morph_analyzer.parse(word)
    )


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

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "й" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "ы" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
    ]

    for form in forms:
        if form in known_forms or is_name(form):
            return form

    return None


def load_dictionary(forms) -> None:
    known_forms.clear()
    known_forms.update(forms)
    matched_form.cache_clear()


def load_char_probabilities(probabilities: dict[str, float]) -> None:
    char_probabilities.clear()
    char_probabilities.update(probabilities)


@lru_cache(maxsize=8192)
def find_acro_variants(inp: str, is_partial=False):
    if len(inp) == 0:
        raise ValueError

    acc = tuple()

    for i in range(len(inp), 0, -1):
        candidate = inp[:i]
        candidate_matched_form = matched_form(candidate)

        if i == len(inp) and (not is_partial or candidate_matched_form):
            acc = (*acc, ((candidate, candidate_matched_form),))

            if candidate_matched_form:
                break
            else:
                continue

        if not candidate_matched_form and is_partial:
            continue

        solved = find_acro_variants(inp[i:], not candidate_matched_form)
        if solved:
            for candidates in solved:
                acc = (*acc, ((candidate, candidate_matched_form), *candidates))

    return acc


def valid_variant(variant):
    in_dict_count = sum(1 for _, in_dict in variant if in_dict)
    in_dict_total_len = sum(len(w) for w, in_dict in variant if in_dict)
    total_len = sum(len(w) for w, _ in variant)

    if not in_dict_count:
        return False

    if total_len != in_dict_total_len and in_dict_total_len < MIN_PARTIAL_LENGTH:
        return False

    # отсекаем варианты, в которых преобладают короткие слова (1-3 символа)
    def count_shorts(threshold):
        return sum(
            len(word) for word, in_dict in variant if len(word) < threshold and in_dict
        )

    if count_shorts(3) / in_dict_total_len > 0.3:
        return False

    if count_shorts(4) / in_dict_total_len > 0.4:
        return False

    # либо несловарное сочетание единственное и находится последним в строке
    if len(variant) - in_dict_count == 1 and not variant[-1][1]:
        return True

    # либо словарные сочетания занимают большую часть строки
    if in_dict_total_len / total_len > 0.7:
        return True

    return False


def detect_acro(inp: str) -> list[Acro]:
    variants = find_acro_variants(inp)

    if not variants:
        return []

    def rank_variants(variant):
        in_dict_words = [word for word, in_dict in variant if in_dict]

        all_in_dict = len(variant) == len(in_dict_words)
        total_len = sum(len(word) for word, _ in variant)
        total_words = len(variant)
        in_dict_len = sum(map(len, in_dict_words))

        return -all_in_dict, -total_len, total_words, -in_dict_len

    variants = sorted(variants, key=rank_variants)

    variants = filter(valid_variant, variants)

    grouped = groupby(variants, rank_variants)

    _, best = next(grouped, (False, []))

    res = []

    for variant in best:
        forms = [dict_form for _, dict_form in variant if dict_form]
        sequence = "".join(word for word, _ in variant)

        res.append(
            Acro(
                sequence=sequence,
                forms=forms,
                total=len(sequence),
                coverage=sum(map(len, forms)),
                statistics=calculate_sequence_statistics(normalize_yo(sequence)),
            )
        )

    return res
