from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

from velimir.nlp import morph_analyzer
from velimir.parsers import is_vowel

MIN_PARTIAL_LENGTH = 6

NAME_GRAMMEMES = {"Name", "Surn", "Patr"}

SINGLE_LETTER_WORDS = ("к", "в", "я", "а", "с", "и", "о", "б")

known_forms: set[str] = set()


@dataclass
class Acro:
    words: list[str]
    forms: list[str]
    coverage: int
    total: int

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

    if count_shorts(3) / in_dict_total_len > 0.4:
        return False

    if count_shorts(4) / in_dict_total_len > 0.5:
        return False

    # либо несловарное сочетание единственное и находится последним в строке
    if len(variant) - in_dict_count == 1 and not variant[-1][1]:
        return True

    # либо словарные сочетания занимают большую часть строки
    if in_dict_total_len / total_len > 0.75:
        return True

    return False


def detect_acro(inp: str) -> list[Acro]:
    variants = find_acro_variants(inp)

    if not variants:
        return []

    variants = sorted(
        variants,
        key=lambda n: (-all(in_dict for _, in_dict in n), len(n)),
    )

    variants = filter(valid_variant, variants)

    grouped = groupby(variants, len)

    _, best = next(grouped, (False, []))

    res = []

    for variant in best:
        words = [word for word, dict_form in variant if dict_form]
        forms = [dict_form for _, dict_form in variant if dict_form]
        total = sum(len(word) for word, _ in variant)
        coverage = sum(map(len, words))

        res.append(Acro(words=words, forms=forms, total=total, coverage=coverage))

    return res
