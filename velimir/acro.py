from functools import lru_cache
from itertools import groupby

from spylls.hunspell import Dictionary
from velimir.parsers import is_vowel

dictionary = Dictionary.from_files("data/wikt/ru_RU")


@lru_cache(maxsize=2048)
def is_word(word):
    if len(word) == 1:
        return word in ["к", "в", "я", "а", "с", "и", "о"]

    if not any(map(is_vowel, word)):
        return False

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "й" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "ы" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
    ]
    return any(map(lambda w: dictionary.lookup(w.capitalize()), forms))


def find_acro_variants(inp: str) -> list[list[str]]:
    if len(inp) == 0:
        raise ValueError

    acc = []
    for i in range(len(inp), 0, -1):
        candidate = inp[:i]

        if not is_word(candidate):
            continue

        if i == len(inp):
            acc.append([candidate])
            continue

        solved = find_acro_variants(inp[i:])
        if solved:
            for candidates in solved:
                acc.append([candidate, *candidates])

    return acc


def detect_acro(inp: str) -> list[str]:
    variants = find_acro_variants(inp)

    if not variants:
        return []

    variants = sorted(variants, key=len)
    grouped = groupby(variants, len)
    _, best = next(grouped)

    res = []

    for variant in best:

        def count_shorts(threshold):
            return sum(1 for word in variant if len(word) < threshold)

        if count_shorts(3) / len(variant) > 0.4:
            continue

        if count_shorts(4) / len(variant) > 0.5:
            continue

        res.append(" ".join(word for word in variant))

    return res
