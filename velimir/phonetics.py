import re
from dataclasses import dataclass
from statistics import mean

from bitarray import bitarray

from velimir import accentuator

voiced = ["б", "з", "д", "в", "г", "ж"]
voiceless = ["п", "с", "т", "ф", "к", "ш"]
consonant_pairs = [a + b for a, b in zip(voiced, voiceless)]


@dataclass
class RhymeInput:
    word: str
    accents: bitarray


def subst_vowel(ch: str, has_accent: bool):
    subst = {
        "ю": "у",
        "я": "а",
        "ё": "о",
        "е": "э",
        "о": "о" if has_accent else "а",
        "ы": "и" if has_accent else "ы",
    }
    return subst.get(ch, ch)


def is_consonant_pair(a: str, b: str):
    return a + b in consonant_pairs or b + a in consonant_pairs


def subst_consonant(ch: str, next_ch: str):
    next_sonoric = next_ch and next_ch in "мнл"
    next_voiced = next_ch and next_ch in voiced
    next_vowel = next_ch and accentuator.is_vowel(next_ch)

    if ch == "ц":
        return "тс"
    elif ch in voiceless:
        return ch
    elif ch in voiced:
        should_be_voiced = next_sonoric or next_voiced or next_vowel

        if should_be_voiced:
            return ch
        return voiceless[voiced.index(ch)]

    return ch


def to_phonetic_repr(inp: RhymeInput) -> str:
    out: str = ""

    word = inp.word.lower()

    # аго / его / ого на конце слова
    word = re.sub(r"(а|е|о)(го)($|\s)", r"\1во\3", word)

    word = re.sub(r"(ж|ш)и", r"\1ы", word)
    word = re.sub(r"(ч|щ)а", r"\1я", word)
    word = re.sub(r"(ч|щ)у", r"\1ю", word)

    word = word.replace("здн", "зн")
    word = word.replace("стн", "сн")

    word = re.sub(r"[^а-яё]|[ьъй]", "", word)

    cur_vowel = 0

    for i, ch in enumerate(word):
        next_ch = "" if i + 1 == len(word) else word[i + 1]
        vowel = accentuator.is_vowel(ch)

        if next_ch == ch and not vowel:
            continue

        has_accent = vowel and inp.accents[cur_vowel]

        cur_vowel += vowel

        out += subst_vowel(ch, has_accent) if vowel else subst_consonant(ch, next_ch)

    return out


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


def calc_rhyming_coef(rhymes: list[RhymeInput]) -> float:
    phonetic_words = list(map(to_phonetic_repr, rhymes))

    phonetic_pairs = []

    for i, word in enumerate(phonetic_words):
        if i + 1 == len(phonetic_words):
            break

        next_word = phonetic_words[i + 1]
        phonetic_pairs.append((word, next_word))

    phonetic_pairs = list(map(trim_phonetic_pair, phonetic_pairs))

    distances = map(lambda pair: levenshtein_distance(*pair), phonetic_pairs)
    weighted_distances = [
        1 - (dist / max(map(len, pair)))
        for dist, pair in zip(distances, phonetic_pairs)
    ]

    return mean(weighted_distances)
