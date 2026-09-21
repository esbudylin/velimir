import re
from dataclasses import dataclass

from velimir import accentuator

voiced = ["б", "з", "д", "в", "г", "ж"]
voiceless = ["п", "с", "т", "ф", "к", "ш"]
consonant_pairs = [a + b for a, b in zip(voiced, voiceless)]

PHONETIC_VOCAB = {
    "а": 1,
    "б": 2,
    "в": 3,
    "г": 4,
    "д": 5,
    "ж": 6,
    "з": 7,
    "и": 8,
    "й": 9,
    "к": 10,
    "л": 11,
    "м": 12,
    "н": 13,
    "о": 14,
    "п": 15,
    "р": 16,
    "с": 17,
    "т": 18,
    "у": 19,
    "ф": 20,
    "х": 21,
    "ч": 22,
    "ш": 23,
    "щ": 24,
    "ы": 25,
    "э": 26,
}


@dataclass
class PhoneticRepr:
    phonetics: str
    accents: list[bool]


def subst_vowel(ch: str, prev_ch: str, has_accent: bool):
    composed_subst = {
        "ю": "у",
        "я": "а",
        "ё": "о",
        "е": "э",
    }

    if prev_ch and accentuator.is_vowel(prev_ch):  # TODO: начало слов
        composed_subst = {k: "й" + v for k, v in composed_subst.items()}

    subst = {
        **composed_subst,
        "о": "о" if has_accent else "а",
    }
    return subst.get(ch, ch)


def is_consonant_pair(a: str, b: str):
    return a + b in consonant_pairs or b + a in consonant_pairs


def subst_consonant(ch: str, next_ch: str):
    next_sonoric = next_ch and next_ch in "мнлр"
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


def to_phonetic_repr(word: str, accents) -> PhoneticRepr:
    res = PhoneticRepr("", [])

    word = word.lower()

    # аго / его / ого на конце слова
    word = re.sub(r"(а|е|о)(го)($|\s)", r"\1во\3", word)

    word = re.sub(r"(ж|ш)и", r"\1ы", word)

    word = word.replace("здн", "зн")
    word = word.replace("стн", "сн")

    word = re.sub(r"[^а-яё]|[ьъ]", "", word)

    cur_vowel = 0

    for i, ch in enumerate(word):
        next_ch = "" if i + 1 == len(word) else word[i + 1]
        prev_ch = "" if i == 0 else word[i - 1]
        vowel = accentuator.is_vowel(ch)

        if next_ch == ch and not vowel:
            continue

        has_accent = vowel and accents[cur_vowel]

        cur_vowel += vowel

        substituted = (
            subst_vowel(ch, prev_ch, has_accent)
            if vowel
            else subst_consonant(ch, next_ch)
        )

        res.phonetics += substituted
        res.accents.extend(accentuator.is_vowel(c) and has_accent for c in substituted)

    if not all(map(lambda n: n in PHONETIC_VOCAB, res.phonetics)):
        raise ValueError("Invalid encdoing: %s" % res.phonetics)

    return res
