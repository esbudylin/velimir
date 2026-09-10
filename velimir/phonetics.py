import re

from velimir.accentuator import is_vowel


def subst_vowel(ch: str, has_accent: bool):
    subst = {
        "ю": "йу",
        "я": "йа",
        "ё": "йо",
        "о": "о" if has_accent else "а",
    }
    return subst.get(ch, ch)


def subst_consonant(ch: str, next_ch: str):
    consonant_pairs = ["пб", "сз", "тд", "фв", "кг"]

    next_sonoric = next_ch in "мнл"

    for pair in consonant_pairs:
        voiceless, voiced = pair

        if ch == voiceless:
            return voiceless
        elif ch == voiced:
            if next_ch and (next_sonoric or is_vowel(next_ch)):
                return voiced
            else:
                return voiceless

    return ch


def to_phonetic_repr(inp: str, accents: list[bool]) -> str:
    out: str = ""

    inp = inp.lower()
    inp = re.sub(r"(е|о)(го)($|\s)", r"\1во\3", inp)
    inp = re.sub(r"[^а-яё]|[ьъ]", "", inp)

    cur_vowel = 0

    for i, ch in enumerate(inp):
        next_ch = "" if i + 1 == len(inp) else inp[i + 1]

        if next_ch == ch:
            continue

        vowel = is_vowel(ch)
        has_accent = vowel and accents[cur_vowel]

        cur_vowel += vowel

        out += subst_vowel(ch, has_accent) if vowel else subst_consonant(ch, next_ch)

    return out
