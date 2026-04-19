import logging
import unicodedata
from enum import IntEnum

from . import accentuator
from .logger import delayed_logger


class DetectionResult(IntEnum):
    LATIN = 0
    CYR = 1
    CYRLAT = 2
    UNKNOWN = 3


STRESS = chr(accentuator.stress_mark_ord)

BASE_LATIN_TO_CYR = {
    "A": "А",
    "a": "а",
    "B": "В",
    "C": "С",
    "c": "с",
    "E": "Е",
    "e": "е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "T": "Т",
    "X": "Х",
    "x": "х",
    "y": "у",
}


def _build_translation_map() -> dict[str, str]:
    extended = dict(BASE_LATIN_TO_CYR)

    for latin_char, cyr_char in BASE_LATIN_TO_CYR.items():
        # Only process lowercase/uppercase letters
        if not latin_char.isalpha():
            continue

        for codepoint in range(0x00C0, 0x0180):  # Latin-1 + Extended-A
            ch = chr(codepoint)

            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue

            # Only Latin letters
            if "LATIN" not in name:
                continue

            # Decompose (e.g., è → e + ̀)
            base, *accent = unicodedata.normalize("NFD", ch)

            if base == latin_char and str(accent) == STRESS:
                # Rebuild Cyrillic + accent
                extended[ch] = cyr_char + STRESS

    return extended


TRANS_MAP = _build_translation_map()
TRANSLATOR = str.maketrans(TRANS_MAP)
LOOKALIKES = frozenset(TRANS_MAP.keys())


def _is_cyrillic(ch: str) -> bool:
    try:
        return "CYRILLIC" in unicodedata.name(ch)
    except ValueError:
        return False


def _is_latin(ch: str) -> bool:
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def detect(text: str) -> DetectionResult:
    has_cyr = False
    has_latin = False

    for ch in text:
        if _is_cyrillic(ch):
            has_cyr = True
        elif _is_latin(ch):
            has_latin = True

            if ch not in LOOKALIKES:
                return DetectionResult.LATIN

    if has_cyr and has_latin:
        return DetectionResult.CYRLAT
    elif has_cyr:
        return DetectionResult.CYR
    elif has_latin:
        return DetectionResult.LATIN
    else:
        return DetectionResult.UNKNOWN


def fix(text: str) -> str:
    delayed_logger.record()
    logging.info("Rebuilding cyrlat string %s", text)

    res = text.translate(TRANSLATOR)

    logging.info("String rebuilt: %s", res)

    return res
