"""
MIT License

Copyright (c) 2021 yuliya1324

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from src.accent_utils import extract_neuro_accents, vowel_count, is_vowel


@dataclass
class AccentEntry:
    endings: set[str]
    capitalized: bool
    accents: list[int]
    secondary_accents: list[int]
    yo: list[int]
    no_accent: bool


accent_dict: defaultdict[str, list[AccentEntry]] = defaultdict(list)


def parse_dict_entry(word: str, accent: str) -> tuple[str, AccentEntry]:
    base = re.sub(r"\(.+$", "", word)

    endings_match = re.findall(r"\(((.*|)+)\)", word)
    if endings_match:
        endings = endings_match[0][0].split("|")
    else:
        endings = [""]

    accents = []
    secondary_accents = []
    yos = []
    no_accent = False

    accent_p = accent.removesuffix("!")
    secondary_accent_mark = "`"
    yo_mark = '"'

    for accent_entry in re.split("[,;]", accent_p):
        if not accent_entry:
            continue

        pos, symbol = re.findall(r"(\d+)(.*)", accent_entry)[0]
        if int(pos) == 0:
            no_accent = True
        elif symbol == secondary_accent_mark:
            secondary_accents.append(int(pos))
        if symbol == yo_mark:
            yos.append(int(pos))
        else:
            accents.append(int(pos))

    entry = AccentEntry(
        endings=set(endings),
        capitalized=accent.endswith("!"),
        accents=accents,
        secondary_accents=secondary_accents,
        yo=yos,
        no_accent=no_accent,
    )

    return base, entry


def build_accent_dict(rows: Iterator[str]):
    for row in rows:
        if row.startswith("#"):
            continue

        if split := row.split():
            word, accent = split
            base, entry = parse_dict_entry(word, accent)
            accent_dict[base].append(entry)


def accent_line(line: str) -> list[bool]:
    words_nacc = extract_neuro_accents(line)

    line_stripped = re.sub(r"[^А-яЁё\s-]+", "", line)
    words = list(filter(vowel_count, line_stripped.split()))

    if len(words) != len(words_nacc):
        raise ValueError(
            f"""Number of words with vowels ({len(words)})
            does not match number of neuro-accented words ({len(words_nacc)})"""
        )

    res = []

    for j, word in enumerate(words):
        if is_word_without_accent(word):
            res += base_mask(word)
            continue

        accent_entry = find_accent_entry(word)

        if not accent_entry or should_use_neuro_accent(word, accent_entry):
            res += words_nacc[j]
        else:
            res += accent_word_by_dict(word, accent_entry)

    return res


def accent_word_by_dict(word: str, accent_entry: AccentEntry) -> list[bool]:
    mask = base_mask(word)

    all_accents = (
        accent_entry.accents + accent_entry.secondary_accents + accent_entry.yo
    )

    for accent in all_accents:
        accent_pos = min(len(mask) - 1, accent - 1)
        mask[accent_pos] = True

    return mask


def should_use_neuro_accent(word: str, accent: AccentEntry) -> bool:
    """
    Слово берется из строки, размеченной нейросетевым
    акцентуатором, если

    1) словарный акцентуатор поставил в этом слове ударение в двух
    местах, или не поставил совсем и при этом в нем нет буквы ё,

    2) словарный акцентуатор поставил и ударение, и
    букву ё (кроме слов через дефис: например, тёмно-си'ний )
    """
    if accent.no_accent:
        return False

    if len(accent.accents) > 1 or len(accent.secondary_accents) > 1:
        return True

    if accent.accents and accent.yo:
        return "-" not in word

    return False


def is_word_without_accent(word: str) -> bool:
    """
    Слово остается без ударения, если
    1) в нем нет гласных,
    2) оно односложное (оно содержит одну гласную)
    """
    vowels = vowel_count(word)

    return vowels < 2


def normalize(s):
    if re.match("^[А-Я]", s):
        caps = True
    else:
        caps = False
    return s.lower(), caps


def base_mask(word):
    return [False for c in word if is_vowel(c)]


def find_accent_entry(word: str) -> AccentEntry | None:
    key, capitalized = normalize(word)
    found_entry = None

    for i in range(len(key), -1, -1):
        entries = accent_dict[key[0:i]]
        ending = key[i:]

        for entry in entries:
            if ending in entry.endings:
                if not capitalized and entry.capitalized:
                    continue
                found_entry = entry
                break
        if found_entry:
            break

    if not found_entry:
        return None

    return found_entry
