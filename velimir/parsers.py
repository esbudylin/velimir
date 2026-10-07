import logging
from fractions import Fraction
from itertools import count
from typing import Iterable, Iterator

from bs4 import BeautifulSoup, NavigableString, Tag

from . import cyrlat
from .accentuation import accent_probabilities
from .domain_models import (
    InputLine,
    Line,
    SyllableFeatures,
)
from .logger import delayed_logger
from .meter.formula import LineFormula, parse_line_formula
from .settings import STRESS_MARK_ORD


def parse_input_lines(
    xml: str,
    *,
    allow_latin: bool = False,
) -> tuple[list[InputLine], list[int]]:
    soup = BeautifulSoup(xml, "xml")

    line_count = count()
    lines = []
    stanza_breaks = []

    for verse in soup.find_all("p", class_="verse"):
        if stanza := list(
            extract_lines(verse, line_count=line_count, allow_latin=allow_latin)
        ):
            stanza_breaks.append(len(lines))
            lines.extend(stanza)
        else:
            delayed_logger.record()
            logging.warning("Skipping empty stanza")

    return lines, stanza_breaks


def transform_poem(xml: str) -> dict:
    input_lines, stanza_breaks = parse_input_lines(xml)

    lines = list(parse_lines(input_lines))

    return dict(
        lines=lines,
        stanza_breaks=stanza_breaks,
    )


def clean_line(s: str) -> str:
    # non-breaking spaces
    s = s.replace("\xa0", " ")

    # tabs
    s = s.replace("\t", " ")

    return s


def extract_syllable_features(
    line: str,
    rhythm_accents: list[bool] = None,
) -> SyllableFeatures:
    poetic_accents = extract_accent_mask(line)
    rhythm_accents = rhythm_accents or []

    if not sum(poetic_accents) and sum(rhythm_accents):
        delayed_logger.record()
        logging.warning("Accents are not marked. Using line formula rhythm instead")
        poetic_accents = rhythm_accents

    cleaned_line = clean_line(remove_accent_marks(line))
    probabilities = accent_probabilities(cleaned_line)

    return SyllableFeatures(
        poetic_accents=poetic_accents,
        last_in_word=extract_word_ending_mask(cleaned_line),
        accent_probabilities=probabilities,
    )


def collect_line_text(line) -> tuple[str, str]:
    parts = []
    rhyme_zone = -1

    stack = [iter(line.next_siblings)]

    while stack:
        try:
            node = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue

        if isinstance(node, Tag) and node.name == "line":
            break

        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag):
            if node.name == "rhyme-zone":
                rhyme_zone = len(parts)

            if node.find("line") or node.find("rhyme-zone"):
                stack.append(iter(node.contents))
            else:
                parts.append(node.get_text())

    full_line_text = "".join(parts).strip()
    rhyme_zone_text = "" if rhyme_zone == -1 else "".join(parts[rhyme_zone:]).strip()

    return full_line_text, rhyme_zone_text


def parse_line(line: InputLine, line_formula: LineFormula) -> Line:
    syllable_features = extract_syllable_features(
        line.text,
        line_formula.rhythm_accents,
    )
    caesura = extract_caesura(
        line_formula,
        syllable_features.poetic_accents,
    )

    return Line(
        idx=line.idx,
        meters=line_formula.meters,
        syllables=syllable_features,
        caesura=caesura,
    )


def extract_lines(
    soup,
    *,
    line_count: Iterator[int] | None = None,
    allow_latin: bool = False,
) -> Iterator[InputLine]:
    for line, idx in zip(soup.find_all("line"), line_count or count()):
        if meter := line.get("meter"):
            text, rhyme_zone = collect_line_text(line)

            if not text:
                delayed_logger.record()
                logging.error("Cannot collect text from line %s", line)
                continue

            match cyrlat.detect(text):
                case cyrlat.DetectionResult.LATIN:
                    if not allow_latin:
                        delayed_logger.record()
                        logging.warning(
                            "Skipping line (latin script detected) %s",
                            text,
                        )
                        continue
                case cyrlat.DetectionResult.CYRLAT:
                    text = cyrlat.fix(text)
                    rhyme_zone = cyrlat.fix(rhyme_zone)

            yield InputLine(
                idx=idx,
                text=text,
                meter=meter.strip(),
                rhyme_zone=rhyme_zone.strip(),
            )


def parse_lines(lines: Iterable[InputLine]) -> Iterator[Line]:
    for line in lines:
        if line_formula := parse_line_formula(line.meter):
            try:
                yield parse_line(line, line_formula)
            except Exception as e:
                delayed_logger.record()
                logging.error(
                    "Error while processing line: %s, %s",
                    line,
                    str(e),
                )


def extract_caesura(
    formula: LineFormula,
    poetic_accents: list[bool],
) -> list[Fraction]:
    if formula.caesura:
        feet = sum(poetic_accents)
        return [Fraction(c, feet) for c in formula.caesura]

    # Определяем положение цезуры для строк, в
    # которых не был размечен ритм, исходя из схемы метра
    if len(formula.meters) > 1 and not formula.caesura:
        feet = sum(meter.feet for meter in formula.meters)
        feet_acc = 0
        caesura = []

        for meter in formula.meters[:-1]:
            feet_acc += meter.feet
            caesura.append(Fraction(feet_acc, feet))

        return caesura

    return []


def is_vowel(char):
    vowels = "аеиоуыэюяёАЕИОУЫЭЮЯЁ"

    return char in vowels


def vowel_count(word):
    return sum(map(is_vowel, word))


def extract_accent_mask(text: str) -> list[bool]:
    result = []

    def is_accent_mark(char):
        return char and ord(char) == STRESS_MARK_ORD

    for i, char in enumerate(text):
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if is_vowel(char):
            if is_accent_mark(next_char):
                result.append(True)
            else:
                result.append(False)

    return result


def extract_word_ending_mask(text: str) -> list[bool]:
    result = []

    for word in text.split():
        word_vowels = list(filter(lambda c: is_vowel(c), word))

        if word_vowels:
            result += [False] * (len(word_vowels) - 1)
            result.append(True)

    return result


def remove_accent_marks(text: str) -> str:
    return "".join(c for c in text if ord(c) != STRESS_MARK_ORD)
