import logging
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import count
from typing import Iterable, Iterator

from bs4 import BeautifulSoup, NavigableString, Tag
from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from . import accentuator, cyrlat
from .domain_models import (
    Clausula,
    InputLine,
    Line,
    Meter,
    MeterType,
    SyllableFeatures,
)
from .logger import delayed_logger

grammar = Grammar(
    """
    expr = meter_schema ( "~" meter_schema )* ( ws rhythm_schema )?

    meter_schema = meter unstable? feet clausula

    meter = ( "Гек" / "Пен" / "Ан" / "Аф" / "Дк" / "Тк" / "Ак" / "Я" / "Х" / "Д" / "Л" / "С" )
    unstable = "*"
    feet = ~r"[0-9]+"
    clausula = ( "г" / "д" / "м" / "ж" )

    rhythm_schema = ( interval ( accent / caesura ) )+ interval

    interval = ~r"[0-9]"
    accent = "*"
    caesura = "|"

    ws = ~r"\s+" 
    """
)


@dataclass(slots=True)
class LineFormula:
    meters: list[Meter]
    # абсолютные позиции ударных слогов, после которых располагается цезура
    caesura: list[int]
    rhythm_accents: list[bool]


class LineFormulaVisitor(NodeVisitor):
    def __init__(self):
        self.meters = []
        self.caesura = []
        self.rhythm_accents = []

        super().__init__()

    def visit_expr(self, node, visited_children):
        return LineFormula(
            meters=[Meter(**meter) for meter in self.meters],
            caesura=self.caesura,
            rhythm_accents=self.rhythm_accents,
        )

    def visit_meter(self, node, *_):
        self._current_meter = {}

        self._current_meter["meter"] = MeterType.from_str(node.text)

        self.meters.append(self._current_meter)

    def visit_feet(self, node, *_):
        self._current_meter["feet"] = int(node.text)

    def visit_clausula(self, node, *_):
        self._current_meter["clausula"] = Clausula.from_str(node.text)

    def visit_unstable(self, *_):
        self._current_meter["unstable"] = True

    def visit_caesura(self, *_):
        self.caesura.append(sum(self.rhythm_accents))

    def visit_interval(self, node, *_):
        self.rhythm_accents.extend(False for _ in range(int(node.text)))

    def visit_accent(self, node, *_):
        self.rhythm_accents.append(True)

    def generic_visit(self, node, visited_children):
        return visited_children or node


def parse_input_lines(xml: str) -> tuple[list[InputLine], list[int]]:
    soup = BeautifulSoup(xml, "xml")

    line_count = count()
    lines = []
    stanza_breaks = []

    for verse in soup.find_all("p", class_="verse"):
        if stanza := list(extract_lines(verse, line_count)):
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


@cache
def parse_line_formula(formula: str) -> LineFormula | None:
    try:
        tree = grammar.parse(formula)

    except ParseError as e:
        delayed_logger.record()

        if isinstance(e, IncompleteParseError):
            logging.warning("Can't fully parse the line meter: %s", formula)
            tree = grammar.match(formula)

        else:
            logging.error("Can't parse the line meter: %s Continuing...", formula)
            return None

    return LineFormulaVisitor().visit(tree)


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
    poetic_accents = accentuator.extract_accent_mask(line)
    rhythm_accents = rhythm_accents or []

    if not sum(poetic_accents) and sum(rhythm_accents):
        delayed_logger.record()
        logging.warning("Accents are not marked. Using line formula rhythm instead")
        poetic_accents = rhythm_accents

    cleaned_line = clean_line(accentuator.remove_accent_marks(line))

    return SyllableFeatures(
        poetic_accents=poetic_accents,
        last_in_word=accentuator.extract_word_ending_mask(cleaned_line),
        linguistic_accents=accentuator.accent_line(cleaned_line),
    )


def extract_text_until_next_line(element) -> str:
    parts = []
    for child in element.contents:
        if isinstance(child, Tag) and child.name == "line":
            break
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(extract_text_until_next_line(child))
    return "".join(parts)


def collect_line_text(line_tag) -> str:
    parts = []
    for node in line_tag.next_siblings:
        if isinstance(node, Tag) and node.name == "line":
            break

        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag):
            if node.find("line") is not None:
                parts.append(extract_text_until_next_line(node))
                break
            else:
                parts.append(node.get_text())

    return "".join(parts).strip()


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


def extract_lines(soup, line_count: Iterator[int] | None = None) -> Iterator[InputLine]:
    for line, idx in zip(soup.find_all("line"), line_count or count()):
        if meter := line.get("meter"):
            text = collect_line_text(line)

            if not text:
                delayed_logger.record()
                logging.error("Cannot collect text from line %s", line)
                continue

            match cyrlat.detect(text):
                case cyrlat.DetectionResult.LATIN:
                    delayed_logger.record()
                    logging.warning("Skipping line (latin script detected) %s", text)
                    continue
                case cyrlat.DetectionResult.CYRLAT:
                    text = cyrlat.fix(text)

            yield InputLine(idx=idx, text=text, meter=meter.strip())


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
