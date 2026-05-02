import logging
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Iterator

from bs4 import BeautifulSoup
from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from . import accentuator, cyrlat
from .domain_models import Clausula, InputLine, Line, Meter, MeterType, SyllableMasks
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


def transform_poem(xml: str) -> dict:
    soup = BeautifulSoup(xml, "xml")

    lines = []
    stanza_breaks = []

    for verse in soup.find_all("p", class_="verse"):
        if stanza := list(parse_lines(extract_lines(verse))):
            stanza_breaks.append(len(lines))
            lines.extend(stanza)
        else:
            delayed_logger.record()
            logging.warning("Skipping empty stanza")

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


def extract_word_ending_mask(text: str) -> list[bool]:
    result = []

    for word in text.split():
        word_vowels = list(filter(lambda c: accentuator.is_vowel(c), word))

        if word_vowels:
            result += [False] * (len(word_vowels) - 1)
            result.append(True)

    return result


def clean_line(s: str) -> str:
    # non-breaking spaces
    s = s.replace("\xa0", " ")

    # tabs
    s = s.replace("\t", " ")

    return s


def extract_syllable_masks(
    line: str,
    rhythm_accents: list[bool] = None,
) -> SyllableMasks:
    poetic_accent_mask = accentuator.extract_accent_mask(line)

    if not poetic_accent_mask:
        # Ударения не размечены
        # Используем разметку ритма вместо них
        poetic_accent_mask = rhythm_accents or []

    cleaned_line = clean_line(accentuator.remove_accent_marks(line))

    return SyllableMasks(
        linguistic_accent_mask=accentuator.accent_line(cleaned_line),
        poetic_accent_mask=poetic_accent_mask,
        last_in_word_mask=extract_word_ending_mask(cleaned_line),
    )


def collect_line_text(line_tag) -> str:
    parts = []
    for node in line_tag.next_siblings:
        if node.name == "line":
            break
        parts.append(node.get_text())

    return "".join(parts).strip()


def parse_line(line_text: str, line_formula: LineFormula) -> Line:
    syllable_masks = extract_syllable_masks(line_text, line_formula.rhythm_accents)
    caesura = extract_caesura(line_formula, syllable_masks.poetic_accent_mask)

    return Line(
        meters=line_formula.meters,
        syllable_masks=syllable_masks,
        caesura=caesura,
    )


def extract_lines(soup) -> Iterator[InputLine]:
    for line in soup.find_all("line"):
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

            yield InputLine(text=text, meter=meter.strip())


def parse_lines(lines: Iterator[InputLine]) -> Iterator[Line]:
    for line in lines:
        if line_formula := parse_line_formula(line.meter):
            try:
                yield parse_line(line.text, line_formula)
            except Exception as e:
                delayed_logger.record()
                logging.error(
                    "Error while processing line: %s, %s",
                    line,
                    str(e),
                )


def extract_caesura(
    formula: LineFormula,
    poetic_accent_mask: list[bool],
) -> list[Fraction]:
    if formula.caesura:
        feet = sum(poetic_accent_mask)
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
