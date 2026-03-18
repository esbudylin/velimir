import logging
from typing import Iterator

from bs4 import BeautifulSoup
from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

import src.accentuator as accentuator
from src.accent_utils import extract_accent_mask, is_vowel, remove_accent_marks
from src.logger import delayed_logger
from src.models import Clausula, InputLine, Line, Meter, MeterType, SyllableMasks

grammar = Grammar(
    """
    expr = meter_schema ( "~" meter_schema )* ( " " rhythm_schema )?

    meter_schema = meter unstable? feet clausula

    meter = ( "Гек" / "Пен" / "Ан" / "Аф" / "Дк" / "Тк" / "Ак" / "Я" / "Х" / "Д" / "Л" / "С" )
    unstable = "*"
    feet = ~r"[0-9]+"
    clausula = ( "г" / "д" / "м" / "ж" )

    rhythm_schema = ( interval ( accent / caesura ) )+ interval

    interval = ~r"[0-9]"
    accent = "*"
    caesura = "|"
    """
)


class MeterVisitor(NodeVisitor):
    def __init__(self):
        self.meters = []
        self.caesura = []
        self.syllable_accents = []

        super().__init__()

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
        self.caesura.append(len(self.syllable_accents))

    def visit_interval(self, node, *_):
        self.syllable_accents.extend(False for _ in range(int(node.text)))

    def visit_accent(self, node, *_):
        self.syllable_accents.append(True)

    def generic_visit(self, node, visited_children):
        return visited_children or node

    def collect_data(self):
        # TODO: стоит ли определять положение цезуры для строк, в
        # которых не был размечен ритм?
        # Это возможно сделать исходя из схемы метра

        return dict(
            meters=self.meters,
            caesura=self.caesura,
            syllables=self.syllable_accents,
        )


def transform_lines(xml_str: str) -> Iterator[Line]:
    return parse_lines(extract_lines(xml_str))


def parse_line_meter(meter: str) -> dict:
    try:
        tree = grammar.parse(meter)

    except ParseError as e:
        delayed_logger.record()

        if isinstance(e, IncompleteParseError):
            logging.warning("Can't fully parse the line meter: %s", meter)
            tree = grammar.match(meter)

        else:
            logging.error("Can't parse the line meter: %s Continuing...", meter)
            return {}

    vis = MeterVisitor()
    vis.visit(tree)

    return vis.collect_data()


def extract_word_ending_mask(text: str) -> list[bool]:
    result = []

    for word in text.split():
        word_vowels = list(filter(lambda c: is_vowel(c), word))

        if word_vowels:
            result += [False] * (len(word_vowels) - 1)
            result.append(True)

    return result


def extract_syllable_masks(
    line: str,
    rhythm_accents: list[bool] = None,
) -> SyllableMasks:
    poetic_accent_mask = extract_accent_mask(line)

    if not poetic_accent_mask:
        # Ударения не размечены
        # Используем разметку ритма вместо них
        poetic_accent_mask = rhythm_accents or []

    cleaned_line = remove_accent_marks(line)

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


def parse_line(line_text: str, parsed_meter: dict) -> Line:
    meters = [Meter(**meter) for meter in parsed_meter["meters"]]

    syllable_masks = extract_syllable_masks(line_text, parsed_meter["syllables"])

    return Line(
        meters=meters,
        syllable_masks=syllable_masks,
        caesura=parsed_meter["caesura"],
    )


def extract_lines(xml: str) -> Iterator[InputLine]:
    soup = BeautifulSoup(xml, "xml")

    for line in soup.find_all("line"):
        if meter := line.get("meter"):
            text = collect_line_text(line)

            yield InputLine(text=text, meter=meter.strip())


def parse_lines(lines: Iterator[InputLine]) -> Iterator[Line]:
    for line in lines:
        if parsed_meter := parse_line_meter(line.meter):
            try:
                yield parse_line(line.text, parsed_meter)
            except Exception:
                delayed_logger.record()
                logging.error(
                    "Error while processing line: %s",
                    line,
                )
