import logging
from typing import Iterator

from bs4 import BeautifulSoup
from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from . import accentuator
from .logger import delayed_logger
from .domain_models import (
    Clausula,
    InputLine,
    Line,
    Meter,
    MeterType,
    SyllableMasks,
)

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


class MeterVisitor(NodeVisitor):
    def __init__(self):
        self.meters = []
        self.caesura = []
        self.syllable_accents = []

        super().__init__()

    def visit_expr(self, node, visited_children):
        # Определяем положение цезуры для строк, в
        # которых не был размечен ритм, исходя из схемы метра
        if len(self.meters) > 1 and not self.caesura:
            try:
                self.caesura = extract_caesura(self.meters)
            except ValueError as e:
                # Невозможно разметить цезуру из-за нерегулярного метра (например, дольника)
                delayed_logger.record()
                logging.error(e)

        return dict(
            meters=self.meters,
            caesura=self.caesura,
            syllables=self.syllable_accents,
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
        self.caesura.append(len(self.syllable_accents))

    def visit_interval(self, node, *_):
        self.syllable_accents.extend(False for _ in range(int(node.text)))

    def visit_accent(self, node, *_):
        self.syllable_accents.append(True)

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

    return MeterVisitor().visit(tree)


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


def parse_line(line_text: str, parsed_meter: dict) -> Line:
    meters = [Meter(**meter) for meter in parsed_meter["meters"]]

    syllable_masks = extract_syllable_masks(line_text, parsed_meter["syllables"])

    return Line(
        meters=meters,
        syllable_masks=syllable_masks,
        caesura=parsed_meter["caesura"],
    )


def extract_lines(soup) -> Iterator[InputLine]:
    for line in soup.find_all("line"):
        if meter := line.get("meter"):
            text = collect_line_text(line)

            if not text:
                delayed_logger.record()
                logging.error(f"Cannot collect text from line {line}")
                continue

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


def match_foot_syllables_from_meter(meter: MeterType) -> int:
    match meter:
        case MeterType.IAMB | MeterType.TROCHEE:
            return 2
        case MeterType.ANAPEST | MeterType.AMPHIBRACH | MeterType.DACTYL:
            return 3
        case _:
            raise ValueError(f"Can't match feet syllables from meter: {meter}")


def stress_position_in_foot(meter: MeterType) -> int:
    match meter:
        case MeterType.TROCHEE | MeterType.DACTYL:
            return 0
        case MeterType.IAMB | MeterType.AMPHIBRACH:
            return 1
        case MeterType.ANAPEST:
            return 2
        case _:
            raise ValueError(f"Unsupported meter for stress offset: {meter}")


def find_final_foot_size(meter: MeterType, clausula: Clausula) -> int:
    stress_pos = stress_position_in_foot(meter)

    return stress_pos + 1 + clausula


def extract_caesura(meters: list[dict]) -> list[int]:
    meter_pairs = list(zip(meters, meters[1:]))
    last_caesura_position = 0

    res = []

    for meter, _ in meter_pairs:
        feet = meter["feet"]

        foot_syllables = match_foot_syllables_from_meter(meter["meter"])
        final_foot_size = find_final_foot_size(meter["meter"], meter["clausula"])

        last_caesura_position += foot_syllables * (feet - 1) + final_foot_size

        res.append(last_caesura_position)

    return res
