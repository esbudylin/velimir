import logging
import unicodedata
from typing import Iterator, List

from bs4 import BeautifulSoup
from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from stressrnn import StressRNN

from src.logger import delayed_logger
from src.models import InputPoem, Line, Meter, OutputPoem, SyllableMasks

vowels = "аеиоуыэюяёАЕИОУЫЭЮЯЁ"

stress_rnn = StressRNN()
accent_mark = "+"

grammar = Grammar(
    """
    expr = meter_schema ( "~" meter_schema )* ( " " rhythm_schema )?

    meter_schema = meter unstable? feet clausula

    meter = ~r"[А-Яа-я]{1,3}"
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
        self.caesura = -1
        self.syllable_accents = []

        super().__init__()

    def visit_meter(self, node, *_):
        self._current_meter = {}
        self._current_meter["meter"] = node.text
        self.meters.append(self._current_meter)

    def visit_feet(self, node, *_):
        self._current_meter["feet"] = int(node.text)

    def visit_clausula(self, node, *_):
        self._current_meter["clausula"] = node.text

    def visit_unstable(self, *_):
        self._current_meter["unstable"] = True

    def visit_caesura(self, *_):
        if self.caesura != -1:
            delayed_logger.record()
            logging.warning("Several caesura breaks are found in a single line")

            return

        self.caesura = len(self.syllable_accents)

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


def transform_poem(poem: InputPoem, xml_str: str) -> OutputPoem:
    return OutputPoem(
        path=poem.path,
        lines=list(parse_lines(xml_str)),
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

    vis = MeterVisitor()
    vis.visit(tree)

    return vis.collect_data()


def remove_accent_marks(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


def extract_accent_mask(text: str) -> List[bool]:
    stress_mark_ord = 768

    result = []

    def accent_test(char):
        return ord(char) in [stress_mark_ord, ord(accent_mark)]

    for i, char in enumerate(text):
        if char in vowels:
            if i + 1 < len(text) and accent_test(text[i + 1]):
                result.append(True)
            else:
                result.append(False)

    return result


def extract_word_ending_mask(text: str) -> List[bool]:
    result = []

    for word in text.split():
        word_vowels = list(filter(lambda c: c in vowels, word))
        result += [False] * (len(word_vowels) - 1)
        result.append(True)

    return result


def extract_syllable_masks(poetic_accent_mask: List[bool], line: str) -> SyllableMasks:
    if not poetic_accent_mask:
        # в корпусе не размечен ритм строки
        # собираем эти данные из размеченных поэтических ударений
        poetic_accent_mask = extract_accent_mask(line)

    cleaned_line = remove_accent_marks(line)

    line_with_linguistic_accents = stress_rnn.put_stress(
        cleaned_line,
        accent_mark,
        use_batch_mode=True,
    )

    return SyllableMasks(
        linguistic_accent_mask=extract_accent_mask(line_with_linguistic_accents),
        poetic_accent_mask=poetic_accent_mask,
        last_in_word_mask=extract_word_ending_mask(cleaned_line),
    )


def collect_line_text(line_tag):
    parts = []
    for node in line_tag.next_siblings:
        if node.name == "line":
            break
        parts.append(node.get_text())

    return "".join(parts).strip()


def parse_lines(xml: str) -> Iterator[Line]:
    soup = BeautifulSoup(xml, "xml")

    for line in soup.find_all("line"):
        if meter := line.get("meter"):
            if parsed := parse_line_meter(meter.strip()):
                meters = [Meter(**meter) for meter in parsed["meters"]]

                line_text = collect_line_text(line)
                syllable_masks = extract_syllable_masks(parsed["syllables"], line_text)

                yield Line(
                    meters=meters,
                    syllable_masks=syllable_masks,
                    caesura=parsed["caesura"],
                )
