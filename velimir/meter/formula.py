import logging
from dataclasses import dataclass
from functools import cache

from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from ..domain_models import Clausula, Meter, MeterType
from ..logger import delayed_logger

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
