from dataclasses import dataclass, field
from enum import IntEnum

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from .domain_models import CodeIntEnum

RHYME_SCHEMA_ALPHABET = "абвгдежзийкл"


rhyme_grammar = Grammar(
    """
    expr = entry ( separator_sharp entry )*
    separator_sharp = ws* "#" ws*

    entry = chain_type / type_with_schema / schemaless_type

    schemaless_type = ( "монорим" / "вольная" / "спорадическая" / "затянутая" / "0" / "неизвестно" )

    type_with_schema = ~r"[а-я]+" separator_colon schema
    separator_colon = ws* ":" ws*

    chain_type = "цепная" separator_colon schema ws* ellipsis

    schema = schema_entry ( ws+ schema_entry )*
    schema_entry = ~r"[А-ЕХа-лхтмр]+"

    ws = ~r"\s+" 
    ellipsis = "..." / "…" / ".."
"""
)


class RhymeType(CodeIntEnum):
    CROSS = 0, "перекрестная"
    PAIRED = 1, "парная"
    ENCIRCLING = 2, "охватная"
    COMPLEX = 3, "сложная"
    FREE = 4, "вольная"
    SPORADIC = 5, "спорадическая"
    MONORHYME = 6, "монорим"
    EVEN = 7, "четная"
    ODD = 8, "нечетная"
    DELAYED = 9, "затянутая"
    SLIDING = 10, "скользящая"
    TRIPLE = 11, "тройная"
    QUADRUPLE = 12, "четверная"
    QUINTUPLE = 13, "пятерная"
    REGULAR = 14, "регулярная"
    CHAIN = 15, "цепная"
    NONE = 16, "0"
    UNKNOWN = 17, "неизвестно"


class SpecialRhymeEntry(IntEnum):
    NO_RHYME = -1
    TAUTO = -2
    MONO = -3
    REFRAIN = -4


@dataclass
class RhymeFormula:
    rhyme_type: RhymeType
    formula: list[list[int]] = field(default_factory=list)

    def to_str(self) -> str:
        if self.formula:
            formatted_schema = " ".join(map(format_rhyme_subschema, self.formula))
            return f"{self.rhyme_type.to_str()} : {formatted_schema}"

        else:
            return self.rhyme_type.to_str()


def format_rhyme_subschema(schema: list[int]) -> str:
    letters = []

    for entry in schema:
        match entry:
            case SpecialRhymeEntry.NO_RHYME:
                letters.append("х")
            case SpecialRhymeEntry.TAUTO:
                letters.append("т")
            case SpecialRhymeEntry.MONO:
                letters.append("м")
            case SpecialRhymeEntry.REFRAIN:
                letters.append("р")
            case _:
                letters.append(RHYME_SCHEMA_ALPHABET[entry])

    return "".join(letters)


def schema_letter_to_int(let: str) -> int:
    match let.lower():
        case "х":  # нет рифмы
            return SpecialRhymeEntry.NO_RHYME
        case "т":  # тавторифма
            return SpecialRhymeEntry.TAUTO
        case "м":  # монотонная рифма
            return SpecialRhymeEntry.MONO
        case "р":  # рефрен
            return SpecialRhymeEntry.REFRAIN
        case _:
            return abs(1072 - ord(let))


class RhymeVisitor(NodeVisitor):
    def visit_expr(self, _, visited_children):
        output = []
        output.append(visited_children[0])

        for child in visited_children[1]:
            _, entry = child
            output.append(entry)

        return output

    def visit_entry(self, _, visited_children):
        return visited_children[0]

    def visit_schemaless_type(self, node, _):
        return RhymeFormula(
            RhymeType.from_str(node.text),
        )

    def visit_type_with_schema(self, _, visited_children):
        rhyme_type, _, schema = visited_children
        return RhymeFormula(
            RhymeType.from_str(rhyme_type.text),
            schema,
        )

    def visit_chain_type(self, _, visited_children):
        rhyme_type, _, schema, *_ = visited_children
        return RhymeFormula(
            RhymeType.from_str(rhyme_type.text),
            schema,
        )

    def visit_schema(self, _, visited_children):
        def text_to_nums(text):
            return list(map(schema_letter_to_int, text))

        output = []

        output.append(text_to_nums(visited_children[0].text))

        for child in visited_children[1]:
            _, entry = child
            output.append(text_to_nums(entry.text))

        return output

    def generic_visit(self, node, visited_children):
        return visited_children or node
