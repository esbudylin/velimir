from enum import IntEnum

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


rhyme_grammar = Grammar(
    """
    expr = entry ( separator_sharp entry )*
    separator_sharp = ws* "#" ws*

    entry = schemaless_type / type_with_schema

    schemaless_type = ( "монорим" / "вольная" / "спорадическая" / "0" )

    type_with_schema = ~r"[а-я]+" separator_colon schema
    separator_colon = ws* ":" ws*

    schema = schema_entry ( ws+ schema_entry )*
    schema_entry = ~r"[А-ГХа-кхтмр]+" 

    ws = ~r"\s+" 
"""
)


class SpecialRhymeEntry(IntEnum):
    NO_RHYME = -1
    TAUTO = -2
    MONO = -3
    REFRAIN = -4


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
        output = {}

        for child in visited_children:
            output.update(child)

        return output

    def visit_schemaless_type(self, node, _):
        return {"type": node.text}

    def visit_type_with_schema(self, _, visited_children):
        type, _, schema = visited_children
        return {
            "type": type.text,
            "schema": schema,
        }

    def visit_schema(self, _, visited_children):
        def text_to_nums(text):
            return map(schema_letter_to_int, text)

        output = []

        output.extend(text_to_nums(visited_children[0].text))

        for child in visited_children[1]:
            _, entry = child
            output.extend(text_to_nums(entry.text))

        return output

    def generic_visit(self, node, visited_children):
        return visited_children or node
