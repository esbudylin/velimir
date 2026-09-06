import logging

from parsimonious import IncompleteParseError, ParseError
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

author_grammar = Grammar(
    """
    expr = author ( separator author )*

    separator = ws? ":" ws?

    author = full_name ( ws secondary_name )?

    full_name = surname_with_initials / ( name ( ws name )* )

    surname_with_initials = initial ws ( initial ws )? name
    initial = ~r"[А-ЯЁ][а-яё]?" "."

    secondary_name = "(" full_name ")"

    name = ( cyr_word ( "-" cyr_word )? ) / ( lat_word ( "-" lat_word )? )
    cyr_word = ~r"[А-ЯЁ]?[а-яё]+"
    lat_word = ~r"[A-Z]?[a-z]+"
    
    ws = ~r"\s+" 
    """
)


# В качестве ключа сортировки либо возвращаем фамилию, либо последнее имя
# Если авторов несколько, возвращаем ключ для первого автора в строке
class AuthorVisitor(NodeVisitor):
    def __init__(self):
        self.surname = ""
        self.visited_names = []

    def visit_expr(self, node, visited_children):
        return self.surname or self.visited_names[-1]

    def visit_surname_with_initials(self, node, visited_children):
        self.surname = self.surname or visited_children[-1].text
        return node

    def visit_name(self, node, visited_children):
        self.visited_names.append(node.text)
        return node

    def generic_visit(self, node, visited_children):
        return node


def get_author_sort_key(author: str):
    try:
        tree = author_grammar.parse(author)

    except ParseError as e:
        if isinstance(e, IncompleteParseError):
            logging.warning("Can't fully parse the author: %s", author)
            tree = author_grammar.match(author)
        else:
            logging.warn("Can't parse author name: %s", author)
            return author

    return AuthorVisitor().visit(tree)
