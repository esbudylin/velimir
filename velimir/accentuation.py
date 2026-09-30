from functools import cache

from stressful import Accentuator


@cache
def get_accentuator() -> Accentuator:
    return Accentuator()


def accent_line(line: str) -> list[bool]:
    return get_accentuator().accentuate(line)
