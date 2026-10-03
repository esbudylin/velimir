from functools import cache

from stressful import Accentuator


@cache
def get_accentuator() -> Accentuator:
    return Accentuator()


def accent_probabilities(line: str) -> list[float]:
    accentuation = get_accentuator().accentuate_detailed(line)

    return [probability for word in accentuation for probability in word.probabilities]
