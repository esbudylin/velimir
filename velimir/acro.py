from dataclasses import dataclass
from functools import lru_cache

from velimir.nlp import morph_analyzer

MIN_PARTIAL_LENGTH = 6

NAME_GRAMMEMES = {"Name", "Surn", "Patr"}

SINGLE_LETTER_WORDS = ("к", "в", "я", "а", "с", "и", "о", "б")

known_forms: set[str] = set()


@dataclass(frozen=True, slots=True)
class Acro:
    words: tuple[str, ...]
    forms: tuple[str, ...]
    coverage: int
    total: int

    @property
    def partial(self) -> bool:
        return self.coverage < self.total


def normalize_yo(word: str) -> str:
    return word.replace("ё", "е").replace("Ё", "Е")


@lru_cache(maxsize=4096)
def is_name(word: str) -> bool:
    word = normalize_yo(word.lower())

    return any(
        parse.is_known and parse.tag.grammemes & NAME_GRAMMEMES
        for parse in morph_analyzer.parse(word)
    )


@lru_cache(maxsize=8192)
def matched_form(word: str) -> str | None:
    word = normalize_yo(word.lower())

    if len(word) == 1:
        return word if word in SINGLE_LETTER_WORDS else None

    forms = [
        word,
        *(word[:i] + "ь" + word[i:] for i in range(1, len(word) + 1)),
        *(word[:i] + "й" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
        *(word[:i] + "ы" + word[i + 1 :] for i, c in enumerate(word) if c == "и"),
    ]

    for form in forms:
        if form in known_forms or is_name(form):
            return form

    return None


def is_word(word) -> bool:
    return matched_form(word) is not None


def load_dictionary(forms) -> None:
    known_forms.clear()
    known_forms.update(forms)
    matched_form.cache_clear()


def find_acro_variants(inp: str) -> list[list[str]]:
    if len(inp) == 0:
        raise ValueError

    unreachable = len(inp) + 1
    word_counts = [unreachable] * (len(inp) + 1)
    prev_starts: list[list[int]] = [[] for _ in range(len(inp) + 1)]
    word_counts[0] = 0

    for end in range(1, len(inp) + 1):
        best = unreachable
        starts = []

        for start in range(end):
            if word_counts[start] == unreachable:
                continue

            if not is_word(inp[start:end]):
                continue

            count = word_counts[start] + 1

            if count < best:
                best = count
                starts = [start]
            elif count == best:
                starts.append(start)

        word_counts[end] = best
        prev_starts[end] = starts

    shortest = min(MIN_PARTIAL_LENGTH, len(inp))
    ends = range(len(inp), shortest - 1, -1)

    cache: dict[int, list[tuple[str, ...]]] = {0: [()]}

    def build(pos: int) -> list[tuple[str, ...]]:
        if pos in cache:
            return cache[pos]

        variants = []

        for start in prev_starts[pos]:
            for prefix in build(start):
                variants.append((*prefix, inp[start:pos]))

        cache[pos] = variants
        return variants

    res = []

    for end in ends:
        if word_counts[end] == unreachable:
            continue

        for variant in build(end):
            res.append(list(variant))

    return res


def validate_variant(variant: list[str]) -> bool:

    def count_shorts(threshold):
        return sum(1 for word in variant if len(word) < threshold)

    if count_shorts(3) / len(variant) > 0.4:
        return False

    if count_shorts(4) / len(variant) > 0.5:
        return False

    return True


def detect_acros(inp: str) -> list[Acro]:
    variants = find_acro_variants(inp)

    grouped: dict[int, list[list[str]]] = {}

    for variant in variants:
        coverage = sum(map(len, variant))
        grouped.setdefault(coverage, []).append(variant)

    for coverage in sorted(grouped, reverse=True):
        group = grouped[coverage]
        min_words = min(map(len, group))
        group = [variant for variant in group if len(variant) == min_words]

        matches = []

        for variant in group:
            if not validate_variant(variant):
                continue

            forms = []

            for word in variant:
                form = matched_form(word)

                if form is None:
                    break

                forms.append(form)
            else:
                matches.append(
                    Acro(
                        words=tuple(variant),
                        forms=tuple(forms),
                        coverage=coverage,
                        total=len(inp),
                    )
                )

        if matches:
            return matches

    return []


def detect_acro(inp: str) -> list[str]:
    return [" ".join(acro.words) for acro in detect_acros(inp)]
