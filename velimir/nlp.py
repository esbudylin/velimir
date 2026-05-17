import re
from dataclasses import dataclass
from functools import cache, partial

from pymorphy2 import MorphAnalyzer

from .accentuator import vowel_count
from .domain_models import CodeIntEnum

morph_analyzer = MorphAnalyzer()

CYRILLIC_EDGE_RE = re.compile(r"^[^А-Яа-яЁё]+|[^А-Яа-яЁё]+$")


class PartOfSpeech(CodeIntEnum):
    UNKNOWN = (0, "UNKNOWN")
    NOUN = (1, "NOUN")  # имя существительное
    ADJF = (2, "ADJF")  # имя прилагательное (полное)
    ADJS = (3, "ADJS")  # имя прилагательное (краткое)
    COMP = (4, "COMP")  # компаратив
    VERB = (5, "VERB")  # глагол (личная форма)
    INFN = (6, "INFN")  # глагол (инфинитив)
    PRTF = (7, "PRTF")  # причастие (полное)
    PRTS = (8, "PRTS")  # причастие (краткое)
    GRND = (9, "GRND")  # деепричастие
    NUMR = (10, "NUMR")  # числительное
    ADVB = (11, "ADVB")  # наречие
    NPRO = (12, "NPRO")  # местоимение-существительное
    PRED = (13, "PRED")  # предикатив
    PREP = (14, "PREP")  # предлог
    CONJ = (15, "CONJ")  # союз
    PRCL = (16, "PRCL")  # частица
    INTJ = (17, "INTJ")  # междометие


@dataclass(slots=True)
class GrammarFeatures:
    """
    грамматические характеристики для каждого слова в строке,
    включающего один или более гласный звук
    """

    part_of_speech: list[PartOfSpeech]

    def expand(self, last_in_word: list[bool]):
        if sum(last_in_word) != len(self.part_of_speech):
            raise ValueError(
                "Mismatch between grammar features length and number of words in last_in_word mask"
            )

        current_word = 0

        expanded_pos = []

        for is_end in last_in_word:
            expanded_pos.append(self.part_of_speech[current_word])

            if is_end:
                current_word += 1

        return GrammarFeatures(expanded_pos)

    def encode(self):
        return self.part_of_speech

    @classmethod
    def decode(cls, data):
        return cls(part_of_speech=data)


def from_str_safe(enum, s):
    try:
        return enum.from_str(s)
    except ValueError:
        return None


def extract_words_for_morph(line: str):
    clean_word = partial(CYRILLIC_EDGE_RE.sub, "")
    return [clean_word(w) for w in line.split() if vowel_count(w)]


@cache
def extract_part_of_speech(word: str) -> PartOfSpeech:
    pos = sorted(morph_analyzer.parse(word), key=lambda t: -t.score)[0].tag.POS
    return from_str_safe(PartOfSpeech, str(pos)) or PartOfSpeech.UNKNOWN


def markup(line: str) -> GrammarFeatures:
    pos = map(extract_part_of_speech, extract_words_for_morph(line))
    return GrammarFeatures(part_of_speech=list(pos))
