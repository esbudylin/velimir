import itertools
from bisect import bisect_right
from dataclasses import dataclass

import stanza

from .accentuator import vowel_count
from .domain_models import CodeIntEnum


class PartOfSpeech(CodeIntEnum):
    ADJ = 1, "ADJ"  # adjective
    ADP = 2, "ADP"  # adposition
    ADV = 3, "ADV"  # adverb
    AUX = 4, "AUX"  # auxiliary
    CCONJ = 5, "CCONJ"  # coordinating conjunction
    DET = 6, "DET"  # determiner
    INTJ = 7, "INTJ"  # interjection
    NOUN = 8, "NOUN"  # noun
    NUM = 9, "NUM"  # numeral
    PART = 10, "PART"  # particle
    PRON = 11, "PRON"  # pronoun
    PROPN = 12, "PROPN"  # proper noun
    SCONJ = 13, "SCONJ"  # subordinating conjunction
    VERB = 14, "VERB"  # verb
    X = 0, "X"  # other


class DependencyRelation(CodeIntEnum):
    # Core arguments
    NSUBJ = 1, "nsubj"
    OBJ = 2, "obj"
    IOBJ = 3, "iobj"
    CSUBJ = 4, "csubj"
    CCOMP = 5, "ccomp"
    XCOMP = 6, "xcomp"

    # Non-core dependents
    OBL = 7, "obl"
    VOCATIVE = 8, "vocative"
    EXPL = 9, "expl"
    ADVCL = 10, "advcl"
    ADVMOD = 11, "advmod"
    DISCOURSE = 12, "discourse"
    AUX = 13, "aux"
    COP = 14, "cop"
    MARK = 15, "mark"

    # Nominal dependents
    NMOD = 16, "nmod"
    APPOS = 17, "appos"
    NUMMOD = 18, "nummod"
    ACL = 19, "acl"
    AMOD = 20, "amod"
    DET = 21, "det"
    CASE = 22, "case"

    # Coordination
    CONJ = 23, "conj"
    CC = 24, "cc"

    # Multiword expressions
    FIXED = 25, "fixed"
    FLAT = 26, "flat"
    COMPOUND = 27, "compound"

    # Loose / special
    LIST = 28, "list"
    PARATAXIS = 29, "parataxis"
    ORPHAN = 30, "orphan"

    # Root of the sentence
    ROOT = 31, "root"

    # Fallback
    UNKNOWN = 0, "_"


@dataclass(slots=True)
class GrammarFeatures:
    """
    грамматические характеристики для каждого слова в строке,
    включающего один или более гласный звук
    """

    part_of_speech: list[PartOfSpeech]
    dep_rels: list[DependencyRelation]


def initialize():
    stanza.download(
        lang="ru",
        download_json=False,
    )

    return stanza.Pipeline(
        lang="ru",
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    )


def from_str_safe(enum, s):
    try:
        return enum.from_str(s)
    except ValueError:
        return None


def markup(nlp, lines: list[str]) -> list[GrammarFeatures]:
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    pos_tags: list[list[PartOfSpeech]] = [[] for _ in lines]
    dep_rels: list[list[DependencyRelation]] = [[] for _ in lines]

    joined_lines = " ".join(lines)
    text = " ".join(joined_lines.split())  # normalize spaces

    word_ends = list(
        itertools.chain.from_iterable(
            [[False] * len(word) + [True] for word in text.split()]
        )
    )

    doc = nlp(text)

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.start_char is None:
                continue

            if not vowel_count(word.text):
                continue

            # лемматизация stanza может отличать от деления на слова на основе пробелов.
            # игнорируем слова, которые не выделяются в velimir/parsers
            if word.start_char and not word_ends[word.start_char - 1]:
                continue

            line_idx = bisect_right(line_starts, word.start_char) - 1

            pos = from_str_safe(PartOfSpeech, word.upos)
            dep_rel = from_str_safe(DependencyRelation, word.deprel)

            pos_tags[line_idx].append(pos or PartOfSpeech.X)
            dep_rels[line_idx].append(dep_rel or DependencyRelation.UNKNOWN)

    return [
        GrammarFeatures(part_of_speech=pos_tags[i], dep_rels=dep_rels[i])
        for i in range(len(lines))
    ]
