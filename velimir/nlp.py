import itertools
import logging
import re
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

    def __post_init__(self):
        if len(self.part_of_speech) != len(self.dep_rels):
            raise ValueError("part_of_speech and dep_rels must be of same length")

    def expand(self, last_in_word: list[bool]):
        if sum(last_in_word) != len(self.part_of_speech):
            raise ValueError(
                "Mismatch between grammar features length and number of words in last_in_word mask"
            )

        current_word = 0

        expanded_pos = []
        expanded_deprels = []

        for is_end in last_in_word:
            expanded_pos.append(self.part_of_speech[current_word])
            expanded_deprels.append(self.dep_rels[current_word])

            if is_end:
                current_word += 1

        return GrammarFeatures(expanded_pos, expanded_deprels)


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


def markup_stanzas(
    nlp,
    verses: list[list[str]],
) -> list[GrammarFeatures]:
    MAX_LINES_PER_GROUP = 32

    result = []
    group_lines: list[str] = []

    for stanza_lines in verses:
        if group_lines and len(group_lines) + len(stanza_lines) > MAX_LINES_PER_GROUP:
            result.extend(markup(nlp, group_lines))
            group_lines = []

        group_lines.extend(stanza_lines)

    if group_lines:
        result.extend(markup(nlp, group_lines))

    return result


def markup(nlp, lines: list[str]) -> list[GrammarFeatures]:
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    pos_tags: list[list[PartOfSpeech]] = [[] for _ in lines]
    dep_rels: list[list[DependencyRelation]] = [[] for _ in lines]

    # убираем прописные буквы в начале строк,
    # чтобы уменьшить число ложнопозитивных определений имён собственных
    lines = [line[0].lower() + line[1:] for line in lines]

    joined_lines = " ".join(lines)
    text = " ".join(joined_lines.split())  # normalize spaces

    # разделяем текст на слова на основе пробелов
    # оставляем слова с гласными звуками
    ws_word_ranges = [
        (m.start(), m.end())
        for m in re.finditer(r"\S+", text)
        if vowel_count(m.group())
    ]

    doc = nlp(text)

    nlp_words = itertools.chain.from_iterable(
        sentence.words for sentence in doc.sentences
    )

    # Лемматизация stanza может отличаться от деления на слова на
    # основе пробелов. Для случаев, когда stanza выделяет несколько
    # лемм для одного слова, используем данные только первой леммы
    for word_start, word_end in ws_word_ranges:
        found_word = None
        for word in nlp_words:
            if not vowel_count(word.text):
                continue

            if word_start <= word.start_char < word_end:
                found_word = word
                break

        if not found_word:
            logging.error("Can't find matching nlp word")
            continue

        line_idx = bisect_right(line_starts, found_word.start_char) - 1

        # use only the base for composed forms
        dep_rel_str = found_word.deprel.split(":")[0]

        pos = from_str_safe(PartOfSpeech, found_word.upos)
        dep_rel = from_str_safe(DependencyRelation, dep_rel_str)

        pos_tags[line_idx].append(pos or PartOfSpeech.X)
        dep_rels[line_idx].append(dep_rel or DependencyRelation.UNKNOWN)

    return [
        GrammarFeatures(part_of_speech=pos_tags[i], dep_rels=dep_rels[i])
        for i in range(len(lines))
    ]
