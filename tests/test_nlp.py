import unittest
from unittest.mock import MagicMock

from velimir.nlp import (
    DependencyRelation,
    GrammarFeatures,
    PartOfSpeech,
    initialize,
    markup,
)


def _make_word(start_char, text, upos, deprel):
    """Helper to create a mock stanza word object."""
    w = MagicMock()
    w.start_char = start_char
    w.text = text
    w.upos = upos
    w.deprel = deprel
    return w


def _make_doc(words_by_sentence):
    """Create a mock stanza doc from nested lists of words."""
    doc = MagicMock()
    sentences = []
    for words in words_by_sentence:
        sent = MagicMock()
        sent.words = words
        sentences.append(sent)
    doc.sentences = sentences
    return doc


class TestLanguageMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = initialize()

    def test_basic_functionality(self):
        result = markup(self.nlp, ["пришла весна"])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], GrammarFeatures)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(result[0].part_of_speech[0], PartOfSpeech.VERB)
        self.assertEqual(result[0].part_of_speech[1], PartOfSpeech.NOUN)
        self.assertEqual(result[0].dep_rels[0], DependencyRelation.ROOT)
        self.assertEqual(result[0].dep_rels[1], DependencyRelation.NSUBJ)

    def test_multi_line(self):
        result = markup(self.nlp, ["пришла весна, и", "снег растаял"])

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [p.code for p in result[0].part_of_speech],
            ["VERB", "NOUN", "CCONJ"],
        )
        self.assertEqual(
            [d.code for d in result[0].dep_rels],
            ["root", "nsubj", "cc"],
        )
        self.assertEqual(
            [p.code for p in result[1].part_of_speech],
            ["NOUN", "VERB"],
        )
        self.assertEqual(
            [d.code for d in result[1].dep_rels],
            ["nsubj", "conj"],
        )

    def test_lemmatization_word_split_mismatch(self):
        nlp = MagicMock()
        # Text: "кое-как пришли"
        # Space-delimited words: "кое-как" (pos 0-6), "пришли" (pos 8-13)
        # Stanza tokens: "кое" (start=0), "как" (start=4), "пришли" (start=8)
        doc = _make_doc(
            [
                [
                    _make_word(0, "кое", "ADV", "advmod"),
                    # not at a space word boundary → should be skipped
                    _make_word(4, "как", "X", "advmod"),
                    _make_word(8, "пришли", "VERB", "root"),
                ],
            ]
        )
        nlp.return_value = doc

        result = markup(nlp, ["кое-как пришли"])

        self.assertEqual(len(result), 1)
        # Only "кое" and "пришли" should remain; "как" is skipped
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(
            [p.code if p else None for p in result[0].part_of_speech],
            ["ADV", "VERB"],
        )

    def test_unknown(self):
        nlp = MagicMock()
        doc = _make_doc([[_make_word(0, "что-то", "UNKNOWN_POS", "UNKNOWN_DEPREL")]])
        nlp.return_value = doc

        result = markup(nlp, ["что-то"])

        self.assertEqual(result[0].part_of_speech[0], PartOfSpeech.X)
        self.assertEqual(result[0].dep_rels[0], DependencyRelation.UNKNOWN)

    def test_words_without_vowels_are_skipped(self):
        result = markup(self.nlp, ["в дом"])

        self.assertEqual(len(result[0].part_of_speech), 1)
        self.assertEqual(result[0].part_of_speech[0], PartOfSpeech.NOUN)
        self.assertEqual(result[0].dep_rels[0], DependencyRelation.ROOT)

    def test_leading_punctuation(self):
        """Stanza splits «Куда into « (PUNCT) and Куда (ADV). « has no vowels
        so it is skipped by vowel filter; Куда should be kept as the first
        content token for the whitespace word «Куда."""
        result = markup(self.nlp, ["«Куда мир"])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(
            [p.code if p else None for p in result[0].part_of_speech],
            ["ADV", "NOUN"],
        )

    def test_trailing_punctuation(self):
        """Stanza splits рекла: into рекла (VERB) and : (PUNCT, no vowels).
        : is skipped by vowel filter; рекла is kept for the whitespace word."""
        result = markup(self.nlp, ["Сказала: слово"])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(
            [p.code if p else None for p in result[0].part_of_speech],
            ["VERB", "NOUN"],
        )

    def test_multiple_sentences(self):
        result = markup(self.nlp, ["Пришла весна. Снег растаял."])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 4)
        self.assertEqual(
            [p.code for p in result[0].part_of_speech],
            ["VERB", "NOUN", "NOUN", "VERB"],
        )
