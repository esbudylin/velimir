import unittest
from unittest.mock import MagicMock, patch

from velimir.nlp import (
    DependencyRelation,
    GrammarFeatures,
    PartOfSpeech,
    initialize,
    markup,
    markup_stanzas,
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


class TestMarkup(unittest.TestCase):
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


class TestMarkupStanzas(unittest.TestCase):
    def _make_features(self, line_count):
        """Return a list of GrammarFeatures for testing."""
        return [
            GrammarFeatures(
                part_of_speech=[PartOfSpeech.NOUN],
                dep_rels=[DependencyRelation.ROOT],
            )
            for _ in range(line_count)
        ]

    @patch("velimir.nlp.markup")
    def test_small_stanzas_joined(self, mock_markup):
        """Stanzas that fit within 32 lines should be processed together."""
        mock_markup.side_effect = lambda nlp, lines: self._make_features(len(lines))

        verses = [
            ["line a", "line b"],  # 2 lines
            ["line c", "line d"],  # 2 lines
            ["line e"],  # 1 line
        ]

        nlp = MagicMock()
        result = markup_stanzas(nlp, verses)

        mock_markup.assert_called_once()
        self.assertEqual(len(result), 5)

    @patch("velimir.nlp.markup")
    def test_stanzas_split_by_limit(self, mock_markup):
        """Stanzas that together exceed 32 lines should be split into groups."""
        mock_markup.side_effect = lambda nlp, lines: self._make_features(len(lines))

        verses = [
            ["line"] * 20,  # 20 lines
            ["line"] * 20,  # 20 lines — together 40 > 32, triggers split
        ]

        nlp = MagicMock()
        result = markup_stanzas(nlp, verses)

        self.assertEqual(mock_markup.call_count, 2)
        self.assertEqual(len(result), 40)

    @patch("velimir.nlp.markup")
    def test_oversized_stanza_split(self, mock_markup):
        """A single stanza exceeding 32 lines should be split into chunks."""
        mock_markup.side_effect = lambda nlp, lines: self._make_features(len(lines))

        verses = [["line"] * 70]  # 70 lines — needs 3 chunks

        nlp = MagicMock()
        result = markup_stanzas(nlp, verses)

        self.assertEqual(mock_markup.call_count, 3)
        self.assertEqual(len(result), 70)

        # Verify chunk sizes: 32, 32, 6
        calls = mock_markup.call_args_list
        self.assertEqual(len(calls[0][0][1]), 32)
        self.assertEqual(len(calls[1][0][1]), 32)
        self.assertEqual(len(calls[2][0][1]), 6)

    @patch("velimir.nlp.markup")
    def test_oversized_stanza_with_pending_group(self, mock_markup):
        """Oversized stanza triggers flush of pending group first."""
        mock_markup.side_effect = lambda nlp, lines: self._make_features(len(lines))

        verses = [
            ["line"] * 10,  # fits, goes into group
            ["line"] * 100,  # oversized, triggers flush + split
        ]

        nlp = MagicMock()
        result = markup_stanzas(nlp, verses)

        self.assertEqual(len(result), 110)

        calls = mock_markup.call_args_list
        self.assertEqual(len(calls), 5)
        # First call: the pending 10-line group
        self.assertEqual(len(calls[0][0][1]), 10)
        # Next 4 calls: 32 + 32 + 32 + 4 = 100
        self.assertEqual(len(calls[1][0][1]), 32)
        self.assertEqual(len(calls[2][0][1]), 32)
        self.assertEqual(len(calls[3][0][1]), 32)
        self.assertEqual(len(calls[4][0][1]), 4)
