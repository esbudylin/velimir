import unittest

from velimir.nlp import (
    GrammarFeatures,
    PartOfSpeech,
    markup,
)


class TestMarkup(unittest.TestCase):
    def test_basic_functionality(self):
        result = markup(["пришла весна"])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], GrammarFeatures)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(result[0].part_of_speech[0], PartOfSpeech.VERB)
        self.assertEqual(result[0].part_of_speech[1], PartOfSpeech.NOUN)

    def test_multi_line(self):
        result = markup(["пришла весна, и", "снег растаял"])

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [p.code for p in result[0].part_of_speech],
            ["VERB", "NOUN", "CONJ"],
        )
        self.assertEqual(
            [p.code for p in result[1].part_of_speech],
            ["NOUN", "VERB"],
        )

    def test_words_without_vowels_are_skipped(self):
        result = markup(["в дом"])

        self.assertEqual(len(result[0].part_of_speech), 1)
        self.assertEqual(result[0].part_of_speech[0], PartOfSpeech.NOUN)

    def test_leading_punctuation(self):
        result = markup(["«Куда мир"])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(
            [p.code if p else None for p in result[0].part_of_speech],
            ["CONJ", "NOUN"],
        )

    def test_trailing_punctuation(self):
        result = markup(["Сказала: слово"])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 2)
        self.assertEqual(
            [p.code if p else None for p in result[0].part_of_speech],
            ["VERB", "NOUN"],
        )

    def test_multiple_sentences(self):
        result = markup(["Пришла весна. Снег растаял."])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].part_of_speech), 4)
        self.assertEqual(
            [p.code for p in result[0].part_of_speech],
            ["VERB", "NOUN", "NOUN", "VERB"],
        )
