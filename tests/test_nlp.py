import unittest

from velimir.nlp import (
    GrammarFeatures,
    PartOfSpeech,
    markup,
)


class TestMarkup(unittest.TestCase):
    def test_basic_functionality(self):
        result = markup("пришла весна")

        self.assertIsInstance(result, GrammarFeatures)
        self.assertEqual(len(result.part_of_speech), 2)
        self.assertEqual(result.part_of_speech[0], PartOfSpeech.VERB)
        self.assertEqual(result.part_of_speech[1], PartOfSpeech.NOUN)

    def test_words_without_vowels_are_skipped(self):
        result = markup("в дом")

        self.assertEqual(len(result.part_of_speech), 1)
        self.assertEqual(result.part_of_speech[0], PartOfSpeech.NOUN)

    def test_leading_punctuation(self):
        result = markup("«Куда мир")

        self.assertEqual(len(result.part_of_speech), 2)
        self.assertEqual(
            result.part_of_speech,
            [PartOfSpeech.CONJ, PartOfSpeech.NOUN],
        )

    def test_trailing_punctuation(self):
        result = markup("Сказала: слово")

        self.assertEqual(len(result.part_of_speech), 2)
        self.assertEqual(
            result.part_of_speech,
            [PartOfSpeech.VERB, PartOfSpeech.NOUN],
        )

    def test_multiple_sentences(self):
        result = markup("Пришла весна. Снег растаял.")

        self.assertEqual(len(result.part_of_speech), 4)
        self.assertEqual(
            result.part_of_speech,
            [
                PartOfSpeech.VERB,
                PartOfSpeech.NOUN,
                PartOfSpeech.NOUN,
                PartOfSpeech.VERB,
            ],
        )
