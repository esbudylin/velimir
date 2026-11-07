import csv
import unittest
from unittest.mock import patch

from parameterized import parameterized

from velimir.acro import WordInfo, detect_acro, load_dictionary
from velimir.settings import ACRO_DICTIONARY_CACHE_PATH


def setUpModule():
    with open(ACRO_DICTIONARY_CACHE_PATH, encoding="utf8", newline="") as f:
        forms = {
            row["form"]: WordInfo(int(row["frequency"]), row["is_name"] == "1")
            for row in csv.DictReader(f)
        }

    load_dictionary(forms)


CASES = [
    ("тюрма", ["тюрьма"]),
    ("анна", ["анна"]),
    ("норе", ["норе"]),
    ("дусе", ["дусе"]),
    ("дружба", ["дружба"]),
    ("россия", ["россия"]),
    ("димековалёву", ["диме", "ковалеву"]),
    ("игорюсеверянину", ["игорю", "северянину"]),
    ("аннаахматова", ["анна", "ахматова"]),
    ("машиналбом", ["машин", "альбом"]),
    ("книжнаялавкаписателеи", ["книжная", "лавка", "писателей"]),
    ("юрииверховскии", ["юрии", "верховский"]),
    ("цехкушаетяства", ["цех", "кушает", "яства"]),
    ("колцов", ["кольцов"]),
    ("эллис", ["эллис"]),
    ("наталия", ["наталия"]),
    ("валериюбрюсову", ["валерию", "брюсову"]),
    ("михаилукузмину", ["михаилу", "кузмину"]),
    ("объясниипрости", ["объясни", "и", "прости"]),
    ("левъмей", ["лев", "мей"]),
]


class TestDetectAcro(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_cleaned_acrostic(self, sequence, expected_forms):
        with patch("velimir.acro.calculate_sequence_statistics", return_value={}):
            acro = detect_acro(sequence)

        self.assertIsNotNone(acro)
        self.assertEqual(acro.sequence, sequence)
        self.assertListEqual(acro.forms, expected_forms)
        self.assertFalse(acro.partial)

    def test_partial(self):
        # xviii/hemnitser/hemn-133
        sequence = "лниколаилвовивб"

        with patch("velimir.acro.calculate_sequence_statistics", return_value={}):
            acro = detect_acro(sequence)

        self.assertIsNotNone(acro)
        self.assertEqual(acro.sequence, "николаилвов")
        self.assertListEqual(acro.forms, ["николаи", "львов"])
        self.assertTrue(acro.partial)

    def test_partial_whitman(self):
        # xx/ssoloviev/ssolov-217
        sequence = "уитменичоитсниа"

        with patch("velimir.acro.calculate_sequence_statistics", return_value={}):
            acro = detect_acro(sequence)

        self.assertIsNotNone(acro)
        self.assertEqual(acro.sequence, "уитмен")
        self.assertListEqual(acro.forms, ["уитмен"])
        self.assertTrue(acro.partial)
