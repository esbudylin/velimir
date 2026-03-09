import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from src.models import Line, Meter
from src.parsers import collect_line_text, parse_lines


class TestParseLine(unittest.TestCase):
    def setUp(self):
        self.xml_line = '<p class="verse"><line meter="Я4ж"/>Ещѐ вкруг со̀лнцев нѐ <rhyme-zone/>враща̀лись<br/>'

    def test_collect_text_from_line(self):
        soup = BeautifulSoup(self.xml_line, "xml")
        line = soup.find("line")
        self.assertEqual(collect_line_text(line), "Ещѐ вкруг со̀лнцев нѐ враща̀лись")

    @patch("src.parsers.accentuate")
    def test_parse_line_with_meter(self, mock_accent_line):
        mock_accent_line.return_value = "Еще' вкруг со'лнцев не враща'лись"

        result = list(parse_lines(self.xml_line))

        mock_accent_line.assert_called_once_with("Еще вкруг солнцев не вращались")

        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertIsInstance(line, Line)

        # Метр
        self.assertEqual(len(line.meters), 1)
        meter = line.meters[0]
        self.assertIsInstance(meter, Meter)
        self.assertEqual(meter.meter, "Я")
        self.assertEqual(meter.feet, 4)
        self.assertEqual(meter.clausula, "ж")
        self.assertFalse(meter.unstable)

        self.assertEqual(line.caesura, -1)

        # Маски
        self.assertListEqual(
            line.syllable_masks.linguistic_accent_mask,
            [False, True, False, True, False, False, False, True, False],
        )

        self.assertListEqual(
            line.syllable_masks.poetic_accent_mask,
            [False, True, False, True, False, True, False, True, False],
        )

        self.assertLessEqual(
            line.syllable_masks.last_in_word_mask,
            [False, True, True, False, True, True, False, False, True],
        )


if __name__ == "__main__":
    unittest.main()
