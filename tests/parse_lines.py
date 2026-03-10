import unittest

from bs4 import BeautifulSoup
from parameterized import parameterized

from src.models import Line, Meter, Clausula
from src.parsers import collect_line_text, parse_lines


xml_line = '<p class="verse"><line meter="Я4ж"/>Ещѐ вкруг со̀лнцев нѐ <rhyme-zone/>враща̀лись<br/>'

xml_line_with_date = """<p class="verse">
<line meter="Я4ж"/>Божѐственно̀ю нѐгой <rhyme-zone/>ды̀шит.</p>

<p class="date"><noindex>1823<br/>
Одесса</noindex></p>
"""

# ритм в поле метра должен иметь приоритет над разметкой ударений
xml_line_with_rhythm = """
<p class="verse"><line meter="Дк3м 2*4*0"/>Но̀гу на̀ ногу <rhyme-zone/>заложѝв<br/>
"""


class TestParseLine(unittest.TestCase):
    @parameterized.expand(
        [
            (xml_line, "Ещѐ вкруг со̀лнцев нѐ враща̀лись"),
            (xml_line_with_date, "Божѐственно̀ю нѐгой ды̀шит."),
        ]
    )
    def test_collect_text_from_line(self, xml_line, text):
        soup = BeautifulSoup(xml_line, "xml")
        line = soup.find("line")
        self.assertEqual(collect_line_text(line), text)

    def test_parse_line_with_rhythm(self):
        result = list(parse_lines(xml_line_with_rhythm))
        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertIsInstance(line, Line)

        self.assertListEqual(
            line.syllable_masks.poetic_accent_mask,
            [False, False, True, False, False, False, False, True],
        )

    def test_parse_line_with_meter(self):
        result = list(parse_lines(xml_line))

        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertIsInstance(line, Line)

        # Метр
        self.assertEqual(len(line.meters), 1)
        meter = line.meters[0]
        self.assertIsInstance(meter, Meter)
        self.assertEqual(meter.meter, "Я")
        self.assertEqual(meter.feet, 4)
        self.assertEqual(meter.clausula, Clausula.FEMININE)
        self.assertFalse(meter.unstable)

        self.assertEqual(line.caesura, -1)

        # Маски
        self.assertListEqual(
            line.syllable_masks.poetic_accent_mask,
            [False, True, False, True, False, True, False, True, False],
        )

        self.assertLessEqual(
            line.syllable_masks.last_in_word_mask,
            [False, True, True, False, True, True, False, False, True],
        )
