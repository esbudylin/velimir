import unittest

from bs4 import BeautifulSoup
from parameterized import parameterized

from src.models import Line, Meter, Clausula, MeterType, OutputPoem
from src.parsers import collect_line_text, parse_lines
from src.accentuator import accent_line


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

xml_line_with_caesura = """
<line meter="Д3м~Д3ж 0*2*2*0|0*2*2*1"/>сло̀вно скита̀льцы в века̀х, вѐрой скреплѐнные <rhyme-zone/>па̀льцы</p
"""

xml_line_with_multiple_caesuras = """
<p class="verse"><line meter="Дк7м 1*2*1|1*2*2|2*2*2*0"/>Велѝчество Со̀лнца велѝкие по̀прища в небеса̀х пробега̀ет легко̀,<br/>
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

    @parameterized.expand(
        [
            (xml_line_with_caesura, [7], 15),
            (xml_line_with_multiple_caesuras, [6, 13], 22),
        ]
    )
    def test_parse_caesuras(self, xml_line, caesuras, syllable_count):
        result = list(parse_lines(xml_line))
        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertEqual(len(line.syllable_masks.poetic_accent_mask), syllable_count)
        self.assertListEqual(line.caesura, caesuras)

    def test_parse_line_with_rhythm(self):
        result = list(parse_lines(xml_line_with_rhythm))
        self.assertEqual(len(result), 1)
        line = result[0]

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
        self.assertEqual(meter.meter, MeterType.IAMB)
        self.assertEqual(meter.feet, 4)
        self.assertEqual(meter.clausula, Clausula.FEMININE)
        self.assertFalse(meter.unstable)

        self.assertListEqual(line.caesura, [])

        # Маски
        self.assertListEqual(
            line.syllable_masks.poetic_accent_mask,
            [False, True, False, True, False, True, False, True, False],
        )

        self.assertLessEqual(
            line.syllable_masks.last_in_word_mask,
            [False, True, True, False, True, True, False, False, True],
        )

        self.assertEqual(
            len(line.syllable_masks.poetic_accent_mask),
            len(line.syllable_masks.linguistic_accent_mask),
        )


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.xml_path = "data/rnc/texts/xix/1830/1830-001.xml"

    def test_data_round_trip(self):
        with open(self.xml_path, "r", encoding="utf8") as f:
            xml = f.read()

        poem = OutputPoem(path=self.xml_path, lines=list(parse_lines(xml)))

        encoded = poem.encode()
        decoded = OutputPoem.decode(encoded)

        self.assertDictEqual(poem.model_dump(), decoded.model_dump())


class TestAccentuator(unittest.TestCase):
    def test_accent_line(self):
        line = "Еще вкруг солнцев не вращались"

        res = accent_line(line)

        self.assertEqual(res, "Ещё вкруг со'лнцев не враща'лись")
