import unittest
from dataclasses import asdict

from bitarray import bitarray
from parameterized import parameterized

from src.accentuator import accent_line, build_accent_dict, accent_word
from src.accent_utils import extract_neuro_accents
from src.io import read_accent_dicts
from src.models import Clausula, Line, Meter, MeterType, OutputPoem
from src.parsers import (
    extract_lines,
    extract_syllable_masks,
    transform_lines,
)
from src.settings import ACCENT_DICT_PATHS

xml_line = '<p class="verse"><line meter="Я4ж"/>Ещѐ вкруг со̀лнцев нѐ <rhyme-zone/>враща̀лись<br/>'

xml_line_with_date = """<p class="verse">
<line meter="Я4ж"/>Божѐственно̀ю нѐгой <rhyme-zone/>ды̀шит.</p>

<p class="date"><noindex>1823<br/>
Одесса</noindex></p>
"""

xml_line_with_caesura = """<p class="verse">
<line meter="Д3м~Д3ж 0*2*2*0|0*2*2*1"/>сло̀вно скита̀льцы в века̀х, вѐрой скреплѐнные <rhyme-zone/>па̀льцы</p>
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
        extracted = next(extract_lines(xml_line))
        self.assertEqual(extracted.text, text)

    @parameterized.expand(
        [
            ("single_caesura", xml_line_with_caesura, [7], 15),
            ("multiple_caesuras", xml_line_with_multiple_caesuras, [6, 13], 22),
        ]
    )
    def test_parse_caesuras(self, name, xml_line, caesuras, syllable_count):
        result = list(transform_lines(xml_line))
        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertEqual(len(line.syllable_masks.poetic_accent_mask), syllable_count)
        self.assertListEqual(line.caesura, caesuras)

    @parameterized.expand(
        [
            (
                "Йо̀шкин кот",
                [True, False, False],
                [False, True, True],
            ),
            (
                "куй желѐзо пока",
                [False, False, True, False, False, False],
                [True, False, False, True, False, True],
            ),
            (
                "в доро+гу",
                [False, True, False],
                [False, False, True],
            ),
            (
                "отправля+юсь в доро+гу",
                [False, False, True, False, False, True, False],
                [False, False, False, True, False, False, True],
            ),
        ]
    )
    def test_mask_extraction(self, input, accent_mask, last_in_word_mask):
        masks = extract_syllable_masks(input)
        self.assertEqual(masks.poetic_accent_mask, bitarray(accent_mask))
        self.assertEqual(masks.last_in_word_mask, bitarray(last_in_word_mask))

    def test_parse_line_with_meter(self):
        result = list(transform_lines(xml_line))

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
        self.assertEqual(
            line.syllable_masks.poetic_accent_mask,
            bitarray([False, True, False, True, False, True, False, True, False]),
        )

        self.assertEqual(
            line.syllable_masks.last_in_word_mask,
            bitarray([False, True, True, False, True, True, False, False, True]),
        )


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.xml_path = "data/rnc/texts/xix/1830/1830-001.xml"

    def test_data_round_trip(self):
        with open(self.xml_path, "r", encoding="utf8") as f:
            xml = f.read()

        poem = OutputPoem(path=self.xml_path, lines=list(transform_lines(xml)))

        encoded = poem.encode()
        decoded = OutputPoem.decode(encoded)

        self.assertDictEqual(asdict(poem), asdict(decoded))


class TestAccentuator(unittest.TestCase):
    def setUpClass():
        build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    @parameterized.expand(
        [
            ("Еще вкруг солнцев не вращались", "010100010"),
            ("Ваше Величество, мы прибыли ко дворцу", "1001000100001"),
            ("йошкин кот", "100"),
        ]
    )
    def test_accent_line(self, line, with_accents):
        res = bitarray(accent_line(line))

        self.assertEqual(res, bitarray(with_accents))

    @parameterized.expand(
        [
            ("легкий", "10"),
            ("темно-синий", "1010"),
            ("ёлка", "10"),
            ("еще", "01"),
            ("Еще", "01"),
            ("какой-нибудь", "0101"),
        ]
    )
    def test_accent_word(self, word, with_accents):
        res = bitarray(accent_word(word, extract_neuro_accents(word)[0]))

        self.assertEqual(res, bitarray(with_accents))
