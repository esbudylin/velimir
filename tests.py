import unittest
from dataclasses import asdict

from bitarray import bitarray
from parameterized import parameterized

from velimir.accentuator import accent_line, build_accent_dict
from velimir.identifier import ProcessedLine
from velimir.domain_models import (
    Clausula,
    Line,
    Meter,
    MeterType,
    OutputPoem,
    SyllableDistances,
)
from velimir.io import read_accent_dicts
from velimir.parsers import (
    extract_lines,
    extract_syllable_masks,
    transform_lines,
)
from velimir.settings import ACCENT_DICT_PATHS

xml_line = '<p class="verse"><line meter="Я4ж"/>Ещѐ вкруг со̀лнцев нѐ <rhyme-zone/>враща̀лись<br/>'

xml_line_with_date = """<p class="verse">
<line meter="Я4ж"/>Божѐственно̀ю нѐгой <rhyme-zone/>ды̀шит.</p>

<p class="date"><noindex>1823<br/>
Одесса</noindex></p>
"""

caesura = """<p class="verse">
<line meter="Д3м~Д3ж 0*2*2*0|0*2*2*1"/>сло̀вно скита̀льцы в века̀х, вѐрой скреплѐнные <rhyme-zone/>па̀льцы</p>
"""

multiple_caesuras = """
<p class="verse"><line meter="Дк7м 1*2*1|1*2*2|2*2*2*0"/>Велѝчество Со̀лнца велѝкие по̀прища в небеса̀х пробега̀ет легко̀,<br/></p>
"""

caesura_without_rhythm = """<p class="verse">
<line meter="Д2м~Д2м"/>ро̀сы в кровѝ, му̀зыка <rhyme-zone/>тра̀в<br/></p>
"""

multiple_caesuras_without_rhythm = """<p class="verse">
<line meter="Я2ж~Я2ж~Я2ж"/>тела̀ на ла̀пах в лохмо̀тьях ѐлок, -- о, жѝзни <rhyme-zone/>дрѐво!</p>
"""

af_caesura_without_rhythm = """
<p class="verse"><line meter="Аф2м~Аф3ж"/>Угрю̀мая тѐнь / Стано̀вится о̀тблеском свѐта.<br/></p>
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
            ("single_caesura", caesura, [7], 15),
            ("multiple_caesuras", multiple_caesuras, [6, 13], 22),
            ("without_rhythm", caesura_without_rhythm, [4], 8),
            ("multiple_without_rhythm", multiple_caesuras_without_rhythm, [5, 10], 15),
            ("af_without_rhythm", af_caesura_without_rhythm, [5], 14),
        ]
    )
    def test_parse_caesuras(self, name, xml_line, caesuras, syllable_count):
        result = list(transform_lines(xml_line))
        self.assertEqual(len(result), 1)
        line = result[0]

        self.assertListEqual(line.caesura, caesuras)
        self.assertEqual(len(line.syllable_masks.poetic_accent_mask), syllable_count)

    @parameterized.expand(
        [
            (
                "Йо̀шкин кот",
                "100",
                "011",
            ),
            (
                "куй желѐзо пока",
                "001000",
                "100101",
            ),
            (
                "в доро+гу",
                "010",
                "001",
            ),
            (
                "отправля+юсь в доро+гу",
                "0010010",
                "0001001",
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
            bitarray("010101010"),
        )

        self.assertEqual(
            line.syllable_masks.last_in_word_mask,
            bitarray("011011001"),
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
            # ("какой-нибудь", "0100"),
            # ("что-то", "10"),
            # ("какие-нибудь", "0100"),
        ]
    )
    def test_accent_word(self, word, with_accents):
        res = bitarray(accent_line(word))

        self.assertEqual(res, bitarray(with_accents))


class TestSyllableDistances(unittest.TestCase):
    # анакруса и клаузула не должны учитываться при подсчёте статистики
    @parameterized.expand(
        [
            ("010010010", [1, 2, 2, 2]),
            ("1010100", [0, 1, 1, 1]),
            ("00101001000", [2, 1, 2, 1.5]),
            ("01", [1, 0, 0, 0]),
            ("1", [0, 0, 0, 0]),
        ]
    )
    def test_syllable_distances(self, mask, expected):
        poetic_accent_mask = bitarray(mask)
        res = SyllableDistances(poetic_accent_mask).to_array()
        self.assertListEqual(res, expected)


class TestProcessedLine(unittest.TestCase):
    def setUp(self):
        self.regular_line = ProcessedLine(
            caesura=[],
            meters=[Meter(MeterType.IAMB, 4, Clausula.FEMININE)],
            poetic_accent_mask=list(bitarray("010101010")),
        )

        self.line_with_caesura = ProcessedLine(
            caesura=[7],
            meters=[
                Meter(MeterType.DACTYL, 3, Clausula.MASCULINE),
                Meter(MeterType.DACTYL, 3, Clausula.FEMININE),
            ],
            poetic_accent_mask=list(bitarray("100100110010010")),
        )

    def test_line_to_string(self):
        self.assertEqual(self.regular_line.to_str(), "Я4ж 1*1*1*1*1")

        self.assertEqual(self.line_with_caesura.to_str(), "Д3м~Д3ж 0*2*2*0|0*2*2*1")
