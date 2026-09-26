import unittest

from bs4 import BeautifulSoup
from velimir.accentuator import (
    extract_accent_mask,
    stress_mark_ord,
)
from velimir.markup import rhyme_zone
from velimir.parsers import extract_lines

xml = """
<p class="verse"><line meter="Ан4м"/>И, садя̀сь, запева̀ли <i>Варя̀га</i> <rhyme-zone/>однѝ,<br/>
<line meter="Ан3м"/>А другѝе -- не в ла̀д -- <i><rhyme-zone/>Ермака̀</i>,<br/>
<line meter="Ан4м"/>И крича̀ли <i>ура̀</i>, и шутѝли <rhyme-zone/>онѝ,<br/>
<line meter="Ан3м"/>И тихо̀нько крестѝлась <rhyme-zone/>рука̀.</p>
"""

MARK = chr(stress_mark_ord)


def get_zone(line: str):
    return rhyme_zone(line, extract_accent_mask(line))


class TestRhymeZoneParsing(unittest.TestCase):
    def test_rhyme_zone_extraction(self):
        soup = BeautifulSoup(xml, "xml")

        expected_rhyme_zones = ["однѝ,", "Ермака̀,", "онѝ,", "рука̀."]
        extracted = list(extract_lines(soup))

        self.assertEqual(len(expected_rhyme_zones), len(extracted))
        for i, line in enumerate(extracted):
            self.assertEqual(line.rhyme_zone, expected_rhyme_zones[i])


class TestRhymeZoneExtraction(unittest.TestCase):
    def test_accented_last_word(self):
        zone = get_zone("Отнюдь не вдохновение, а гру" + MARK + "сть")

        self.assertEqual(zone.word, "гру" + MARK + "сть")
        self.assertEqual(zone.accents, [True])

    def test_keeps_trailing_clitic(self):
        line = "Лете" + MARK + "ть ли вдаль, подняться ввы" + MARK + "сь ли,"

        zone = get_zone(line)

        self.assertEqual(zone.word, "ввы" + MARK + "сь ли,")
        self.assertEqual(zone.accents, [True, False])

    def test_keeps_trailing_punctuation(self):
        line = "Жизни вы" + MARK + "питая ча" + MARK + "ша --"

        zone = get_zone(line)

        self.assertEqual(zone.word, "ча" + MARK + "ша --")
        self.assertEqual(zone.accents, [True, False])

    def test_full_accents(self):
        line = "Мѝлая О̀ля, за о̀кном ю̀жно-ура̀льский, сѐрый"

        zone = get_zone(line)

        self.assertEqual(zone.word, "сѐрый")
        self.assertEqual(zone.accents, [True, False])
