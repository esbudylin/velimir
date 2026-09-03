import unittest

from bs4 import BeautifulSoup
from velimir.parsers import extract_lines

xml = """
<p class="verse"><line meter="Ан4м"/>И, садя̀сь, запева̀ли <i>Варя̀га</i> <rhyme-zone/>однѝ,<br/>
<line meter="Ан3м"/>А другѝе -- не в ла̀д -- <i><rhyme-zone/>Ермака̀</i>,<br/>
<line meter="Ан4м"/>И крича̀ли <i>ура̀</i>, и шутѝли <rhyme-zone/>онѝ,<br/>
<line meter="Ан3м"/>И тихо̀нько крестѝлась <rhyme-zone/>рука̀.</p>
"""


class TestRhymeZone(unittest.TestCase):
    def test_rhyme_zone_extraction(self):
        soup = BeautifulSoup(xml, "xml")

        expected_rhyme_zones = ["однѝ,", "Ермака̀,", "онѝ,", "рука̀."]
        extracted = list(extract_lines(soup))

        self.assertEqual(len(expected_rhyme_zones), len(extracted))
        for i, line in enumerate(extracted):
            self.assertEqual(line.rhyme_zone, expected_rhyme_zones[i])
