import unittest

from src.parsers import parse_lines
from src.models import OutputPoem


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.xml_path = "data/texts/xix/1830/1830-001.xml"

    def test_data_round_trip(self):
        with open(self.xml_path, "r", encoding="utf8") as f:
            xml = f.read()

        poem = OutputPoem(path=self.xml_path, lines=list(parse_lines(xml)))

        encoded = poem.encode()
        decoded = OutputPoem.decode(encoded)

        self.assertDictEqual(poem.model_dump(), decoded.model_dump())
