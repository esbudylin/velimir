import unittest
from datetime import datetime, timezone

from velimir.creation_date import CreationDate
from velimir.domain_models import InputPoem


def ts(year, month=1, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def poem(created, exact=""):
    return InputPoem(
        author="",
        created=created,
        exact=exact,
        header="",
        formula="",
        meter="",
        clausula="",
        feet="",
        rhyme="",
        extra="",
        path="",
    )


class TestCreationDate(unittest.TestCase):
    def test_year(self):
        result = CreationDate.extract(poem("1785"))
        self.assertEqual(result.lower, ts(1785))
        self.assertIsNone(result.upper)
        self.assertTrue(result.is_exact)

    def test_year_month(self):
        result = CreationDate.extract(poem("1785.05"))
        self.assertEqual(result.lower, ts(1785, 5))
        self.assertIsNone(result.upper)

    def test_year_month_day(self):
        result = CreationDate.extract(poem("1750.08.18"))
        self.assertEqual(result.lower, ts(1750, 8, 18))
        self.assertIsNone(result.upper)

    def test_day_month_year(self):
        result = CreationDate.extract(poem("21.01.1924"))
        self.assertEqual(result.lower, ts(1924, 1, 21))
        self.assertIsNone(result.upper)

    def test_year_range(self):
        result = CreationDate.extract(poem("1959-1989"))
        self.assertEqual(result.lower, ts(1959))
        self.assertEqual(result.upper, ts(1989))

    def test_mixed_precision_range(self):
        result = CreationDate.extract(poem("1788.08-1795"))
        self.assertEqual(result.lower, ts(1788, 8))
        self.assertEqual(result.upper, ts(1795))

    def test_full_date_range(self):
        result = CreationDate.extract(poem("1750.12-1751.02.15"))
        self.assertEqual(result.lower, ts(1750, 12))
        self.assertEqual(result.upper, ts(1751, 2, 15))

    def test_more_than_two_parts(self):
        with self.assertRaises(ValueError):
            CreationDate.extract(poem("1785-1790-1800"))

    def test_dubious(self):
        result = CreationDate.extract(poem("1785", exact="неточная"))
        self.assertFalse(result.is_exact)

    def test_exact(self):
        result = CreationDate.extract(poem("1785", exact=""))
        self.assertTrue(result.is_exact)


if __name__ == "__main__":
    unittest.main()
