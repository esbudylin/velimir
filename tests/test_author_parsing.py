import unittest

from parameterized import parameterized

from velimir.author import get_author_sort_key


class TestAuthorKeyExtraction(unittest.TestCase):
    @parameterized.expand(
        [
            ("А. А. Ахматова", "Ахматова"),
            ("Б. Бета", "Бета"),
            ("П. С. Соловьева (Allegro)", "Соловьева"),
            ("А. А. Бестужев-Марлинский", "Бестужев-Марлинский"),
            ("Андрей Белый", "Белый"),
            ("А. С. Пушкин : Е. А. Баратынский", "Пушкин"),
            ("А. С. Пушкин : лицеисты", "Пушкин"),
        ]
    )
    def test_sort_key(self, full_name, key):
        res = get_author_sort_key(full_name)
        self.assertEqual(res, key)
