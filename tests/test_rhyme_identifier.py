import unittest

from parameterized import parameterized

from velimir.rhyme_identifier import (
    PatternEntry,
    build_patterns,
    build_rhyme_formulas,
    render_rhyme_formulas,
)

CASES = [
    (
        [0, 1, 0, 1, 2, 3, 2, 3],
        [
            PatternEntry(pattern=[0, 1, 0, 1], diff=[2, 2, 2, 2], repeats=2),
        ],
        "перекрестная : абаб",
    ),
    (
        [0, 1, 1, 0, 2, 3, 3, 2],
        [
            PatternEntry(pattern=[0, 1, 1, 0], diff=[2, 2, 2, 2], repeats=2),
        ],
        "охватная : абба",
    ),
    (
        [0, 1, 0, 1, 2, 1, 2, 3, 2],  # цепная
        [
            PatternEntry(pattern=[0, 1, 0], diff=[1, 1, 1], repeats=3),
        ],
        "цепная : аба",
    ),
    (
        [0, 1, 0, 0, 1, 2, 3, 2, 2, 3],
        [
            PatternEntry(pattern=[0, 1, 0, 0, 1], diff=[2, 2, 2, 2, 2], repeats=2),
        ],
        "затянутая : абааб",
    ),
    (
        # сочетание нескольких типов
        [0, 1, 0, 1, 2, 3, 2, 3, 4, 4, 4, 5, 5, 5],
        [
            PatternEntry(pattern=[0, 1, 0, 1], diff=[2, 2, 2, 2], repeats=2),
            PatternEntry(pattern=[4, 4, 4], diff=[1, 1, 1], repeats=2),
        ],
        "перекрестная : абаб # тройная : ааа",
    ),
    (
        [-1, 1, -1, 1, -1, 2, -1, 2],
        [
            PatternEntry(pattern=[-1, 1, -1, 1], diff=[0, 1, 0, 1], repeats=2),
        ],
        "четная : хаха",
    ),
]


class TestBuildRhymePatterns(unittest.TestCase):
    @parameterized.expand([(inp, patterns) for inp, patterns, _ in CASES])
    def test_build_patterns(self, inp, out):
        res = build_patterns(inp, [])

        self.assertListEqual(res, out)


class TestBuildRhymeFormulaStrings(unittest.TestCase):
    @parameterized.expand([(inp, formula_str) for inp, _, formula_str in CASES])
    def test_build_rhyme_formula_strings(self, inp, out):
        res = render_rhyme_formulas(build_rhyme_formulas(build_patterns(inp, [])))

        self.assertEqual(res, out)
