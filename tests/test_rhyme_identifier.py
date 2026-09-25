import unittest

import numpy as np
from parameterized import parameterized

from velimir.rhyme_identifier import (
    PatternEntry,
    build_patterns,
    build_rhyme_formulas,
    render_rhyme_formulas,
    calc_diff,
)

CASES = [
    (
        [0, 1, 0, 1, 2, 3, 2, 3],
        [
            PatternEntry(
                pattern=np.array([0, 1, 0, 1]), diff=np.array([2, 2, 2, 2]), repeats=2
            ),
        ],
        "перекрестная : абаб",
    ),
    (
        [0, 1, 1, 0, 2, 3, 3, 2],
        [
            PatternEntry(
                pattern=np.array([0, 1, 1, 0]), diff=np.array([2, 2, 2, 2]), repeats=2
            ),
        ],
        "охватная : абба",
    ),
    (
        [0, 1, 0, 1, 2, 1, 2, 3, 2],  # цепная
        [
            PatternEntry(
                pattern=np.array([0, 1, 0]), diff=np.array([1, 1, 1]), repeats=3
            ),
        ],
        "цепная : аба",
    ),
    (
        [0, 1, 0, 0, 1, 2, 3, 2, 2, 3],
        [
            PatternEntry(
                pattern=np.array([0, 1, 0, 0, 1]),
                diff=np.array([2, 2, 2, 2, 2]),
                repeats=2,
            ),
        ],
        "затянутая : абааб",
    ),
    (
        # сочетание нескольких типов
        [0, 1, 0, 1, 2, 3, 2, 3, 4, 4, 4, 5, 5, 5],
        [
            PatternEntry(
                pattern=np.array([0, 1, 0, 1]), diff=np.array([2, 2, 2, 2]), repeats=2
            ),
            PatternEntry(
                pattern=np.array([4, 4, 4]), diff=np.array([1, 1, 1]), repeats=2
            ),
        ],
        "перекрестная : абаб # тройная : ааа",
    ),
    (
        [-1, 1, -1, 1, -1, 2, -1, 2],
        [
            PatternEntry(
                pattern=np.array([-1, 1, -1, 1]), diff=np.array([0, 1, 0, 1]), repeats=2
            ),
        ],
        "четная : хаха",
    ),
    (
        # совпадение между первой и последней строфой
        [0, 1, 0, 1, 2, 3, 2, 3, 0, 4, 0, 4],
        [
            PatternEntry(
                pattern=np.array([0, 1, 0, 1]), diff=np.array([2, 2, 2, 2]), repeats=3
            ),
        ],
        "перекрестная : абаб",
    ),
    (
        [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5],
        [
            PatternEntry(
                pattern=np.array([0, 1, 2, 0, 1, 2]),
                diff=np.array([3] * 6),
                repeats=2,
            )
        ],
        "скользящая : абвабв",
    ),
    (
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7],
        [
            PatternEntry(
                pattern=np.array([0, 1, 2, 3, 0, 1, 2, 3]),
                diff=np.array([4] * 8),
                repeats=2,
            )
        ],
        "скользящая : абвгабвг",
    ),
]


class TestBuildRhymePatterns(unittest.TestCase):
    @parameterized.expand([(inp, patterns) for inp, patterns, _ in CASES])
    def test_build_patterns(self, inp, out):
        res = build_patterns(inp)

        self.assertEqual(len(res), len(out))
        for actual, expected in zip(res, out):
            np.testing.assert_array_equal(actual.pattern, expected.pattern)
            np.testing.assert_array_equal(actual.diff, expected.diff)
            self.assertEqual(actual.repeats, expected.repeats)


class TestBuildRhymeFormulaStrings(unittest.TestCase):
    @parameterized.expand([(inp, formula_str) for inp, _, formula_str in CASES])
    def test_build_rhyme_formula_strings(self, inp, out):
        res = render_rhyme_formulas(build_rhyme_formulas(build_patterns(inp), inp))

        self.assertEqual(res, out)


class TestCalculateArrayDiff(unittest.TestCase):
    def test_calc_diff(self):
        arr1 = np.array([2, 3, 2, 3])
        arr2 = np.array([0, 4, 0, 4])

        np.testing.assert_array_equal(calc_diff(arr1, arr2), np.array([2] * 4))

    def test_calc_diff_chain(self):
        arr1 = np.array([1, 2, 1])
        arr2 = np.array([2, 0, 2])

        np.testing.assert_array_equal(calc_diff(arr1, arr2), np.array([1] * 3))

    def test_calc_diff_negatives(self):
        arr1 = np.array([-1, 1, -1, 1])
        arr2 = np.array([-1, 3, -1, 3])

        np.testing.assert_array_equal(calc_diff(arr1, arr2), np.array([0, 1, 0, 1]))
