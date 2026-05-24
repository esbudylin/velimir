import unittest

from parameterized import parameterized

from velimir.parsers import parse_input_lines

# Wrapping <i> tag spanning multiple <line> elements
# Reproduces: xx/ellis/ellis-190, xx/merezhkovsky/merezh-109,
# xviii/derzhavin/derz-032-8
xml_wrapping_i = """<p class="verse">
<line meter="Я6ж"/><i>Ты зна̀ешь, му̀дрецы̀ с изда̀вних по̀р <rhyme-zone/>мечта̀ли,<br/>
<line meter="Я6ж"/>(Хотя̀ зада̀ча ѝх разрѐшена̀ <rhyme-zone/>едва̀ ли!)</i></p>
"""

# Malformed nested <i> tags (second <i> opens instead of closing)
# Reproduces: xix/zhukovsky/zhuk-636
xml_nested_i = """<p class="verse">
<line meter="Я4ж"/>Пѐрвая строка̀,<br/>
<line meter="Я4м"/>Сло̀вно как са̀м Ива̀н-царѐвич, отвѐтствуют: <i>бу̀ду<i>.<br/>
<line meter="Я4ж"/>Э̀тот отвѐт придво̀рные слу̀ги отно̀сят к Кощѐю;</i></i></p>
"""

# Wrapping <i> that closes on same <p> line but wraps inner <line> tags
xml_wrapping_i_multi = """<p class="verse"><line meter="Я4м"/><i>О, Бо̀же мо̀й, <rhyme-zone/>благо̀дарю̀<br/>
<line meter="Я4м"/>За то̀, что да̀л моѝм <rhyme-zone/>оча̀м<br/>
<line meter="Я4м"/>Ты вѝдеть мѝр, Твой вѐчный <rhyme-zone/>хра̀м,</i></p>
"""

# The first <line> is outside <i>, subsequent <line>s are inside <i>
# Like hemnitser hemn-133 where a valid stanza has <i> wrapping
# Wrapped in <html> because BS4/xml needs a single root for multi-stanza
xml_wrapping_i_inner_lines = """<html>
<p class="verse"><line meter="Я2м"/><i>Любѐзный <rhyme-zone/>дру̀г!</i></p>

<p class="verse"><line meter="Я6ж"/><i>Нелѐстной дру̀жбе тру̀д усѐрдный <rhyme-zone/>по̀свяща̀ю<br/>
<line meter="Я6м"/>И зна̀нью пра̀вому̀ судѝти <rhyme-zone/>прѐдлага̀ю;</i></p>
</html>
"""


class TestLineTextExtraction(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "wrapping_i_ellis",
                xml_wrapping_i,
                [
                    "Ты зна̀ешь, му̀дрецы̀ с изда̀вних по̀р мечта̀ли,",
                    "(Хотя̀ зада̀ча ѝх разрѐшена̀ едва̀ ли!)",
                ],
            ),
            (
                "nested_malformed_i",
                xml_nested_i,
                [
                    "Пѐрвая строка̀,",
                    "Сло̀вно как са̀м Ива̀н-царѐвич, отвѐтствуют: бу̀ду.",
                    "Э̀тот отвѐт придво̀рные слу̀ги отно̀сят к Кощѐю;",
                ],
            ),
            (
                "wrapping_i_multi",
                xml_wrapping_i_multi,
                [
                    "О, Бо̀же мо̀й, благо̀дарю̀",
                    "За то̀, что да̀л моѝм оча̀м",
                    "Ты вѝдеть мѝр, Твой вѐчный хра̀м,",
                ],
            ),
            (
                "wrapping_i_inner_lines",
                xml_wrapping_i_inner_lines,
                [
                    "Любѐзный дру̀г!",
                    "Нелѐстной дру̀жбе тру̀д усѐрдный по̀свяща̀ю",
                    "И зна̀нью пра̀вому̀ судѝти прѐдлага̀ю;",
                ],
            ),
        ]
    )
    def test_wrapping_i_does_not_merge_lines(self, name, xml_str, expected_texts):
        input_lines, _ = parse_input_lines(xml_str)
        actual_texts = [line.text for line in input_lines]

        self.assertEqual(
            len(actual_texts),
            len(expected_texts),
            f"Mismatched line count for case {name}",
        )
        for i, (actual, expected) in enumerate(zip(actual_texts, expected_texts)):
            self.assertEqual(
                actual,
                expected,
                f"Line {i} mismatch for case {name}:\nExpected: {expected!r}\nGot:      {actual!r}",
            )
