import logging
from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

from .accentuator import (
    build_accent_dict,
    extract_word_ending_mask,
    is_vowel,
    stress_mark_ord,
)
from .identifier import FailedLine, ProcessedLine, process_lines
from .io import read_accent_dicts
from .ml_preprocess import MeterClassRegistry
from .onnx import load_onnx_models, load_rhyme_onnx_model
from .rhyme_identifier import (
    RhymeInput,
    identify_rhyme_schema,
    render_rhyme_formulas,
)
from .settings import ACCENT_DICT_PATHS


@dataclass
class MarkupLine:
    text: str
    accented: str
    meter: str
    failed: bool


@dataclass
class MarkupResult:
    meta: dict[str, str]
    verses: list[list[MarkupLine]]


def split_verses(text: str) -> list[list[str]]:
    verses = []
    current = []

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        if not line.strip():
            if current:
                verses.append(current)
                current = []
        else:
            current.append(line)

    if current:
        verses.append(current)

    return verses


def flatten_verses(verses: list[list[str]]):
    flat = []
    stanza_breaks = []

    for verse in verses:
        stanza_breaks.append(len(flat))
        flat.extend(verse)

    return flat, stanza_breaks


def unflatten(processed: list[ProcessedLine], stanza_breaks: list[int]):
    """Split flat processed lines back into verses."""
    res = []
    current_stanza = []

    for i, line in enumerate(processed):
        if i in stanza_breaks and current_stanza:
            res.append(current_stanza)
            current_stanza = []
        current_stanza.append(line)

    if current_stanza:
        res.append(current_stanza)

    return res


def put_accents(line: str, mask: list[bool]):
    res = ""
    vowel_pos = 0

    for c in line:
        res += c
        if is_vowel(c):
            if mask[vowel_pos]:
                res += chr(stress_mark_ord)
            vowel_pos += 1

    return res


def rhyme_zone(
    line: str,
    accents: list[bool],
) -> RhymeInput:
    word_endings = extract_word_ending_mask(line)

    if len(word_endings) != len(accents):
        raise ValueError

    start = 0

    for i in range(len(accents) - 1, -1, -1):
        if not accents[i]:
            continue

        start = i

        while start > 0 and not word_endings[start - 1]:
            start -= 1

        break

    words = line.split()
    vowel_pos = 0
    first_word = len(words)

    for j, word in enumerate(words):
        word_vowels = sum(is_vowel(c) for c in word)

        if vowel_pos + word_vowels > start:
            first_word = j
            break

        vowel_pos += word_vowels

    return RhymeInput(" ".join(words[first_word:]), accents[start:])


def extract_rhyme_inputs(
    verses: list[list[str]],
    processed_verses: list[list[ProcessedLine | FailedLine]],
) -> list[RhymeInput]:
    inputs = []

    for verse_lines, verse_processed in zip(verses, processed_verses):
        for line, processed in zip(verse_lines, verse_processed):
            if isinstance(processed, FailedLine):
                accents = [False] * sum(is_vowel(c) for c in line)
            else:
                accents = processed.poetic_accents

            inputs.append(rhyme_zone(line, accents))

    return inputs


def format_verse(lines: list[MarkupLine]) -> str:
    parts = ['<p class="verse">']

    for line in lines:
        parts.append(
            f"<line meter={quoteattr(line.meter)}/>{escape(line.accented)}<br/>"
        )

    parts.append("</p>")
    return "\n".join(parts)


def render_xml(result: MarkupResult) -> str:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<body>"]

    for name, value in result.meta.items():
        if value:
            parts.append(f'<meta id="{escape(name)}">{escape(value)}</meta>')

    for verse in result.verses:
        parts.append(format_verse(verse))

    parts.append("</body>")
    return "\n".join(parts) + "\n"


class MarkupEngine:
    def __init__(self):
        MeterClassRegistry.initialize()

        build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

        self.meter_model, self.accent_model = load_onnx_models()
        self.rhyme_model = load_rhyme_onnx_model()

    def markup_text(self, text: str) -> MarkupResult:
        verses = split_verses(text)
        processed_verses = self._process_verses(verses)

        marked = [
            [
                self._markup_line(line, processed)
                for line, processed in zip(verse_lines, verse_processed)
            ]
            for verse_lines, verse_processed in zip(verses, processed_verses)
        ]

        rhymes = extract_rhyme_inputs(verses, processed_verses)

        return MarkupResult(
            meta={"rhyme": self._identify_rhyme(rhymes)},
            verses=marked,
        )

    def _process_verses(
        self,
        verses: list[list[str]],
    ) -> list[list[ProcessedLine | FailedLine]]:
        flat_lines, stanza_breaks = flatten_verses(verses)

        processed_flat = process_lines(
            self.meter_model,
            self.accent_model,
            flat_lines,
            stanza_breaks,
        )

        if len(processed_flat) != len(flat_lines):
            raise ValueError(
                "Mismatch: processed %d lines, expected %d"
                % (len(processed_flat), len(flat_lines))
            )

        return unflatten(processed_flat, stanza_breaks)

    def _identify_rhyme(self, rhymes: list[RhymeInput]) -> str:
        if not rhymes:
            return ""

        try:
            formulas = identify_rhyme_schema(rhymes, self.rhyme_model)
            return render_rhyme_formulas(formulas)
        except Exception:
            logging.exception("Failed to identify rhyme schema")
            return ""

    @staticmethod
    def _markup_line(line: str, processed: ProcessedLine | FailedLine) -> MarkupLine:
        if isinstance(processed, FailedLine):
            logging.error("Failed to process line: %s", line)
            return MarkupLine(text=line, accented=line, meter="???", failed=True)

        return MarkupLine(
            text=line,
            accented=put_accents(line, processed.poetic_accents),
            meter=processed.to_str(),
            failed=False,
        )
