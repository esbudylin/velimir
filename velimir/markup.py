import logging
from dataclasses import dataclass

from .accentuator import build_accent_dict, is_vowel, stress_mark_ord
from .identifier import FailedLine, ProcessedLine, process_lines
from .io import read_accent_dicts
from .ml_preprocess import MeterClassRegistry
from .onnx import load_onnx_models
from .settings import ACCENT_DICT_PATHS


@dataclass
class MarkupLine:
    text: str
    accented: str
    meter: str
    failed: bool


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


def format_verse(lines: list[MarkupLine]) -> str:
    parts = ['<p class="verse">']

    for line in lines:
        parts.append(f'<line meter="{line.meter}"/>{line.accented}<br/>')

    parts.append("</p>")
    return "\n".join(parts)


def render_xml(verses: list[list[MarkupLine]]) -> str:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<body>"]

    for verse in verses:
        parts.append(format_verse(verse))

    parts.append("</body>")
    return "\n".join(parts) + "\n"


class MarkupEngine:
    def __init__(self):
        MeterClassRegistry.initialize()

        build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

        self.meter_model, self.accent_model = load_onnx_models()

    def markup_text(self, text: str) -> list[list[MarkupLine]]:
        return self._markup_verses(split_verses(text))

    def _markup_verses(self, verses: list[list[str]]) -> list[list[MarkupLine]]:
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

        processed_verses = unflatten(processed_flat, stanza_breaks)

        return [
            [
                self._markup_line(line, processed)
                for line, processed in zip(verse_lines, verse_processed)
            ]
            for verse_lines, verse_processed in zip(verses, processed_verses)
        ]

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
