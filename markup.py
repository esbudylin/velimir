import logging
import sys

from velimir.accentuator import build_accent_dict, is_vowel, stress_mark_ord
from velimir.identifier import ProcessedLine, process_lines
from velimir.io import read_accent_dicts
from velimir.ml_loader import MeterClassRegistry
from velimir.settings import ACCENT_DICT_PATHS, LoggingSettings


def read_verses_from_stdin() -> list[list[str]]:
    verses = []
    current = []

    for raw in sys.stdin:
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


def emit_result(verses, processed_verses):
    print('<?xml version="1.0" encoding="utf-8"?>')
    print("<body>")

    for verse_lines, verse_processed in zip(verses, processed_verses):
        print(format_verse(verse_lines, verse_processed))

    print("</body>")


def format_verse(lines: list[str], processed_lines: list[ProcessedLine | None]) -> str:
    parts = ['<p class="verse">']

    for line, processed in zip(lines, processed_lines):
        meter = processed.to_str() if processed else "???"
        accline = put_accents(line, processed.poetic_accent_mask) if processed else line
        parts.append(f'<line meter="{meter}"/>{accline}<br/>')

    parts.append("</p>")
    return "\n".join(parts)


def main():
    MeterClassRegistry.initialize()
    LoggingSettings.setup()

    build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    verses = read_verses_from_stdin()

    if not verses:
        logging.error("No input provided")
        sys.exit(1)

    flat_lines, stanza_breaks = flatten_verses(verses)

    processed_flat = process_lines(flat_lines, stanza_breaks)

    if len(processed_flat) != len(flat_lines):
        logging.error(
            "Mismatch: processed %d lines, expected %d",
            len(processed_flat),
            len(flat_lines),
        )
        sys.exit(1)

    processed_verses = unflatten(processed_flat, stanza_breaks)
    emit_result(verses, processed_verses)


if __name__ == "__main__":
    main()
