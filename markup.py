import logging
import sys

from velimir.accentuator import build_accent_dict, is_vowel, stress_mark_ord
from velimir.identifier import ProcessedLine, process_lines
from velimir.io import read_accent_dicts
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
    lengths = []

    for verse in verses:
        flat.extend(verse)
        lengths.append(len(verse))

    return flat, lengths


def unflatten(processed: list[ProcessedLine], lengths: list[int]):
    """Split flat processed lines back into verses."""
    res = []
    idx = 0

    for length in lengths:
        res.append(processed[idx : idx + length])
        idx += length

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


def format_verse(lines: list[str], processed_lines: list[ProcessedLine | None]) -> str:
    parts = ['<p class="verse">']

    for line, processed in zip(lines, processed_lines):
        meter = processed.to_str() if processed else "???"
        accline = put_accents(line, processed.poetic_accent_mask) if processed else line
        parts.append(f'<line meter="{meter}"/>{accline}<br/>')

    parts.append("</p>")
    return "\n".join(parts)


def main():
    build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))
    logging.basicConfig(**LoggingSettings().model_dump())

    verses = read_verses_from_stdin()

    if not verses:
        logging.error("No input provided")
        sys.exit(1)

    flat_lines, lengths = flatten_verses(verses)

    processed_flat = process_lines(flat_lines)

    if len(processed_flat) != len(flat_lines):
        logging.error(
            "Mismatch: processed %d lines, expected %d",
            len(processed_flat),
            len(flat_lines),
        )
        sys.exit(1)

    processed_verses = unflatten(processed_flat, lengths)

    for verse_lines, verse_processed in zip(verses, processed_verses):
        print(format_verse(verse_lines, verse_processed))


if __name__ == "__main__":
    main()
