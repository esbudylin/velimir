"""Build permalink URLs to Russian National Corpus full-text pages.

The ``search`` parameter of ``https://ruscorpora.ru/full-text`` is a
base64-encoded protobuf message.  The document to open is identified by a
nested field (top-level field 1 -> field 5 -> field 18 -> field 1 -> field 1)
holding a string of the form ``<subcorpus>/<path>.xml`` (e.g.
``poetic/xx/kornilovb/bkorn-033.xml``).
"""

import base64
import urllib.parse

CORPUS_FULL_TEXT_URL = "https://ruscorpora.ru/full-text"


def _varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1

    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _varint_field(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _string_field(field: int, text: str) -> bytes:
    return _bytes_field(field, text.encode("utf-8"))


def _attribute(name: str, value: str) -> bytes:
    return _bytes_field(
        1, _string_field(1, name) + _bytes_field(2, _string_field(1, value))
    )


def _lex_entry(word: str, dist: tuple[int, int] | None = None) -> bytes:
    parts = [
        _attribute("lex", word),
        _attribute("form", ""),
        _attribute("gramm", ""),
        _attribute("sem", ""),
        _attribute("flags", ""),
    ]

    if dist is not None:
        lo, hi = dist
        parts.append(
            _bytes_field(
                1,
                _string_field(1, "dist")
                + _bytes_field(4, _varint_field(1, lo) + _varint_field(2, hi)),
            )
        )

    return b"".join(parts)


def _search_query(words: list[str]) -> bytes:
    disambmod = _bytes_field(
        1,
        _string_field(1, "disambmod") + _bytes_field(2, _string_field(1, "all")),
    )

    main = disambmod
    for i, word in enumerate(words):
        dist = (-2, 2) if i > 0 else None
        main += _bytes_field(2, _lex_entry(word, dist))

    return _bytes_field(2, _bytes_field(1, main))


def _doc_selector(path: str) -> bytes:
    document = (
        _bytes_field(1, _string_field(1, path))
        + _varint_field(2, 0)
        + _varint_field(3, 100)
    )

    return (
        _bytes_field(1, b"")
        + _varint_field(4, 0)
        + _varint_field(8, 5)
        + _varint_field(15, 1)
        + _bytes_field(18, document)
    )


def build_rnc_url(path: str, words: list[str] | None = None) -> str:
    """Return a full-text permalink for a corpus text.

    ``path`` is the raw corpus file path (e.g. ``xx/kornilovb/bkorn-033``),
    as stored in the ``path`` column of ``poetic.csv``.  ``words`` are the
    search terms used to highlight matches in the text.
    """
    full_path = f"poetic/{path}.xml"

    message = _bytes_field(
        1,
        _search_query(words or [])
        + _bytes_field(5, _doc_selector(full_path))
        + _bytes_field(6, _varint_field(1, 9))
        + _bytes_field(7, b"\x03"),
    )

    search = base64.b64encode(message).decode("ascii")
    search = urllib.parse.quote(search, safe="")

    return f"{CORPUS_FULL_TEXT_URL}?search={search}"
