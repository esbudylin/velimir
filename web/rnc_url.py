"""Build permalink URLs to Russian National Corpus full-text pages.

The ``search`` parameter of ``https://ruscorpora.ru/full-text`` is a
base64-encoded ``FrontendSearchState`` protobuf message (see the schema that
ships with the RNC frontend).  It is produced by RNC's own ``encodeSearchQuery``
helper, which serialises::

    FrontendSearchState {
        SearchQuery            query          = 1;
        FrontendPreviousParams previousParams = 2;
    }

A permalink that opens a single text is built the same way the RNC frontend
does it (``searchStore.fullTextQueryParams``): the text is identified by
``query.params.expdiap.docSource.docId`` (a ``DocSource`` inside a
``DiapSource``), the result type is ``FULL_TEXT`` and the search parameters
that were active when the link was copied are kept in
``previousParams``.

This module only needs the corpus file path, so it synthesises the base search
parameters RNC uses by default.  RNC additionally preserves the subcorpus
filters that were active when a link was copied (e.g. an author facet); we do
not have that context here and omit it.  The path is stored without the
``poetic/`` prefix and the ``.xml`` suffix in the ``path`` column of
``poetic.csv``.
"""

import base64
import urllib.parse

CORPUS_FULL_TEXT_URL = "https://ruscorpora.ru/full-text"

# ``Corpus.type`` enum values (only the one we care about).
CORPUS_POETIC = 9

# ``SearchResultType`` enum values.
RESULT_TYPE_FULL_TEXT = 3
RESULT_TYPE_META = 6

# Defaults taken from the RNC frontend (``CgaYUsgE`` search store).
DEFAULT_KWSZ = 5
DEFAULT_PAGE_SIZE = 100
DEFAULT_DOCS_PER_PAGE = 10
DEFAULT_SNIPPETS_PER_PAGE = 50
DEFAULT_SNIPPETS_PER_DOC = 10


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


def _packed_field(field: int, values: list[int]) -> bytes:
    return _bytes_field(field, b"".join(_varint(value) for value in values))


# --- leaf messages ---------------------------------------------------------


def _condition(field_name: str, value: str) -> bytes:
    """``ConditionValue`` holding a text condition (``text`` is field 2)."""
    return _string_field(1, field_name) + _bytes_field(
        2, _string_field(1, value)
    )


def _formsection_value(conditions: list[bytes]) -> bytes:
    """``FormSectionValue`` with a list of ``ConditionValue`` (field 1)."""
    return b"".join(_bytes_field(1, condition) for condition in conditions)


def _doc_source(doc_id: str) -> bytes:
    """``DocSource``: the corpus document id, e.g. ``poetic/.../mandl-603.xml``."""
    return _string_field(1, doc_id)


def _diap_source(doc_id: str, start: int, end: int) -> bytes:
    """``DiapSource``: a document plus a fragment range."""
    return (
        _bytes_field(1, _doc_source(doc_id))
        + _varint_field(2, start)
        + _varint_field(3, end)
    )


def _pagination(
    page: int = 0,
    docs_per_page: int | None = None,
    snippets_per_page: int | None = None,
    snippets_per_doc: int | None = None,
) -> bytes:
    """``PaginationParams``."""
    out = _varint_field(1, page)
    if docs_per_page is not None:
        out += _varint_field(2, docs_per_page)
    if snippets_per_page is not None:
        out += _varint_field(3, snippets_per_page)
    if snippets_per_doc is not None:
        out += _varint_field(4, snippets_per_doc)
    return out


def _search_params(
    page_params: bytes,
    *,
    expdiap: bytes | None = None,
    kwsz: int = DEFAULT_KWSZ,
    sampling: int = 0,
    no_diacritic: bool = True,
) -> bytes:
    """``SearchParams`` with the fields RNC sets for a full-text permalink."""
    out = _bytes_field(1, page_params)
    out += _varint_field(4, sampling)
    out += _varint_field(8, kwsz)
    out += _varint_field(15, int(no_diacritic))
    if expdiap is not None:
        out += _bytes_field(18, expdiap)
    return out


def _corpus(corpus_type: int) -> bytes:
    """``Corpus``: just the corpus type."""
    return _varint_field(1, corpus_type)


def _lex_gramm(words: list[str]) -> bytes:
    """``SearchFormValue`` (``SearchQuery.lexGramm``) highlighting ``words``.

    Mirrors RNC's form builder: a single top-level section whose ``extra``
    conditions select the disambiguation mode, and whose subsections carry the
    actual lexical conditions.
    """
    extras = [
        _condition("disambmod", "main"),
        _condition("distmod", "no_zeros"),
    ]
    section = b"".join(_bytes_field(1, condition) for condition in extras)
    for word in words:
        subsection = _formsection_value([_condition("lex", word)])
        section += _bytes_field(2, subsection)
    return _bytes_field(1, section)


def _search_query(
    *,
    params: bytes,
    corpus_type: int,
    result_type: list[int],
    lex_gramm: bytes | None = None,
) -> bytes:
    """``SearchQuery``."""
    out = b""
    if lex_gramm is not None:
        out += _bytes_field(2, lex_gramm)
    out += _bytes_field(5, params)
    out += _bytes_field(6, _corpus(corpus_type))
    out += _packed_field(7, result_type)
    return out


def _frontend_previous_params(search_params: bytes, result_type: int) -> bytes:
    """``FrontendPreviousParams``."""
    return _bytes_field(1, search_params) + _varint_field(2, result_type)


def _frontend_search_state(query: bytes, previous_params: bytes) -> bytes:
    """``FrontendSearchState``."""
    return _bytes_field(1, query) + _bytes_field(2, previous_params)


def build_rnc_url(path: str, words: list[str] | None = None) -> str:
    """Return a full-text permalink for a corpus text.

    ``path`` is the raw corpus file path (e.g. ``xx/kornilovb/bkorn-033``), as
    stored in the ``path`` column of ``poetic.csv``.  ``words`` are optional
    search terms that will be highlighted in the text.
    """
    full_path = f"poetic/{path}.xml"

    # Parameters of the search the permalink was "copied" from; RNC keeps them
    # in ``previousParams`` and uses them as the base for the opened document.
    base_params = _search_params(
        _pagination(
            0,
            DEFAULT_DOCS_PER_PAGE,
            DEFAULT_SNIPPETS_PER_PAGE,
            DEFAULT_SNIPPETS_PER_DOC,
        )
    )

    # The opened document is selected through ``params.expdiap`` (a document
    # source plus the [0, page_size) fragment range) and shown as full text.
    query_params = _search_params(
        _pagination(0),
        expdiap=_diap_source(full_path, 0, DEFAULT_PAGE_SIZE),
    )
    query = _search_query(
        params=query_params,
        corpus_type=CORPUS_POETIC,
        result_type=[RESULT_TYPE_FULL_TEXT],
        lex_gramm=_lex_gramm(words) if words else None,
    )

    state = _frontend_search_state(
        query,
        _frontend_previous_params(base_params, RESULT_TYPE_META),
    )

    # RNC's ``encodeSearchQuery`` percent-encodes the base64 and the router
    # encodes the resulting query value once more, so the canonical link is
    # double-encoded (e.g. the ``=`` padding ends up as ``%253D``).
    encoded = base64.b64encode(state).decode("ascii")
    search = urllib.parse.quote(urllib.parse.quote(encoded, safe=""), safe="")

    return f"{CORPUS_FULL_TEXT_URL}?search={search}"
