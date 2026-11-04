import csv
import logging
import os
import re
from collections import Counter

from bs4 import BeautifulSoup
from spylls.hunspell import Dictionary

from velimir import cyrlat
from velimir.acro import detect_acro, load_dictionary, normalize_yo
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings
from velimir.parsers import parse_input_lines, remove_accent_marks
from velimir.settings import (
    ACRO_DICTIONARY_CACHE_PATH,
    ACRO_TABLE_PATH,
    METADATA_TABLE,
    InputDialect,
)
from web.rnc_url import build_rnc_url

WORD_RE = re.compile(r"[а-яё]+")
WIKI_DICTIONARY_PATH = "data/wikt/ru_RU"


def find_acros():
    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        for i, row in enumerate(input_reader):
            if (i + 1) % 100 == 0:
                logging.info("Processing %d poem", i + 1)

            poem = InputPoem.from_row(row)
            xml_str = read_poem_xml(poem.path)
            lines, _ = parse_input_lines(xml_str, allow_latin=True)

            if len(lines) < 4:
                continue

            if len(lines) > 15:
                lines = lines[:15]

            def extract_first_char(line):
                subbed = re.sub("[^а-яё]", "", line.text.lower())
                return subbed[0] if subbed else ""

            first_letters = "".join(map(extract_first_char, lines))

            if not first_letters:
                continue

            acros = detect_acro(first_letters)

            if acros:
                logging.info(
                    "Acro found %s, %s",
                    poem.path,
                    "; ".join(" ".join(acro.forms) for acro in acros),
                )
                yield poem, acros


def extract_xml_words(xml_str: str):
    soup = BeautifulSoup(xml_str, "xml")

    for text in soup.stripped_strings:
        if cyrlat.detect(text) == cyrlat.DetectionResult.CYRLAT:
            text = cyrlat.fix(text)

        text = remove_accent_marks(text)

        yield from WORD_RE.findall(text.lower())


def build_corpus_forms() -> Counter[str]:
    forms: Counter[str] = Counter()

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        for i, row in enumerate(input_reader):
            if (i + 1) % 100 == 0:
                logging.info("Building dictionary, processed %d poem", i + 1)

            poem = InputPoem.from_row(row)
            xml_str = read_poem_xml(poem.path)

            forms.update(normalize_yo(word) for word in extract_xml_words(xml_str))

    return forms


def load_cached_dictionary(cache_path: str) -> set[str] | None:
    try:
        with open(cache_path, "r", encoding="utf8") as cache_file:
            forms = {line.strip() for line in cache_file if line.strip()}
    except FileNotFoundError:
        return None
    except OSError as error:
        logging.warning("Could not read dictionary cache %s: %s", cache_path, error)
        return None

    logging.info(
        "Loaded dictionary with %d forms from cache %s", len(forms), cache_path
    )

    return forms


def save_cached_dictionary(cache_path: str, forms: set[str]) -> None:
    try:
        cache_directory = os.path.dirname(cache_path)
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)

        with open(cache_path, "w", encoding="utf8") as cache_file:
            for form in sorted(forms):
                cache_file.write(form + "\n")
    except OSError as error:
        logging.warning("Could not write dictionary cache %s: %s", cache_path, error)
        return

    logging.info("Saved dictionary cache to %s", cache_path)


def build_dictionary(cache_path: str | None = None) -> set[str]:
    cache_path = cache_path or ACRO_DICTIONARY_CACHE_PATH
    cached_forms = load_cached_dictionary(cache_path)

    if cached_forms is not None:
        return cached_forms

    corpus_forms = build_corpus_forms()

    wiki = Dictionary.from_files(WIKI_DICTIONARY_PATH)
    forms = {
        form
        for form, count in corpus_forms.items()
        if len(form) > 4 or (count > 20 and wiki.lookup(form.capitalize()))
    }

    logging.info(
        "Corpus forms: %d, after wiki validation: %d",
        len(corpus_forms),
        len(forms),
    )

    save_cached_dictionary(cache_path, forms)

    return forms


def write_output(acros):
    with open(ACRO_TABLE_PATH, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "author",
                "path",
                "created",
                "acro",
                "forms",
                "partial",
                "coverage",
                "total",
                "corpus_url",
            ]
        )

        for poem, matches in acros:
            writer.writerow(
                [
                    poem.author,
                    poem.path,
                    poem.created,
                    "; ".join(" ".join(acro.words) for acro in matches),
                    "; ".join(" ".join(acro.forms) for acro in matches),
                    matches[0].partial,
                    matches[0].coverage,
                    matches[0].total,
                    build_rnc_url(poem.path),
                ]
            )


if __name__ == "__main__":
    LoggingSettings.setup()
    load_dictionary(build_dictionary())
    write_output(find_acros())
