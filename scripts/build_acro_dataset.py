import csv
import logging
import os
import re
from collections import Counter

from bs4 import BeautifulSoup
from spylls.hunspell import Dictionary

from velimir import cyrlat
from velimir.acro import (
    detect_acro,
    load_char_probabilities,
    load_dictionary,
    normalize_yo,
)
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings
from velimir.parsers import parse_input_lines, remove_accent_marks
from velimir.settings import (
    ACRO_CHAR_PROBABILITIES_CACHE_PATH,
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


def build_corpus_first_letter_counts(corpus_forms: Counter[str]) -> Counter[str]:
    letter_counts: Counter[str] = Counter()

    for form, count in corpus_forms.items():
        if form:
            letter_counts[form[0]] += count

    return letter_counts


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


def load_cached_char_probabilities(cache_path: str) -> dict[str, float] | None:
    try:
        with open(cache_path, "r", encoding="utf8") as cache_file:
            rows = [
                line.rstrip("\n").split("\t") for line in cache_file if line.strip()
            ]
    except FileNotFoundError:
        return None
    except OSError as error:
        logging.warning(
            "Could not read char probabilities cache %s: %s", cache_path, error
        )
        return None

    try:
        probabilities = {char: float(value) for char, value in rows}
    except ValueError:
        logging.warning("Invalid char probabilities cache %s", cache_path)
        return None

    if not probabilities or abs(sum(probabilities.values()) - 1.0) > 1e-6:
        logging.warning("Invalid char probabilities cache %s", cache_path)
        return None

    logging.info(
        "Loaded %d char probabilities from cache %s",
        len(probabilities),
        cache_path,
    )

    return probabilities


def save_cached_char_probabilities(
    cache_path: str, probabilities: dict[str, float]
) -> None:
    try:
        cache_directory = os.path.dirname(cache_path)
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)

        with open(cache_path, "w", encoding="utf8") as cache_file:
            for char, probability in sorted(probabilities.items()):
                cache_file.write(f"{char}\t{probability!r}\n")
    except OSError as error:
        logging.warning(
            "Could not write char probabilities cache %s: %s", cache_path, error
        )
        return

    logging.info("Saved char probabilities cache to %s", cache_path)


def build_dictionary(corpus_forms: Counter[str]) -> set[str]:
    cached_forms = load_cached_dictionary(ACRO_DICTIONARY_CACHE_PATH)

    if cached_forms is not None:
        return cached_forms

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

    save_cached_dictionary(ACRO_DICTIONARY_CACHE_PATH, forms)

    return forms


def build_char_probabilities(corpus_forms: Counter[str]) -> dict[str, float]:
    cached_probabilities = load_cached_char_probabilities(
        ACRO_CHAR_PROBABILITIES_CACHE_PATH
    )

    if cached_probabilities is not None:
        return cached_probabilities

    letter_counts = build_corpus_first_letter_counts(corpus_forms)
    total = sum(letter_counts.values())

    if not total:
        raise ValueError("Corpus contains no words")

    probabilities = {
        letter: count / total for letter, count in sorted(letter_counts.items())
    }

    logging.info(
        "Built char probabilities for %d first letters from %d corpus words",
        len(probabilities),
        total,
    )

    save_cached_char_probabilities(ACRO_CHAR_PROBABILITIES_CACHE_PATH, probabilities)

    return probabilities


def build_acro_data() -> tuple[set[str], dict[str, float]]:
    cached_forms = load_cached_dictionary(ACRO_DICTIONARY_CACHE_PATH)
    cached_probabilities = load_cached_char_probabilities(
        ACRO_CHAR_PROBABILITIES_CACHE_PATH
    )

    if cached_forms is not None and cached_probabilities is not None:
        return cached_forms, cached_probabilities

    corpus_forms = build_corpus_forms()

    forms = cached_forms if cached_forms is not None else build_dictionary(corpus_forms)
    probabilities = (
        cached_probabilities
        if cached_probabilities is not None
        else build_char_probabilities(corpus_forms)
    )

    return forms, probabilities


def write_output(acros):
    with open(ACRO_TABLE_PATH, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "author",
                "path",
                "created",
                "sequence",
                "forms",
                "partial",
                "coverage",
                "total",
                "p_value",
                "corpus_url",
            ]
        )

        for poem, matches in acros:
            writer.writerow(
                [
                    poem.author,
                    poem.path,
                    poem.created,
                    "; ".join(acro.sequence for acro in matches),
                    "; ".join(" ".join(acro.forms) for acro in matches),
                    matches[0].partial,
                    matches[0].coverage,
                    matches[0].total,
                    matches[0].statistics["p_value"],
                    build_rnc_url(poem.path),
                ]
            )


if __name__ == "__main__":
    LoggingSettings.setup()

    forms, probabilities = build_acro_data()
    load_dictionary(forms)
    load_char_probabilities(probabilities)

    write_output(find_acros())
