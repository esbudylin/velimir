import csv
import logging
import os
import re
from collections import Counter
from math import log10

from bs4 import BeautifulSoup

from velimir import cyrlat
from velimir.acro import (
    WordInfo,
    detect_acro,
    load_char_probabilities,
    load_dictionary,
    normalize_yo,
)
from velimir.nlp import morph_analyzer
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

OUTPUT_PERCENTILE = 98.0


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

            if len(lines) > 30:
                lines = lines[:30]

            def extract_first_char(line):
                subbed = re.sub("[^а-яё]", "", line.text.lower())
                return subbed[0] if subbed else ""

            first_letters = "".join(map(extract_first_char, lines))

            if not first_letters:
                continue

            if acro := detect_acro(first_letters):
                yield poem, acro


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


def load_cached_dictionary(cache_path: str) -> dict[str, WordInfo] | None:
    try:
        with open(cache_path, "r", encoding="utf8", newline="") as cache_file:
            reader = csv.reader(cache_file)

            if next(reader, None) != ["form", "frequency", "is_name"]:
                logging.warning("Invalid dictionary cache header %s", cache_path)
                return None

            forms = {}

            for row in reader:
                if not row:
                    continue

                form, frequency, is_name = row
                forms[form] = WordInfo(int(frequency), is_name == "1")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logging.warning("Could not read dictionary cache %s: %s", cache_path, error)
        return None

    logging.info(
        "Loaded dictionary with %d forms from cache %s", len(forms), cache_path
    )

    return forms


def save_cached_dictionary(cache_path: str, forms: dict[str, WordInfo]) -> None:
    try:
        cache_directory = os.path.dirname(cache_path)
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)

        with open(cache_path, "w", encoding="utf8", newline="") as cache_file:
            writer = csv.writer(cache_file)
            writer.writerow(["form", "frequency", "is_name"])

            for form, info in sorted(forms.items()):
                writer.writerow([form, info.frequency, 1 if info.is_name else 0])
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


def normalize_corpus_forms(
    corpus_forms: Counter[str],
) -> tuple[Counter[str], set[str]]:
    normalized_forms: Counter[str] = Counter()
    known_normals: set[str] = set()

    for form, count in corpus_forms.items():
        parses = [parse for parse in morph_analyzer.parse(form) if parse.is_known]

        if not parses:
            normalized_forms[form] += count
            continue

        for normal_form in {normalize_yo(parse.normal_form) for parse in parses}:
            normalized_forms[normal_form] += count
            known_normals.add(normal_form)

    return normalized_forms, known_normals


def passes_threshold(form: str, count: int, known_normals: set[str]) -> bool:
    if len(form) >= 5:
        return True

    if len(form) == 4:
        return count > 20

    if len(form) == 3:
        return count > 30

    if len(form) == 2:
        return count > 50 and form in known_normals

    return False


NAME_GRAMMEMES = {"Name", "Surn", "Patr"}


def extract_name_forms() -> set[str]:
    candidates = {
        normalize_yo(word)
        for word, tag, *_ in morph_analyzer.dictionary.iter_known_words()
        if tag.grammemes & NAME_GRAMMEMES
    }

    forms = {
        form
        for form in candidates
        if (parses := morph_analyzer.parse(form))
        and parses[0].tag.grammemes & NAME_GRAMMEMES
    }

    logging.info(
        "Extracted %d name forms from %d pymorphy candidates",
        len(forms),
        len(candidates),
    )

    return forms


def build_dictionary(corpus_forms: Counter[str]) -> dict[str, WordInfo]:
    normalized_forms, known_normals = normalize_corpus_forms(corpus_forms)

    forms: dict[str, WordInfo] = {}

    for form, count in normalized_forms.items():
        if not passes_threshold(form, count, known_normals):
            continue

        forms[form] = WordInfo(count)

        if form not in known_normals:
            continue

        for parse in morph_analyzer.parse(form):
            if parse.is_known and normalize_yo(parse.normal_form) == form:
                for word in morph_analyzer.get_lexeme(parse):
                    lexeme_form = normalize_yo(word.word)
                    forms.setdefault(lexeme_form, WordInfo(count))

    logging.info(
        "Corpus forms: %d, normalized: %d, after threshold validation: %d",
        len(corpus_forms),
        len(normalized_forms),
        len(forms),
    )

    added_name_forms = 0

    for form in extract_name_forms():
        info = forms.get(form)

        if info is None:
            forms[form] = WordInfo(corpus_forms.get(form, 0), is_name=True)
            added_name_forms += 1
        elif not info.is_name:
            forms[form] = WordInfo(info.frequency, is_name=True)

    logging.info(
        "Added %d name forms, total dictionary forms: %d",
        added_name_forms,
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


def build_acro_data() -> tuple[dict[str, WordInfo], dict[str, float]]:
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


def acro_combined_score(acro) -> float:
    p_value = acro.statistics["p_value"]

    return acro.score / log10(max(p_value, 1e-12))


def write_output(acros):
    scored = sorted(
        ((acro_combined_score(acro), poem, acro) for poem, acro in acros),
        key=lambda item: item[0],
        reverse=True,
    )

    if scored and OUTPUT_PERCENTILE > 0:
        scores = sorted(score for score, _, _ in scored)
        index = min(len(scores) - 1, int(len(scores) * OUTPUT_PERCENTILE / 100))
        threshold = scores[index]
        scored = [item for item in scored if item[0] >= threshold]

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
                "p_value",
                "score",
                "combined_score",
                "corpus_url",
            ]
        )

        for combined_score, poem, match in scored:
            writer.writerow(
                [
                    poem.author,
                    poem.path,
                    poem.created,
                    match.sequence,
                    " ".join(match.forms),
                    match.partial,
                    match.statistics["p_value"],
                    match.score,
                    combined_score,
                    build_rnc_url(poem.path),
                ]
            )


if __name__ == "__main__":
    LoggingSettings.setup()

    forms, probabilities = build_acro_data()
    load_dictionary(forms)
    load_char_probabilities(probabilities)

    write_output(find_acros())
