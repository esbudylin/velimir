"""Сравнение акцентуаторов на акцентниках РНК.

Сравниваются собственный акцентуатор проекта (словари + StressRNN),
ruaccent и silero-stress. Внешние библиотеки объявлены в
dependency-группе accent, запуск:

    make compare-accentuators
    # или
    uv run --group accent python scripts/compare_accentuators.py

Если библиотека не установлена, соответствующий акцентуатор
пропускается с предупреждением.
"""

import argparse
import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterator

from bs4 import BeautifulSoup

import velimir.accentuator as accentuator
from velimir.domain_models import InputPoem, SyllableFeatures
from velimir.io import read_accent_dicts, read_poem_xml
from velimir.logger import LoggingSettings, delayed_logger
from velimir.parsers import clean_line, extract_lines
from velimir.settings import ACCENT_DICT_PATHS, METADATA_TABLE, InputDialect

Predictor = Callable[[str], list[bool]]


@dataclass(slots=True)
class ContextLine:
    prev: str
    text: str
    next: str


def extract_ak_contexts(csv_reader: csv.DictReader) -> Iterator[ContextLine]:
    for row in csv_reader:
        poem = InputPoem.from_row(row)

        if "Ак" not in poem.meter:
            continue

        delayed_logger.create(
            logging.INFO,
            "Transforming poem: %s, meter: %s",
            poem.path,
            poem.formula,
        )

        xml_str = read_poem_xml(poem.path)
        soup = BeautifulSoup(xml_str, "xml")
        extracted = list(extract_lines(soup))

        for i, line in enumerate(extracted):
            if "Ак" not in line.meter:
                continue

            prev_text = extracted[i - 1].text if i > 0 else ""
            next_text = extracted[i + 1].text if i + 1 < len(extracted) else ""

            yield ContextLine(prev=prev_text, text=line.text, next=next_text)


def marked_to_mask(clean: str, marked: str) -> list[bool]:
    mask = []
    stress_next = False

    for char in marked:
        if char == "+":
            stress_next = True
        elif accentuator.is_vowel(char):
            mask.append(stress_next or char in "ёЁ")
            stress_next = False

    expected = accentuator.vowel_count(clean)

    if len(mask) != expected:
        raise ValueError(f"Vowel count mismatch: expected {expected}, got {len(mask)}")

    return mask


def predict_ours(text: str) -> list[bool]:
    return accentuator.accent_line(text)


def build_predictors() -> dict[str, Predictor]:
    predictors: dict[str, Predictor] = {"ours": predict_ours}

    try:
        from ruaccent import RUAccent
    except ImportError:
        logging.warning("ruaccent is not installed, skipping it")
    else:
        ruaccent_model = RUAccent()
        ruaccent_model.load(omograph_model_size="turbo3.1", use_dictionary=True)

        def predict_ruaccent(text: str) -> list[bool]:
            return marked_to_mask(text, ruaccent_model.process_all(text))

        predictors["ruaccent"] = predict_ruaccent

    try:
        from silero_stress import load_accentor
    except ImportError:
        logging.warning("silero-stress is not installed, skipping it")
    else:
        silero_model = load_accentor()

        def predict_silero(text: str) -> list[bool]:
            return marked_to_mask(text, silero_model(text))

        predictors["silero"] = predict_silero

    return predictors


def word_diff_stats(masks: SyllableFeatures) -> tuple[list[int], int, int]:
    error_indexes = []
    analysed = 0
    correct = 0

    word_ling = []
    word_poet = []
    word_index = 0

    for ling, poet, last in zip(
        masks.linguistic_accents,
        masks.poetic_accents,
        masks.last_in_word,
    ):
        word_ling.append(ling)
        word_poet.append(poet)

        if last:
            if len(word_poet) > 1 and sum(word_poet):
                analysed += 1

                if sum(a != b for a, b in zip(word_ling, word_poet)):
                    error_indexes.append(word_index)
                else:
                    correct += 1

            word_ling = []
            word_poet = []
            word_index += 1

    return error_indexes, analysed, correct


def evaluate_lines(
    contexts: list[ContextLine],
    predict: Predictor,
    name: str,
    use_context: bool,
) -> dict:
    total_lines = 0
    total_words = 0
    total_analysed = 0
    total_correct = 0
    total_diff = 0.0
    diffed_words: Counter[str] = Counter()

    for context in contexts:
        line = context.text
        try:
            reference = accentuator.extract_accent_mask(line)
            cleaned = clean_line(accentuator.remove_accent_marks(line))

            if use_context:
                prev_clean = clean_line(accentuator.remove_accent_marks(context.prev))
                next_clean = clean_line(accentuator.remove_accent_marks(context.next))
                model_input = "\n".join((prev_clean, cleaned, next_clean))
            else:
                model_input = cleaned

            predicted = predict(model_input)

            if use_context:
                start = accentuator.vowel_count(prev_clean)
                end = start + accentuator.vowel_count(cleaned)
                predicted = predicted[start:end]

            masks = SyllableFeatures(
                poetic_accents=reference,
                last_in_word=accentuator.extract_word_ending_mask(cleaned),
                linguistic_accents=predicted,
            )

            error_indexes, analysed, correct = word_diff_stats(masks)

            line_stripped = re.sub(r"[^А-яЁё\s-]+", "", line)
            line_words = list(
                filter(
                    lambda w: sum(map(accentuator.is_vowel, w)),
                    line_stripped.split(),
                )
            )

            for di in error_indexes:
                if di < len(line_words):
                    diffed_words[line_words[di].lower()] += 1
        except Exception as e:
            delayed_logger.record()
            logging.error("[%s] error while processing line %s", name, line)
            logging.exception(e)
            continue

        word_count = sum(masks.last_in_word)
        if word_count:
            total_diff += len(error_indexes) / word_count
            total_lines += 1
            total_words += word_count
            total_analysed += analysed
            total_correct += correct

    return dict(
        name=name,
        total_lines=total_lines,
        total_words=total_words,
        total_analysed=total_analysed,
        total_correct=total_correct,
        word_accuracy=total_correct / total_analysed if total_analysed else 0.0,
        avg_diff=total_diff / total_lines if total_lines else 0.0,
        diffed_words=diffed_words,
    )


def main(max_lines: int | None = None, use_context: bool = False):
    LoggingSettings.setup()

    accentuator.build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        reader: Iterator[dict] = csv.DictReader(csv_file, dialect=InputDialect)

        collected = extract_ak_contexts(reader)

        if max_lines:
            collected = islice(collected, max_lines)

        contexts = list(collected)

    logging.info("Collected %d accented lines", len(contexts))

    results = [
        evaluate_lines(contexts, predict, name, use_context)
        for name, predict in build_predictors().items()
    ]

    print(
        f"{'accentuator':<12} {'lines':>8} {'words':>10} "
        f"{'analysed':>10} {'correct':>10} {'word_acc':>9} {'avg_diff':>10}"
    )

    for result in results:
        print(
            f"{result['name']:<12} "
            f"{result['total_lines']:>8} "
            f"{result['total_words']:>10} "
            f"{result['total_analysed']:>10} "
            f"{result['total_correct']:>10} "
            f"{result['word_accuracy']:>9.4f} "
            f"{result['avg_diff']:>10.4f}"
        )

        logging.info(
            "[%s] lines=%d words=%d analysed=%d correct=%d "
            "word_accuracy=%.4f avg_diff=%.4f",
            result["name"],
            result["total_lines"],
            result["total_words"],
            result["total_analysed"],
            result["total_correct"],
            result["word_accuracy"],
            result["avg_diff"],
        )

        for word, count in result["diffed_words"].most_common(20):
            logging.info("[%s] %s: %d", result["name"], word, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare accentuation models on RNC accented poems."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N accented lines",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Prepend and append the previous and next lines as context",
    )
    args = parser.parse_args()

    main(max_lines=args.limit, use_context=args.context)
