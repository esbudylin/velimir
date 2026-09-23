import argparse
import csv
import logging
import random
import time
from dataclasses import dataclass

from velimir import accentuator
from velimir.logger import LoggingSettings
from velimir.onnx import load_rhyme_onnx_model
from velimir.rhyme_identifier import (
    RhymeInput,
    identify_rhyme_schema,
    render_rhyme_formulas,
)
from velimir.domain_models import InputPoem
from velimir.settings import METADATA_TABLE, InputDialect
from velimir.parsers import parse_input_lines
from velimir.io import read_poem_xml

MAX_POEM_LINES = 20


@dataclass
class Comparison:
    path: str
    annotation: str
    ours: str
    string_outcome: str


def sample_rows(sample_size: int, seed: int) -> list[InputPoem]:
    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        rows = [
            InputPoem.from_row(row)
            for row in csv.DictReader(csv_file, dialect=InputDialect)
            if row["rhyme"].strip()
        ]

    random.Random(seed).shuffle(rows)

    return rows[:sample_size]


def compare_poem(row: InputPoem, lines, model) -> Comparison:
    annotation = row.rhyme.strip()

    ri = []

    for line in lines:
        if line.rhyme_zone:
            rz = line.rhyme_zone
        else:
            rz = line.text.split()[-1]

        am = accentuator.extract_accent_mask(rz)
        ri.append(RhymeInput(rz, am))

    try:
        ours = render_rhyme_formulas(identify_rhyme_schema(ri, model))
    except Exception:
        logging.exception("Can't identify rhyme schema for %s", row.path)
        return Comparison(row.path, annotation, "", "processing_error")

    return Comparison(
        row.path,
        annotation,
        ours,
        "match" if ours == annotation else "mismatch",
    )


def iter_comparisons(rows: list[InputPoem]):
    model = load_rhyme_onnx_model()

    for row in rows:
        logging.info("Processing poem: %s", row.path)
        started = time.monotonic()

        xml = read_poem_xml(row.path)
        lines, _ = parse_input_lines(xml)

        if len(lines) > MAX_POEM_LINES:
            logging.info(
                "Skipping large poem (%d > %d lines): %s",
                len(lines),
                MAX_POEM_LINES,
                row.path,
            )
            continue

        comparison = compare_poem(row, lines, model)

        logging.info(
            "Finished poem: %s (%s) in %.2fs",
            row.path,
            comparison.string_outcome,
            time.monotonic() - started,
        )

        yield comparison


def write_comparisons(comparisons, output: str) -> tuple[int, int]:
    matched = 0
    total = 0

    with open(output, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["path", "annotation", "ours", "string_outcome"])

        for comparison in comparisons:
            total += 1

            if comparison.string_outcome == "match":
                matched += 1

            writer.writerow(
                [
                    comparison.path,
                    comparison.annotation,
                    comparison.ours,
                    comparison.string_outcome,
                ]
            )

    return matched, total


def main(sample_size: int, seed: int, output: str):
    LoggingSettings.setup()

    rows = sample_rows(sample_size, seed)

    matched, total = write_comparisons(iter_comparisons(rows), output)

    percent = matched / total * 100 if total else 0.0

    print(f"Correctly identified: {matched}/{total} ({percent:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate extracted rhyme schemas.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--output",
        default="data/rhyme_validation_mismatches.csv",
        help="Where to write mismatching texts",
    )
    args = parser.parse_args()

    main(sample_size=args.sample_size, seed=args.seed, output=args.output)
