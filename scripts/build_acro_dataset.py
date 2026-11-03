import csv
import logging
import re

from velimir.acro import detect_acro
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.logger import LoggingSettings
from velimir.parsers import parse_input_lines
from velimir.settings import ACRO_TABLE_PATH, METADATA_TABLE, InputDialect
from web.rnc_url import build_rnc_url


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

            acro = "; ".join(detect_acro(first_letters))

            if acro:
                logging.info("Acro found %s, %s", poem.path, acro)
                yield poem, acro


def write_output(acros):
    with open(ACRO_TABLE_PATH, "w", encoding="utf8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["author", "path", "created", "acro", "corpus_url"])

        for poem, acro in acros:
            writer.writerow(
                [
                    poem.author,
                    poem.path,
                    poem.created,
                    acro,
                    build_rnc_url(poem.path),
                ]
            )


if __name__ == "__main__":
    LoggingSettings.setup()
    write_output(find_acros())
