import csv

from velimir.acro import detect_acro
from velimir.domain_models import InputPoem
from velimir.io import read_poem_xml
from velimir.parsers import parse_input_lines
from velimir.settings import ACRO_TABLE_PATH, METADATA_TABLE, InputDialect
from web.rnc_url import build_rnc_url


def find_acros():
    with open(METADATA_TABLE, "r", encoding="utf8") as csv_file:
        input_reader = csv.DictReader(csv_file, dialect=InputDialect)

        for row in input_reader:
            poem = InputPoem.from_row(row)
            xml_str = read_poem_xml(poem.path)
            lines, _ = parse_input_lines(xml_str, allow_latin=True)

            if not (4 < len(lines) < 30):
                continue

            first_letters = "".join(line.text.lower().strip()[0] for line in lines)

            if not first_letters:
                continue

            acro = detect_acro(first_letters)

            if acro:
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
                    " ".join(acro),
                    build_rnc_url(poem.path),
                ]
            )


if __name__ == "__main__":
    write_output(find_acros())
