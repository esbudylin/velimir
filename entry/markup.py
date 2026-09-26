import logging
import sys

from velimir.logger import LoggingSettings
from velimir.markup import MarkupEngine, render_xml


def main():
    LoggingSettings.setup()

    text = sys.stdin.read()

    if not text.strip():
        logging.error("No input provided")
        sys.exit(1)

    engine = MarkupEngine()
    result = engine.markup_text(text)

    print(render_xml(result), end="")


if __name__ == "__main__":
    main()
