# Validates break_into_chunks against the on-disk training corpus.
import argparse
import collections
from dataclasses import dataclass

from velimir.io import load_poems_from_msgpack
from velimir.logger import LoggingSettings
from velimir.meter.ml_preprocess import break_into_chunks


@dataclass(slots=True)
class ChunkFailure:
    path: str
    line_count: int
    reason: str
    detail: str


def validate_poem(
    path: str, lines: list, stanza_breaks: list[int]
) -> ChunkFailure | None:
    try:
        break_into_chunks(lines, stanza_breaks)
    except Exception as error:
        return ChunkFailure(path, len(lines), type(error).__name__, str(error))

    return None


def iter_failures(limit: int):
    total = 0

    for poem in load_poems_from_msgpack():
        total += 1

        if limit and total > limit:
            break

        failure = validate_poem(poem.path, poem.lines, poem.stanza_breaks)

        if failure is not None:
            yield failure

    print("Validated %d poems" % total)


def report(failures) -> int:
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, ChunkFailure] = {}
    total = 0

    for failure in failures:
        total += 1
        counts[failure.reason] += 1
        examples.setdefault(failure.reason, failure)

    if not total:
        print("Chunking validation passed")
        return 0

    print(f"Chunking validation FAILED for {total} poems")
    print("Failure breakdown:")

    for reason, count in counts.most_common():
        example = examples[reason]
        print(f"  {count:8d}  {reason}")
        print(
            f"            e.g. {example.path} ({example.line_count} lines): {example.detail}"
        )

    return total


def main(limit: int) -> None:
    LoggingSettings.setup()

    failed = report(iter_failures(limit))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate poem chunking logic.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max poems to check, 0 = all"
    )
    args = parser.parse_args()

    main(limit=args.limit)
