import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .domain_models import InputPoem


DATE_YEAR_FIRST = re.compile(r"^(\d{4})(?:\.(\d{1,2}))?(?:\.(\d{1,2}))?$")
DATE_DAY_FIRST = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")


@dataclass(frozen=True, slots=True)
class CreationDate:
    lower: int
    upper: int | None
    is_exact: bool

    def extract(poem: InputPoem):
        value = poem.created.strip()
        parts = value.split("-")

        if len(parts) > 2:
            raise ValueError(f"Unsupported date range: {poem.created!r}")

        lower = parse_date_part(parts[0])
        upper = parse_date_part(parts[1]) if len(parts) == 2 else None

        return CreationDate(
            lower=lower,
            upper=upper,
            is_exact=not poem.exact.strip(),
        )


def to_timestamp(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def parse_date_part(part: str) -> int:
    match = DATE_YEAR_FIRST.match(part)
    if match:
        year = int(match.group(1))
        month = int(match.group(2)) if match.group(2) else 1
        day = int(match.group(3)) if match.group(3) else 1
        return to_timestamp(year, month, day)

    match = DATE_DAY_FIRST.match(part)
    if match:
        day, month, year = (int(group) for group in match.groups())
        return to_timestamp(year, month, day)

    raise ValueError(f"Unsupported date part: {part!r}")
