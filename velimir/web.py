import atexit
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone

from flask import Flask, abort, current_app, g, render_template, request, url_for
from markupsafe import Markup, escape

from velimir.settings import RHYME_DB_PATH

SORT_KEY_RE = re.compile(r"[^а-яёa-z0-9]")

AUTHOR_COLUMNS = [
    ("rhymes", "Слова", "rhymes"),
    ("date", "Дата", "creation_dates.date_low"),
    ("title", "Стихотворение", "sort_key(poems.header)"),
]

SEARCH_COLUMNS = [
    ("word", "Слово", "rhymes.word"),
    ("author", "Автор", "authors.name"),
    ("date", "Дата", "creation_dates.date_low"),
    ("title", "Стихотворение", "sort_key(poems.header)"),
]


def sort_key(text: str) -> str:
    return SORT_KEY_RE.sub("", (text or "").lower())


def timestamp_year(timestamp: int) -> int:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).year


def format_creation_date(date_low: int, date_high: int | None, is_exact: int) -> str:
    low = timestamp_year(date_low)
    high = timestamp_year(date_high) if date_high is not None else None

    if high is not None and high != low:
        text = f"{low}–{high}"
    else:
        text = str(low)

    return text if is_exact else f"≈{text}"


def prepare_database() -> str:
    temp_dir = tempfile.mkdtemp(prefix="velimir_rhymes_")
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    temp_path = os.path.join(temp_dir, "rhyme.db")
    shutil.copyfile(RHYME_DB_PATH, temp_path)

    conn = sqlite3.connect(temp_path)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_poems_author ON poems(author_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rhymes_word ON rhymes(word)")
        conn.commit()
    finally:
        conn.close()

    return temp_path


def resolve_sort(columns, args, default: str) -> tuple[str, str]:
    keys = {key for key, _label, _expr in columns}

    sort = args.get("sort", default)
    if sort not in keys:
        sort = default

    order = args.get("order", "asc")
    if order not in ("asc", "desc"):
        order = "asc"

    return sort, order


def order_expr(columns, sort: str) -> str:
    for key, _label, expr in columns:
        if key == sort:
            return expr

    raise ValueError(f"Unknown sort key: {sort}")


def header_columns(columns, sort: str, order: str, href_builder) -> list[dict]:
    headers = []

    for key, label, _expr in columns:
        next_order = "desc" if (sort == key and order == "asc") else "asc"
        headers.append(
            {
                "key": key,
                "label": label,
                "href": href_builder(key, next_order),
                "arrow": ("▲" if order == "asc" else "▼") if sort == key else "",
            }
        )

    return headers


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.create_function("sort_key", 1, sort_key, deterministic=True)
    return g.db


def close_db(_error=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def create_app() -> Flask:
    app = Flask(__name__)
    app.teardown_appcontext(close_db)
    app.config["DB_PATH"] = prepare_database()

    @app.get("/")
    def index():
        return render_template("index.html", database=RHYME_DB_PATH)

    @app.get("/authors")
    def authors():
        rows = (
            get_db()
            .execute(
                """
            SELECT authors.name AS name,
                   COUNT(DISTINCT rhymes.poem_id || '-' || rhymes.seq
                          || '-' || rhymes.rhyme_group) AS groups
            FROM authors
            JOIN poems ON poems.author_id = authors.ROWID
            JOIN rhymes ON rhymes.poem_id = poems.ROWID
            WHERE rhymes.rhyme_group <> -1
            GROUP BY authors.name
            ORDER BY authors.name
            """
            )
            .fetchall()
        )

        return render_template("authors.html", authors=rows)

    @app.get("/authors/<path:name>")
    def author(name):
        db = get_db()

        if (
            db.execute("SELECT 1 FROM authors WHERE name = ?", (name,)).fetchone()
            is None
        ):
            abort(404)

        sort, order = resolve_sort(AUTHOR_COLUMNS, request.args, "rhymes")

        order_by = f"{order_expr(AUTHOR_COLUMNS, sort)} {order.upper()}"

        rows = db.execute(
            f"""
            SELECT poems.header AS title,
                   creation_dates.date_low AS date_low,
                   creation_dates.date_high AS date_high,
                   creation_dates.is_exact AS is_exact,
                   json_group_array(DISTINCT rhymes.word) AS rhymes
            FROM rhymes
            JOIN poems ON rhymes.poem_id = poems.ROWID
            JOIN authors ON poems.author_id = authors.ROWID
            JOIN creation_dates ON creation_dates.poem_id = poems.ROWID
            WHERE rhymes.rhyme_group <> -1 AND authors.name = ?
            GROUP BY rhymes.poem_id, rhymes.seq, rhymes.rhyme_group
            HAVING COUNT(DISTINCT rhymes.word) > 1
            ORDER BY {order_by}
            """,
            (name,),
        ).fetchall()

        groups = [
            {
                "rhymes": ", ".join(json.loads(row["rhymes"])),
                "title": row["title"],
                "date": format_creation_date(
                    row["date_low"], row["date_high"], row["is_exact"]
                ),
            }
            for row in rows
        ]

        columns = header_columns(
            AUTHOR_COLUMNS,
            sort,
            order,
            lambda key, next_order: url_for(
                "author", name=name, sort=key, order=next_order
            ),
        )

        return render_template(
            "author.html",
            name=name,
            groups=groups,
            columns=columns,
        )

    @app.get("/search")
    def search():
        query = request.args.get("q", "").strip().lower()

        sort, order = resolve_sort(SEARCH_COLUMNS, request.args, "word")

        columns = header_columns(
            SEARCH_COLUMNS,
            sort,
            order,
            lambda key, next_order: url_for(
                "search", q=query, sort=key, order=next_order
            ),
        )

        results = []

        if query:
            db = get_db()

            order_by = f"{order_expr(SEARCH_COLUMNS, sort)} {order.upper()}"

            rows = db.execute(
                f"""
                WITH seqs AS (
                    SELECT poem_id, seq, rhyme_group
                    FROM rhymes
                    WHERE word = ?
                )
                SELECT authors.name AS author,
                       poems.header AS title,
                       creation_dates.date_low AS date_low,
                       creation_dates.date_high AS date_high,
                       creation_dates.is_exact AS is_exact,
                       rhymes.word AS word
                FROM rhymes
                JOIN seqs
                    ON rhymes.poem_id = seqs.poem_id
                    AND seqs.seq = rhymes.seq
                    AND seqs.rhyme_group = rhymes.rhyme_group
                    AND seqs.rhyme_group <> -1
                JOIN poems ON rhymes.poem_id = poems.ROWID
                JOIN authors ON poems.author_id = authors.ROWID
                JOIN creation_dates ON creation_dates.poem_id = poems.ROWID
                WHERE rhymes.word <> ?
                ORDER BY {order_by}
                """,
                (query, query),
            ).fetchall()

            results = [
                {
                    "word": row["word"],
                    "author": Markup('<a href="{}">{}</a>').format(
                        url_for("author", name=row["author"]),
                        escape(row["author"]),
                    ),
                    "date": format_creation_date(
                        row["date_low"], row["date_high"], row["is_exact"]
                    ),
                    "title": row["title"],
                }
                for row in rows
            ]

        return render_template(
            "search.html",
            query=query,
            results=results,
            columns=columns,
        )

    return app
