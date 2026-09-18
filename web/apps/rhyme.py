import atexit
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from enum import Enum

from flask import Flask, abort, current_app, g, render_template, request, url_for
from markupsafe import Markup, escape

from web.rnc_url import build_rnc_url
from velimir.settings import DATASETS_DIRECTORY, RHYME_DB_PATH

SORT_KEY_RE = re.compile(r"[^а-яёa-z0-9]")


class SortKey(str, Enum):
    RHYMES = "rhymes"
    DATE = "date"
    TITLE = "title"
    NAME = "name"
    GROUPS = "groups"
    WORD = "word"
    AUTHOR = "author"


class Order(Enum):
    ASC = "asc"
    DESC = "desc"

    @classmethod
    def from_str(cls, value: str) -> "Order":
        return cls.DESC if value == "desc" else cls.ASC

    @property
    def sql(self) -> str:
        return self.name

    @property
    def opposite(self) -> "Order":
        return Order.DESC if self is Order.ASC else Order.ASC


@dataclass(frozen=True)
class Column:
    key: SortKey
    label: str
    default_order: Order = Order.ASC


AUTHOR_COLUMNS = [
    Column(SortKey.RHYMES, "Слова"),
    Column(SortKey.DATE, "Дата"),
    Column(SortKey.TITLE, "Стихотворение"),
]

AUTHORS_COLUMNS = [
    Column(SortKey.NAME, "Автор"),
    Column(SortKey.GROUPS, "Рифменных групп", default_order=Order.DESC),
]

SEARCH_COLUMNS = [
    Column(SortKey.WORD, "Слово"),
    Column(SortKey.AUTHOR, "Автор"),
    Column(SortKey.DATE, "Дата"),
    Column(SortKey.TITLE, "Стихотворение"),
]

AUTHOR_ORDER_BY = {
    SortKey.RHYMES: "rhymes",
    SortKey.DATE: "date_low",
    SortKey.TITLE: "sort_key(title)",
}

AUTHORS_ORDER_BY = {
    SortKey.NAME: "sort_key",
    SortKey.GROUPS: "groups",
}

SEARCH_ORDER_BY = {
    SortKey.WORD: "rhymes.word",
    SortKey.AUTHOR: "authors.sort_key",
    SortKey.DATE: "date_low",
    SortKey.TITLE: "sort_key(poems.header)",
}


def sort_key(text: str) -> str:
    return SORT_KEY_RE.sub("", (text or "").lower())


def prepare_database() -> str:
    temp_dir = tempfile.mkdtemp(prefix="velimir_rhymes_", dir=DATASETS_DIRECTORY)
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    temp_path = os.path.join(temp_dir, "rhyme.db")
    shutil.copyfile(RHYME_DB_PATH, temp_path)

    conn = sqlite3.connect(temp_path)
    try:
        conn.execute(
            """
            CREATE TABLE creation_dates AS
            SELECT ROWID AS poem_id,
                IF(is_date_exact, '', '≈') ||
                strftime('%Y', datetime(date_low, 'unixepoch')) ||
                CASE
                    WHEN date_high IS NOT NULL
                         AND strftime('%Y', datetime(date_high, 'unixepoch'))
                             <> strftime('%Y', datetime(date_low, 'unixepoch'))
                    THEN '-' || strftime('%Y', datetime(date_high, 'unixepoch'))
                    ELSE ''
                END AS year_range
            FROM poems
        """
        )

        conn.execute(
            """
            CREATE TABLE rhymes_by_author AS
            SELECT poems.header AS title,
                   poems.path AS path,
                   poems.date_low AS date_low,
                   authors.name AS name,
                   authors.sort_key as sort_key,
                   json_group_array(DISTINCT r.word) AS rhymes,
                   creation_dates.year_range
            FROM rhymes r
            JOIN poems ON r.poem_id = poems.ROWID
            JOIN authors ON poems.author_id = authors.ROWID
            JOIN creation_dates ON creation_dates.poem_id = poems.ROWID
            WHERE r.rhyme_group <> -1
            GROUP BY r.poem_id, r.seq, r.rhyme_group
        """
        )

        conn.commit()
    finally:
        conn.close()

    return temp_path


def resolve_sort(order_by, args, default: SortKey) -> tuple[SortKey, Order]:
    try:
        sort = SortKey(args.get("sort", default.value))
    except ValueError:
        sort = default
    if sort not in order_by:
        sort = default

    order = Order.from_str(args.get("order", Order.ASC.value))

    return sort, order


def header_columns(columns, sort: SortKey, order: Order, href_builder) -> list[dict]:
    headers = []

    for col in columns:
        next_order = order.opposite if sort is col.key else col.default_order
        headers.append(
            {
                "key": col.key.value,
                "label": col.label,
                "href": href_builder(col.key.value, next_order.value),
                "arrow": ("▲" if order is Order.ASC else "▼")
                if sort is col.key
                else "",
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
        return render_template("index.html")

    @app.get("/authors")
    def authors():
        sort, order = resolve_sort(AUTHORS_ORDER_BY, request.args, SortKey.NAME)

        rows = (
            get_db()
            .execute(
                f"""
            SELECT name, COUNT(*) AS groups FROM rhymes_by_author
            GROUP BY name
            ORDER BY {AUTHORS_ORDER_BY[sort]} {order.sql}
            """
            )
            .fetchall()
        )

        columns = header_columns(
            AUTHORS_COLUMNS,
            sort,
            order,
            lambda key, next_order: url_for("authors", sort=key, order=next_order),
        )

        return render_template("authors.html", authors=rows, columns=columns)

    @app.get("/authors/<path:name>")
    def author(name):
        db = get_db()

        if (
            db.execute("SELECT 1 FROM authors WHERE name = ?", (name,)).fetchone()
            is None
        ):
            abort(404)

        sort, order = resolve_sort(AUTHOR_ORDER_BY, request.args, SortKey.RHYMES)

        rows = db.execute(
            f"""
            SELECT * FROM rhymes_by_author WHERE name = ?
            ORDER BY {AUTHOR_ORDER_BY[sort]} {order.sql}
            """,
            (name,),
        ).fetchall()

        groups = [
            {
                "rhymes": ", ".join(json.loads(row["rhymes"])),
                "title": Markup('<a href="{}">{}</a>').format(
                    build_rnc_url(row["path"]),
                    escape(row["title"]),
                ),
                "date": row["year_range"],
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

        sort, order = resolve_sort(SEARCH_ORDER_BY, request.args, SortKey.WORD)

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

            order_by = f"{SEARCH_ORDER_BY[sort]} {order.sql}"

            rows = db.execute(
                f"""
                SELECT authors.name AS author,
                    poems.header AS title,
                    poems.path   AS path,
                    poems.date_low AS date_low,
                    creation_dates.year_range AS year_range,
                    rhymes.word  AS word
                FROM rhymes
                JOIN poems   ON poems.ROWID = rhymes.poem_id
                JOIN authors ON authors.ROWID = poems.author_id
                JOIN creation_dates ON creation_dates.poem_id = poems.ROWID
                WHERE (rhymes.poem_id, rhymes.seq, rhymes.rhyme_group) IN (
                    SELECT poem_id, seq, rhyme_group FROM rhymes WHERE word = ?
                )
                AND rhymes.word <> ? AND rhymes.rhyme_group <> -1
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
                    "date": row["year_range"],
                    "title": Markup('<a href="{}">{}</a>').format(
                        build_rnc_url(row["path"]),
                        escape(row["title"]),
                    ),
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
