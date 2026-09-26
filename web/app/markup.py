import threading

from flask import Blueprint, Response, current_app, render_template, request

from velimir.markup import MarkupEngine, MarkupResult, render_xml

bp = Blueprint("markup", __name__)

MAX_TEXT_LENGTH = 10000

_engine_lock = threading.Lock()


@bp.app_context_processor
def inject_limits() -> dict:
    return {"max_text_length": MAX_TEXT_LENGTH}


def get_engine() -> MarkupEngine:
    if "markup_engine" not in current_app.extensions:
        with _engine_lock:
            if "markup_engine" not in current_app.extensions:
                current_app.extensions["markup_engine"] = MarkupEngine()

    return current_app.extensions["markup_engine"]


def run_markup(text: str) -> tuple[MarkupResult | None, str | None]:
    if not text.strip():
        return None, "Введите текст стихотворения"

    if len(text) > MAX_TEXT_LENGTH:
        return None, f"Текст слишком длинный: максимум {MAX_TEXT_LENGTH} символов"

    try:
        return get_engine().markup_text(text), None
    except Exception:
        current_app.logger.exception("Failed to markup the poem")
        return None, "Не удалось разметить стихотворение"


@bp.get("/")
def index():
    return render_template("markup.html")


@bp.post("/")
def process():
    text = request.form.get("text", "")
    result, error = run_markup(text)

    return render_template("markup.html", text=text, result=result, error=error)


@bp.post("/download")
def download():
    text = request.form.get("text", "")
    result, error = run_markup(text)

    if error:
        return render_template("markup.html", text=text, error=error)

    return Response(
        render_xml(result),
        mimetype="application/xml",
        headers={"Content-Disposition": "attachment; filename=markup.xml"},
    )
