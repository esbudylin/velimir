from flask import Flask

from web.app import markup, rhyme


def create_app(debug: bool = False) -> Flask:
    app = Flask(__name__)
    app.debug = debug

    app.register_blueprint(rhyme.bp)
    app.register_blueprint(markup.bp, url_prefix="/markup")

    return app
