# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "flask>=3.0,<4",
#     "gunicorn>=22,<24",
# ]
# ///
import argparse

from gunicorn.app.base import BaseApplication

from web.apps.rhyme import create_app


class WebApplication(BaseApplication):
    def __init__(self, options: dict):
        self.options = options
        super().__init__()

    def load_config(self) -> None:
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the rhyme dataset web UI in production."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    args = parser.parse_args()

    WebApplication(
        {
            "bind": f"{args.host}:{args.port}",
            "workers": args.workers,
        }
    ).run()


if __name__ == "__main__":
    main()
