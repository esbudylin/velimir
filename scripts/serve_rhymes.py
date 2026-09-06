# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "flask>=3.0,<4",
# ]
# ///
import argparse

from velimir.web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the rhyme dataset web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
