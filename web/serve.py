import argparse

from web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Velimir web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    args = parser.parse_args()

    create_app(debug=True).run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
