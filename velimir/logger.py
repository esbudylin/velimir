import logging
import os
from dataclasses import asdict, dataclass


class DelayedLogRecord:
    def __init__(self):
        self.log_record = None

    def create(self, level, message, *args):
        self.log_record = logging.LogRecord(
            name="delayed_logger",
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=args,
            exc_info=None,
        )

    def record(self):
        if self.log_record:
            logging.getLogger(self.log_record.name).handle(self.log_record)

        self.log_record = None


@dataclass
class LoggingSettings:
    filename: str = "main.log"
    encoding: str = "utf-8"
    level: int = logging.INFO
    filemode: str = "w"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"

    @classmethod
    def setup(cls):
        config = asdict(cls())

        if log_file := os.environ.get("LOG_FILE"):
            config["filename"] = log_file

        if log_dir := os.path.dirname(config["filename"]):
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(**config)

        # supress stanza logging to stdout
        import stanza

        stanza.logger.removeHandler(stanza.log_handler)


delayed_logger = DelayedLogRecord()
