import logging
from logging import LogRecord


class DelayedLogRecord:
    def __init__(self):
        self.log_record = None

    def create(self, level, message, *args):
        self.log_record = LogRecord(
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


delayed_logger = DelayedLogRecord()
