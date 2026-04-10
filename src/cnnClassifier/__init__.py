import logging
import os
import sys
from logging.handlers import RotatingFileHandler

PROJECT_AUTHOR = "Pratyush Mishra"
DEFAULT_LOG_DIR = "/tmp/logs" if os.getenv("VERCEL") else "logs"
LOG_DIR = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

try:
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError:
    LOG_DIR = "/tmp/logs"
    LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")
    os.makedirs(LOG_DIR, exist_ok=True)


class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        if not hasattr(record, "author"):
            record.author = PROJECT_AUTHOR
        return True


def configure_logging() -> logging.Logger:
    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(module)s | "
        "author=%(author)s | trace_id=%(trace_id)s | %(message)s"
    )

    formatter = logging.Formatter(log_format)
    trace_filter = TraceContextFilter()

    file_handler = RotatingFileHandler(
        LOG_FILEPATH,
        maxBytes=int(os.getenv("LOG_MAX_BYTES", "1048576")),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(trace_filter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(trace_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
    else:
        for handler in root_logger.handlers:
            handler.addFilter(trace_filter)

    project_logger = logging.getLogger("cnnClassifierLogger")
    project_logger.setLevel(LOG_LEVEL)
    project_logger.propagate = True
    return project_logger


logger = configure_logging()
logger.info("Logging initialized for Chest Cancer Classifier.")
