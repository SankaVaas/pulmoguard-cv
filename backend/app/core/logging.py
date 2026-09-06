"""
Structured logging setup.

Emits JSON-formatted log lines in production (machine-parseable by log
aggregators like CloudWatch/Datadog/ELK) and human-readable lines in
development. Configured once at app startup via `configure_logging()`.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from app.core.config import get_settings

settings = get_settings()


class JsonFormatter(logging.Formatter):
    """Renders log records as single-line JSON for production log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """Configure the root logger once at application startup.

    Idempotent: safe to call multiple times (e.g. under pytest and uvicorn
    reload) without duplicating handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    if settings.ENVIRONMENT == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Quiet noisy third-party loggers unless we're actively debugging them.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
