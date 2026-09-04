"""Structured logging built on the standard library.

Two things matter for this service:

1. Every log line can be correlated to the HTTP request that produced it, via
   a ``request_id`` carried in a :class:`~contextvars.ContextVar`. That id also
   travels back to the caller in the ``X-Request-ID`` response header, so a
   user report can be traced through the API and, in later phases, through the
   AI layer, the weather providers and the datastore.
2. Deployed environments emit machine-parseable JSON while local development
   stays readable.

No third-party logging dependency is required for either.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Final

from app.config.settings import LogFormat, Settings

#: Correlation id for the request currently being handled, if any.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Header used to accept and echo the correlation id.
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

#: Placeholder used when a log record is emitted outside a request.
NO_REQUEST_ID: Final[str] = "-"

_CONSOLE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] | %(message)s"
)

# Attributes present on every LogRecord; anything else was supplied by the
# caller through ``extra=`` and is treated as structured context.
_STANDARD_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "request_id",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class RequestIdFilter(logging.Filter):
    """Attach the active correlation id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or NO_REQUEST_ID
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", NO_REQUEST_ID),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


def _build_handler(settings: Settings) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if settings.LOG_FORMAT is LogFormat.JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(_CONSOLE_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z")
        )
    return handler


def configure_logging(settings: Settings) -> None:
    """Install the application logging configuration.

    Idempotent: existing root handlers are replaced, so calling this more than
    once (for example when a test builds several app instances) will not
    duplicate output.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    root.addHandler(_build_handler(settings))
    root.setLevel(settings.LOG_LEVEL)

    # Let uvicorn's loggers flow through the root handler so every line shares
    # one format. Access logging is handled by our own middleware, which also
    # records the correlation id and duration, so uvicorn's version is muted.
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = True
    access_logger.setLevel(logging.WARNING)

    # httpx logs an INFO line per outbound call. Our own provider and ingestion
    # logs already record those with correlation ids, so this is pure noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Thin wrapper kept for a single import site."""
    return logging.getLogger(name)
