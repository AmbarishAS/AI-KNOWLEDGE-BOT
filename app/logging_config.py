"""Structured JSON logging with request-id correlation and secret redaction.

Why structured logs? In an enterprise system you want machine-parseable logs that carry a
correlation id across the ingestion/retrieval/generation pipeline (see
``docs/OBSERVABILITY`` notes). We also redact obvious secrets so API keys never leak.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Correlation id propagated per-request (set in API middleware).
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{8,})"),
    re.compile(r"(api[_-]?key\"?\s*[:=]\s*\"?)([^\s\"]+)", re.IGNORECASE),
    re.compile(r"(bearer\s+)([A-Za-z0-9._-]+)", re.IGNORECASE),
]


def redact(text: str) -> str:
    """Mask common secret shapes in a string."""
    redacted = _SECRET_PATTERNS[0].sub("sk-***REDACTED***", text)
    redacted = _SECRET_PATTERNS[1].sub(r"\1***REDACTED***", redacted)
    redacted = _SECRET_PATTERNS[2].sub(r"\1***REDACTED***", redacted)
    return redacted


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        # Attach structured extras passed via logger.info(..., extra={"extra": {...}})
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(dict(extra))
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        prefix = f"[{rid}] " if rid else ""
        return f"{record.levelname:<7} {record.name}: {prefix}{redact(record.getMessage())}"


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure the root logger. Idempotent."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
