"""Structured logging: one JSON object per line, with the request's identity on every line.

Why JSON: a log aggregator can filter and chart `latency_ms` or `status` only if they are
fields, not words inside a sentence. Why the guard against sensitive fields: see
docs/logging.md, "What is never logged".
"""

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from app.observability.context import client_var, request_id_var

LogFormat = Literal["json", "text"]

REDACTED = "[redacted]"

# Field names whose values must never reach a log. This is the second line of defence: the code
# is written not to pass such values at all, and this list catches a slip (a debugging line
# added in a hurry) before it becomes a leak. Matching ignores case and "-" versus "_".
SENSITIVE_FIELDS = frozenset(
    {
        "prompt",
        "messages",
        "content",
        "completion",
        "reply",
        "text",
        "body",
        "response_body",
        "api_key",
        "x_api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "cookie",
        "idempotency_key",
    }
)

_STANDARD_ATTRIBUTES = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    # uvicorn adds a copy of its message with terminal colour codes: noise in a log file
    "color_message",
}


def _is_sensitive(name: str) -> bool:
    return name.lower().replace("-", "_") in SENSITIVE_FIELDS


def redact(name: str, value: Any) -> Any:
    if _is_sensitive(name):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(k): redact(str(k), v) for k, v in value.items()}
    return value


def install_context_in_records() -> None:
    """Stamp request_id and client onto every log record when it is created.

    A record factory rather than a handler filter: it applies to records whichever handler
    ends up writing them, including the ones tests attach.
    """
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_carries_request_context", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = request_id_var.get()
        record.client = client_var.get()
        return record

    factory._carries_request_context = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: redact(key, value)
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRIBUTES and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # None values are dropped: a line without request_id is outside any request
        payload.update({k: v for k, v in _extras(record).items() if v is not None})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Readable lines for a terminal: the same fields, as key=value after the message."""

    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in _extras(record).items() if v is not None)
        when = self.formatTime(record, "%H:%M:%S")
        line = f"{when} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if fields:
            line += f"  {fields}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", fmt: LogFormat = "json") -> None:
    """Replace the root handler and route uvicorn's loggers through it.

    Without this the application's loggers have no handler and a WARNING level: INFO lines
    are dropped and warnings come out unformatted.
    """
    install_context_in_records()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn = logging.getLogger(name)
        uvicorn.handlers.clear()
        uvicorn.propagate = True
    # Our request line replaces uvicorn's access log: same facts, plus the request ID
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
