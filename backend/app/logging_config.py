"""Logging setup: human-readable text or one-JSON-object-per-line.

``LOG_FORMAT=json`` is meant for containers / log shippers; ``text`` (default)
for local dev. Every record carries the current request id (``-`` outside a
request), taken from a context variable the HTTP middleware sets, so log lines
emitted deep inside services correlate with the request that caused them.
Structured fields passed via ``extra={...}`` are emitted as top-level JSON keys.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Attributes every LogRecord has; anything else came from ``extra=`` and is
# emitted as a structured field.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "request_id",
    "color_message",  # uvicorn's ANSI-coloured duplicate of msg
}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_var.get()),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """``ts LEVEL logger [rid] message key=value ...``"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        base = super().format(record)
        extras = " ".join(
            f"{k}={v}"
            for k, v in vars(record).items()
            if k not in _RESERVED and not k.startswith("_")
        )
        return f"{base} {extras}" if extras else base


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install a single stdout handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through the same handler/format. Access logs
    # are dropped: the request middleware already logs every request with its
    # id and latency.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    # Third-party chatter (per-request httpx lines, chroma telemetry) is noise at INFO.
    for name in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
