"""Structured logging: JSON lines carry level, logger, request id and extras."""
from __future__ import annotations

import json
import logging

from app.logging_config import JsonFormatter, RequestIdFilter, TextFormatter, request_id_var


def _record(msg="hello", **extra):
    rec = logging.LogRecord("app.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    RequestIdFilter().filter(rec)
    return rec


def test_json_formatter_emits_structured_fields():
    token = request_id_var.set("rid123")
    try:
        line = JsonFormatter().format(_record(status=200, path="/health"))
    finally:
        request_id_var.reset(token)
    obj = json.loads(line)
    assert obj["msg"] == "hello"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "app.test"
    assert obj["request_id"] == "rid123"
    assert obj["status"] == 200 and obj["path"] == "/health"
    assert obj["ts"].endswith("+00:00")


def test_json_formatter_includes_exception():
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    obj = json.loads(JsonFormatter().format(rec))
    assert "ValueError: bad" in obj["exc"]


def test_json_formatter_never_crashes_on_unserialisable_extras():
    obj = json.loads(JsonFormatter().format(_record(thing=object())))
    assert "object" in obj["thing"]


def test_text_formatter_appends_extras_and_rid():
    line = TextFormatter().format(_record(ms=1.5))
    assert "[-]" in line and line.endswith("ms=1.5")
