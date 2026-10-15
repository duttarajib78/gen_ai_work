"""Structured JSON logging with request/job/unit context and secret redaction.

Never log prompts, document text, tool inputs, or request bodies - only identifiers,
sizes, counts, and timings. The redaction filter is a safety net, not a licence to log
sensitive values.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

_log_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("log_ctx", default={})

_SENSITIVE_KEY = re.compile(r"(secret|token|password|passwd|authorization|credential|api[_-]?key|session)", re.I)
_SENSITIVE_TEXT = [
    re.compile(r"(?i)\b(authorization|x-api-key|api[_-]?key|secret|token|password)\b\s*[=:]\s*\S+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
]
_MAX_FIELD_CHARS = 2000


def redact(text: str) -> str:
    for pattern in _SENSITIVE_TEXT:
        text = pattern.sub("***REDACTED***", text)
    return text


def _scrub(value: Any, key: str = "") -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {str(k): _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value][:50]
    if isinstance(value, str):
        text = redact(value)
        return text if len(text) <= _MAX_FIELD_CHARS else text[:_MAX_FIELD_CHARS] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(str(value))[:_MAX_FIELD_CHARS]


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach identifiers (request_id, job_id, unit_id, ...) to every log record in this scope."""
    merged = {**_log_ctx.get(), **{k: v for k, v in fields.items() if v is not None}}
    token = _log_ctx.set(merged)
    try:
        yield
    finally:
        _log_ctx.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage())[:_MAX_FIELD_CHARS],
        }
        payload.update(_scrub(_log_ctx.get()))
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(_scrub(extra))
        if record.exc_info:
            etype, evalue, _ = record.exc_info
            payload["exc_type"] = getattr(etype, "__name__", str(etype))
            payload["exc_message"] = redact(str(evalue))[:_MAX_FIELD_CHARS]
            payload["exc"] = redact(self.formatException(record.exc_info))[:4000]
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, json_logs: bool = True, stream: Any = None) -> None:
    """Idempotent root logger setup (stdout is what SageMaker ships to CloudWatch Logs)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("botocore", "boto3", "urllib3", "claude_agent_sdk", "mcp", "pdfminer", "pdfplumber"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log ``message`` with structured ``fields`` (scrubbed by the formatter)."""
    logger.log(level, message, extra={"fields": fields})
