"""Structured logging that cannot leak episode content by construction.

* One JSON object per line on stdout (SageMaker forwards it to CloudWatch Logs).
* ``log_event`` accepts only an allowlist of field names. Anything else is
  dropped and counted, so a future change cannot accidentally log codes,
  reasoning, prompts, or model output.
* ``admission_no`` is logged only when ``AUDIT_LOG_RAW_IDENTIFIERS=true``.
* Exceptions are logged as type, a short scrubbed message, and stack *frames*
  (no locals, no source lines of data).
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from typing import Any, Optional

from .textutil import clean_line

SAFE_FIELDS = frozenset(
    {
        "request_id", "admission_no", "event", "status", "flagged", "potential_alert", "disposition",
        "error_code", "retryable", "http_status", "mode", "stream", "n_diagnoses", "n_failed_adx",
        "llm_calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "elapsed_ms", "latency_ms", "attempt", "stop_reason", "reason", "size_bytes", "model_id",
        "aws_error_code", "aws_http_status", "aws_request_id", "backoff_s", "phase", "exception_type",
        "exception_message", "stack", "component", "settings", "n_ignored_fields", "max_tokens",
        "routing_scope", "version", "count", "queue_depth", "outcome",
    }
)

_RESERVED = {"ts", "level", "logger", "msg"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": clean_line(record.getMessage(), 500),
        }
        fields = getattr(record, "audit_fields", None)
        if fields:
            payload.update({k: v for k, v in fields.items() if k not in _RESERVED})
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quiet chatty libraries that can include request details at DEBUG.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    log_raw_identifiers: bool = False,
    **fields: Any,
) -> None:
    safe: dict[str, Any] = {"event": event}
    dropped = 0
    for key, value in fields.items():
        if key not in SAFE_FIELDS or value is None:
            if value is not None:
                dropped += 1
            continue
        if key == "admission_no" and not log_raw_identifiers:
            continue
        safe[key] = value
    if dropped:
        safe["dropped_fields"] = dropped
    logger.log(level, event, extra={"audit_fields": safe})


def exception_fields(exc: BaseException, *, max_frames: int = 8) -> dict:
    frames = traceback.extract_tb(exc.__traceback__)[-max_frames:]
    return {
        "exception_type": type(exc).__name__,
        "exception_message": clean_line(str(exc), 200),
        "stack": [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames],
    }


def elapsed_ms(start: float, now: Optional[float] = None) -> int:
    return int(((now if now is not None else time.monotonic()) - start) * 1000)
