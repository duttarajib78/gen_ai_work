"""Structured logging.

One JSON object per line on stdout, which SageMaker ships to CloudWatch Logs (``/aws/sagemaker/Endpoints/<name>``). A request id is attached
automatically through a context variable. Prompts, model output and request bodies are NEVER logged; only identifiers, counts, token usage,
cost and durations. Anything that looks like a credential is redacted from every message.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Optional

from dx_support.text import redact

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("dx_request_id", default="-")

_STD_ATTRS = set(logging.LogRecord("x", 0, "x", 0, "x", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = redact(record.getMessage())
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname, "logger": record.name, "msg": msg, "request_id": request_id_var.get(),
        }
        for k, v in record.__dict__.items():            # fields passed with logger.info(..., extra={...})
            if k not in _STD_ATTRS and not k.startswith("_"):
                out[k] = v
        if record.exc_info and record.exc_info[0] is not None:
            out["exc_type"] = record.exc_info[0].__name__          # the type only: tracebacks can embed request data
        return json.dumps(out, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        base = f"{record.levelname} {record.name}: {redact(record.getMessage())}"
        return base if rid == "-" else f"{base} [request {rid}]"


def configure_logging(level: str = "INFO", json_logs: bool = True, stream=None) -> None:
    """Idempotent: replaces the root handlers so repeated calls (tests, reloads) never duplicate lines."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_request_id(value: Optional[str]) -> contextvars.Token:
    return request_id_var.set(value or "-")
