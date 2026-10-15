"""Flask application implementing the SageMaker real-time inference contract.

  GET|POST /ping         readiness/liveness (must answer within 2 s; reflects local process health only)
  POST     /invocations  JSON envelope {"operation": ..., ...}; every operation returns in milliseconds because
                         long-running extraction work is queued (SageMaker real-time requests are capped at 60 s)

Errors are JSON with a stable ``code``; unexpected errors never leak internals (only a request id to correlate with logs).
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from ..domain.api import Envelope
from ..errors import RuleExtractorError
from ..logging_setup import event, log_context
from ..settings import ALL_OPERATIONS
from .container import ServiceContainer
from .handlers import dispatch

log = logging.getLogger(__name__)
_ATTR_SAFE = re.compile(r"[^A-Za-z0-9_.:=\- ]")


def create_app(container: ServiceContainer) -> Flask:
    app = Flask("rule_extractor")
    app.config["MAX_CONTENT_LENGTH"] = container.settings.service.max_request_bytes
    app.json.sort_keys = False

    def _error(status: int, code: str, message: str, details: list[str] | None = None):
        body: dict[str, Any] = {"ok": False, "request_id": getattr(g, "request_id", None), "error": {"code": code, "message": message}}
        if details:
            body["error"]["details"] = details
        resp = jsonify(body)
        resp.status_code = status
        return resp

    @app.before_request
    def _begin() -> None:
        g.request_id = uuid.uuid4().hex[:16]
        g.started = time.monotonic()
        attrs = request.headers.get("X-Amzn-SageMaker-Custom-Attributes", "")
        g.client_attrs = _ATTR_SAFE.sub("", attrs)[:128]

    @app.after_request
    def _end(resp):
        resp.headers["X-Amzn-SageMaker-Custom-Attributes"] = f"request_id={getattr(g, 'request_id', '')}"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/ping", methods=["GET", "POST"])
    def ping():
        ok, detail = container.health()
        resp = jsonify({"status": "ok" if ok else "unhealthy", **detail})
        resp.status_code = 200 if ok else 503
        return resp

    @app.post("/invocations")
    def invocations():
        ctype = (request.content_type or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return _error(415, "unsupported_media_type", "Content-Type must be application/json")
        try:
            payload = json.loads(request.get_data(cache=False, as_text=False))
        except (ValueError, UnicodeDecodeError):
            return _error(400, "invalid_json", "request body is not valid JSON")
        envelope = Envelope.from_dict(payload, known_operations=ALL_OPERATIONS)
        with log_context(request_id=g.request_id, operation=envelope.operation, client_attrs=g.client_attrs or None):
            data = dispatch(container, envelope.operation, envelope.body)
            event(log, logging.INFO, "invocation ok", duration_ms=round((time.monotonic() - g.started) * 1000, 1))
        return jsonify({"ok": True, "operation": envelope.operation, "request_id": g.request_id, "data": data})

    @app.errorhandler(RuleExtractorError)
    def _known(exc: RuleExtractorError):
        with log_context(request_id=getattr(g, "request_id", None)):
            event(log, logging.INFO if exc.http_status < 500 else logging.ERROR, "invocation rejected",
                  code=exc.code, status=exc.http_status)
        return _error(exc.http_status, exc.code, exc.message, exc.details)

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        code = {413: "payload_too_large", 404: "not_found", 405: "method_not_allowed"}.get(exc.code or 500, "http_error")
        return _error(exc.code or 500, code, exc.name)

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        log.exception("unhandled error", extra={"fields": {"request_id": getattr(g, "request_id", None)}})
        return _error(500, "internal_error", "internal error; quote the request_id when reporting it")

    return app
