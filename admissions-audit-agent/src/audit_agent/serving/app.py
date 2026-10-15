"""Flask app implementing the SageMaker hosting contract.

* ``GET  /ping``         200 once start-up finished (never calls Bedrock, so a
                         model outage cannot take every instance out of service)
* ``POST /invocations``  one audit per request

Two response modes share this route:

* **Regular** (``InvokeEndpoint``): the whole answer as one JSON document. SageMaker
  cuts invocations at 60 seconds, so the audit gets ``AUDIT_REQUEST_DEADLINE_S``
  (default 50) and a structured ``deadline_exceeded`` response is returned before
  the platform would cut the connection.
* **Streaming** (``InvokeEndpointWithResponseStream``): send ``Accept:
  application/x-ndjson`` or ``options.stream=true``. The platform allows up to 8
  minutes, so the audit gets ``AUDIT_STREAM_DEADLINE_S`` (default 420). The body is
  newline-delimited JSON: ``accepted``, periodic ``heartbeat``, then one ``result``
  event carrying the HTTP status and the same body a regular call would return.

HTTP status says who must act: 4xx the caller, 5xx the platform or a dependency
(retry later), 200 the request was processed (inspect ``status``). Request bodies
are never logged.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Optional

from flask import Flask, Response, request, stream_with_context
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from ..config import Settings
from ..deadline import Deadline
from ..errors import (
    AuditError, OverloadedError, PayloadTooLargeError, RequestValidationError, UnsupportedMediaTypeError,
)
from ..logging_utils import configure_logging, exception_fields, log_event
from ..service import AuditService, build_service

logger = logging.getLogger(__name__)

NDJSON = "application/x-ndjson"
_STREAM_GRACE_S = 1.0


def _reject_constant(name: str) -> Any:
    raise ValueError(f"invalid JSON constant {name}")


def _reject_duplicates(pairs: list) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key in JSON object")
        out[key] = value
    return out


def parse_json_body(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise RequestValidationError("request body must be UTF-8 encoded JSON") from None
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates)
    except (ValueError, RecursionError):
        raise RequestValidationError(
            "request body is not valid JSON (NaN, Infinity, duplicate keys, and excessive nesting are rejected)"
        ) from None


def _dumps(body: Any) -> str:
    return json.dumps(body, separators=(",", ":"), default=str)


@dataclass
class _State:
    settings: Settings
    service: AuditService
    executor: ThreadPoolExecutor
    gate: threading.BoundedSemaphore


def create_app(
    settings: Optional[Settings] = None,
    service: Optional[AuditService] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Flask:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    service = service or build_service(settings)
    state = _State(
        settings=settings,
        service=service,
        executor=executor or ThreadPoolExecutor(max_workers=settings.max_concurrent_requests, thread_name_prefix="audit"),
        gate=threading.BoundedSemaphore(settings.max_concurrent_requests),
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes
    app.extensions["audit"] = state
    log_event(logger, "service_started", component="serving", settings=settings.describe())

    # -- helpers ---------------------------------------------------------------
    def respond(body: Any, status: int, request_id: Optional[str] = None) -> Response:
        response = Response(_dumps(body), status=status, mimetype="application/json")
        if request_id:
            response.headers["X-Request-Id"] = request_id
        return response

    def error_response(exc: AuditError) -> Response:
        return respond({"request_id": None, "status": "failed", "error": exc.to_error_body()}, exc.http_status)

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # -- routes ----------------------------------------------------------------
    @app.get("/ping")
    def ping() -> Response:
        return respond({"status": "ok"}, 200)

    @app.post("/invocations")
    def invocations() -> Response:
        if (request.mimetype or "").lower() != "application/json":
            raise UnsupportedMediaTypeError("Content-Type must be application/json")
        areq = state.service.validate(parse_json_body(request.get_data(cache=False)))

        wants_stream = areq.stream or NDJSON in (request.headers.get("Accept") or "").lower()
        limit = state.settings.stream_deadline_s if wants_stream else state.settings.request_deadline_s
        deadline = Deadline(min(areq.deadline_s, limit) if areq.deadline_s else limit)

        if not state.gate.acquire(blocking=False):
            raise OverloadedError("the service is at capacity; retry shortly")
        try:
            future: Future = state.executor.submit(state.service.audit, areq, deadline)
        except Exception:
            state.gate.release()
            raise
        future.add_done_callback(lambda _f: state.gate.release())  # capacity is held until the work ends

        if not wants_stream:
            try:
                result = future.result(timeout=deadline.remaining() + _STREAM_GRACE_S)
            except FutureTimeout:
                deadline.cancel()
                result = state.service.deadline_exceeded_result(areq, deadline)
            return respond(result.body, result.status_code, areq.request_id)

        def events():
            try:
                yield _dumps({"type": "accepted", "request_id": areq.request_id}) + "\n"
                while True:
                    try:
                        result = future.result(timeout=state.settings.stream_heartbeat_s)
                    except FutureTimeout:
                        if deadline.elapsed() > deadline.budget_s + _STREAM_GRACE_S:
                            deadline.cancel()
                            result = state.service.deadline_exceeded_result(areq, deadline)
                        else:
                            yield _dumps({"type": "heartbeat", "elapsed_ms": int(deadline.elapsed() * 1000)}) + "\n"
                            continue
                    yield _dumps({"type": "result", "http_status": result.status_code, "body": result.body}) + "\n"
                    return
            finally:
                if not future.done():  # the client disconnected: stop starting new work
                    deadline.cancel()

        response = Response(stream_with_context(events()), status=200, mimetype=NDJSON)
        response.headers["X-Request-Id"] = areq.request_id
        response.headers["X-Accel-Buffering"] = "no"
        return response

    # -- errors ----------------------------------------------------------------
    @app.errorhandler(AuditError)
    def on_audit_error(exc: AuditError) -> Response:
        return error_response(exc)

    @app.errorhandler(RequestEntityTooLarge)
    def on_too_large(_exc: RequestEntityTooLarge) -> Response:
        return error_response(PayloadTooLargeError(f"request body exceeds {state.settings.max_request_bytes} bytes"))

    @app.errorhandler(HTTPException)
    def on_http_exception(exc: HTTPException) -> Response:
        body = {"request_id": None, "status": "failed",
                "error": {"code": f"http_{exc.code}", "message": exc.name, "retryable": False}}
        return respond(body, exc.code or 500)

    @app.errorhandler(Exception)
    def on_unexpected(exc: Exception) -> Response:
        log_event(logger, "unexpected_error", level=logging.ERROR, phase="http", **exception_fields(exc))
        return error_response(AuditError("an unexpected internal error occurred"))

    return app
