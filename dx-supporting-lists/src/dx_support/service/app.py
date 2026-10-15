"""The Flask application: the SageMaker container contract (GET /ping, POST /invocations on port 8080).

Responses
  200  complete (or partial: some families could not finish, see `status` and per-category `error`)
  202  in_progress: work is still running on the instance. Ask again with the same body (results are stored, nothing is recomputed);
       or call InvokeEndpointWithResponseStream with `"stream": true` to wait up to 8 minutes on one connection
  4xx  the request is wrong (400 malformed, 415 not JSON, 413 too large, 422 invalid values, 429 this instance is at its request limit)
  503  the instance is not ready

Streaming (`"stream": true`, via InvokeEndpointWithResponseStream) answers with application/jsonlines: an `accepted` event, `progress` events
about every few seconds (which also keep the connection alive), and a final `result` (or `error`) event.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
import uuid
from typing import Iterator, Optional

from flask import Flask, Response, request
from werkzeug.exceptions import HTTPException

from dx_support.logging_setup import request_id_var, set_request_id
from dx_support.service.runtime import JobHandle, ServiceRuntime
from dx_support.service.schemas import DESCRIBE_KEYS, RequestError, SupportRequest, check_keys, parse_body, parse_support_request, request_id_of

logger = logging.getLogger(__name__)
RETRY_AFTER_S = 20


def _json(status: int, payload: dict, headers: Optional[dict] = None) -> Response:
    resp = Response(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), status=status, mimetype="application/json")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    for k, v in (headers or {}).items():
        resp.headers[k] = v
    return resp


def _error(status: int, code: str, message: str, headers: Optional[dict] = None) -> Response:
    return _json(status, {"error": {"code": code, "message": message}, "request_id": request_id_var.get()}, headers)


class SupportRun:
    """One `supporting_lists` request: answers what is already stored, starts jobs for the rest, and reports on them."""

    def __init__(self, rt: ServiceRuntime, req: SupportRequest):
        self.rt, self.req, self.t0 = rt, req, time.monotonic()
        self.results: dict[str, dict] = {}
        self.handles: dict[str, JobHandle] = {}
        self.cache_hit = True

    def start(self) -> "SupportRun":
        rt, req = self.rt, self.req
        for cat in req.categories:
            if rt.is_cached(cat, req.families):
                self.results[cat] = self._assemble(cat)
                continue
            self.cache_hit = False
            if req.cache_only or not rt.settings.allow_compute:
                self.results[cat] = {"status": "not_cached", "title": rt.universe.categories[cat].title,
                                     "detail": "No stored result for these families, and this request may not start model calls."}
            else:
                self.handles[cat] = rt.submit(cat, req.families)
        return self

    def _assemble(self, cat: str) -> dict:
        r = self.req
        return self.rt.assemble(cat, r.families, r.merge_tiers, r.prosthesis_extract, r.include_audit)

    def pending_futures(self) -> list:
        return [h.future for c, h in self.handles.items() if c not in self.results]

    def collect_finished(self) -> None:
        for cat, h in self.handles.items():
            if cat in self.results or not h.future.done():
                continue
            res = self._assemble(cat)
            p = h.progress
            if p.error:
                res["error"] = p.error
            if res["status"] == "pending":                                  # the job produced nothing for this category
                res["status"] = "failed" if p.outcome in ("failed", "stopped") or p.error else "partial"
            self.results[cat] = res

    def progress_events(self) -> dict:
        return {c: h.progress.snapshot() for c, h in self.handles.items() if c not in self.results}

    def payload(self) -> tuple[dict, int]:
        self.collect_finished()
        results = {}
        for cat in self.req.categories:
            h = self.handles.get(cat)
            results[cat] = self.results.get(cat) or {"status": "in_progress", "title": self.rt.universe.categories[cat].title,
                                                    "progress": h.progress.snapshot() if h else {}}
        statuses = [r["status"] for r in results.values()]
        overall = "in_progress" if "in_progress" in statuses else "complete" if all(s == "complete" for s in statuses) else "partial"
        gates = [h.progress.gate for h in self.handles.values() if h.progress.gate is not None]
        s = self.rt.settings
        out = {"request_id": self.req.request_id, "status": overall, "fingerprint": self.rt.fingerprint, "notice": [s.notice_line_1, s.notice_line_2],
               "results": results,
               "usage": {"cache_hit": self.cache_hit, "llm_calls": sum(g.n_calls for g in gates), "cost_usd": round(sum(g.total_cost for g in gates), 4),
                         "elapsed_ms": int((time.monotonic() - self.t0) * 1000)}}
        if overall == "in_progress":
            out["pending"] = [c for c, r in results.items() if r["status"] == "in_progress"]
            out["retry_after_s"] = RETRY_AFTER_S
        return out, 202 if overall == "in_progress" else 200


def _event(name: str, request_id: str, **fields) -> bytes:
    return (json.dumps({"event": name, "request_id": request_id, **fields}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def stream_events(run: SupportRun) -> Iterator[bytes]:
    req, s = run.req, run.rt.settings
    try:
        yield _event("accepted", req.request_id, fingerprint=run.rt.fingerprint, categories=list(req.categories), families=list(req.families),
                     cache_hit=run.cache_hit, pending=list(run.handles))
        deadline = run.t0 + s.stream_max_s
        while run.pending_futures() and time.monotonic() < deadline:
            concurrent.futures.wait(run.pending_futures(), timeout=s.stream_heartbeat_s, return_when=concurrent.futures.FIRST_COMPLETED)
            run.collect_finished()
            if run.pending_futures():
                yield _event("progress", req.request_id, elapsed_s=round(time.monotonic() - run.t0, 1), categories=run.progress_events())
        payload, _ = run.payload()
        yield _event("result", req.request_id, **{k: v for k, v in payload.items() if k != "request_id"})
    except GeneratorExit:
        raise                                                               # the client went away: the jobs keep running and stay resumable
    except Exception as e:                                                  # noqa: BLE001
        logger.error("stream failed: %s", type(e).__name__)
        yield _event("error", req.request_id, error={"code": "internal_error", "message": "An internal error occurred."})


def create_app(runtime: ServiceRuntime) -> Flask:
    app = Flask("dx_support")
    s = runtime.settings
    app.config["MAX_CONTENT_LENGTH"] = s.max_request_bytes
    app.json.sort_keys = False                                              # type: ignore[attr-defined]
    admission = threading.BoundedSemaphore(s.max_concurrent_requests)

    @app.before_request
    def _begin() -> None:
        set_request_id(uuid.uuid4().hex[:12])

    @app.teardown_request
    def _end(_exc) -> None:
        set_request_id(None)

    @app.get("/ping")
    def ping():
        if runtime.ready:
            return _json(200, {"status": "ok"})
        return _json(503, {"status": "starting", "detail": runtime.init_error or "initialising"})

    @app.post("/invocations")
    def invocations():
        if not runtime.ready:
            return _error(503, "not_ready", "The service is starting or shutting down.", {"Retry-After": "10"})
        if (request.mimetype or "").lower() != "application/json":
            return _error(415, "unsupported_media_type", "Send Content-Type: application/json.")
        if not admission.acquire(blocking=False):
            runtime.counters["rejected_busy"] += 1
            return _error(429, "too_many_requests", "This instance is at its concurrent-request limit. Retry with back-off.", {"Retry-After": "5"})
        released, handed_off, lock = False, False, threading.Lock()

        def release() -> None:                                              # idempotent: called from the view and from response close
            nonlocal released
            with lock:
                if not released:
                    released = True
                    admission.release()
        try:
            body = parse_body(request.get_data(cache=False))
            action = body.get("action", "supporting_lists")
            if action == "describe":
                check_keys(body, DESCRIBE_KEYS)
                set_request_id(request_id_of(body))
                return _json(200, {"request_id": request_id_var.get(), **runtime.snapshot()})
            if action != "supporting_lists":
                raise RequestError(400, "unknown_action", "`action` must be \"supporting_lists\" or \"describe\".")
            req = parse_support_request(body, s, runtime.universe, runtime.leaf_map)
            set_request_id(req.request_id)
            runtime.counters["requests"] += 1
            run = SupportRun(runtime, req).start()
            logger.info("request accepted", extra={"categories": list(req.categories), "families": list(req.families), "cache_hit": run.cache_hit,
                                                   "stream": req.stream})
            if req.stream:
                resp = Response(stream_events(run), mimetype="application/jsonlines")
                resp.headers["Cache-Control"] = "no-store"
                resp.headers["X-Accel-Buffering"] = "no"
                resp.call_on_close(release)
                handed_off = True                                           # the response, not this view, frees the admission slot
                return resp
            deadline = run.t0 + s.sync_deadline_s
            while run.pending_futures() and time.monotonic() < deadline:
                concurrent.futures.wait(run.pending_futures(), timeout=max(0.0, deadline - time.monotonic()), return_when=concurrent.futures.FIRST_COMPLETED)
                run.collect_finished()
            payload, status = run.payload()
            logger.info("request answered", extra={"status": payload["status"], "http": status, "elapsed_ms": payload["usage"]["elapsed_ms"]})
            return _json(status, payload, {"Retry-After": str(RETRY_AFTER_S)} if status == 202 else None)
        except RequestError as e:
            return _error(e.status, e.code, e.message)
        finally:
            if not handed_off:
                release()

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        codes = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}
        return _error(e.code or 500, codes.get(e.code or 500, "http_error"), e.name)

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        logger.error("unhandled error: %s", type(e).__name__)
        return _error(500, "internal_error", "An internal error occurred. Quote the request id when reporting it.")

    return app

