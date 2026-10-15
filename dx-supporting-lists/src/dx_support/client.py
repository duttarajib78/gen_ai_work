"""A small reference client for callers of the endpoint (boto3 is imported lazily; pass your own `client` in tests).

    from dx_support.client import EndpointClient
    c = EndpointClient("prod-dx-support", region="ap-southeast-2")
    result = c.get_lists(categories=["H25"], families=["achi", "mbs"])        # waits (streaming) until the lists exist, up to `timeout_s`
    for item in result["results"]["H25"]["families"]["achi"]["items"]: ...

It handles the three things a raw InvokeEndpoint call does not: a 202 (work still running -> keep waiting on a stream, then resume), 429 (this
instance is busy -> back off), and the 60-second limit of the regular API (it uses the 8-minute streaming API for anything not already cached).
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Iterator, Optional


class EndpointError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, code: str = ""):
        super().__init__(message)
        self.status, self.code = status, code


def iter_ndjson(event_stream: Any) -> Iterator[dict]:
    """Re-assembles JSON lines from the PayloadPart events of InvokeEndpointWithResponseStream (a part can end mid-line)."""
    buf = b""
    for event in event_stream:
        part = event.get("PayloadPart")
        if not part:
            continue
        buf += part.get("Bytes", b"")
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)
    if buf.strip():
        yield json.loads(buf)


def _model_error_status(exc: BaseException) -> Optional[int]:
    resp = getattr(exc, "response", None) or {}
    return resp.get("OriginalStatusCode") or (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")


class EndpointClient:
    def __init__(self, endpoint_name: str, region: Optional[str] = None, client: Any = None, sleep=time.sleep):
        self.endpoint_name = endpoint_name
        self._sleep = sleep
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("sagemaker-runtime", region_name=region, config=Config(read_timeout=70, retries={"max_attempts": 0}))
        self.client = client

    # ------------------------------------------------------------------ raw calls
    def _invoke(self, payload: dict) -> tuple[int, dict]:
        try:
            r = self.client.invoke_endpoint(EndpointName=self.endpoint_name, ContentType="application/json", Accept="application/json",
                                            Body=json.dumps(payload).encode("utf-8"))
        except Exception as e:                                                   # noqa: BLE001
            status = _model_error_status(e)
            if status is None:
                raise
            raise EndpointError(f"endpoint returned HTTP {status}: {str(e)[:300]}", status=status) from e
        body = json.loads(r["Body"].read())
        return int(r.get("ResponseMetadata", {}).get("HTTPStatusCode", 200)), body

    def describe(self) -> dict:
        return self._invoke({"action": "describe"})[1]

    def stream(self, payload: dict) -> Iterator[dict]:
        r = self.client.invoke_endpoint_with_response_stream(EndpointName=self.endpoint_name, ContentType="application/json", Accept="application/jsonlines",
                                                             Body=json.dumps({**payload, "stream": True}).encode("utf-8"))
        yield from iter_ndjson(r["Body"])

    # ------------------------------------------------------------------ the convenient call
    def get_lists(self, categories: Optional[list[str]] = None, codes: Optional[list[str]] = None, families: Optional[list[str]] = None,
                  merge_tiers: Optional[list[str]] = None, include_audit: bool = False, cache_only: bool = False, timeout_s: float = 900.0,
                  on_progress=None) -> dict:
        payload: dict = {"action": "supporting_lists", "include_audit": include_audit, "cache_only": cache_only}
        for k, v in (("categories", categories), ("codes", codes), ("families", families), ("merge_tiers", merge_tiers)):
            if v:
                payload[k] = v
        deadline = time.monotonic() + timeout_s
        busy_attempts = 0
        streaming = False
        while True:
            try:
                if not streaming:
                    status, body = self._invoke(payload)
                    if status == 200 or body.get("status") in ("complete", "partial"):
                        return body
                    streaming = not cache_only                           # 202: switch to the 8-minute streaming API for the wait
                else:
                    last: Optional[dict] = None
                    for ev in self.stream(payload):
                        if ev.get("event") == "progress" and on_progress:
                            on_progress(ev)
                        elif ev.get("event") == "result":
                            last = ev
                        elif ev.get("event") == "error":
                            raise EndpointError(ev.get("error", {}).get("message", "stream error"), code=ev.get("error", {}).get("code", ""))
                    if last is not None and last.get("status") != "in_progress":
                        return {k: v for k, v in last.items() if k != "event"}
                    # the stream ended (8-minute cap) with work still running: reconnect -- everything finished so far is stored
            except EndpointError as e:
                if e.status not in (429, 503):
                    raise                                                # a request the endpoint rejected will be rejected again
                busy_attempts += 1                                       # this instance is busy or starting: back off and retry
                self._sleep(min(30.0, 2.0 ** busy_attempts) * random.uniform(0.8, 1.2))
            if time.monotonic() >= deadline:
                raise EndpointError(f"timed out after {timeout_s:.0f}s; the work continues on the endpoint, so calling again resumes it")
