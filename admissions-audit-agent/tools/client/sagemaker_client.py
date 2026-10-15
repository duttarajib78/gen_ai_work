"""Helpers for calling the endpoint, in regular or streaming mode.

* Regular:   ``InvokeEndpoint``                    (SageMaker cuts it at 60 seconds)
* Streaming: ``InvokeEndpointWithResponseStream``  (up to 8 minutes; NDJSON events)

Both return an ``InvokeResult`` with the HTTP status the container produced and
its JSON body, so callers handle one shape. Retryable outcomes (throttling,
5xx, ``error.retryable``) are retried with jittered exponential backoff; the
audit is read-only, so retrying is always safe.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

NDJSON = "application/x-ndjson"
_RETRYABLE_STATUS = {429, 502, 503, 504}
_RETRYABLE_AWS_CODES = {"ThrottlingException", "ServiceUnavailable", "InternalFailure", "ModelNotReadyException"}


@dataclass(frozen=True)
class InvokeResult:
    http_status: int
    body: dict

    @property
    def ok(self) -> bool:
        return self.http_status == 200 and self.body.get("status") == "completed"

    @property
    def retryable(self) -> bool:
        if self.http_status in _RETRYABLE_STATUS:
            return True
        err = self.body.get("error") if isinstance(self.body, dict) else None
        return bool(isinstance(err, dict) and err.get("retryable"))


def make_runtime_client(region: str, *, streaming: bool = False):
    import boto3
    from botocore.config import Config

    # SageMaker's own limit is 60 s (8 min streaming); leave headroom so the platform answers first.
    config = Config(region_name=region, read_timeout=500 if streaming else 70, connect_timeout=10,
                    retries={"max_attempts": 0, "mode": "standard"})
    return boto3.client("sagemaker-runtime", config=config)


def parse_ndjson(chunks: Iterable[bytes]) -> Iterator[dict]:
    """Yield JSON objects from byte chunks that may split lines anywhere."""
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)
    if buffer.strip():
        yield json.loads(buffer)


def _failure(status: int, code: str, message: str, retryable: bool) -> InvokeResult:
    return InvokeResult(status, {"status": "failed", "error": {"code": code, "message": message, "retryable": retryable}})


def _from_client_error(exc: Exception) -> InvokeResult:
    response = getattr(exc, "response", None) or {}
    code = (response.get("Error") or {}).get("Code", "")
    if code == "ModelError":
        status = int(response.get("OriginalStatusCode") or 500)
        message = response.get("OriginalMessage") or ""
        try:
            body = json.loads(message)
            if isinstance(body, dict):
                return InvokeResult(status, body)
        except ValueError:
            pass
        return _failure(status, "model_error", str(message)[:300], status in _RETRYABLE_STATUS)
    return _failure(503 if code in _RETRYABLE_AWS_CODES else 500, code or type(exc).__name__,
                    str((response.get("Error") or {}).get("Message") or exc)[:300], code in _RETRYABLE_AWS_CODES)


def invoke(runtime: Any, endpoint_name: str, payload: dict, *, stream: bool = False,
           inference_id: Optional[str] = None) -> InvokeResult:
    body = json.dumps(payload).encode("utf-8")
    extra = {"InferenceId": inference_id[:64]} if inference_id else {}
    try:
        if not stream:
            resp = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json",
                                           Accept="application/json", Body=body, **extra)
            return InvokeResult(200, json.loads(resp["Body"].read()))
        resp = runtime.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name, ContentType="application/json", Accept=NDJSON, Body=body, **extra)
        return _consume_stream(resp["Body"])
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "response", None) is not None:
            return _from_client_error(exc)
        if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__module__.startswith(("botocore", "urllib3")):
            return _failure(503, type(exc).__name__, "transport error calling the endpoint", True)
        raise


def _consume_stream(events: Iterable[dict]) -> InvokeResult:
    def chunks() -> Iterator[bytes]:
        for event in events:
            if "PayloadPart" in event:
                yield event["PayloadPart"]["Bytes"]
            elif "ModelStreamError" in event or "InternalStreamFailure" in event:
                raise ConnectionError("the endpoint reported a stream failure")

    final: Optional[dict] = None
    try:
        for message in parse_ndjson(chunks()):
            if message.get("type") == "result":
                final = message
    except (ConnectionError, ValueError):
        return _failure(503, "stream_interrupted", "the response stream failed before a result arrived", True)
    if final is None:
        return _failure(503, "stream_interrupted", "the response stream ended without a result", True)
    return InvokeResult(int(final.get("http_status", 500)), final.get("body") or {})


def invoke_with_retry(
    runtime: Any, endpoint_name: str, payload: dict, *, stream: bool = False, max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep, rand: Callable[[], float] = random.random,
) -> tuple[InvokeResult, int]:
    """Returns (final result, attempts used)."""
    inference_id = str(payload.get("request_id") or "")[:64] or None
    result = _failure(500, "not_attempted", "no attempt was made", False)
    for attempt in range(1, max_attempts + 1):
        result = invoke(runtime, endpoint_name, payload, stream=stream, inference_id=inference_id)
        if not result.retryable or attempt == max_attempts:
            return result, attempt
        sleep(min(30.0, 1.0 * 2 ** (attempt - 1)) * (0.5 + 0.5 * rand()))
    return result, max_attempts
