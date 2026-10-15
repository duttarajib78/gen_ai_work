"""Amazon Bedrock client for Claude Sonnet 5.5 (Converse API, answer via tool call).

Request shape, and why (Sonnet 5.5 behaviour as documented by AWS and Anthropic):

* ``toolChoice`` is ``auto``. Forced tool use (``any`` / a named tool) is
  rejected by this model with HTTP 400, so the prompt instructs the model to call
  the tool and the caller validates and re-asks when it does not.
* No ``temperature`` / ``top_p`` / ``top_k``. Non-default values are rejected.
* Thinking is adaptive, with depth set by ``output_config.effort`` passed in
  ``additionalModelRequestFields`` (the Converse route for model-specific fields).
  Disabled or fixed-budget thinking is rejected.
* Responses can contain reasoning blocks before the tool call, so content is
  selected by block type, never by position.
* Strict tool schemas and native structured output are not available on Bedrock
  for this model, so ``llm/verdicts.py`` validates every answer itself.

Retry policy: botocore retries are disabled (a hidden retry could blow the
request deadline). This class retries only transient transport failures
(throttling, 5xx, timeouts) with jittered backoff, and only while the deadline
leaves room for another attempt. Unusable *answers* are handled one level up.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, Mapping, Optional

from .. import __version__
from ..config import Settings
from ..deadline import Deadline
from ..errors import (
    AuditError, LlmAccessError, LlmOutputError, LlmRefusalError, LlmRequestError,
    LlmThrottledError, LlmUnavailableError,
)
from ..logging_utils import log_event
from ..textutil import clean_line
from .base import LlmUsage, ToolCallResult, ToolSpec, UsageMeter

logger = logging.getLogger(__name__)

_THROTTLE_CODES = {
    "ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException", "RequestLimitExceeded",
}
_UNAVAILABLE_CODES = {
    "ModelTimeoutException", "ServiceUnavailableException", "InternalServerException", "ModelNotReadyException",
    "ModelErrorException", "InternalFailure", "ServiceUnavailable", "ExpiredTokenException", "RequestExpired",
}
_ACCESS_CODES = {
    "AccessDeniedException", "UnrecognizedClientException", "InvalidSignatureException", "ResourceNotFoundException",
}
_REFUSAL_STOP_REASONS = {"guardrail_intervened", "content_filtered", "refusal"}
_MALFORMED_STOP_REASONS = {"malformed_tool_use", "malformed_model_output"}
_SDK_MODULE_PREFIXES = ("botocore", "boto3", "urllib3")


def classify_exception(exc: BaseException) -> Optional[AuditError]:
    """Map an SDK exception to a typed error, or None if it is not an SDK/network error.

    Duck-typed on ``exc.response`` so this module does not import botocore, which
    keeps unit tests free of the SDK. Unknown exceptions return None and are
    re-raised by the caller: a programming error must not look like an outage."""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error") or {}
        meta = response.get("ResponseMetadata") or {}
        code = str(error.get("Code") or "")
        status = meta.get("HTTPStatusCode")
        details = {"aws_error_code": code or None, "aws_http_status": status, "aws_request_id": meta.get("RequestId")}
        details = {k: v for k, v in details.items() if v is not None}
        message = clean_line(error.get("Message") or "", 200)
        if code in _THROTTLE_CODES or status == 429:
            return LlmThrottledError(f"Bedrock throttled the request ({code or status})", details=details)
        if code in _UNAVAILABLE_CODES or (isinstance(status, int) and status >= 500):
            return LlmUnavailableError(f"Bedrock was unavailable ({code or status})", details=details)
        if code in _ACCESS_CODES or status in (401, 403, 404):
            return LlmAccessError(
                f"Bedrock denied access or the model is unavailable to this account/role ({code or status}): {message}",
                details=details,
            )
        return LlmRequestError(f"Bedrock rejected the request ({code or status}): {message}", details=details)

    name = type(exc).__name__
    module = type(exc).__module__ or ""
    if isinstance(exc, (TimeoutError, ConnectionError)) or module.startswith(_SDK_MODULE_PREFIXES):
        return LlmUnavailableError(f"Bedrock call failed ({name})", details={"exception_type": name})
    return None


class BedrockClaudeClient:
    def __init__(
        self,
        runtime_client: Any,
        *,
        model_id: str,
        effort: str = "high",
        thinking_mode: str = "adaptive",
        max_output_tokens: int = 16000,
        prompt_caching: bool = False,
        extra_request_fields: Optional[Mapping[str, Any]] = None,
        guardrail_id: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        max_transport_retries: int = 2,
        min_attempt_budget_s: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rand: Callable[[], float] = random.random,
        log_raw_identifiers: bool = False,
    ) -> None:
        self._runtime = runtime_client
        self._model_id = model_id
        self._effort = effort
        self._thinking_mode = thinking_mode
        self._max_output_tokens = max_output_tokens
        self._prompt_caching = prompt_caching
        self._extra = dict(extra_request_fields or {})
        self._guardrail = (guardrail_id, guardrail_version) if guardrail_id and guardrail_version else None
        self._max_transport_retries = max_transport_retries
        self._min_attempt_budget_s = min_attempt_budget_s
        self._sleep = sleep
        self._clock = clock
        self._rand = rand
        self._log_ids = log_raw_identifiers

    @classmethod
    def from_settings(cls, settings: Settings, runtime_client: Any = None) -> "BedrockClaudeClient":
        return cls(
            runtime_client if runtime_client is not None else create_bedrock_runtime(settings),
            model_id=settings.bedrock_model_id,
            effort=settings.llm_effort,
            thinking_mode=settings.llm_thinking_mode,
            max_output_tokens=settings.llm_max_output_tokens,
            prompt_caching=settings.llm_prompt_caching,
            extra_request_fields=settings.llm_extra_request_fields,
            guardrail_id=settings.guardrail_id,
            guardrail_version=settings.guardrail_version,
            max_transport_retries=settings.llm_max_transport_retries,
            min_attempt_budget_s=settings.llm_min_attempt_budget_s,
            log_raw_identifiers=settings.log_raw_identifiers,
        )

    # -- request ---------------------------------------------------------------
    def build_request(
        self, *, system_prompt: str, user_text: str, tool: ToolSpec, max_output_tokens: Optional[int] = None
    ) -> dict:
        system: list[dict] = [{"text": system_prompt}]
        if self._prompt_caching:
            system.append({"cachePoint": {"type": "default"}})
        additional: dict[str, Any] = {
            "thinking": {"type": self._thinking_mode},
            "output_config": {"effort": self._effort},
        }
        additional.update(self._extra)  # operator override, shallow
        request: dict[str, Any] = {
            "modelId": self._model_id,
            "system": system,
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": {"maxTokens": int(max_output_tokens or self._max_output_tokens)},
            "toolConfig": {
                "tools": [{
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": dict(tool.input_schema)},
                    }
                }],
                "toolChoice": {"auto": {}},
            },
            "additionalModelRequestFields": additional,
        }
        if self._guardrail:
            request["guardrailConfig"] = {
                "guardrailIdentifier": self._guardrail[0], "guardrailVersion": self._guardrail[1],
            }
        return request

    # -- response --------------------------------------------------------------
    def parse_response(
        self, response: Mapping[str, Any], tool: ToolSpec, wall_ms: int, meter: Optional[UsageMeter] = None
    ) -> ToolCallResult:
        usage_raw = response.get("usage") or {}
        metrics = response.get("metrics") or {}
        usage = LlmUsage(
            input_tokens=int(usage_raw.get("inputTokens") or 0),
            output_tokens=int(usage_raw.get("outputTokens") or 0),
            cache_read_tokens=int(usage_raw.get("cacheReadInputTokens") or 0),
            cache_write_tokens=int(usage_raw.get("cacheWriteInputTokens") or 0),
            latency_ms=int(metrics.get("latencyMs") or wall_ms),
        )
        if meter is not None:
            meter.add(usage)
        stop = str(response.get("stopReason") or "")
        log_event(
            logger, "llm_call", stop_reason=stop, model_id=self._model_id, latency_ms=usage.latency_ms,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
        )

        if stop in _REFUSAL_STOP_REASONS:
            raise LlmRefusalError("the model declined to answer or a guardrail intervened", details={"stop_reason": stop})
        if stop == "max_tokens":
            raise LlmOutputError(
                "the response was cut off at the output token limit",
                code="llm_output_truncated", details={"stop_reason": stop},
            )
        if stop in _MALFORMED_STOP_REASONS:
            raise LlmOutputError("the model produced a malformed tool call", details={"stop_reason": stop})

        message = (response.get("output") or {}).get("message") or {}
        blocks = message.get("content") or []
        calls = [b["toolUse"] for b in blocks if isinstance(b, Mapping) and isinstance(b.get("toolUse"), Mapping)]
        ours = [c for c in calls if c.get("name") == tool.name]
        if len(ours) != 1:
            raise LlmOutputError(
                "the model did not call the answer tool exactly once", details={"stop_reason": stop, "tool_calls": len(ours)}
            )
        arguments = ours[0].get("input")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if not isinstance(arguments, Mapping):
            raise LlmOutputError("the tool input was not a JSON object", details={"stop_reason": stop})
        return ToolCallResult(tool_name=tool.name, arguments=arguments, usage=usage, stop_reason=stop)

    # -- call ------------------------------------------------------------------
    def _backoff(self, attempt: int) -> float:
        return min(4.0, 0.5 * (2 ** attempt)) * (0.5 + 0.5 * self._rand())

    def invoke_tool(
        self,
        *,
        system_prompt: str,
        user_text: str,
        tool: ToolSpec,
        deadline: Deadline,
        max_output_tokens: Optional[int] = None,
        meter: Optional[UsageMeter] = None,
    ) -> ToolCallResult:
        request = self.build_request(
            system_prompt=system_prompt, user_text=user_text, tool=tool, max_output_tokens=max_output_tokens
        )
        for attempt in range(self._max_transport_retries + 1):
            deadline.check(self._min_attempt_budget_s, what="an LLM call")
            started = self._clock()
            try:
                response = self._runtime.converse(**request)
            except Exception as exc:  # noqa: BLE001 - classified below; unknown types are re-raised
                error = classify_exception(exc)
                if error is None:
                    raise
                log_event(
                    logger, "llm_call_failed", level=logging.WARNING, error_code=error.code, attempt=attempt + 1,
                    retryable=error.retryable, model_id=self._model_id,
                    aws_error_code=(error.details or {}).get("aws_error_code"),
                    aws_request_id=(error.details or {}).get("aws_request_id"),
                )
                if error.retryable and attempt < self._max_transport_retries:
                    delay = self._backoff(attempt)
                    if deadline.remaining() > delay + self._min_attempt_budget_s:
                        self._sleep(delay)
                        continue
                raise error from exc
            wall_ms = int((self._clock() - started) * 1000)
            return self.parse_response(response, tool, wall_ms, meter)
        raise AssertionError("unreachable: the loop always returns or raises")  # pragma: no cover


def create_bedrock_runtime(settings: Settings) -> Any:
    """Build the boto3 ``bedrock-runtime`` client. Credentials come from the
    execution role through the default chain; nothing is read from config."""
    import boto3
    from botocore.config import Config

    config = Config(
        region_name=settings.bedrock_region,
        connect_timeout=settings.llm_connect_timeout_s,
        read_timeout=settings.llm_read_timeout_s,
        retries={"max_attempts": 0, "mode": "standard"},  # retries are owned by BedrockClaudeClient
        max_pool_connections=max(10, settings.max_concurrent_requests * 2),
        user_agent_extra=f"admissions-audit-agent/{__version__}",
    )
    return boto3.client("bedrock-runtime", config=config)
