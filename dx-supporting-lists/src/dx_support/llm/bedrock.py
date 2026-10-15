"""Claude on Amazon Bedrock, through the Converse API. The ONLY module that imports boto3 for model calls.

How a call works
-----------------
* Credentials: the default boto3 chain, which on SageMaker is the endpoint's execution role. No key is read from a file or the environment
  by this project.
* Structured output: each request offers ONE tool, ``submit_result``, whose input schema is the stage's JSON Schema, and the model is asked
  to answer by calling it. ``toolChoice`` is ``auto`` because Claude 5.x on Bedrock rejects forced tool choice (and forced tool choice cannot
  be combined with thinking anyway). Whatever arrives is then validated by the caller (every id must be one that was sent), so a model that
  ignores the tool, or returns a malformed answer, produces a failed attempt, never a bad verdict. If the model answers in plain text, the
  outermost JSON object in it is used as a fallback.
* No sampling parameters are sent (Claude 5.x rejects them). Only ``maxTokens``, which also bounds the worst-case cost of a call.
* A call whose WORST-CASE cost (estimated input + the full output cap) exceeds the per-call budget is refused before it is sent.
* Retries and back-off live in LLMGate, not here: boto3's own retries are switched off so every attempt is counted and costed exactly once.
* Prompts, responses and request bodies are never logged.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from dx_support import __version__
from dx_support.llm.catalog import (Catalog, ResolvedModel, cost_from_usage, load_catalog, min_output_tokens, resolve_model, thinking_fields,
                                    worst_case_cost)
from dx_support.llm.types import LLMCallError, LLMRequest, LLMResult, extract_json_fallback
from dx_support.settings import Settings
from dx_support.text import redact

logger = logging.getLogger(__name__)

TOOL_NAME = "submit_result"
STRUCTURED_OUTPUT_NOTE = ("Deliver your final answer by calling the `submit_result` tool exactly once, with arguments that satisfy its input "
                          "schema. Do not reply in plain text.")
_CACHE_PREFIX_MIN_CHARS = 2000        # below this a cache checkpoint cannot reach the model's minimum cacheable size anyway

_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ModelNotReadyException", "ServiceQuotaExceededException", "RequestLimitExceeded",
                   "Throttling", "ThrottledException"}
_TRANSIENT_CODES = {"ServiceUnavailableException", "InternalServerException", "ModelTimeoutException", "ModelErrorException",
                    "ModelStreamErrorException", "InternalFailure", "ServiceUnavailable", "RequestTimeout", "RequestTimeoutException"}
_FATAL_CODES = {"AccessDeniedException", "AccessDenied", "UnrecognizedClientException", "ExpiredTokenException", "ExpiredToken", "InvalidSignatureException",
                "InvalidClientTokenId", "SignatureDoesNotMatch", "ResourceNotFoundException", "UnauthorizedOperation", "InvalidIdentityToken"}
_TRANSIENT_CLASS_NAMES = {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError", "ConnectionClosedError", "ProxyConnectionError",
                          "SSLError", "IncompleteReadError", "ResponseStreamingError", "TimeoutError"}
_FATAL_CLASS_NAMES = {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError", "NoRegionError", "ProfileNotFound",
                      "InvalidRegionError", "UnknownEndpointError", "ParamValidationError", "InvalidConfigError"}
_TOO_LONG_RE = re.compile(r"too long|too many tokens|input length|exceeds? the (?:maximum|max)|context (?:length|window)", re.I)


def translate_error(exc: BaseException) -> LLMCallError:
    """Maps a boto3 / botocore exception to an LLMCallError with the right `kind`. Duck-typed (no botocore import) so it is testable anywhere."""
    response = getattr(exc, "response", None)
    code = ""
    status = 0
    message = str(exc)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        code = str(err.get("Code") or "")
        message = str(err.get("Message") or message)
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    name = type(exc).__name__
    text = redact(f"{code or name}: {message}")[:400]
    if code == "ValidationException":
        # A validation error is almost always a configuration problem that would fail every call the same way (bad model id, an option the
        # model rejects). Stop the run -- unless the request itself was too large, which is specific to that one call.
        return LLMCallError(text, kind="budget" if _TOO_LONG_RE.search(message) else "fatal")
    if code in _THROTTLE_CODES or status == 429:
        return LLMCallError(text, kind="throttle")
    if code in _FATAL_CODES or name in _FATAL_CLASS_NAMES or status in (401, 403, 404):
        return LLMCallError(text, kind="fatal")
    if code in _TRANSIENT_CODES or name in _TRANSIENT_CLASS_NAMES or status >= 500 or status == 408:
        return LLMCallError(text, kind="transient")
    if code and status == 400:
        return LLMCallError(text, kind="fatal")
    return LLMCallError(text, kind="transient")


class BedrockBackend:
    name = "bedrock-converse"

    def __init__(self, settings: Settings, client: Any = None, catalog: Optional[Catalog] = None):
        self.settings = settings
        self.catalog = catalog or load_catalog(settings.model_catalog_path)
        self._client = client
        self._client_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.llm_concurrency + 2, thread_name_prefix="dx-bedrock")
        self._resolved: dict[str, ResolvedModel] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3                                # lazy: the stub backend, the tests and the offline tools never need it
                    from botocore.config import Config
                    cfg = Config(region_name=self.settings.aws_region, retries={"max_attempts": 0}, connect_timeout=10,
                                 read_timeout=int(self.settings.llm_call_timeout_s), max_pool_connections=self.settings.llm_concurrency + 4,
                                 user_agent_extra=f"dx-support/{__version__}")
                    self._client = boto3.session.Session().client("bedrock-runtime", config=cfg)
        return self._client

    def resolve(self, model: str) -> ResolvedModel:
        r = self._resolved.get(model)
        if r is None:
            r = self._resolved[model] = resolve_model(model, self.settings.inference_profile_prefix, self.catalog)
            if r.warning and r.warning not in self._warned:
                self._warned.add(r.warning)
                logger.warning(r.warning)
        return r

    def describe(self, models: dict[str, str]) -> dict[str, dict]:
        out = {}
        for stage, name in models.items():
            r = self.resolve(name)
            out[stage] = {"alias": name, "model_id": r.model_id, "price_in_per_m": r.spec.price_in, "price_out_per_m": r.spec.price_out,
                          "price_multiplier": r.price_multiplier, "thinking_style": r.spec.thinking_style, "known": r.spec.known}
        return out

    # ------------------------------------------------------------------
    def build_request(self, req: LLMRequest, resolved: Optional[ResolvedModel] = None) -> dict:
        """The exact keyword arguments for `converse()`."""
        resolved = resolved or self.resolve(req.model)
        extra, warning = thinking_fields(resolved.spec, req.thinking)
        if warning and warning not in self._warned:
            self._warned.add(warning)
            logger.warning(warning)
        caching = self.settings.prompt_caching
        system: list[dict] = [{"text": f"{req.system_prompt}\n\n{STRUCTURED_OUTPUT_NOTE}"}]
        if caching:
            system.append({"cachePoint": {"type": "default"}})
        if caching and req.cache_prefix_chars >= _CACHE_PREFIX_MIN_CHARS:
            content = [{"text": req.prompt[: req.cache_prefix_chars]}, {"cachePoint": {"type": "default"}}, {"text": req.prompt[req.cache_prefix_chars:]}]
            if not content[2]["text"]:
                content = content[:2] + [{"text": "Proceed."}]
        else:
            content = [{"text": req.prompt}]
        kwargs: dict[str, Any] = {
            "modelId": resolved.model_id,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": max(req.max_tokens, min_output_tokens(resolved.spec, req.thinking))},
            "toolConfig": {"tools": [{"toolSpec": {"name": TOOL_NAME, "description": "Submit the final structured result for this step.",
                                                   "inputSchema": {"json": req.schema}}}],
                           "toolChoice": {"auto": {}}},
        }
        if extra:
            kwargs["additionalModelRequestFields"] = extra
        if self.settings.guardrail_id:
            kwargs["guardrailConfig"] = {"guardrailIdentifier": self.settings.guardrail_id, "guardrailVersion": self.settings.guardrail_version,
                                         "trace": "disabled"}
        return kwargs

    def parse_response(self, resp: dict, resolved: ResolvedModel, started: float) -> LLMResult:
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        tool_input: Optional[dict] = None
        texts: list[str] = []
        for b in blocks:                                    # select by block type, never by position: thinking blocks may come first
            tu = b.get("toolUse")
            if tu and tu.get("name") == TOOL_NAME and tool_input is None:
                inp = tu.get("input")
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except json.JSONDecodeError:
                        inp = None
                if isinstance(inp, dict):
                    tool_input = inp
            elif "text" in b:
                texts.append(str(b["text"]))
        text = "".join(texts)
        structured = tool_input if tool_input is not None else extract_json_fallback(text)
        usage = resp.get("usage") or {}
        cost = cost_from_usage(resolved, usage, self.catalog)
        stop = str(resp.get("stopReason") or "")
        latency = (resp.get("metrics") or {}).get("latencyMs")
        diag = {"input_tokens": usage.get("inputTokens"), "output_tokens": usage.get("outputTokens"), "thinking_tokens": None,
                "cache_read_tokens": usage.get("cacheReadInputTokens"), "cache_creation_tokens": usage.get("cacheWriteInputTokens"),
                "duration_ms": int(latency) if latency is not None else int((time.monotonic() - started) * 1000), "num_turns": 1,
                "stop_reason": stop, "model_id": resolved.model_id}
        if stop in ("guardrail_intervened", "content_filtered"):
            raise LLMCallError(f"response blocked ({stop})", cost_usd=cost, kind="budget")      # a retry would be blocked again
        if structured is None and stop == "max_tokens":
            raise LLMCallError("output was cut off at max_tokens before a result was produced; raise DX_MAX_OUTPUT_TOKENS_<STAGE>",
                               cost_usd=cost, kind="transient")
        return LLMResult(structured=structured, cost_usd=cost, diag=diag, text=text)

    # ------------------------------------------------------------------
    async def complete(self, req: LLMRequest) -> LLMResult:
        resolved = self.resolve(req.model)
        kwargs = self.build_request(req, resolved)
        worst = worst_case_cost(resolved, len(req.system_prompt) + len(req.prompt) + len(json.dumps(req.schema)), kwargs["inferenceConfig"]["maxTokens"])
        if worst > req.max_budget_usd:
            raise LLMCallError(f"worst-case cost ${worst:.2f} exceeds the per-call budget ${req.max_budget_usd:.2f} "
                               "(lower the output cap or raise DX_MAX_BUDGET_USD_PER_CALL)", cost_usd=0.0, kind="budget")
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        try:
            resp = await loop.run_in_executor(self._pool, functools.partial(self._get_client().converse, **kwargs))
        except asyncio.CancelledError:
            raise
        except Exception as e:                                          # noqa: BLE001 - translated, never swallowed
            raise translate_error(e) from e
        return self.parse_response(resp, resolved, started)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
