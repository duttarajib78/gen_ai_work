"""Exception hierarchy.

Every exception that can reach the HTTP layer derives from ``AuditError`` and
carries three things the caller can act on:

* ``code``       stable, machine-readable identifier (documented in the README)
* ``http_status`` who should act: 4xx = caller fixes the request, 5xx = platform
                 or dependency problem, 200 = the request was processed but this
                 one episode could not be completed (inspect ``status``)
* ``retryable``  whether re-sending the same request unchanged may succeed

Messages must never embed episode content (codes, reasoning, identifiers).
"""
from __future__ import annotations

from typing import Any, Optional


class AuditError(Exception):
    code = "internal_error"
    http_status = 500
    retryable = False

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        http_status: Optional[int] = None,
        retryable: Optional[bool] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message or (code or self.code))
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        if retryable is not None:
            self.retryable = retryable
        self.details = details

    def to_error_body(self) -> dict:
        body = {"code": self.code, "message": str(self), "retryable": self.retryable}
        if self.details:
            body["details"] = self.details
        return body


# ---- startup / configuration -------------------------------------------------
class ConfigError(AuditError):
    code = "configuration_error"


class ReferenceDataError(AuditError):
    code = "reference_data_error"


# ---- request handling --------------------------------------------------------
class RequestValidationError(AuditError):
    code = "invalid_request"
    http_status = 400


class PayloadTooLargeError(AuditError):
    code = "payload_too_large"
    http_status = 413


class UnsupportedMediaTypeError(AuditError):
    code = "unsupported_media_type"
    http_status = 415


class OverloadedError(AuditError):
    code = "overloaded"
    http_status = 503
    retryable = True


class DeadlineExceededError(AuditError):
    code = "deadline_exceeded"
    http_status = 504
    retryable = True


class RequestCancelledError(AuditError):
    code = "request_cancelled"
    http_status = 503
    retryable = True


# ---- LLM ---------------------------------------------------------------------
class LlmError(AuditError):
    code = "llm_error"
    http_status = 502


class LlmThrottledError(LlmError):
    code = "llm_throttled"
    http_status = 503
    retryable = True


class LlmUnavailableError(LlmError):
    code = "llm_unavailable"
    http_status = 503
    retryable = True


class LlmAccessError(LlmError):
    """Model not enabled for the account/role, wrong model id, or IAM denial."""

    code = "llm_access_denied"
    http_status = 500


class LlmRequestError(LlmError):
    """Bedrock rejected the request shape (a bug or a configuration problem)."""

    code = "llm_request_rejected"
    http_status = 500


class LlmOutputError(LlmError):
    """The model answered, but not with a usable result after all attempts.

    Reported with HTTP 200 and ``status="failed"``: it is a property of this
    episode (or of the prompt), not a platform outage."""

    code = "llm_output_invalid"
    http_status = 200


class LlmRefusalError(LlmError):
    code = "llm_refused"
    http_status = 200


# ---- grouper / corrections ---------------------------------------------------
class GrouperUnavailableError(AuditError):
    code = "grouper_unavailable"
    http_status = 503
    retryable = True


class GrouperInputError(AuditError):
    code = "grouper_input_invalid"
    http_status = 200


class CorrectionError(AuditError):
    """The proposed correction cannot be applied unambiguously; needs a human coder."""

    code = "correction_not_applicable"
    http_status = 200
