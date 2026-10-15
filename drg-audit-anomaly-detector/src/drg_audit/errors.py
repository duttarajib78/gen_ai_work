"""Exception hierarchy.

Security note: exception messages are logged and may be returned to API clients. They must never
contain request data (admission numbers, codes, ages, ...). Use field *names* and stable error codes.
"""

from __future__ import annotations


class DrgAuditError(Exception):
    """Base class for all project errors."""


class ConfigError(DrgAuditError):
    """Invalid or inconsistent configuration."""


class DataValidationError(DrgAuditError):
    """Training or batch data does not satisfy the data contract."""


class ArtifactError(DrgAuditError):
    """A model bundle is missing, malformed, or fails integrity verification."""


class RequestValidationError(DrgAuditError):
    """The client request is invalid. Carries an HTTP status and a stable machine-readable code."""

    def __init__(self, message: str, *, code: str = "invalid_request", status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class PayloadTooLargeError(RequestValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="payload_too_large", status_code=413)


class UnsupportedMediaTypeError(RequestValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_media_type", status_code=415)


class NotAcceptableError(RequestValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="not_acceptable", status_code=406)
