"""Exception hierarchy. Each error carries a stable machine-readable code and an HTTP status."""
from __future__ import annotations


class RuleExtractorError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str = "", *, details: list[str] | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = list(details or [])


class ConfigError(RuleExtractorError):
    code = "config_error"


class RequestValidationError(RuleExtractorError):
    code = "invalid_request"
    http_status = 400


class NotFoundError(RuleExtractorError):
    code = "not_found"
    http_status = 404


class ConflictError(RuleExtractorError):
    code = "conflict"
    http_status = 409


class PolicyError(RuleExtractorError):
    """The request is well-formed but disabled by deployment policy."""

    code = "forbidden_by_policy"
    http_status = 403


class IngestionError(RuleExtractorError):
    code = "ingestion_error"


class OutputError(RuleExtractorError):
    code = "output_error"


class IntegrityError(RuleExtractorError):
    code = "integrity_error"


class AgentError(RuleExtractorError):
    """Raised by agent runners. ``transient`` errors are retried with back-off."""

    code = "agent_error"

    def __init__(self, message: str = "", *, transient: bool = False, reason: str = "error", stats: object = None) -> None:
        super().__init__(message)
        self.transient = transient
        self.reason = reason
        self.stats = stats          # AgentRunStats of the failed attempt (cost accounting)


class UnavailableError(RuleExtractorError):
    """A required dependency (for example the corpus) is not available on this instance."""

    code = "unavailable"
    http_status = 503
