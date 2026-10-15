"""SageMaker real-time endpoint entry point (PyTorch inference container contract).

    model_fn -> input_fn -> predict_fn -> output_fn

All business logic lives in the ``drg_audit`` package; this file is deliberately thin glue.

Security behaviour:
  * fail closed: ``model_fn`` raises (and the endpoint never becomes healthy) if the bundle fails
    integrity verification or the model self-check
  * client errors become 4xx with a stable ``code: message`` string (no submitted values echoed)
  * unexpected errors become a generic 500; details go to CloudWatch Logs, never to the caller
  * the Hugging Face hub is disabled; nothing is downloaded at runtime
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")  # noqa: S108  (container scratch space)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from drg_audit.errors import ArtifactError, ConfigError, RequestValidationError  # noqa: E402
from drg_audit.logging_utils import configure_logging, get_logger, log_event  # noqa: E402
from drg_audit.serving import contract  # noqa: E402
from drg_audit.serving.service import InferenceService  # noqa: E402

try:  # present inside the SageMaker inference container
    from sagemaker_inference.errors import GenericInferenceToolkitError as _ToolkitError
except ImportError:  # local development / unit tests
    _ToolkitError = None

configure_logging()
logger = get_logger("entrypoint")


class HttpError(Exception):
    """Fallback used outside the SageMaker container; mirrors the toolkit error's attributes."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(status_code, message)
        self.status_code = status_code
        self.message = message


def _http_error(status_code: int, message: str) -> Exception:
    if _ToolkitError is not None:
        return _ToolkitError(status_code, message)
    return HttpError(status_code, message)


class RawRequest(NamedTuple):
    body: bytes | str
    content_type: str


def model_fn(model_dir: str) -> InferenceService:
    """Load the bundle from ``model_dir`` (SageMaker extracts model.tar.gz there)."""
    try:
        return InferenceService.from_model_dir(model_dir)
    except (ArtifactError, ConfigError) as exc:
        log_event(logger, "model_load_failed", level=logging.ERROR, error_type=type(exc).__name__, detail=str(exc))
        raise


def input_fn(request_body: bytes | str, request_content_type: str = contract.JSON_MEDIA_TYPE) -> RawRequest:
    """Cheap media-type allowlist. Full validation needs the bundle's limits, so it runs in ``predict_fn``."""
    media_type = (request_content_type or "").split(";")[0].strip().lower()
    if media_type != contract.JSON_MEDIA_TYPE:
        raise _http_error(415, "unsupported_media_type: Content-Type must be application/json")
    return RawRequest(request_body, media_type)


def predict_fn(raw: RawRequest, service: InferenceService) -> dict:
    try:
        return service.score(service.parse(raw.body, raw.content_type))
    except RequestValidationError as exc:
        raise _http_error(exc.status_code, f"{exc.code}: {exc.message}") from None
    except Exception:
        logger.exception("scoring_failed")
        raise _http_error(500, "internal_error: the request could not be processed") from None


def output_fn(prediction: dict, accept: str = contract.JSON_MEDIA_TYPE) -> tuple[str, str]:
    try:
        media_type = contract.negotiate_accept(accept)
        return contract.serialize_response(prediction), media_type
    except RequestValidationError as exc:
        raise _http_error(exc.status_code, f"{exc.code}: {exc.message}") from None
    except ValueError:
        logger.exception("response_serialization_failed")
        raise _http_error(500, "internal_error: the response could not be produced") from None
