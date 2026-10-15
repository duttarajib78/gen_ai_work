"""Exercises sagemaker/code/inference.py end to end with a stub engine (no torch required)."""
import json
import unittest
from unittest import mock

from tests.helpers import valid_payload_record  # noqa: I001

import inference  # sagemaker/code/inference.py (path added by tests.helpers)
from drg_audit.schemas import AuditResult
from drg_audit.serving.service import InferenceService
from drg_audit.settings import RequestLimits
from tests.test_service import StubEngine


def make_service(limits=None):
    return InferenceService(StubEngine(), limits or RequestLimits())


def invoke(service, payload, content_type="application/json", accept="application/json"):
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    raw = inference.input_fn(body, content_type)
    prediction = inference.predict_fn(raw, service)
    return inference.output_fn(prediction, accept)


class EntrypointTests(unittest.TestCase):
    def test_happy_path(self):
        service = make_service()
        body, content_type = invoke(service, {"records": [valid_payload_record(), valid_payload_record(drg="G62A")]})
        self.assertEqual(content_type, "application/json")
        parsed = json.loads(body)
        self.assertEqual(parsed["summary"]["n_records"], 2)
        self.assertEqual(parsed["summary"]["n_anomalies"], 1)

    def test_client_errors_map_to_4xx_without_echoing_input(self):
        service = make_service()
        cases = [
            ("{bad json", "application/json", 400, "invalid_json"),
            (json.dumps({"records": []}), "application/json", 400, "invalid_records"),
            (json.dumps({"records": [valid_payload_record()]}), "text/csv", 415, "unsupported_media_type"),
        ]
        for body, content_type, status, code in cases:
            with self.subTest(code=code), self.assertRaises(Exception) as ctx:
                invoke(service, body, content_type=content_type)
            self.assertEqual(ctx.exception.status_code, status)
            self.assertIn(code, str(ctx.exception.args))

    def test_payload_too_large_is_413(self):
        service = make_service(RequestLimits(max_payload_bytes=2048))
        with self.assertRaises(Exception) as ctx:
            invoke(service, {"records": [valid_payload_record(admission_no=str(i)) for i in range(100)]})
        self.assertEqual(ctx.exception.status_code, 413)

    def test_unacceptable_accept_header_is_406(self):
        with self.assertRaises(Exception) as ctx:
            invoke(make_service(), {"records": [valid_payload_record()]}, accept="text/html")
        self.assertEqual(ctx.exception.status_code, 406)

    def test_internal_errors_are_generic_500s(self):
        service = make_service()
        service.engine.predict = mock.Mock(side_effect=RuntimeError("CUDA exploded with SENSITIVE-ADM-9"))
        with self.assertLogs("drg_audit.entrypoint", level="ERROR"), self.assertRaises(Exception) as ctx:
            invoke(service, {"records": [valid_payload_record(admission_no="SENSITIVE-ADM-9")]})
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn("SENSITIVE", str(ctx.exception.args))
        self.assertNotIn("CUDA", str(ctx.exception.args))

    def test_nan_in_response_is_a_500_not_invalid_json(self):
        service = make_service()
        service.engine.predict = lambda records, confidence_threshold=None: [
            AuditResult(admission_no=r.admission_no, audit_priority_score=float("nan")) for r in records]
        with self.assertLogs("drg_audit.entrypoint", level="ERROR"), self.assertRaises(Exception) as ctx:
            invoke(service, {"records": [valid_payload_record()]})
        self.assertEqual(ctx.exception.status_code, 500)

    def test_model_fn_fails_closed_on_bad_bundle(self):
        import tempfile
        from drg_audit.errors import ArtifactError
        with tempfile.TemporaryDirectory() as empty, self.assertLogs("drg_audit.entrypoint", level="ERROR"):
            with self.assertRaises(ArtifactError):
                inference.model_fn(empty)


if __name__ == "__main__":
    unittest.main()
