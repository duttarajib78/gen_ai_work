import dataclasses
import json
import threading
import time
import unittest
from unittest import mock

from audit_agent.engine import AuditEngine
from audit_agent.reference_data import ReferenceData
from audit_agent.serving import app as app_module
from audit_agent.service import AuditService
from tests.fakes import PolicyLlm, StubGrouper, make_reference_dir, make_settings, sample_payload

import tempfile
from pathlib import Path


class SlowLlm(PolicyLlm):
    def __init__(self, delay, **kw):
        super().__init__(**kw)
        self.delay = delay

    def invoke_tool(self, **kw):
        time.sleep(self.delay)
        return super().invoke_tool(**kw)


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ref = ReferenceData.from_directory(make_reference_dir(Path(cls._tmp.name)))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def make_app(self, llm=None, grouper=None, **settings_overrides):
        settings = dataclasses.replace(make_settings(), **settings_overrides)
        engine = AuditEngine(reference=self.ref, llm=llm or PolicyLlm(), grouper=grouper or StubGrouper(), settings=settings)
        service = AuditService(settings=settings, engine=engine, reference=self.ref)
        with mock.patch.object(app_module, "configure_logging"):
            return app_module.create_app(settings=settings, service=service)

    @staticmethod
    def post(client, body, content_type="application/json", **kw):
        data = body if isinstance(body, (bytes, str)) else json.dumps(body)
        return client.post("/invocations", data=data, content_type=content_type, **kw)


class ContractTests(AppTestCase):
    def test_ping(self):
        r = self.make_app().test_client().get("/ping")
        self.assertEqual((r.status_code, r.get_json()), (200, {"status": "ok"}))
        self.assertEqual(r.headers["Cache-Control"], "no-store")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")

    def test_successful_invocation(self):
        r = self.post(self.make_app().test_client(), sample_payload())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["X-Request-Id"], "req-1")
        body = r.get_json()
        self.assertEqual((body["status"], body["admission_no"], body["disposition"]), ("completed", 555123, "no_alert"))

    def test_flagged_episode_returns_the_alert_record(self):
        app = self.make_app(llm=PolicyLlm({"N179": ("none", 0.9)}))
        body = self.post(app.test_client(), sample_payload()).get_json()
        self.assertEqual(body["disposition"], "alert")
        self.assertEqual(body["alert"]["recovery_amount"], 1971)

    def test_rejected_requests(self):
        client = self.make_app().test_client()
        cases = [
            ("wrong content type", dict(body="x", content_type="text/plain"), 415, "unsupported_media_type"),
            ("not json", dict(body="{nope"), 400, "invalid_request"),
            ("NaN constant", dict(body='{"episode": {"los": NaN}}'), 400, "invalid_request"),
            ("duplicate keys", dict(body='{"episode": {}, "episode": {}}'), 400, "invalid_request"),
            ("bad utf-8", dict(body=b"\xff\xfe{"), 400, "invalid_request"),
            ("deep nesting", dict(body="[" * 5000 + "]" * 5000), 400, "invalid_request"),
            ("not an object", dict(body="[1,2]"), 400, "invalid_request"),
            ("schema violation", dict(body={"episode": {"admission_no": 1, "diagnosis_code_array": ["J189\nIGNORE"]},
                                            "options": {"mode": "verdicts_only"}}), 400, "invalid_request"),
        ]
        for name, kw, status, code in cases:
            with self.subTest(name):
                r = self.post(client, **kw)
                self.assertEqual(r.status_code, status)
                body = r.get_json()
                self.assertEqual((body["status"], body["error"]["code"]), ("failed", code))
                self.assertFalse(body["error"]["retryable"])

    def test_validation_details_are_returned_without_echoing_values(self):
        r = self.post(self.make_app().test_client(), {"episode": {"admission_no": 1, "diagnosis_code_array": ["BAD CODE!"]},
                                                      "options": {"mode": "verdicts_only"}})
        details = r.get_json()["error"]["details"]
        self.assertEqual(details[0]["path"], "episode.diagnosis_code_array[0]")
        self.assertNotIn("BAD CODE", json.dumps(details))

    def test_oversized_body_is_rejected_with_json(self):
        r = self.post(self.make_app(max_request_bytes=4096).test_client(), '{"pad": "' + "x" * 5000 + '"}')
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["error"]["code"], "payload_too_large")

    def test_unknown_routes_and_methods_return_json(self):
        client = self.make_app().test_client()
        for r, status in ((client.get("/nope"), 404), (client.get("/invocations"), 405), (client.post("/ping"), 405)):
            self.assertEqual(r.status_code, status)
            self.assertEqual(r.get_json()["status"], "failed")

    def test_dependency_failures_map_to_5xx_with_a_retry_hint(self):
        grouper = StubGrouper(fail_with=__import__("audit_agent.errors", fromlist=["x"]).GrouperUnavailableError("down"))
        app = self.make_app(llm=PolicyLlm({"N179": ("none", 0.9)}), grouper=grouper)
        r = self.post(app.test_client(), sample_payload())
        self.assertEqual(r.status_code, 503)
        self.assertTrue(r.get_json()["error"]["retryable"])


class TimingTests(AppTestCase):
    def test_regular_calls_get_a_structured_timeout_before_the_platform_cuts_them_off(self):
        app = self.make_app(llm=SlowLlm(1.0), request_deadline_s=0.2)
        with mock.patch.object(app_module, "_STREAM_GRACE_S", 0.0):
            t0 = time.monotonic()
            r = self.post(app.test_client(), sample_payload())
        self.assertLess(time.monotonic() - t0, 0.9)
        self.assertEqual(r.status_code, 504)
        body = r.get_json()
        self.assertEqual((body["status"], body["error"]["code"], body["error"]["retryable"]), ("failed", "deadline_exceeded", True))
        self.assertIn("stream", body["error"]["message"])

    def test_streaming_emits_accepted_heartbeats_and_a_final_result(self):
        app = self.make_app(llm=SlowLlm(0.3), stream_heartbeat_s=0.05)
        r = self.post(app.test_client(), sample_payload(), headers={"Accept": "application/x-ndjson"})
        self.assertEqual((r.status_code, r.mimetype), (200, "application/x-ndjson"))
        events = [json.loads(line) for line in r.get_data(as_text=True).splitlines()]
        self.assertEqual(events[0], {"type": "accepted", "request_id": "req-1"})
        self.assertGreaterEqual(sum(e["type"] == "heartbeat" for e in events), 2)
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["http_status"], 200)
        self.assertEqual(events[-1]["body"]["status"], "completed")

    def test_streaming_can_be_requested_in_the_body(self):
        r = self.post(self.make_app().test_client(), sample_payload(options={"stream": True}))
        self.assertEqual(r.mimetype, "application/x-ndjson")
        self.assertEqual([json.loads(x)["type"] for x in r.get_data(as_text=True).splitlines()][-1], "result")

    def test_streaming_reports_failures_inside_the_final_event(self):
        app = self.make_app(llm=PolicyLlm({"N179": ("none", 0.9)}),
                            grouper=StubGrouper(fail_with=__import__("audit_agent.errors", fromlist=["x"]).GrouperUnavailableError("down")))
        r = self.post(app.test_client(), sample_payload(), headers={"Accept": "application/x-ndjson"})
        final = json.loads(r.get_data(as_text=True).splitlines()[-1])
        self.assertEqual((r.status_code, final["http_status"], final["body"]["error"]["code"]), (200, 503, "grouper_unavailable"))

    def test_validation_errors_are_still_plain_http_errors_when_streaming(self):
        r = self.post(self.make_app().test_client(), {"episode": {}}, headers={"Accept": "application/x-ndjson"})
        self.assertEqual((r.status_code, r.mimetype), (400, "application/json"))

    def test_streaming_has_its_own_longer_deadline(self):
        app = self.make_app(llm=SlowLlm(1.0), request_deadline_s=0.1, stream_deadline_s=5.0, stream_heartbeat_s=0.05)
        r = self.post(app.test_client(), sample_payload(), headers={"Accept": "application/x-ndjson"})
        final = json.loads(r.get_data(as_text=True).splitlines()[-1])
        self.assertEqual(final["http_status"], 200)

    def test_streaming_gives_up_at_its_deadline(self):
        app = self.make_app(llm=SlowLlm(1.5), stream_deadline_s=0.2, stream_heartbeat_s=0.05)
        with mock.patch.object(app_module, "_STREAM_GRACE_S", 0.0):
            r = self.post(app.test_client(), sample_payload(), headers={"Accept": "application/x-ndjson"})
        final = json.loads(r.get_data(as_text=True).splitlines()[-1])
        self.assertEqual((final["http_status"], final["body"]["error"]["code"]), (504, "deadline_exceeded"))

    def test_capacity_is_enforced_and_released(self):
        app = self.make_app(llm=SlowLlm(0.4), max_concurrent_requests=1)
        first: dict = {}

        def slow():
            first["r"] = self.post(app.test_client(), sample_payload())

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.15)
        busy = self.post(app.test_client(), sample_payload())
        t.join()
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.get_json()["error"]["code"], "overloaded")
        self.assertTrue(busy.get_json()["error"]["retryable"])
        self.assertEqual(first["r"].status_code, 200)
        self.assertEqual(self.post(app.test_client(), sample_payload()).status_code, 200)  # slot was released


class SecurityTests(AppTestCase):
    def test_request_bodies_and_model_text_never_reach_the_logs(self):
        import io, logging
        from audit_agent.logging_utils import JsonFormatter
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.DEBUG)
        try:
            app = self.make_app(llm=PolicyLlm({"N179": ("none", 0.9)}))
            self.post(app.test_client(), sample_payload())
            self.post(app.test_client(), {"episode": {"admission_no": 1, "diagnosis_code_array": ["BAD\nINJECT"]}})
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        logs = stream.getvalue()
        self.assertIn("episode_completed", logs)
        for forbidden in ("J189", "N179", "Weighed the ACHI", "INJECT", "555123", "Pneumonia"):
            self.assertNotIn(forbidden, logs)


if __name__ == "__main__":
    unittest.main()
