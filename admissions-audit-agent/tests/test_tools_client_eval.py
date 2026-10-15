import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_agent.schemas import parse_request
from tests.fakes import FakeClientError, PolicyLlm, make_reference_dir, make_settings, sample_episode
from tools.client import batch_invoke
from tools.client.sagemaker_client import InvokeResult, invoke, invoke_with_retry, parse_ndjson
from tools.evaluation import benchmark, check_reference_coverage, sweep_decision_rules
from tools.evaluation.metrics import compute_metrics, confusion, percentile
from tools.tabular import coerce_to_list, jsonable, row_to_payload

OK_BODY = {"status": "completed", "disposition": "no_alert", "verdicts": [{}, {}], "provenance": {}}


class FakeBody:
    def __init__(self, data): self.data = data
    def read(self): return self.data


class FakeRuntime:
    def __init__(self, *script): self.script, self.calls = list(script), []
    def _next(self, kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
    def invoke_endpoint(self, **kw): return self._next(kw)
    def invoke_endpoint_with_response_stream(self, **kw): return self._next(kw)


def model_error(status, body):
    err = FakeClientError("ModelError", 424)
    err.response["OriginalStatusCode"] = status
    err.response["OriginalMessage"] = body if isinstance(body, str) else json.dumps(body)
    return err


class SageMakerClientTests(unittest.TestCase):
    def test_ndjson_parser_handles_lines_split_across_chunks(self):
        data = b'{"type":"accepted"}\n{"type":"heartbeat","elapsed_ms":5}\n{"type":"result","http_status":200}'
        for size in (1, 3, 7, 1000):
            with self.subTest(size=size):
                chunks = [data[i:i + size] for i in range(0, len(data), size)]
                self.assertEqual([e["type"] for e in parse_ndjson(chunks)], ["accepted", "heartbeat", "result"])

    def test_regular_invoke_success_and_request_shape(self):
        rt = FakeRuntime({"Body": FakeBody(json.dumps(OK_BODY).encode())})
        result = invoke(rt, "ep", {"request_id": "r1", "episode": {}}, inference_id="r1")
        self.assertTrue(result.ok)
        kw = rt.calls[0]
        self.assertEqual((kw["EndpointName"], kw["ContentType"], kw["InferenceId"]), ("ep", "application/json", "r1"))

    def test_model_errors_surface_the_containers_own_status_and_body(self):
        body = {"status": "failed", "error": {"code": "invalid_request", "retryable": False}}
        r = invoke(FakeRuntime(model_error(400, body)), "ep", {})
        self.assertEqual((r.http_status, r.body["error"]["code"], r.retryable), (400, "invalid_request", False))
        r = invoke(FakeRuntime(model_error(503, "not json at all")), "ep", {})
        self.assertEqual((r.http_status, r.body["error"]["code"], r.retryable), (503, "model_error", True))

    def test_platform_errors_are_classified(self):
        r = invoke(FakeRuntime(FakeClientError("ThrottlingException", 429)), "ep", {})
        self.assertTrue(r.retryable)
        r = invoke(FakeRuntime(FakeClientError("ValidationError", 400)), "ep", {})
        self.assertFalse(r.retryable)
        r = invoke(FakeRuntime(TimeoutError("read timeout")), "ep", {})
        self.assertTrue(r.retryable)
        with self.assertRaises(KeyError):  # programming errors are not disguised
            invoke(FakeRuntime(KeyError("bug")), "ep", {})

    def test_streaming_returns_the_final_result_event(self):
        lines = [{"type": "accepted"}, {"type": "heartbeat"}, {"type": "result", "http_status": 200, "body": OK_BODY}]
        raw = "".join(json.dumps(x) + "\n" for x in lines).encode()
        events = [{"PayloadPart": {"Bytes": raw[i:i + 11]}} for i in range(0, len(raw), 11)]
        rt = FakeRuntime({"Body": events})
        r = invoke(rt, "ep", {}, stream=True)
        self.assertTrue(r.ok)
        self.assertEqual(rt.calls[0]["Accept"], "application/x-ndjson")

    def test_stream_failures_are_retryable(self):
        for events in ([{"PayloadPart": {"Bytes": b'{"type":"accepted"}\n'}}],
                       [{"PayloadPart": {"Bytes": b'{"type":"accepted"}\n'}}, {"ModelStreamError": {"Message": "x"}}],
                       [{"PayloadPart": {"Bytes": b"{broken"}}]):
            with self.subTest(events=len(events)):
                r = invoke(FakeRuntime({"Body": events}), "ep", {}, stream=True)
                self.assertEqual((r.http_status, r.body["error"]["code"], r.retryable), (503, "stream_interrupted", True))

    def test_retry_policy(self):
        sleeps = []
        busy = model_error(503, {"status": "failed", "error": {"code": "llm_throttled", "retryable": True}})
        rt = FakeRuntime(busy, busy, {"Body": FakeBody(json.dumps(OK_BODY).encode())})
        result, attempts = invoke_with_retry(rt, "ep", {"request_id": "r"}, sleep=sleeps.append, rand=lambda: 1.0)
        self.assertEqual((result.ok, attempts, sleeps), (True, 3, [1.0, 2.0]))
        bad = model_error(400, {"status": "failed", "error": {"code": "invalid_request", "retryable": False}})
        result, attempts = invoke_with_retry(FakeRuntime(bad), "ep", {}, sleep=sleeps.append)
        self.assertEqual((result.http_status, attempts), (400, 1))
        result, attempts = invoke_with_retry(FakeRuntime(*[busy] * 2), "ep", {}, max_attempts=2, sleep=lambda _s: None)
        self.assertEqual((result.retryable, attempts), (True, 2))

    def test_in_body_retryable_flag_is_honoured_even_on_http_200(self):
        self.assertTrue(InvokeResult(200, {"status": "failed", "error": {"retryable": True}}).retryable)
        self.assertFalse(InvokeResult(200, {"status": "failed", "error": {"retryable": False}}).retryable)


class TabularTests(unittest.TestCase):
    def test_coerce_to_list_handles_every_shape_seen_in_extracts(self):
        cases = [
            (np.array(["J211", "N179"]), ["J211", "N179"]),
            ("['J211' 'N179' 'E119']", ["J211", "N179", "E119"]),   # numpy repr: no commas
            ("['3208400', '3209100']", ["3208400", "3209100"]),
            ('["a", "b"]', ["a", "b"]), ("J211;N179", ["J211", "N179"]), ("[]", []), (None, []), (float("nan"), []),
            ((1, 2), [1, 2]),
        ]
        for cell, expected in cases:
            with self.subTest(cell=repr(cell)):
                self.assertEqual(coerce_to_list(cell), expected)

    def test_jsonable(self):
        self.assertEqual(jsonable(np.int64(5)), 5)
        self.assertIsNone(jsonable(float("nan")))
        self.assertIsNone(jsonable(pd.NaT))
        self.assertEqual(jsonable(pd.Timestamp("2026-08-01")), "2026-08-01T00:00:00")
        self.assertEqual(jsonable({"poa": np.int64(1), "code": "J189"}), {"poa": 1, "code": "J189"})

    def test_row_to_payload_produces_a_request_the_service_accepts(self):
        ep = sample_episode()
        row = {
            **ep, "admission_no": np.int64(555123),
            "diagnosis_code_array": np.array(ep["diagnosis_code_array"]),
            "diagnosis_poa_array": np.array([{"code": c, "poa": np.int64(1)} for c in ep["diagnosis_code_array"]], dtype=object),
            "condition_onset_array": str(ep["condition_onset_array"]),
            "dx_code_idx_array": np.array(ep["dx_code_idx_array"]),
            "procedure_code_array": "['3237500' '9200900']",
            "procedure_date_array": np.array([pd.Timestamp("2026-08-02"), pd.NaT], dtype=object),
            "dte_admitted": pd.Timestamp("2026-08-01"), "dte_separated": pd.Timestamp("2026-08-06"),
            "member_dob": pd.Timestamp("1952-01-01"), "infant_weight": float("nan"), "not_a_field": "dropped",
        }
        payload = row_to_payload(row)
        self.assertNotIn("not_a_field", payload["episode"])
        self.assertEqual(payload["request_id"], "adm-555123")
        req = parse_request(payload, make_settings())
        self.assertEqual(req.episode.diagnosis_codes, ("J189", "I509", "N179", "E119"))
        self.assertEqual(req.episode.procedure_dates[1], None)
        self.assertEqual(row_to_payload(row, mode="verdicts_only")["options"], {"mode": "verdicts_only"})
        json.dumps(payload)  # fully JSON-serialisable


class MetricsTests(unittest.TestCase):
    def test_known_values(self):
        pairs = [(True, True)] * 6 + [(False, True)] * 2 + [(True, False)] * 4 + [(False, False)] * 8
        m = compute_metrics(confusion(pairs))
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (6, 2, 4, 8))
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.6)
        self.assertAlmostEqual(m["f1"], 2 * 0.75 * 0.6 / 1.35)
        self.assertAlmostEqual(m["accuracy"], 0.7)
        self.assertAlmostEqual(m["specificity"], 0.8)
        self.assertAlmostEqual(m["mcc"], (6 * 8 - 2 * 4) / (8 * 10 * 10 * 12) ** 0.5)

    def test_degenerate_cases_return_none_not_errors(self):
        m = compute_metrics(confusion([(False, False)] * 3))
        self.assertIsNone(m["precision"]); self.assertIsNone(m["recall"]); self.assertIsNone(m["mcc"])
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)


class BenchmarkAndSweepTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_rows(self):
        rows = []
        # (admission_no, ground_truth, levels for J189/I509/N179/E119)
        spec = [(1, True, {"N179": ("none", 0.9)}), (2, True, {"I509": ("weak", 0.3)}), (3, False, {}),
                (4, False, {"E119": ("weak", 0.9)}), (5, True, {}), (6, False, {"N179": ("none", 0.9)})]
        self.levels = {}
        for adm, truth, levels in spec:
            rows.append({**sample_episode(admission_no=adm), "Recommended": truth})
            self.levels[adm] = levels
        return rows

    def service_invoker(self):
        from audit_agent.deadline import Deadline
        from audit_agent.engine import AuditEngine
        from audit_agent.reference_data import ReferenceData
        from audit_agent.service import AuditService
        settings = make_settings()
        ref = ReferenceData.from_directory(make_reference_dir(self.root / "ref"))

        def invoke(payload):
            adm = payload["episode"]["admission_no"]
            engine = AuditEngine(reference=ref, llm=PolicyLlm(self.levels[adm]), grouper=None, settings=settings)
            service = AuditService(settings=settings, engine=engine, reference=ref)
            r = service.audit(service.validate(payload), Deadline(50))
            return r.status_code, r.body
        return invoke

    def test_benchmark_outputs_and_metrics(self):
        results = benchmark.run_benchmark(self.make_rows(), self.service_invoker(), workers=2)
        metrics = benchmark.summarise(results)
        # episodes 1 and 2 (truth True) flagged; 5 (truth True) missed; 6 (truth False) flagged; 3 and 4 clean
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 1, 1, 2))
        self.assertEqual(metrics["n_errored"], 0)
        self.assertEqual(metrics["provenance"]["prompt_version"], "pdx-adx-support/2026-10-15.1")
        out = self.root / "out"
        benchmark.write_outputs(results, out, metrics)
        self.assertEqual({p.name for p in out.iterdir()},
                         {"benchmark_results.csv", "benchmark_episodes.csv", "benchmark_metrics.json", "benchmark_report.md"})
        df = pd.read_csv(out / "benchmark_results.csv")
        self.assertEqual(list(df.columns), benchmark.VERDICT_COLUMNS)
        self.assertEqual(len(df), 6 * 4)  # PDx + 3 ADx per episode
        report = (out / "benchmark_report.md").read_text()
        self.assertIn("Caution: only 6 scored episodes", report)
        self.assertIn("| Human found unsupported | 2 (TP) | 1 (FN) |", report)
        self.assertNotIn("Cotiviti", report)

    def test_errors_are_excluded_from_metrics_and_reported(self):
        def flaky(payload):
            if payload["episode"]["admission_no"] == 2:
                raise RuntimeError("boom")
            return self.service_invoker()(payload)
        results = benchmark.run_benchmark(self.make_rows(), flaky, workers=1)
        m = benchmark.summarise(results)
        self.assertEqual((m["n_total"], m["n_scored"], m["n_errored"]), (6, 5, 1))
        self.assertEqual(m["error_codes"], {"RuntimeError": 1})
        out = self.root / "out"
        benchmark.write_outputs(results, out, m)
        df = pd.read_csv(out / "benchmark_results.csv")
        self.assertEqual(df[df["error"].notna()]["admission_no"].tolist(), [2])

    def test_sweep_current_rule_matches_what_the_service_decided(self):
        results = benchmark.run_benchmark(self.make_rows(), self.service_invoker(), workers=1)
        out = self.root / "out"
        benchmark.write_outputs(results, out, benchmark.summarise(results))
        df = pd.read_csv(out / "benchmark_results.csv")
        df["ground_truth_unsupported"] = df["ground_truth_unsupported"].astype(bool)
        rows = dict(sweep_decision_rules.evaluate(df, 0.5))
        current = next(m for name, m in rows.items() if name.startswith("CURRENT"))
        service_metrics = benchmark.summarise(results)
        for key in ("tp", "fp", "fn", "tn"):
            self.assertEqual(current[key], service_metrics[key])
        self.assertIn('"none" only', rows)
        self.assertEqual(rows['"none" only']["tp"], 1)  # only episode 1 has a "none" verdict among the true positives
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(sweep_decision_rules.main([str(out / "benchmark_results.csv")]), 0)
        self.assertIn("CURRENT", buf.getvalue())

    def test_sweep_rejects_a_per_episode_csv(self):
        path = self.root / "old.csv"
        pd.DataFrame({"admission_no": [1], "support_level": ["none"]}).to_csv(path, index=False)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sweep_decision_rules.main([str(path)]), 2)

    def test_reference_coverage_gate(self):
        ref_dir = make_reference_dir(self.root / "cov_ref")
        # CSV round-trips arrays as their text form, the exact shape coerce_to_list exists for
        good = pd.DataFrame([{"admission_no": 1, "diagnosis_code_array": np.array(["J189", "I509"]),
                              "procedure_code_array": np.array(["3237500"]), "medical_claims_array": np.array(["3"])}])
        bad = pd.DataFrame([{"admission_no": 2, "diagnosis_code_array": np.array(["J189", "ZZ999"]),
                             "procedure_code_array": np.array(["1"]), "medical_claims_array": np.array(["99999"])}])
        for name, df, expected in (("good", good, 0), ("bad", bad, 1)):
            path = self.root / f"{name}.csv"
            df.to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                code = check_reference_coverage.main(["--input", str(path), "--reference-dir", str(ref_dir)])
            self.assertEqual(code, expected, name)


class BatchTests(unittest.TestCase):
    def test_alert_file_is_rebuilt_without_duplicates_and_honours_the_latest_result(self):
        alert = lambda n: {"claim_seqno": str(n), "alert_info": "x", "recovery_amount": n, "alert_instance_id": 1}
        records = [
            {"admission_no": 1, "body": {"flagged": True, "alert": alert(1)}},
            {"admission_no": 2, "body": {"flagged": True, "alert": alert(2)}},
            {"admission_no": 1, "body": {"flagged": True, "alert": alert(11)}},   # re-run supersedes
            {"admission_no": 2, "body": {"flagged": False, "alert": None}},       # no longer an alert
            {"admission_no": 3, "body": {"flagged": False, "alert": None}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts_json.txt"
            self.assertEqual(batch_invoke.write_alert_file(records, path), 1)
            lines = path.read_text().splitlines()
            self.assertEqual([json.loads(x)["claim_seqno"] for x in lines], ["11"])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_jsonl_reader_skips_a_torn_final_line_and_files_are_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = batch_invoke.private_dir(Path(tmp) / "out")
            path = d / "results.jsonl"
            with batch_invoke.private_append(path) as fh:
                fh.write('{"admission_no": 1}\n{"admission_no": 2')
            self.assertEqual(batch_invoke.read_jsonl(path), [{"admission_no": 1}])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700)


if __name__ == "__main__":
    unittest.main()
