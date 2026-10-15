"""The HTTP service (Flask test client), result assembly, and the reference client."""
from __future__ import annotations

import json
import logging
import threading
import time
import unittest

from dx_support.client import EndpointClient, EndpointError, iter_ndjson
from dx_support.llm.fake import FakeBackend
from dx_support.logging_setup import JsonFormatter
from dx_support.sample_data import synthetic_universe
from dx_support.service.app import create_app
from dx_support.service.results import assemble_category
from dx_support.service.runtime import ServiceRuntime
from dx_support.service.schemas import RequestError, parse_support_request
from dx_support.settings import Settings
from dx_support.storage.checkpoint import Checkpoint
from dx_support.storage.objectstore import LocalObjectStore
from tests.helpers import run, tmpdir

UNIVERSE = synthetic_universe()
RUNTIMES: list[ServiceRuntime] = []


def make_runtime(delay_s: float = 0.0, **overrides) -> ServiceRuntime:
    env = {"DX_ENV": "test", "DX_BACKEND": "fake", "DX_STORE_URI": tmpdir().as_uri()}
    settings = Settings.from_env(env).with_overrides(**overrides)
    rt = ServiceRuntime(settings, universe=UNIVERSE, backend=FakeBackend(delay_s=delay_s)).start()
    RUNTIMES.append(rt)
    return rt


def post(client, body=None, ctype="application/json", raw=None):
    data = raw if raw is not None else json.dumps(body)
    return client.post("/invocations", data=data, content_type=ctype)


def tearDownModule():
    for rt in RUNTIMES:
        rt.shutdown(timeout=2)


class ContractTests(unittest.TestCase):
    def test_ping_reflects_readiness(self):
        rt = ServiceRuntime(Settings.from_env({"DX_ENV": "test", "DX_BACKEND": "fake"}))
        c = create_app(rt).test_client()
        self.assertEqual(c.get("/ping").status_code, 503)
        self.assertEqual(post(c, {"categories": ["H25"]}).status_code, 503)
        ready = create_app(make_runtime()).test_client()
        r = ready.get("/ping")
        self.assertEqual((r.status_code, r.get_json()), (200, {"status": "ok"}))

    def test_malformed_requests_are_rejected_with_json_errors(self):
        c = create_app(make_runtime()).test_client()
        cases = [
            (dict(raw="{not json"), 400, "invalid_json"), (dict(raw="[1,2]"), 400, "invalid_json"), (dict(raw=b"\xff\xfe"), 400, "invalid_json"),
            (dict(body={"categories": ["H25"], "mode": "exhaustive"}), 400, "unknown_field"),
            (dict(body={"action": "drop_tables"}), 400, "unknown_action"),
            (dict(body={}), 422, "missing_categories"),
            (dict(body={"categories": ["h25;rm"]}), 422, "invalid_category"), (dict(body={"categories": "H25"}), 422, "invalid_categories"),
            (dict(body={"categories": ["Z99"]}), 422, "unknown_category"), (dict(body={"categories": ["V01"]}), 422, "context_only_category"),
            (dict(body={"categories": ["H25", "I48", "K35", "K37"]}), 422, "too_many_categories"),
            (dict(body={"categories": ["H25"], "families": ["achi", "evil"]}), 422, "invalid_families"),
            (dict(body={"categories": ["H25"], "merge_tiers": ["great"]}), 422, "invalid_merge_tiers"),
            (dict(body={"categories": ["H25"], "include_audit": "yes"}), 422, "invalid_include_audit"),
            (dict(body={"categories": ["H25"], "prosthesis_extract": "newest"}), 422, "invalid_prosthesis_extract"),
            (dict(body={"categories": ["H25"], "request_id": "x y"}), 422, "invalid_request_id"),
            (dict(body={"codes": ["H999X"]}), 422, "unknown_code"),
            (dict(body={"categories": ["H25"]}, ctype="text/plain"), 415, "unsupported_media_type"),
        ]
        for kw, status, code in cases:
            r = post(c, **kw)
            self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (status, code), kw)
        big = post(c, None, raw=json.dumps({"categories": ["H25"], "pad": "x" * 70_000}))
        self.assertEqual((big.status_code, big.get_json()["error"]["code"]), (413, "payload_too_large"))
        self.assertEqual(c.get("/nope").get_json()["error"]["code"], "not_found")
        self.assertEqual(c.get("/invocations").status_code, 405)

    def test_error_bodies_never_echo_hostile_input(self):
        c = create_app(make_runtime()).test_client()
        r = post(c, {"categories": ["<script>alert(1)</script>"]})
        self.assertNotIn("script", r.get_data(as_text=True))
        r = post(c, {"categories": ["H25"], "evil<b>": 1})
        self.assertEqual(r.status_code, 400)

    def test_codes_are_mapped_to_their_category(self):
        rt = make_runtime()
        req = parse_support_request({"codes": ["h251", "I480"], "categories": ["H25"]}, rt.settings, UNIVERSE, rt.leaf_map)
        self.assertEqual(req.categories, ("H25", "I48"))
        with self.assertRaises(RequestError):
            parse_support_request({"categories": ["V01"]}, rt.settings, UNIVERSE, rt.leaf_map)
        ok = parse_support_request({"categories": ["V01"]}, rt.settings.with_overrides(include_context_chapters=True), UNIVERSE, rt.leaf_map)
        self.assertEqual(ok.categories, ("V01",))

    def test_describe_reports_state_without_leaking_the_store(self):
        rt = make_runtime()
        r = post(create_app(rt).test_client(), {"action": "describe"})
        body = r.get_json()
        self.assertEqual((r.status_code, body["fingerprint"], body["backend"]), (200, rt.fingerprint, "fake"))
        self.assertIn("limits", body)
        self.assertNotIn("store", json.dumps(body).lower().replace("restore", ""))


class ServingTests(unittest.TestCase):
    def test_cold_request_computes_then_the_same_request_is_a_free_cache_hit(self):
        rt = make_runtime()
        c = create_app(rt).test_client()
        r1 = post(c, {"categories": ["I48"], "families": ["achi", "mbs"], "request_id": "t-1"})
        b1 = r1.get_json()
        self.assertEqual((r1.status_code, b1["status"], b1["request_id"]), (200, "complete", "t-1"))
        self.assertFalse(b1["usage"]["cache_hit"])
        self.assertGreater(b1["usage"]["llm_calls"], 0)
        fams = b1["results"]["I48"]["families"]
        self.assertEqual({f: v["state"] for f, v in fams.items()}, {"achi": "complete", "mbs": "complete"})
        self.assertTrue(any(i["support_tier"] in ("core", "broad") for v in fams.values() for i in v["items"]))
        self.assertEqual(b1["notice"], [rt.settings.notice_line_1, rt.settings.notice_line_2])
        calls_before = rt.backend.n_calls
        b2 = post(c, {"categories": ["I48"], "families": ["achi", "mbs"]}).get_json()
        self.assertEqual((b2["status"], b2["usage"]["cache_hit"], b2["usage"]["llm_calls"], b2["usage"]["cost_usd"]), ("complete", True, 0, 0.0))
        self.assertEqual(rt.backend.n_calls, calls_before)                            # a hit makes no model call
        self.assertEqual(b2["results"]["I48"]["families"]["achi"]["items"], fams["achi"]["items"])
        b3 = post(c, {"categories": ["I48"], "families": ["achi"], "merge_tiers": ["core"]}).get_json()     # a subset of what is stored: still free
        self.assertTrue(b3["usage"]["cache_hit"])
        self.assertTrue(all(i["support_tier"] == "core" for i in b3["results"]["I48"]["families"]["achi"]["items"]))
        self.assertEqual(rt.backend.n_calls, calls_before)

    def test_slow_work_answers_202_keeps_running_and_the_retry_is_free(self):
        rt = make_runtime(delay_s=0.04, sync_deadline_s=0.05)
        c = create_app(rt).test_client()
        r = post(c, {"categories": ["I48"], "families": ["achi"]})
        b = r.get_json()
        self.assertEqual((r.status_code, b["status"], b["pending"], r.headers.get("Retry-After")), (202, "in_progress", ["I48"], "20"))
        self.assertIn("progress", b["results"]["I48"])
        for _ in range(100):                                                           # the job keeps going after the response has gone
            if rt.is_cached("I48", ("achi",)):
                break
            time.sleep(0.05)
        else:
            self.fail("the background job never finished")
        calls = rt.backend.n_calls
        again = post(c, {"categories": ["I48"], "families": ["achi"]})
        self.assertEqual((again.status_code, again.get_json()["status"]), (200, "complete"))
        self.assertEqual(rt.backend.n_calls, calls)

    def test_concurrent_identical_requests_do_not_pay_twice(self):
        solo = make_runtime()
        post(create_app(solo).test_client(), {"categories": ["K35"], "families": ["achi", "mbs"]})
        expected = solo.backend.n_calls
        rt = make_runtime(delay_s=0.01, sync_deadline_s=20)
        app = create_app(rt)
        out = []

        def hit():
            out.append(post(app.test_client(), {"categories": ["K35"], "families": ["achi", "mbs"]}).get_json()["status"])
        threads = [threading.Thread(target=hit) for _ in range(4)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        self.assertEqual(out, ["complete"] * 4)
        self.assertEqual(rt.backend.n_calls, expected)

    def test_cache_only_and_disabled_compute_never_call_the_model(self):
        for kw in ({}, {"allow_compute": False}):
            rt = make_runtime(**kw)
            c = create_app(rt).test_client()
            body = {"categories": ["H25"], "families": ["achi"]}
            if not kw:
                body["cache_only"] = True
            r = post(c, body)
            b = r.get_json()
            self.assertEqual((r.status_code, b["status"], b["results"]["H25"]["status"]), (200, "partial", "not_cached"))
            self.assertEqual(rt.backend.n_calls, 0)

    def test_the_per_category_cost_cap_stops_a_job_and_the_response_says_so(self):
        rt = make_runtime(max_cost_usd_per_category=0.0001)
        r = post(create_app(rt).test_client(), {"categories": ["H25"], "families": ["achi"]})
        b = r.get_json()
        self.assertEqual((r.status_code, b["status"]), (200, "partial"))
        self.assertIn(b["results"]["H25"]["status"], ("partial", "failed"))
        self.assertIn("cap", b["results"]["H25"].get("error", ""))
        self.assertLess(rt.backend.n_calls, 4)

    def test_the_hourly_budget_is_a_circuit_breaker_for_the_instance(self):
        rt = make_runtime(max_cost_usd_per_hour=0.0001)
        b = post(create_app(rt).test_client(), {"categories": ["H25"], "families": ["achi"]}).get_json()
        self.assertEqual(b["status"], "partial")
        self.assertIn("rolling budget", b["results"]["H25"].get("error", ""))

    def test_admission_control_returns_429_when_the_instance_is_busy(self):
        rt = make_runtime(delay_s=0.15, max_concurrent_requests=1, sync_deadline_s=20)
        app = create_app(rt)
        first = threading.Thread(target=lambda: post(app.test_client(), {"categories": ["H25"], "families": ["achi"]}))
        first.start()
        time.sleep(0.1)
        r = post(app.test_client(), {"action": "describe"})
        self.assertEqual((r.status_code, r.get_json()["error"]["code"], r.headers.get("Retry-After")), (429, "too_many_requests", "5"))
        first.join(30)
        self.assertEqual(post(app.test_client(), {"action": "describe"}).status_code, 200)     # the slot was released

    def test_streaming_emits_accepted_progress_and_a_final_result(self):
        rt = make_runtime(delay_s=0.03, stream_heartbeat_s=0.05)
        c = create_app(rt).test_client()
        r = post(c, {"categories": ["H25"], "families": ["achi", "mbs"], "stream": True, "request_id": "s-1"})
        self.assertEqual((r.status_code, r.mimetype), (200, "application/jsonlines"))
        events = [json.loads(line) for line in r.get_data(as_text=True).splitlines() if line.strip()]
        names = [e["event"] for e in events]
        self.assertEqual((names[0], names[-1]), ("accepted", "result"))
        self.assertIn("progress", names)
        self.assertTrue(all(e["request_id"] == "s-1" for e in events))
        self.assertEqual(events[-1]["status"], "complete")
        self.assertEqual(events[0]["cache_hit"], False)
        again = [json.loads(x) for x in post(c, {"categories": ["H25"], "families": ["achi", "mbs"], "stream": True}).get_data(as_text=True).splitlines() if x]
        self.assertEqual([e["event"] for e in again], ["accepted", "result"])             # stored results: no waiting, no progress events
        self.assertTrue(again[0]["cache_hit"])
        self.assertEqual(post(c, {"action": "describe"}).status_code, 200)                  # the stream released its admission slot

    def test_audit_and_dx_family(self):
        rt = make_runtime()
        c = create_app(rt).test_client()
        b = post(c, {"categories": ["H25"], "families": ["achi", "dx"], "include_audit": True}).get_json()
        res = b["results"]["H25"]
        self.assertEqual(res["status"], "complete")
        self.assertTrue(any(a["support_tier"] == "not_supporting" for a in res["audit"]))     # every verdict is available for review
        self.assertTrue(all("unit_text" in a and "merged" in a for a in res["audit"]))
        dx_items = res["families"]["dx"]["items"]
        self.assertFalse(any(i["code"] == "H25" for i in dx_items))                           # never its own supporting diagnosis

    def test_logs_carry_no_reference_text_prompts_or_model_output(self):
        records: list[str] = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(JsonFormatter().format(record))
        root = logging.getLogger()
        h = Grab()
        old_level = root.level
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        logging.getLogger("dx_support.calls").setLevel(logging.INFO)
        try:
            rt = make_runtime()
            post(create_app(rt).test_client(), {"categories": ["H25"], "families": ["achi", "mbs"]})
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)
            logging.getLogger("dx_support.calls").setLevel(logging.NOTSET)
        text = "\n".join(records)
        self.assertIn("request accepted", text)
        for secret in ("Insertion of intraocular lens", "Cardioversion", "dry-run stub brief", "stub: "):
            self.assertNotIn(secret, text)
        self.assertTrue(all(json.loads(line)["request_id"] for line in records))


class ResultAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.store = LocalObjectStore(tmpdir())
        self.ck = Checkpoint(self.store)
        self.pros = UNIVERSE.indices["prosthesis"]
        self.lens = next(u for u in self.pros.units.values() if "EG009" in u.codes)

    def v(self, family, uid, tier, ev="treated", scope="all", ts="2026-01-01T00:00:00"):
        return {"category": "H25", "family": family, "unit_id": uid, "support_tier": tier, "evidence_type": ev, "rationale": "r", "member_scope": scope,
                "source": "route", "model": "m", "ts": ts}

    def build(self, records, families_complete):
        run(self.ck.write_verdicts(records))
        run(self.ck.write_status({"category": "H25", "status": "complete", "mode": "routed", "families_complete": families_complete,
                                  "families": {f: {"n_candidates": 3} for f in families_complete}}))

    def test_a_prosthesis_verdict_expands_to_every_billing_code_and_the_latest_extract_filter_applies(self):
        self.build([self.v("prosthesis", self.lens.unit_id, "core")], ["prosthesis"])
        kw = dict(ckpt=self.ck, universe=UNIVERSE, category="H25", families=["prosthesis"], merge_tiers=["core", "broad"])
        items = assemble_category(**kw)["families"]["prosthesis"]["items"]
        self.assertEqual(sorted(i["code"] for i in items), ["EG009", "EG010"])
        self.assertTrue(all(i["device_type"] and i["unit_id"] == self.lens.unit_id for i in items))
        latest = assemble_category(**kw, prosthesis_extract="latest")["families"]["prosthesis"]["items"]
        self.assertEqual([i["code"] for i in latest], ["EG009"])

    def test_merge_tiers_sorting_unknown_units_and_states(self):
        achi = UNIVERSE.indices["achi"]
        ids = sorted(achi.units)[:4]
        self.build([self.v("achi", ids[0], "broad"), self.v("achi", ids[1], "core"), self.v("achi", ids[2], "not_supporting", "none"),
                    self.v("achi", "9999999", "core"), self.v("achi", ids[3], "unclear", "none")], ["achi"])
        kw = dict(ckpt=self.ck, universe=UNIVERSE, category="H25", families=["achi"])
        both = assemble_category(**kw, merge_tiers=["core", "broad"], include_audit=True)
        items = both["families"]["achi"]["items"]
        self.assertEqual([i["support_tier"] for i in items], ["core", "broad"])           # core first
        self.assertEqual(both["families"]["achi"]["counts"], {"core": 1, "broad": 1})
        self.assertNotIn("9999999", [i["unit_id"] for i in both["audit"]])                # a unit the index does not know is never served
        self.assertEqual(len(both["audit"]), 4)
        self.assertEqual(sum(a["merged"] for a in both["audit"]), 2)
        strict = assemble_category(**kw, merge_tiers=["core"])
        self.assertEqual(len(strict["families"]["achi"]["items"]), 1)
        self.assertEqual(strict["status"], "complete")
        mixed = assemble_category(ckpt=self.ck, universe=UNIVERSE, category="H25", families=["achi", "mbs"], merge_tiers=["core"])
        self.assertEqual((mixed["status"], mixed["families"]["mbs"]["state"]), ("partial", "pending"))

    def test_the_newest_verdict_wins(self):
        achi_id = sorted(UNIVERSE.indices["achi"].units)[0]
        self.build([self.v("achi", achi_id, "not_supporting", "none", ts="2026-01-01T00:00:00")], ["achi"])
        run(self.ck.write_verdicts([self.v("achi", achi_id, "core", ts="2026-03-01T00:00:00")]))
        res = assemble_category(ckpt=self.ck, universe=UNIVERSE, category="H25", families=["achi"], merge_tiers=["core"])
        self.assertEqual([i["support_tier"] for i in res["families"]["achi"]["items"]], ["core"])


class FakeSageMakerRuntime:
    """Scripted sagemaker-runtime client: `sync` and `streams` are consumed in order."""

    def __init__(self, sync, streams=()):
        self.sync, self.streams, self.calls = list(sync), list(streams), []

    def invoke_endpoint(self, **kw):
        self.calls.append(("sync", json.loads(kw["Body"])))
        r = self.sync.pop(0)
        if isinstance(r, BaseException):
            raise r
        status, body = r
        return {"Body": type("B", (), {"read": lambda s: json.dumps(body).encode()})(), "ResponseMetadata": {"HTTPStatusCode": status}}

    def invoke_endpoint_with_response_stream(self, **kw):
        self.calls.append(("stream", json.loads(kw["Body"])))
        chunks = self.streams.pop(0)
        return {"Body": [{"PayloadPart": {"Bytes": c}} for c in chunks]}


class ModelError(Exception):
    def __init__(self, status):
        super().__init__(f"ModelError {status}")
        self.response = {"OriginalStatusCode": status, "Error": {"Code": "ModelError"}}


def line(**kw):
    return (json.dumps(kw) + "\n").encode()


class ClientTests(unittest.TestCase):
    def client(self, sync, streams=()):
        fake = FakeSageMakerRuntime(sync, streams)
        return EndpointClient("ep", client=fake, sleep=lambda s: None), fake

    def test_ndjson_is_reassembled_across_part_boundaries(self):
        blob = line(event="accepted") + line(event="result", status="complete")
        parts = [{"PayloadPart": {"Bytes": blob[:7]}}, {"PayloadPart": {"Bytes": blob[7:30]}}, {"PayloadPart": {"Bytes": blob[30:]}}, {"Other": 1}]
        self.assertEqual([e["event"] for e in iter_ndjson(parts)], ["accepted", "result"])

    def test_a_cached_answer_needs_one_call(self):
        c, fake = self.client([(200, {"status": "complete", "results": {}})])
        self.assertEqual(c.get_lists(categories=["H25"])["status"], "complete")
        self.assertEqual([k for k, _ in fake.calls], ["sync"])

    def test_a_202_switches_to_the_streaming_api_and_returns_the_final_result(self):
        seen = []
        c, fake = self.client([(202, {"status": "in_progress"})], [[line(event="accepted"), line(event="progress", elapsed_s=5), line(event="result", status="complete", results={"H25": {}})]])
        out = c.get_lists(categories=["H25"], on_progress=seen.append)
        self.assertEqual(out["status"], "complete")
        self.assertNotIn("event", out)
        self.assertEqual(len(seen), 1)
        self.assertEqual([k for k, _ in fake.calls], ["sync", "stream"])
        self.assertTrue(fake.calls[1][1]["stream"])

    def test_a_cut_stream_reconnects_and_a_busy_instance_is_retried_with_back_off(self):
        c, fake = self.client([ModelError(429), (202, {"status": "in_progress"})],
                              [[line(event="accepted"), line(event="progress")], [line(event="accepted"), line(event="result", status="complete")]])
        self.assertEqual(c.get_lists(categories=["H25"])["status"], "complete")
        self.assertEqual([k for k, _ in fake.calls], ["sync", "sync", "stream", "stream"])

    def test_rejected_requests_are_not_retried_and_timeouts_say_the_work_continues(self):
        c, fake = self.client([ModelError(422)])
        with self.assertRaises(EndpointError) as e:
            c.get_lists(categories=["ZZZ"])
        self.assertEqual(e.exception.status, 422)
        self.assertEqual(len(fake.calls), 1)
        c2, _ = self.client([(202, {"status": "in_progress"})], [[line(event="result", status="in_progress")]])
        with self.assertRaises(EndpointError) as t:
            c2.get_lists(categories=["H25"], timeout_s=0)
        self.assertIn("calling again resumes", str(t.exception))
        c3, _ = self.client([(202, {"status": "in_progress"})], [[line(event="error", error={"code": "internal_error", "message": "boom"})]])
        with self.assertRaises(EndpointError):
            c3.get_lists(categories=["H25"])


if __name__ == "__main__":
    unittest.main()
