import asyncio
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from rule_extractor import cli
from rule_extractor.domain.api import JobOptions
from rule_extractor.errors import AgentError, ConfigError
from rule_extractor.ingestion.corpus import CorpusIndex
from rule_extractor.llm.bedrock import BLANKED_ENV
from rule_extractor.model_package import build_model_dir
from rule_extractor.pipeline.local_run import run_local
from rule_extractor.service import supervisor
from rule_extractor.service.app import create_app
from rule_extractor.service.container import build_container
from rule_extractor.service.health import HEARTBEAT_FILE, READY_FILE, check_health, touch_heartbeat, write_ready
from rule_extractor.settings import Settings
from rule_extractor.storage.interfaces import Backends
from rule_extractor.storage.memory import InMemoryArtifactStore, InMemoryStateStore, InMemoryWorkQueue

from .fakes import FakeRunner, emit_default
from .helpers import make_env, make_env_dict

RULE_LINE = json.dumps({"rule_id": "RULE_004", "recommended_action": "remove_code", "automatable": True,
                        "flag_condition": {"all": [{"has_diagnosis": "A41.0"}, {"has_diagnosis": "B95.6"}]},
                        "correction_action": {"action": "remove_code", "target": "diagnosis", "match": "B95.6"}})


class AppBase(unittest.TestCase):
    ENV: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_env(self.tmp.name, **self.ENV)
        self.backends = Backends(InMemoryStateStore(), InMemoryArtifactStore(), InMemoryWorkQueue())
        self.container = build_container(self.settings, backends=self.backends, corpus=CorpusIndex(self.settings.paths.corpus_dir),
                                         with_finalizer=True)
        self.healthy = [True]
        self.container.health = lambda: (self.healthy[0], {} if self.healthy[0] else {"reason": "test"})
        self.client = create_app(self.container).test_client()

    def call(self, payload, *, content_type="application/json", raw=None, headers=None):
        data = raw if raw is not None else json.dumps(payload)
        return self.client.post("/invocations", data=data, content_type=content_type, headers=headers or {})

    def ok(self, payload):
        resp = self.call(payload)
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertTrue(body["ok"])
        return body["data"]


class HttpContractTests(AppBase):
    def test_ping_reflects_local_health_only(self):
        self.assertEqual(self.client.get("/ping").status_code, 200)
        self.assertEqual(self.client.post("/ping").status_code, 200)
        self.healthy[0] = False
        resp = self.client.get("/ping")
        self.assertEqual((resp.status_code, resp.get_json()["status"]), (503, "unhealthy"))

    def test_plan_submit_get_cancel_roundtrip(self):
        plan = self.ok({"operation": "plan", "options": {"topics": ["0110"], "include_tabular_sweep": False}})
        self.assertEqual(plan["summary"]["units_total"], 1)
        sub = self.ok({"operation": "submit_job", "idempotency_key": "client-key-0001", "options": {"topics": ["0110"], "include_tabular_sweep": False}})
        self.assertEqual((sub["status"], sub["units_total"]), ("RUNNING", 1))
        again = self.ok({"operation": "submit_job", "idempotency_key": "client-key-0001", "options": {"topics": ["0110"], "include_tabular_sweep": False}})
        self.assertTrue(again["deduplicated"])
        got = self.ok({"operation": "get_job", "job_id": sub["job_id"], "include_units": True})
        self.assertEqual(got["units"][0]["unit_id"], "acs-0110")
        self.assertEqual(self.ok({"operation": "cancel_job", "job_id": sub["job_id"]})["cancel_requested"], True)
        self.assertEqual(len(self.ok({"operation": "list_jobs", "status": "RUNNING"})["jobs"]), 1)

    def test_health_operation_exposes_versions_but_no_secrets(self):
        data = self.ok({"operation": "health"})
        self.assertEqual(data["status"], "ok")
        self.assertIn("prompt_digest", data["versions"])
        text = json.dumps(data).lower()
        for needle in ("secret", "token", "password", "credential"):
            self.assertNotIn(needle, text)

    def test_request_validation(self):
        cases = [
            (dict(raw="{not json"), 400, "invalid_json"),
            (dict(raw=json.dumps([1, 2])), 400, "invalid_request"),
            (dict(payload={"operation": "drop_tables"}), 400, "invalid_request"),
            (dict(payload={}), 400, "invalid_request"),
            (dict(payload={"operation": "plan", "options": {"topics": ["abc"]}}), 400, "invalid_request"),
            (dict(payload={"operation": "plan", "options": {"bogus": 1}}), 400, "invalid_request"),
            (dict(payload={"operation": "plan", "surprise": 1}), 400, "invalid_request"),
            (dict(payload={"operation": "get_job", "job_id": "../../etc/passwd"}), 400, "invalid_request"),
            (dict(payload={"operation": "get_job", "job_id": "job-ffffffffffff"}), 404, "not_found"),
            (dict(payload={"operation": "get_job", "job_id": "job-ffffffffffff", "limit": 0}), 400, "invalid_request"),
            (dict(payload={"operation": "submit_job", "idempotency_key": "x"}), 400, "invalid_request"),
            (dict(payload={"operation": "plan", "options": {"include_web_research": True, "web_research_topics": ["Coding rules"]}}), 403, "forbidden_by_policy"),
            (dict(payload={"operation": "evaluate_episodes", "episodes": [{}]}), 403, "forbidden_by_policy"),
            (dict(payload={"operation": "retry_failed", "job_id": "job-ffffffffffff"}), 404, "not_found"),
            (dict(payload={"operation": "plan"}, content_type="text/plain"), 415, "unsupported_media_type"),
        ]
        for kwargs, status, code in cases:
            with self.subTest(kwargs=str(kwargs)[:80]):
                resp = self.call(kwargs.get("payload"), **{k: v for k, v in kwargs.items() if k != "payload"})
                self.assertEqual(resp.status_code, status, resp.get_data(as_text=True))
                body = resp.get_json()
                self.assertEqual((body["ok"], body["error"]["code"]), (False, code))
                self.assertTrue(body["request_id"])

    def test_conflict_when_retrying_a_running_job(self):
        job = self.ok({"operation": "submit_job", "options": {"topics": ["0110"], "include_tabular_sweep": False}})
        resp = self.call({"operation": "retry_failed", "job_id": job["job_id"]})
        self.assertEqual((resp.status_code, resp.get_json()["error"]["code"]), (409, "conflict"))

    def test_unexpected_errors_do_not_leak_internals(self):
        with mock.patch.object(self.container.jobs, "plan", side_effect=RuntimeError("secret path /etc/shadow")):
            resp = self.call({"operation": "plan"})
        self.assertEqual(resp.status_code, 500)
        text = resp.get_data(as_text=True)
        self.assertNotIn("secret path", text)
        self.assertNotIn("Traceback", text)
        self.assertEqual(resp.get_json()["error"]["code"], "internal_error")

    def test_headers_and_unknown_routes(self):
        resp = self.call({"operation": "health"}, headers={"X-Amzn-SageMaker-Custom-Attributes": "trace=abc<script>alert(1)</script>"})
        self.assertTrue(resp.headers["X-Amzn-SageMaker-Custom-Attributes"].startswith("request_id="))
        self.assertNotIn("<script>", json.dumps(dict(resp.headers)))
        self.assertEqual(resp.headers["Cache-Control"], "no-store")
        self.assertEqual(self.client.get("/invocations").status_code, 405)
        self.assertEqual(self.client.get("/nope").status_code, 404)


class SmallLimitTests(AppBase):
    ENV = {"RULE_EXTRACTOR_MAX_REQUEST_BYTES": "2048", "RULE_EXTRACTOR_ALLOWED_OPERATIONS": "health,plan"}

    def test_oversize_body_and_operation_allowlist(self):
        resp = self.call(None, raw=json.dumps({"operation": "plan", "pad": "x" * 5000}))
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.get_json()["error"]["code"], "payload_too_large")
        resp = self.call({"operation": "submit_job"})
        self.assertEqual((resp.status_code, resp.get_json()["error"]["code"]), (403, "forbidden_by_policy"))
        self.assertEqual(self.call({"operation": "health"}).status_code, 200)


class EvaluateTests(AppBase):
    ENV = {"RULE_EXTRACTOR_ENABLE_EVALUATE": "true", "RULE_EXTRACTOR_MAX_EPISODES_PER_REQUEST": "2"}

    def setUp(self):
        super().setUp()
        rules = Path(self.settings.paths.model_dir) / "rules"
        rules.mkdir(parents=True)
        (rules / "executable_rules.jsonl").write_text(RULE_LINE + "\n")

    def test_flags_and_previews_without_echoing_episode_content(self):
        resp = self.call({"operation": "evaluate_episodes", "episodes": [
            {"episode_id": "e1", "diagnosis_code_array": ["A41.0", "B95.6"]}, {"episode_id": "e2", "diagnosis_code_array": ["E11.9"]}]})
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        data = resp.get_json()["data"]
        self.assertTrue(data["preview_only"] and data["grouper"]["is_mock"])
        e1, e2 = data["results"]
        self.assertEqual(e1["flagged"][0]["rule_id"], "RULE_004")
        self.assertEqual(e1["flagged"][0]["cost_weight_delta"], -1.0)
        self.assertEqual(e2["flagged"], [])
        self.assertNotIn("A41.0", resp.get_data(as_text=True))

    def test_bounds_and_strict_episode_validation(self):
        too_many = [{"episode_id": str(i)} for i in range(3)]
        for payload in ({"operation": "evaluate_episodes", "episodes": too_many},
                        {"operation": "evaluate_episodes", "episodes": [{"surprise": 1}]},
                        {"operation": "evaluate_episodes", "episodes": "x"},
                        {"operation": "evaluate_episodes", "episodes": [{"diagnosis_code_array": ["A"]}], "rule_ids": "x"}):
            self.assertEqual(self.call(payload).status_code, 400, payload)


class HealthFileTests(unittest.TestCase):
    def test_states(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(check_health(d)[0], False)
            write_ready(d, {"x": 1})
            self.assertIn("worker has not started", check_health(d)[1]["reason"])
            touch_heartbeat(d)
            self.assertTrue(check_health(d)[0])
            self.assertFalse(check_health(d, now=os.stat(d / HEARTBEAT_FILE).st_mtime + 500)[0])


class FakeProc:
    def __init__(self, rc=None):
        self._rc, self.signals, self.killed = rc, [], False

    def poll(self):
        return self._rc

    def send_signal(self, sig):
        self.signals.append(sig)
        self._rc = 0

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        self.killed, self._rc = True, -9


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_env(self.tmp.name, RULE_EXTRACTOR_SHUTDOWN_GRACE_SECONDS="1", RULE_EXTRACTOR_RUN_AS_UID=str(os.geteuid()))
        build_model_dir(self.settings.paths.corpus_dir, Path(self.tmp.name) / "template.xlsx", self.settings.paths.model_dir)
        self.procs: dict[str, FakeProc] = {}

    def popen_factory(self, **rcs):
        def popen(cmd, **kwargs):
            name = "web" if "gunicorn" in cmd else "worker"
            self.procs[name] = FakeProc(rcs.get(name))
            return self.procs[name]
        return popen

    class OkRunner:
        def __init__(self, settings): pass
        def preflight(self): return {"sdk_version": "test"}

    class BadRunner(OkRunner):
        def preflight(self): raise ConfigError("sdk too old")

    def run_sup(self, popen, runner=OkRunner, stop_after=0.4):
        stop = threading.Event()
        threading.Timer(stop_after, stop.set).start()
        return supervisor.run_supervisor(self.settings, popen=popen, runner_factory=runner, stop_event=stop, install_signals=False)

    def test_clean_shutdown_signals_both_children(self):
        self.assertEqual(self.run_sup(self.popen_factory()), 0)
        self.assertEqual(set(self.procs), {"web", "worker"})
        for proc in self.procs.values():
            self.assertEqual(len(proc.signals), 1)
        self.assertTrue((self.settings.paths.work_dir / "run" / READY_FILE).is_file())

    def test_child_death_stops_the_other_and_propagates_exit_code(self):
        self.assertEqual(self.run_sup(self.popen_factory(web=3), stop_after=5), 3)
        self.assertEqual(len(self.procs["worker"].signals), 1)

    def test_tampered_model_refuses_to_start(self):
        (self.settings.paths.model_dir / "corpus" / "acs" / "p0003.txt").write_text("tampered")
        self.assertEqual(self.run_sup(self.popen_factory()), 2)
        self.assertEqual(self.procs, {})

    def test_sdk_preflight_failure_refuses_to_start(self):
        self.assertEqual(self.run_sup(self.popen_factory(), runner=self.BadRunner), 3)
        self.assertEqual(self.procs, {})

    def test_command_and_env_construction(self):
        cmds = supervisor.child_commands(self.settings)
        self.assertIn("0.0.0.0:8080", cmds["web"])
        self.assertEqual(cmds["web"][cmds["web"].index("--timeout") + 1], "55")      # below SageMaker's 60 s limit
        self.assertEqual(cmds["worker"][-2:], ["rule_extractor", "worker"])
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x", "CLAUDE_CODE_OAUTH_TOKEN": "y", "AWS_REGION": "ap-southeast-2"}):
            env = supervisor.child_env(self.settings, Path("/tmp/h"))
        for name in BLANKED_ENV:
            self.assertNotIn(name, env)
        self.assertEqual((env["HOME"], env["AWS_REGION"]), ("/tmp/h", "ap-southeast-2"))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = make_env_dict(self.tmp.name)
        make_env(self.tmp.name)           # builds the corpus under the same paths
        patcher = mock.patch.dict(os.environ, self.env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_plan_and_filters(self):
        code, out, _ = self.run_cli("plan")
        self.assertEqual((code, json.loads(out)["summary"]["units_total"]), (0, 10))
        code, out, _ = self.run_cli("plan", "--topics", "0110", "--no-tabular-sweep")
        self.assertEqual([u["unit_id"] for u in json.loads(out)["units"]], ["acs-0110"])
        code, out, _ = self.run_cli("run", "--dry-run", "--limit-topics", "1", "--no-tabular-sweep")
        self.assertEqual(json.loads(out)["summary"]["units_total"], 1)

    def test_errors_exit_non_zero_with_message_on_stderr(self):
        code, out, err = self.run_cli("plan", "--topics", "9999")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("unknown ACS topic", err)
        code, _, err = self.run_cli("plan", "--web-research", "--web-topics", "Coding rules", "--no-acs", "--no-tabular-sweep")
        self.assertEqual(code, 1)
        self.assertIn("web research is disabled", err)

    def test_corpus_verify_build_package_and_check(self):
        code, out, _ = self.run_cli("corpus", "verify")
        self.assertEqual((code, json.loads(out)["ok"]), (0, True))
        code, out, _ = self.run_cli("corpus", "build", "--force", "--workers", "1")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["mbs"], {"items": 3, "facts": 2})
        dist = Path(self.tmp.name) / "dist" / "model.tar.gz"
        stage = Path(self.tmp.name) / "stage"
        code, out, _ = self.run_cli("package", "--out", str(dist), "--model-dir", str(stage), "--template", str(Path(self.tmp.name) / "template.xlsx"))
        report = json.loads(out)
        self.assertEqual((code, report["ok"], len(report["sha256"])), (0, True, 64))
        self.assertTrue(dist.is_file())
        code, out, _ = self.run_cli("check-model", "--model-dir", str(stage))
        self.assertEqual((code, json.loads(out)["ok"]), (0, True))
        (stage / "corpus" / "acs" / "p0001.txt").write_text("x")
        code, out, _ = self.run_cli("check-model", "--model-dir", str(stage))
        self.assertEqual(code, 1)

    def test_evaluate_command(self):
        rules = Path(self.tmp.name) / "r.jsonl"
        rules.write_text(RULE_LINE + "\n")
        eps = Path(self.tmp.name) / "e.jsonl"
        eps.write_text(json.dumps({"episode_id": "e1", "diagnosis_code_array": ["A41.0", "B95.6"]}) + "\n")
        code, out, _ = self.run_cli("evaluate", "--episodes", str(eps), "--rules", str(rules))
        result = json.loads(out)
        self.assertEqual((code, result["flagged"][0]["cost_weight_delta"]), (0, -1.0))


class LocalRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_env(self.tmp.name)

    def go(self, runner, resume=True, **opts):
        return asyncio.run(run_local(JobOptions.from_dict(opts), self.settings, resume=resume, runner=runner, poll_interval=0.05))

    def test_full_run_writes_outputs_and_resume_is_a_noop(self):
        runner = FakeRunner()
        view = self.go(runner)
        self.assertEqual((view["status"], view["progress"]["succeeded"]), ("SUCCEEDED", 10))
        out = self.settings.paths.output_dir
        for rel in (f"jobs/{view['job_id']}/outputs/rule_catalogue.xlsx", "rule_catalogue.xlsx", "executable_rules.jsonl",
                    "mbs_billing_facts_reference.jsonl"):
            self.assertTrue((out / rel).is_file(), rel)
        self.assertTrue((self.settings.paths.state_dir / "state.json").is_file())
        again_runner = FakeRunner()
        again = self.go(again_runner)
        self.assertEqual((again["job_id"], again_runner.tasks), (view["job_id"], []))

    def test_resume_reruns_only_failed_units(self):
        failing = FakeRunner(lambda t, n: (_ for _ in ()).throw(AgentError("nope")) if t.unit_id == "acs-0110" else emit_default(t))
        first = self.go(failing)
        self.assertEqual((first["status"], first["failures"][0]["unit_id"]), ("PARTIAL", "acs-0110"))
        healthy = FakeRunner()
        second = self.go(healthy)
        self.assertEqual([t.unit_id for t in healthy.tasks], ["acs-0110"])
        self.assertEqual((second["job_id"], second["status"]), (first["job_id"], "SUCCEEDED"))

    def test_no_resume_starts_a_new_job(self):
        a = self.go(FakeRunner(), topics=["0110"], include_tabular_sweep=False)
        b = self.go(FakeRunner(), resume=False, topics=["0110"], include_tabular_sweep=False)
        self.assertNotEqual(a["job_id"], b["job_id"])

    def test_interrupted_run_is_resumed(self):
        from rule_extractor.service.jobs import JobService
        from rule_extractor.storage.local import LocalArtifactStore, LocalStateStore
        options = JobOptions.from_dict({"topics": ["0110"], "include_tabular_sweep": False})
        backends = Backends(LocalStateStore(self.settings.paths.state_dir), LocalArtifactStore(self.settings.paths.output_dir), InMemoryWorkQueue())
        jobs = JobService(self.settings, backends, CorpusIndex(self.settings.paths.corpus_dir))
        view = jobs.submit(options, f"local-{options.digest()[:24]}")       # the process "dies" before any worker runs
        backends.state.update_unit(view["job_id"], "acs-0110", lambda u: setattr(u, "status", "RUNNING") or setattr(u, "lease_owner", "dead") or True)
        runner = FakeRunner()
        result = asyncio.run(run_local(options, self.settings, runner=runner, poll_interval=0.05))
        self.assertEqual((result["job_id"], result["status"], len(runner.tasks)), (view["job_id"], "SUCCEEDED", 1))


if __name__ == "__main__":
    unittest.main()
