import asyncio
import tempfile
import time
import unittest

from rule_extractor.domain.api import JobOptions
from rule_extractor.domain.jobs import JobRecord, JobStatus, UnitRecord, UnitStatus, WorkItem
from rule_extractor.errors import AgentError, ConflictError, NotFoundError, RequestValidationError
from rule_extractor.ingestion.corpus import CorpusIndex
from rule_extractor.pipeline.finalize import Finalizer
from rule_extractor.pipeline.processor import UnitProcessor
from rule_extractor.service.janitor import Janitor
from rule_extractor.service.jobs import JobService
from rule_extractor.service.metrics import ListMetrics
from rule_extractor.service.worker import Worker
from rule_extractor.storage.interfaces import Backends
from rule_extractor.storage.memory import InMemoryArtifactStore, InMemoryStateStore, InMemoryWorkQueue

from .fakes import FakeRunner, emit_default
from .helpers import make_env


async def _no_sleep(_):
    return None


class ServiceBase(unittest.TestCase):
    ENV: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_env(self.tmp.name, **self.ENV)
        self.corpus = CorpusIndex(self.settings.paths.corpus_dir)
        self.backends = Backends(InMemoryStateStore(), InMemoryArtifactStore(), InMemoryWorkQueue())
        self.jobs = JobService(self.settings, self.backends, self.corpus, finalizer=Finalizer(self.settings, self.backends.artifacts, self.corpus))
        self.metrics = ListMetrics()

    def worker(self, runner, **kw):
        proc = UnitProcessor(self.settings, self.corpus, runner, self.backends.artifacts, sleep=_no_sleep)
        kw.setdefault("poll_wait", 0.02)
        kw.setdefault("idle_sleep", 0.01)
        kw.setdefault("metrics", self.metrics)
        return Worker(settings=self.settings, jobs=self.jobs, processor=proc, **kw)

    def run_until_terminal(self, worker, job_id, timeout=30.0):
        async def main():
            stop = asyncio.Event()

            async def watch():
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if self.backends.state.get_job(job_id).is_terminal:
                        break
                    await asyncio.sleep(0.02)
                stop.set()

            await asyncio.gather(worker.run(stop), watch())

        asyncio.run(main())
        return self.backends.state.get_job(job_id)

    def submit(self, key=None, **opts):
        return self.jobs.submit(JobOptions.from_dict(opts), key)


class LifecycleTests(ServiceBase):
    def test_end_to_end_success(self):
        view = self.submit("key-0000-0001")
        self.assertEqual((view["status"], view["units_total"], view["deduplicated"]), ("RUNNING", 10, False))
        job = self.run_until_terminal(self.worker(FakeRunner()), view["job_id"])
        self.assertEqual(job.status, "SUCCEEDED")
        got = self.jobs.get(view["job_id"], include_units=True)
        self.assertEqual(got["progress"], {"pending": 0, "running": 0, "succeeded": 10, "failed": 0, "cancelled": 0, "done": 10, "total": 10})
        self.assertAlmostEqual(got["cost_usd"], 0.5)
        self.assertEqual(len(got["units"]), 10)
        self.assertEqual(self.backends.queue.depth(), (0, 0))
        page = self.jobs.get_rules(view["job_id"], 0, 4)
        self.assertEqual((page["total"], len(page["rules"]), page["next_offset"]), (10, 4, 4))
        self.assertIsNone(self.jobs.get_rules(view["job_id"], 8, 4)["next_offset"])
        self.assertIn("catalogue", self.jobs.get_results(view["job_id"])["artifacts"])
        for name in ("UnitsSucceeded", "UnitCostUSD", "RulesEmitted", "UnitDurationSeconds"):
            self.assertIn(name, self.metrics.names())

    def test_idempotent_submit(self):
        a = self.submit("key-0000-0002", topics=["0110"], include_tabular_sweep=False)
        b = self.submit("key-0000-0002", topics=["0110"], include_tabular_sweep=False)
        self.assertEqual((a["job_id"], b["job_id"], b["deduplicated"]), (a["job_id"], a["job_id"], True))
        self.assertEqual(self.backends.queue.depth(), (1, 0))                         # no second set of messages
        with self.assertRaises(ConflictError):
            self.submit("key-0000-0002", topics=["0111"], include_tabular_sweep=False)

    def test_non_transient_failure_gives_partial_job_with_failure_details(self):
        def flaky(task, n):
            if task.unit_id == "acs-0110":
                raise AgentError("bad request", transient=False)
            return emit_default(task)

        view = self.submit()
        job = self.run_until_terminal(self.worker(FakeRunner(flaky)), view["job_id"])
        self.assertEqual(job.status, "PARTIAL")
        got = self.jobs.get(view["job_id"])
        self.assertEqual(got["failures"][0]["unit_id"], "acs-0110")
        self.assertIn("bad request", got["failures"][0]["error"])
        self.assertEqual(got["progress"]["succeeded"], 9)
        self.assertEqual(self.jobs.get_rules(view["job_id"], 0, 100)["total"], 9)      # partial results are still delivered

    def test_retry_failed_path(self):
        state = {"fail": True}

        def flaky(task, n):
            if task.unit_id == "acs-0110" and state["fail"]:
                raise AgentError("bad request", transient=False)
            return emit_default(task)

        view = self.submit()
        self.assertEqual(self.run_until_terminal(self.worker(FakeRunner(flaky)), view["job_id"]).status, "PARTIAL")
        state["fail"] = False
        reopened = self.jobs.retry_failed(view["job_id"])
        self.assertEqual((reopened["status"], reopened["progress"]["pending"]), ("RUNNING", 1))
        job = self.run_until_terminal(self.worker(FakeRunner(flaky)), view["job_id"])
        self.assertEqual(job.status, "SUCCEEDED")
        self.assertEqual(self.jobs.get(view["job_id"])["progress"]["succeeded"], 10)
        with self.assertRaises(ConflictError):
            self.jobs.retry_failed(view["job_id"])

    def test_cancel_before_start_cancels_every_unit(self):
        view = self.submit()
        self.jobs.cancel(view["job_id"])
        job = self.run_until_terminal(self.worker(FakeRunner()), view["job_id"])
        self.assertEqual(job.status, "CANCELLED")
        self.assertEqual(self.jobs.get(view["job_id"])["progress"]["cancelled"], 10)

    def test_cancel_in_flight_aborts_the_session(self):
        async def slow(task, n):
            await asyncio.sleep(30)

        view = self.submit(topics=["0110"], include_tabular_sweep=False)

        async def main():
            stop = asyncio.Event()
            worker = self.worker(FakeRunner(slow), heartbeat_interval=0.05)

            async def canceller():
                while self.backends.state.get_unit(view["job_id"], "acs-0110").status != "RUNNING":
                    await asyncio.sleep(0.02)
                self.jobs.cancel(view["job_id"])
                while not self.backends.state.get_job(view["job_id"]).is_terminal:
                    await asyncio.sleep(0.02)
                stop.set()

            await asyncio.wait_for(asyncio.gather(worker.run(stop), canceller()), timeout=20)

        asyncio.run(main())
        self.assertEqual(self.backends.state.get_job(view["job_id"]).status, "CANCELLED")
        self.assertEqual(self.backends.state.get_unit(view["job_id"], "acs-0110").status, "CANCELLED")

    def test_graceful_shutdown_hands_units_back(self):
        async def slow(task, n):
            await asyncio.sleep(30)

        view = self.submit(topics=["0110"], include_tabular_sweep=False)

        async def main():
            stop = asyncio.Event()
            worker = self.worker(FakeRunner(slow))

            async def stopper():
                while self.backends.state.get_unit(view["job_id"], "acs-0110").status != "RUNNING":
                    await asyncio.sleep(0.02)
                stop.set()

            await asyncio.wait_for(asyncio.gather(worker.run(stop), stopper()), timeout=20)

        asyncio.run(main())
        unit = self.backends.state.get_unit(view["job_id"], "acs-0110")
        self.assertEqual((unit.status, unit.attempts, unit.lease_owner), ("PENDING", 0, None))
        self.assertEqual(self.backends.queue.depth(), (1, 0))                          # visible again for another instance
        job = self.run_until_terminal(self.worker(FakeRunner()), view["job_id"])       # a new worker finishes the job
        self.assertEqual(job.status, "SUCCEEDED")

    def test_poison_and_duplicate_messages(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        self.backends.queue.send_batch(["not json", WorkItem("job-ffffffffffff", "acs-0110").to_json()])
        job = self.run_until_terminal(self.worker(FakeRunner()), view["job_id"])
        self.assertEqual(job.status, "SUCCEEDED")
        self.assertIn("PoisonMessages", self.metrics.names())
        self.backends.queue.send_batch([WorkItem(view["job_id"], "acs-0110").to_json()])     # duplicate for a finished unit

        async def drain():
            stop = asyncio.Event()
            w = self.worker(FakeRunner())

            async def later():
                await asyncio.sleep(0.4)
                stop.set()
            await asyncio.gather(w.run(stop), later())

        asyncio.run(drain())
        visible, in_flight = self.backends.queue.depth()
        self.assertEqual(visible, 0)
        self.assertEqual(in_flight, 1)                                                       # only the poison message remains (-> DLQ)

    def test_validation_and_not_found(self):
        with self.assertRaises(RequestValidationError):
            self.jobs.get("bogus")
        with self.assertRaises(NotFoundError):
            self.jobs.get("job-ffffffffffff")
        with self.assertRaises(NotFoundError):
            self.jobs.cancel("job-ffffffffffff")
        self.assertEqual(self.jobs.plan(JobOptions.from_dict({}))["summary"]["units_total"], 10)

    def test_list_jobs(self):
        a = self.submit()
        self.submit(topics=["0110"], include_tabular_sweep=False)
        listed = self.jobs.list_jobs("RUNNING", 10)["jobs"]
        self.assertEqual(len(listed), 2)
        self.assertEqual(len(self.jobs.list_jobs(None, 1)["jobs"]), 1)
        self.assertEqual(self.jobs.list_jobs("SUCCEEDED")["jobs"], [])


class BudgetTests(ServiceBase):
    ENV = {"RULE_EXTRACTOR_CONCURRENCY": "1"}

    def test_budget_stops_dispatch(self):
        view = self.submit(max_cost_usd=0.06)
        job = self.run_until_terminal(self.worker(FakeRunner()), view["job_id"])
        got = self.jobs.get(view["job_id"])
        self.assertEqual(job.status, "PARTIAL")
        self.assertEqual(got["progress"]["succeeded"], 2)
        self.assertEqual(got["progress"]["cancelled"], 8)


class ClaimTests(ServiceBase):
    def test_crashed_workers_unit_is_taken_over_after_lease_expiry(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        job_id = view["job_id"]

        def crash(u):
            u.status, u.attempts, u.lease_owner, u.lease_expires = "RUNNING", 1, "dead-worker", time.time() - 10
            return True
        self.backends.state.update_unit(job_id, "acs-0110", crash)
        job = self.run_until_terminal(self.worker(FakeRunner()), job_id)
        self.assertEqual(job.status, "SUCCEEDED")
        self.assertEqual(self.backends.state.get_unit(job_id, "acs-0110").attempts, 2)

    def test_live_lease_held_by_another_worker_is_respected(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        job_id = view["job_id"]
        self.backends.state.update_unit(job_id, "acs-0110", lambda u: setattr(u, "status", "RUNNING") or setattr(u, "lease_owner", "other")
                                        or setattr(u, "lease_expires", time.time() + 600) or True)
        runner = FakeRunner()

        async def main():
            stop = asyncio.Event()
            w = self.worker(runner)

            async def later():
                await asyncio.sleep(0.4)
                stop.set()
            await asyncio.gather(w.run(stop), later())

        asyncio.run(main())
        self.assertEqual(runner.tasks, [])                                            # not processed twice
        self.assertEqual(self.backends.state.get_unit(job_id, "acs-0110").lease_owner, "other")

    def test_claim_budget_exhaustion_fails_the_unit(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        job_id = view["job_id"]
        self.backends.state.update_unit(job_id, "acs-0110", lambda u: setattr(u, "attempts", 3) or True)   # max_retries=2 -> 3 claims
        job = self.run_until_terminal(self.worker(FakeRunner()), job_id)
        self.assertEqual(job.status, "FAILED")
        self.assertIn("gave up", self.backends.state.get_unit(job_id, "acs-0110").error)


class JanitorTests(ServiceBase):
    def setUp(self):
        super().setUp()
        self.now = [time.time()]
        self.janitor = Janitor(self.settings, self.jobs, clock=lambda: self.now[0])

    def test_half_submitted_job_is_repaired(self):
        job = JobRecord(job_id="job-aaaaaaaaaaaa", status="SUBMITTING", updated_epoch=self.now[0] - 1000, units_total=1,
                        deadline_epoch=self.now[0] + 10_000)
        self.backends.state.put_job(job)
        self.backends.state.put_units([UnitRecord(job.job_id, "acs-0110", "acs_topic", 0, {"code": "0110", "title": "t", "start_page": 6, "end_page": 7})])
        self.assertEqual(self.janitor.sweep_once()["resubmitted"], 1)
        self.assertEqual(self.backends.state.get_job(job.job_id).status, "RUNNING")
        self.assertEqual(self.backends.queue.depth(), (1, 0))

    def test_exhausted_lease_and_deadline_are_resolved_and_job_finalized(self):
        view = self.submit(topics=["0110", "0111"], include_tabular_sweep=False)
        job_id = view["job_id"]
        self.backends.state.update_unit(job_id, "acs-0110", lambda u: setattr(u, "status", "RUNNING") or setattr(u, "attempts", 3)
                                        or setattr(u, "lease_expires", self.now[0] - 10_000) or True)
        self.backends.state.update_unit(job_id, "acs-0111", lambda u: setattr(u, "status", "SUCCEEDED") or setattr(u, "rules_key", None) or True)
        stats = self.janitor.sweep_once()
        self.assertEqual((stats["lease_exhausted"], stats["finalized"]), (1, 1))
        self.assertEqual(self.backends.state.get_job(job_id).status, "PARTIAL")

    def test_deadline_fails_pending_units(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        self.now[0] += self.settings.service.job_max_runtime_seconds + 5
        self.assertEqual(self.janitor.sweep_once()["aborted_units"], 1)
        unit = self.backends.state.get_unit(view["job_id"], "acs-0110")
        self.assertEqual((unit.status, unit.error), ("FAILED", "job deadline exceeded"))
        self.assertEqual(self.backends.state.get_job(view["job_id"]).status, "FAILED")

    def test_cancel_requested_cancels_pending_units(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        self.jobs.cancel(view["job_id"])
        self.janitor.sweep_once()
        self.assertEqual(self.backends.state.get_job(view["job_id"]).status, "CANCELLED")

    def test_stale_finalizing_is_retried(self):
        view = self.submit(topics=["0110"], include_tabular_sweep=False)
        job_id = view["job_id"]
        self.backends.state.update_unit(job_id, "acs-0110", lambda u: setattr(u, "status", "SUCCEEDED") or True)
        self.backends.state.update_job(job_id, lambda j: setattr(j, "status", "FINALIZING") or setattr(j, "updated_epoch", self.now[0] - 5000) or True)
        self.assertEqual(self.janitor.sweep_once()["finalize_retried"], 1)
        self.assertEqual(self.backends.state.get_job(job_id).status, "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
