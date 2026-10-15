"""Run a whole job locally (CLI / SageMaker Code Editor) with the SAME worker/processor/finalizer used in production.

State is file-backed (``checkpoints/state.json``) so an interrupted run resumes: the job id is derived from the options
(idempotency key), finished units are skipped and only failed/unfinished ones are retried - the original ``--resume``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..agents.sdk_runner import ClaudeSdkRunner
from ..agents.types import AgentRunner
from ..domain.api import JobOptions
from ..domain.jobs import ACTIVE_JOB_STATUSES, JobStatus
from ..service.container import open_corpus
from ..service.janitor import Janitor
from ..service.jobs import JobService
from ..service.worker import Worker
from ..settings import Settings
from ..storage.interfaces import Backends
from ..storage.local import LocalArtifactStore, LocalStateStore
from ..storage.memory import InMemoryWorkQueue
from .finalize import Finalizer
from .processor import UnitProcessor

log = logging.getLogger(__name__)
CONVENIENCE_COPIES = (("catalogue", "rule_catalogue.xlsx"), ("executable_rules", "executable_rules.jsonl"),
                      ("mbs_facts", "mbs_billing_facts_reference.jsonl"))


async def run_local(options: JobOptions, settings: Settings, *, resume: bool = True, runner: AgentRunner | None = None,
                    poll_interval: float = 0.5, on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if runner is None:
        settings.validate_for_agents()
        runner = ClaudeSdkRunner(settings)
    corpus = open_corpus(settings)
    backends = Backends(state=LocalStateStore(settings.paths.state_dir), artifacts=LocalArtifactStore(settings.paths.output_dir),
                        queue=InMemoryWorkQueue())
    jobs = JobService(settings, backends, corpus, finalizer=Finalizer(settings, backends.artifacts, corpus))
    worker = Worker(settings=settings, jobs=jobs, processor=UnitProcessor(settings, corpus, runner, backends.artifacts),
                    janitor=Janitor(settings, jobs), poll_wait=0.2, idle_sleep=0.1, janitor_interval=30.0)

    view = jobs.submit(options, f"local-{options.digest()[:24]}" if resume else None)
    job_id = view["job_id"]
    if view.get("deduplicated"):
        status = backends.state.get_job(job_id).status
        if status in (JobStatus.PARTIAL.value, JobStatus.FAILED.value):
            jobs.retry_failed(job_id)                       # re-run only what failed
        elif status in ACTIVE_JOB_STATUSES:
            jobs.resume(job_id, reset_running=True)         # interrupted run: nothing is running any more
            jobs.activate(job_id)
        else:
            return _finish(jobs, backends, settings, job_id)

    stop = asyncio.Event()

    async def watch() -> None:
        while not backends.state.get_job(job_id).is_terminal:
            if on_progress:
                on_progress(jobs.get(job_id))
            await asyncio.sleep(poll_interval)
        stop.set()

    started = time.monotonic()
    await asyncio.gather(worker.run(stop), watch())
    summary = _finish(jobs, backends, settings, job_id)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return summary


def _finish(jobs: JobService, backends: Backends, settings: Settings, job_id: str) -> dict[str, Any]:
    view = jobs.get(job_id)
    store = backends.artifacts
    for artifact, filename in CONVENIENCE_COPIES:      # copy headline deliverables next to the job folder for convenience
        if artifact in view["artifacts"]:
            key = f"jobs/{job_id}/outputs/{filename}"
            if store.exists(key):
                (settings.paths.output_dir / filename).write_bytes(store.get_bytes(key))
    return view
