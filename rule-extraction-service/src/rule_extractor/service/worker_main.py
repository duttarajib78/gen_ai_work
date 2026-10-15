"""Worker process entry point: ``python -m rule_extractor worker``."""
from __future__ import annotations

import asyncio
import logging
import os
import signal

from ..agents.sdk_runner import ClaudeSdkRunner
from ..logging_setup import configure_logging
from ..pipeline.processor import UnitProcessor
from ..settings import Settings
from .container import build_backends, build_metrics, open_corpus
from .health import run_dir
from .janitor import Janitor
from .jobs import JobService
from .worker import Worker
from ..pipeline.finalize import Finalizer

log = logging.getLogger(__name__)


def main() -> int:
    settings = Settings.load()
    configure_logging(os.environ.get("RULE_EXTRACTOR_LOG_LEVEL", "INFO"))
    backends, corpus = build_backends(settings), open_corpus(settings)
    jobs = JobService(settings, backends, corpus, finalizer=Finalizer(settings, backends.artifacts, corpus))
    processor = UnitProcessor(settings, corpus, ClaudeSdkRunner(settings), backends.artifacts)
    worker = Worker(settings=settings, jobs=jobs, processor=processor, metrics=build_metrics(settings),
                    janitor=Janitor(settings, jobs), run_dir=run_dir(settings))

    async def runner() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await worker.run(stop)

    asyncio.run(runner())
    return 0
