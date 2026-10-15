"""Dependency wiring shared by the web process, the worker process, and the local CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..engine.grouper import GrouperAdapter, load_grouper
from ..engine.rules_store import ExecutableRule, load_all
from ..errors import UnavailableError
from ..ingestion.corpus import CorpusIndex
from ..pipeline.finalize import Finalizer
from ..settings import Settings
from ..storage.interfaces import Backends
from ..storage.local import LocalArtifactStore, LocalStateStore
from ..storage.memory import InMemoryWorkQueue
from .health import run_dir
from .jobs import JobService
from .metrics import CloudWatchMetrics, MetricsSink, NullMetrics


def build_backends(settings: Settings) -> Backends:
    if settings.service.backend == "aws":
        from ..storage.aws import build_aws_backends
        return build_aws_backends(settings)
    return Backends(state=LocalStateStore(settings.paths.state_dir),
                    artifacts=LocalArtifactStore(settings.paths.output_dir),
                    queue=InMemoryWorkQueue())


def build_metrics(settings: Settings) -> MetricsSink:
    if settings.service.backend != "aws":
        return NullMetrics()
    import boto3
    from botocore.config import Config
    client = boto3.client("cloudwatch", config=Config(region_name=settings.bedrock.region, retries={"max_attempts": 3, "mode": "standard"},
                                                      connect_timeout=3, read_timeout=10))
    return CloudWatchMetrics(client, settings.aws.metrics_namespace, settings.aws.endpoint_name)


def open_corpus(settings: Settings) -> CorpusIndex | None:
    return CorpusIndex(settings.paths.corpus_dir) if (settings.paths.corpus_dir / "manifest.json").is_file() else None


@dataclass
class ServiceContainer:
    settings: Settings
    jobs: JobService
    corpus: CorpusIndex | None
    backends: Backends
    run_dir: Path
    health: Callable[[], tuple[bool, dict[str, Any]]]
    _rules: list[ExecutableRule] | None = field(default=None, repr=False)
    _grouper: GrouperAdapter | None = field(default=None, repr=False)

    def executable_rules(self) -> list[ExecutableRule]:
        if self._rules is None:
            path = self.settings.paths.model_dir / "rules" / "executable_rules.jsonl"
            rules = load_all(path)
            if not rules:
                raise UnavailableError("no executable rules are packaged in this model artifact (rules/executable_rules.jsonl)")
            self._rules = rules
        return self._rules

    def grouper(self) -> GrouperAdapter:
        if self._grouper is None:
            self._grouper = load_grouper(self.settings.grouper_factory)
        return self._grouper


def build_container(settings: Settings, *, backends: Backends | None = None, corpus: CorpusIndex | None = None,
                    health: Callable[[], tuple[bool, dict[str, Any]]] | None = None, with_finalizer: bool = False) -> ServiceContainer:
    backends = backends or build_backends(settings)
    corpus = corpus if corpus is not None else open_corpus(settings)
    finalizer = Finalizer(settings, backends.artifacts, corpus) if with_finalizer else None
    jobs = JobService(settings, backends, corpus, finalizer=finalizer)
    return ServiceContainer(settings=settings, jobs=jobs, corpus=corpus, backends=backends, run_dir=run_dir(settings),
                            health=health or (lambda: (True, {})))
