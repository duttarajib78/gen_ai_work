"""Ports. The service layer only depends on these protocols, so the same code runs against DynamoDB/SQS/S3 in
SageMaker and against local files / memory for development and tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ..domain.jobs import JobRecord, UnitRecord


class StateStore(Protocol):
    def reserve_idempotency(self, key: str, job_id: str, options_digest: str) -> tuple[str, str] | None:
        """Atomically bind ``key`` to ``job_id``. ``None`` means reserved by the caller; otherwise the existing
        ``(job_id, options_digest)`` that already owns the key."""

    def put_job(self, job: JobRecord) -> None:
        """Create a job; raises ``ConflictError`` if the id exists."""

    def get_job(self, job_id: str) -> JobRecord | None: ...

    def update_job(self, job_id: str, mutate: Callable[[JobRecord], bool]) -> tuple[bool, JobRecord | None]:
        """Optimistic read-modify-write. ``mutate`` edits a copy in place and returns True to persist. It may be
        re-applied on contention, so it must be a pure function of the record."""

    def put_units(self, units: Sequence[UnitRecord]) -> None: ...

    def get_unit(self, job_id: str, unit_id: str) -> UnitRecord | None: ...

    def update_unit(self, job_id: str, unit_id: str, mutate: Callable[[UnitRecord], bool]) -> tuple[bool, UnitRecord | None]: ...

    def list_units(self, job_id: str) -> list[UnitRecord]: ...

    def list_jobs_by_status(self, status: str, limit: int = 100) -> list[JobRecord]: ...


class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str: ...
    def put_file(self, key: str, path: Path, *, content_type: str = "application/octet-stream") -> str: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def uri(self, key: str) -> str: ...


@dataclass
class QueueMessage:
    handle: Any
    body: str
    receive_count: int = 1
    message_id: str = ""


class WorkQueue(Protocol):
    def send_batch(self, bodies: Sequence[str]) -> None: ...
    def receive(self, max_messages: int, wait_seconds: int, visibility_timeout: int) -> list[QueueMessage]: ...
    def ack(self, message: QueueMessage) -> None: ...
    def extend(self, message: QueueMessage, visibility_timeout: int) -> None:
        """Set the remaining invisibility to ``visibility_timeout`` seconds (0 releases the message immediately)."""
    def depth(self) -> tuple[int, int]:
        """(visible, in_flight)"""


@dataclass
class Backends:
    state: StateStore
    artifacts: ArtifactStore
    queue: WorkQueue
    metrics_sink: Any = None
