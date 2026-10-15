"""Thread-safe in-memory backends (tests and the base of the local backends)."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Sequence

from ..domain.jobs import JobRecord, UnitRecord
from ..errors import ConflictError, NotFoundError
from ..util import validate_object_key, validate_object_prefix
from .interfaces import QueueMessage


class InMemoryStateStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._units: dict[str, dict[str, dict]] = {}
        self._idem: dict[str, tuple[str, str]] = {}

    # hook for subclasses that persist
    def _persist(self) -> None:  # pragma: no cover - overridden
        pass

    def reserve_idempotency(self, key: str, job_id: str, options_digest: str) -> tuple[str, str] | None:
        with self._lock:
            if key in self._idem:
                return self._idem[key]
            self._idem[key] = (job_id, options_digest)
            self._persist()
            return None

    def put_job(self, job: JobRecord) -> None:
        with self._lock:
            if job.job_id in self._jobs:
                raise ConflictError(f"job {job.job_id} already exists")
            self._jobs[job.job_id] = job.to_dict()
            self._units.setdefault(job.job_id, {})
            self._persist()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            data = self._jobs.get(job_id)
            return JobRecord.from_dict(data) if data else None

    def update_job(self, job_id: str, mutate: Callable[[JobRecord], bool]) -> tuple[bool, JobRecord | None]:
        with self._lock:
            data = self._jobs.get(job_id)
            if data is None:
                return False, None
            record = JobRecord.from_dict(data)
            if not mutate(record):
                return False, record
            self._jobs[job_id] = record.to_dict()
            self._persist()
            return True, record

    def put_units(self, units: Sequence[UnitRecord]) -> None:
        with self._lock:
            for unit in units:
                self._units.setdefault(unit.job_id, {})[unit.unit_id] = unit.to_dict()
            self._persist()

    def get_unit(self, job_id: str, unit_id: str) -> UnitRecord | None:
        with self._lock:
            data = self._units.get(job_id, {}).get(unit_id)
            return UnitRecord.from_dict(data) if data else None

    def update_unit(self, job_id: str, unit_id: str, mutate: Callable[[UnitRecord], bool]) -> tuple[bool, UnitRecord | None]:
        with self._lock:
            data = self._units.get(job_id, {}).get(unit_id)
            if data is None:
                return False, None
            record = UnitRecord.from_dict(data)
            if not mutate(record):
                return False, record
            self._units[job_id][unit_id] = record.to_dict()
            self._persist()
            return True, record

    def list_units(self, job_id: str) -> list[UnitRecord]:
        with self._lock:
            units = [UnitRecord.from_dict(d) for d in self._units.get(job_id, {}).values()]
        return sorted(units, key=lambda u: (u.order, u.unit_id))

    def list_jobs_by_status(self, status: str, limit: int = 100) -> list[JobRecord]:
        with self._lock:
            jobs = [JobRecord.from_dict(d) for d in self._jobs.values() if d["status"] == status]
        return sorted(jobs, key=lambda j: j.created_at)[:limit]


class InMemoryArtifactStore:
    def __init__(self, bucket: str = "memory") -> None:
        self._data: dict[str, tuple[bytes, str]] = {}
        self._bucket = bucket
        self._lock = threading.Lock()

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        validate_object_key(key)
        with self._lock:
            self._data[key] = (bytes(data), content_type)
        return self.uri(key)

    def put_file(self, key: str, path: Path, *, content_type: str = "application/octet-stream") -> str:
        return self.put_bytes(key, Path(path).read_bytes(), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        validate_object_key(key)
        with self._lock:
            if key not in self._data:
                raise NotFoundError(f"artifact not found: {key}")
            return self._data[key][0]

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def list_keys(self, prefix: str) -> list[str]:
        validate_object_prefix(prefix)
        with self._lock:
            return sorted(k for k in self._data if k.startswith(prefix))

    def uri(self, key: str) -> str:
        return f"memory://{self._bucket}/{key}"


class InMemoryWorkQueue:
    """SQS-like semantics: receive hides a message for ``visibility_timeout``; ack deletes; extend(0) releases."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._messages: dict[str, dict] = {}

    def send_batch(self, bodies: Sequence[str]) -> None:
        with self._lock:
            for body in bodies:
                mid = uuid.uuid4().hex
                self._messages[mid] = {"body": body, "visible_at": 0.0, "receives": 0, "handle": None}

    def receive(self, max_messages: int, wait_seconds: int, visibility_timeout: int) -> list[QueueMessage]:
        deadline = self._clock() + max(0, wait_seconds)
        while True:
            with self._lock:
                now = self._clock()
                ready = [(mid, m) for mid, m in self._messages.items() if m["visible_at"] <= now][:max_messages]
                out = []
                for mid, m in ready:
                    m["visible_at"] = now + visibility_timeout
                    m["receives"] += 1
                    m["handle"] = uuid.uuid4().hex
                    out.append(QueueMessage(handle=(mid, m["handle"]), body=m["body"], receive_count=m["receives"], message_id=mid))
            if out or self._clock() >= deadline:
                return out
            time.sleep(0.01)

    def ack(self, message: QueueMessage) -> None:
        mid, handle = message.handle
        with self._lock:
            current = self._messages.get(mid)
            if current and current["handle"] == handle:
                del self._messages[mid]

    def extend(self, message: QueueMessage, visibility_timeout: int) -> None:
        mid, handle = message.handle
        with self._lock:
            current = self._messages.get(mid)
            if current and current["handle"] == handle:
                current["visible_at"] = self._clock() + visibility_timeout

    def depth(self) -> tuple[int, int]:
        with self._lock:
            now = self._clock()
            visible = sum(1 for m in self._messages.values() if m["visible_at"] <= now)
            return visible, len(self._messages) - visible
