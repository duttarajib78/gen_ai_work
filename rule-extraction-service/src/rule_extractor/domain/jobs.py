"""Job / unit records and the work-queue message format."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

JOB_ID_RE = re.compile(r"^job-[0-9a-f]{12,32}$")
UNIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}$")


class JobStatus(str, Enum):
    SUBMITTING = "SUBMITTING"
    RUNNING = "RUNNING"
    FINALIZING = "FINALIZING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_JOB_STATUSES = frozenset({JobStatus.SUBMITTING.value, JobStatus.RUNNING.value, JobStatus.FINALIZING.value})
TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.SUCCEEDED.value, JobStatus.PARTIAL.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
)


class UnitStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_UNIT_STATUSES = frozenset(
    {UnitStatus.SUCCEEDED.value, UnitStatus.FAILED.value, UnitStatus.CANCELLED.value}
)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class JobRecord:
    job_id: str
    kind: str = "extraction"
    status: str = JobStatus.SUBMITTING.value
    options: dict[str, Any] = field(default_factory=dict)
    options_digest: str = ""
    idempotency_key: str | None = None
    created_at: str = ""
    updated_at: str = ""
    created_epoch: float = 0.0
    updated_epoch: float = 0.0
    deadline_epoch: float = 0.0
    units_total: int = 0
    cost_usd: float = 0.0
    plan: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    version_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        return _from_dict(cls, data)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES


@dataclass
class UnitRecord:
    job_id: str
    unit_id: str
    kind: str
    order: int
    spec: dict[str, Any] = field(default_factory=dict)
    status: str = UnitStatus.PENDING.value
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    rules_key: str | None = None
    rules_count: int = 0
    rejected_count: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnitRecord":
        return _from_dict(cls, data)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_UNIT_STATUSES


@dataclass(frozen=True)
class WorkItem:
    """Queue message body. Deliberately tiny: identifiers only, never document text."""

    job_id: str
    unit_id: str

    def to_json(self) -> str:
        return json.dumps({"v": 1, "job_id": self.job_id, "unit_id": self.unit_id}, separators=(",", ":"))

    @classmethod
    def from_json(cls, body: str) -> "WorkItem":
        if len(body) > 2048:
            raise ValueError("work item too large")
        data = json.loads(body)
        if not isinstance(data, dict) or data.get("v") != 1:
            raise ValueError("unsupported work item version")
        job_id, unit_id = data.get("job_id"), data.get("unit_id")
        if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
            raise ValueError("invalid job_id")
        if not isinstance(unit_id, str) or not UNIT_ID_RE.match(unit_id):
            raise ValueError("invalid unit_id")
        return cls(job_id=job_id, unit_id=unit_id)
