"""Readiness / liveness shared between the supervisor, the worker and the web process via small files.

``/ping`` must answer within 2 seconds and must reflect *this process group's* health only - never Bedrock or
DynamoDB reachability, otherwise a downstream outage would make SageMaker replace healthy instances.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..settings import Settings
from ..util import atomic_write_text

READY_FILE = "ready.json"
HEARTBEAT_FILE = "worker.heartbeat"


def run_dir(settings: Settings) -> Path:
    return settings.paths.work_dir / "run"


def write_ready(directory: Path, info: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / READY_FILE, json.dumps(info, default=str))


def touch_heartbeat(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / HEARTBEAT_FILE, str(time.time()))


def check_health(directory: Path, *, stale_after: float = 90.0, now: float | None = None) -> tuple[bool, dict[str, Any]]:
    now = time.time() if now is None else now
    if not (directory / READY_FILE).is_file():
        return False, {"reason": "startup checks have not completed"}
    beat = directory / HEARTBEAT_FILE
    if not beat.is_file():
        return False, {"reason": "worker has not started"}
    age = now - beat.stat().st_mtime
    if age > stale_after:
        return False, {"reason": "worker heartbeat is stale", "age_s": round(age, 1)}
    return True, {"worker_heartbeat_age_s": round(age, 1)}
