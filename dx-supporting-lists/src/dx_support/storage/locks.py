"""Single-run guard on an object store.

Created atomically (O_EXCL locally, an S3 conditional write remotely), so two overlapping runs against the same namespace are refused even if
started at the same instant. A stale lock (a crashed run) is REPORTED with its holder, never deleted automatically: delete it yourself once you
are sure nothing is running.
"""
from __future__ import annotations

import os
import socket
from typing import Optional

from dx_support.storage.checkpoint import now_ts
from dx_support.storage.objectstore import ObjectStore

LOCK_KEY = ".run.lock"


class RunLock:
    def __init__(self, store: ObjectStore, key: str = LOCK_KEY):
        self.store = store
        self.key = key
        self._held = False

    def __enter__(self) -> "RunLock":
        info = f"pid={os.getpid()} host={socket.gethostname().replace(' ', '_')} started={now_ts()}"
        if not self.store.put_if_absent(self.key, info.encode("utf-8")):
            holder = (self.store.get(self.key) or b"unreadable")[:120].decode("utf-8", errors="replace")
            raise SystemExit(
                f"{self.store.uri}/{self.key} already exists ({holder}). Either another run is in progress, or a previous run crashed without "
                "cleaning up. If you are sure nothing is running, delete this object and try again.")
        self._held = True
        return self

    def __exit__(self, *exc) -> None:
        if self._held:
            self.store.delete(self.key)
            self._held = False

    @staticmethod
    def read(store: ObjectStore, key: str = LOCK_KEY) -> Optional[dict[str, str]]:
        raw = store.get(key)
        if raw is None:
            return None
        return dict(tok.split("=", 1) for tok in raw.decode("utf-8", errors="replace").split() if "=" in tok)
