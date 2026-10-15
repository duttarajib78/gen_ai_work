"""File-backed backends for development, the CLI, and resumable local runs (single process only)."""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import NotFoundError
from ..util import atomic_write_bytes, atomic_write_text, validate_object_key, validate_object_prefix
from .memory import InMemoryStateStore


class LocalStateStore(InMemoryStateStore):
    """In-memory store that snapshots to ``<root>/state.json`` after every write and reloads it on start (resume)."""

    def __init__(self, root: Path | str) -> None:
        super().__init__()
        self._path = Path(root) / "state.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.is_file():
            snapshot = json.loads(self._path.read_text(encoding="utf-8"))
            self._jobs = snapshot.get("jobs", {})
            self._units = snapshot.get("units", {})
            self._idem = {k: tuple(v) for k, v in snapshot.get("idem", {}).items()}

    def _persist(self) -> None:
        atomic_write_text(self._path, json.dumps({"jobs": self._jobs, "units": self._units,
                                                  "idem": {k: list(v) for k, v in self._idem.items()}}))


class LocalArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_object_key(key)
        return self.root / key

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        atomic_write_bytes(self._path(key), data)
        return self.uri(key)

    def put_file(self, key: str, path: Path, *, content_type: str = "application/octet-stream") -> str:
        return self.put_bytes(key, Path(path).read_bytes(), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise NotFoundError(f"artifact not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str) -> list[str]:
        validate_object_prefix(prefix)
        base = self.root
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and str(p.relative_to(base)).startswith(prefix))

    def uri(self, key: str) -> str:
        return str(self._path(key))
