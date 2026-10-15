"""The resumable record of a run, kept in an ObjectStore.

Every decision the pipeline pays for is persisted the moment it exists, under a deterministic key, so an interrupted run (or a second instance of
the service) never buys the same decision twice:

    briefs/<CAT>.json                     the clinical brief                                   (so a brief is never re-bought)
    routes/<CAT>/<family>/<key>.json      one routing decision (selected node ids)             (so a routing decision is never re-bought)
    verdicts/<CAT>/<family>/<batch>.jsonl one judged batch, one verdict per line               (so a verdict is never re-bought)
    status/<CAT>.json                     the last status record (complete / partial, and why)
    runlog/<YYYYMMDD>/<run>-<seq>.jsonl   one record per model-call attempt (tokens, cost, duration; never a prompt)

Objects are never appended to (S3 cannot append) and every key is a pure function of its content, so writes are idempotent: two writers racing
on the same decision overwrite each other with an equivalent object.

Loading is lazy and per category (`hydrate`), which is what lets many instances share one store without loading all of it. Writes are
`async` and complete only when the object is durable, so ordering is simple: the pipeline writes a category's status record only after every
verdict batch it describes has been awaited.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable, Iterator, Optional

from dx_support.storage.objectstore import ObjectStore

logger = logging.getLogger(__name__)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def parse_jsonl(data: Optional[bytes]) -> Iterator[dict]:
    """Line-by-line parse (never pandas.read_json, whose dtype inference would coerce numeric-looking string codes to integers). Corrupt or
    partial lines are skipped, not fatal."""
    if not data:
        return
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield rec


def _dumps(rec: Any) -> bytes:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def latest_by_key(records: Iterable[dict], key_fields: tuple[str, ...] = ("category", "family", "unit_id")) -> dict[tuple, dict]:
    """The newest record (by its `ts`) for each key: objects are not ordered, so recency is read from the record, not from file order."""
    out: dict[tuple, dict] = {}
    for r in records:
        k = tuple(r.get(f) for f in key_fields)
        cur = out.get(k)
        if cur is None or str(r.get("ts", "")) >= str(cur.get("ts", "")):
            out[k] = r
    return out


class Checkpoint:
    def __init__(self, store: ObjectStore, io_workers: int = 8, run_log_flush_every: int = 100):
        self.store = store
        self.briefs: dict[str, dict] = {}
        self.routes: dict[tuple[str, str, str], list[str]] = {}
        self.verdict_keys: set[tuple[str, str, str]] = set()
        self.status: dict[str, dict] = {}
        self._hydrated: set[str] = set()
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="dx-ckpt")
        self._log_lock = threading.Lock()
        self._log_buf: list[dict] = []
        self._log_pending: list[Future] = []
        self._log_seq = 0
        self._run_id = uuid.uuid4().hex[:8]
        self._flush_every = run_log_flush_every

    # ------------------------------------------------------------------ reading
    def _get_many(self, keys: list[str]) -> list[Optional[bytes]]:
        return list(self._io.map(self.store.get, keys)) if keys else []

    def _iter_objects(self, keys: list[str], chunk: int = 64) -> Iterator[tuple[str, Optional[bytes]]]:
        for i in range(0, len(keys), chunk):
            part = keys[i: i + chunk]
            yield from zip(part, self._io.map(self.store.get, part))

    def hydrate(self, category: str, force: bool = False) -> None:
        """Loads everything stored for one category (brief, routing decisions, verdict keys, status). Blocking I/O: call `ahydrate` from async code."""
        if category in self._hydrated and not force:
            return
        brief_raw = self.store.get(f"briefs/{category}.json")
        status_raw = self.store.get(f"status/{category}.json")
        route_keys = self.store.list(f"routes/{category}/")
        verdict_objs = self.store.list(f"verdicts/{category}/")
        route_raws, verdict_raws = self._get_many(route_keys), self._get_many(verdict_objs)
        for r in parse_jsonl(brief_raw):
            if isinstance(r.get("brief"), dict):
                self.briefs[category] = r["brief"]
        for r in parse_jsonl(status_raw):
            self.status[category] = r
        for raw in route_raws:
            for r in parse_jsonl(raw):
                if {"category", "family", "key", "selected"} <= set(r):
                    self.routes[(r["category"], r["family"], r["key"])] = list(r["selected"])
        for raw in verdict_raws:
            for r in parse_jsonl(raw):
                if {"category", "family", "unit_id"} <= set(r):
                    self.verdict_keys.add((r["category"], r["family"], r["unit_id"]))
        self._hydrated.add(category)

    async def ahydrate(self, category: str, force: bool = False) -> None:
        await asyncio.to_thread(self.hydrate, category, force)

    def refresh_status(self, category: str) -> dict:
        """Re-reads ONLY the status object (one GET): the cheap check used to answer a cache hit."""
        for r in parse_jsonl(self.store.get(f"status/{category}.json")):
            self.status[category] = r
        return self.status.get(category, {})

    def load_status_all(self, categories: Optional[Iterable[str]] = None) -> dict[str, dict]:
        keys = [k for k in self.store.list("status/") if k.endswith(".json")]
        if categories is not None:
            want = {f"status/{c}.json" for c in categories}
            keys = [k for k in keys if k in want]
        out: dict[str, dict] = {}
        for raw in self._get_many(keys):
            for r in parse_jsonl(raw):
                if "category" in r:
                    out[r["category"]] = r
        self.status.update(out)
        return out

    def cached_brief_ids(self) -> set[str]:
        return {k.split("/", 1)[1][: -len(".json")] for k in self.store.list("briefs/") if k.endswith(".json")}

    def read_verdicts(self, category: str, family: Optional[str] = None) -> dict[tuple[str, str, str], dict]:
        """The newest verdict record per (category, family, unit) as stored."""
        prefix = f"verdicts/{category}/" + (f"{family}/" if family else "")
        recs = (r for _k, raw in self._iter_objects(self.store.list(prefix)) for r in parse_jsonl(raw))
        return {(k[0], k[1], k[2]): v for k, v in latest_by_key(recs).items()}

    def iter_verdicts(self, categories: Optional[Iterable[str]] = None) -> Iterator[dict]:
        prefixes = [f"verdicts/{c}/" for c in categories] if categories is not None else ["verdicts/"]
        for p in prefixes:
            for _k, raw in self._iter_objects(self.store.list(p)):
                yield from parse_jsonl(raw)

    def iter_briefs(self) -> Iterator[dict]:
        for _k, raw in self._iter_objects([k for k in self.store.list("briefs/") if k.endswith(".json")]):
            yield from parse_jsonl(raw)

    def iter_run_log(self) -> Iterator[dict]:
        for _k, raw in self._iter_objects(self.store.list("runlog/")):
            yield from parse_jsonl(raw)

    def read_json(self, key: str) -> Optional[dict]:
        for r in parse_jsonl(self.store.get(key)):
            return r
        return None

    def write_json(self, key: str, obj: dict) -> None:
        self.store.put(key, _dumps(obj))

    # ------------------------------------------------------------------ writing (durable when the await returns)
    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._io, fn, *args)

    async def write_brief(self, category: str, brief: dict) -> None:
        await self._run(self.store.put, f"briefs/{category}.json", _dumps({"category": category, "brief": brief, "ts": now_ts()}))
        self.briefs[category] = brief

    async def write_route(self, category: str, family: str, level: int, key: str, selected: list[str]) -> None:
        rec = {"category": category, "family": family, "level": level, "key": key, "selected": list(selected), "ts": now_ts()}
        await self._run(self.store.put, f"routes/{category}/{family}/{key}.json", _dumps(rec))
        self.routes[(category, family, key)] = list(selected)

    async def write_verdicts(self, records: list[dict]) -> None:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in records:
            groups[(r["category"], r["family"])].append(r)
        for (cat, fam), recs in groups.items():
            batch = hashlib.sha1("|".join(sorted(r["unit_id"] for r in recs)).encode("utf-8")).hexdigest()[:16]   # noqa: S324 - identifier only
            await self._run(self.store.put, f"verdicts/{cat}/{fam}/{batch}.jsonl", b"\n".join(_dumps(r) for r in recs) + b"\n")
            for r in recs:
                self.verdict_keys.add((cat, fam, r["unit_id"]))

    async def write_status(self, rec: dict) -> None:
        rec = {**rec, "ts": now_ts()}
        await self._run(self.store.put, f"status/{rec['category']}.json", _dumps(rec))
        self.status[rec["category"]] = rec

    # ------------------------------------------------------------------ run log (buffered; never contains prompts)
    def log_run(self, rec: dict) -> None:
        with self._log_lock:
            self._log_buf.append(rec)
            if len(self._log_buf) < self._flush_every:
                return
            batch, self._log_buf = self._log_buf, []
        self._submit_log(batch)

    def _submit_log(self, batch: list[dict]) -> None:
        try:
            fut = self._io.submit(self._write_log, batch)
        except RuntimeError:                       # the pool is already shut down: write inline rather than lose the records
            self._write_log(batch)
            return
        with self._log_lock:
            self._log_pending = [f for f in self._log_pending if not f.done()] + [fut]

    def _write_log(self, batch: list[dict]) -> None:
        with self._log_lock:
            self._log_seq += 1
            seq = self._log_seq
        key = f"runlog/{time.strftime('%Y%m%d')}/{self._run_id}-{seq:06d}.jsonl"
        try:
            self.store.put(key, b"\n".join(_dumps(r) for r in batch) + b"\n")
        except Exception as e:                     # noqa: BLE001 - the run log is telemetry: losing a batch must never fail a run
            logger.warning("could not write %d run-log record(s): %s", len(batch), type(e).__name__)

    def flush_run_log(self) -> None:
        with self._log_lock:
            batch, self._log_buf = self._log_buf, []
            pending = list(self._log_pending)
            self._log_pending = []
        if batch:
            self._write_log(batch)
        for f in pending:
            f.result()

    async def aflush_run_log(self) -> None:
        await asyncio.to_thread(self.flush_run_log)

    def close(self) -> None:
        self.flush_run_log()
        self._io.shutdown(wait=True)
