"""Plumbing shared by the commands: settings, store / namespace resolution, run configuration, output naming, graceful shutdown."""
from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dx_support.llm.bedrock import BedrockBackend
from dx_support.llm.fake import FakeBackend
from dx_support.logging_setup import configure_logging
from dx_support.reference.chapters import VALID_CHAPTER_IDS
from dx_support.settings import ConfigError, Settings, environment_warnings
from dx_support.storage.checkpoint import Checkpoint
from dx_support.storage.objectstore import ObjectStore, namespace, open_store

logger = logging.getLogger("dx_support.cli")
RUN_CONFIG_KEY = "run_config.json"


def add_run_args(p: argparse.ArgumentParser, *, tag: bool = False) -> None:
    p.add_argument("--store-uri", default=None, help="Where checkpoints live: file:///path or s3://bucket/prefix (default: DX_STORE_URI, else ./output).")
    p.add_argument("--namespace", default="default", help="Name of this batch run inside the store (runs/<namespace>/). Default: default.")
    p.add_argument("--data-dir", default=None, help="Folder holding the four reference files (default: DX_DATA_DIR, else ./data).")
    p.add_argument("--dry-run", action="store_true", help="Use the offline non-clinical stub instead of a model: no credentials, no cost. Uses the 'dry_run' namespace.")
    if tag:
        p.add_argument("--tag", type=str, default=None, help="Read the sub-run <namespace>/<tag> (for example 'exhaustive').")


def load_settings(args: argparse.Namespace, quiet: bool = False) -> Settings:
    try:
        s = Settings.from_env()
    except ConfigError as e:
        sys.exit(f"Configuration error: {e}")
    over = {}
    if getattr(args, "store_uri", None):
        over["store_uri"] = args.store_uri
    if getattr(args, "data_dir", None):
        over["data_dir"] = Path(args.data_dir)
    if getattr(args, "include_coarse_prostheses", False):
        over["include_coarse_prostheses"] = True
    s = s.with_overrides(**over) if over else s
    configure_logging(s.log_level, json_logs=False)
    logging.getLogger("dx_support.calls").setLevel(logging.INFO if s.log_calls else logging.WARNING)
    if not quiet:
        for w in environment_warnings():
            logger.warning(w)
    return s


def namespace_path(ns: str, dry_run: bool, tag: Optional[str] = None) -> str:
    base = "dry_run" if dry_run else ns
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,59}$", base) or (tag and not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,59}$", tag)):
        sys.exit("--namespace and --tag may only contain letters, digits, '_' and '-' (at most 60 characters).")
    return f"runs/{base}" + (f"/{tag}" if tag else "")


@dataclass
class Run:
    settings: Settings
    base: ObjectStore
    store: ObjectStore
    ckpt: Checkpoint
    ns: str

    def close(self) -> None:
        self.ckpt.close()


def open_run(settings: Settings, ns_path: str, *, create_ckpt: bool = True) -> Run:
    base = open_store(settings.store_uri, region=settings.aws_region, kms_key_id=settings.store_kms_key_id)
    store = namespace(base, ns_path)
    return Run(settings, base, store, Checkpoint(store, io_workers=16), ns_path)


def make_backend(settings: Settings, dry_run: bool, fake_fail_rate: float = 0.0, fake_delay_s: float = 0.0):
    return FakeBackend(fail_rate=fake_fail_rate, delay_s=fake_delay_s) if dry_run else BedrockBackend(settings)


def install_graceful_shutdown() -> None:
    """Make SIGTERM behave like Ctrl-C: a plain `kill <pid>` (or a SageMaker stop signal) raises KeyboardInterrupt, so the checkpoints are flushed and
    the run lock is released instead of the process vanishing mid-write. (SIGKILL cannot be caught: the lock then has to be deleted by hand.)"""
    def _handler(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):          # not the main thread, or no SIGTERM on this platform
        pass


# ---------------------------------------------------------------------------
# Output file naming (a scoped compile never overwrites a full one)
# ---------------------------------------------------------------------------
_SUFFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,59}$")


def with_name_suffix(path: Path, suffix: str) -> Path:
    """`x.csv` -> `x_<suffix>.csv`. The suffix arrives on the command line and becomes part of a FILE NAME, so it is validated: it can never carry a
    path separator, `..` or a space."""
    if not _SUFFIX_RE.match(suffix or ""):
        raise ValueError(f"{suffix!r} is not a valid name suffix: use letters, digits, '_' and '-' (at most 60 characters)")
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _chapter_tag(chapters: tuple) -> str:
    """'IX' ; two or more: 'IX_X' ; consecutive runs of three or more are shortened ('I-XIX_XXI'), so a long list stays a sensible filename."""
    order = {c: i for i, c in enumerate(VALID_CHAPTER_IDS)}
    ids = sorted(set(chapters), key=lambda c: order.get(c, 999))
    runs: list[list[str]] = []
    for c in ids:
        if runs and order.get(c, -9) == order.get(runs[-1][-1], -99) + 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    parts = ["_".join(r) if len(r) <= 2 else f"{r[0]}-{r[-1]}" for r in runs]
    return ("chapter_" if len(ids) == 1 else "chapters_") + "_".join(parts)


def with_chapter(path: Path, chapter) -> Path:
    """Chapter-suffixed filename. One chapter: `x_chapter_X`; several: `x_chapters_IX_X`. All 22 chapters is the same as no scope."""
    if not chapter:
        return path
    chapters = (chapter,) if isinstance(chapter, str) else tuple(chapter)
    if set(chapters) >= set(VALID_CHAPTER_IDS):
        return path
    return path.with_name(f"{path.stem}_{_chapter_tag(chapters)}{path.suffix}")


def fmt_ts(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
