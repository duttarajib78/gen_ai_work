"""PID-1 supervisor for the SageMaker container.

Startup (fail fast, before any traffic): validate config -> verify the model artifact's checksums -> verify the
SDK supports every security-relevant option -> mark ready -> start the web (gunicorn) and worker processes as an
unprivileged user. If either child dies the supervisor exits non-zero so SageMaker replaces the instance; on SIGTERM
it forwards the signal and escalates to SIGKILL inside SageMaker's 30 s window.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..agents.sdk_runner import ClaudeSdkRunner
from ..errors import ConfigError
from ..llm.bedrock import BLANKED_ENV
from ..logging_setup import configure_logging
from ..model_package import verify_model_dir
from ..settings import Settings
from .health import HEARTBEAT_FILE, READY_FILE, run_dir, write_ready

log = logging.getLogger(__name__)
PORT = 8080


def child_commands(settings: Settings) -> dict[str, list[str]]:
    s = settings.service
    web = [sys.executable, "-m", "gunicorn", "rule_extractor.service.wsgi:app", "--bind", f"0.0.0.0:{PORT}",
           "--workers", str(s.web_workers), "--threads", str(s.web_threads), "--worker-class", "gthread",
           "--timeout", "55", "--graceful-timeout", "25", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-"]
    if os.path.isdir("/dev/shm"):
        web += ["--worker-tmp-dir", "/dev/shm"]
    return {"web": web, "worker": [sys.executable, "-m", "rule_extractor", "worker"]}


def child_env(settings: Settings, home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in BLANKED_ENV}
    env.update({"HOME": str(home), "PYTHONUNBUFFERED": "1"})
    return env


def popen_kwargs(settings: Settings) -> dict[str, Any]:
    uid = settings.service.run_as_uid
    if hasattr(os, "geteuid") and os.geteuid() == 0 and uid != 0:
        return {"user": uid, "group": uid, "extra_groups": []}
    return {}


def prepare_dirs(settings: Settings) -> tuple[Path, Path]:
    directory, home = run_dir(settings), settings.paths.work_dir / "home"
    for path in (settings.paths.work_dir, directory, home):
        path.mkdir(parents=True, exist_ok=True)
    for stale in (READY_FILE, HEARTBEAT_FILE):
        (directory / stale).unlink(missing_ok=True)
    if hasattr(os, "geteuid") and os.geteuid() == 0 and settings.service.run_as_uid != 0:
        uid = settings.service.run_as_uid
        for path in (settings.paths.work_dir, directory, home):
            os.chown(path, uid, uid)
    return directory, home


def run_supervisor(settings: Settings, *, popen: Callable[..., Any] = subprocess.Popen,
                   runner_factory: Callable[[Settings], ClaudeSdkRunner] = ClaudeSdkRunner,
                   stop_event: threading.Event | None = None, install_signals: bool = True) -> int:
    configure_logging(os.environ.get("RULE_EXTRACTOR_LOG_LEVEL", "INFO"))
    try:
        if settings.service.backend == "aws":
            settings.validate_for_aws_backend()
        settings.validate_for_agents()
        problems = verify_model_dir(settings.paths.model_dir)
        if problems:
            log.error("model artifact failed integrity verification", extra={"fields": {"problems": problems[:10]}})
            return 2
        sdk_info = runner_factory(settings).preflight()
    except ConfigError as exc:
        log.error("startup configuration error: %s", exc.message)
        return 3
    directory, home = prepare_dirs(settings)
    write_ready(directory, {"started": time.time(), "sdk": sdk_info, "backend": settings.service.backend})

    stopping = stop_event or threading.Event()
    if install_signals:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopping.set())
    kwargs, env = popen_kwargs(settings), child_env(settings, home)
    procs = {name: popen(cmd, env=env, **kwargs) for name, cmd in child_commands(settings).items()}
    log.info("supervisor started children", extra={"fields": {"children": list(procs), "port": PORT}})

    exit_code = 0
    while not stopping.is_set():
        for name, proc in procs.items():
            rc = proc.poll()
            if rc is not None:
                log.error("child exited unexpectedly", extra={"fields": {"child": name, "returncode": rc}})
                exit_code = rc if rc else 1
                stopping.set()
                break
        stopping.wait(0.5)

    for proc in procs.values():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + settings.service.shutdown_grace_seconds + 3
    for proc in procs.values():
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("child did not exit in time; killing")
            proc.kill()
    return exit_code
