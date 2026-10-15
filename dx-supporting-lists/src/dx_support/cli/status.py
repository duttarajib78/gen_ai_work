"""dx-support status -- read-only progress report for a build. Run it from ANOTHER terminal while `dx-support build` is running, or afterwards. It only
reads the store (checkpoints, run log, lock object); it changes nothing and makes no model calls.

    dx-support status                    # the default namespace
    dx-support status --chapter 10       # one chapter (number or Roman numeral)
    dx-support status --watch 30         # refresh every 30 seconds (Ctrl-C quits the watcher, never the run)
    dx-support status --dry-run          # the dry_run namespace
    dx-support status --namespace serve/<fingerprint>   (see below)

It opens with a one-line VERDICT (RUNNING / FINISHED / STOPPED / CRASHED OR KILLED / NOT STARTED), then reports: whether a run is alive (lock object and, on the
same host, a process check), how many categories are complete / partial / not started, the time since the last model call, calls and cost so far
(API-rate estimates), the recent failure pattern, and a pace-based estimate of the time and cost still to go.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from collections import Counter
from typing import Optional

from dx_support.cli.common import RUN_CONFIG_KEY, add_run_args, load_settings, namespace_path, open_run
from dx_support.constants import FAMILIES, parse_families
from dx_support.indexing.universe import load_universe
from dx_support.pipeline.projection import fmt_hours, fmt_usd
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS, VALID_CHAPTER_IDS, parse_chapters
from dx_support.storage.checkpoint import Checkpoint
from dx_support.storage.locks import RunLock
from dx_support.storage.objectstore import ObjectStore


def _parse_ts(s) -> Optional[float]:
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def _pid_alive(pid: int) -> Optional[bool]:
    """True/False on POSIX; None where it cannot be checked safely (on Windows os.kill TERMINATES the process)."""
    if os.name != "posix" or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _partial_detail(cat: str, rec: dict) -> str:
    bits = []
    if rec.get("error"):
        bits.append(str(rec["error"])[:70])
    for fam, fs in (rec.get("families") or {}).items():
        if not fs.get("complete", True):
            bits.append(f"{fam}: {str(fs['error'])[:70]}" if fs.get("error") else f"{fam}: {fs.get('n_missing', '?')} units without a verdict")
    return cat + (f" ({'; '.join(bits)})" if bits else "")


def build_report(store: ObjectStore, ckpt: Checkpoint, categories: dict, chapter=None, now: Optional[float] = None, families: Optional[tuple] = None) -> str:
    now = now or time.time()
    L: list[str] = []
    chapters = None if not chapter else ({chapter} if isinstance(chapter, str) else set(chapter))
    scope = [c for c, i in categories.items() if (i.chapter in chapters if chapters else i.chapter not in CONTEXT_ONLY_CHAPTERS)]
    scope_set = set(scope)

    # ---- is a run alive? ----
    lock_info = RunLock.read(store) or {}
    lock_state = "none"
    if lock_info:
        pid = int(lock_info["pid"]) if lock_info.get("pid", "").isdigit() else 0
        same_host = lock_info.get("host") == socket.gethostname().replace(" ", "_")
        alive = _pid_alive(pid) if same_host else None
        lock_state = "alive" if alive else ("unknown" if alive is None else "stale")
        if alive:
            state = f"RUNNING (pid {pid}, started {lock_info.get('started', '?')})"
        elif alive is None:
            state = f"LOCK PRESENT (host {lock_info.get('host', '?')}, pid {pid}, started {lock_info.get('started', '?')}); this tool cannot check whether it is alive from here"
        else:
            state = f"NOT RUNNING -- stale lock left by pid {pid}; delete the object {store.uri}/.run.lock before re-running"
    else:
        state = "NOT RUNNING (no lock)"
    L.append(f"Store      : {store.uri}")
    L.append(f"State      : {state}")

    # ---- categories ----
    last = ckpt.load_status_all()
    fam_set = set(families) if families else None
    done_for = ((lambda c: last.get(c, {}).get("status") == "complete" and fam_set <= set(last[c].get("families_complete", [])))
                if fam_set else (lambda c: last.get(c, {}).get("status") == "complete"))
    counts = Counter(last[c]["status"] for c in scope if c in last)
    not_started = sum(1 for c in scope if c not in last)
    n_done = sum(1 for c in scope if done_for(c))
    n_other = sum(1 for c in scope if last.get(c, {}).get("status") == "complete" and not done_for(c))
    label = (f"chapter{'s' if len(chapters) > 1 else ''} " + ", ".join(c for c in VALID_CHAPTER_IDS if c in chapters)) if chapters else "all non-context chapters"
    L.append(f"Scope      : {label} -- {len(scope)} categories")
    if fam_set:
        L.append(f"Families   : {', '.join(f for f in FAMILIES if f in fam_set)}")
    L.append(f"Categories : complete {n_done} | partial {counts.get('partial', 0)} | not started {not_started}"
             + (f" | finished other families only {n_other}" if n_other else "")
             + (f" | skipped {sum(v for k, v in counts.items() if k.startswith('skipped'))}" if any(k.startswith('skipped') for k in counts) else ""))

    # ---- calls and cost ----
    attempts = failed = 0
    cost = cost_since = 0.0
    n_since = failed_since = 0
    by_stage: dict[str, list] = {}
    last_call_ts: Optional[float] = None
    failures: list[dict] = []
    since_calls: Counter = Counter()
    since_cost: Counter = Counter()
    since = _parse_ts(lock_info.get("started")) if lock_info else None
    for r in ckpt.iter_run_log():
        if r.get("category") not in scope_set:
            continue
        attempts += 1
        c = r.get("cost_usd") or 0.0
        cost += c
        failed += 0 if r.get("ok") else 1
        if not r.get("ok"):
            failures.append(r)
        st = by_stage.setdefault(r.get("stage", "?"), [0, 0.0])
        st[0] += 1
        st[1] += c
        t = _parse_ts(r.get("ts"))
        if t:
            last_call_ts = max(last_call_ts or 0, t)
            if since and t >= since:
                n_since += 1
                cost_since += c
                failed_since += 0 if r.get("ok") else 1
                since_calls[r.get("category")] += 1
                since_cost[r.get("category")] += c
    if last_call_ts:
        idle = max(now - last_call_ts, 0)
        L.append(f"Last call  : {idle:,.0f} s ago" + ("  <-- no recent activity; stalled, throttled, or the machine slept?" if state.startswith("RUNNING") and idle > 600 else ""))
    else:
        L.append("Last call  : none logged yet")
    L.append(f"Calls/cost : {attempts:,} attempts ({failed:,} failed) | {fmt_usd(cost)} so far (API-rate estimate)" + (f" | this run: {n_since:,} calls, {fmt_usd(cost_since)}" if since else ""))
    if by_stage:
        L.append("  by stage : " + ", ".join(f"{s} {n:,} calls {fmt_usd(c)}" for s, (n, c) in sorted(by_stage.items())))

    # ---- what is left ----
    done = [c for c in scope if done_for(c)]
    todo = [c for c in scope if not done_for(c) and last.get(c, {}).get("status") not in ("skipped_context_only", "skipped_by_brief")]
    if todo:
        # While a run is alive, prefer THIS run's own call throughput (the pace of the last few completions is misleading: categories started
        # together finish together, so a burst of completions looks like a fast steady pace).
        eta_line = None
        if lock_state == "alive" and since and n_since >= 30 and now - since >= 120:
            finished_here = [c for c in done if (_parse_ts(last[c].get("ts")) or 0) >= since]
            if finished_here:
                rate = n_since / (now - since)
                per_calls = sum(since_calls[c] for c in finished_here) / len(finished_here)
                per_cost = sum(since_cost[c] for c in finished_here) / len(finished_here)
                spent_calls = sum(v for c, v in since_calls.items() if c in scope_set and not done_for(c))
                spent_cost = sum(v for c, v in since_cost.items() if c in scope_set and not done_for(c))
                rem_calls = max(len(todo) * per_calls - spent_calls, 0.0)
                rem_cost = max(len(todo) * per_cost - spent_cost, 0.0)
                eta_line = (f"~{fmt_hours(rem_calls / rate / 3600)} (about {rem_calls:,.0f} calls left at {rate * 60:.0f} calls/min this run); "
                            f"~{fmt_usd(rem_cost)} more at {fmt_usd(per_cost)} per category finished in this run")
        if eta_line:
            L.append(f"Remaining  : {len(todo)} categories to go")
            L.append("ETA        : " + eta_line)
        else:
            times = sorted(t for t in (_parse_ts(last[c].get("ts")) for c in done) if t)[-11:]
            gaps = [b - a for a, b in zip(times, times[1:]) if 0 <= b - a <= 600]
            avg = cost / len(done) if done else None
            pace = (len(gaps) / sum(gaps)) if gaps and sum(gaps) > 0 else None
            bits = [f"{len(todo)} categories to go"]
            if avg is not None:
                bits.append(f"~{fmt_usd(avg * len(todo))} at {fmt_usd(avg)} per completed category")
            if pace:
                bits.append(f"~{fmt_hours(len(todo) / pace / 3600)} at the recent pace ({pace * 60:.1f} categories/min; rough, completions arrive in bursts)")
            L.append("Remaining  : " + "; ".join(bits))
    elif scope:
        L.append("Remaining  : nothing -- every category in scope is complete or skipped. Next: dx-support compile")

    # ---- problems ----
    if failures:
        L.append(f"Problems   : {len(failures)} failed attempts in total" + (f" ({failed_since} in this run)" if since else "") + "; most recent:")
        for r in failures[-5:]:
            L.append(f"    {r.get('label') or r.get('category')}: {(r.get('error') or 'unknown')[:110]}")
    partial = [c for c in scope if last.get(c, {}).get("status") == "partial"]
    if partial:
        L.append("Partial    : " + "; ".join(_partial_detail(c, last[c]) for c in partial[:8]) + (" ..." if len(partial) > 8 else "") + "  -> re-run the same command to finish them")
    if lock_state == "alive":
        verdict = "RUNNING"
    elif lock_state == "unknown":
        verdict = "RUNNING? -- a lock exists, but this tool cannot check whether the process is alive from here"
    elif lock_state == "stale":
        verdict = f"CRASHED OR KILLED -- stale lock; {len(todo)} categories not complete. Delete the lock, then re-run the same command"
    elif not scope:
        verdict = "NOTHING IN SCOPE"
    elif not todo:
        verdict = f"FINISHED -- all {len(scope)} categories complete (or skipped). Next: dx-support compile"
    elif attempts or counts:
        verdict = (f"STOPPED -- {len(todo)} of {len(scope)} categories not complete (cost cap, error, interrupt, or not all started). "
                   "Check the end of the log, then re-run the same command to continue")
    else:
        verdict = "NOT STARTED"
    L.insert(2, f"Verdict    : {verdict}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dx-support status", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, tag=True)
    p.add_argument("--chapter", "--chapters", type=parse_chapters, default=None, metavar="CHAPTERS", help="1-22 or Roman numerals: one, a list, or a range (9, 9,10, IX,X, 9-11).")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Refresh every N seconds until Ctrl-C.")
    p.add_argument("--families", type=str, default=None, help="Families that must be complete for a category to count as done (comma list, or `all`). Default: the last run's, from run_config.json.")
    args = p.parse_args(argv)
    settings = load_settings(args, quiet=True)
    ns = args.namespace
    path = ns if ns.startswith(("runs/", "serve/")) else namespace_path(ns, args.dry_run, args.tag)
    try:
        universe = load_universe(settings.data_dir, settings.include_coarse_prostheses, warm_indexes=False)
    except FileNotFoundError as e:
        sys.exit(f"{e}\nThe diagnosis file is needed to know which categories belong to a chapter.")
    run = open_run(settings, path)
    try:
        fams = None
        if args.families:
            try:
                fams = parse_families(args.families)
            except ValueError as e:
                sys.exit(str(e))
        else:
            cfg = run.ckpt.read_json(RUN_CONFIG_KEY) or {}
            fams = tuple(cfg.get("families") or ()) or None
        try:
            while True:
                print(build_report(run.store, run.ckpt, universe.categories, args.chapter, families=fams))
                if not args.watch:
                    return 0
                print("-" * 72)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    finally:
        run.close()


if __name__ == "__main__":
    raise SystemExit(main())
