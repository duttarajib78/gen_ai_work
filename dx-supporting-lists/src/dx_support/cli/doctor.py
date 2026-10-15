"""dx-support doctor -- preflight. Run it before the first deployment, after changing a model, and from the endpoint's own environment.

It checks, in order: the configuration (with the production rules), the surrounding environment, the reference data, the model names and what they
resolve to on Bedrock, the result store (a write and a read-back), your AWS identity, and finally that EVERY configured model really answers a tiny
request in the shape the pipeline uses (tool call, thinking settings, inference profile, IAM permissions). The model probe costs a fraction of a cent
per distinct model. Exit status is 1 if anything failed, so it can gate a deployment pipeline.

    dx-support doctor                    # everything
    dx-support doctor --offline          # no network: configuration, data, model resolution, local store only
    dx-support doctor --skip-data        # skip loading the reference files
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable, Optional

from dx_support.cli.common import load_settings
from dx_support.indexing.universe import load_universe
from dx_support.llm.bedrock import BedrockBackend
from dx_support.llm.catalog import load_catalog, resolve_model
from dx_support.llm.types import LLMCallError, LLMRequest
from dx_support.settings import Settings, environment_warnings
from dx_support.storage.objectstore import namespace, open_store
from dx_support.text import redact

PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def run_checks(settings: Settings, *, offline: bool = False, skip_data: bool = False, skip_models: bool = False, backend: Optional[Any] = None,
               store: Optional[Any] = None) -> list[tuple[str, str, str]]:
    """[(name, 'ok'|'warn'|'fail', detail)]. `backend` / `store` can be injected (tests)."""
    out: list[tuple[str, str, str]] = []

    def check(name: str, fn: Callable[[], tuple[str, str]]) -> None:
        try:
            level, detail = fn()
        except Exception as e:                                      # noqa: BLE001 - a failing check is a result, not a crash
            level, detail = "fail", redact(f"{type(e).__name__}: {e}")[:300]
        out.append((name, level, detail))

    check("configuration", lambda: ("ok", f"env={settings.env} backend={settings.backend} region={settings.aws_region} model={settings.model} thinking={settings.thinking} "
                                          f"profile={settings.inference_profile_prefix} concurrency={settings.llm_concurrency}"))
    warns = environment_warnings()
    out.append(("environment", "warn" if warns else "ok", "; ".join(warns) or "no endpoint overrides or static credentials in the environment"))
    if not skip_data:
        def data():
            u = load_universe(settings.data_dir, settings.include_coarse_prostheses)
            n = {f: len(i.units) for f, i in u.indices.items()}
            return "ok", f"{len(u.categories)} categories; reviewable units {n}; data digest {u.digest[:12]}"
        check("reference data", data)

    def models():
        cat = load_catalog(settings.model_catalog_path)
        lines, level = [], "ok"
        for stage, name in settings.stage_models().items():
            r = resolve_model(name, settings.inference_profile_prefix, cat)
            lines.append(f"{stage}: {r.model_id} (${r.spec.price_in * r.price_multiplier:.2f}/${r.spec.price_out * r.price_multiplier:.2f} per M tokens)")
            if r.warning:
                level = "warn"
                lines.append("  ! " + r.warning)
        return level, "; ".join(lines)
    check("models", models)

    def store_check():
        s = store or namespace(open_store(settings.store_uri, region=settings.aws_region, kms_key_id=settings.store_kms_key_id), "serve", "doctor")
        s.put("probe.json", b"{}")
        if s.get("probe.json") != b"{}":
            raise RuntimeError("the store did not return what was written")
        return "ok", f"wrote and read back {s.uri}/probe.json"
    if offline and (settings.store_uri.startswith("s3://") and store is None):
        out.append(("result store", "warn", "skipped (--offline) for an S3 store"))
    else:
        check("result store", store_check)

    if not offline and backend is None and settings.backend == "bedrock":
        def identity():
            import boto3
            who = boto3.client("sts", region_name=settings.aws_region).get_caller_identity()
            return "ok", f"account {who['Account']} as {who['Arn']}"
        check("aws identity", identity)

    if not offline and not skip_models and settings.backend == "bedrock":
        be = backend or BedrockBackend(settings)
        seen: dict[str, str] = {}
        for stage, name in settings.stage_models().items():
            seen.setdefault(be.resolve(name).model_id, name)
        for model_id, name in seen.items():
            def probe(name=name):
                req = LLMRequest(stage="probe", system_prompt="You are a connectivity probe for a deployment check.", prompt="Call submit_result with ok set to true.",
                                 schema=PROBE_SCHEMA, model=name, thinking=settings.thinking, max_tokens=2048, max_budget_usd=settings.max_budget_usd_per_call)
                try:
                    res = asyncio.run(be.complete(req))
                except LLMCallError as e:
                    return "fail", f"{e.kind}: {e}"[:300]
                if isinstance(res.structured, dict) and res.structured.get("ok") is True:
                    return "ok", f"answered via the tool in {res.diag.get('duration_ms')} ms, {res.diag.get('input_tokens')}+{res.diag.get('output_tokens')} tokens, ${res.cost_usd or 0:.4f}"
                return "warn", "the model answered but did not return the expected tool input; the pipeline will retry such calls, but check the model / thinking settings"
            check(f"bedrock probe {model_id}", probe)
        close = getattr(be, "close", None)
        if close and backend is None:
            close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dx-support doctor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-uri", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--offline", action="store_true", help="No network checks (Bedrock, STS, S3).")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-models", action="store_true", help="Do not call the models (no cost).")
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = p.parse_args(argv)
    settings = load_settings(args, quiet=True)
    results = run_checks(settings, offline=args.offline, skip_data=args.skip_data, skip_models=args.skip_models)
    if args.json:
        print(json.dumps([{"check": n, "result": r, "detail": d} for n, r, d in results], indent=1))
    else:
        for name, level, detail in results:
            print(f"[{level.upper():>4}] {name}: {detail}")
    failed = [n for n, r, _ in results if r == "fail"]
    if not args.json:
        print("-" * 72)
        print("FAILED: " + ", ".join(failed) if failed else "All checks passed" + (" (with warnings)" if any(r == "warn" for _, r, _ in results) else "") + ".")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
