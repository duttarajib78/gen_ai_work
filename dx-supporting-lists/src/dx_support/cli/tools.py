"""Small commands: `make-sample-data` and `serve-local`."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from dx_support.cli.common import load_settings
from dx_support.sample_data import write_sample_data


def make_sample_data(argv=None) -> int:
    """Writes the four reference files with tiny, invented, NON-CLINICAL content in the real formats, so everything (build --dry-run, the service, compile)
    can be tried with no licensed data and no credentials."""
    p = argparse.ArgumentParser(prog="dx-support make-sample-data", description=make_sample_data.__doc__)
    p.add_argument("--out", default="data-sample", help="Folder to write (default ./data-sample).")
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = p.parse_args(argv)
    out = Path(args.out)
    existing = [f for f in ("diagnosis_codes.tsv", "achi_codes.tsv", "all_pros.csv", "MBS-XML-20260801.XML") if (out / f).exists()]
    if existing and not args.force:
        sys.exit(f"{out} already contains {existing}; pass --force to overwrite.")
    paths = write_sample_data(out)
    print("Wrote (invented, non-clinical) sample reference files:")
    for k, v in paths.items():
        print(f"  {k:>10}: {v}")
    print(f"Try:  DX_DATA_DIR={out} dx-support build --dry-run --max-categories 5 --yes")
    return 0


def serve_local(argv=None) -> int:
    """Runs the HTTP service on this machine with the Flask development server -- for trying the API, NOT for production (the container uses gunicorn).
    With --backend fake and --sample-data nothing needs credentials or licensed data."""
    p = argparse.ArgumentParser(prog="dx-support serve-local", description=serve_local.__doc__)
    p.add_argument("--backend", choices=["fake", "bedrock"], default="fake")
    p.add_argument("--sample-data", action="store_true", help="Generate the invented sample reference files in a temp folder and use them.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--store-uri", default=None)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1: this machine only).")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)
    from dx_support.service.app import create_app
    from dx_support.service.runtime import ServiceRuntime
    settings = load_settings(args)
    over = {"backend": args.backend}
    if args.sample_data:
        tmp = Path(tempfile.mkdtemp(prefix="dx_sample_"))
        write_sample_data(tmp)
        over["data_dir"] = tmp
        print(f"Using invented sample data in {tmp}")
    if args.backend == "fake" and settings.is_production:
        sys.exit("The stub backend is refused when DX_ENV=production.")
    settings = settings.with_overrides(**over)
    runtime = ServiceRuntime(settings).start()
    app = create_app(runtime)
    print(f"Serving on http://{args.host}:{args.port}  (POST /invocations, GET /ping) -- fingerprint {runtime.fingerprint}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        runtime.shutdown()
    return 0


def bedrock_arns(argv=None, bedrock_client=None) -> int:
    """Prints the exact IAM resources the endpoint must be allowed to invoke for the models in your DX_* settings (same variables the endpoint uses), in the
    form the foundation stack's BedrockResourceArns parameter takes. Least privilege: the named profiles and the foundation models they route to, nothing else."""
    p = argparse.ArgumentParser(prog="dx-support bedrock-arns", description=bedrock_arns.__doc__)
    p.add_argument("--account", required=True, help="Your 12-digit AWS account id.")
    p.add_argument("--region", default=None, help="The endpoint's region (default DX_AWS_REGION / AWS_REGION, else ap-southeast-2).")
    p.add_argument("--partition", default="aws")
    p.add_argument("--destination-regions", default=None, help="Comma-separated regions the inference profile routes to (default: au -> ap-southeast-2,ap-southeast-4; else the region itself).")
    p.add_argument("--discover", action="store_true", help="Ask Bedrock which regions each profile routes to (needs bedrock:GetInferenceProfile). The most reliable option.")
    args = p.parse_args(argv)
    import re
    from dx_support.deploy_tools import build_arns, discover_destinations
    from dx_support.llm.catalog import load_catalog, resolve_model
    if not re.fullmatch(r"\d{12}", args.account):
        sys.exit("--account must be a 12-digit account id")
    settings = load_settings(args, quiet=True)
    region = args.region or settings.aws_region
    catalog = load_catalog(settings.model_catalog_path)
    ids = list(dict.fromkeys(resolve_model(m, settings.inference_profile_prefix, catalog).model_id for m in settings.stage_models().values()))
    dest = None
    if args.discover:
        if bedrock_client is None:
            import boto3
            bedrock_client = boto3.client("bedrock", region_name=region)
        dest = discover_destinations(bedrock_client, ids)
    elif args.destination_regions:
        regions = [r.strip() for r in args.destination_regions.split(",") if r.strip()]
        dest = {i: regions for i in ids}
    arns = build_arns(args.partition, region, args.account, ids, dest)
    print(f"# models in force: {', '.join(ids)}")
    for a in arns:
        print(a)
    print()
    print("BedrockResourceArns=" + ",".join(arns))
    return 0


def invoke(argv=None, client=None) -> int:
    """Calls a deployed endpoint (the reference client: waits on the streaming API for uncached work, reconnects, backs off when the instance is busy)."""
    import json
    p = argparse.ArgumentParser(prog="dx-support invoke", description=invoke.__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--region", default=None)
    p.add_argument("--categories", default=None, help="Comma-separated, e.g. H25,I48")
    p.add_argument("--codes", default=None, help="Comma-separated ICD-10-AM codes (mapped to their category)")
    p.add_argument("--families", default=None, help="Comma-separated: achi,mbs,prosthesis,dx (default achi,mbs,prosthesis)")
    p.add_argument("--merge-tiers", default=None, help="Default core,broad")
    p.add_argument("--include-audit", action="store_true")
    p.add_argument("--cache-only", action="store_true", help="Never start model calls; return only what is already stored")
    p.add_argument("--describe", action="store_true", help="Ask the endpoint to describe itself and stop")
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--json", action="store_true", help="Print the full response as JSON")
    p.add_argument("--out", default=None, help="Write the full response to this file")
    args = p.parse_args(argv)
    from dx_support.client import EndpointClient, EndpointError
    c = EndpointClient(args.endpoint, region=args.region, client=client)
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None          # noqa: E731
    try:
        if args.describe:
            body = c.describe()
        else:
            if not (args.categories or args.codes):
                sys.exit("give --categories and/or --codes (or --describe)")
            body = c.get_lists(categories=split(args.categories), codes=split(args.codes), families=split(args.families), merge_tiers=split(args.merge_tiers),
                               include_audit=args.include_audit, cache_only=args.cache_only, timeout_s=args.timeout_s,
                               on_progress=lambda ev: print(f"  ... {ev.get('elapsed_s')} s: {ev.get('categories')}", file=sys.stderr))
    except EndpointError as e:
        sys.exit(f"endpoint error: {e}")
    if args.out:
        Path(args.out).write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.json or args.describe:
        print(json.dumps(body, indent=1, ensure_ascii=False))
    else:
        print(f"status: {body['status']}  (fingerprint {body['fingerprint']}, cache hit: {body['usage']['cache_hit']}, {body['usage']['llm_calls']} model calls, ${body['usage']['cost_usd']})")
        for cat, r in body["results"].items():
            fams = {f: (v["state"], v["n_items"]) for f, v in r.get("families", {}).items()}
            print(f"  {cat} [{r['status']}] {r.get('title', '')[:60]}  " + "  ".join(f"{f}: {n} codes ({st})" for f, (st, n) in fams.items()))
        for line in body.get("notice", []):
            print(line)
    return 0 if body.get("status") in ("complete", None) or args.describe else 3


def prewarm(argv=None, sm_client=None) -> int:
    """Starts a SageMaker Processing job that builds supporting lists INTO the result store the endpoint reads, using the endpoint's own configuration
    (read from its SageMaker model), so categories built here are served instantly. You must give a spend cap: nothing here chooses one for you."""
    import json
    p = argparse.ArgumentParser(prog="dx-support prewarm", description=prewarm.__doc__)
    p.add_argument("--endpoint", required=True, help="Name of the deployed endpoint (for example prod-dx-support).")
    p.add_argument("--region", default=None)
    p.add_argument("--max-total-cost-usd", type=float, required=True, help="Hard spend cap for the job (required).")
    p.add_argument("--chapter", default=None, help="Chapters to build: 9, 9,10, IX,X, 9-11 or all.")
    p.add_argument("--categories", default=None, help="Comma-separated categories instead of chapters.")
    p.add_argument("--families", default=None, help="Default achi,mbs,prosthesis.")
    p.add_argument("--max-categories", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--category-concurrency", type=int, default=None)
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    p.add_argument("--max-runtime-hours", type=float, default=24.0)
    p.add_argument("--print-only", action="store_true", help="Print the CreateProcessingJob request and stop.")
    args = p.parse_args(argv)
    if not (args.chapter or args.categories or args.max_categories):
        sys.exit("say what to build: --chapter, --categories or --max-categories (a whole-universe build must be an explicit --chapter all)")
    from dx_support.prewarm import build_processing_request, describe_endpoint_model
    build_args = ["--max-total-cost-usd", str(args.max_total_cost_usd)]
    for flag, val in (("--chapter", args.chapter), ("--categories", args.categories), ("--families", args.families), ("--max-categories", args.max_categories),
                      ("--concurrency", args.concurrency), ("--category-concurrency", args.category_concurrency)):
        if val is not None:
            build_args += [flag, str(val)]
    sm = sm_client
    if sm is None:
        import boto3
        sm = boto3.client("sagemaker", region_name=args.region)
    model = describe_endpoint_model(sm, args.endpoint)
    req = build_processing_request(model, build_args, instance_type=args.instance_type, max_runtime_s=int(args.max_runtime_hours * 3600))
    if args.print_only:
        print(json.dumps(req, indent=1))
        return 0
    sm.create_processing_job(**req)
    print(f"started processing job {req['ProcessingJobName']} (cap ${args.max_total_cost_usd:,.0f}, builds into the endpoint's own result store)")
    print(f"follow it:  aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix {req['ProcessingJobName']} --follow")
    print(f"stop it:    aws sagemaker stop-processing-job --processing-job-name {req['ProcessingJobName']}   (re-running resumes; nothing is paid twice)")
    return 0
