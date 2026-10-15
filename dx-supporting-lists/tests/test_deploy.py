"""Deployment artifacts: template invariants (with negative tests that prove the checks bite), container files, IAM resource generation, scripts."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from dx_support.cli import tools
from dx_support.deploy_tools import build_arns, discover_destinations, split_profile_id
from tests.helpers import tmpdir
from tests.test_service import FakeSageMakerRuntime

ROOT = Path(__file__).resolve().parents[1]
CFN = ROOT / "deploy" / "cloudformation"
try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:                                           # pragma: no cover
    HAVE_YAML = False


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_templates", ROOT / "deploy" / "scripts" / "validate_templates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = load_validator()

    def mutated(self, which: str, old: str, new: str) -> list[str]:
        d = Path(tempfile.mkdtemp())
        texts = {n: (CFN / n).read_text() for n in ("foundation.yaml", "endpoint.yaml")}
        name = f"{which}.yaml"
        self.assertEqual(texts[name].count(old), 1, f"{old!r} must occur exactly once in {name}")
        texts[name] = texts[name].replace(old, new)
        for n, t in texts.items():
            (d / n).write_text(t)
        return self.v.validate(d / "foundation.yaml", d / "endpoint.yaml")

    def test_the_shipped_templates_are_consistent_and_secure(self):
        self.assertEqual(self.v.validate(CFN / "foundation.yaml", CFN / "endpoint.yaml"), [])

    def test_the_checks_actually_catch_regressions(self):
        cases = [
            ("foundation", 'Action: "s3:GetObject"\n                Resource: !Sub "${ArtifactsBucket.Arn}/model/*"', 'Action: "s3:*"\n                Resource: !Sub "${ArtifactsBucket.Arn}/model/*"', "wildcard action"),
            ("foundation", 'Resource: !Sub "${ArtifactsBucket.Arn}/serve/*"', 'Resource: "*"', "need a specific resource"),
            ("foundation", "CidrIp: !Ref VpcCidr", "CidrIp: 0.0.0.0/0", "world-open rule"),
            ("foundation", 'Condition: { Bool: { "aws:SecureTransport": "false" } }', 'Condition: { Bool: { "aws:SecureTransport": "true" } }', "deny non-TLS"),
            ("foundation", "ImageTagMutability: IMMUTABLE", "ImageTagMutability: MUTABLE", "immutable"),
            ("foundation", "BlockPublicAcls: true", "BlockPublicAcls: false", "block all public access"),
            ("foundation", "Status: Enabled\n      LifecycleConfiguration", "Status: Suspended\n      LifecycleConfiguration", "versioning"),
            ("foundation", "VpcId: !Ref VpcId\n      SecurityGroupEgress", "VpcId: !Ref NoSuchParameter\n      SecurityGroupEgress", "unknown"),
            ("endpoint", "DX_BACKEND: bedrock", "DX_BACKEND: fake", "bedrock backend"),
            ("endpoint", "DX_ENV: production", "DX_ENV: development", "production"),
            ("endpoint", "EnableNetworkIsolation: false ", "# ", "EnableNetworkIsolation"),
            ("endpoint", "          DX_LOG_CALLS: !Ref LogCalls", "          DX_LOG_CALLS: !Ref LogCalls\n          AWS_SECRET_ACCESS_KEY: nope", "looks like a secret"),
            ("endpoint", "HasKmsKey: !Not [!Equals [!Ref KmsKeyArn, none]]", 'HasKmsKey: !Not [!Equals [{"Fn::ImportValue": "x"}, none]]', "not allowed inside a condition"),
            ("endpoint", "- AlarmName: !Ref ModelErrorsAlarm", "- AlarmName: !Ref DoesNotExist", "unknown"),
        ]
        for which, old, new, needle in cases:
            errs = self.mutated(which, old, new) if (CFN / f"{which}.yaml").read_text().count(old) == 1 else ["__pattern missing__"]
            self.assertTrue(any(needle in e for e in errs), f"{needle!r} not reported for {old!r}: {errs}")

    def test_imports_match_exports_and_the_role_is_scoped(self):
        f, e = self.v.load(CFN / "foundation.yaml"), self.v.load(CFN / "endpoint.yaml")
        self.assertEqual(self.v.check_exports(f, e), [])
        role = f["Resources"]["ExecutionRole"]["Properties"]
        sids = {s["Sid"]: s for s in role["Policies"][0]["PolicyDocument"]["Statement"] if isinstance(s, dict) and "Sid" in s}
        self.assertEqual(sids["InvokeTheNamedBedrockModelsOnly"]["Action"], ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"])
        self.assertEqual(sids["InvokeTheNamedBedrockModelsOnly"]["Resource"], {"Ref": "BedrockResourceArns"})       # exactly what the operator names, no wildcard
        deletes = [s for s in sids.values() if "s3:DeleteObject" in json.dumps(s.get("Action"))]
        self.assertEqual([s["Sid"] for s in deletes], ["ReleaseTheRunLock"])                                       # nothing can be deleted except a run-lock object
        self.assertTrue(json.dumps(deletes[0]["Resource"]).endswith('/serve/*/.run.lock"}'))
        self.assertIn("aws:SourceAccount", json.dumps(role["AssumeRolePolicyDocument"]))                           # confused-deputy protection
        self.assertFalse(any("s3:PutObject" in json.dumps(s) and "model/" in json.dumps(s) for s in sids.values()))   # the model artifact is read-only to the container


class ContainerFileTests(unittest.TestCase):
    def test_dockerfile_runs_unprivileged_and_never_copies_data_or_secrets(self):
        d = (ROOT / "Dockerfile").read_text()
        self.assertIn("USER 10001:10001", d)
        self.assertIn('CMD ["serve"]', d)
        self.assertIn("EXPOSE 8080", d)
        self.assertIn("--require-hashes", d)                                      # a locked, hash-pinned install is used whenever requirements.lock exists
        for bad in ("COPY data", "ADD ", "COPY . ", "AWS_SECRET", "AWS_ACCESS_KEY", "ANTHROPIC_API_KEY", "sudo"):
            self.assertNotIn(bad, d)
        ignore = (ROOT / ".dockerignore").read_text().split()
        for must in ("data/", "output/", "*.env", ".git/", "tests/"):
            self.assertIn(must, ignore)

    def test_entrypoint_and_gunicorn_configuration(self):
        e = ROOT / "deploy" / "container" / "entrypoint.sh"
        self.assertTrue(e.stat().st_mode & stat.S_IXUSR)
        text = e.read_text()
        self.assertIn("serve)", text)
        self.assertIn("exec gunicorn", text)
        ns: dict = {}
        exec((ROOT / "deploy" / "container" / "gunicorn.conf.py").read_text(), ns)       # noqa: S102 - our own config file
        self.assertEqual((ns["workers"], ns["worker_class"]), (1, "gthread"))               # one process owns the loop, the limiter and the budget
        self.assertTrue(ns["bind"].endswith(":8080"))
        self.assertGreaterEqual(ns["threads"], 12)                                          # headroom beyond the request limit for /ping
        self.assertIsNone(ns["accesslog"])                                                  # request bodies are never access-logged
        self.assertTrue(callable(ns["worker_exit"]))

    def test_requirements_have_no_unbounded_or_dev_only_packages(self):
        req = [x for x in (ROOT / "requirements.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
        self.assertTrue(all(("<" in r and ">=" in r) for r in req), req)
        self.assertFalse(any(name in " ".join(req) for name in ("pytest", "ruff", "pip-tools")))
        self.assertFalse(any("anthropic" in r.lower() or "claude-agent" in r.lower() for r in req))       # the Agent SDK is gone: Bedrock via boto3


class BedrockArnTests(unittest.TestCase):
    def test_a_geo_profile_needs_the_profile_and_every_destination_foundation_model(self):
        arns = build_arns("aws", "ap-southeast-2", "111122223333", ["au.anthropic.claude-sonnet-5"])
        self.assertEqual(arns, ["arn:aws:bedrock:ap-southeast-2:111122223333:inference-profile/au.anthropic.claude-sonnet-5",
                                "arn:aws:bedrock:ap-southeast-2::foundation-model/anthropic.claude-sonnet-5",
                                "arn:aws:bedrock:ap-southeast-4::foundation-model/anthropic.claude-sonnet-5"])

    def test_global_and_base_ids_and_duplicates(self):
        g = build_arns("aws", "us-east-1", "111122223333", ["global.anthropic.claude-sonnet-5-5"])
        self.assertIn("arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-5-5", g)
        self.assertIn("arn:aws:bedrock:us-east-1:111122223333:inference-profile/global.anthropic.claude-sonnet-5-5", g)
        self.assertEqual(build_arns("aws", "us-east-1", "1", ["anthropic.claude-sonnet-5"]), ["arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5"])
        twice = build_arns("aws", "ap-southeast-2", "1", ["au.anthropic.claude-sonnet-5", "au.anthropic.claude-sonnet-5"])
        self.assertEqual(len(twice), len(set(twice)))
        self.assertEqual(split_profile_id("eu.anthropic.claude-x"), ("eu", "anthropic.claude-x"))
        self.assertEqual(split_profile_id("anthropic.claude-x"), ("", "anthropic.claude-x"))

    def test_discovery_asks_bedrock_for_the_real_routing(self):
        class Client:
            def get_inference_profile(self, inferenceProfileIdentifier):
                return {"models": [{"modelArn": "arn:aws:bedrock:ap-southeast-2::foundation-model/m"}, {"modelArn": "arn:aws:bedrock:ap-southeast-4::foundation-model/m"},
                                   {"modelArn": "arn:aws:bedrock:ap-southeast-2::foundation-model/m"}]}
        found = discover_destinations(Client(), ["au.anthropic.claude-sonnet-5", "anthropic.claude-base"])
        self.assertEqual(found, {"au.anthropic.claude-sonnet-5": ["ap-southeast-2", "ap-southeast-4"]})

    def test_the_cli_prints_the_parameter_in_the_stack_format(self):
        saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("DX_")}
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = tools.bedrock_arns(["--account", "111122223333"])
            line = next(x for x in buf.getvalue().splitlines() if x.startswith("BedrockResourceArns="))
            self.assertEqual(rc, 0)
            self.assertNotIn(" ", line)
            self.assertEqual(line.count("inference-profile/au.anthropic.claude-sonnet-5"), 1)          # three stages, one model: one profile, no duplicates
            os.environ["DX_ROUTE_MODEL"] = "claude-haiku-4-5-20251001"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tools.bedrock_arns(["--account", "111122223333", "--destination-regions", "ap-southeast-2"])
            self.assertIn("au.anthropic.claude-haiku-4-5-20251001-v1:0", buf.getvalue())
            with self.assertRaises(SystemExit):
                tools.bedrock_arns(["--account", "not-an-account"])
        finally:
            for k in [k for k in os.environ if k.startswith("DX_")]:
                os.environ.pop(k)
            os.environ.update(saved)


class InvokeCommandTests(unittest.TestCase):
    def test_summary_and_describe(self):
        body = {"status": "complete", "fingerprint": "abc123", "usage": {"cache_hit": True, "llm_calls": 0, "cost_usd": 0.0}, "notice": ["n1", "n2"],
                "results": {"I48": {"status": "complete", "title": "Atrial fibrillation", "families": {"achi": {"state": "complete", "n_items": 3}}}}}
        fake = FakeSageMakerRuntime([(200, body)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = tools.invoke(["--endpoint", "ep", "--categories", "I48", "--families", "achi"], client=fake)
        self.assertEqual(rc, 0)
        self.assertIn("I48 [complete]", buf.getvalue())
        self.assertIn("achi: 3 codes (complete)", buf.getvalue())
        self.assertEqual(fake.calls[0][1]["categories"], ["I48"])
        fake2 = FakeSageMakerRuntime([(200, {"version": "2.0.0", "fingerprint": "abc"})])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(tools.invoke(["--endpoint", "ep", "--describe"], client=fake2), 0)
        self.assertEqual(json.loads(buf.getvalue())["version"], "2.0.0")
        with self.assertRaises(SystemExit):
            tools.invoke(["--endpoint", "ep"], client=FakeSageMakerRuntime([]))


class ScriptTests(unittest.TestCase):
    def test_source_zip_leaves_out_data_tests_and_env_files(self):
        out = tmpdir() / "s.zip"
        subprocess.run(["python3", str(ROOT / "deploy" / "scripts" / "make_source_zip.py"), str(out)], check=True, capture_output=True)
        names = zipfile.ZipFile(out).namelist()
        self.assertIn("Dockerfile", names)
        self.assertIn("buildspec.yml", names)
        self.assertIn("src/dx_support/__init__.py", names)
        for n in names:
            self.assertFalse(n.startswith(("data/", "tests/", "docs/", ".git/", "deploy/cloudformation/", "output/")), n)
            self.assertFalse(n.endswith((".env", ".pem", ".pyc")), n)

    def test_package_model_data_builds_a_readable_artifact_and_leaves_encryption_to_the_bucket(self):
        from dx_support.sample_data import write_sample_data
        work = tmpdir()
        data = work / "data"
        write_sample_data(data)
        bin_dir = work / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "aws"
        stub.write_text(f'#!/bin/sh\necho "$@" >> "{work}/aws.log"\n[ "$1" = s3 ] && [ "$2" = cp ] && cp "$3" "{work}/uploaded.tar.gz"\nexit 0\n')
        stub.chmod(0o755)
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
        r = subprocess.run([str(ROOT / "deploy" / "scripts" / "package_model_data.sh"), str(data), "my-bucket", "v1"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ModelDataUrl=s3://my-bucket/model/v1/model.tar.gz", r.stdout)
        self.assertEqual(len([ln for ln in r.stdout.splitlines() if "  ./" in ln]), 4)             # four files with their SHA-256
        log = (work / "aws.log").read_text()
        self.assertIn("s3://my-bucket/model/v1/model.tar.gz", log)
        self.assertNotIn("--sse", log)                                                           # the bucket's default KMS encryption applies
        with tarfile.open(work / "uploaded.tar.gz") as t:
            members = [m for m in t.getmembers() if m.isfile()]
            self.assertEqual(sorted(Path(m.name).name for m in members), ["MBS-XML-20260801.XML", "achi_codes.tsv", "all_pros.csv", "diagnosis_codes.tsv"])
            self.assertTrue(all(m.uid == 0 and m.gid == 0 and (m.mode & 0o044) == 0o044 for m in members))   # root-owned, readable by the unprivileged container user
        incomplete = work / "incomplete"
        incomplete.mkdir()
        r = subprocess.run([str(ROOT / "deploy" / "scripts" / "package_model_data.sh"), str(incomplete), "b"], env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)


if __name__ == "__main__":
    unittest.main()


class PrewarmTests(unittest.TestCase):
    MODEL = {"ExecutionRoleArn": "arn:aws:iam::111122223333:role/r", "VpcConfig": {"SecurityGroupIds": ["sg-1"], "Subnets": ["subnet-a", "subnet-b"]},
             "PrimaryContainer": {"Image": "111122223333.dkr.ecr.ap-southeast-2.amazonaws.com/dx-support-prod:abc", "ModelDataUrl": "s3://b/model/v1/model.tar.gz",
                                  "Environment": {"DX_ENV": "production", "DX_MODEL": "claude-sonnet-5", "DX_STORE_URI": "s3://b"}}}

    class SM:
        def __init__(self, model):
            self.model, self.created = model, []

        def describe_endpoint(self, EndpointName):
            return {"EndpointConfigName": "cfg-1"}

        def describe_endpoint_config(self, EndpointConfigName):
            return {"ProductionVariants": [{"ModelName": "model-1"}]}

        def describe_model(self, ModelName):
            return self.model

        def create_processing_job(self, **kw):
            self.created.append(kw)

    def test_the_job_copies_the_endpoints_own_configuration_so_the_fingerprint_matches(self):
        from dx_support.prewarm import build_processing_request
        req = build_processing_request(self.MODEL, ["--max-total-cost-usd", "50", "--chapter", "9-10"], job_name="dx-prewarm-x")
        self.assertEqual(req["Environment"], self.MODEL["PrimaryContainer"]["Environment"])
        self.assertEqual((req["RoleArn"], req["AppSpecification"]["ImageUri"]), (self.MODEL["ExecutionRoleArn"], self.MODEL["PrimaryContainer"]["Image"]))
        self.assertEqual(req["NetworkConfig"]["VpcConfig"]["Subnets"], ["subnet-a", "subnet-b"])          # the same private network: Bedrock and S3 over the VPC endpoints
        self.assertFalse(req["NetworkConfig"]["EnableNetworkIsolation"])
        self.assertEqual(req["ProcessingInputs"][0]["S3Input"]["S3Uri"], "s3://b/model/v1/model.tar.gz")
        script = req["AppSpecification"]["ContainerArguments"][0]
        self.assertIn("exec dx-support build --for-service --yes --max-total-cost-usd 50 --chapter 9-10", script)
        self.assertEqual(req["AppSpecification"]["ContainerEntrypoint"], ["/bin/sh", "-c"])
        self.assertNotIn("SECRET", json.dumps(req))
        hostile = build_processing_request(self.MODEL, ["--categories", "H25; rm -rf /"], job_name="j")["AppSpecification"]["ContainerArguments"][0]
        self.assertIn("'H25; rm -rf /'", hostile)                                                           # arguments are shell-quoted, never interpolated
        with self.assertRaises(ValueError):
            build_processing_request({**self.MODEL, "PrimaryContainer": {"Image": "i", "Environment": {}}}, [])

    def test_command_requires_a_spend_cap_and_a_scope_and_submits(self):
        with self.assertRaises(SystemExit):
            tools.prewarm(["--endpoint", "e", "--chapter", "9"], sm_client=self.SM(self.MODEL))                   # no cap
        with self.assertRaises(SystemExit):
            tools.prewarm(["--endpoint", "e", "--max-total-cost-usd", "5"], sm_client=self.SM(self.MODEL))        # nothing named to build
        sm = self.SM(self.MODEL)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(tools.prewarm(["--endpoint", "e", "--max-total-cost-usd", "75", "--chapter", "VII", "--families", "achi,mbs"], sm_client=sm), 0)
        self.assertEqual(len(sm.created), 1)
        self.assertIn("--max-total-cost-usd 75.0 --chapter VII --families achi,mbs", sm.created[0]["AppSpecification"]["ContainerArguments"][0])
        self.assertIn("stop-processing-job", buf.getvalue())
        sm2 = self.SM(self.MODEL)
        with contextlib.redirect_stdout(io.StringIO()):
            tools.prewarm(["--endpoint", "e", "--max-total-cost-usd", "5", "--categories", "H25", "--print-only"], sm_client=sm2)
        self.assertEqual(sm2.created, [])                                                                    # --print-only submits nothing
