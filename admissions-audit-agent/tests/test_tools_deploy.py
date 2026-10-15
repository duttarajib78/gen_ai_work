import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from audit_agent.errors import ReferenceDataError
from audit_agent.reference_data import ReferenceData
from tools.deploy import iam
from tools.deploy.deploy_endpoint import (
    Problems, apply_plan, build_plan, container_environment, find_placeholders, names, validate_config,
)
from tools.deploy.package_model_artifact import PackagingError, build_artifact, main as package_main
from tests.fakes import make_reference_dir

ACCOUNT = "123456789012"
GLOBAL = "global.anthropic.claude-sonnet-5-5"


class IamTests(unittest.TestCase):
    def test_global_profile_uses_the_documented_three_statements(self):
        doc = iam.bedrock_invoke_policy(region="ap-southeast-2", account_id=ACCOUNT, model_id=GLOBAL)
        s = doc["Statement"]
        profile = f"arn:aws:bedrock:ap-southeast-2:{ACCOUNT}:inference-profile/{GLOBAL}"
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0]["Resource"], [profile])
        self.assertEqual(s[1]["Resource"], ["arn:aws:bedrock:ap-southeast-2::foundation-model/anthropic.claude-sonnet-5-5"])
        self.assertEqual(s[2]["Resource"], ["arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-5-5"])
        self.assertEqual(s[2]["Condition"]["StringEquals"]["aws:RequestedRegion"], "unspecified")
        for st in s[1:]:
            self.assertEqual(st["Condition"]["StringEquals"]["bedrock:InferenceProfileArn"], profile)
        self.assertTrue(all(st["Action"] == "bedrock:InvokeModel" for st in s))

    def test_geo_profile_needs_explicit_destination_regions(self):
        with self.assertRaises(ValueError):
            iam.bedrock_invoke_policy(region="eu-west-1", account_id=ACCOUNT, model_id="eu.anthropic.claude-sonnet-5-5")
        doc = iam.bedrock_invoke_policy(region="eu-west-1", account_id=ACCOUNT, model_id="eu.anthropic.claude-sonnet-5-5",
                                        destination_regions=["eu-west-3", "eu-west-1", "eu-west-3"])
        self.assertEqual(doc["Statement"][1]["Resource"], [
            "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-sonnet-5-5",
            "arn:aws:bedrock:eu-west-3::foundation-model/anthropic.claude-sonnet-5-5"])

    def test_inputs_are_validated(self):
        for kw in ({"account_id": "123"}, {"model_id": "gpt-4"}, {"model_id": "anthropic.claude-sonnet-5-5"}):
            args = {"region": "ap-southeast-2", "account_id": ACCOUNT, "model_id": GLOBAL, **kw}
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                iam.bedrock_invoke_policy(**args)

    def test_execution_role_is_least_privilege(self):
        doc = iam.execution_role_policy(
            region="ap-southeast-2", account_id=ACCOUNT, endpoint_name="audit-prod", ecr_repository="audit",
            artifact_s3_uri="s3://bkt/audit/v1/model.tar.gz", kms_key_arn=f"arn:aws:kms:ap-southeast-2:{ACCOUNT}:key/k",
            model_id=GLOBAL)
        json.dumps(doc)
        by_sid = {s["Sid"]: s for s in doc["Statement"]}
        self.assertEqual(by_sid["ReadModelArtifact"]["Resource"], ["arn:aws:s3:::bkt/audit/v1/model.tar.gz"])
        self.assertEqual(by_sid["PullImage"]["Resource"], [f"arn:aws:ecr:ap-southeast-2:{ACCOUNT}:repository/audit"])
        self.assertIn("/aws/sagemaker/Endpoints/audit-prod", by_sid["EndpointLogs"]["Resource"][0])
        self.assertIn("VpcNetworking", by_sid)
        actions = [a for s in doc["Statement"] for a in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]])]
        self.assertFalse([a for a in actions if a.endswith("*")], "no wildcard actions")
        bare_wildcards = sorted({s["Sid"] for s in doc["Statement"] if s["Resource"] == "*"})
        self.assertEqual(bare_wildcards, ["EcrToken", "EndpointMetrics", "VpcNetworking"])  # APIs that cannot be scoped
        no_vpc = iam.execution_role_policy(
            region="ap-southeast-2", account_id=ACCOUNT, endpoint_name="e", ecr_repository="r",
            artifact_s3_uri="s3://b/k.tar.gz", kms_key_arn="arn:aws:kms:x", model_id=GLOBAL, vpc=False)
        self.assertNotIn("VpcNetworking", {s["Sid"] for s in no_vpc["Statement"]})

    def test_trust_and_deployer(self):
        self.assertEqual(iam.trust_policy()["Statement"][0]["Principal"], {"Service": "sagemaker.amazonaws.com"})
        doc = iam.deployer_policy(region="ap-southeast-2", account_id=ACCOUNT, endpoint_name="audit-prod",
                                  execution_role_arn=f"arn:aws:iam::{ACCOUNT}:role/r", kms_key_arn="arn:aws:kms:x",
                                  ecr_repository="audit", artifact_bucket="bkt")
        passrole = next(s for s in doc["Statement"] if s["Sid"] == "PassEndpointRole")
        self.assertEqual(passrole["Condition"], {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}})

    def test_cli_emits_json_and_reports_errors(self):
        import contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = iam.main(["bedrock", "--region", "ap-southeast-2", "--account", ACCOUNT, "--model-id", GLOBAL])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out.getvalue())["Statement"]), 3)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(iam.main(["bedrock", "--region", "r", "--account", "bad", "--model-id", GLOBAL]), 2)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ref_dir = make_reference_dir(self.root / "ref")
        self.drg = self.root / "drg_src"
        (self.drg / "reference.service_metadata").mkdir(parents=True)
        (self.drg / "reference.service_metadata" / "part-0.parquet").write_bytes(b"PAR1-fake")
        (self.drg / "reference.service_master").write_bytes(b"PAR1-fake-2")

    def extract(self, tar_path: Path) -> Path:
        dest = self.root / "extracted"
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest, filter="data")
        return dest

    def test_round_trip_is_verified_by_the_service_loader(self):
        out = self.root / "model.tar.gz"
        info = build_artifact(self.ref_dir, self.drg, out)
        self.assertEqual(info["counts"], {"diagnosis": 5, "achi": 2, "mbs": 2})
        extracted = self.extract(out)
        files = set(info["manifest"]["files"])
        self.assertEqual(files, {"reference/diagnosis_codes.tsv", "reference/achi_codes.tsv",
                                 "reference/MBS-XML-20260801.XML", "drg/reference.service_metadata/part-0.parquet",
                                 "drg/reference.service_master"})
        ref = ReferenceData.from_model_dir(extracted)
        self.assertEqual(ref.resolve_diagnosis("J189"), "Pneumonia, unspecified")
        (extracted / "reference" / "achi_codes.tsv").write_text("tampered")
        with self.assertRaises(ReferenceDataError):
            ReferenceData.from_model_dir(extracted)

    def test_tarball_is_deterministic_and_normalised(self):
        os.environ["SOURCE_DATE_EPOCH"] = "1700000000"
        self.addCleanup(os.environ.pop, "SOURCE_DATE_EPOCH", None)
        a, b = self.root / "a.tar.gz", self.root / "b.tar.gz"
        build_artifact(self.ref_dir, self.drg, a)
        build_artifact(self.ref_dir, self.drg, b)
        # the manifest's created_at differs by clock, so compare members rather than raw bytes
        with tarfile.open(a) as ta, tarfile.open(b) as tb:
            self.assertEqual([m.name for m in ta.getmembers()], [m.name for m in tb.getmembers()])
            for m in ta.getmembers():
                self.assertEqual((m.uid, m.gid, m.uname, m.gname, m.mtime), (0, 0, "", "", 1700000000))
                self.assertEqual(m.mode, 0o755 if m.isdir() else 0o644)
                self.assertFalse(m.name.startswith("/") or ".." in m.name)

    def test_symlinks_and_empty_drg_are_refused(self):
        os.symlink("/etc/passwd", self.drg / "link")
        with self.assertRaises(PackagingError):
            build_artifact(self.ref_dir, self.drg, self.root / "x.tar.gz")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(PackagingError):
            build_artifact(self.ref_dir, empty, self.root / "y.tar.gz")

    def test_wrong_reference_files_fail_before_packaging(self):
        (self.ref_dir / "achi_codes.tsv").write_text("0\tICD10_DIAGNOSIS\t1\t2\t3\t4\tx\ty\n")
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            code = package_main(["--reference-dir", str(self.ref_dir), "--no-drg", "--output", str(self.root / "z.tar.gz")])
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "z.tar.gz").exists())

    def test_no_drg_option(self):
        out = self.root / "nodrg.tar.gz"
        info = build_artifact(self.ref_dir, None, out)
        self.assertFalse([f for f in info["manifest"]["files"] if f.startswith("drg/")])


def good_config():
    return {
        "endpoint_name": "audit-prod", "region": "ap-southeast-2",
        "image_uri": f"{ACCOUNT}.dkr.ecr.ap-southeast-2.amazonaws.com/audit:abc123",
        "model_data_url": "s3://bkt/audit/v1/model.tar.gz",
        "execution_role_arn": f"arn:aws:iam::{ACCOUNT}:role/audit-exec",
        "kms_key_id": f"arn:aws:kms:ap-southeast-2:{ACCOUNT}:key/abc",
        "vpc": {"subnets": ["subnet-a", "subnet-b"], "security_group_ids": ["sg-1"]},
        "instance": {"type": "ml.m5.large", "count": 2},
        "autoscaling": {"min_capacity": 2, "max_capacity": 6, "target_invocations_per_instance": 20},
        "monitoring": {"sns_topic_arn": f"arn:aws:sns:ap-southeast-2:{ACCOUNT}:alerts"},
        "environment": {"AUDIT_BEDROCK_MODEL_ID": GLOBAL, "AUDIT_ALLOW_GLOBAL_INFERENCE": True, "AUDIT_LLM_EFFORT": "high"},
        "tags": {"Application": "audit"},
    }


class DeployConfigTests(unittest.TestCase):
    def problems(self, mutate=None, **flags):
        cfg = good_config()
        if mutate:
            mutate(cfg)
        return validate_config(cfg, **flags)

    def test_good_config_passes(self):
        self.assertEqual(self.problems(), [])

    def test_residency_guard_is_enforced_before_deploy(self):
        out = self.problems(lambda c: c["environment"].update(AUDIT_ALLOW_GLOBAL_INFERENCE="false"))
        self.assertTrue(any("AUDIT_ALLOW_GLOBAL_INFERENCE" in p for p in out))

    def test_rejections(self):
        cases = {
            "placeholder": (lambda c: c.update(model_data_url="s3://<bucket>/x.tar.gz"), "placeholder"),
            "latest tag": (lambda c: c.update(image_uri=f"{ACCOUNT}.dkr.ecr.ap-southeast-2.amazonaws.com/audit:latest"), "latest"),
            "not ecr": (lambda c: c.update(image_uri="docker.io/library/python:3.12"), "ECR"),
            "bad role": (lambda c: c.update(execution_role_arn="role"), "IAM role"),
            "one instance": (lambda c: c["instance"].update(count=1), "instance.count"),
            "no kms": (lambda c: c.pop("kms_key_id"), "kms_key_id"),
            "one subnet": (lambda c: c["vpc"].update(subnets=["subnet-a"]), "vpc"),
            "no vpc": (lambda c: c.pop("vpc"), "vpc"),
            "secret env": (lambda c: c["environment"].update(AUDIT_API_KEY="x"), "secret"),
            "foreign env": (lambda c: c["environment"].update(HTTP_PROXY="x"), "not an AUDIT_"),
            "bad autoscaling": (lambda c: c["autoscaling"].update(max_capacity=1), "autoscaling"),
            "bad name": (lambda c: c.update(endpoint_name="bad_name!"), "endpoint_name"),
            "bad artifact": (lambda c: c.update(model_data_url="s3://b/model.zip"), "model_data_url"),
        }
        for name, (mutate, expect) in cases.items():
            with self.subTest(name):
                self.assertTrue(any(expect in p for p in self.problems(mutate)), self.problems(mutate))

    def test_escape_hatches_are_explicit(self):
        def mutate(c):
            c.pop("kms_key_id"); c.pop("vpc"); c["instance"]["count"] = 1; c["autoscaling"] = None
        self.assertEqual(self.problems(mutate, allow_no_vpc=True, allow_no_kms=True, allow_single_instance=True), [])

    def test_environment_values_become_strings(self):
        env = container_environment(good_config())
        self.assertEqual(env["AUDIT_ALLOW_GLOBAL_INFERENCE"], "true")
        self.assertEqual(env["AUDIT_AWS_REGION"], "ap-southeast-2")
        self.assertTrue(all(isinstance(v, str) for v in env.values()))

    def test_find_placeholders(self):
        self.assertEqual(find_placeholders({"a": ["<x>", "ok"], "b": {"c": "s3://<bucket>/k"}}), ["a[0]", "b.c"])


class DeployPlanTests(unittest.TestCase):
    def test_first_deployment_creates_everything_in_a_safe_order(self):
        plan = build_plan(good_config(), release="r1", endpoint_exists=False)
        ops = [s.operation for s in plan]
        self.assertEqual(ops[:2], ["create_model", "create_endpoint_config"])
        self.assertEqual(ops.count("put_metric_alarm"), 4)
        self.assertLess(max(i for i, o in enumerate(ops) if o == "put_metric_alarm"), ops.index("create_endpoint"))
        self.assertLess(ops.index("create_endpoint"), ops.index("endpoint_in_service"))
        self.assertLess(ops.index("endpoint_in_service"), ops.index("register_scalable_target"))
        self.assertLess(ops.index("register_scalable_target"), ops.index("put_metric_filter"))
        self.assertNotIn("update_endpoint", ops)

    def test_model_and_config_follow_the_security_posture(self):
        plan = {s.operation: s for s in build_plan(good_config(), release="r1", endpoint_exists=False)}
        model = plan["create_model"].params
        self.assertFalse(model["EnableNetworkIsolation"])
        self.assertEqual(model["VpcConfig"]["Subnets"], ["subnet-a", "subnet-b"])
        self.assertEqual(model["PrimaryContainer"]["Environment"]["AUDIT_ALLOW_GLOBAL_INFERENCE"], "true")
        self.assertEqual(model["ModelName"], "audit-prod-r1")
        config = plan["create_endpoint_config"].params
        self.assertEqual(config["KmsKeyId"], good_config()["kms_key_id"])
        self.assertNotIn("DataCaptureConfig", config)
        variant = config["ProductionVariants"][0]
        self.assertEqual(variant["RoutingConfig"], {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"})
        self.assertEqual((variant["InstanceType"], variant["InitialInstanceCount"]), ("ml.m5.large", 2))

    def test_update_uses_a_canary_with_automatic_rollback_on_existing_alarms(self):
        plan = build_plan(good_config(), release="r2", endpoint_exists=True)
        ops = [s.operation for s in plan]
        self.assertIn("update_endpoint", ops)
        self.assertNotIn("create_endpoint", ops)
        update = next(s for s in plan if s.operation == "update_endpoint").params
        bg = update["DeploymentConfig"]["BlueGreenUpdatePolicy"]
        self.assertEqual(bg["TrafficRoutingConfiguration"]["Type"], "CANARY")
        self.assertEqual(bg["TerminationWaitInSeconds"], 600)
        alarm_names = {s.params["AlarmName"] for s in plan if s.operation == "put_metric_alarm"}
        rollback = {a["AlarmName"] for a in update["DeploymentConfig"]["AutoRollbackConfiguration"]["Alarms"]}
        self.assertTrue(rollback and rollback <= alarm_names)

    def test_alarms_and_autoscaling_values(self):
        plan = build_plan(good_config(), release="r1", endpoint_exists=False)
        alarms = {s.params["AlarmName"]: s.params for s in plan if s.operation == "put_metric_alarm"}
        latency = alarms["audit-prod-latency-p99"]
        self.assertEqual((latency["ExtendedStatistic"], latency["Threshold"]), ("p99", 55000.0 * 1000))  # microseconds
        self.assertEqual(alarms["audit-prod-5xx"]["AlarmActions"], [good_config()["monitoring"]["sns_topic_arn"]])
        policy = next(s for s in plan if s.operation == "put_scaling_policy").params
        self.assertEqual(policy["ResourceId"], "endpoint/audit-prod/variant/AllTraffic")
        self.assertEqual(policy["TargetTrackingScalingPolicyConfiguration"]["TargetValue"], 20.0)

    def test_metric_filters_target_the_endpoint_log_group_and_are_optional(self):
        filters = [s for s in build_plan(good_config(), release="r1", endpoint_exists=False) if s.operation == "put_metric_filter"]
        self.assertEqual(len(filters), 7)
        self.assertTrue(all(s.optional and s.params["logGroupName"] == "/aws/sagemaker/Endpoints/audit-prod" for s in filters))
        tokens = next(s for s in filters if s.params["filterName"].endswith("llm-input-tokens")).params
        self.assertEqual(tokens["metricTransformations"][0]["metricValue"], "$.input_tokens")

    def test_resource_names_are_validated(self):
        with self.assertRaises(Problems):
            names(good_config(), "bad release!")

    def test_apply_plan_runs_steps_in_order_tolerates_optional_failures_and_explains_a_failed_wait(self):
        calls = []

        class Waiter:
            def __init__(self, fail): self.fail = fail
            def wait(self, **kw):
                if self.fail: raise RuntimeError("timeout")
                calls.append(("wait", kw["EndpointName"]))

        class Client:
            def __init__(self, name, fail_logs=False, fail_wait=False): self.name, self.fail_logs, self.fail_wait = name, fail_logs, fail_wait
            def __getattr__(self, op):
                def call(**kw):
                    if self.name == "logs" and self.fail_logs: raise RuntimeError("ResourceNotFoundException: no log group")
                    calls.append((self.name, op))
                    return {"FailureReason": "image pull failed"}
                return call
            def get_waiter(self, name): return Waiter(self.fail_wait)

        class Session:
            def __init__(self, **kw): self.kw = kw
            def client(self, service): return Client(service, **self.kw)

        plan = build_plan(good_config(), release="r1", endpoint_exists=False)
        logs = []
        apply_plan(plan, Session(fail_logs=True), log=logs.append)
        self.assertEqual(calls[0], ("sagemaker", "create_model"))
        self.assertTrue(any("warning" in line for line in logs))
        calls.clear()
        with self.assertRaises(RuntimeError) as ctx:
            apply_plan(plan, Session(fail_wait=True), log=lambda _m: None)
        self.assertIn("image pull failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
