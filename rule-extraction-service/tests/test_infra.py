import importlib.util
import json
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PSEUDO = {"AWS::Region", "AWS::AccountId", "AWS::Partition", "AWS::StackName", "AWS::NoValue", "AWS::URLSuffix"}


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"Fn::" + suffix if suffix not in ("Ref", "Condition") else suffix: value}


_Loader.add_multi_constructor("!", _tag)


def load(name):
    return yaml.load((ROOT / "infra" / "cloudformation" / name).read_text(), Loader=_Loader)


def walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


class TemplateLint(unittest.TestCase):
    def check_references(self, template):
        names = set(template.get("Parameters", {})) | set(template["Resources"]) | PSEUDO
        conditions = set(template.get("Conditions", {}))
        for node in walk(template):
            if not isinstance(node, dict):
                continue
            if "Ref" in node and isinstance(node["Ref"], str):
                self.assertIn(node["Ref"], names, f"Ref to unknown {node['Ref']}")
            if "Fn::GetAtt" in node:
                target = node["Fn::GetAtt"]
                resource = (target[0] if isinstance(target, list) else target.split(".")[0])
                self.assertIn(resource, template["Resources"], f"GetAtt on unknown {resource}")
            if "Fn::Sub" in node:
                text = node["Fn::Sub"] if isinstance(node["Fn::Sub"], str) else node["Fn::Sub"][0]
                for ref in re.findall(r"\$\{([^}!]+)\}", text):
                    self.assertIn(ref.split(".")[0], names, f"Sub references unknown {ref}")
            if "Fn::If" in node:
                self.assertIn(node["Fn::If"][0], conditions)
        for res in template["Resources"].values():
            self.assertIn("Type", res)
            if "Condition" in res:
                self.assertIn(res["Condition"], conditions)
            for dep in ([res["DependsOn"]] if isinstance(res.get("DependsOn"), str) else res.get("DependsOn", [])):
                self.assertIn(dep, template["Resources"])
        for cond in template.get("Conditions", {}).values():
            for node in walk(cond):
                if isinstance(node, dict) and "Ref" in node:
                    self.assertIn(node["Ref"], names)

    def test_foundation_references_resolve(self):
        self.check_references(load("foundation.yaml"))

    def test_endpoint_references_resolve(self):
        self.check_references(load("endpoint.yaml"))

    def test_fn_equals_only_used_in_conditions(self):
        for name in ("foundation.yaml", "endpoint.yaml"):
            t = load(name)
            for res in t["Resources"].values():
                for node in walk(res.get("Properties", {})):
                    if isinstance(node, dict):
                        self.assertNotIn("Fn::Equals", node, f"{name}: Fn::Equals is invalid inside resource properties")

    def test_iam_least_privilege(self):
        role = load("foundation.yaml")["Resources"]["ExecutionRole"]["Properties"]
        statements = role["Policies"][0]["PolicyDocument"]["Statement"]
        text = json.dumps(statements)
        for forbidden in ('"bedrock:*"', '"s3:*"', '"dynamodb:*"', '"sqs:*"', '"kms:*"', "aws-marketplace", '"iam:'):
            self.assertNotIn(forbidden, text)
        for s in statements:
            actions = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
            self.assertNotIn("*", actions)
            if s["Resource"] == "*":
                self.assertTrue(set(actions) <= {"bedrock:ListInferenceProfiles", "bedrock:GetInferenceProfile", "ecr:GetAuthorizationToken",
                                                 "cloudwatch:PutMetricData", "ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission",
                                                 "ec2:DeleteNetworkInterface", "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces",
                                                 "ec2:DescribeVpcs", "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups"}, actions)
        invoke = [s for s in statements if "bedrock:InvokeModel" in s["Action"]]
        self.assertEqual(len(invoke), 2)
        self.assertTrue(all(s["Resource"] != "*" for s in invoke))
        metrics = next(s for s in statements if s["Sid"] == "PublishMetrics")
        self.assertIn("cloudwatch:namespace", json.dumps(metrics["Condition"]))
        self.assertEqual(role["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["Service"], "sagemaker.amazonaws.com")

    def test_data_stores_are_encrypted_and_protected(self):
        res = load("foundation.yaml")["Resources"]
        bucket = res["ArtifactBucket"]["Properties"]
        self.assertTrue(all(bucket["PublicAccessBlockConfiguration"].values()))
        self.assertEqual(bucket["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]["SSEAlgorithm"], "aws:kms")
        self.assertEqual(bucket["VersioningConfiguration"]["Status"], "Enabled")
        self.assertIn("aws:SecureTransport", json.dumps(res["ArtifactBucketPolicy"]))
        self.assertTrue(res["StateTable"]["Properties"]["SSESpecification"]["SSEEnabled"])
        self.assertTrue(res["StateTable"]["Properties"]["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"])
        self.assertIn("KmsMasterKeyId", res["WorkQueue"]["Properties"])
        self.assertEqual(res["Repository"]["Properties"]["ImageTagMutability"], "IMMUTABLE")
        self.assertTrue(res["Repository"]["Properties"]["ImageScanningConfiguration"]["ScanOnPush"])
        self.assertTrue(res["Key"]["Properties"]["EnableKeyRotation"])
        for retained in ("ArtifactBucket", "StateTable", "Key"):
            self.assertEqual(res[retained]["DeletionPolicy"], "Retain")

    def test_endpoint_environment_matches_settings_and_validation(self):
        from rule_extractor.settings import ENV_PREFIX
        env = load("endpoint.yaml")["Resources"]["Model"]["Properties"]["PrimaryContainer"]["Environment"]
        source = (ROOT / "src" / "rule_extractor" / "settings.py").read_text()
        known = {ENV_PREFIX + n for n in re.findall(r'reader\.(?:str|int|float|bool|csv|choice)\(\s*"([A-Z_]+)"', source)}
        known |= {ENV_PREFIX + "BEDROCK_REGION", ENV_PREFIX + "CONFIG_FILE"}
        for key in env:
            if key.startswith(ENV_PREFIX):
                self.assertIn(key, known, f"{key} is not a recognised setting")
        for required in ("RULE_EXTRACTOR_STATE_TABLE", "RULE_EXTRACTOR_QUEUE_URL", "RULE_EXTRACTOR_ARTIFACT_BUCKET", "AWS_REGION"):
            self.assertIn(required, env)
        self.assertEqual(env["RULE_EXTRACTOR_ENV"], "prod")
        self.assertEqual(env["RULE_EXTRACTOR_BACKEND"], "aws")

    def test_endpoint_resilience_settings(self):
        t = load("endpoint.yaml")
        variant = t["Resources"]["EndpointConfig"]["Properties"]["ProductionVariants"][0]
        self.assertGreaterEqual(variant["ContainerStartupHealthCheckTimeoutInSeconds"], 600)
        self.assertEqual(t["Parameters"]["InstanceCount"]["Default"], 2)
        deploy = t["Resources"]["Endpoint"]["Properties"]["DeploymentConfig"]
        self.assertTrue(deploy["AutoRollbackConfiguration"]["Alarms"])
        self.assertEqual(t["Resources"]["Model"]["Properties"]["EnableNetworkIsolation"], False)   # Bedrock calls need egress via VPC endpoints
        for conditional in ("ScalableTarget", "ScalingPolicy"):
            self.assertEqual(t["Resources"][conditional]["Condition"], "UseAutoScaling")


class ScriptTests(unittest.TestCase):
    def load_script(self, name):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_deploy_stack_create_update_and_noop(self):
        deploy = self.load_script("deploy")

        class Waiter:
            def __init__(self, log, name): self.log, self.name = log, name
            def wait(self, **kw): self.log.append(("wait", self.name))

        class Cfn:
            def __init__(self, exists, update_error=None):
                self.exists, self.update_error, self.log = exists, update_error, []
            def describe_stacks(self, StackName):
                if not self.exists:
                    raise deploy.ClientError("does not exist")
                return {"Stacks": [{"Outputs": [{"OutputKey": "K", "OutputValue": "V"}]}]}
            def create_stack(self, **kw):
                self.log.append(("create", kw["Parameters"], kw["Capabilities"], kw.get("EnableTerminationProtection")))
                self.exists = True
            def update_stack(self, **kw):
                self.log.append(("update",))
                if self.update_error:
                    raise self.update_error
            def get_waiter(self, name): return Waiter(self.log, name)

        tpl = str(ROOT / "infra" / "cloudformation" / "foundation.yaml")
        c = Cfn(False)
        out = deploy.deploy_stack(c, "s", tpl, {"ProjectName": "x"})
        self.assertEqual(out, {"K": "V"})
        self.assertEqual(c.log[0], ("create", [{"ParameterKey": "ProjectName", "ParameterValue": "x"}], ["CAPABILITY_NAMED_IAM"], True))
        c2 = Cfn(True)
        deploy.deploy_stack(c2, "s", tpl, {})
        self.assertEqual(c2.log[0], ("update",))
        c3 = Cfn(True, update_error=deploy.ClientError("No updates are to be performed."))
        self.assertEqual(deploy.deploy_stack(c3, "s", tpl, {}), {"K": "V"})
        c4 = Cfn(True, update_error=deploy.ClientError("AccessDenied"))
        with self.assertRaises(Exception):
            deploy.deploy_stack(c4, "s", tpl, {})

    def test_invoke_options_mapping(self):
        mod = self.load_script("invoke_endpoint")
        import argparse
        ns = argparse.Namespace(topics="0110,0002", limit_topics=3, limit_sweep_windows=None, no_tabular_sweep=True, no_acs=False, max_cost_usd=25.0)
        self.assertEqual(mod.options_from(ns), {"topics": ["0110", "0002"], "limit_topics": 3, "include_tabular_sweep": False, "max_cost_usd": 25.0})


class RepoHygieneTests(unittest.TestCase):
    SECRET_PATTERNS = [r"AKIA[0-9A-Z]{16}", r"ASIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"(?i)aws_secret_access_key\s*=",
                       r"sk-ant-[A-Za-z0-9_-]{10,}", r"(?i)\bapi[_-]?key\s*[:=]\s*['\"][A-Za-z0-9]{16,}"]

    def files(self, *globs):
        for g in globs:
            for p in ROOT.glob(g):
                if p.is_file() and "__pycache__" not in p.parts and ".venv" not in p.parts:
                    yield p

    def test_no_secrets_anywhere(self):
        for path in self.files("**/*.py", "**/*.yaml", "**/*.yml", "**/*.md", "**/*.sh", "**/*.toml", "**/*.in", "**/.env.example", "Makefile", "Dockerfile"):
            text = path.read_text(errors="ignore")
            for pattern in self.SECRET_PATTERNS:
                self.assertIsNone(re.search(pattern, text), f"{path.relative_to(ROOT)} matches secret pattern {pattern}")

    def test_no_notice_footer_in_readme_or_scripts(self):
        for path in self.files("README.md", "scripts/*", "docker/*", "Makefile", "Dockerfile", "src/**/*.py", "src/**/*.md"):
            text = path.read_text(errors="ignore")
            self.assertNotIn("Generated by AI", text, path.name)
            self.assertNotIn("All rights reserved", text, path.name)
            self.assertNotIn("sole and exclusive property", text, path.name)

    def test_dangerous_constructs_are_absent_from_source(self):
        bad = [r"\beval\(", r"\bexec\(", r"shell\s*=\s*True", r"pickle\.loads?\(", r"yaml\.load\(", r"os\.system\(",
               r"permission_mode\s*=\s*[\"']bypassPermissions", r"dangerously", r"skipWebFetchPreflight", r"verify\s*=\s*False"]
        for path in self.files("src/**/*.py", "scripts/*.py"):
            text = path.read_text()
            for pattern in bad:
                self.assertIsNone(re.search(pattern, text), f"{path.relative_to(ROOT)} matches {pattern}")

    def test_image_never_receives_licensed_data_and_installs_hash_locked(self):
        ignore = (ROOT / ".dockerignore").read_text().split()
        for needed in ("source_docs", "corpus", "dist", "build", "**/*.pdf", ".env"):
            self.assertIn(needed, ignore)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("--require-hashes", dockerfile)
        self.assertNotIn("COPY source_docs", dockerfile)
        self.assertIn('ENTRYPOINT ["/opt/program/entrypoint.sh"]', dockerfile)       # exec form -> SIGTERM reaches PID 1
        self.assertIn("RULE_EXTRACTOR_ENV=prod", dockerfile)
        gitignore = (ROOT / ".gitignore").read_text()
        for needed in ("corpus/", "source_docs/*.pdf", ".env"):
            self.assertIn(needed, gitignore)

    def test_env_example_has_no_credentials(self):
        text = (ROOT / ".env.example").read_text()
        self.assertNotRegex(text, r"(?im)^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN|AWS_BEARER_TOKEN_BEDROCK)\s*=")

    def test_default_config_loads_and_stores_no_notice_text(self):
        from rule_extractor.settings import Settings
        s = Settings.load(env={"RULE_EXTRACTOR_CONFIG_FILE": str(ROOT / "configs" / "base.yaml")})
        self.assertEqual(s.output.footer_lines, ())
        self.assertFalse(s.web.enabled)
        self.assertEqual(s.web.allowed_hosts, ("www.ihacpa.gov.au", "ihacpa.gov.au"))


if __name__ == "__main__":
    unittest.main()
