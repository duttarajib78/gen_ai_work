#!/usr/bin/env python3
"""Offline structural checks for the CloudFormation templates (a stand-in where cfn-lint is not installed -- run cfn-lint as well in CI).

It parses the YAML with the CloudFormation short-form tags and verifies: every Ref / GetAtt / Sub variable / Condition / DependsOn resolves; parameter
patterns compile and accept their defaults; exports and imports match; and a set of SECURITY invariants that must never regress -- no wildcard actions, a
wildcard resource only for the few actions that require one, no world-open security-group rule, a private encrypted versioned bucket that refuses plain HTTP,
and an endpoint that is in a VPC with automatic rollback.
    python deploy/scripts/validate_templates.py [foundation.yaml endpoint.yaml]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PSEUDO = {"AWS::AccountId", "AWS::NotificationARNs", "AWS::NoValue", "AWS::Partition", "AWS::Region", "AWS::StackId", "AWS::StackName", "AWS::URLSuffix"}
TAGS = ["Ref", "Sub", "GetAtt", "If", "Equals", "Not", "Or", "And", "Join", "Split", "Select", "FindInMap", "Condition", "ImportValue", "Base64", "Cidr", "GetAZs", "Transform"]
WILDCARD_RESOURCE_ACTIONS = {"ecr:GetAuthorizationToken", "ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
                             "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs", "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets",
                             "ec2:DescribeSecurityGroups"}


class Loader(yaml.SafeLoader):
    pass


def _tag(name):
    key = "Condition" if name == "Condition" else f"Fn::{name}" if name != "Ref" else "Ref"

    def construct(loader, node):
        if isinstance(node, yaml.ScalarNode):
            val = loader.construct_scalar(node)
            if name == "GetAtt":
                val = val.split(".", 1)
        elif isinstance(node, yaml.SequenceNode):
            val = loader.construct_sequence(node, deep=True)
        else:
            val = loader.construct_mapping(node, deep=True)
        return {key: val}
    return construct


for _t in TAGS:
    Loader.add_constructor(f"!{_t}", _tag(_t))


def walk(node, path=""):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)  # noqa: S506 - SafeLoader subclass with only CloudFormation tags


def check_references(t: dict, name: str) -> list[str]:
    errs: list[str] = []
    params, res, conds = set(t.get("Parameters", {})), set(t.get("Resources", {})), set(t.get("Conditions", {}))
    known = params | res | PSEUDO
    for path, node in walk(t):
        if not isinstance(node, dict):
            continue
        if "Ref" in node and len(node) == 1 and node["Ref"] not in known:
            errs.append(f"{name}{path}: Ref to unknown {node['Ref']!r}")
        if "Fn::GetAtt" in node:
            target = node["Fn::GetAtt"][0] if isinstance(node["Fn::GetAtt"], list) else str(node["Fn::GetAtt"]).split(".")[0]
            if target not in res:
                errs.append(f"{name}{path}: GetAtt on unknown resource {target!r}")
        if "Fn::Sub" in node:
            sub = node["Fn::Sub"]
            text, local = (sub[0], set(sub[1])) if isinstance(sub, list) else (sub, set())
            for var in re.findall(r"\$\{([^}!]+)\}", text):
                base = var.split(".")[0]
                if base not in known and base not in local:
                    errs.append(f"{name}{path}: Sub variable ${{{var}}} is not a parameter, resource or local name")
        if "Fn::If" in node and node["Fn::If"][0] not in conds:
            errs.append(f"{name}{path}: Fn::If on unknown condition {node['Fn::If'][0]!r}")
        if "Condition" in node and isinstance(node["Condition"], str) and path.count("/") <= 3 and node["Condition"] not in conds:
            errs.append(f"{name}{path}: Condition {node['Condition']!r} is not defined")
    for rname, r in t.get("Resources", {}).items():
        dep = r.get("DependsOn", [])
        for d in ([dep] if isinstance(dep, str) else dep):
            if d not in res:
                errs.append(f"{name}/Resources/{rname}: DependsOn unknown {d!r}")
        if "Condition" in r and r["Condition"] not in conds:
            errs.append(f"{name}/Resources/{rname}: Condition {r['Condition']!r} is not defined")
    used = {n["Fn::If"][0] for _, n in walk(t) if isinstance(n, dict) and "Fn::If" in n}
    used |= {r["Condition"] for r in t.get("Resources", {}).values() if "Condition" in r}
    used |= {n["Condition"] for _, n in walk(t.get("Conditions", {})) if isinstance(n, dict) and "Condition" in n}
    for c in conds - used:
        errs.append(f"{name}: condition {c!r} is defined but never used")
    for path, node in walk(t.get("Conditions", {})):
        if isinstance(node, dict) and "Fn::ImportValue" in node:
            errs.append(f"{name}/Conditions{path}: Fn::ImportValue is not allowed inside a condition")
    return errs


def check_parameters(t: dict, name: str) -> list[str]:
    errs = []
    for pname, p in t.get("Parameters", {}).items():
        pat, default = p.get("AllowedPattern"), p.get("Default")
        if pat:
            try:
                rx = re.compile(pat)
            except re.error as e:
                errs.append(f"{name}: parameter {pname} has an invalid AllowedPattern: {e}")
                continue
            if default not in (None, "") and not rx.match(str(default)):
                errs.append(f"{name}: parameter {pname} default {default!r} does not match its own AllowedPattern")
        if "AllowedValues" in p and default is not None and default not in p["AllowedValues"] and str(default) not in map(str, p["AllowedValues"]):
            errs.append(f"{name}: parameter {pname} default {default!r} is not in AllowedValues")
        if "Type" not in p:
            errs.append(f"{name}: parameter {pname} has no Type")
    return errs


def statements(t: dict):
    for rname, r in t.get("Resources", {}).items():
        for path, node in walk(r.get("Properties", {})):
            if isinstance(node, dict) and "Effect" in node and ("Action" in node or "NotAction" in node):
                yield rname, node


def as_list(v):
    return v if isinstance(v, list) else [v]


def rules_of(items: list) -> list[dict]:
    """Security-group rules with any Fn::If expanded to its dictionary branches (a NoValue branch is not a rule)."""
    out = []
    for x in items:
        if isinstance(x, dict) and "Fn::If" in x:
            out += [b for b in x["Fn::If"][1:] if isinstance(b, dict) and "IpProtocol" in b]
        elif isinstance(x, dict) and "IpProtocol" in x:
            out.append(x)
    return out


def check_security(t: dict, name: str) -> list[str]:
    errs = []
    for rname, st in statements(t):
        actions, res, principal = [str(a) for a in as_list(st.get("Action", []))], as_list(st.get("Resource", [])), st.get("Principal")
        if "NotAction" in st or "NotResource" in st:
            errs.append(f"{name}/{rname}: NotAction / NotResource is not allowed")
        if st.get("Effect") == "Allow":
            if any(a in ("*", "*:*") or a.endswith(":*") for a in actions) and not (rname == "KmsKey"):
                errs.append(f"{name}/{rname}: wildcard action {actions}")
            if any(r == "*" for r in res) and not set(actions) <= WILDCARD_RESOURCE_ACTIONS and rname != "KmsKey":
                errs.append(f"{name}/{rname}: Resource '*' with actions {sorted(set(actions) - WILDCARD_RESOURCE_ACTIONS)} that need a specific resource")
            if principal == "*" or (isinstance(principal, dict) and principal.get("AWS") == "*"):
                errs.append(f"{name}/{rname}: Allow to every principal")
    for rname, r in t.get("Resources", {}).items():
        p = r.get("Properties", {})
        if r["Type"] == "AWS::EC2::SecurityGroup":
            if p.get("SecurityGroupIngress") and "VpcEndpointSecurityGroup" != rname:
                errs.append(f"{name}/{rname}: only the VPC endpoint group may have ingress rules")
            for rule in rules_of(p.get("SecurityGroupIngress", [])) + rules_of(p.get("SecurityGroupEgress", [])):
                if rule.get("CidrIp") in ("0.0.0.0/0",) or rule.get("CidrIpv6") in ("::/0",):
                    errs.append(f"{name}/{rname}: world-open rule")
                if rule.get("IpProtocol") != "tcp" or rule.get("FromPort") != 443 or rule.get("ToPort") != 443:
                    errs.append(f"{name}/{rname}: every rule must be tcp/443")
        if r["Type"] == "AWS::S3::Bucket":
            pab = p.get("PublicAccessBlockConfiguration", {})
            if not all(pab.get(k) is True for k in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets")):
                errs.append(f"{name}/{rname}: the bucket must block all public access")
            if "BucketEncryption" not in p:
                errs.append(f"{name}/{rname}: the bucket must have default encryption")
            if p.get("VersioningConfiguration", {}).get("Status") != "Enabled":
                errs.append(f"{name}/{rname}: versioning must be enabled")
            if r.get("DeletionPolicy") != "Retain":
                errs.append(f"{name}/{rname}: data stores must have DeletionPolicy: Retain")
        if r["Type"] == "AWS::S3::BucketPolicy":
            if not any(s.get("Effect") == "Deny" and s.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport") == "false" for s in as_list(p["PolicyDocument"]["Statement"])):
                errs.append(f"{name}/{rname}: the bucket policy must deny non-TLS requests")
        if r["Type"] == "AWS::ECR::Repository" and p.get("ImageTagMutability") != "IMMUTABLE":
            errs.append(f"{name}/{rname}: image tags must be immutable")
        if r["Type"] == "AWS::SageMaker::Model":
            if p.get("EnableNetworkIsolation") is not False:
                errs.append(f"{name}/{rname}: EnableNetworkIsolation must be stated (false: the container has to reach Bedrock)")
            if not p.get("VpcConfig"):
                errs.append(f"{name}/{rname}: the model must run in a VPC")
            env = p["PrimaryContainer"].get("Environment", {})
            for must in ("DX_ENV", "DX_BACKEND", "DX_STORE_URI", "DX_DATA_DIR"):
                if must not in env:
                    errs.append(f"{name}/{rname}: container environment lacks {must}")
            if env.get("DX_ENV") != "production" or env.get("DX_BACKEND") != "bedrock":
                errs.append(f"{name}/{rname}: the endpoint must run DX_ENV=production with the bedrock backend")
            for k in env:
                if re.search(r"KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", k) and k != "DX_STORE_KMS_KEY_ID":
                    errs.append(f"{name}/{rname}: environment variable {k} looks like a secret; the endpoint authenticates with its IAM role only")
        if r["Type"] == "AWS::SageMaker::Endpoint":
            if not p.get("DeploymentConfig", {}).get("AutoRollbackConfiguration", {}).get("Alarms"):
                errs.append(f"{name}/{rname}: updates must roll back automatically on alarms")
    text = Path(name).read_text(encoding="utf-8") if Path(name).exists() else ""
    if re.search(r"AKIA[0-9A-Z]{16}", text):
        errs.append(f"{name}: contains something that looks like an access key id")
    return errs


def check_exports(found: dict, endpoint: dict) -> list[str]:
    exported = {}
    for oname, o in found.get("Outputs", {}).items():
        if "Export" in o:
            nm = o["Export"]["Name"]
            exported[oname] = nm["Fn::Sub"] if isinstance(nm, dict) else nm
    errs = []
    for path, node in walk(endpoint):
        if isinstance(node, dict) and "Fn::ImportValue" in node:
            iv = node["Fn::ImportValue"]
            text = iv["Fn::Sub"] if isinstance(iv, dict) else iv
            m = re.fullmatch(r"\$\{FoundationStackName\}-(\w+)", str(text))
            if not m:
                errs.append(f"endpoint{path}: unexpected import {text!r}")
            elif f"${{AWS::StackName}}-{m.group(1)}" not in exported.values():
                errs.append(f"endpoint{path}: the foundation stack exports nothing named {m.group(1)!r}")
    return errs


def validate(foundation: Path, endpoint: Path) -> list[str]:
    f, e = load(foundation), load(endpoint)
    errs: list[str] = []
    for t, p in ((f, foundation), (e, endpoint)):
        n = str(p)
        errs += [x for x in (f"{n}: missing top-level key {k}" for k in ("AWSTemplateFormatVersion", "Resources")) if k_missing(t, x)]
        errs += check_references(t, n) + check_parameters(t, n) + check_security(t, n)
    return errs + check_exports(f, e)


def k_missing(t: dict, msg: str) -> bool:
    return msg.rsplit(" ", 1)[-1] not in t


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1] / "cloudformation"
    foundation = Path(argv[0]) if argv else root / "foundation.yaml"
    endpoint = Path(argv[1]) if len(argv) > 1 else root / "endpoint.yaml"
    errs = validate(foundation, endpoint)
    for e in errs:
        print("ERROR", e)
    print("OK: templates are structurally consistent" if not errs else f"{len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
