# Admissions audit agent

LLM-assisted review of whether the diagnoses coded on a hospital admission are clinically supported by the procedures and services billed for it. Served from a managed Amazon SageMaker real-time endpoint, with Claude Sonnet 5.5 on Amazon Bedrock as the model.

- **Input:** one hospital episode (ICD-10-AM diagnoses, ACHI procedures, MBS items, plus the fields the DRG grouper needs)
- **Output:** a per-diagnosis verdict, and for flagged episodes a proposed correction, a regrouped DRG, and an alert record in the existing alert schema
- **Blind by design:** the model sees code descriptions only. It never sees identifiers, dates, age, sex, DRG, or any financial field

## Contents

1. [Read this first](#1-read-this-first)
2. [Architecture](#2-architecture)
3. [Decisions required before go-live](#3-decisions-required-before-go-live)
4. [Project structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Configuration reference](#6-configuration-reference)
7. [Develop in SageMaker Code Editor](#7-develop-in-sagemaker-code-editor)
8. [Build and deploy](#8-build-and-deploy)
9. [Inference](#9-inference)
10. [Operations](#10-operations)
11. [Security controls](#11-security-controls)
12. [Evaluating accuracy and the decision threshold](#12-evaluating-accuracy-and-the-decision-threshold)
13. [Testing](#13-testing)
14. [Changes from the previous implementation](#14-changes-from-the-previous-implementation)
15. [Verification status and known limits](#15-verification-status-and-known-limits)

---

## 1. Read this first

| Item | Why it matters |
|---|---|
| The service **will not start** with a `global.` inference profile until `AUDIT_ALLOW_GLOBAL_INFERENCE=true` | Routing can leave Australia. This is a deliberate fail-closed guard; see [section 3](#3-decisions-required-before-go-live) |
| `vendor/au_grouper.py` is **not in this repository** | The internal grouper must be vendored into the image build. Without it, build with `REQUIRE_GROUPER=false` and run `AUDIT_GROUPER_BACKEND=disabled` |
| Nothing here has been run against real AWS services | Run `make smoke-bedrock` first. See [section 15](#15-verification-status-and-known-limits) |
| The 0.5 low-confidence threshold was set for a different model | Re-validate it with [section 12](#12-evaluating-accuracy-and-the-decision-threshold) before alerts reach reviewers |
| Regular calls are cut at **60 seconds** by SageMaker | At effort `high` large episodes can exceed that. Use streaming (8 minutes) or a lower effort; see [section 9](#9-inference) |

---

## 2. Architecture

```
 Client (batch job, application)                       AWS account, private subnets, 2+ instances
 -------------------------------                       ------------------------------------------------------
  boto3 sagemaker-runtime  --- SigV4 / IAM --->  SageMaker real-time endpoint (ml.m5.large, CPU only)
  invoke_endpoint (<= 60 s)                       |
  invoke_endpoint_with_response_stream (<= 8 min) |  container: gunicorn + Flask   GET /ping   POST /invocations
                                                  |
                                                  |  1. validate request (strict allowlists, size limits)
                                                  |  2. format de-identified episode text (reference descriptions)
                                                  |  3. Claude Sonnet 5.5 via Bedrock Converse  --- VPC endpoint ---> Amazon Bedrock
                                                  |       phase 1: verdict for the PDx and every ADx (one call)
                                                  |       phase 2: replacement principal search (only if the PDx failed)
                                                  |  4. decision rule -> correction -> AR-DRG regroup (internal grouper)
                                                  |  5. alert text (HTML-encoded) and alert record
                                                  |
                                                  +-- reads /opt/ml/model  (reference data, verified against manifest.json)
                                                  +-- writes JSON logs (no episode content) -> CloudWatch Logs
                                                         -> metric filters -> CloudWatch alarms -> SNS and automatic rollback
```

### Request lifecycle

1. SageMaker routes the call to the least-busy instance (`LEAST_OUTSTANDING_REQUESTS`).
2. `schemas.parse_request` validates the body. Anything malformed is a 400 before any model call is made.
3. `episode_text.build_episode_text` builds the prompt text from reference descriptions (duplicate codes collapsed, PDx wins ties).
4. **Phase 1** (`engine.evaluate`): one Bedrock call returns an independent verdict for the PDx and each ADx.
5. The **decision rule** (`rules.diagnosis_fails`) marks a diagnosis as failing. The episode is a *potential alert* if the PDx or any ADx fails.
6. **Phase 2** (`engine.plan_correction`):
   - every failed ADx is removed;
   - if the PDx failed, the model picks a replacement among the ADx that did not fail;
   - if none fits, the first surviving ADx is used mechanically (and the alert says so);
   - if no ADx survives, the episode is dropped (no alert).
7. **Phase 3** (`engine.finalize`): the grouper regroups the corrected episode; the alert text and record are built.
8. The response returns verdicts, correction, regroup figures, the alert record, provenance, and token usage.

### Decision rule

| Verdict | Fails? |
|---|---|
| `none` (any confidence) | yes |
| `weak` with confidence below the threshold (default 0.5) | yes |
| `weak` with confidence at or above the threshold | no |
| `clear` (any confidence) | no |

Failed ADx are **removed**. A failed PDx is **resequenced** (demoted to the end, kept), because a real condition can still contribute a CC or MCC to the regrouped DRG.

### What the model sees

- **Sent:** descriptions of the diagnoses, ACHI procedures with day offsets from admission, and MBS items
- **Never sent:** admission number, provider, fund, dates, date of birth, age, sex, length of stay, DRG, benefit amount, any claim sequence number
- The prompt is wrapped in `<episode>` tags and the model is told to treat it strictly as data. Codes are validated against a character allowlist before they can reach it

### Why this deployment shape

| Choice | Reason |
|---|---|
| Bring-your-own container | Dependencies are baked in and scanned; no `pip install` at start-up, so it works in a VPC with no internet |
| CPU instances | The model runs in Bedrock, so no GPU is needed |
| Model artifact = reference data | `/opt/ml/model` holds the code tables and a hash manifest; the service refuses to start if any file differs |
| Per-episode requests | Stateless and idempotent (the audit is read-only), so retries are always safe |
| Streaming mode | Reasoning-heavy calls can exceed SageMaker's 60-second limit; streaming allows 8 minutes |
| Tool-use answers, validated | Sonnet 5.5 rejects forced tool choice and offers no structured-output mode on Bedrock, so the service validates every answer itself |

---

## 3. Decisions required before go-live

Each decision lists the options, the safest default, and what you give up. **The code applies the safest path until you choose otherwise.**

### D1. Where may prompts be processed? (data residency)

As of the AWS model card checked on October 15, 2026, Claude Sonnet 5.5 lists `global.`, `us.`, and `eu.` inference profiles only. **No Australian or APAC profile is listed**, so from Sydney the only route for 5.5 is global. Verify with `aws bedrock list-inference-profiles --region <region>` before deciding; AWS adds profiles over time.

| Option | Setting | Data location | Trade-off |
|---|---|---|---|
| **A (safest)** | `AUDIT_BEDROCK_MODEL_ID=au.anthropic.claude-sonnet-5` | Australian geography | Previous-generation model, not 5.5. The request shape here works with it; re-benchmark |
| B | `global.anthropic.claude-sonnet-5-5` and `AUDIT_ALLOW_GLOBAL_INFERENCE=true` | Any commercial AWS Region | The model you asked for, best availability. Residency is not guaranteed |
| C | `us.` or `eu.` profile, `AUDIT_BEDROCK_REGION=<that region>`, `AUDIT_ALLOW_CROSS_REGION_BEDROCK=true` | US/Canada or EU | Data leaves Australia for a defined geography; needs a network path to that Region |

- **Default behaviour:** no model id is set, and a `global.` id without the opt-in refuses to start. `deploy_endpoint` runs the same check, so a bad choice is caught before deployment.
- **Mitigating fact (not a decision for you):** the prompt contains no direct identifiers. Whether that satisfies your privacy obligations is for your governance owner.
- For any geography profile, enumerate the real destination Regions with `aws bedrock get-inference-profile` and do not rely on the prefix alone.

### D2. Network posture

| Option | Result |
|---|---|
| **A (default)** VPC-only endpoint, interface VPC endpoints, no internet route | Traffic to Bedrock stays on the AWS network. Needs endpoints for `bedrock-runtime`, `ecr.api`, `ecr.dkr`, `logs`, and an S3 gateway endpoint, plus a private path to the grouper |
| B Network isolation | **Not possible**: it blocks the calls to Bedrock |
| C NAT or internet egress | Works, but widens the attack surface. Avoid |

`deploy_endpoint` requires a VPC with two or more subnets and a security group. `--allow-no-vpc` exists for experiments only.

### D3. Defence against prompt injection

| Option | Notes |
|---|---|
| **A (default)** Layered controls in code | Strict input validation, delimited data, tool-only output that is validated, HTML-encoded alert text, and no model-driven actions. Deterministic and adds no latency |
| B A plus Bedrock Guardrails | Set `AUDIT_BEDROCK_GUARDRAIL_ID` and `_VERSION`. Adds latency and cost and may block legitimate clinical text. **Not exercised against live Bedrock here**: test with tool use before enabling |
| C None | Not recommended |

### D4. Storing prompts and responses

| Option | Notes |
|---|---|
| **A (default)** Store nothing | No content in logs, SageMaker data capture is off, Bedrock invocation logging is a separate account setting (leave it off) |
| B Bedrock model invocation logging to KMS-encrypted S3 | Only with approval and strict access control; prompts are de-identified but still clinical |
| C SageMaker data capture | Would write full request bodies, including demographics, to S3. Avoid |

---

## 4. Project structure

```
admissions-audit-agent/
├── README.md
├── pyproject.toml                       package metadata and tool settings
├── Makefile                             test, artifact, image, deploy shortcuts
├── requirements/
│   ├── runtime.txt                      image dependencies (bounded ranges; see the lock note inside)
│   ├── grouper.txt                      internal groupy client (private index)
│   ├── dev.txt                          development tooling
│   └── deploy.txt                       what tools/ needs
├── container/
│   ├── Dockerfile                       non-root, no secrets, grouper presence checked at build time
│   ├── serve                            entry point SageMaker runs
│   └── gunicorn.conf.py                 threads, timeouts, request limits
├── vendor/                              put au_grouper.py here (not in version control)
├── src/audit_agent/
│   ├── config.py                        environment settings and the fail-closed residency guards
│   ├── schemas.py                       request contract, strict validation, the Episode model
│   ├── reference_data.py                code tables, manifest verification, safe XML parsing
│   ├── episode_text.py                  de-identified prompt text
│   ├── prompts.py                       system prompts, tool schemas, PROMPT_VERSION
│   ├── rules.py                         the decision rule (pure functions), RULE_VERSION
│   ├── engine.py                        phases 1 to 3
│   ├── alerts.py                        alert text (HTML-encoded) and alert record
│   ├── service.py                       payload in, status and JSON out; provenance; build_service()
│   ├── deadline.py, errors.py, textutil.py, logging_utils.py
│   ├── llm/
│   │   ├── base.py                      provider-neutral types
│   │   ├── bedrock.py                   Converse client for Claude Sonnet 5.5
│   │   └── verdicts.py                  the two model tasks and answer validation
│   ├── grouper/
│   │   ├── corrections.py               paired-array removal and resequencing
│   │   ├── au_drg.py                    adapter for the internal grouper
│   │   ├── drg_version.py, interface.py
│   └── serving/app.py                   /ping and /invocations, streaming, deadlines
├── tools/                               run from a terminal; not shipped in the image
│   ├── deploy/                          package_model_artifact, iam, deploy_endpoint, build scripts, example config
│   ├── client/                          invoke_endpoint, batch_invoke, sagemaker_client
│   ├── evaluation/                      benchmark, sweep_decision_rules, check_reference_coverage, metrics
│   ├── diagnostics/                     smoke_test_bedrock, diagnose_episode, inspect_grouper_payload
│   ├── tabular.py                       extract rows to request payloads
│   └── dev_server.py                    local server with a stub model
└── tests/                               194 tests, no AWS access needed
```

---

## 5. Prerequisites

- [ ] **Bedrock:** Sonnet 5.5 (or your chosen model) enabled for the account; quotas checked for tokens and requests per minute
- [ ] **KMS key** for the endpoint volume and the S3 artifact
- [ ] **S3 bucket** for the artifact (block public access, default encryption)
- [ ] **ECR repository** (the build script creates one with immutable tags and scan-on-push)
- [ ] **VPC:** two private subnets in different Availability Zones, a security group allowing outbound 443 to the VPC endpoints and the grouper, and the endpoints listed under D2
- [ ] **IAM:** an endpoint execution role and a deployer role; generate both policies with `tools.deploy.iam` (section 8, step 4)
- [ ] **Reference data:** `diagnosis_codes.tsv`, `achi_codes.tsv`, `MBS-XML-YYYYMMDD.XML`, and the DRG cost-weight reference parquet
- [ ] **Grouper:** `vendor/au_grouper.py` and access to the `groupy` package
- [ ] **SNS topic** for alarms
- [ ] Decisions D1 to D4 recorded

---

## 6. Configuration reference

All settings are environment variables on the SageMaker model. They are validated together at start-up; every problem is reported in one message. No credentials are accepted: AWS access comes from the execution role.

### Model and Bedrock

| Variable | Default | Notes |
|---|---|---|
| `AUDIT_AWS_REGION` | `AWS_REGION` | Endpoint Region |
| `AUDIT_BEDROCK_MODEL_ID` | required | Inference profile id or ARN, for example `global.anthropic.claude-sonnet-5-5` |
| `AUDIT_BEDROCK_REGION` | endpoint Region | A different value needs `AUDIT_ALLOW_CROSS_REGION_BEDROCK=true` |
| `AUDIT_ALLOW_GLOBAL_INFERENCE` | `false` | Required for any `global.` profile |
| `AUDIT_ALLOW_CROSS_REGION_BEDROCK` | `false` | |
| `AUDIT_LLM_EFFORT` | `high` | `low`, `medium`, `high`, `xhigh`, `max` (depth of adaptive thinking) |
| `AUDIT_LLM_THINKING_MODE` | `adaptive` | `between_tools` skips up-front thinking (effort `high` or below only) |
| `AUDIT_LLM_MAX_OUTPUT_TOKENS` | `16000` | Thinking and the answer share this budget |
| `AUDIT_LLM_MAX_OUTPUT_TOKENS_CEILING` | `64000` | Doubling limit when an answer is cut off |
| `AUDIT_LLM_MAX_OUTPUT_ATTEMPTS` | `3` | Re-asks when an answer is unusable |
| `AUDIT_LLM_MAX_TRANSPORT_RETRIES` | `2` | Throttling, 5xx, timeouts |
| `AUDIT_LLM_CONNECT_TIMEOUT_S` / `AUDIT_LLM_READ_TIMEOUT_S` | `5` / `120` | |
| `AUDIT_LLM_MIN_ATTEMPT_BUDGET_S` | `8` | A call is not started with less time than this left |
| `AUDIT_BEDROCK_PROMPT_CACHING` | `false` | Caches the static prompt; **untested against live Bedrock** |
| `AUDIT_BEDROCK_GUARDRAIL_ID` / `_VERSION` | unset | Set together; see D3 |
| `AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON` | unset | Escape hatch merged into `additionalModelRequestFields` |

### Behaviour

| Variable | Default | Notes |
|---|---|---|
| `AUDIT_LOW_CONFIDENCE_WEAK_THRESHOLD` | `0.5` | Provisional; see section 12 |
| `AUDIT_MBS_ARRAY_FIELD` | `medical_claims_array` | Or `hospital_claims_array`. Which array holds MBS items is still unconfirmed |
| `AUDIT_GROUPER_BACKEND` | `au_drg` | `disabled` returns verdicts and the proposed correction only |
| `AUDIT_GROUPER_STACK_NAME` | `hambs-shed` | Passed to the grouper client |
| `AUDIT_GROUPER_SERIALIZE_CALLS` | `true` | One grouper call at a time per worker (thread safety of the internal client is unknown) |
| `AUDIT_RULE_ID` | `62248` | `alert_instance_id` in alert records. Carried over from the prototype; confirm per environment |

### Serving and safety limits

| Variable | Default | Notes |
|---|---|---|
| `AUDIT_REQUEST_DEADLINE_S` | `50` | Regular calls (range 5 to 58) |
| `AUDIT_STREAM_DEADLINE_S` | `420` | Streaming calls (range 30 to 470) |
| `AUDIT_STREAM_HEARTBEAT_S` | `10` | |
| `AUDIT_MAX_REQUEST_BYTES` | `1048576` | |
| `AUDIT_MAX_CONCURRENT_REQUESTS` | `16` | Per worker process |
| `AUDIT_MAX_DIAGNOSES` / `_PROCEDURES` / `_MBS_ITEMS` | `100` / `200` / `300` | Bounds prompt size and cost |
| `AUDIT_MODEL_DIR` | `/opt/ml/model` | |
| `AUDIT_REQUIRE_MANIFEST` | `true` | Set `false` for local use only |
| `AUDIT_LOG_LEVEL` | `INFO` | |
| `AUDIT_LOG_RAW_IDENTIFIERS` | `false` | Logs `admission_no` when true |
| `AUDIT_WORKERS` / `AUDIT_THREADS` | `2` / `8` | Gunicorn. Concurrency per instance is the product (16) |

---

## 7. Develop in SageMaker Code Editor

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
pip install -e .
make test                      # 194 tests, no AWS access needed
```

- **Credentials:** the Code Editor session uses the Studio execution role automatically. For a realistic check, use credentials as close to the endpoint role as you can.
- **Check Bedrock access and the request shape** (also the first thing to run after any model or SDK change):

```bash
export AUDIT_AWS_REGION=ap-southeast-2
export AUDIT_BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-5-5
export AUDIT_ALLOW_GLOBAL_INFERENCE=true        # only after D1 is decided
make smoke-bedrock
```

- **Try the HTTP contract with no AWS at all** (stub model, tiny built-in reference data):

```bash
python -m tools.dev_server --fail-code N179 &
curl -s localhost:8080/invocations -H 'content-type: application/json' -d @tools/deploy/smoke_payload.json
```

- **Check how much of your evidence resolves to a description** (below 95 percent, the model is judging with a partial picture):

```bash
python -m tools.evaluation.check_reference_coverage --input episodes.parquet --reference-dir ./reference
```

---

## 8. Build and deploy

### Step 1. Package the reference data

```bash
python -m tools.deploy.package_model_artifact \
  --reference-dir ./reference --drg-reference-dir ./drg_reference \
  --output artifacts/model.tar.gz \
  --s3-uri s3://<bucket>/admissions-audit-agent/<version>/model.tar.gz --kms-key-id <key>
```

- Loads and validates the files with the service's own loaders, so a wrong file fails here.
- Writes `manifest.json` (SHA-256 of every file). The endpoint verifies it at start-up and refuses to run on a mismatch.
- Uploads with SSE-KMS. The tarball is deterministic and contains no symlinks.

### Step 2. Put the grouper in place

Copy the internal `au_grouper.py` into `vendor/` (see `vendor/README.md` for the interface the adapter expects). If you have no grouper yet, build with `REQUIRE_GROUPER=false` and deploy with `AUDIT_GROUPER_BACKEND=disabled`.

### Step 3. Build the image

```bash
AWS_REGION=ap-southeast-2 ECR_REPOSITORY=admissions-audit-agent \
  PIP_CONF_FILE=./pip.conf bash tools/deploy/build_and_push.sh       # needs Docker with BuildKit
```

- Creates the repository with immutable tags and scan-on-push, builds, pushes, and prints the image URI and digest.
- `PIP_CONF_FILE` (private package index credentials) is passed as a BuildKit secret and is never stored in a layer.
- **No Docker in your Studio domain?** Use `tools/deploy/buildspec.yml` in AWS CodeBuild (privileged mode).
- Review the ECR scan findings before deploying. For reproducible builds, generate a hash-pinned lock (`pip-compile --generate-hashes`) and pin the base image by digest (`BASE_IMAGE=python:3.12-slim-bookworm@sha256:...`).

### Step 4. Create the roles

```bash
python -m tools.deploy.iam trust > trust.json
python -m tools.deploy.iam execution-role --region ap-southeast-2 --account <acct> \
  --endpoint-name admissions-audit-agent-prod --ecr-repository admissions-audit-agent \
  --artifact-s3-uri s3://<bucket>/admissions-audit-agent/<version>/model.tar.gz \
  --kms-key-arn <key-arn> --model-id global.anthropic.claude-sonnet-5-5 > execution-policy.json
python -m tools.deploy.iam deployer ... > deployer-policy.json
```

- For a `global.` profile the policy has the three statements AWS documents (the profile, the foundation model in the calling Region, and the Region-less model ARN). For a geography profile, pass each destination Region with `--destination-region`.
- Validate with a real invocation. If SCPs restrict Regions, confirm they allow the destinations (including `aws:RequestedRegion = unspecified` for global profiles).
- Account administrators may also need Marketplace permissions the first time a third-party model is used.

### Step 5. Configure and deploy

```bash
cp tools/deploy/config/example.yaml tools/deploy/config/prod.yaml      # replace every <placeholder>
make deploy-dry-run                                                     # prints every API call, changes nothing
make deploy                                                             # creates or updates, then runs a smoke test
```

What the deployment does:

| Step | Detail |
|---|---|
| Validation | Placeholders, `latest` tags, missing VPC or KMS key, single instance, secret-like environment keys, and the **container's own start-up validation** (so the residency guard fires here) |
| Model and config | New immutable names per release; network isolation off (required for Bedrock); data capture off |
| Alarms | 5xx, p99 model latency, deadline exceeded, Bedrock throttling |
| Create | First deployment creates the endpoint |
| Update | Blue/green canary, with automatic rollback on the 5xx and latency alarms; old fleet kept for in-flight streaming requests |
| Autoscaling | Target tracking on invocations per instance (default 20 per minute, 2 to 6 instances) |
| Metrics | Log metric filters for episodes, alerts, failures, deadline hits, throttling, and token counts |

### Step 6. Validate before alerts reach reviewers

1. Run the benchmark on a labelled slice ([section 12](#12-evaluating-accuracy-and-the-decision-threshold)).
2. Choose the effort level and threshold, then confirm on a fresh slice.
3. Run a small `batch_invoke` in `verdicts_only` mode, then `full`, and have coders review a sample of alerts.

### Rollback

- **Automatic:** a canary that breaches the 5xx or latency alarm is rolled back by SageMaker.
- **Manual:** redeploy the previous image tag (immutable) and artifact with the same command; SageMaker shifts traffic blue/green.

---

## 9. Inference

### Request

`POST /invocations`, `Content-Type: application/json`, body up to 1 MiB.

```json
{
  "request_id": "batch-17-000123",
  "episode": {
    "admission_no": 555123,
    "diagnosis_code_array": ["J189", "I509", "N179", "E119"],
    "diagnosis_poa_array": [1, 1, 0, 1],
    "condition_onset_array": ["ADMIT", "ADMIT", "DURING", "ADMIT"],
    "dx_code_idx_array": [0, 1, 2, 3],
    "procedure_code_array": ["3237500"],
    "procedure_date_array": ["2026-08-02"],
    "medical_claims_array": ["3"],
    "dte_admitted": "2026-08-01",
    "dte_separated": "2026-08-06",
    "drg": "E62B", "drg_vers": "100", "hos_benefit": 9200.0, "fund_id": 101, "los": 5,
    "hospital_claims_seqno_array": [1001, 1002],
    "hospital_claims_service_class_code_array": ["DRG_CASE_PAYMENT", "OTHER"]
  },
  "options": {"mode": "full"}
}
```

| Field | Rules |
|---|---|
| `request_id` | Optional, `[A-Za-z0-9._-]{1,64}`; generated if absent; returned in the response and the `X-Request-Id` header |
| `episode.admission_no` | Required positive integer |
| `episode.diagnosis_code_array` | Required, non-empty. **Position 0 is the PDx**, the rest are ADx |
| Code arrays | Letters, digits, `.` and `-` only, up to 16 characters. Anything else is a 400 (this is what keeps instructions out of the prompt) |
| Paired arrays | `diagnosis_poa_array`, `condition_onset_array`, `dx_code_idx_array` must match the diagnosis array in length. POA entries may be scalars or `{"code", "poa"}` objects |
| Dates | ISO-8601. The wall-clock date is kept; time zones are not converted |
| Grouper fields | `drg`, `hos_benefit`, `dte_admitted`, `dte_separated` are required in `full` mode. Other recognised fields (`fund_id`, `los`, `member_*`, and so on) are passed to the grouper only |
| Unknown episode fields | Dropped and counted in `warnings` (the envelope itself is strict) |

**Options**

| Option | Default | Meaning |
|---|---|---|
| `mode` | `full` | `verdicts_only` stops after phase 1 and needs no grouper fields |
| `stream` | `false` | Same as `Accept: application/x-ndjson` |
| `deadline_s` | per mode | Shorten the time budget (5 seconds or more; it never exceeds the limit for the mode) |
| `include_episode_text` | `false` | Return the exact text shown to the model |
| `mbs_array_field` | service setting | `medical_claims_array` or `hospital_claims_array` |

### Response

```json
{
  "request_id": "batch-17-000123",
  "status": "completed",
  "admission_no": 555123,
  "mode": "full",
  "potential_alert": true,
  "flagged": true,
  "disposition": "alert",
  "verdicts": [
    {"code": "J189", "role": "PDx", "support_level": "clear", "supported": true, "failed": false,
     "flag_reason": null, "confidence": 0.92, "reasoning": "..."},
    {"code": "N179", "role": "ADx", "support_level": "none", "supported": false, "failed": true,
     "flag_reason": "none", "confidence": 0.85, "reasoning": "..."}
  ],
  "failed_adx_codes": ["N179"],
  "correction": {"removed_adx_codes": ["N179"], "pdx_resequence": null,
                 "corrected_diagnosis_fields": {"diagnosis_code_array": ["J189", "I509", "E119"]}},
  "regroup": {"original_drg": "E62B", "regrouped_drg": "E62A", "drg_changed": true,
              "cost_weight_drop": 0.15, "possible_recovery": 1971.43},
  "alert": {"claim_seqno": "1001", "alert_info": "...", "recovery_amount": 1971, "alert_instance_id": 62248},
  "warnings": [],
  "provenance": {"model_id": "global.anthropic.claude-sonnet-5-5", "reasoning_effort": "high",
                 "prompt_version": "pdx-adx-support/2026-10-15.1", "rule_version": "any-verdict:none-or-lowconf-weak/1",
                 "low_confidence_weak_threshold": 0.5, "reference_data_manifest_sha256": "..."},
  "usage": {"llm_calls": 1, "input_tokens": 1500, "output_tokens": 2300, "elapsed_ms": 18000}
}
```

- `alert` uses the schema downstream consumers already read: `claim_seqno`, `alert_info`, `recovery_amount`, `alert_instance_id`.
- `reasoning` and `alert_info` contain model-written text. `alert_info` is HTML-encoded (it carries one intentional `<br>`); encode `reasoning` before rendering it as HTML.
- `provenance` records exactly what produced the result, so an alert can be reproduced and audited.

**`disposition` values**

| Value | Meaning |
|---|---|
| `no_alert` | Every diagnosis passed |
| `alert` | Correction applied, regrouped, alert record produced |
| `dropped_no_surviving_adx` | The PDx failed and no ADx survives to promote; no alert (a warning says so) |
| `correction_proposed` | Grouper disabled: correction returned, no regroup figures or alert |
| `verdicts_only` | `mode=verdicts_only` |

`potential_alert` is the phase-1 outcome. `flagged` is true only when an alert record was produced.

### Errors and retries

HTTP status says who acts: **4xx** the caller, **5xx** the platform or a dependency (retry later), **200** the request was processed (read `status`).

| Code | HTTP | Retry? | Meaning |
|---|---|---|---|
| `invalid_request` | 400 | no | Details list each field path (values are never echoed) |
| `unsupported_media_type` / `payload_too_large` | 415 / 413 | no | |
| `overloaded` | 503 | yes | Per-process capacity reached |
| `deadline_exceeded` | 504 | yes | Use streaming or a lower effort |
| `llm_throttled` / `llm_unavailable` | 503 | yes | Bedrock quota or outage |
| `llm_access_denied` / `llm_request_rejected` | 500 | no | Model access, IAM, or a rejected parameter |
| `llm_output_invalid` / `llm_refused` | 200 | no | Episode-level failure after the allowed attempts. For `llm_output_invalid`, `error.details.last_reason` says why (for example `llm_output_truncated`, `missing_code`, `bad_support_level`) |
| `grouper_unavailable` | 503 | yes | |
| `grouper_input_invalid` / `correction_not_applicable` | 200 | no | Needs a human coder (for example an ambiguous repeated code) |
| `internal_error` | 500 | no | Details are in the logs only |

A failure after phase 1 still returns the verdicts under `partial`, so paid-for work is not lost. Failure bodies always carry `status: "failed"` and `error.retryable`. The audit is read-only, so retrying any request is safe.

### Regular and streaming calls

```python
import json, boto3
from botocore.config import Config

rt = boto3.client("sagemaker-runtime", region_name="ap-southeast-2",
                  config=Config(read_timeout=70, retries={"max_attempts": 0}))
resp = rt.invoke_endpoint(EndpointName="admissions-audit-agent-prod", ContentType="application/json",
                          Accept="application/json", Body=json.dumps(payload))
result = json.loads(resp["Body"].read())
```

- **Regular:** the platform cuts calls at 60 seconds. The service answers with `deadline_exceeded` at about 50 seconds (`AUDIT_REQUEST_DEADLINE_S`) so the failure is structured rather than a platform timeout.
- **Streaming (up to 8 minutes):** call `invoke_endpoint_with_response_stream` with `Accept: application/x-ndjson`. Events are `accepted`, periodic `heartbeat`, then one `result` carrying `http_status` and the same body a regular call returns. Chunk boundaries do not align with lines; `tools/client/sagemaker_client.py` shows the buffering.

### Command-line clients

```bash
python -m tools.client.invoke_endpoint --endpoint-name NAME --region ap-southeast-2 --payload episode.json --stream
python -m tools.client.batch_invoke --input episodes.parquet --endpoint-name NAME --region ap-southeast-2 \
       --out-dir output/batch --mode full --workers 4 --stream
```

- `batch_invoke` retries retryable failures, keeps a resumable checkpoint (`results.jsonl`), writes failures to `failures.jsonl`, and rebuilds the alert file (`<concept>_json.txt`, the schema above) from the checkpoint each run, so it never holds duplicates.
- Output files are created readable by the owner only.
- For very large backfills, size `--workers` to your Bedrock quota. The same image can also back an asynchronous endpoint (not provided here).

---

## 10. Operations

### Logs and metrics

One JSON line per event on stdout, forwarded to `/aws/sagemaker/Endpoints/<endpoint>`. A field allowlist makes it impossible to log codes, reasoning, prompts, or model output; `admission_no` is logged only with `AUDIT_LOG_RAW_IDENTIFIERS=true`.

| Event | Use |
|---|---|
| `episode_completed` / `episode_failed` | One per request: status, disposition, tokens, elapsed time, error code |
| `llm_call` / `llm_call_failed` | Per model call: stop reason, tokens, latency, AWS error code and request id |
| `llm_answer_rejected` | The model answered but validation rejected it (reason category only) |
| `service_started` | Operating parameters (model id, routing scope, effort, threshold) |

CloudWatch namespace `AdmissionsAuditAgent/<endpoint>` (from log metric filters): `EpisodesCompleted`, `EpisodesFailed`, `AlertsFlagged`, `DeadlineExceeded`, `LlmThrottled`, `LlmInputTokens`, `LlmOutputTokens`. Standard `AWS/SageMaker` endpoint metrics apply as usual.

Watch the **alert rate** (`AlertsFlagged / EpisodesCompleted`). A sudden shift after a model, prompt, or threshold change is the first sign of drift.

### Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Endpoint will not reach `InService` | Start-up validation failed | Read the endpoint's CloudWatch log: configuration errors list every problem. Common: residency opt-in, missing or mismatched `manifest.json`, `au_grouper` not in the image |
| `llm_access_denied` | Model not enabled, IAM statements, or SCP | Run `make smoke-bedrock` with the endpoint's role; regenerate the policy with `tools.deploy.iam`; check SCPs |
| `llm_request_rejected` | Bedrock refused a parameter | Read the message. Sonnet 5.5 rejects `temperature`/`top_p`/`top_k`, forced tool choice, and disabled or budgeted thinking. Remove `AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON` and retry |
| `deadline_exceeded` often | Effort too high for 60 seconds | Use streaming, lower `AUDIT_LLM_EFFORT`, or try `between_tools` |
| `llm_throttled` | Bedrock quota | Request an increase; lower `--workers` or the autoscaling target |
| `llm_output_invalid` | Unusable or cut-off answers (see `error.details.last_reason`) | `python -m tools.diagnostics.diagnose_episode`; if the reason is `llm_output_truncated`, raise `AUDIT_LLM_MAX_OUTPUT_TOKENS` or lower the effort |
| `grouper_input_invalid` | A grouper input field is missing or malformed | `python -m tools.diagnostics.inspect_grouper_payload` (values hidden by default) |
| `correction_not_applicable` | Repeated PDx or principal code, or every diagnosis failed | Route to a human coder |
| Many 5xx after a release | Bad release | SageMaker rolls the canary back automatically; otherwise redeploy the previous tag |

### Capacity

- Concurrency per instance is `AUDIT_WORKERS` x `AUDIT_THREADS` (16 by default). Throughput is roughly concurrency divided by mean latency: at 30 seconds per episode, about 32 episodes per minute per instance.
- Bedrock tokens-per-minute and requests-per-minute quotas usually bind before instance capacity. Adaptive thinking tokens count as output tokens.
- Lower `target_invocations_per_instance` if requests are slow; use two or more instances for zone resilience.

### Changing the model, prompt, or threshold

- Any text or schema change in `prompts.py` needs a new `PROMPT_VERSION`; any change to the rule needs a new `RULE_VERSION`. Both appear in every response.
- Benchmark first ([section 12](#12-evaluating-accuracy-and-the-decision-threshold)), then deploy through the normal blue/green path.
- Model deprecations: AWS announces lifecycle dates per model. Pin the model id in the configuration and plan upgrades; re-run the smoke test and benchmark each time.

---

## 11. Security controls

| Area | Control |
|---|---|
| Authentication | Endpoints accept only SigV4/IAM callers; grant `sagemaker:InvokeEndpoint` on the one endpoint ARN |
| Network | VPC-only endpoint, private Bedrock access through a VPC endpoint, no inbound ports beyond SageMaker's |
| Secrets | None in code, config, images, logs, or tests. Credentials come from the execution role; the private-index credential is a BuildKit secret; the deploy tool rejects secret-like environment keys |
| Input | Strict envelope, allowlisted episode fields, character allowlist on every code, size and range limits, strict JSON (no NaN, duplicate keys, or deep nesting), error messages that never echo values |
| Prompt injection | Only validated codes and trusted descriptions reach the prompt; descriptions are sanitised; episode text is delimited and declared as data; the answer is a validated tool call; nothing the model says triggers an action |
| Output | Model text is control-character-stripped and length-capped; `alert_info` is HTML-encoded |
| Data minimisation | The model receives no identifiers or financial data; the grouper receives only allowlisted fields |
| Integrity | Reference data verified against a SHA-256 manifest at start-up; MBS XML parsed with `defusedxml` (no entity expansion) |
| Encryption | Endpoint volume and S3 artifact under KMS; TLS to Bedrock and SageMaker |
| Logging | Content-free by construction; identifiers off by default |
| Least privilege | Policies generated from parameters; the execution role cannot call anything except the one Bedrock profile, its image, its artifact, and its logs |
| Supply chain | Immutable image tags, scan-on-push, non-root container, bounded dependency ranges (pin with hashes in CI) |
| Availability | Per-request deadlines, bounded retries, capacity limits, canary deployment with automatic rollback |

---

## 12. Evaluating accuracy and the decision threshold

Confidence calibration differs between models, so the 0.5 threshold needs re-validating whenever the model, effort, or prompt changes.

```bash
# 1. Benchmark: the service's own code against human review outcomes (column "Recommended" = human found it unsupported)
python -m tools.evaluation.benchmark --input labelled.parquet --reference-dir ./reference --out-dir output/benchmark
#    or through a deployed endpoint:
python -m tools.evaluation.benchmark --input labelled.parquet --endpoint-name NAME --region ap-southeast-2 --stream

# 2. Sweep per-verdict rules over the same results (no new model calls)
python -m tools.evaluation.sweep_decision_rules output/benchmark/benchmark_results.csv --threshold 0.5
```

- The benchmark writes `benchmark_results.csv` (one row per verdict), `benchmark_episodes.csv`, `benchmark_metrics.json`, and `benchmark_report.md` (confusion matrix, precision, recall, F1, MCC, token use, latency).
- The sweep marks the deployed rule as `CURRENT` and compares stricter and looser alternatives.
- **Judge on a fresh slice.** A threshold chosen on the data it is scored on looks better than it is.
- If every rule scores near "none only", the signal itself is the limit: change the evidence, prompt, effort, or model, not the line.
- Compare effort levels (`medium` against `high`) on both accuracy and p95 latency before choosing one.
- Agreement with human review measures consistency, not clinical truth; reviewers may have had evidence the model never sees.

---

## 13. Testing

```bash
make test        # python -m unittest discover -s tests -t . -v   (pytest also works)
```

194 tests cover configuration guards, request validation (including injection-shaped input), reference-data integrity, prompt formatting, the Bedrock request and response handling, retries and deadlines, answer validation, correction mechanics, alert text, the grouper adapter, the HTTP contract (streaming, timeouts, capacity), start-up wiring, deployment planning, IAM generation, packaging, and the benchmark-to-sweep pipeline. The Bedrock SDK and the grouper are replaced by fakes; one test drives the real HTTP app, service, engine, and Bedrock client against a fake Converse runtime end to end.

---

## 14. Changes from the previous implementation

| Before | Now |
|---|---|
| gpt-oss-20b on a GPU box (llama.cpp or Transformers) | Claude Sonnet 5.5 on Bedrock (Converse API); CPU-only endpoint |
| Batch scripts, lock file, checkpoints on EFS | Stateless per-episode endpoint; `batch_invoke` keeps the checkpoint |
| `paths.py`, `.env` loader, hard-coded EFS and S3 locations | Environment-only settings validated at start-up; reference data in the model artifact |
| `temperature=0.1`, a `Reasoning:` prompt line, optional grammar constraint | Removed (Sonnet 5.5 rejects sampling parameters); adaptive thinking with `effort` |
| JSON parsed out of free text with a brace scanner | Tool-call answers, strictly validated, re-asked with a specific reason |
| Verdict fields ordered label first | `reasoning` precedes `support_level` in the schema |
| Prompt asked for a JSON reply | The none/weak/clear rubric is byte-identical (checked by diff); new output and data-handling paragraphs. One sentence in the replacement-search prompt was reworded: it said the candidates were "the full list" of ADx, but failed ADx are no longer offered |
| Alert text named internal files and embedded model text raw | File names removed; model text HTML-encoded |
| Day offsets from date-time differences | Calendar-day offsets |
| Paired-array mismatch found in phase 3 | Rejected at request time |
| Ambiguous resequence raised mid-batch | Reported as `correction_not_applicable`, verdicts preserved |
| `drg_cost_weight.py` (scoring and joins) | Only the DRG version mapping is kept; the rest was not used by the audit |
| `model_data_prep.py` (S3 and EFS pulls) | Upstream data engineering, out of scope; `tools/tabular.py` converts extracts to requests |
| `benchmark_report.py` (Word report) | Markdown, JSON, and CSV outputs; render a document from them if stakeholders need one |
| `pilot_llm_only_smoketest`, `diagnose_stuck_episode`, `check_evidence_resolution`, `inspect_grouper_payload`, `sweep_decision_rules` | `verdicts_only` mode and `dev_server`, `diagnose_episode`, `check_reference_coverage`, `inspect_grouper_payload` (values hidden by default), `sweep_decision_rules` (now fed by the benchmark's per-verdict CSV) |
| Tests that called functions which no longer existed | 194 new tests |

---

## 15. Verification status and known limits

**Verified here**
- All 194 tests pass locally (Python 3.12, standard-library `unittest`, Flask 3.1, pandas 3.0).
- The request shape, retry policy, and answer handling are exercised against fakes written from the AWS and Anthropic documentation as of October 15, 2026.

**Not verified here (no AWS access, no Docker, no internal modules)**
- Any call to Amazon Bedrock, SageMaker, ECR, CloudWatch, KMS, or IAM, and the generated IAM policies against a real account
- The Docker build, gunicorn start-up, and `serve`
- The adapter against the real `au_grouper` and `groupy` modules (it follows the prototype's calls exactly, tested with a fake)
- Latency at effort `high`, throughput under load, and accuracy against human review with Claude
- Prompt caching and Guardrails (both off by default)

**Known limits**
- The 0.5 threshold is provisional (section 12).
- Which array holds MBS items (`medical_claims_array` or `hospital_claims_array`) and the meaning of `dx_code_idx_array` are still unconfirmed, as in the prototype.
- A repeated PDx or new-principal code makes a resequence ambiguous; it is refused and routed to a human coder.
- Model output is not bit-for-bit reproducible (sampling parameters cannot be set on this model); `provenance` and the returned verdicts are the record.
- Reference-data parsing is tested with fixtures shaped like the prototype's loaders, not your production files; the packaging step validates the real ones.
- Dependency ranges are bounded, not hash-pinned; generate a lock in CI.
