# DRG audit anomaly detector

Flags inpatient admissions whose billed AR-DRG disagrees with what a transformer model infers from the admission's clinical codes and resource use. Built for AWS SageMaker Code Editor and served from a managed SageMaker real-time endpoint.

**Contents**

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Getting started](#4-getting-started)
5. [Data contract](#5-data-contract)
6. [Training](#6-training)
7. [Packaging](#7-packaging)
8. [Deployment](#8-deployment)
9. [Inference API](#9-inference-api)
10. [Batch scoring and alert generation](#10-batch-scoring-and-alert-generation)
11. [Security](#11-security)
12. [Operations](#12-operations)
13. [Testing and pre-production checklist](#13-testing-and-pre-production-checklist)
14. [Migration from the prototype](#14-migration-from-the-prototype)
15. [Configuration reference](#15-configuration-reference)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Overview

- **Goal**: find likely upcoding or mis-grouping by comparing the assigned DRG with a prediction made *without* the principal diagnosis.
- **Approach**: a contrastive-pretrained transformer encoder feeds a cascaded classifier (MDC, then base DRG, then complexity split).
- **Output per admission**: predicted DRG parts, confidences, three discrepancy flags, an anomaly flag, and an audit priority score (`benefit x (1 - p(assigned split))`).
- **Serving**: JSON in, JSON out, up to 256 admissions per call, CPU or GPU.
- **What this repository contains**
  - A tested Python package (`drg_audit`) with one shared feature pipeline for training and serving.
  - Training for both stages, runnable in Code Editor or as a SageMaker Training Job.
  - A hardened SageMaker endpoint handler, a package builder, and a boto3 deployer with autoscaling, alarms, and blue/green updates.
  - Alert generation (replaces the prototype's `main.py`) and a converter for existing prototype weights.

---

## 2. Architecture

### 2.1 System view

```
 TRAIN  (Code Editor GPU space, or a SageMaker Training Job)
 curated Parquet --> stage 1: contrastive pre-training --> stage 2: joint fine-tuning --> serving bundle
 (data contract)     encoder checkpoint                    encoder + cascaded classifier   safetensors + JSON + SHA-256 manifest
                                                                                                  |
                                              sagemaker/build_model_package.py  (+ serving code, + offline wheelhouse)
                                                                                                  v
                                                                         dist/model.tar.gz + package_meta.json (digest)
                                                                                                  |
                              sagemaker/deploy_endpoint.py  (S3 upload with SSE-KMS, model, endpoint config, endpoint,
                                                             autoscaling, CloudWatch alarms, blue/green updates)
                                                                                                  v
 clients --SigV4--> SageMaker real-time endpoint (private VPC, network isolation)
 (batch scorer,        inference.py --> InferenceService --> AuditEngine
  applications)        validate request --> encode --> model --> decision rules --> JSON response
                                                                                                  |
                       scored admissions --> alert generation (regroup without PDx, recovery amount) --> alert JSON lines
```

### 2.2 Model

```
 admission codes --> [CLS] [MASK] secondary-dx... procedures... claims... [SEP]     principal diagnosis is masked (no leakage)
                 --> encoder: 6-layer RoBERTa, hidden 256, 8 heads, relative_key_query attention, trained from scratch
                 --> CLS state --> LayerNorm --> Linear-GELU-LayerNorm projector --> embedding (256)

 embedding (256) + meta (4) + DRG-version embedding (16) --> shared MLP (256 -> 128; LayerNorm, GELU, dropout)
      |--> MDC head                          --> p(MDC)
      |--> base-DRG head  (shared + p(MDC))   --> p(base)
      '--> split head     (shared + p(base))  --> p(split)
```

- **Meta features**: length-of-stay bucket, `log1p(theatre minutes)`, `log1p(ventilation hours)`, age / 100.
- **Stage 1 loss**: in-batch contrastive (InfoNCE, temperature 0.30) between a clean view and a randomly masked view (15% of tokens) of each admission.
- **Stage 2 loss**: `0.5 x contrastive + 1.0 x CE(MDC) + 1.2 x CE(base) + 0.8 x CE(split)` with square-root inverse-frequency class weights.
- **Model selection**: best mean macro-F1 over the three heads, measured on clean validation views; early stopping patience 6 (stage 2) and 10 (stage 1).
- **Splits**: hash of `admission_no`, identical in both stages, so pre-training validation data never leaks into fine-tuning training data.

### 2.3 Decision rules

An admission is a **billing anomaly** when the model confidently disagrees with the assigned DRG (`threshold` defaults to 0.85, comparison is strictly greater-than):

| Flag | Condition |
| --- | --- |
| `mdc_discrepancy_flag` | predicted MDC differs from assigned MDC and p(predicted MDC) > threshold |
| `base_discrepancy_flag` | predicted base DRG differs from assigned and p(predicted base) > threshold |
| `split_inflation_flag` | predicted split ranks below the assigned split (A > B > C > D) and p(predicted split) > threshold |
| `is_billing_anomaly` | any of the three flags |

`audit_priority_score = drg_benefit x (1 - p(assigned split))` for anomalies, otherwise 0.

### 2.4 Endpoint internals

| Layer | Module | Responsibility |
| --- | --- | --- |
| SageMaker glue | `sagemaker/code/inference.py` | `model_fn` / `input_fn` / `predict_fn` / `output_fn`; maps errors to 4xx or a generic 500 |
| Contract | `drg_audit/serving/contract.py` | strict JSON parsing, bounds, per-record rejection |
| Service | `drg_audit/serving/service.py` | orchestration, response assembly, PHI-free logging |
| Engine | `drg_audit/inference/engine.py` | batching, device and precision handling, deterministic inference, load-time self-check |
| Decision | `drg_audit/inference/decision.py` | flags and priority score (NumPy only, unit-tested) |
| Artifacts | `drg_audit/artifacts.py` | safetensors weights, SHA-256 manifest verification, fail-closed loading |

### 2.5 Key design decisions

| Decision | Reason |
| --- | --- |
| One feature module for training and serving | The prototype tokenized differently in training and inference (see section 14) |
| Tokenization spec stored in the bundle | Serving always matches how the weights were trained; legacy weights keep their legacy spec |
| safetensors + JSON, no pickle | Loading a model can never execute code |
| Hash-verified bundle, fail-closed load | Detects tampering and truncated downloads before any traffic is served |
| CPU-first serving | The model is small; CPU endpoints are cheaper and avoid GPU-only code paths |
| Per-record `rejected` / `skipped` results | One bad admission cannot fail a 256-record batch |
| boto3 deployer instead of the SageMaker SDK | Immune to SDK v2/v3 API differences; every security parameter is explicit and reviewable |

---

## 3. Project structure

```
drg-audit-anomaly-detector/
|-- README.md
|-- Makefile                          common tasks (make help)
|-- pyproject.toml                    package metadata, optional dependency groups, lint settings
|-- environment.yml                   Code Editor / conda development environment
|-- requirements/
|   |-- inference.in                  packages added on top of the AWS PyTorch inference container
|   |-- train_job.in                  training-container additions
|   |-- train.in                      local GPU training (adds torch)
|   |-- ops.in                        boto3 and PyYAML for the operations scripts
|   '-- dev.in                        everything, plus pytest, ruff, pip-tools
|-- configs/
|   |-- pretrain.yaml                 stage 1 hyper-parameters (equal to the prototype's)
|   |-- finetune.yaml                 stage 2 hyper-parameters and the serving defaults baked into the bundle
|   |-- alerts.yaml                   alert generation rules and paths
|   |-- deploy.example.yaml           endpoint deployment template
|   '-- train_job.example.yaml        training-job template
|-- examples/
|   |-- request.json                  valid example request (synthetic)
|   '-- response.json                 example response
|-- iam/
|   |-- endpoint_execution_role_trust.json
|   |-- endpoint_execution_role_policy.json    least privilege for the endpoint role
|   '-- deployer_policy.json                   least privilege for whoever runs the deploy script
|-- src/drg_audit/
|   |-- features.py                   shared tokenization and meta features (NumPy only)
|   |-- schemas.py                    records, results, label maps, tokenization spec
|   |-- settings.py                   request limits, runtime settings, DRG_AUDIT_* overrides
|   |-- artifacts.py                  bundle read / write / verify
|   |-- archive.py                    safe tar extraction for artifacts from other jobs
|   |-- config.py, _mapping.py        strict typed YAML configuration
|   |-- logging_utils.py, errors.py   JSON logging, exception hierarchy
|   |-- synthetic.py                  PHI-free synthetic data for smoke tests
|   |-- client.py                     endpoint client (batching, retries, bounded concurrency)
|   |-- cli.py                        `drg-audit` command
|   |-- models/                       encoder, cascaded classifier, joint model
|   |-- inference/                    decision rules and scoring engine
|   |-- serving/                      request contract and service layer
|   |-- data/                         dataset IO and validation, vocabulary, torch datasets
|   |-- training/                     stage 1, stage 2, losses, checkpoints
|   '-- postprocess/alerts.py         alert generation
|-- sagemaker/
|   |-- code/inference.py             endpoint entry point (packaged into model.tar.gz)
|   |-- build_model_package.py        bundle -> model.tar.gz (+ digest, + metadata)
|   |-- deploy_endpoint.py            deploy | status | rollback | delete
|   |-- launch_training_job.py        SageMaker Training Job launcher
|   |-- train_entry.py                training-container entry point
|   '-- _awscfg.py, _packaging.py     shared helpers
|-- scripts/
|   |-- convert_legacy_checkpoint.py  prototype .pth -> verified bundle
|   |-- export_dataset.py             prototype data_prep -> Parquet data contract
|   '-- make_synthetic_data.py
'-- tests/                            unittest suites (run with pytest or unittest)
```

Only these modules are packaged into the endpoint: `features`, `schemas`, `settings`, `artifacts`, `errors`, `logging_utils`, `_mapping`, and the `models`, `inference`, and `serving` packages. Training, data, alerts, client, and CLI code never reach the container, and the build fails if a packaged file imports anything missing.

---

## 4. Getting started

### 4.1 Environment (SageMaker Code Editor)

- Use a **GPU instance type** (for example `ml.g5.xlarge`) only for real training. Everything else runs on a CPU space.
- From the repository root:

```bash
conda env create -f environment.yml && conda activate drg-audit
make test        # unit tests; torch tests run automatically because torch is installed
make smoke       # tiny end-to-end run on synthetic data, CPU, a few minutes
```

- `make smoke` generates synthetic admissions, trains a tiny model, verifies the bundle, and scores locally. Use it to confirm the toolchain before touching real data.
- Without conda: `python -m pip install -r requirements/dev.in && python -m pip install -e .`.

### 4.2 Version pins (important)

| Component | Pin | Where |
| --- | --- | --- |
| Endpoint and training container | AWS PyTorch 2.6.0, Python 3.12 | `container.framework_version` / `py_version` in the deploy config |
| `torch` (local training only) | 2.6.0 | `requirements/train.in` |
| `transformers`, `safetensors` | 4.46.3, 0.4.5 | `requirements/inference.in` |

- Train and serve with the **same** `transformers` and `torch` versions. Parameter names are stable across versions, but numerics are not guaranteed to be identical.
- AWS documents Python 3.12 support in the PyTorch 2.6 inference and training containers. Confirm that your region offers the container and instance type you choose.
- To change a pin, change it in `requirements/*.in`, `pyproject.toml`, and the deploy config together, then rebuild, rerun `make test`, and redeploy to a non-production endpoint first.
- `make lock` produces hash-pinned lock files with `pip-compile` for fully reproducible builds.

---

## 5. Data contract

The upstream ETL (the prototype's `data_prep.py`, which was not part of the supplied code) publishes admissions as **Parquet or Feather**. A file or a directory of files is accepted.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `admission_no` | string / integer | yes | unique per admission; drives the train / validation split |
| `diagnosis_code_array` | list of string | yes | index 0 is the principal diagnosis; at least one code |
| `procedure_code_array` | list of string | yes | may be empty |
| `medical_claims_array` | list of string | yes | claim item numbers; may be empty |
| `drg_vers` | string | yes | for example `10.0`; must be a version seen in training |
| `drg` | string | yes | assigned AR-DRG such as `F62A`; shorter than 3 characters means "cannot be scored" |
| `los` | integer | yes | length of stay in days |
| `minutes_operating_theatre` | float | no | default 0 |
| `hours_of_mechanical_ventilation` | float | no | default 0 |
| `member_age` | float | no | default 50 (the prototype's imputation) |
| `drg_benefit` | float | no | default 0; scales the priority score |

```bash
python scripts/export_dataset.py --split train --output data/train.parquet   # run next to your data_prep.py
drg-audit validate-data --input data/train.parquet
```

- Invalid rows are dropped and counted. If more than 1% are invalid, the whole dataset is rejected (quality gate).
- The report lists counts only, never values.

---

## 6. Training

### 6.1 In Code Editor (GPU space)

```bash
drg-audit train --stage all --input data/train.parquet --output-dir artifacts/run
#  artifacts/run/pretrain   stage 1 encoder checkpoint
#  artifacts/run/bundle     serving bundle (what you package and deploy)
```

- Run one stage at a time with `--stage pretrain` or `--stage finetune --pretrained-dir artifacts/run/pretrain`.
- Override any setting without editing files: `--pretrain-set epochs=20 --finetune-set encoder_learning_rate=1e-5`. Unknown keys are rejected.
- Progress is emitted as JSON log lines (`pretrain_epoch`, `finetune_epoch`) with losses, per-head macro-F1, alignment, and uniformity.
- Set `runtime.checkpoint_dir` to enable atomic, resumable checkpoints. Loading them uses `weights_only=True`.
- Training refuses to run without a GPU unless `runtime.require_gpu: false`.

### 6.2 As a SageMaker Training Job

```bash
cp configs/train_job.example.yaml configs/train_job.dev.yaml     # replace every REPLACE_ME
python sagemaker/launch_training_job.py --config configs/train_job.dev.yaml --wait
aws s3 cp <model_artifact printed above> training_model.tar.gz
python sagemaker/build_model_package.py --model-tar training_model.tar.gz --output dist/model.tar.gz
```

- The launcher builds a source bundle, uploads it with SSE-KMS, and starts a job with network isolation, VPC attachment, KMS-encrypted output and volume, and inter-container encryption.
- Provide the curated Parquet files under `inputs.train_s3_uri`. The job writes the serving bundle to the root of its `model.tar.gz`.
- `hyperparameters.stage` is `all`, `pretrain`, or `finetune` (the last needs `inputs.pretrained_s3_uri`).
- Managed spot training is supported when `output.checkpoint_s3_uri` is set.
- Archives from other jobs are extracted only through `safe_extract_tar` (no links, no path traversal, bounded size).

### 6.3 Reusing your existing prototype weights

```bash
python scripts/convert_legacy_checkpoint.py \
  --legacy-checkpoint model_artefacts/integrated_drg_pipeline.pth \
  --vocab model_artefacts/vocab.json --output-dir artifacts/bundle
```

- The pickle is loaded with `weights_only=True` and an allow-list containing only `RobertaConfig` (needs torch 2.4 or later). `--allow-unsafe-pickle` exists only for a file you created yourself.
- The converter checks shapes and vocabulary, loads weights strictly, writes the bundle, and verifies a serialization round trip.
- **Tokenization choice (`--tokenization`)**
  - `legacy-training` (default): claims in source order and `[SEP]` dropped on truncation. This is exactly how the prototype *trained* these weights.
  - `standard`: the spec used by new models. The prototype's inference script effectively used this, so its predictions can differ from the default.
- Before relying on converted weights, score a representative sample with the old pipeline and the new bundle and compare flag agreement.

---

## 7. Packaging

```bash
python sagemaker/build_model_package.py --bundle-dir artifacts/run/bundle --output dist/model.tar.gz
```

The build verifies the bundle first and refuses to package a tampered one. It produces:

- `dist/model.tar.gz`: bundle files at the root plus `code/` (`inference.py`, the serving modules, `requirements.txt`, optional `wheelhouse/`).
- `dist/model.tar.gz.sha256` and `dist/package_meta.json`: digest, bundle id, dependency mode, wheel list. The deployer refuses a package that does not match its metadata.
- Reproducible output: the same inputs produce the same bytes (sorted members, zeroed timestamps and owners).

### Decision: how dependencies reach the container

| Option | How | Trade-offs |
| --- | --- | --- |
| **A. Offline wheelhouse (default, safest)** | `--dependency-mode wheelhouse` vendors wheels into the package and installs them with `--no-index` | No outbound network at start-up, works with network isolation, nothing fetched at runtime. Larger package; needs a build machine with network access and the matching `--python-version` |
| B. PyPI at start-up | `--dependency-mode pypi` | Simplest. Needs outbound internet (NAT or mirror), cannot use network isolation, and widens the supply-chain surface at every container start |
| C. Custom image | Derive from the AWS container and bake dependencies in | Strongest provenance and scan story, but needs Docker or CodeBuild (not available in Code Editor by default). Not implemented here |

- The deploy script blocks option B combined with network isolation.
- If the build machine cannot reach PyPI, point pip at your internal mirror through the standard `PIP_INDEX_URL` environment variable.

---

## 8. Deployment

### 8.1 Prerequisites

| Item | Requirement |
| --- | --- |
| Region | One that satisfies your data-residency policy (for example `ap-southeast-2` or `ap-southeast-4`) and offers the container and instance type |
| S3 bucket | Private, block public access, default SSE-KMS, versioning on |
| KMS key | Customer-managed key used for the package, endpoint volume, and (if enabled) data capture |
| Endpoint role | Created from `iam/endpoint_execution_role_trust.json` and `iam/endpoint_execution_role_policy.json` after replacing the `<PLACEHOLDERS>` |
| Deployer permissions | `iam/deployer_policy.json` for whoever runs `deploy_endpoint.py` (validate the guardrail statements with the IAM policy simulator) |
| VPC | At least two private subnets in different Availability Zones and a security group with no inbound rules from the internet |
| VPC endpoints | S3 (gateway), ECR API and ECR DKR (interface), CloudWatch Logs (interface). Clients inside the VPC also need SageMaker Runtime (interface) |

### 8.2 Deploy

```bash
cp configs/deploy.example.yaml configs/deploy.dev.yaml     # replace every REPLACE_ME
python sagemaker/deploy_endpoint.py deploy --config configs/deploy.dev.yaml \
    --package dist/model.tar.gz --smoke-test examples/request.json
```

What the command does, in order:

1. Validates the config and refuses to continue on any `REPLACE_ME`, missing VPC, package/metadata mismatch, or stage policy violation (all before any AWS call).
2. Checks that the execution role belongs to the calling account.
3. Uploads the package to a content-addressed S3 key with SSE-KMS and an expected-bucket-owner check; an existing object with a different digest aborts the run.
4. Creates the model (network isolation, VPC, offline Hugging Face), the endpoint config (KMS key, optional data capture), and three CloudWatch alarms.
5. Creates the endpoint, or updates it with a **blue/green canary** and automatic rollback on the 5xx and latency alarms.
6. Waits for `InService`, then **verifies the endpoint is serving the new configuration** (an alarm-driven rollback is reported as a failure).
7. Registers target-tracking autoscaling and runs the smoke test (aggregate counts only are printed).

### 8.3 Operate

```bash
python sagemaker/deploy_endpoint.py status   --config configs/deploy.dev.yaml
python sagemaker/deploy_endpoint.py rollback --config configs/deploy.dev.yaml --endpoint-config-name <previous-config>
python sagemaker/deploy_endpoint.py delete   --config configs/deploy.dev.yaml --yes [--delete-models]
```

- Each deploy creates new, uniquely named model and endpoint-config resources, so the previous configuration stays available for rollback.
- `stage: prod` additionally requires two or more instances, subnets in two Availability Zones, network isolation, and an SNS alarm topic.

### 8.4 Decisions with safest defaults

| Topic | Options | Default and trade-off |
| --- | --- | --- |
| Network posture | **Isolation + VPC**; VPC only; neither | Isolation + VPC. Needs the wheelhouse package and VPC endpoints. `allow_no_vpc` must be set explicitly, with security approval |
| Payload capture | **Off**; on with KMS | Off. Captured payloads contain clinical data, so enabling needs approval, a dedicated bucket, and retention rules |
| Rollout | **Canary + auto rollback**; all at once | Canary at 25% for 300 seconds. Slower deploys, safer failures |
| Per-request threshold | Allowed within bounds; locked | Allowed within `[min_confidence_threshold, 1.0]`. To make it stricter-only, set `limits.min_confidence_threshold` equal to the default threshold |

### 8.5 Sizing and scaling

- Start with `ml.m5.xlarge` (CPU), two instances, `SAGEMAKER_MODEL_SERVER_WORKERS=2`, and `DRG_AUDIT_TORCH_THREADS=2` (about vCPUs / workers). Each worker holds one copy of the model.
- The autoscaling target (1000 invocations per instance per minute) and the alarm thresholds are **starting points**. Derive them from a load test: sustained requests per minute per instance x 0.5 to 0.7.
- Use a GPU instance only if profiling shows CPU cannot meet your latency target; set `DRG_AUDIT_PRECISION=fp16` there if desired.

---

## 9. Inference API

SageMaker authenticates callers with IAM (SigV4). The container accepts and returns `application/json` only.

### 9.1 Request

```json
{
  "records": [ { "admission_no": "A100234", "diagnosis_code_array": ["I50.0", "I10"], "...": "see table" } ],
  "options": { "confidence_threshold": 0.85 }
}
```

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `records` | array | yes | 1 to 256 records |
| `options.confidence_threshold` | number | no | between `min_confidence_threshold` (0.5) and 1.0 |
| `admission_no` | string or integer | yes | letters, digits, `_`, `-`; up to 64 characters |
| `diagnosis_code_array` | array | yes | 1 to 512 codes; index 0 is the principal diagnosis |
| `procedure_code_array`, `medical_claims_array` | array | no | up to 512 codes each |
| `drg_vers` | string | yes | up to 16 characters |
| `drg` | string | yes | up to 8 alphanumeric characters (shorter than 3 is skipped) |
| `los` | integer | yes | 0 to 3650 |
| `minutes_operating_theatre`, `hours_of_mechanical_ventilation` | number | no | 0 to 100000, 0 to 8760 |
| `member_age` | number | no | 0 to 130; missing becomes 50 with a warning |
| `drg_benefit` | number | no | absolute value up to 1e9; missing becomes 0 with a warning |

- A code is a string or integer of up to 32 characters matching `[A-Za-z0-9][A-Za-z0-9._\-:/]*`.
- Booleans, `NaN`, `Infinity`, unknown fields, and out-of-range values are refused, never clamped.
- Limits are configurable (`configs/finetune.yaml`, `serving.limits`) and overridable per endpoint with `DRG_AUDIT_*` variables.

### 9.2 Response

See `examples/response.json`. Top level: `request_id`, `model_version`, `confidence_threshold`, `summary`, `results` (same order as the request).

| Result field | Meaning |
| --- | --- |
| `status` | `ok`, `skipped` (cannot be scored), or `rejected` (invalid input) |
| `reason` | `skipped`: `invalid_assigned_drg`, `unknown_drg_version`. `rejected`: `<code>:<field>` with code `missing_field`, `invalid_field`, `unknown_fields`, `not_an_object` |
| `predicted_mdc`, `predicted_base_drg`, `predicted_complexity_split` | model prediction |
| `mdc_confidence`, `base_confidence`, `split_confidence` | probability of each prediction |
| `prob_assigned_split` | probability the model gives the billed split |
| `*_discrepancy_flag`, `split_inflation_flag`, `is_billing_anomaly` | decision rules (section 2.3) |
| `audit_priority_score` | prioritisation score (0 unless anomaly) |
| `warnings` | `member_age_missing_defaulted`, `drg_benefit_missing`, `unknown_drg_version_fallback` |

### 9.3 Errors

Client errors reach callers as a SageMaker `ModelError` carrying the HTTP status.

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `invalid_json`, `invalid_encoding`, `empty_body`, `invalid_shape`, `invalid_records`, `unknown_fields`, `invalid_options` | malformed request |
| 406 | `not_acceptable` | `Accept` is not JSON |
| 413 | `payload_too_large` | over the byte or record limit |
| 415 | `unsupported_media_type` | `Content-Type` is not `application/json` |
| 500 | `internal_error` | generic; details are in CloudWatch Logs with a `request_id` |

### 9.4 Calling the endpoint

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name drg-audit-anomaly-dev --region <region> \
  --content-type application/json --accept application/json \
  --body fileb://examples/request.json response.json && cat response.json
```

```python
import json, boto3
runtime = boto3.client("sagemaker-runtime", region_name="<region>")
reply = runtime.invoke_endpoint(EndpointName="drg-audit-anomaly-dev", ContentType="application/json",
                                Accept="application/json", Body=open("examples/request.json", "rb").read())
print(json.loads(reply["Body"].read())["summary"])
```

- For datasets, use the bundled client, which batches, retries, and bounds concurrency: `drg-audit score --endpoint-name <name> --region <region> --input data/inference.parquet --output scores.parquet`.

---

## 10. Batch scoring and alert generation

```bash
drg-audit score --endpoint-name <name> --region <region> --input data/inference.parquet --output artifacts/scores.csv
drg-audit alerts --config configs/alerts.yaml --run-id 2026-10-14
```

- `drg-audit score` accepts `--bundle-dir` instead of `--endpoint-name` to score locally. Output files are owner-readable only.
- `drg-audit alerts` replaces the prototype's `main.py`:
  - Candidates are flagged admissions that also pass the business rules in `configs/alerts.yaml` (fund ids, principal diagnosis not starting with `C`, first secondary diagnosis not starting with `R` or `Z`, not previously alerted).
  - The internal AR-DRG grouper (`au_grouper`, not part of this repository) regroups each candidate with the principal diagnosis removed. Install a pinned version from your private package index; the adapter is `AuGrouperBackend`.
  - One JSON line per alert (`claim_seqno`, `alert_info`, `recovery_amount`, `alert_instance_id`) is written atomically to a run-specific file with owner-only permissions. Re-running with the same `--run-id` overwrites instead of duplicating.
- Expected grouper output columns: `admission_no`, `principal_diagnosis_code`, `drg`, `drg_rw`, `regroup_drg`, `regroup_drg_desc`, `regroup_drg_value`, `drg_ver_grouper`, `possible_recovery`, and the three `hospital_claims_*` arrays. The code validates them and fails with a clear message.
- Set `io.output_filename: "{concept_name}_json.txt"` to keep the legacy file name. Behaviour changes versus `main.py` are listed in section 14.

---

## 11. Security

### 11.1 Controls implemented

| Area | Control |
| --- | --- |
| Model loading | safetensors and JSON only; SHA-256 manifest verified before use; load fails closed; self-check at start-up |
| Request handling | JSON only; size, record, array, code-length, and numeric bounds; strict field allow-lists; code pattern check; refusal (not clamping) of bad values |
| Information disclosure | Errors carry field names and stable codes, never submitted values; unexpected errors return a generic 500 |
| Logging | Counts, durations, request ids, model version only. No payloads, codes, or admission numbers (covered by a test) |
| Network | Network isolation, VPC attachment, no runtime downloads (wheelhouse), Hugging Face hub disabled |
| Encryption | Customer-managed KMS key for the package, endpoint volume, training output and volume, and data capture; inter-container encryption in training |
| Supply chain | Pinned versions, binary-only wheels, deterministic package, digest verified before deploy, content-addressed S3 keys, `pip install` removed from application code |
| Identity | Separate least-privilege roles (`iam/`), `iam:PassRole` restricted to SageMaker, trust policy with confused-deputy conditions, role-account check, expected-bucket-owner on S3 calls |
| Data at rest on disk | Prediction files, alert files, and exported datasets are written mode 0600; the Excel copy of alerts is opt-in |
| Archives and YAML | Safe tar extraction (no links, no traversal, bounded); `yaml.safe_load` only; no `eval`, no `pickle` (except the explicit legacy opt-in) |
| Training | Checkpoints load with `weights_only=True`; GPU requirement enforced; data quality gate |

### 11.2 PHI handling

- Requests, responses, and alert files contain clinical context. Treat them as PHI under your policies.
- Payload capture is **off** by default. If enabled: dedicated bucket, KMS key, retention policy, restricted access, and approval.
- Business KPIs such as anomaly rate come from client-side aggregation (`drg-audit score` prints counts) because an isolated container cannot publish custom metrics.

### 11.3 Residual risks to review

- The endpoint accepts a per-request threshold. Bound it as described in section 8.4 if your governance requires a fixed threshold.
- `au_grouper` and `data_prep` are outside this repository; apply the same supply-chain controls to them.
- IAM policy templates need your account values and a policy-simulator check before use.

---

## 12. Operations

### 12.1 Signals

| Signal | Source | Action |
| --- | --- | --- |
| `Invocation5XXErrors`, `ModelLatency` p95, `MemoryUtilization` | CloudWatch alarms created by the deployer | Notify the SNS topic; 5xx and latency also trigger automatic rollback during updates |
| Request log lines (`request_scored`) | Endpoint log group `/aws/sagemaker/Endpoints/<name>` | Logs Insights query below |
| `model_loaded`, `model_load_failed` | Same log group | Confirms version, device, and tokenization at start-up |

```
fields @timestamp, @message
| filter @message like /"event":"request_scored"/
| parse @message /"latency_ms":(?<latency>[0-9.]+)/
| parse @message /"n_anomalies":(?<anomalies>[0-9]+)/
| stats count() as requests, avg(latency) as avg_ms, max(latency) as max_ms, sum(anomalies) as anomalies by bin(5m)
```

### 12.2 Model lifecycle

- Retrain on a schedule or when drift is suspected, then package, deploy as a canary, and watch alarms. The previous endpoint configuration remains available for rollback.
- Track the anomaly rate and priority-score distribution over time from your batch scoring; a sudden shift often means upstream data changed.
- Recalibrate the threshold from reviewed audit outcomes rather than from validation F1 alone.

### 12.3 Runbook

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Endpoint stuck `Creating` then `Failed` | Container could not start: bundle integrity failure, missing wheel, wrong Python version | Read `FailureReason` and the log group; rebuild with the matching `--python-version`; run `drg-audit verify-bundle` |
| Start-up fails with pip errors | Package built in `pypi` mode but network isolated, or wheel platform mismatch | Rebuild in `wheelhouse` mode; check `--platform` tags |
| Deploy reports a different configuration is serving | Alarm-driven blue/green rollback | Inspect alarms and logs, fix, redeploy |
| Many `rejected` results | Client sends unsupported shapes (floats for codes, unknown fields) | Compare with `examples/request.json` and section 9.1 |
| `unknown_drg_version_fallback` warnings | A DRG version the model never saw | Retrain with that version, or set `DRG_AUDIT_UNKNOWN_VERSION_POLICY=reject` |
| High latency | Too few workers or threads, large batches | Tune workers and threads; lower client batch size; scale out |
| 413 responses | Payload over limit | Reduce records per call (client default 128) |

---

## 13. Testing and pre-production checklist

```bash
make test        # all suites
make lint        # ruff with security rules
make smoke       # end-to-end on synthetic data
```

| Suite | Covers |
| --- | --- |
| `test_features`, `test_decision`, `test_config` | tokenization, padding equivalence, decision rules, strict config and env overrides |
| `test_contract`, `test_service`, `test_inference_entrypoint` | request validation and rejection, PHI-free logging, 4xx / 500 mapping, response strictness |
| `test_artifacts`, `test_archive` | manifest integrity, tamper detection, safe extraction |
| `test_build_package`, `test_deploy`, `test_training_launcher`, `test_configs` | package layout and reproducibility, deploy policy and AWS request assembly (fake clients), shipped YAML and examples |
| `test_alerts` | candidate selection, output contract, idempotent writes |
| `test_torch_pipeline` | tiny end-to-end training, bundle round trip, engine determinism and batch invariance, handler round trip, tamper refusal (needs torch) |

**Before the first production deployment**

- [ ] `make test` and `make lint` pass in the dev environment (torch suite included).
- [ ] `make smoke` completes on your Code Editor space.
- [ ] Deploy to a sandbox account or stage `dev` and run the smoke test against the live endpoint.
- [ ] Load test with realistic payloads; set autoscaling and alarm thresholds from the results.
- [ ] Compare predictions of any converted prototype weights with the old pipeline on a labelled sample.
- [ ] Review IAM templates, KMS policy, and VPC endpoints with your security team; run the IAM policy simulator.
- [ ] Decide on payload capture, threshold governance, and retention; document the decisions.

During authoring, the 131 torch-free tests passed (the 10 torch tests were skipped there). The torch suite, `make lint`, `make smoke`, and every live AWS call have not been exercised yet; run them in your environment before relying on the project.

---

## 14. Migration from the prototype

| Prototype | Now |
| --- | --- |
| `common.py`, `data_prep` import | `drg_audit/data/io.py` and the Parquet data contract; `scripts/export_dataset.py` bridges your `data_prep.py` |
| `pipeline_train_embedding.py` | `drg_audit/training/pretrain.py` |
| `pipeline_integrated_joint_trainer.py` | `drg_audit/training/finetune.py` |
| `pipeline_inference_by_joint_ft_classifier.py` | `drg_audit/inference/` + `serving/` + `sagemaker/code/inference.py` |
| `main.py` | `drg_audit/postprocess/alerts.py` + `configs/alerts.yaml` |
| `environment.yml` | `environment.yml` and `requirements/*.in` |
| `.pth` pickles, `vocab.json` | safetensors bundle (`scripts/convert_legacy_checkpoint.py`) |

### Behaviour changes and fixes

- **Train / serve skew removed.** The prototype trained with medical claims in source order and dropped `[SEP]` on truncation, but its inference script sorted claims and kept `[SEP]`. Now one function serves both, and the spec is stored in the bundle.
- **Gradient clipping** now happens after AMP unscaling. The prototype clipped loss-scaled gradients.
- **Validation is deterministic and clean.** Stage 2 metrics use clean views; the prototype scored randomly masked views. Stage 1 validation uses fixed masks.
- **Hash-based split** on `admission_no`, shared by both stages.
- **Vocabulary** is built from the training split only.
- **Endpoint returns every record** with a status; the prototype dropped non-anomalous and short-DRG records. `base_discrepancy_flag` and confidences are now in the output.
- **Unknown DRG versions** produce a warning (or a skip) instead of silently using index 0.
- **CPU serving** is supported; the prototype aborted without a GPU. Default precision is fp32.
- **Memory**: token ids are stored as one compact int32 array instead of two int64 tensors per record.
- **`main.py` issues fixed**: `NameError` when nothing qualified, `int(None)` for admissions without a matching claim, appended duplicates on re-runs, runtime `pip install` (including `groupy` from public PyPI, a dependency-confusion risk), hard-coded user paths, a typo in the alert text ("amout"), and an unconditional Excel copy.
- **Torch version mismatch**: the prototype's inference script needed torch 2.4 or later (`add_safe_globals`) while `environment.yml` pinned 2.1.2. Versions are now pinned consistently.
- **Same-day stays**: the length-of-stay bucket keeps its legacy behaviour (LOS 1 maps to 0.0, LOS 0 maps to 0.25). Changing it requires retraining.

### Assumptions and open items

- `data_prep.py` and `au_grouper` were not supplied. The data contract and the grouper adapter are inferred from how the prototype used them.
- Autoscaling targets and alarm thresholds are not measured values.
- Single variant, single container; shadow variants and multi-model endpoints are out of scope.

---

## 15. Configuration reference

### 15.1 Environment variables on the endpoint

| Variable | Default | Purpose |
| --- | --- | --- |
| `DRG_AUDIT_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `DRG_AUDIT_PRECISION` | `fp32` | `fp32`, `fp16` (CUDA only), `bf16` |
| `DRG_AUDIT_MAX_BATCH_SIZE` | 64 | records per forward pass |
| `DRG_AUDIT_TORCH_THREADS` | torch default | intra-op threads per worker |
| `DRG_AUDIT_DYNAMIC_PADDING` | `true` | trim padding per batch (numerically equivalent) |
| `DRG_AUDIT_UNKNOWN_VERSION_POLICY` | `fallback` | `fallback` or `reject` |
| `DRG_AUDIT_CONFIDENCE_THRESHOLD` | from bundle (0.85) | overrides the bundle default |
| `DRG_AUDIT_MAX_PAYLOAD_BYTES`, `DRG_AUDIT_MAX_RECORDS` | 8 MiB, 256 | request bounds |
| `DRG_AUDIT_LOG_LEVEL` | `INFO` | log verbosity |
| `SAGEMAKER_MODEL_SERVER_WORKERS`, `SAGEMAKER_MODEL_SERVER_TIMEOUT` | container default, 60 | worker count and request timeout |

Invalid values stop the container at start-up with a clear error.

### 15.2 Files

| File | Purpose |
| --- | --- |
| `configs/pretrain.yaml`, `configs/finetune.yaml` | training hyper-parameters; `finetune.yaml` also holds `serving:` defaults baked into the bundle |
| `configs/alerts.yaml` | alert rule, eligibility, grouper, and IO paths |
| `configs/deploy.example.yaml` | every deployment setting, with comments |
| `configs/train_job.example.yaml` | training-job settings |

---

## 16. Troubleshooting

| Problem | Check |
| --- | --- |
| `error: ConfigError: ... REPLACE_ME` | Complete the config; the deployer lists the keys still unset |
| `integrity check failed for ...` | The bundle or package was modified or truncated; rebuild or re-download |
| `package does not match package_meta.json` | The package changed after the build; rebuild it |
| `no GPU was detected` | Use a GPU space or job, or set `runtime.require_gpu: false` for tests |
| Legacy conversion: `torch >= 2.4 is required` | Upgrade torch; use `--allow-unsafe-pickle` only for a file you created |
| `container.image_uri is not set` | Set it explicitly, or `pip install sagemaker` so the Deep Learning Container URI can be resolved |
| `pip download` fails while packaging | Check network access and that `--python-version` and `--platform` match the container |
| Torch tests are skipped | Run in the dev environment where torch, transformers, safetensors, and scikit-learn are installed |
