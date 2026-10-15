# Dx supporting lists

LLM-assisted **supporting lists** for ICD-10-AM diagnosis categories, served from an **Amazon SageMaker real-time endpoint** that calls **Claude on Amazon Bedrock**.

For each 3-character ICD-10-AM category the system produces four lists: **ACHI procedure codes**, **MBS billing items**, **prosthesis billing codes**, and (opt-in) **other diagnosis categories**, whose presence on a hospital episode indicates the diagnosis was **treated**, **investigated**, or **examined** (or, for the diagnosis list, makes it clinically plausible). A verdict is a screening signal, never a coding determination: every verdict stays available for review by a qualified clinical coder.

## Contents

1. [What changed from the earlier build](#1-what-changed-from-the-earlier-build)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Configuration reference](#4-configuration-reference)
5. [Prerequisites](#5-prerequisites)
6. [Develop and test in SageMaker Code Editor](#6-develop-and-test-in-sagemaker-code-editor)
7. [Deploy to production](#7-deploy-to-production)
8. [Inference guide](#8-inference-guide)
9. [Batch builds and pre-warming](#9-batch-builds-and-pre-warming)
10. [Operations](#10-operations)
11. [Verification status and known limitations](#11-verification-status-and-known-limitations)
12. [Governance](#12-governance)
13. [Appendix: old to new](#13-appendix-old-to-new)

---

## 1. What changed from the earlier build

The method, the prompts, the verdict tiers, and the outputs are unchanged. The system around them was rebuilt for production.

| Area | Earlier build | This build |
|---|---|---|
| Model access | Claude Agent SDK: a Claude Code subprocess per call, API key or OAuth token | **Amazon Bedrock Converse API** through `boto3`, using the endpoint's **IAM role**. No keys, no `.env`, no subprocess |
| Per-call overhead | About 20,000 input tokens of wrapper context per call (measured in the earlier README) | None: the request contains only the prompt. The agent already ran with `tools=[]`, so no tool loop was lost |
| Structured output | SDK `output_format` | One tool, `submit_result`, with `toolChoice: auto` (Claude 5.x on Bedrock rejects forced tool choice), then strict validation of every returned id |
| Run state | Local JSONL files, a local lock file | **Object store** (S3 in production, local files for development), one object per decision, atomic conditional-write lock |
| Interface | Batch CLI only | **HTTP service** (`/ping`, `/invocations`) with streaming, plus the full batch CLI under `dx-support` |
| Stale results | Resume trusted whatever was on disk | **Cache fingerprint** (data, prompts, models, thinking, tunables): a changed configuration never reuses old verdicts |
| Cost control | Per-run cap | Per-call worst-case refusal, per-category cap, **per-hour per-instance circuit breaker**, adaptive concurrency under throttling |
| Deployment | A folder you copy | Container image, two CloudFormation stacks, blue/green endpoint with automatic rollback, autoscaling, alarms |
| Security | Guards in code | The same guards, plus VPC-only egress, KMS encryption, least-privilege role, production start-up refusals, no payload logging |

**Why not keep the Agent SDK?** On a SageMaker endpoint it would mean shipping Node and the Claude Code CLI in the image, spawning a process for every one of tens of thousands of calls, and paying for the wrapper's context on each. The `LLMBackend` interface (`src/dx_support/llm/types.py`) is a single method, so another backend can be added without touching the pipeline.

---

## 2. Architecture

### 2.1 Overview

```
                      callers (IAM SigV4: sagemaker:InvokeEndpoint on this one endpoint)
                                              |
                  InvokeEndpoint (60 s)  /  InvokeEndpointWithResponseStream (8 min)
                                              |
        +-------------------------------------v--------------------------------------+
        |  SageMaker real-time endpoint  (>= 2 instances, VPC, no inbound, blue/green) |
        |  +------------------------------------------------------------------------+ |
        |  | container: gunicorn (1 process, N threads) -> Flask                    | |
        |  |   GET /ping   POST /invocations                                        | |
        |  |   validate -> answer from store | start jobs -> wait / stream / 202    | |
        |  |                                                                        | |
        |  |   event-loop thread: jobs  ->  LLMGate (shared adaptive limiter,       | |
        |  |   per-category cap, per-hour budget)  ->  BedrockBackend (Converse)    | |
        |  +--------------------+---------------------------------+-----------------+ |
        +-----------------------|---------------------------------|--------------------+
               private subnets, SG egress 443 to the VPC only      |
                                |                                   |
         VPC endpoints (Bedrock Runtime, S3, KMS, Logs, ECR)        |
                |                              |                    |
     Amazon Bedrock (inference profile,   S3 artifacts bucket (KMS, versioned, TLS-only)
      e.g. au.anthropic.claude-sonnet-5)    model/<v>/model.tar.gz  -> /opt/ml/model/data (reference files)
                                            serve/<fingerprint>/    -> briefs, routes, verdicts, status, run log

   batch:  dx-support build --for-service  (laptop, Code Editor, or a SageMaker Processing job via `dx-support prewarm`)
           writes into the SAME serve/<fingerprint>/ namespace, so the endpoint answers those categories instantly.
```

### 2.2 Serving model: a read-through cache

A SageMaker real-time endpoint allows **60 seconds** for a regular response and **8 minutes** for a streaming response, and neither limit can be changed. Computing one category from scratch takes many sequential model calls (a brief, then one routing call per tree level, then the judge): the earlier build measured about 160 seconds for one category with a single family, and about 19 calls per category with all three. So the endpoint is a **read-through cache in front of the pipeline**:

| Situation | What the endpoint does |
|---|---|
| Result already stored (the normal case once pre-warmed) | Answers in well under a second with `status: complete`, **0 model calls** |
| Not stored, caller used the regular API | Starts the work, waits up to `DX_SYNC_DEADLINE_S` (50 s), then answers **HTTP 202** `in_progress`. The work **keeps running** on the instance; calling again with the same body is free once it finishes |
| Not stored, caller used the streaming API (`"stream": true`) | Holds the connection up to `DX_STREAM_MAX_S` (450 s), sends `progress` events every few seconds, and ends with a `result` event |
| Stream cut at the 8-minute cap or dropped | Nothing is lost: every decision is stored the moment it exists. Reconnect and the work resumes |
| `cache_only: true`, or `DX_ALLOW_COMPUTE=false` | Never calls a model. Returns what is stored and marks the rest `not_cached` |

Pre-warming (section 9) is how production traffic stays on the first row. The reference client (`dx_support.client.EndpointClient`, `dx-support invoke`) implements the whole dance for callers.

### 2.3 Request lifecycle

1. SageMaker authenticates the caller (SigV4) and forwards the body to `POST /invocations`.
2. **Admission**: at most `DX_MAX_CONCURRENT_REQUESTS` requests per instance; more get **429** with `Retry-After`. `/ping` is never starved (extra threads are reserved).
3. **Validation** (`service/schemas.py`): JSON object only, unknown fields rejected, categories matched against `^[A-Z][0-9]{2}$` **and** the loaded reference data, families and tiers against allowlists, strict booleans, bounded sizes. Caller text never reaches a prompt.
4. For each category, one `GET` of its status object decides *stored* or *not*. Stored categories are assembled from their verdict objects.
5. Not-stored categories become **jobs** on the event-loop thread. A per-category lock means two requests for the same category never pay twice: the second waits, then finds it done.
6. A job hydrates the category's stored decisions, runs only what is missing (`process_category`), and writes its status record last, after every verdict batch is durable.
7. The handler waits (sync deadline or stream), assembles the response, and returns it.

### 2.4 The pipeline

Per category and per code family, in three stages (unchanged from the validated build; prompts are verbatim, see `pipeline/prompts.py`):

1. **Brief** (1 call per category). The model writes a short clinical work-up brief and plain search phrases. One brief drives every family, keeping the lists consistent.
2. **Route**. Each family is a small-fan-out tree (ACHI: 2-, 3-, then 4-digit groups; MBS: category/group, then sub-group; prostheses: sub-category, then product group; diagnoses: chapter, then block). The model picks every branch that could hold a supporting code, instructed to be inclusive. A branch with one child is passed through unasked. In parallel, **lexical rescue** (BM25 over every unit, driven by the brief's search phrases) adds candidates the walk may have missed. Branches over 150 units are ranked and cut, and flagged.
3. **Judge**. Every candidate is classified, in batches of 100-150, and skipped candidates are re-judged in batches of 25 and then 10:

| Tier | Meaning |
|---|---|
| `core` | Specific to the category's condition or site |
| `broad` | Non-specific treatment, investigation, or examination consistent with the category |
| `unclear` | The description is not enough to decide |
| `not_supporting` | Unrelated, or so generic it appears whatever the diagnosis |

Each `core` or `broad` verdict has an **evidence type** (`treated`, `investigated`, `examined`; for the diagnosis family the **relation**: `underlying_condition`, `complication_or_manifestation`, `associated_comorbidity`) and a one-sentence rationale. For large categories (more than 20 member codes) a verdict can be scoped to member codes. Prosthesis verdicts are per **device type** and expand to every billing code in the group. Chapters XX and XXII (circumstances, not conditions) are skipped by default.

`mode: exhaustive` reviews every unit instead of routing (about 100 calls per category). It exists to measure routing recall and for certainty on single categories; it is a batch tool and is not what the endpoint serves.

### 2.5 Calling Claude on Bedrock

`llm/bedrock.py` is the only module that imports `boto3` for model calls.

* **Request**: one `converse` call per attempt. `system` = the stage's system prompt plus an instruction to answer by calling `submit_result`; one tool whose `inputSchema` is the stage's JSON Schema; `toolChoice: {"auto": {}}`; `inferenceConfig` carries **only** `maxTokens`. No temperature, `top_p`, or `top_k`: Claude 5.x rejects them.
* **Structured output**: the tool call is taken **by block type, never by position** (thinking blocks may come first). If the model answers in text instead, the outermost JSON object is used. Anything else is a failed attempt, retried. Whatever arrives is then validated: every id must be one that was sent.
* **Thinking** (`DX_THINKING`: `disabled`, `low`, `medium`, `high`, `adaptive`) is mapped per model by `data/models.json`, because the models differ:

| Model style | `low` / `medium` / `high` | `adaptive` | `disabled` |
|---|---|---|---|
| Adaptive-effort (Sonnet 5, Sonnet 5.5, Haiku 5.5, Opus 5.5, Fable 5.1) | `thinking: adaptive` + `output_config.effort` | `thinking: adaptive` | Per model: `thinking: disabled` (Sonnet 5, Haiku 5.5), `thinking: between_tools` (Sonnet 5.5, Opus 5.5), or adaptive at low effort with a warning (Fable 5.1) |
| Budget (Haiku 4.5) | `thinking: enabled` + `budget_tokens` 1024 / 4096 / 10000 | 4096 | field omitted |

* **Model ids**: a catalog alias such as `claude-sonnet-5` becomes `<DX_INFERENCE_PROFILE_PREFIX>.anthropic.claude-sonnet-5` (default `au.`). Newer Claude models cannot be invoked on demand by base id; an explicit id or ARN is used verbatim. An unknown alias is an error, so a typo can never silently become another model.
* **Cost accounting** comes from the `usage` block (cache reads and writes priced separately) with a configurable premium for geo profiles. A call whose **worst-case** cost (estimated input plus the full `maxTokens`) exceeds `DX_MAX_BUDGET_USD_PER_CALL` is refused before it is sent.
* **Retries** live in `LLMGate`, not `boto3` (SDK retries are off, so every attempt is counted once): transient errors retry with jittered back-off; throttling retries longer and **halves model-call concurrency** (recovering one step per 20 successes); a validation or permission error is **fatal** and stops the run instead of failing every remaining call the same way.
* **Prompt caching** (`DX_PROMPT_CACHING`, off by default) marks the system prompt and the constant catalogue prefix of brief and routing prompts. Turn it on for large batch builds after checking the minimum cacheable size of your model.
* **Guardrails**: set `DX_GUARDRAIL_ID` and `DX_GUARDRAIL_VERSION` to apply a Bedrock guardrail to every call.

### 2.6 State: layout and the fingerprint

Everything lives under one prefix per configuration:

```
s3://<bucket>/serve/<fingerprint>/
    briefs/<CAT>.json                       the clinical brief
    routes/<CAT>/<family>/<key>.json        one routing decision
    verdicts/<CAT>/<family>/<batch>.jsonl   one judged batch, one verdict per line
    status/<CAT>.json                       complete / partial, per-family stats, errors
    runlog/<YYYYMMDD>/<run>-<seq>.jsonl     one record per model-call attempt (tokens, cost, duration; never a prompt)
```

Objects are never appended to and every key is a pure function of its content, so writes are idempotent and two writers racing on the same decision produce equivalent objects. Loading is per category, which is what lets many instances share one store.

The **fingerprint** is a hash of the reference-data digest, every prompt and schema, the batch and routing tunables, the **resolved** model ids, the thinking setting, token caps, lexical rescue, coarse-prosthesis inclusion, the backend, and `DX_CACHE_SALT`. Change any of them and the endpoint starts a new, empty namespace: stale verdicts are never served for a configuration that did not produce them. Stub-backend results can never share a namespace with real ones.

### 2.7 Concurrency, limits, and cost controls

| Control | Default | Scope | Effect when reached |
|---|---|---|---|
| `DX_MAX_CONCURRENT_REQUESTS` | 8 | per instance | HTTP 429 + `Retry-After` |
| `DX_MAX_CATEGORIES_PER_REQUEST` | 3 | per request | HTTP 422 |
| `DX_LLM_CONCURRENCY` | 16 | per instance | calls queue; shrinks automatically while Bedrock throttles |
| `DX_MAX_BUDGET_USD_PER_CALL` | $0.50 | per call | the call is refused before it is sent |
| `DX_MAX_COST_USD_PER_CATEGORY` | $3.00 | per category job | the job stops launching calls; the category is `partial` and resumable |
| `DX_MAX_COST_USD_PER_HOUR` | $100 | per instance, rolling hour | circuit breaker: no new calls until spend ages out |
| `DX_LLM_CALL_TIMEOUT_S` | 240 | per call | transient failure, retried |
| `DX_ALLOW_COMPUTE=false` | true | instance | read-only endpoint: never calls a model |
| `--max-total-cost-usd` | $50 | batch run | the run stops; re-run to continue |

Worst-case spend per instance is bounded by `DX_MAX_COST_USD_PER_HOUR` plus the calls in flight when it trips (about `DX_LLM_CONCURRENCY` times the per-call budget). Multiply by the instance count for the endpoint.

### 2.8 Failure behaviour

| Event | Behaviour |
|---|---|
| Bedrock throttles | Retries with back-off, concurrency halves, then recovers; after the retry limit the category is `partial` |
| Permission, model-name, or request-shape error | **Fatal**: the job stops at once and the status says why. `dx-support doctor` finds these before deployment |
| Model ignores the tool or returns an unusable answer | Failed attempt (costed), retried; ids it invented are dropped |
| Judge skips candidates | Re-judged in batches of 25, then 10; any still missing keep the category `partial` |
| Client disconnects or the stream hits 8 minutes | Jobs continue; call again to resume at no extra cost |
| Instance replaced or scaled in mid-job | Stored decisions survive; the next call resumes. A decision in flight at that moment is re-bought |
| S3 unreachable at start-up | The container exits with the reason (SageMaker reports the deployment failed quickly) |
| S3 unreachable later | Requests that need it fail with 500; `/ping` stays healthy so the instance is not recycled for a transient store problem |
| Bad configuration | The container refuses to start and names the variable (never its value) |

### 2.9 Security

**Controls implemented**

* **No secrets anywhere.** Models and storage are reached with the endpoint's IAM role. Production **refuses to start** if static AWS keys or AWS endpoint overrides are present (`DX_ALLOW_STATIC_CREDENTIALS` / `DX_ALLOW_ENDPOINT_OVERRIDE` exist for deliberate exceptions), if the backend is the stub, or if the store is not S3.
* **Least-privilege role** (`foundation.yaml`): invoke only the Bedrock ARNs you name; read-only on the model artifact; get/put on `serve/*` (it can delete only run-lock objects, and list only in two folders); ECR pull; SageMaker log groups; ENI management for the VPC; KMS use of one key. The trust policy is pinned to your account.
* **Network**: instances run in private subnets, **no inbound**, security group egress **only tcp/443 to the VPC range** (where the interface VPC endpoints live) and optionally the S3 prefix list. Network isolation (`EnableNetworkIsolation`) cannot be used because the container must reach Bedrock; the VPC and security group are what confine it.
* **Encryption**: customer-managed KMS key with rotation for the bucket, the instance volume, and ECR; bucket versioned, public access blocked, TLS-only, owner-enforced.
* **Untrusted input everywhere**: reference text is sanitised (control characters and this project's delimiters stripped, lengths bounded) and the prompts say to classify it, never obey it; MBS XML is parsed with `defusedxml` and DTDs are refused; request bodies are validated strictly; model output is validated against the ids that were sent; spreadsheet formulas in any text are neutralised in every CSV and workbook the compile step writes.
* **The model has one tool**, the output-submission tool. It cannot read files, browse, or execute anything. Tool calls with any other name are ignored.
* **Logging hygiene**: structured JSON with a request id; ids, counts, tokens, cost, and durations only. Prompts, model output, request bodies, and anything credential-shaped are never logged (a test enforces it).
* **Supply chain**: ranged runtime dependencies with an optional **hash-locked** `requirements.lock` (`make lock`) that the Dockerfile prefers; unprivileged container user; immutable ECR tags with scan-on-push; the licensed data is never baked into an image (`.dockerignore`, and the CodeBuild source zip excludes it).
* **Offline invariants**: `deploy/scripts/validate_templates.py` fails on wildcard actions, a wildcard resource outside the few actions that need one, a world-open or non-443 rule, a public or unencrypted bucket, mutable image tags, a secret-looking environment variable, or an endpoint without a VPC and automatic rollback. Tests prove each check bites.

**Decisions with options** (the safest default is what the templates and settings use)

| Decision | Options | Default and trade-off |
|---|---|---|
| Network egress | **A. Private subnets + interface VPC endpoints, egress to the VPC only** / B. Private subnets + NAT gateway / C. Public subnets | **A.** Smallest blast radius; needs VPC endpoints (`CreateVpcEndpoints=true`, or existing ones). B allows broader egress. C is not supported by the templates |
| Where inference may run | **A. `au` geo profile** / B. another geo profile (`apac`, `us`, `eu`, `jp`) / C. `global` | **A.** Keeps inference in Australia; not every model has an `au` profile (the catalog lists the ones known; `doctor` confirms). C has the most capacity and is about 10% cheaper but may process anywhere. The data sent is code descriptions only, never patient data |
| Who can call it | **A. IAM only: `sagemaker:InvokeEndpoint` on this endpoint** / B. API Gateway or a Lambda in front for non-AWS callers | **A.** It is also the only built-in path. B adds an authorizer and a second hop you must secure; not included |
| Encryption | **A. New customer-managed KMS key** / B. Your existing key / C. S3-managed (SSE-S3) | **A.** Rotation and audit through CloudTrail; C is simpler but gives up key-level control |
| Prompt and response retention | **A. Never logged by this code** / B. Account-level Bedrock invocation logging (your choice, outside this project) | **A.** If B is enabled by an administrator, prompts (reference text only) are retained by that logging |

---

## 3. Project structure

```
dx-supporting-lists/
├── README.md
├── pyproject.toml              package metadata, dependencies, the `dx-support` entry point
├── requirements.txt            runtime dependency ranges (make lock -> requirements.lock, hash-pinned, preferred by the Dockerfile)
├── requirements-dev.txt        + pytest, ruff, pip-tools, pyyaml
├── Makefile                    test, dry-run, lock, validate-templates, deploy-*, package-data, image, describe
├── Dockerfile                  multi-stage, unprivileged user, `serve` entry for SageMaker
├── buildspec.yml               CodeBuild: build and push the image (no local Docker needed)
├── .dockerignore  .gitignore   licensed data, results, env files never enter an image or the repository
├── data/README.txt             the four input files and their formats (the files themselves are never committed)
├── deploy/
│   ├── cloudformation/
│   │   ├── foundation.yaml     KMS key, bucket, ECR, execution role, security groups, optional VPC endpoints, optional CodeBuild
│   │   └── endpoint.yaml       model, endpoint config, blue/green endpoint + rollback alarms, autoscaling, alarms
│   ├── container/
│   │   ├── entrypoint.sh       `serve` -> gunicorn; anything else -> the dx-support CLI
│   │   └── gunicorn.conf.py    one worker process, gthread, graceful shutdown hook
│   ├── scripts/
│   │   ├── package_model_data.sh    reference files -> model.tar.gz -> S3 (bucket default encryption)
│   │   ├── build_push_image.sh      docker build + push (immutable tag)
│   │   ├── build_image_codebuild.sh zip source -> CodeBuild -> ECR (no local Docker)
│   │   ├── make_source_zip.py       source zip that excludes data, tests, env files
│   │   └── validate_templates.py    offline template checks incl. the security invariants
│   └── env.example             non-secret local settings
├── src/dx_support/
│   ├── settings.py             DX_* environment -> validated Settings; production guards
│   ├── constants.py  text.py   vocabulary; text hygiene (sanitise, redact, spreadsheet-safe cells)
│   ├── logging_setup.py        JSON logs with request id; credential redaction
│   ├── sample_data.py          invented, non-clinical sample files for tests and demos
│   ├── deploy_tools.py         exact Bedrock IAM resource ARNs from your settings
│   ├── prewarm.py              Processing-job request copied from the live endpoint
│   ├── client.py               reference client: 202 handling, streaming, back-off
│   ├── reference/              chapters, loaders for the four files, derived categories
│   ├── indexing/               family trees, exclusions with reasons, BM25, universe loader + digest
│   ├── llm/
│   │   ├── bedrock.py          Converse backend: tool-based structured output, error mapping, worst-case budget
│   │   ├── catalog.py          alias -> Bedrock id, prices, per-model thinking rules
│   │   ├── gate.py             limiter (AIMD), retries, cost accounting, rolling budget, stop control
│   │   ├── fake.py             offline non-clinical stub (dry runs, tests; refused in production)
│   │   └── types.py
│   ├── storage/                object store (local, S3, namespaces), checkpoint, run lock
│   ├── pipeline/               prompts, parsing, stages, projection/calibration, fingerprint
│   ├── service/
│   │   ├── wsgi.py             gunicorn entry; loud synchronous start-up
│   │   ├── app.py              Flask: /ping, /invocations, sync / 202 / streaming
│   │   ├── runtime.py          event-loop thread, shared limiter and budget, per-category jobs
│   │   ├── schemas.py          strict request validation
│   │   └── results.py          verdicts -> response (tier merge, prosthesis expansion, audit)
│   ├── cli/                    build, estimate, compile, status, validate-retrieval, validate-judge, doctor, bedrock-arns, invoke, prewarm, ...
│   └── data/models.json        model catalog: ids, prices, thinking rules
└── tests/                      unittest-style (pytest collects them too): LLM, storage, pipeline, service, CLI, deployment
```

---

## 4. Configuration reference

All configuration is environment variables (`DX_*`, plus the standard `AWS_*`). `Settings.from_env()` validates at start-up and fails closed, naming the variable and never its value. **There are no credential variables.**

| Variable | Default | Meaning |
|---|---|---|
| `DX_ENV` | `development` | `production` enables the strict start-up rules; also `test` |
| `DX_BACKEND` | `bedrock` | `fake` is the offline stub (refused in production) |
| `DX_AWS_REGION` / `AWS_REGION` | `ap-southeast-2` | Region of Bedrock and S3 |
| `DX_MODEL` | `claude-sonnet-5` | Catalog alias or explicit Bedrock id / ARN |
| `DX_BRIEF_MODEL` `DX_ROUTE_MODEL` `DX_JUDGE_MODEL` | `DX_MODEL` | Per-stage overrides (measure a cheaper router first) |
| `DX_INFERENCE_PROFILE_PREFIX` | `au` | `au`, `apac`, `us`, `eu`, `jp`, `global`, or `none` (base id) |
| `DX_MODEL_CATALOG` | packaged | Path to a replacement `models.json` |
| `DX_THINKING` | `low` | `disabled`, `low`, `medium`, `high`, `adaptive` |
| `DX_MAX_OUTPUT_TOKENS_BRIEF` `_ROUTE` `_JUDGE` | 4000 / 6000 / 16000 | Output caps (thinking included); also bound the worst-case cost |
| `DX_PROMPT_CACHING` | `false` | Bedrock prompt-cache checkpoints |
| `DX_GUARDRAIL_ID` `DX_GUARDRAIL_VERSION` | unset | Both or neither |
| `DX_LLM_CONCURRENCY` | 16 | Model calls in flight per process |
| `DX_LLM_CALL_TIMEOUT_S` | 240 | Per-call read timeout |
| `DX_MAX_CALL_RETRIES` `DX_MAX_THROTTLE_RETRIES` | 3 / 6 | Retry limits |
| `DX_MAX_BUDGET_USD_PER_CALL` | 0.50 | Worst-case refusal threshold |
| `DX_MAX_COST_USD_PER_CATEGORY` | 3.00 | Service: cap for one category job |
| `DX_MAX_COST_USD_PER_HOUR` | 100 | Service: rolling per-instance circuit breaker |
| `DX_DATA_DIR` | `./data` | The four reference files (`/opt/ml/model/data` in the container) |
| `DX_STORE_URI` | `file://<cwd>/output` | `s3://bucket[/prefix]` (required in production) or a local path |
| `DX_STORE_KMS_KEY_ID` | unset | Sent as the SSE-KMS key on S3 writes; otherwise bucket default encryption applies |
| `DX_MODE` | `routed` | `exhaustive` is for batch certainty runs, not recommended for the endpoint |
| `DX_LEXICAL_RESCUE` | `true` | BM25 rescue candidates |
| `DX_INCLUDE_COARSE_PROSTHESES` `DX_INCLUDE_CONTEXT_CHAPTERS` | `false` | Opt-ins |
| `DX_CACHE_SALT` | empty | Change to force recomputation under a new namespace |
| `DX_MAX_CATEGORIES_PER_REQUEST` | 3 | Per-request limit |
| `DX_MAX_CONCURRENT_REQUESTS` | 8 | Per-instance admission limit |
| `DX_MAX_REQUEST_BYTES` | 65536 | Request body limit |
| `DX_SYNC_DEADLINE_S` | 50 | Wait before answering 202 (max 58) |
| `DX_STREAM_HEARTBEAT_S` `DX_STREAM_MAX_S` | 5 / 450 | Stream progress interval and cap (SageMaker's limit is 480) |
| `DX_ALLOW_COMPUTE` | `true` | `false` = read-only endpoint |
| `DX_LOG_LEVEL` `DX_LOG_JSON` `DX_LOG_CALLS` | `INFO` / `true` / `false` | `DX_LOG_CALLS` logs one line per model call (the container image sets it to true) |
| `DX_OUTPUT_NOTICE_LINE_1` `_LINE_2` | screening-signal notice | Wording of the notice in API responses, workbook footers, and `AI_GENERATED_NOTICE.txt` |
| `DX_ALLOW_LOCAL_STORE` `DX_ALLOW_STATIC_CREDENTIALS` `DX_ALLOW_ENDPOINT_OVERRIDE` | `false` | Deliberate production exceptions |

---

## 5. Prerequisites

* An AWS account and a Region where Amazon Bedrock offers your chosen Claude model. Check access in the Bedrock console; a third-party model may need a first-time use-case form or an administrator with AWS Marketplace permissions to enable it.
* Bedrock **quotas** (requests and tokens per minute) for the inference profile you will use. Instances times `DX_LLM_CONCURRENCY` is your peak concurrent call count; `maxTokens` is reserved against the token quota when a call starts.
* A VPC with **two private subnets in different Availability Zones**. Either existing VPC endpoints for `bedrock-runtime`, `logs`, `kms`, `ecr.api`, `ecr.dkr`, and S3 (gateway), or let the foundation stack create them (`CreateVpcEndpoints=true`).
* The four licensed reference files (section 6 and `data/README.txt`).
* Tools: Python 3.10 or later; the AWS CLI v2; and for the image either Docker access in your environment or the CodeBuild path (below).
* Deployer permissions: CloudFormation, IAM role creation and `iam:PassRole`, S3, ECR, KMS, EC2 (security groups, VPC endpoints), SageMaker, Application Auto Scaling, CloudWatch, and CodeBuild if you use it. In SageMaker Code Editor the AWS CLI and `boto3` use the **space's execution role**, so that role needs these.

---

## 6. Develop and test in SageMaker Code Editor

```bash
unzip dx-supporting-lists.zip && cd dx-supporting-lists
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" pyyaml

make test                       # 119 offline tests: no AWS access, no licensed data
make dry-run                    # the whole pipeline on invented sample data with the stub model
```

Try the HTTP service locally (development server, stub model, invented data; nothing leaves the machine):

```bash
dx-support serve-local --backend fake --sample-data --port 8080
curl -s -X POST localhost:8080/invocations -H 'Content-Type: application/json' -d '{"categories":["I48"],"families":["achi","mbs"]}'
```

Put the real reference files in `./data` (they are `.gitignore`d and `.dockerignore`d), then:

```bash
dx-support estimate --show-universe     # what the data contains: units, exclusions, trees
dx-support doctor --offline             # configuration, data, model resolution, local store
dx-support doctor                       # + your AWS identity, the store, and a REAL probe of every configured model
```

`doctor` sends one tiny request per distinct model in the exact shape the pipeline uses (tool call, thinking settings, inference profile, IAM). It costs a fraction of a cent and exits non-zero on failure, so it can gate a deployment. **Run it before the first deployment and after any model change.**

---

## 7. Deploy to production

Set a few shell variables once:

```bash
export AWS_REGION=ap-southeast-2 ENV=prod ACCOUNT=<12-digit account id>
export VPC_ID=vpc-... VPC_CIDR=10.0.0.0/16 SUBNETS=subnet-aaa,subnet-bbb      # private, different AZs
```

**1. Name the models the endpoint may call.** Use the same `DX_*` variables you will give the endpoint (`DX_MODEL`, `DX_INFERENCE_PROFILE_PREFIX`, ...); this prints exact ARNs, least privilege, no wildcards:

```bash
dx-support bedrock-arns --account $ACCOUNT --discover      # --discover asks Bedrock which regions the profile routes to
export BEDROCK_ARNS='<the BedrockResourceArns= value it prints>'
```

**2. Foundation stack** (key, bucket, registry, role, network rules):

```bash
make deploy-foundation ENV=$ENV REGION=$AWS_REGION VPC_ID=$VPC_ID VPC_CIDR=$VPC_CIDR SUBNETS=$SUBNETS BEDROCK_ARNS="$BEDROCK_ARNS" \
     FOUNDATION_EXTRA="CreateVpcEndpoints=true PrivateRouteTableIds=rtb-aaa,rtb-bbb"      # omit the extra if your VPC already has the endpoints
```

**3. Package the reference data** into the SageMaker model artifact (encrypted by the bucket's default key; printed digests belong in your release record):

```bash
make package-data ENV=$ENV REGION=$AWS_REGION DATA_DIR=./data VERSION=2026-10-a
# prints:  ModelDataUrl=s3://dx-support-prod-<account>-<region>/model/2026-10-a/model.tar.gz
```

**4. Build and push the image** (tags are immutable, so each build gets a new tag):

```bash
make image ENV=$ENV REGION=$AWS_REGION                  # local Docker (not every Code Editor space has Docker access)
make image-codebuild ENV=$ENV REGION=$AWS_REGION        # or CodeBuild: no Docker needed
# either prints:  ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/dx-support-prod:<tag>
```

For a hash-pinned install, run `make lock`, commit `requirements.lock`, and rebuild: the Dockerfile uses it whenever it exists.

**5. Endpoint stack** (model, blue/green endpoint, autoscaling, alarms):

```bash
make deploy-endpoint ENV=$ENV REGION=$AWS_REGION IMAGE_URI=<ImageUri> MODEL_DATA_URL=<ModelDataUrl> \
     ENDPOINT_EXTRA="InstanceType=ml.m5.large MinCapacity=2 MaxCapacity=6 DxModel=claude-sonnet-5 InferenceProfilePrefix=au AlarmTopicArn=<optional sns arn>"
```

**6. Verify.**

```bash
dx-support invoke --endpoint $ENV-dx-support --region $AWS_REGION --describe     # version, fingerprint, models in force, limits, load
dx-support invoke --endpoint $ENV-dx-support --region $AWS_REGION --categories H25 --families achi,mbs
```

The first call for a category computes it (watch the progress lines); the second is instant. `describe` shows the **fingerprint**: the namespace the endpoint reads.

**7. Pre-warm** the categories your callers will ask for (section 9), so the endpoint is serving stored results from day one.

**Updating.** Change the image tag, model data, or any parameter and run `make deploy-endpoint` again. SageMaker performs a **blue/green deployment**: the new fleet takes `CanaryPercent` of capacity first, bakes for `CanaryWaitSeconds` while the 5xx and model-error alarms watch it, and **rolls back automatically** if they fire. A change to the model, thinking setting, prompts, tunables, or reference data changes the **fingerprint**, so the updated fleet starts with an empty cache (callers see 202s or streams until it warms). To avoid a cold period, deploy the new configuration first as a **second endpoint stack** with a different `EnvironmentName` (for example `prod-next`; it reuses the foundation stack and the same bucket), run `dx-support prewarm --endpoint prod-next-dx-support ...`, then apply the identical parameters to the main stack: the fingerprints match, so the main endpoint is warm from its first request. Delete the second stack afterwards.

**Rolling back by hand.** Redeploy the previous `ImageUri` / parameters. The previous fingerprint's results are still in S3 (versioned, retained), so they are served again immediately.

**Tearing down.** Delete the endpoint stack, then the foundation stack. The bucket, KMS key, and registry have `DeletionPolicy: Retain`; remove them deliberately when you are sure (empty the versioned bucket first).

---

## 8. Inference guide

### 8.1 Request

`POST /invocations`, `Content-Type: application/json`. All fields are optional except one of `categories` / `codes`. Unknown fields are rejected.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `action` | string | `supporting_lists` | or `describe` |
| `categories` | list of string | | 3-character categories, for example `["H25"]`; at most `DX_MAX_CATEGORIES_PER_REQUEST` |
| `codes` | list of string | | ICD-10-AM codes from your file, mapped to their category |
| `families` | list of string | `achi, mbs, prosthesis` | any of `achi`, `mbs`, `prosthesis`, `dx` |
| `merge_tiers` | list of string | `core, broad` | tiers that count as supporting; `["core"]` is the strict list |
| `include_audit` | bool | `false` | also return **every** verdict (including `not_supporting` and `unclear`) for coder review |
| `prosthesis_extract` | `all` / `latest` | `all` | `latest` keeps only codes seen in the last stacked extract |
| `cache_only` | bool | `false` | never start model calls |
| `stream` | bool | `false` | answer as JSON lines; use `InvokeEndpointWithResponseStream` |
| `request_id` | string | generated | 1-64 characters of `A-Za-z0-9._-`; echoed back and put on every log line |

### 8.2 Response

```jsonc
{
  "request_id": "smoke-1",
  "status": "complete",                      // complete | partial | in_progress
  "fingerprint": "e6b742ec35b9",
  "notice": ["AI-assisted screening output: ...", "A qualified clinical coder must review ..."],
  "results": {
    "I48": {
      "status": "complete",                  // complete | partial | in_progress | not_cached | failed | pending
      "title": "...", "chapter": "IX", "n_member_codes": 3,
      "families": {
        "achi": {
          "state": "complete", "units_reviewed": 10, "n_candidates": 10, "n_items": 2, "counts": {"core": 2},
          "candidates_truncated": false,
          "items": [{"code": "3880100", "description": "...", "support_tier": "core", "evidence_type": "investigated",
                     "member_scope": "all", "rationale": "...", "unit_id": "3880100", "source": "route+lexical"}]
        }
      },
      "audit": []                            // only with include_audit
    }
  },
  "usage": {"cache_hit": true, "llm_calls": 0, "cost_usd": 0.0, "elapsed_ms": 41}
}
```

Prosthesis items also carry `device_type`, `suffixes`, `in_latest_extract`, `label_resolution`, and `n_label_variants`; one verdict yields one item per billing code in the device type. For the `dx` family `code` is the supporting **category** and `evidence_type` is the **relation**. `in_progress` responses add `pending` (categories still running) and `retry_after_s`.

| HTTP | Meaning | What to do |
|---|---|---|
| 200 | `complete`, or `partial` (some family could not finish: see `error`) | `partial`: call again to resume, or inspect with `include_audit` |
| 202 | `in_progress`: work is running on the instance | Call again with the same body after `Retry-After` seconds, or use streaming |
| 400 | Malformed JSON, unknown field or action | Fix the request |
| 413 / 415 | Body too large / not `application/json` | Fix the request |
| 422 | Invalid values (`invalid_category`, `unknown_category`, `context_only_category`, `too_many_categories`, ...) | Fix the request; the body names the problem |
| 429 | This instance is at its request limit | Retry with exponential back-off |
| 503 | Not ready (starting or shutting down) | Retry |
| 500 | Internal error | Retry; quote `request_id` when reporting it |

SageMaker surfaces the container's 4xx and 5xx as a `ModelError` (the original status is in `OriginalStatusCode`); 2xx, including 202, pass through.

### 8.3 Streaming events

`"stream": true` with `InvokeEndpointWithResponseStream` returns `application/jsonlines` over chunked encoding, one JSON object per line:

```
{"event":"accepted","request_id":"...","fingerprint":"...","categories":["H25"],"families":["achi","mbs"],"cache_hit":false,"pending":["H25"]}
{"event":"progress","request_id":"...","elapsed_s":5.0,"categories":{"H25":{"state":"running","llm_calls":4,"cost_usd":0.12,"elapsed_s":5.0}}}
{"event":"result","request_id":"...","status":"complete","results":{...},"usage":{...}}        <- always the last line, or an "error" event
```

If the connection ends without a `result` (the 8-minute cap, a network drop), reconnect with the same request: everything finished so far is stored.

### 8.4 Calling it

**Reference client** (recommended; handles 202, streaming, reconnects, and back-off):

```python
from dx_support.client import EndpointClient
c = EndpointClient("prod-dx-support", region="ap-southeast-2")
out = c.get_lists(categories=["H25"], families=["achi", "mbs"], on_progress=print)
for item in out["results"]["H25"]["families"]["achi"]["items"]:
    print(item["code"], item["support_tier"], item["evidence_type"])
```

**Command line**: `dx-support invoke --endpoint prod-dx-support --region ap-southeast-2 --categories H25,I48 --families achi,mbs [--include-audit] [--json] [--out result.json]`.

**Plain `boto3`** (cached or short work):

```python
import boto3, json
from botocore.config import Config
rt = boto3.client("sagemaker-runtime", region_name="ap-southeast-2", config=Config(read_timeout=70, retries={"max_attempts": 0}))
r = rt.invoke_endpoint(EndpointName="prod-dx-support", ContentType="application/json", Accept="application/json",
                       Body=json.dumps({"categories": ["H25"], "families": ["achi", "mbs"]}))
body = json.loads(r["Body"].read())          # status 202 / body["status"] == "in_progress": ask again, or stream
```

**Plain `boto3`, streaming** (uncached work):

```python
from dx_support.client import iter_ndjson
r = rt.invoke_endpoint_with_response_stream(EndpointName="prod-dx-support", ContentType="application/json",
                                            Body=json.dumps({"categories": ["H25"], "stream": True}))
for event in iter_ndjson(r["Body"]):
    print(event["event"], event.get("elapsed_s"))
```

**AWS CLI**:

```bash
aws sagemaker-runtime invoke-endpoint --region ap-southeast-2 --endpoint-name prod-dx-support \
  --content-type application/json --cli-binary-format raw-in-base64-out \
  --body '{"categories":["H25"],"families":["achi","mbs"]}' result.json && cat result.json
```

Set the client **read timeout to 70 seconds** for regular calls (SageMaker answers within 60), and disable SDK retries so your own retry policy governs.

### 8.5 Caller permissions

Callers need exactly one action on exactly one endpoint (streaming uses the same action):

```json
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
  "Resource": "arn:aws:sagemaker:ap-southeast-2:<account>:endpoint/prod-dx-support"}]}
```

### 8.6 Handling the output

* Treat every text field (`description`, `rationale`, `title`) as **untrusted**: it came from reference files and a model. If you export to a spreadsheet, neutralise values that start with `=`, `+`, `-`, or `@` (`dx_support.text.safe_cell`).
* The result is a **screening signal**. Keep the `fingerprint` with anything you derive from it: it identifies the exact data, prompts, and models behind the verdicts.
* Merge policy: `core + broad` is the default screening list; `core` alone is the strict one. Lists for generic hardware can legitimately be very large.

---

## 9. Batch builds and pre-warming

The same package is the batch tool. Always estimate first, pilot, then scale:

```bash
dx-support estimate --show-universe                         # data facts; no model calls
dx-support estimate --chapter 9                              # projected calls, cost, time (assumptions, not measurements)
dx-support build --dry-run --max-categories 5                # plumbing check with the stub: free
dx-support build --categories H25,I48,K35 --max-total-cost-usd 15        # a real pilot across chapters
dx-support compile                                           # CSVs, full_audit.csv, workbooks: review the audit with a clinical coder
dx-support estimate --calibrate                              # replace assumptions with the pilot's measured calls and cost
dx-support validate-retrieval --n-categories 6               # how much recall does routing lose versus an exhaustive review?
dx-support validate-judge --gold gold.csv                    # judge accuracy against clinician-reviewed pairs (or --n-pairs 300 for repeatability only)
dx-support build --chapter 9 --max-total-cost-usd 150 --yes  # one chapter at a time; re-run the same command to resume
dx-support status --chapter 9 --watch 30                     # read-only progress from another terminal
```

Exit codes of `build`: `0` finished, `2` fatal error, `3` cost cap reached, `4` finished with partial categories. Nothing is spent without an interactive `yes` or `--yes`, and the default cap is a deliberately modest $50. A run continues only under the configuration it started with (the fingerprint is stored in `run_config.json`); `--accept-config-change` overrides that knowingly.

**Pre-warming the endpoint.** Build **into the namespace the endpoint reads** with `--for-service`, using exactly the endpoint's settings. The easiest correct way is `dx-support prewarm`, which reads the endpoint's own SageMaker model (image, environment, role, VPC, data artifact) and submits a Processing job configured from it, so the fingerprints cannot differ:

```bash
dx-support prewarm --endpoint prod-dx-support --region ap-southeast-2 --chapter 7-10 --max-total-cost-usd 400 --print-only   # inspect the request
dx-support prewarm --endpoint prod-dx-support --region ap-southeast-2 --chapter 7-10 --max-total-cost-usd 400
```

A spend cap is mandatory and nothing is chosen for you. The job resumes if stopped, so a build can be spread over several jobs. Afterwards `dx-support status --namespace serve/<fingerprint> --store-uri s3://<bucket>` reports progress, and `dx-support invoke ... --cache-only` confirms categories are served from the store.

Compiled outputs (`dx-support compile`, written to `--output-dir`, default `./output/compiled/<namespace>`):

| File | Contents |
|---|---|
| `supporting_lists.csv` | Every merged (category, code) pair, all families |
| `supporting_achi.csv`, `supporting_mbs.csv`, `supporting_dx.csv` | Per-family lists with rationale (`supporting_dx.csv` only when the diagnosis family was built) |
| `supporting_prostheses.csv` | One row per **billing code** with its device type |
| `full_audit.csv` | **Every** verdict at unit level, including `not_supporting` and `unclear`: the file for coder review |
| `supporting_lists_summary.xlsx` | `README`, `by_category`, `qa_flags`, `ubiquity`, `excluded_summary`, `orphans`, `run_info`. Read `qa_flags` first |
| `dx_crosswalk_profiles.xlsx` | The rule-building input: one tab per code type, nine columns (`icd10_category`, `description`, `code`, `code_description`, four blank statistics columns, `source`) |
| `excluded_codes.csv`, `orphan_codes.csv`, `AI_GENERATED_NOTICE.txt` | Codes never reviewed (with reasons); reviewable codes no list contains; the output notice |

Options: `--merge-tiers core`, `--chapter 9-11`, `--name-suffix TEXT`, `--prosthesis-extract latest`, `--with-tier-columns`, `--no-xlsx`. CSVs carry a BOM so Excel opens them as UTF-8.

---

## 10. Operations

**Logs.** CloudWatch group `/aws/sagemaker/Endpoints/<endpoint>`: one JSON line per request (`request accepted`, `request answered`), per job event, and (with `DX_LOG_CALLS`) per model call: stage, tokens, cost, duration, and error text. No prompts, no model output, no request bodies. Find a request by its `request_id`. Set a retention period on the log group (`aws logs put-retention-policy`); SageMaker creates the group.

**Alarms** (created by the endpoint stack; add `AlarmTopicArn` for notifications):

| Alarm | Meaning |
|---|---|
| `Errors5xxAlarm`, `ModelErrorsAlarm` | Container failures. **Also the automatic-rollback triggers** during an update |
| `BusyAlarm` | Callers are getting 429 or sending bad requests: add capacity or raise `MaxConcurrentRequests` |
| `SlowAlarm` | p99 model latency near the 60-second limit: callers should stream for uncached work |

**Scaling.** Target tracking on `SageMakerVariantInvocationsPerInstance` (default 10 per minute) between `MinCapacity` and `MaxCapacity`; routing is `LEAST_OUTSTANDING_REQUESTS` because requests last up to minutes. Keep at least two instances for availability. Real capacity is set by Bedrock quotas: `instances x DX_LLM_CONCURRENCY` calls in flight. Streaming requests hold a thread for minutes, so size `MaxConcurrentRequests` for your expected simultaneous cold requests; cached requests are fast and plentiful.

**Cost.** Dominated by model tokens. Controls: pre-warm instead of computing on demand, keep `DX_MAX_COST_USD_PER_HOUR` and `DX_MAX_COST_USD_PER_CATEGORY` realistic, set `DX_ALLOW_COMPUTE=false` for a read-only endpoint, and enable `DX_PROMPT_CACHING` for large batch builds after checking it for your model. The model catalog's prices are point-in-time: verify them before budgeting. Cost figures in this system are estimates from token counts, not a bill.

**Changing a model, prompt, or the data** produces a new fingerprint and a cold cache. Pre-warm the new configuration, then cut over (section 7, Updating). Old namespaces stay in S3 until you delete them.

**Troubleshooting**

| Symptom | Likely cause and fix |
|---|---|
| Container exits at start with `Configuration error: ... DX_...` | The named variable is invalid or a production rule failed. Fix the stack parameter |
| `ValidationException` ... "on-demand throughput isn't supported" | A newer model was given as a base id. Use an alias with `DX_INFERENCE_PROFILE_PREFIX`, or an explicit profile id |
| `AccessDeniedException` on `InvokeModel` | `BedrockResourceArns` is missing the profile or a destination region's foundation model: regenerate with `dx-support bedrock-arns --discover` |
| `thinking`, `tool_choice`, or "Extra inputs are not permitted" errors | A model that rejects a setting: adjust `DX_THINKING` or the catalog entry; `dx-support doctor` shows which model |
| Many 202s | Cold categories: pre-warm, or have callers stream |
| `category ... partial` with `error: ... cap` | Per-category or hourly cap reached: raise it deliberately, or re-ask later; progress is kept |
| Endpoint creation hangs, then fails | Image pull or S3 download blocked: check the VPC endpoints and the security group, and `ContainerStartupHealthCheckTimeoutInSeconds` |
| Results look stale after a change | They are not: a changed configuration reads a new namespace. Check `describe.fingerprint` |

---

## 11. Verification status and known limitations

**Verified here** (offline, by the 119 tests and the smoke run): the loaders and cleaning; indexing and routing (including sole-child pass-through); the Bedrock backend against a fake Converse client (request shape, thinking mapping, tool-call parsing by block type, error classification, cost from usage, worst-case refusal); the gate (retries, throttling, adaptive concurrency, caps, fatal stop); the object stores (local, and S3 against a fake client including conditional writes and pagination); resume after a cost cap producing exactly the uninterrupted run's verdicts, and convergence under 30% simulated call failures; the HTTP service (validation, cache hit, 202 and resume, concurrent de-duplication, admission control, streaming, budgets, log hygiene); every CLI command end to end in `--dry-run`, including that a `--for-service` build is served by the service as a cache hit; the templates (structural and security invariants, with negative tests); and the packaging script. A live run of the real server answered over a real socket with genuine chunked streaming.

**Not verified here, and where to check:**

* **Nothing has run against real Amazon Bedrock, SageMaker, S3, ECR, or CodeBuild**, and the container image has not been built or started. The first real checks are `dx-support doctor`, then a pilot (`dx-support build --categories H25,I48,K35 --max-total-cost-usd 15`), then `dx-support validate-retrieval` and `validate-judge`.
* **Model behaviour is unmeasured on Bedrock.** The thinking and request-shape rules in `models.json` come from public documentation (some third-party) and are probed by `doctor`, not proven. Latency and cost per category are **provisional** (`SECONDS_PER_CALL_ESTIMATE`, the prices): the earlier build's measurements (about 19 calls, 160-plus seconds, and a per-category cost that included about 20,000 tokens of wrapper overhead per call) do not transfer. Use `dx-support estimate --calibrate`.
* **Accuracy is unmeasured** until `validate-judge --gold` runs on your coders' reviewed pairs. The default model and thinking setting were validated in the earlier project for this task; the Bedrock path and any other model were not.
* **The CloudFormation templates** were checked by the offline validator, not by `cfn-lint` or a real deploy. Run `cfn-lint` in CI and review the stack change set before executing it.
* **The `au` inference profile** is documented for Claude Sonnet 5 but not for every model in the catalog (Sonnet 5.5 lists `global`, `us`, `eu`).

**Known limitations**

* **Duplicate work across instances is possible.** One instance never computes a category twice at once, but two instances can occasionally compute the same one (costing money, not correctness). Pre-warming largely avoids it.
* **A decision in flight when an instance is replaced is re-bought** on the next call.
* Routing recall is unmeasured until `validate-retrieval` runs; a unit neither the tree walk nor the lexical search surfaces is never judged.
* A judge call can skip candidates; they are re-judged in smaller batches, and any still missing keep the category `partial`.
* MBS has no place-of-service field; prosthesis verdicts are per device type and "latest extract" is a heuristic; 1,718 categories have no title in the file (the model sees member descriptions instead).
* The 8-minute streaming cap means a very slow category may need a reconnect.
* One worker process per container by design (it owns the event loop, limiter, and budget); CPU-bound work (the BM25 index) shares one interpreter, which is fine for an I/O-bound service but not for very high cached-request rates per instance (scale out instead).
* `DX_MODE=exhaustive` on the endpoint would make every cold request about 100 calls per category per family. It is permitted and capped, but meant for batch runs.

---

## 12. Governance

* **A verdict is a screening signal, not a coding determination.** Every verdict, merged or not, stays available (`include_audit`, `full_audit.csv`) for a qualified clinical coder. The diagnosis family is a plausibility screen, not the coding standard's test for an additional diagnosis (treatment, diagnostic interventions, or increased clinical care).
* **No patient-level data is used anywhere.** Prompts contain code descriptions from the reference files only.
* **Output notice.** API responses, every workbook sheet's print footer and properties, and `AI_GENERATED_NOTICE.txt` carry a notice about the nature of the output. Its wording is set by `DX_OUTPUT_NOTICE_LINE_1` and `DX_OUTPUT_NOTICE_LINE_2`, so your organisation's required text can be applied without a code change.
* **Reproducibility.** The `fingerprint` and `run_config.json` record the data digest, prompts, models, and settings behind any result. Keep the SHA-256 list printed by `package_model_data.sh` with the release.

---

## 13. Appendix: old to new

| Earlier | Now |
|---|---|
| `python estimate_scope.py` | `dx-support estimate` |
| `python main_build_lists.py` | `dx-support build` (`--for-service` writes into the endpoint's store) |
| `python compile_results.py` | `dx-support compile` (`--output-dir`) |
| `python run_status.py` | `dx-support status` |
| `python validate_retrieval.py` | `dx-support validate-retrieval` |
| `python validate_judge.py` | `dx-support validate-judge` |
| `python selftest.py` | `make test` (`python -m unittest discover -s tests -t .`) |
| `.env` with `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` | none: IAM role (endpoint) or your AWS identity (CLI) |
| `output/checkpoints/*.jsonl`, `.run.lock` | objects under `runs/<namespace>/` or `serve/<fingerprint>/` in the store; lock object `.run.lock` |
| `--tag exhaustive`, `--dry-run` | `--mode exhaustive` writes to `<namespace>/exhaustive`; `--dry-run` uses the `dry_run` namespace |
| `--debug` (raw SDK messages) | removed: there is no SDK process to inspect; `DX_LOG_CALLS=true` logs per-call metadata |
| `--max-turns`, per-call `max_budget_usd` as an SDK limit | `maxTokens` caps plus the worst-case per-call budget refusal |
| `claude-sonnet-5` etc. | the same aliases; `DX_INFERENCE_PROFILE_PREFIX` selects the Bedrock inference profile |
