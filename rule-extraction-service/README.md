# Rule extraction service

Upcoding and coding-compliance rule extraction for Australian hospital casemix (AR-DRG) data.
Claude Agent SDK agents running on **Amazon Bedrock** research ICD-10-AM, ACHI, the Australian Coding Standards (ACS), and the MBS Schedule, then produce `rule_catalogue.xlsx` rows plus machine-executable rules.
The same code runs as a **SageMaker real-time endpoint** (bring-your-own-container) and as a local CLI.

- [1. At a glance](#1-at-a-glance) · [2. Architecture](#2-architecture) · [3. Project structure](#3-project-structure) · [4. Configuration reference](#4-configuration-reference)
- [5. Security](#5-security) · [6. Prerequisites](#6-prerequisites) · [7. Develop in SageMaker Code Editor](#7-develop-and-test-in-sagemaker-code-editor)
- [8. Build the artifacts](#8-build-the-artifacts) · [9. Deploy](#9-deploy) · [10. Inference](#10-inference) · [11. Operations](#11-operations)
- [12. Migration notes and assumptions](#12-migration-notes-and-assumptions-to-verify) · [13. Verification status](#13-verification-status) · [14. Known limitations](#14-known-limitations)

## 1. At a glance

- **One image, three entry points:** `serve` (endpoint), `worker` (queue consumer only), `cli ...` (plan, run, corpus, package, evaluate, contract-check).
- **Claude comes from Bedrock** through the IAM execution role. No API keys, OAuth tokens, or secrets exist anywhere in the repo, image, or endpoint environment.
- **The endpoint API is submit/poll.** SageMaker real-time requests are capped at 60 seconds (8 minutes when streaming) and your extraction units take 4 to 8 minutes each, so every operation returns in milliseconds and the work runs on a durable queue.
- **Durable and horizontally scalable:** DynamoDB (job and unit state with lease-based claims), SQS (work items plus DLQ), S3 (artifacts, SSE-KMS). Add instances to add throughput.
- **Human review is preserved:** agents can never write `confidence = verified`; every new row is `needs_review`.
- **Production hardening built in:** prompt-injection defences, least-privilege tool policy, hash-locked installs, model-artifact integrity verification, graceful SIGTERM hand-back, blue/green rollout with alarm-based rollback.
- **214 unit tests** run with no AWS access (see [13. Verification status](#13-verification-status) for what they do and do not prove).

## 2. Architecture

```
  client (scripts/invoke_endpoint.py, boto3, your app)
        │  SigV4 + IAM: sagemaker:InvokeEndpoint
        ▼
┌──────────────────────── SageMaker real-time endpoint (N instances) ─────────────────────────┐
│  supervisor (PID 1, root only to start children)                                            │
│    start-up: validate config → verify model.tar.gz checksums → SDK preflight → mark ready   │
│    ├─ web    gunicorn + Flask (uid 10001)  GET|POST /ping, POST /invocations  (milliseconds)│
│    └─ worker asyncio (uid 10001): claims units, runs Claude Agent SDK sessions, finalizes   │
└───────┬───────────────┬───────────────┬────────────────┬──────────────────┬─────────────────┘
        ▼               ▼               ▼                ▼                  ▼
   DynamoDB          SQS work queue   S3 artifacts    Bedrock Runtime   CloudWatch metrics/logs
   jobs, units,      (+ DLQ)          (SSE-KMS)       (Claude, au.*     /opt/ml/model (read-only):
   leases (CAS)                                       profiles)         corpus, template, manifest
```

### 2.1 Components

| Component | Module | Responsibility |
|---|---|---|
| Supervisor | `service/supervisor.py` | Fail-fast start-up checks; starts web and worker as an unprivileged user; exits non-zero if either dies; forwards SIGTERM and escalates to SIGKILL inside SageMaker's 30 s window |
| Web | `service/app.py`, `handlers.py` | The SageMaker contract; strict request schemas; JSON errors that never leak internals |
| Job service | `service/jobs.py` | submit, get, cancel, retry, resume, finalize; idempotency keys; single-winner finalization |
| Worker | `service/worker.py` | Receives work items, claims each unit with a compare-and-swap lease, heartbeats, handles cancel, deadline and budget, hands units back on shutdown |
| Janitor | `service/janitor.py` | Repairs half-submitted jobs, exhausted leases, missed finalizations, and deadline overruns |
| Unit processor | `pipeline/processor.py` | Builds prompts and the tool profile per unit, runs the agent with retries, persists rules |
| Agent runner | `agents/sdk_runner.py` | One stateless `query()` per unit against Bedrock; fail-closed option checks; per-session throw-away HOME |
| Collector | `agents/collector.py` | `emit_rule` validation boundary (schema, citation, shape, DSL, duplicates, caps) |
| Permission policy | `agents/permissions.py` | Tool allow-list; corpus path confinement; https host allow-list |
| Finalizer | `pipeline/finalize.py` | Dedupe, append to the template workbook, remap executable-rule ids, summary |
| Engine | `engine/` | Safe JSON condition/action DSL, interpreter, grouper port, compare loop |
| Ingestion | `ingestion/` | PDFs and MBS XML to a page-indexed, read-only corpus plus code registry |
| Model package | `model_package.py` | Deterministic `model.tar.gz` with a SHA-256 manifest; verification at start-up |

### 2.2 Why submit/poll on a real-time endpoint

| Constraint (verified in AWS docs, October 2026) | Consequence |
|---|---|
| Real-time invocation: 60 s for regular responses, 8 min for streaming, not adjustable | A single ACS unit (273 to 485 s measured) cannot run inside one request |
| Payload up to 25 MB | Requests here are tiny (options only); results are fetched from S3 or paged |
| `/ping` must answer within 2 s; start-up health window defaults to 8 min (configured to 15) | `/ping` reflects local process health only, never Bedrock or DynamoDB reachability |
| SIGTERM then SIGKILL after 30 s on scale-in or update | In-flight units are cancelled and handed back (state `PENDING`, queue visibility 0) within `SHUTDOWN_GRACE_SECONDS` (default 20) |
| Endpoints bill per instance-hour while `InService` | For rare, full catalogue builds consider SageMaker Asynchronous Inference or a Processing job: **the same image works for both** (`serve` or `cli run`) |

### 2.3 Job and unit lifecycle

```
submit_job ─► SUBMITTING ─► RUNNING ─► FINALIZING ─► SUCCEEDED   all units succeeded
                              │  ▲                    PARTIAL     some succeeded
              cancel_job ─────┘  └── retry_failed     FAILED      none succeeded (or finalization error)
              (flag; workers abort in-flight)         CANCELLED   cancel requested
```

| Unit state | Meaning |
|---|---|
| `PENDING` | Waiting for a worker (message on the queue) |
| `RUNNING` | Claimed with a lease (`LEASE_SECONDS`, extended by heartbeat every lease/3) |
| `SUCCEEDED` | Rules persisted to `jobs/<job_id>/units/<unit_id>.jsonl` (an empty file is a valid result) |
| `FAILED` | Retries exhausted, non-transient error, lease budget exhausted, or deadline exceeded |
| `CANCELLED` | Job cancelled or budget reached before the unit ran |

Unit kinds: `acs_topic` (one ACS standard; tools: Read, Grep, Glob confined to the corpus, plus `emit_rule`), `sweep` (a page window; tool: `emit_rule` only), `web_research` (allow-listed WebFetch; opt-in), `contract_check` (SDK and Bedrock diagnostics).

### 2.4 Failure handling

| Scenario | Behaviour |
|---|---|
| Transient model or API error (throttling, 5xx, timeout) | Up to `MAX_RETRIES + 1` attempts with exponential backoff (base 5 s, jitter 0.5 to 1.5×, cap 120 s) |
| Agent hits the turn limit | Retried with `max_turns × 1.5 + 1` (cap `MAX_TURNS_CAP`); timeout is `max(AGENT_TIMEOUT_SECONDS, turns × SECONDS_PER_TURN)` |
| Non-transient error (bad request, access denied, budget) | Unit `FAILED` immediately; the error text is kept (redacted, truncated) |
| Instance crash mid-unit | Lease and SQS visibility expire; another worker claims the unit (attempts +1); after `MAX_RETRIES + 1` claims the unit is `FAILED` |
| Duplicate or late SQS delivery | Claim is a CAS: duplicates of finished work are acknowledged and ignored; live leases are respected |
| Scale-in, deploy, or SIGTERM | Sessions cancelled, units returned to `PENDING`, messages released immediately |
| Poison message | Ignored and hidden 15 min; moves to the DLQ after 10 receives (alarm on DLQ depth) |
| Job past `JOB_MAX_RUNTIME_SECONDS` | Janitor and workers fail remaining units (`job deadline exceeded`) and finalize |
| Spend guard | `max_cost_usd` stops dispatching new units once the SDK-reported estimate is reached; `MAX_BUDGET_USD_PER_UNIT` caps each session |
| Finalization crash | Janitor re-opens `FINALIZING` jobs older than 15 min; outputs use deterministic keys so reruns are idempotent; raw rules are written first |
| Partial results | `PARTIAL` jobs still deliver workbook, rules, and summary; `retry_failed` re-runs only failed units |

### 2.5 Data model

| Store | Key | Content |
|---|---|---|
| DynamoDB `PK=JOB#<id>`, `SK=META` | `gsi1pk=JOBSTATUS#<status>` | Job record (options, plan, cost, artifacts, summary) as one JSON attribute plus `version` for compare-and-swap; TTL `JOB_RETENTION_DAYS` |
| DynamoDB `PK=JOB#<id>`, `SK=UNIT#<unit_id>` | | Unit record: status, attempts, lease owner and expiry, cost, turns, error |
| DynamoDB `PK=IDEM#<key>`, `SK=META` | | Idempotency binding to a job id and options digest |
| SQS message | | `{"v":1,"job_id":"…","unit_id":"…"}` only; never document text |
| S3 `jobs/<id>/units/<unit>.jsonl` | | Rules (ledger record plus optional executable rule) emitted by one unit |
| S3 `jobs/<id>/outputs/` | | `rule_catalogue.xlsx`, `executable_rules.jsonl`, `rules_final.jsonl`, `rules_raw.jsonl`, `mbs_billing_facts_reference.jsonl`, `units.jsonl`, `summary.json`, `contract_report.json` |
| `/opt/ml/model` (read-only) | | `manifest.json`, `corpus/`, `templates/rule_catalogue.xlsx`, optional `base/executable_rules.jsonl`, `rules/executable_rules.jsonl` |

## 3. Project structure

```
rule-extraction-service/
├── README.md
├── pyproject.toml            package metadata, pinned runtime deps, tool config
├── Makefile                  install, lock, test, lint, corpus, package, image, foundation, upload-model
├── Dockerfile                multi-stage, hash-locked install, non-root drop, exec-form entrypoint
├── buildspec.yml             CodeBuild spec for environments without a Docker daemon
├── .env.example              non-secret local settings (no credentials)
├── configs/base.yaml         structured defaults baked into the image (footer, web hosts, source overrides)
├── docker/entrypoint.sh      handles `serve` (SageMaker), `worker`, `cli`
├── requirements/             base.in, dev.in (→ prod.lock via `make lock`; commit prod.lock)
├── infra/cloudformation/
│   ├── foundation.yaml       KMS, S3, DynamoDB, SQS+DLQ, ECR, least-privilege execution role
│   └── endpoint.yaml         Model, EndpointConfig, blue/green Endpoint, alarms, optional auto scaling
├── scripts/
│   ├── build_and_push.sh     docker or sm-docker (CodeBuild) image build and push
│   ├── deploy.py             stack create/update, outputs, model upload, scale
│   └── invoke_endpoint.py    client: plan, submit, status, wait, rules, results, cancel, retry, raw
├── source_docs/              your licensed PDFs, MBS XML, rule_catalogue.xlsx (git-ignored)
├── src/rule_extractor/
│   ├── settings.py           typed, validated config (env + YAML)       logging_setup.py   JSON logs + redaction
│   ├── errors.py, util.py    error hierarchy, safe helpers              model_package.py   model.tar.gz + integrity
│   ├── cli.py, __main__.py   `rule-extractor` command
│   ├── domain/               schema.py (Rule), jobs.py (records), api.py (strict request models)
│   ├── engine/               code_match, episode, conditions, actions, grammar, grouper, compare, rules_store
│   ├── ingestion/            pdf, toc, chunking, mbs, corpus, sources
│   ├── agents/               prompts.py + prompt_templates/*.md, collector.py, permissions.py,
│   │                         sdk_runner.py, contract_check.py, types.py
│   ├── llm/bedrock.py        model-id resolution and SDK environment for Bedrock
│   ├── pipeline/             planner, processor, dedupe, finalize, local_run
│   ├── output/               excel_writer.py (template append, formula-injection defence), jsonl.py
│   ├── storage/              interfaces + memory, local, aws (DynamoDB, S3, SQS)
│   └── service/              app, handlers, jobs, worker, janitor, supervisor, worker_main, wsgi, health, metrics, container
└── tests/                    214 tests: engine, ingestion, writer, storage, Bedrock env, agents, SDK runner, pipeline,
                              service lifecycle, HTTP contract, supervisor, CLI, templates, repo hygiene
```

## 4. Configuration reference

Configuration precedence: built-in defaults, then `configs/base.yaml` (`RULE_EXTRACTOR_CONFIG_FILE`), then environment variables. Invalid values stop start-up with every problem listed.
On SageMaker the CloudFormation template sets these in the Model `Environment`; the execution role supplies AWS credentials.

| Variable | Default | Meaning |
|---|---|---|
| `AWS_REGION` / `RULE_EXTRACTOR_BEDROCK_REGION` | none (required) | Region for Bedrock, DynamoDB, S3, SQS, CloudWatch |
| `RULE_EXTRACTOR_ENV` | `dev` | `prod` also blanks `AWS_BEARER_TOKEN_BEDROCK` for the CLI (role only) |
| `RULE_EXTRACTOR_BACKEND` | `local` | `aws` (DynamoDB, SQS, S3) or `local` (files and memory) |
| `RULE_EXTRACTOR_CONFIG_FILE` | none | YAML file with structured defaults |
| `RULE_EXTRACTOR_LOG_LEVEL` | `INFO` | Log level for the service and CLI |
| `RULE_EXTRACTOR_BEDROCK_PROFILE_PREFIX` | `au` | Inference profile geography: `au`, `apac`, `us`, `eu`, `jp`, `global`, or empty for base ids |
| `RULE_EXTRACTOR_MODEL_STRONG` | `anthropic.claude-sonnet-5` | ACS topics and Tabular sweeps (prefix is applied automatically) |
| `RULE_EXTRACTOR_MODEL_FAST` | `anthropic.claude-haiku-4-5-20251001-v1:0` | Alphabetic sweeps |
| `RULE_EXTRACTOR_MODEL_WEB` | strong model | Web research (opt-in) |
| `RULE_EXTRACTOR_BEDROCK_PROVIDER` | `invoke` | `invoke` (bedrock-runtime) or `mantle` (Messages API; no invocation logging or Guardrails) |
| `RULE_EXTRACTOR_BEDROCK_SERVICE_TIER` | `default` | `default`, `flex`, or `priority` |
| `RULE_EXTRACTOR_EFFORT` | none | Passed to the CLI as `CLAUDE_CODE_EFFORT_LEVEL` |
| `RULE_EXTRACTOR_CONCURRENCY` | `3` | Concurrent agent sessions per instance |
| `RULE_EXTRACTOR_MAX_RETRIES` | `4` | In-process retries per unit (also bounds claims per unit) |
| `RULE_EXTRACTOR_RETRY_BASE_DELAY_SECONDS` | `5` | Backoff base |
| `RULE_EXTRACTOR_AGENT_TIMEOUT_SECONDS` | `480` | Floor for the per-attempt timeout |
| `RULE_EXTRACTOR_SECONDS_PER_TURN` | `25` | Timeout budget per turn |
| `RULE_EXTRACTOR_MAX_TURNS` | `40` | Turn budget per attempt |
| `RULE_EXTRACTOR_MAX_TURNS_CAP` | `100` | Ceiling when retries escalate turns |
| `RULE_EXTRACTOR_MAX_BUDGET_USD_PER_UNIT` | `5.0` | Per-session spend cap (`0` disables) |
| `RULE_EXTRACTOR_MAX_RULES_PER_UNIT` | `40` | Cap on `emit_rule` calls per unit |
| `RULE_EXTRACTOR_REQUIRE_VERBATIM_PROMPTS` | `true` | Refuse to run unless the SDK supports `verbatim_prompts` (SDK ≥ 0.2.158) |
| `RULE_EXTRACTOR_TABULAR_PAGES_PER_CHUNK` / `_TABULAR_PAGE_OVERLAP` | `6` / `1` | Tabular sweep windows |
| `RULE_EXTRACTOR_ALPHABETIC_PAGES_PER_CHUNK` / `_ALPHABETIC_PAGE_OVERLAP` | `10` / `1` | Alphabetic sweep windows |
| `RULE_EXTRACTOR_MIN_ACS_STANDARDS` | `100` | `corpus build` fails closed below this many located standards |
| `RULE_EXTRACTOR_MODEL_DIR` | `/opt/ml/model` | Model artifact root (corpus, template, manifest) |
| `RULE_EXTRACTOR_CORPUS_DIR` | `<model dir>/corpus` | Page-indexed corpus |
| `RULE_EXTRACTOR_SOURCE_DOCS_DIR` | `./source_docs` | Licensed inputs for `corpus build` |
| `RULE_EXTRACTOR_OUTPUT_DIR` / `_STATE_DIR` | `./sample_output` / `./checkpoints` | Local-run outputs and resume state |
| `RULE_EXTRACTOR_WORK_DIR` | `/tmp/rule-extractor` | Scratch space (session dirs, health files) |
| `RULE_EXTRACTOR_TEMPLATE_PATH` | auto | Workbook the writer copies from (default `model dir/templates/…`, then `source_docs/…`) |
| `RULE_EXTRACTOR_BASE_RULES_PATH` | auto | Executable rules already in the base workbook (default `model dir/base/…`) |
| `RULE_EXTRACTOR_STATE_TABLE` / `_QUEUE_URL` / `_ARTIFACT_BUCKET` | none (required for `aws`) | DynamoDB table, SQS URL, S3 bucket |
| `RULE_EXTRACTOR_ARTIFACT_PREFIX` | `rule-extractor` | S3 key prefix |
| `RULE_EXTRACTOR_KMS_KEY_ID` | none | CMK for S3 writes (ARN) |
| `RULE_EXTRACTOR_METRICS_NAMESPACE` / `_ENDPOINT_NAME` | `RuleExtractor` / `rule-extractor` | CloudWatch namespace and `EndpointName` dimension |
| `RULE_EXTRACTOR_ALLOWED_OPERATIONS` | all except `evaluate_episodes` | Comma list; others return 403 |
| `RULE_EXTRACTOR_ENABLE_EVALUATE` | `false` | Enables `evaluate_episodes` (processes patient-level data; see 5.3) |
| `RULE_EXTRACTOR_MAX_EPISODES_PER_REQUEST` | `50` | Bound for `evaluate_episodes` |
| `RULE_EXTRACTOR_GROUPER_FACTORY` | none (mock grouper) | `module:callable` returning an object with `group(episode)` |
| `RULE_EXTRACTOR_ENABLE_WEB_RESEARCH` | `false` | Allows `include_web_research`; needs egress to allow-listed hosts |
| `RULE_EXTRACTOR_WEB_ALLOWED_HOSTS` | `www.ihacpa.gov.au,ihacpa.gov.au` | WebFetch host allow-list (https only) |
| `RULE_EXTRACTOR_EXCEL_SHEET_NAME` / `_EXCEL_FOOTER_ENABLED` | `rule_catalogue` / `true` | Workbook sheet; footer applies only if `configs/base.yaml` lists footer lines |
| `RULE_EXTRACTOR_MAX_REQUEST_BYTES` | `1048576` | Request size cap |
| `RULE_EXTRACTOR_MAX_UNITS_PER_JOB` | `600` | Plan size cap |
| `RULE_EXTRACTOR_JOB_RETENTION_DAYS` | `90` | DynamoDB TTL for job and unit records |
| `RULE_EXTRACTOR_LEASE_SECONDS` | `600` | Unit lease and initial SQS visibility |
| `RULE_EXTRACTOR_JOB_MAX_RUNTIME_SECONDS` | `43200` | Job deadline (12 h) |
| `RULE_EXTRACTOR_RUN_AS_UID` | `10001` | Unprivileged uid for web and worker |
| `RULE_EXTRACTOR_WEB_WORKERS` / `_WEB_THREADS` | `2` / `4` | gunicorn processes and threads |
| `RULE_EXTRACTOR_SHUTDOWN_GRACE_SECONDS` | `20` | Drain budget inside SageMaker's 30 s window |

## 5. Security

### 5.1 Threats and controls

| Threat | Control |
|---|---|
| Prompt injection through PDF or web text | Untrusted text only in the **user** prompt, between random-nonce markers; single-pass rendering (inserted text is never re-templated); SDK `verbatim_prompts=True` (no `@path` expansion, no slash commands); injection-defence rules in every system prompt; tools limited to what the unit needs |
| A hijacked model tries to act | `tools=` restricts built-ins; `disallowed_tools` removes Bash, Write, Edit, WebSearch, Task, and more; `can_use_tool` gate denies by default and confines reads to the corpus root; `emit_rule` only records data; `bypassPermissions` is never used; only the in-process tool is pre-approved so the gate cannot be shadowed |
| Poisoned or hallucinated output | `emit_rule` validates schema, lengths, citation, rule shape, executable grammar, duplicates, per-unit cap; unknown codes are flagged against the corpus registry; provenance fields cannot be spoofed; `confidence` forced to `needs_review` |
| Code execution through rules | Rules are a fixed JSON grammar interpreted by hand-written code with depth and size limits; no `eval`/`exec` anywhere (enforced by a test) |
| Spreadsheet formula injection | Cells starting `=`, `+`, `-`, `@`, tab, or CR are neutralised and forced to text |
| Malicious PDF or XML | Size and page limits; text sanitised (controls and bidi overrides); XML parsed without entity expansion, DTD loading, or network (verified against an XXE sample) |
| Tampered corpus or template | `model.tar.gz` manifest with SHA-256 per file; container refuses to start on mismatch, extra file, symlink, or special file |
| Malicious API requests | IAM-authenticated invocation only; strict schemas, unknown fields rejected; ids validated; 1 MB default body cap; operation allow-list; no paths or URLs accepted from callers |
| Credential exposure | IAM role only; the CLI environment blanks `ANTHROPIC_*` and OAuth variables; `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`; logs redact secrets; payloads and prompts are never logged; tests scan the repo for secret patterns |
| Supply chain | `--require-hashes` installs from `prod.lock`; pinned SDK; immutable ECR tags with scan-on-push; mirror dependencies in CodeArtifact because upstream removes old SDK releases |
| Data leakage at rest and in transit | SSE-KMS (CMK, rotation) on S3, DynamoDB, SQS, and the endpoint volume; TLS-only bucket and queue policies; PITR on the table; licensed documents excluded from git and Docker context |
| Privilege escalation in the container | Supervisor drops to uid 10001 for the web, worker, and the Claude CLI; no Bash tool; throw-away per-session HOME removed after each unit |

### 5.2 Decisions with options (safest default first)

| Decision | Options and trade-offs | Default |
|---|---|---|
| Inference geography | **`au`** keeps inference in Australian regions (the foundation IAM policy enforces the same scope); `apac` and `us`/`eu` widen the destination regions; `global` gives the highest throughput but may route anywhere. Sonnet 5.5 had no `au.` profile when checked (October 8, 2026) | `au` |
| Tool permission strategy | **`default` mode + `can_use_tool` gate + `disallowed_tools`** (implemented; matches what worked in your 0.2.152 debugging); `dontAsk` is documented as fail-closed but your repro notes say it skips the callback and MCP tools were denied; `bypassPermissions` is refused | gate |
| Container user | **Start as root, drop to uid 10001** (SageMaker expects root at start); pure non-root `USER` risks volume permission problems; running everything as root is the AWS-documented norm but weakest | drop |
| Bedrock provider | **bedrock-runtime** supports model invocation logging and Guardrails; Mantle gives Messages-API parity but neither | `invoke` |
| Web research | **Off.** Opt-in is allow-listed WebFetch only; it needs egress to the allow-listed hosts plus the CLI's hostname safety check to `api.anthropic.com`. WebSearch does not exist on Bedrock | off |
| Dependency installs | **Hash-locked only** (build fails without `prod.lock`); no unlocked fallback is offered | locked |
| Network isolation | Endpoint runs in your VPC with VPC endpoints; `EnableNetworkIsolation` must stay `false` because the container calls Bedrock, DynamoDB, SQS, and S3 | VPC endpoints |

### 5.3 Data handling

- **Licensed content:** the source PDFs and everything derived from them (corpus, model artifact) are IHACPA/AIHW-licensed. Keep the bucket private, KMS-encrypted, in-account; never commit them.
- **Extraction path has no patient data.** Only the optional `evaluate_episodes` operation accepts episode-level data. It is **off by default**, never logs bodies, and echoes only `episode_id`. Enable it only after your privacy review, and use the real grouper plug-in rather than the mock.
- **Retention:** DynamoDB TTL (`JOB_RETENTION_DAYS`), S3 noncurrent versions expire after 90 days, logs after `LogRetentionDays`.

### 5.4 Network requirements

No internet egress is needed with web research off. Required VPC endpoints (private DNS on) in the endpoint subnets:

`bedrock-runtime`, `bedrock` (control plane; profile discovery), `sqs`, `logs`, `monitoring` (CloudWatch metrics), `ecr.api`, `ecr.dkr`, `s3` (gateway), `dynamodb` (gateway). The security group needs egress 443 only.

## 6. Prerequisites

- AWS account with **Bedrock model access** enabled for the Claude models you use, and the one-time Anthropic first-use form submitted. Verify: `aws bedrock list-inference-profiles --region <region> --query "inferenceProfileSummaries[?contains(inferenceProfileId,'au.anthropic')].inferenceProfileId"`.
- Your **deployer role** (the Code Editor execution role) needs: CloudFormation, ECR push, S3 upload to the artifact bucket, `kms:GenerateDataKey` on the key, `iam:PassRole` on the endpoint role, `sagemaker:*Endpoint*`/`CreateModel`/`CreateEndpointConfig`, and `sagemaker:InvokeEndpoint`.
- Private subnets in at least two Availability Zones, a security group, and the VPC endpoints from 5.4.
- Source files in `source_docs/`: the five PDFs, `MBS-XML-<date>.XML`, and `rule_catalogue.xlsx`. File names are overridable via `sources:` in `configs/base.yaml`.
- A way to build the image: Docker in Studio (enable `DockerSettings.EnableDockerAccess` on the domain and restart the app), **or** `sm-docker` (CodeBuild; its role needs push access to the ECR repository), **or** the supplied `buildspec.yml`.

## 7. Develop and test in SageMaker Code Editor

```bash
git clone <your repo> rule-extraction-service && cd rule-extraction-service
make install                 # .venv, runtime + dev deps, editable install
make test                    # 214 tests, no AWS access needed
export AWS_REGION=<region>   # credentials come from the Studio execution role (or `aws sso login`)
```

Local dry-run and smoke tests (these call Bedrock and cost money; start small):

```bash
make corpus                                   # build + verify corpus (compare counts, see section 12)
rule-extractor plan                           # free: shows units (ACS topics + sweep windows)
rule-extractor contract-check                 # live: proves tool dispatch, nested args, denial, both model tiers
rule-extractor run --limit-topics 3 --limit-sweep-windows 3 --max-cost-usd 10
rule-extractor run                            # resumable: re-run the same command after an interruption
```

`run` uses the same worker, processor, and finalizer as production with file-backed state (`checkpoints/state.json`). Outputs land in `sample_output/jobs/<job_id>/outputs/`, with `rule_catalogue.xlsx`, `executable_rules.jsonl`, and `mbs_billing_facts_reference.jsonl` also copied to `sample_output/`.

## 8. Build the artifacts

### 8.1 Model artifact (corpus + template + manifest)

```bash
make corpus                   # corpus/ (page text, acs_toc.json, known_codes.json, MBS lookup and facts)
make package                  # dist/model.tar.gz (deterministic; prints sha256)
rule-extractor check-model --model-dir build/model     # re-verify the staged directory
```

Options: `--base-rules <jsonl>` (executable rules already in your base workbook, for example RULE_049 and RULE_050; continue numbering by using that workbook as `--template`), `--evaluation-rules <jsonl>` (served by `evaluate_episodes`).

### 8.2 Dependency lock and image

```bash
make lock                                     # requirements/prod.lock with hashes (use your CodeArtifact/PyPI mirror); commit it
ECR_REPOSITORY=<RepositoryName output> make image                       # Docker daemon available
ECR_REPOSITORY=<RepositoryName output> BUILDER=sm-docker make image     # no daemon: builds in CodeBuild
# Studio with Docker access:  DOCKER_BUILD_ARGS="--network sagemaker" make image
```

Tags are immutable in ECR (default tag: short git sha; override with `IMAGE_TAG`). The script prints `IMAGE_URI=…`.

## 9. Deploy

```bash
export AWS_REGION=<region>
FOUNDATION=rule-extractor-dev-foundation

# 1. Foundation: KMS, S3, DynamoDB, SQS+DLQ, ECR, execution role
python scripts/deploy.py stack --stack-name $FOUNDATION --template infra/cloudformation/foundation.yaml \
  --param ProjectName=rule-extractor --param Environment=dev
eval "$(python scripts/deploy.py outputs --stack-name $FOUNDATION | python -c 'import json,sys; [print(f"export {k}={v}") for k,v in json.load(sys.stdin).items()]')"

# 2. Image and model artifact
ECR_REPOSITORY=$RepositoryName make image                  # note IMAGE_URI
make package
python scripts/deploy.py upload-model --file dist/model.tar.gz --foundation-stack $FOUNDATION   # prints ModelDataUrl + ModelVersion

# 3. Endpoint (blue/green with auto-rollback on 5xx alarm)
python scripts/deploy.py stack --stack-name rule-extractor-dev-endpoint --template infra/cloudformation/endpoint.yaml \
  --param EndpointName=rule-extractor-dev --param ModelVersion=<ModelVersion> \
  --param ImageUri=<IMAGE_URI> --param ModelDataUrl=<ModelDataUrl> --param ExecutionRoleArn=$ExecutionRoleArn \
  --param SubnetIds=subnet-aaa,subnet-bbb --param SecurityGroupIds=sg-xxx \
  --param StateTableName=$StateTableName --param QueueUrl=$QueueUrl --param DlqName=$DlqName \
  --param ArtifactBucket=$ArtifactBucket --param KmsKeyArn=$KmsKeyArn
```

**Verify the deployment (do this every time):**

```bash
python scripts/invoke_endpoint.py --endpoint rule-extractor-dev health
echo '{"operation":"submit_job","options":{"kind":"contract_check"}}' | python scripts/invoke_endpoint.py --endpoint rule-extractor-dev raw
python scripts/invoke_endpoint.py --endpoint rule-extractor-dev wait <job_id>        # expect SUCCEEDED
python scripts/invoke_endpoint.py --endpoint rule-extractor-dev results <job_id>     # artifacts.contract_report -> aws s3 cp <uri> -
```

**Update, rollback, scale**

| Task | How |
|---|---|
| New code or model | Build a new image tag and artifact; redeploy the endpoint stack with a **new `ModelVersion`** (immutable Model/Config; SageMaker shifts traffic blue/green and rolls back if `Invocation5XXErrors` alarms) |
| Roll back | Redeploy with the previous `ImageUri`, `ModelDataUrl`, and `ModelVersion` |
| More throughput | `python scripts/deploy.py scale --endpoint rule-extractor-dev --instances 4`, or set `EnableAutoScaling=true` (target-tracks `WorkerSlotUtilization`; scale-in is disabled by default because it interrupts running units) |
| New SDK or CLI version | Bump the pin, `make lock`, rebuild, deploy, then run the contract-check job |
| Stop spending | Delete the endpoint stack (the foundation stack retains data); cancel running jobs first with `cancel` |

## 10. Inference

### 10.1 Request and response envelope

`POST /invocations` (SageMaker `InvokeEndpoint`), `Content-Type: application/json`:

```json
{ "operation": "submit_job", "idempotency_key": "catalogue-2026-q4", "options": { "topics": ["0110"], "include_tabular_sweep": false } }
```

Success: `{"ok": true, "operation": "...", "request_id": "...", "data": { ... }}`. Error: `{"ok": false, "request_id": "...", "error": {"code": "...", "message": "...", "details": [...]}}`.
Set a short `CustomAttributes` header if you want a correlation id (sanitised; the response returns the server `request_id`). Use a socket timeout of 70 s.

### 10.2 Operations

| Operation | Fields | Returns |
|---|---|---|
| `plan` | `options` | Unit counts by kind and document, preview of the first 25 units (no state written) |
| `submit_job` | `options`, `idempotency_key` (optional, 8 to 128 chars) | `job_id`, `status`, `units_total`, `plan`, `deduplicated` |
| `get_job` | `job_id`, `include_units`, `unit_status`, `limit` (≤ 600) | Status, `progress` counts, cost, failures (first 20), artifacts, summary |
| `list_jobs` | `status`, `limit` (≤ 50) | Recent jobs |
| `get_rules` | `job_id`, `offset`, `limit` (≤ 100) | Page of final rules (`rule` ledger record plus `executable`), `next_offset` |
| `get_results` | `job_id` | Artifact URIs, summary, error |
| `cancel_job` | `job_id` | Sets the cancel flag; workers abort in-flight units |
| `retry_failed` | `job_id` (job must be `PARTIAL` or `FAILED`) | Resets failed units to `PENDING` and re-enqueues only them |
| `health` | none | Versions (service, prompt digest, SDK, models, corpus digest), queue depth, public settings |
| `evaluate_episodes` | `episodes` (≤ `MAX_EPISODES_PER_REQUEST`), `rule_ids` | **Disabled by default.** Flagged rules, correction preview, cost-weight delta; `grouper.is_mock` tells you if the proxy grouper was used |

`options` (mirrors the original CLI): `kind` (`extraction` or `contract_check`), `topics`, `limit_topics`, `include_acs_topics`, `include_tabular_sweep`, `sweep_documents`, `limit_sweep_windows`, `include_web_research`, `web_research_topics`, `max_cost_usd`. Unknown fields are rejected.

### 10.3 Errors

| HTTP | `code` | Typical cause |
|---|---|---|
| 400 | `invalid_request`, `invalid_json` | Schema violation (details list every problem), unknown operation or field |
| 403 | `forbidden_by_policy` | Operation disabled (`evaluate_episodes`, web research, allow-list) |
| 404 | `not_found` | Unknown job id |
| 409 | `conflict` | Idempotency key reused with different options; retrying a job that is not `PARTIAL`/`FAILED`; rules not ready |
| 413 / 415 | `payload_too_large`, `unsupported_media_type` | Body over `MAX_REQUEST_BYTES`; not `application/json` |
| 500 | `internal_error` | Quote the `request_id`; details are only in the logs |
| 503 | `unavailable` | Corpus or rules not packaged in this artifact |

### 10.4 Typical session

```bash
EP=rule-extractor-dev
python scripts/invoke_endpoint.py --endpoint $EP plan --limit-topics 3 --limit-sweep-windows 3
python scripts/invoke_endpoint.py --endpoint $EP submit --limit-topics 3 --limit-sweep-windows 3 --idempotency-key smoke-001 --max-cost-usd 15
python scripts/invoke_endpoint.py --endpoint $EP wait <job_id> --interval 30
python scripts/invoke_endpoint.py --endpoint $EP rules <job_id> --limit 50
aws s3 cp s3://<bucket>/rule-extractor/jobs/<job_id>/outputs/rule_catalogue.xlsx .
```

Python:

```python
import json, boto3
from botocore.config import Config
rt = boto3.client("sagemaker-runtime", config=Config(read_timeout=70, retries={"max_attempts": 3, "mode": "standard"}))
call = lambda p: json.loads(rt.invoke_endpoint(EndpointName="rule-extractor-dev", ContentType="application/json",
                                                Body=json.dumps(p).encode())["Body"].read())
job = call({"operation": "submit_job", "idempotency_key": "catalogue-2026-q4", "options": {}})["data"]
print(call({"operation": "get_job", "job_id": job["job_id"]})["data"]["progress"])
```

Poll no faster than every 15 s. Re-submitting with the same `idempotency_key` and options returns the same job (`deduplicated: true`); the same key with different options returns 409.

### 10.5 Outputs

| Artifact | Content |
|---|---|
| `rule_catalogue.xlsx` | Copy of your template plus new rows, continuing `RULE_NNN` numbering; all new rows `needs_review`; footer lines only if configured |
| `executable_rules.jsonl` | Base rules plus new executable rules, keyed by the final `RULE_NNN` ids (the join to the workbook) |
| `rules_final.jsonl` | Deduplicated rules with final ids, internal provenance (`job_id`, `unit_id`, `prompt_version`, `source_model`, `validation_warnings`), and executable parts |
| `rules_raw.jsonl` | Every validated rule before deduplication |
| `mbs_billing_facts_reference.jsonl` | Deterministic EMSN-cap and derived-fee facts (no LLM) |
| `units.jsonl`, `summary.json` | Per-unit table and job summary (counts, cost, failures, versions) |

`validation_warnings` such as `unknown_icd10am_code:XXX` mean a cited code was not found in the corpus registry. Review those first.

## 11. Operations

### 11.1 Monitoring

| Signal | Where | Alarm |
|---|---|---|
| `Invocation5XXErrors` | AWS/SageMaker | `<endpoint>-invocation-5xx` (also gates blue/green rollback) |
| DLQ depth | AWS/SQS | `<endpoint>-dlq-not-empty` |
| `UnitsFailed` (sum per hour) | `RuleExtractor` namespace | `<endpoint>-units-failed` (≥ 5) |
| `UnitsSucceeded`, `UnitDurationSeconds`, `UnitCostUSD`, `RulesEmitted`, `TokensInput`, `TokensOutput`, `ActiveUnits`, `WorkerSlotUtilization`, `QueueVisible`, `QueueInFlight`, `PoisonMessages` | `RuleExtractor` namespace, dimension `EndpointName` | Dashboards; `WorkerSlotUtilization` drives optional auto scaling |
| Structured JSON logs (`job_id`, `unit_id`, `request_id`, `worker`) | `/aws/sagemaker/Endpoints/<endpoint>` | Query with CloudWatch Logs Insights |

Cost is the SDK-reported estimate; reconcile with Cost Explorer and Bedrock invocation logging (supported on the `invoke` provider). Set Bedrock quotas (RPM and TPM) per source region before large runs.

### 11.2 Capacity and cost planning

- Full plan: about 118 ACS topics plus about 306 sweep windows (123 + 84 + 67 + 32) = about 424 units. Your earlier measured data point: 9 units cost $3.23 and took 10.7 minutes at concurrency 3 (about 3.6 minutes per unit).
- Wall-clock estimate = units × 3.6 min ÷ (`CONCURRENCY` × instances): one instance about 8.5 h, two about 4.2 h, four about 2.1 h. Spend scales with units, not instances.
- Memory is dominated by one Claude CLI process per concurrent session (hundreds of MB each); `ml.m5.xlarge` (16 GB) suits concurrency up to 4.
- The endpoint bills while `InService`. For occasional builds, deploy, run, then delete the endpoint stack, or run `cli run` as a Processing job.

### 11.3 Runbook

| Symptom | Check and action |
|---|---|
| Endpoint stuck `Creating`/`Failed` | CloudWatch log group for the endpoint. Exit code 2 = model integrity failure (rebuild and re-upload the artifact); 3 = configuration or SDK preflight (message in the log); check VPC endpoints for `ecr.*` and `s3` |
| Many `FAILED` units with access errors | Bedrock model access, inference-profile id (`list-inference-profiles`), IAM resources match the geography, VPC endpoint for `bedrock-runtime` |
| Throttling | Lower `CONCURRENCY`, request quota increases for the source region, or add instances in a different quota scope |
| Job stuck `RUNNING` | `get_job` with `include_units`; janitor resolves lapsed leases within about one lease plus 60 s; `cancel_job` then `retry_failed` if needed |
| DLQ not empty | Receive and inspect messages (identifiers only); usually a poison message or a crash loop; redrive after fixing |
| `contract_check` job fails | `deny_outside_root` failure means a file read outside the corpus was not denied: stop and review permission settings before any further jobs |
| Rules with `unknown_*_code` warnings | Hallucinated or mis-formatted codes; reviewers should reject or correct |

### 11.4 Grouper plug-in (real AR-DRG regrouping)

AR-DRG grouping is licensed software and is not reimplemented. The default grouper is a labelled mock (`is_mock: true`).
Set `RULE_EXTRACTOR_GROUPER_FACTORY=your_pkg.grouper:build_grouper`, where the callable returns any object with `group(episode) -> GroupResult(drg_code, cost_weight, is_mock, details)`; wrap your existing `AuGrouperAdapter` there and install `.[grouper]` in the image if it needs pandas or pyarrow. Results are always labelled `preview_only`.

## 12. Migration notes and assumptions to verify

**You supplied six files** (`README.md`, `config.py`, `schema.py`, `run_extraction.py`, `requirements.txt`, `minimal_repro.py`). The other modules were rebuilt from the README's specification. Treat the reconstructed parts as new code to review.

| Original | Now | Status |
|---|---|---|
| `config.py` | `settings.py`, `configs/base.yaml`, `ingestion/sources.py` | Migrated; same defaults for models, chunking, retries, timeouts, turns (40, cap 100), concurrency 3 |
| `schema.py` | `domain/schema.py` | Migrated and hardened (limits, sanitising, provenance fields) |
| `run_extraction.py` | `cli.py`, `pipeline/local_run.py` | Migrated; **web research is now opt-in** (`--web-research`) |
| `minimal_repro.py` | `agents/contract_check.py`, `contract-check`, job kind `contract_check` | Evolved into a deploy-time check that also tests denial and both model tiers |
| `requirements.txt` | `pyproject.toml`, `requirements/*.in`, `prod.lock` | SDK pin moves from 0.2.152 to 0.2.159 (needed for `verbatim_prompts`) |
| `ingestion/*`, `agents/*`, `engine/*`, `pipeline/*`, `output/*` | same package names | **Reconstructed** |
| `engine/vendor/au_grouper.py`, `au_grouper_adapter.py` | not included | Plug in through `RULE_EXTRACTOR_GROUPER_FACTORY` |

Verify against your real environment before relying on it:

1. **Corpus build on the real PDFs:** `make corpus` must locate at least 100 ACS standards (you previously saw about 116 to 118). Compare code-registry counts (earlier: about 13,526 ICD-10-AM and 6,647 ACHI) and MBS counts (earlier: 6,046 items, 1,132 facts). Adjust `ingestion/toc.py` or `ingestion/mbs.FIELD_ALIASES` if they differ. Printed-page detection is a heuristic.
2. **Prompts** (`agents/prompt_templates/*.md`) are new. Diff them against your originals; you can paste your versions in (same placeholder names) without code changes, and `PROMPT_VERSION` plus a content digest are recorded on every rule.
3. **DSL grammar** (`engine/conditions.py`, `actions.py`) is designed from your README's description (exact, prefix, and wildcard matching; about ten operators; `automatable=false` override). Confirm it can express your 48 examples and update `engine/episode.py` if your claims schema differs.
4. **Bedrock ids and access:** defaults are `au.anthropic.claude-sonnet-5` and `au.anthropic.claude-haiku-4-5-20251001-v1:0`. Upgrade to Sonnet 5.5 or Haiku 5.5 only after confirming a profile exists for your geography.
5. **SDK behaviour on 0.2.159:** run `rule-extractor contract-check` locally and the contract-check job on the endpoint. The runner uses a streaming prompt, `permission_mode="default"`, a `can_use_tool` gate, and in-process MCP tools; the contract check proves they work together in your environment.
6. **Excel footer:** the original writer added a mandatory notice at runtime. The hook is preserved (`output.excel_footer` in `configs/base.yaml`) but the text is intentionally not stored in this repository; add your policy text there.
7. **Web research topics:** your original `DEFAULT_TOPICS` were not supplied; supply `web_research_topics` per request or list defaults in `configs/base.yaml`.

## 13. Verification status

| Area | Verified here | Needs your environment |
|---|---|---|
| Engine, schema, ingestion (synthetic PDFs and XML), writer, storage adapters (hand-written fakes), Bedrock env, permissions, collector, pipeline, worker lifecycle (crash takeover, cancel, shutdown hand-back, budget, janitor), HTTP contract, supervisor, CLI, local resume | 214 unit tests pass (Python 3.12; code avoids 3.11-only features for the 3.10 floor) | `make test` in Code Editor, plus `make lint` (ruff, mypy, cfn-lint) |
| Claude Agent SDK and bundled CLI | Runner exercised against a fake SDK mirroring the documented API | Real SDK 0.2.159, real Bedrock calls, `contract-check` |
| Licensed PDFs | Not available | `make corpus` and the counts in section 12 |
| CloudFormation | YAML, reference, IAM least-privilege, encryption, and env-name checks | `cfn-lint`, a real deploy |
| Docker image, SageMaker invocation, VPC endpoints, quotas | Not executed (no Docker, AWS, or PyPI access in the build sandbox) | Section 9 verification steps |

## 14. Known limitations

- Duplicate detection is exact on normalised logic text; near-duplicates across the ACS and sweep passes (or against your existing rows) are not merged. Review for overlap before treating the catalogue as final.
- Executable rules cover the grammar only; rules needing clinical judgement are narrative-only by design.
- `evaluate_episodes` is a preview mechanism and uses a mock grouper unless you plug in the real one.
- The SDK-reported cost is an estimate; Bedrock bills the authoritative amount.
- A real-time endpoint cannot scale to zero; use Asynchronous Inference or a Processing job if idle cost matters.
- SQS and DynamoDB are regional: a regional outage pauses jobs (they resume afterwards); there is no cross-region failover.
