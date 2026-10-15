# syntax=docker/dockerfile:1.7
# SageMaker real-time endpoint image. Build: scripts/build_and_push.sh (see README).
# Requires requirements/prod.lock (hash-pinned; create it with `make lock`). Nothing licensed (source PDFs/corpus) is ever
# baked into the image: those arrive via the model artifact at /opt/ml/model.
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm

FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements/prod.lock requirements/prod.lock
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --require-hashes -r requirements/prod.lock

FROM ${BASE_IMAGE} AS runtime
ARG VERSION=dev
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="rule-extraction-service" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

# ripgrep backs the Claude Code Grep tool; everything else comes from the venv (the SDK wheel bundles the Claude CLI).
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*
# Unprivileged identity for the app and the Claude CLI. SageMaker starts containers as root; the supervisor (PID 1) drops
# privileges for its children (RULE_EXTRACTOR_RUN_AS_UID), so model code never runs as root.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY src /opt/program/src
COPY configs /opt/program/configs
COPY docker/entrypoint.sh /opt/program/entrypoint.sh
RUN chmod 0555 /opt/program/entrypoint.sh \
    && mkdir -p /tmp/rule-extractor && chmod 1777 /tmp/rule-extractor

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/program/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RULE_EXTRACTOR_CONFIG_FILE=/opt/program/configs/base.yaml \
    RULE_EXTRACTOR_BACKEND=aws \
    RULE_EXTRACTOR_ENV=prod \
    RULE_EXTRACTOR_MODEL_DIR=/opt/ml/model \
    RULE_EXTRACTOR_WORK_DIR=/tmp/rule-extractor
WORKDIR /opt/program
EXPOSE 8080
# Exec form so SIGTERM from SageMaker reaches the supervisor; SageMaker appends "serve".
ENTRYPOINT ["/opt/program/entrypoint.sh"]
CMD ["serve"]
