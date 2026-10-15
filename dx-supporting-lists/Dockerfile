# syntax=docker/dockerfile:1.7
# Production image for the SageMaker real-time endpoint (and, with a different command, for SageMaker Processing batch builds).
#
#   docker build -t dx-support:local .
#
# BASE_IMAGE: the ECR Public mirror avoids Docker Hub rate limits. For a locked-down build, point it at your own mirror and pin it by digest:
#   --build-arg BASE_IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/mirror/python:3.12-slim@sha256:<digest>
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim

FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Dependencies first (cached layer). requirements.lock (hash-pinned, from `make lock`) wins when present.
COPY requirements.txt requirements.lock* ./
RUN if [ -f requirements.lock ]; then pip install --require-hashes -r requirements.lock; else pip install -r requirements.txt; fi
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

FROM ${BASE_IMAGE} AS runtime
LABEL org.opencontainers.image.title="dx-supporting-lists" \
      org.opencontainers.image.description="Supporting lists for ICD-10-AM categories: SageMaker real-time endpoint using Claude on Amazon Bedrock"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:${PATH}" \
    DX_ENV=production DX_DATA_DIR=/opt/ml/model/data DX_LOG_JSON=true DX_LOG_CALLS=true
# A fixed unprivileged user: nothing in the container needs root, and nothing needs to write outside /tmp.
RUN groupadd --system --gid 10001 dx \
 && useradd --system --uid 10001 --gid dx --no-create-home --shell /usr/sbin/nologin dx
COPY --from=builder /opt/venv /opt/venv
COPY deploy/container/entrypoint.sh deploy/container/gunicorn.conf.py /opt/program/
RUN chmod 0555 /opt/program/entrypoint.sh && chmod 0444 /opt/program/gunicorn.conf.py
WORKDIR /opt/program
USER 10001:10001
EXPOSE 8080
# SageMaker starts the container as `docker run <image> serve`.
ENTRYPOINT ["/opt/program/entrypoint.sh"]
CMD ["serve"]
