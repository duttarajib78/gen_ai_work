# syntax=docker/dockerfile:1.7
#
# SageMaker real-time inference image (bring-your-own-container).
# Build from the repository root:   docker build -f container/Dockerfile -t <repo>:<tag> .
# See README.md ("Build the image") and tools/deploy/build_and_push.sh.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Fail the build (not the deployment) when the internal grouper module was not supplied.
ARG REQUIRE_GROUPER=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/program/src:/opt/program/vendor

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /opt/program

COPY requirements/runtime.txt requirements/runtime.txt
RUN pip install -r requirements/runtime.txt

# The internal grouper client may live on a private index. Credentials are supplied as a
# BuildKit secret (docker build --secret id=pip_conf,src=<pip.conf>), mounted only for this
# step, and never written to an image layer.
COPY requirements/grouper.txt requirements/grouper.txt
RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf,required=false \
    pip install -r requirements/grouper.txt

COPY src/ src/
COPY vendor/ vendor/
RUN if [ "${REQUIRE_GROUPER}" = "true" ] && [ ! -f vendor/au_grouper.py ]; then \
      echo "ERROR: vendor/au_grouper.py is missing. Copy the internal au_grouper module into vendor/ " \
           "(or build with --build-arg REQUIRE_GROUPER=false and AUDIT_GROUPER_BACKEND=disabled)." >&2; \
      exit 1; \
    fi

COPY container/serve container/gunicorn.conf.py ./
RUN chmod 0555 /opt/program/serve \
 && chown -R root:root /opt/program \
 && chmod -R go-w /opt/program

USER 10001:10001
EXPOSE 8080

# SageMaker runs `docker run <image> serve`; the script ignores its arguments.
ENTRYPOINT ["/opt/program/serve"]
