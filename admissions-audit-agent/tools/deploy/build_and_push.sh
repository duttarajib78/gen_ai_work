#!/usr/bin/env bash
# Build the inference image and push it to Amazon ECR with an immutable tag.
#
#   AWS_REGION=ap-southeast-2 ECR_REPOSITORY=admissions-audit-agent bash tools/deploy/build_and_push.sh
#
# Optional environment:
#   IMAGE_TAG      default: short git commit, plus "-dirty" if the tree has uncommitted changes
#   BASE_IMAGE     pin the base image by digest, e.g. python:3.12-slim-bookworm@sha256:...
#   PIP_CONF_FILE  pip.conf with credentials for a private package index (passed as a BuildKit
#                  secret, never stored in the image)
#   ECR_KMS_KEY    KMS key for repository encryption when the repository is created (default AES256)
#   REQUIRE_GROUPER=false  build without vendor/au_grouper.py (use AUDIT_GROUPER_BACKEND=disabled)
#
# Needs Docker with BuildKit. In a Studio domain without Docker access, run the same build in
# AWS CodeBuild with tools/deploy/buildspec.yml.
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${ECR_REPOSITORY:?set ECR_REPOSITORY}"

cd "$(dirname "$0")/../.."
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [ -z "${IMAGE_TAG:-}" ]; then
  IMAGE_TAG="$(git rev-parse --short=12 HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)"
  if [ -n "$(git status --porcelain 2>/dev/null || true)" ]; then IMAGE_TAG="${IMAGE_TAG}-dirty"; fi
fi
case "${IMAGE_TAG}" in latest|"") echo "refusing to use a mutable tag" >&2; exit 1;; esac
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

if ! aws ecr describe-repositories --region "${AWS_REGION}" --repository-names "${ECR_REPOSITORY}" >/dev/null 2>&1; then
  echo "Creating repository ${ECR_REPOSITORY} (immutable tags, scan on push)"
  if [ -n "${ECR_KMS_KEY:-}" ]; then
    ENCRYPTION="encryptionType=KMS,kmsKey=${ECR_KMS_KEY}"
  else
    ENCRYPTION="encryptionType=AES256"
  fi
  aws ecr create-repository --region "${AWS_REGION}" --repository-name "${ECR_REPOSITORY}" \
    --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true \
    --encryption-configuration "${ENCRYPTION}" >/dev/null
fi

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"

BUILD_ARGS=(--build-arg "REQUIRE_GROUPER=${REQUIRE_GROUPER:-true}")
[ -n "${BASE_IMAGE:-}" ] && BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
SECRET_ARGS=()
[ -n "${PIP_CONF_FILE:-}" ] && SECRET_ARGS+=(--secret "id=pip_conf,src=${PIP_CONF_FILE}")

DOCKER_BUILDKIT=1 docker build "${BUILD_ARGS[@]}" ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"} -f container/Dockerfile -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

DIGEST="$(aws ecr describe-images --region "${AWS_REGION}" --repository-name "${ECR_REPOSITORY}" \
  --image-ids imageTag="${IMAGE_TAG}" --query 'imageDetails[0].imageDigest' --output text)"
echo
echo "image_uri: ${IMAGE_URI}"
echo "digest:    ${DIGEST}"
echo "Put image_uri in your deployment config. Review the ECR scan findings before deploying."
