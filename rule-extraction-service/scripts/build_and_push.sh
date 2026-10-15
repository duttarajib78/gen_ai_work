#!/usr/bin/env bash
# Build the endpoint image and push it to ECR with an immutable tag.
#   BUILDER=docker    (default)  needs a Docker daemon; in SageMaker Studio enable Docker access and set
#                                DOCKER_BUILD_ARGS="--network sagemaker"
#   BUILDER=sm-docker            no daemon needed: builds in CodeBuild via the sagemaker-studio-image-build CLI
# Required: AWS_REGION, ECR_REPOSITORY (foundation stack output RepositoryName). Optional: IMAGE_TAG.
set -euo pipefail
: "${AWS_REGION:?set AWS_REGION}"
: "${ECR_REPOSITORY:?set ECR_REPOSITORY (foundation stack output RepositoryName)}"
BUILDER="${BUILDER:-docker}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short=12 HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

if [[ ! -f requirements/prod.lock ]]; then
  echo "requirements/prod.lock is missing - run 'make lock' first (hash-pinned installs are mandatory)." >&2
  exit 1
fi

case "$BUILDER" in
  docker)
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
    # shellcheck disable=SC2086
    docker build ${DOCKER_BUILD_ARGS:-} --build-arg VERSION="$IMAGE_TAG" --build-arg GIT_SHA="$IMAGE_TAG" -t "$IMAGE_URI" .
    docker push "$IMAGE_URI"
    ;;
  sm-docker)
    sm-docker build . --repository "${ECR_REPOSITORY}:${IMAGE_TAG}" \
      --build-arg VERSION="$IMAGE_TAG" --build-arg GIT_SHA="$IMAGE_TAG"
    ;;
  *)
    echo "unknown BUILDER '$BUILDER' (use docker or sm-docker)" >&2
    exit 1
    ;;
esac
echo "IMAGE_URI=${IMAGE_URI}"
