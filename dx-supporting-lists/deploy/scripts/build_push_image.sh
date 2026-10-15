#!/usr/bin/env bash
# Builds the image with Docker and pushes it to ECR (needs Docker access in your environment; otherwise use build_image_codebuild.sh).
#
#   AWS_REGION=ap-southeast-2 ECR_REPO_URI=<account>.dkr.ecr.<region>.amazonaws.com/dx-support-prod deploy/scripts/build_push_image.sh [tag]
#
# ECR tags are immutable in the foundation stack, so every build needs a new tag (default: git short SHA + UTC time).
set -euo pipefail
: "${AWS_REGION:?set AWS_REGION}"
: "${ECR_REPO_URI:?set ECR_REPO_URI (the foundation stack output EcrRepositoryUri)}"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG="${1:-${SHA}-$(date -u +%Y%m%d%H%M%S)}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ECR_REPO_URI%%/*}"
docker build --pull -t "${ECR_REPO_URI}:${TAG}" .
docker push "${ECR_REPO_URI}:${TAG}"
echo
echo "ImageUri=${ECR_REPO_URI}:${TAG}"
