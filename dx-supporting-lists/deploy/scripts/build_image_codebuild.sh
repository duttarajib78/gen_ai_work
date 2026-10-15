#!/usr/bin/env bash
# Builds and pushes the image with AWS CodeBuild (no local Docker needed). Requires the foundation stack with CreateBuildProject=true.
#
#   STACK=prod-dx-support-foundation deploy/scripts/build_image_codebuild.sh [tag]
set -euo pipefail
: "${STACK:?set STACK to the foundation stack name}"
out() { aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
BUCKET="$(out ArtifactsBucketName)"; PROJECT="$(out BuildProjectName)"; REPO="$(out EcrRepositoryUri)"
[ "$PROJECT" != "none" ] || { echo "the foundation stack was deployed with CreateBuildProject=false" >&2; exit 1; }
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG="${1:-${SHA}-$(date -u +%Y%m%d%H%M%S)}"
ZIP="$(mktemp -d)/source.zip"
python3 "$(dirname "$0")/make_source_zip.py" "$ZIP"
aws s3 cp "$ZIP" "s3://${BUCKET}/source/source.zip" --only-show-errors
BUILD_ID="$(aws codebuild start-build --project-name "$PROJECT" --environment-variables-override "name=IMAGE_TAG,value=${TAG},type=PLAINTEXT" --query 'build.id' --output text)"
echo "started $BUILD_ID"
while true; do
  STATUS="$(aws codebuild batch-get-builds --ids "$BUILD_ID" --query 'builds[0].buildStatus' --output text)"
  [ "$STATUS" = "IN_PROGRESS" ] || break
  sleep 15
done
echo "build finished: $STATUS"
[ "$STATUS" = "SUCCEEDED" ] || { echo "see the CodeBuild logs for $BUILD_ID" >&2; exit 1; }
echo "ImageUri=${REPO}:${TAG}"
