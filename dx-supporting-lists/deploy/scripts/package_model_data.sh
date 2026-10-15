#!/usr/bin/env bash
# Packages the four licensed reference files into the SageMaker "model artifact" (model.tar.gz) and uploads it. SageMaker downloads and unpacks it into
# /opt/ml/model before the container starts, so the data is versioned, immutable, encrypted at rest, and never baked into an image.
#
#   deploy/scripts/package_model_data.sh <data-dir> <artifacts-bucket> [version]
#
# Encryption: no --sse flags are passed on purpose, so the bucket's default encryption (the KMS key created by the foundation stack) applies.
set -euo pipefail

DATA_DIR="${1:?usage: package_model_data.sh <data-dir> <artifacts-bucket> [version]}"
BUCKET="${2:?usage: package_model_data.sh <data-dir> <artifacts-bucket> [version]}"
VERSION="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"

for f in diagnosis_codes.tsv achi_codes.tsv all_pros.csv; do
  [ -f "$DATA_DIR/$f" ] || { echo "missing $DATA_DIR/$f" >&2; exit 1; }
done
MBS="$(ls "$DATA_DIR"/MBS-XML-*.XML 2>/dev/null | sort | tail -n 1 || true)"
[ -n "$MBS" ] || { echo "missing $DATA_DIR/MBS-XML-*.XML" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/data"
cp "$DATA_DIR/diagnosis_codes.tsv" "$DATA_DIR/achi_codes.tsv" "$DATA_DIR/all_pros.csv" "$MBS" "$TMP/data/"
( cd "$TMP/data" && sha256sum ./* > "$TMP/SHA256SUMS" )
echo "Files and SHA-256 digests (keep these with your release record):"
cat "$TMP/SHA256SUMS"

# World-readable inside the artifact, owned by root: the container runs as an unprivileged user and must be able to read it.
tar --owner=0 --group=0 --numeric-owner --mode='u=rwX,go=rX' -czf "$TMP/model.tar.gz" -C "$TMP" data
KEY="model/${VERSION}/model.tar.gz"
aws s3 cp "$TMP/model.tar.gz" "s3://${BUCKET}/${KEY}" --only-show-errors
echo
echo "ModelDataUrl=s3://${BUCKET}/${KEY}"
