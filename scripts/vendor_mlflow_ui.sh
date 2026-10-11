#!/usr/bin/env bash
# TD-MLOPS-3: vendor the stock MLflow UI web bundle into third_party/.
# Downloads the pinned FULL wheel (mlflow-skinny has no UI), verifies its
# sha256, and extracts mlflow/server/js/build/ — reproducibly.
set -euo pipefail

MLFLOW_VERSION="3.16.0"
WHEEL_SHA256="c4ac5e8634aacad1a3d7d5a1a31be9279593ebd48b84272843682db11970cf71"
DEST="third_party/mlflow-ui/${MLFLOW_VERSION}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [ -d "${DEST}/build" ] && [ -f "${DEST}/PROVENANCE.adoc" ]; then
  echo "vendored tree already present at ${DEST} (idempotent)"
  exit 0
fi

pip download "mlflow==${MLFLOW_VERSION}" --no-deps -q -d "$TMP" >&2
WHEEL=$(ls "$TMP"/mlflow-*.whl)
ACTUAL_SHA=$(shasum -a 256 "$WHEEL" | awk '{print $1}')

if [ -z "${MLFLOW_UI_WHEEL_SHA256:-}" ]; then
  MLFLOW_UI_WHEEL_SHA256="$WHEEL_SHA256"
fi

if [ "$ACTUAL_SHA" != "$MLFLOW_UI_WHEEL_SHA256" ]; then
  if [ "$WHEEL_SHA256" = "placeholder; set on first run" ]; then
    echo "NOTE: first run — record this sha256 as WHEEL_SHA256 in the script:"
    echo "  $ACTUAL_SHA"
    MLFLOW_UI_WHEEL_SHA256="$ACTUAL_SHA"
  else
    echo "ERROR: wheel sha256 mismatch (expected $MLFLOW_UI_WHEEL_SHA256, got $ACTUAL_SHA)" >&2
    exit 1
  fi
fi

mkdir -p "$DEST"
unzip -q -o "$WHEEL" "mlflow/server/js/build/*" -d "$TMP/extracted"
mv "$TMP/extracted/mlflow/server/js/build" "$DEST/build"
unzip -p "$WHEEL" "mlflow/server/js/build/static/js/main.2408fedd.js.LICENSE.txt" > /dev/null 2>&1 || true

cat > "${DEST}/PROVENANCE.adoc" <<EOF
= Vendored MLflow UI — provenance

|===
| Field | Value
| Upstream | https://github.com/mlflow/mlflow (Apache-2.0)
| Version | ${MLFLOW_VERSION}
| Wheel | mlflow-${MLFLOW_VERSION}-py3-none-any.whl
| Wheel SHA-256 | ${MLFLOW_UI_WHEEL_SHA256}
| Extracted | mlflow/server/js/build/ -> build/ (unmodified)
| Vendored on | $(date -u +%Y-%m-%d)
| License | Apache-2.0 (upstream; see LICENSE below)
|===

The MLflow project is Apache-2.0. The full upstream LICENSE text is
available at https://github.com/mlflow/mlflow/blob/${MLFLOW_VERSION}/LICENSE;
per-file webpack license notices are preserved inside build/static/js/*.js.LICENSE.txt.
Re-extract or bump via: scripts/vendor_mlflow_ui.sh
EOF

curl -sf "https://raw.githubusercontent.com/mlflow/mlflow/v${MLFLOW_VERSION}/LICENSE" -o "${DEST}/LICENSE" || \
  echo "NOTE: fetch LICENSE manually from upstream" >&2

echo "vendored ${MLFLOW_VERSION} UI -> ${DEST} ($(du -sh ${DEST}/build | awk '{print $1}'))"
echo "wheel sha256: ${MLFLOW_UI_WHEEL_SHA256}"
