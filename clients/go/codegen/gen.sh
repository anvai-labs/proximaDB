#!/usr/bin/env bash
# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Generate the typed Go REST transport (internal/genrest) from the published
# OpenAPI spec (TD-126 Phase 2). Invoked by `make gen-go-sdk`.
#
# Pipeline:
#   1. down-convert docs/openapi/proximadb-openapi.yaml (OpenAPI 3.1, generated +
#      drift-gated by Phase 1) to 3.0.3 in a temp file — the source spec is never
#      edited; oapi-codegen/kin-openapi does not yet support 3.1.
#   2. run oapi-codegen (version pinned in codegen/go.mod) on the temp file.
#
# Regenerating must be deterministic: the CI drift gate runs this and then
# `git diff --exit-code`s internal/genrest.
set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GO_DIR}/../.." && pwd)"

SPEC_31="${REPO_ROOT}/docs/openapi/proximadb-openapi.yaml"
OUT="${GO_DIR}/proximadb/internal/genrest/genrest.gen.go"
PYTHON="${PYTHON:-python3}"

TMP_SPEC="$(mktemp -t proximadb-openapi-30.XXXXXX.yaml)"
trap 'rm -f "${TMP_SPEC}"' EXIT

echo "==> down-converting OpenAPI 3.1 -> 3.0 (${SPEC_31})"
"${PYTHON}" "${SCRIPT_DIR}/openapi_31_to_30.py" \
  --go-optional-nullable-omitempty "${SPEC_31}" "${TMP_SPEC}"

echo "==> generating ${OUT} (oapi-codegen, pinned in codegen/go.mod)"
mkdir -p "$(dirname "${OUT}")"
( cd "${SCRIPT_DIR}" && \
  go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen \
    -config "${SCRIPT_DIR}/oapi-codegen.yaml" \
    -o "${OUT}" \
    "${TMP_SPEC}" )

echo "==> formatting ${OUT}"
gofmt -w "${OUT}"

echo "==> done"
