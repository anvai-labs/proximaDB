#!/usr/bin/env bash
# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Generate the typed Rust REST transport (clients/rust/src/genrest.rs) from the
# published OpenAPI spec (TD-126 Phase 4). Invoked by `make gen-rust-sdk`.
#
# Pipeline (mirrors the Go pilot, clients/go/codegen/gen.sh):
#   1. down-convert docs/openapi/proximadb-openapi.yaml (OpenAPI 3.1, generated +
#      drift-gated by Phase 1) to 3.0.3 in a temp file — the source spec is never
#      edited; progenitor (like oapi-codegen) does not support OpenAPI 3.1. We
#      reuse the exact same down-converter the Go pilot uses.
#   2. run the progenitor-based generator (version pinned in codegen/Cargo.toml)
#      on the temp file, emitting a single `// @generated` module.
#
# Regenerating must be deterministic: the CI drift gate runs this and then
# `git diff --exit-code`s clients/rust/src/genrest.rs.
set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RUST_DIR}/../.." && pwd)"

SPEC_31="${REPO_ROOT}/docs/openapi/proximadb-openapi.yaml"
OUT="${RUST_DIR}/src/genrest.rs"
PYTHON="${PYTHON:-python3}"

TMP_SPEC="$(mktemp -t proximadb-openapi-30.XXXXXX.yaml)"
trap 'rm -f "${TMP_SPEC}"' EXIT

echo "==> down-converting OpenAPI 3.1 -> 3.0 (${SPEC_31})"
"${PYTHON}" "${SCRIPT_DIR}/openapi_31_to_30.py" "${SPEC_31}" "${TMP_SPEC}"

echo "==> building the generator (progenitor, pinned in codegen/Cargo.toml)"
( cd "${SCRIPT_DIR}" && cargo build --locked --release --quiet )

echo "==> generating ${OUT}"
mkdir -p "$(dirname "${OUT}")"
"${SCRIPT_DIR}/target/release/gen" "${TMP_SPEC}" "${OUT}"

# Canonicalise with the SDK crate's rustfmt (analogous to the Go pilot's
# `gofmt -w`). The generator already converts doc comments to `///` line form
# (see codegen/src/main.rs) so rustfmt preserves the ```` ```ignore ```` example
# fences; running `cargo fmt` here keeps the committed artifact identical to
# what `cargo fmt --check` expects, so the format gate and the codegen-drift
# gate agree. Deterministic because CI runs the same stable rustfmt.
echo "==> formatting ${OUT} (cargo fmt)"
( cd "${RUST_DIR}" && cargo fmt -- "${OUT}" )

echo "==> done"
