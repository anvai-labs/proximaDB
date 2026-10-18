#!/usr/bin/env bash
# Download the server-side embedding model (BAAI/bge-small-en-v1.5, ONNX) into
# ./models — mounted into the engine container by docker-compose.yml.
# Idempotent: skips files that already exist non-empty.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="${1:-$HERE/models}"
mkdir -p "${MODELS}"

REV="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"   # pinned HF revision of BAAI/bge-small-en-v1.5

fetch() { # fetch <file-path-in-repo> <dest> — atomic, revision-pinned
  local path="$1" dest="$2"
  if [ -s "$dest" ]; then echo "✓ $(basename "$dest") already present"; return 0; fi
  local url="https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/${REV}/${path}"
  echo "▶ downloading $(basename "$dest") (rev ${REV:0:8}) …"
  # tmp + mv: an interrupted download must never leave a truncated model
  # that later runs treat as "already present".
  curl -sL --fail -o "$dest.tmp" "$url" && mv "$dest.tmp" "$dest" || { rm -f "$dest.tmp"; echo "download failed" >&2; return 1; }
}

ls -la "${MODELS}"
