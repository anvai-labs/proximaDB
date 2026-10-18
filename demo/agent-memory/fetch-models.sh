#!/usr/bin/env bash
# Download the server-side embedding model (BAAI/bge-small-en-v1.5, ONNX) into
# ./models — mounted into the engine container by docker-compose.yml.
# Idempotent: skips files that already exist non-empty.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="${1:-$HERE/models}"
mkdir -p "${MODELS}"

fetch() { # fetch <url> <dest>
  if [ -s "$2" ]; then echo "✓ $(basename "$2") already present"; else
    echo "▶ downloading $(basename "$2") …"; curl -sL --fail -o "$2" "$1"
  fi
}

fetch "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" \
      "${MODELS}/bge-small-en-v1.5.onnx"
fetch "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json" \
      "${MODELS}/tokenizer.json"

ls -la "${MODELS}"
