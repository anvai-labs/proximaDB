#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a throwaway cargo project that depends on proximadb-sdk ONLY from
# crates.io at exactly the release version, outside any workspace.
#
# Usage: render_manifest.sh <version> <dest_dir>
set -euo pipefail
VERSION="${1:?version}"; DEST="${2:?dest dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DEST/src"
cp "$HERE/src/main.rs" "$DEST/src/main.rs"
cat > "$DEST/Cargo.toml" <<EOF
[package]
name = "rust-sdk-smoke"
version = "0.0.0"
edition = "2024"
publish = false

# An empty workspace table: never adopted by an enclosing workspace.
[workspace]

[dependencies]
proximadb-sdk = "=${VERSION}"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time"] }

[profile.dev]
debug = 0
EOF
echo "rendered ${DEST}/Cargo.toml (proximadb-sdk = \"=${VERSION}\", crates.io only)"
