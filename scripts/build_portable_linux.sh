#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the released Linux x86_64 binaries so they run on every distribution the
# packages claim: glibc >= 2.28 (RHEL/Rocky/Alma 8+, Debian 10+, Ubuntu 20.04+)
# and NO dependency on the system OpenSSL.
#
# Why: the v0.4.1 binaries were built on ubuntu-22.04, so they required
# GLIBC_2.35 and libssl.so.3. The `.el8.rpm` repackaging them could not start on
# Rocky 8 (no libssl.so.3, glibc 2.28) nor Rocky 9 (glibc 2.34) -- found by
# release-smoke.yml's packages job.
#
#   * glibc: compile inside the manylinux_2_28 image (AlmaLinux 8, glibc 2.28,
#     gcc-toolset); a binary links against the OLDEST glibc it is built on.
#   * OpenSSL: the server's `vendored-openssl` feature statically links OpenSSL
#     (openssl-sys reaches the tree only through reqwest 0.11 / lettre
#     native-tls). Same TLS stack and behaviour; native-tls still locates the
#     system CA bundle through openssl-probe.
#
# Runs on the host: starts the container and builds in it. Then fails unless
# every binary needs glibc <= 2.28 and links no libssl/libcrypto.
#
# Usage: build_portable_linux.sh <features> [target_dir]
#   features    e.g. "cloud-full,vendored-openssl"
#   target_dir  host dir for cargo's target (default: ./target)
# Env: PORTABLE_IMAGE (pinned below), CARGO_BUILD_JOBS, PORTABLE_CARGO_HOME
#      (host dir to persist the registry between runs; default ~/.cargo).
set -euo pipefail
FEATURES="${1:?features}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$(mkdir -p "${2:-$REPO/target}" && cd "${2:-$REPO/target}" && pwd)"
TRIPLE="x86_64-unknown-linux-gnu"
# manylinux_2_28 (AlmaLinux 8.10, glibc 2.28, gcc-toolset-14), pinned by digest.
IMAGE="${PORTABLE_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64@sha256:1e7b8460ad8fb3fb0fbe1cd818845159ac896b4fd17452d56e892c291f3b97a8}"
CARGO_HOME_HOST="${PORTABLE_CARGO_HOME:-$HOME/.cargo}"
mkdir -p "$CARGO_HOME_HOST/registry" "$CARGO_HOME_HOST/git"
MAX_GLIBC="2.28"

docker run --rm \
  -v "$REPO:/work" -w /work \
  -v "$TARGET_DIR:/target" \
  -v "$CARGO_HOME_HOST/registry:/cargo/registry" \
  -v "$CARGO_HOME_HOST/git:/cargo/git" \
  -e CARGO_HOME=/cargo -e RUSTUP_HOME=/rustup -e CARGO_TARGET_DIR=/target \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-}" -e CARGO_INCREMENTAL=0 \
  -e CARGO_NET_RETRY=10 -e RUSTUP_MAX_RETRIES=10 \
  -e FEATURES="$FEATURES" -e TRIPLE="$TRIPLE" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  "$IMAGE" bash -euo pipefail -c '
    # Hand every file the container (root) writes back to the host user on ANY
    # exit: root-owned target/ or registry files left by a failed build break
    # the next step (rust-cache post) and the next run.
    trap '"'"'chown -R "$HOST_UID:$HOST_GID" /target /cargo/registry /cargo/git'"'"' EXIT
    # vendored OpenSSL builds with perl (IPC::Cmd etc.).
    dnf install -y -q perl-IPC-Cmd perl-core >/dev/null
    curl --proto =https --tlsv1.2 -sSf --retry 5 https://sh.rustup.rs \
      | sh -s -- -y -q --profile minimal --default-toolchain none
    export PATH=/cargo/bin:$PATH
    # rust-toolchain.toml in the checkout selects the toolchain.
    rustup show active-toolchain || rustup toolchain install
    [ -n "${CARGO_BUILD_JOBS}" ] || unset CARGO_BUILD_JOBS
    cargo build --locked --release --target "$TRIPLE" --features "$FEATURES" \
      -p proximadb-server -p proximadb-bench -p proximadb-migrate
  '

bash "$REPO/scripts/check_linux_portability.sh" "$TARGET_DIR/$TRIPLE/release" "$MAX_GLIBC"
