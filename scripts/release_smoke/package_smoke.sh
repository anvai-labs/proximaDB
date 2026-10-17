#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Inside a clean distro container: install a released .deb or .rpm with the
# distro's package manager, check `proximadb-server --version`, start the
# server on the config the package installs, and require /health to report
# healthy at the release version.
#
# No curl/python needed (minimal images lack them): /health is read with bash's
# /dev/tcp.
#
# Usage (in the container): package_smoke.sh <deb|rpm> <package file> <version>
set -euo pipefail
KIND="${1:?deb|rpm}"; PKG="${2:?package file}"; V="${3:?version}"
# shellcheck disable=SC1091  # the container's own os-release
DISTRO="$(. /etc/os-release && echo "${ID}:${VERSION_ID}")"

case "$KIND" in
  deb)
    apt-get update -qq >/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG" >/tmp/install.log 2>&1 \
      || { tail -20 /tmp/install.log; echo "::error::${KIND} on ${DISTRO}: install failed"; exit 1; } ;;
  rpm)
    dnf install -y -q "$PKG" >/tmp/install.log 2>&1 \
      || { tail -20 /tmp/install.log; echo "::error::${KIND} on ${DISTRO}: install failed"; exit 1; } ;;
  *) echo "::error::unknown package kind ${KIND}"; exit 1 ;;
esac

got="$(proximadb-server --version 2>&1)" \
  || { echo "::error::${KIND} on ${DISTRO}: proximadb-server does not run: ${got}"; exit 1; }
[ "$got" = "proximadb-server ${V}" ] \
  || { echo "::error::${KIND} on ${DISTRO}: --version printed ${got}"; exit 1; }

CONFIG=/etc/proximadb/config.toml
[ -f "$CONFIG" ] || { echo "::error::${KIND} on ${DISTRO}: package installs no ${CONFIG}"; exit 1; }
PORT="$(sed -n 's/^unified_port = \([0-9]*\)/\1/p; s/^rest_port = \([0-9]*\)/\1/p' "$CONFIG" | head -1)"
PORT="${PORT:-5678}"
proximadb-server --config "$CONFIG" >/tmp/server.log 2>&1 &
SERVER=$!

health() {
  exec 3<>"/dev/tcp/127.0.0.1/${PORT}" || return 1
  printf 'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n' >&3
  cat <&3
  exec 3<&-
}
body=""
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    tail -30 /tmp/server.log
    echo "::error::${KIND} on ${DISTRO}: server exited before answering /health"; exit 1
  fi
  if body="$(health 2>/dev/null)" && grep -q '"status":"healthy"' <<<"$body"; then
    break
  fi
  body=""
  sleep 2
done
kill "$SERVER"
[ -n "$body" ] || { tail -30 /tmp/server.log; echo "::error::${KIND} on ${DISTRO}: /health not healthy in 120s"; exit 1; }
grep -q "\"version\":\"${V}\"" <<<"$body" \
  || { echo "::error::${KIND} on ${DISTRO}: /health is not version ${V}: ${body}"; exit 1; }
echo "${KIND} on ${DISTRO}: ${got}; /health healthy at ${V}"
