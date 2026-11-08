#!/bin/bash
# Pre-publish DEB verifier (release-review F4 symmetry with verify_rpm.sh).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" debian:11 bash verify_deb.sh <version>
# The .deb must be at /pkg/proximadb_*_amd64.deb.
set -euo pipefail

VER="${1:?usage: verify_deb.sh <version> — the .deb must be at /pkg/proximadb_*_amd64.deb}"

apt-get update -qq
apt-get install -y -qq /pkg/proximadb_*_amd64.deb
test -x /usr/bin/proximadb-server
test -f /etc/proximadb/config.toml
test -f /lib/systemd/system/proximadb.service
# F4 symmetry with the RPM leg. Two dpkg -V semantics matter (empirically
# verified on debian:11 / dpkg 1.20; re-pin at each debian bump —
# src/verify.c): it verifies md5sums + presence of packaged regular files
# only (no dirs/ownership/modes — the TD-PKG-1 chown needs no tolerance),
# and it exits 0 while PRINTING deviations (rc=1 only for a
# not-installed package). So the gate REQUIRES empty output — but capture
# with `|| true` AND guard the rc: a nonzero rc with NO output is a hard
# failure (db error / not installed), exactly like the RPM leg's rc guard.
rc=0
out="$(dpkg -V proximadb)" || rc=$?
if [ "$rc" -ne 0 ] && [ -z "$out" ]; then
    echo "dpkg -V failed with rc=$rc and no verify output" >&2
    exit 1
fi
if [ -n "$out" ]; then
    echo "dpkg -V reported deviations:" >&2
    printf '%s\n' "$out" >&2
    exit 1
fi
# (Mode-only payload defects are out of dpkg -V's reach — covered by the
# post-release distro smoke, not here.)

got="$(/usr/bin/proximadb-server --version)"
test "$got" = "proximadb-server $VER"
dpkg -r proximadb
test ! -e /usr/bin/proximadb-server
echo "DEB verify PASSED (install, payload md5s clean, version, erase)"
