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
# F4 symmetry with the RPM leg. CAUTION: dpkg -V (debian:11 ships 1.20.x)
# verifies md5sums + presence of packaged regular files only — no
# dirs/ownership/modes, so the TD-PKG-1 chown needs no tolerance here —
# AND it exits 0 even when deviations are printed (its rc is 1 only for a
# not-installed package, dpkg 1.20 src/verify.c). So the gate REQUIRES
# empty output: any printed deviation fails.
# dpkg -V exits 0 while PRINTING deviations (rc=1 only for a
# not-installed package, dpkg 1.20/1.22 src/verify.c) — capture with
# || true so set -e never kills the diagnostics before they print.
out="$(dpkg -V proximadb)" || true
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
