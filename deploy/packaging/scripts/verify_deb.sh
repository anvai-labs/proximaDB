#!/bin/bash
# Pre-publish DEB verifier (release-review F4 symmetry with verify_rpm.sh).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" debian:11 bash verify_deb.sh <version>
# (plain `bash` — the script's own `set -euo pipefail` is the strictness
# authority). The .deb must be at /pkg/proximadb_*_amd64.deb.
#
# MODES (same shape as verify_rpm.sh):
#   verify_deb.sh <version>   the REAL verify branch (container path).
#   verify_deb.sh selftest    host rehearsal: the happy state is a CLEAN
#     `dpkg -V` (its deviations print but exit 0), so the positive fixture
#     is empty output; VERIFY_DEB_DEV_FIXTURE provides a non-empty output
#     the gate must REJECT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=verify_payload_common.sh
. "$SCRIPT_DIR/verify_payload_common.sh"
POSTINSTALL_SRC="${POSTINSTALL_SRC:-$SCRIPT_DIR/postinstall.sh}"

if [ "${1:-}" = "selftest" ]; then
    SELFTEST=1
    VER="selftest"
    rc=0
    out="${VERIFY_DEB_DEV_FIXTURE-}"
else
    SELFTEST=0
    VER="${1:?real-branch usage: verify_deb.sh <version>}"
    apt-get update -qq
    apt-get install -y -qq /pkg/proximadb_*_amd64.deb
    assert_common_payload /lib/systemd/system/proximadb.service
    # TD-PKG-1 asserted at the SAME depth as the RPM leg: dpkg -V is blind
    # to dirs/ownership/modes, so the contract dirs are stat-ed directly.
    contract="$(grep -E '^CHOWN_DIRS=' "$POSTINSTALL_SRC" || true)"
    [ -n "$contract" ] || { echo "cannot read CHOWN_DIRS from $POSTINSTALL_SRC" >&2; exit 1; }
    contract_dirs="${contract#CHOWN_DIRS=\"}"; contract_dirs="${contract_dirs%\"}"
    for dir in $contract_dirs; do
        owner="$(stat -c %U "$dir")"
        mode="$(stat -c %a "$dir")"
        [ "$owner" = "proximadb" ] || { echo "TD-PKG-1: $dir owner is $owner, not proximadb" >&2; exit 1; }
        [ "$mode" = "750" ] || { echo "TD-PKG-1: $dir mode is $mode, not 750" >&2; exit 1; }
    done
    assert_version
    # Two dpkg -V semantics (verified on debian:11 / dpkg 1.20; re-pin at
    # each debian bump): deviations print to stdout with rc=0, and
    # dirs/ownership/modes are invisible. So the gate REQUIRES empty
    # output, and a nonzero rc with output still fails.
    rc=0
    out="$(dpkg -V proximadb)" || rc=$?
fi

if [ "$rc" -ne 0 ] && [ -z "$out" ]; then
    echo "dpkg -V failed with rc=$rc and no verify output" >&2
    exit 1
fi
if [ -n "$out" ]; then
    echo "dpkg -V reported deviations (contract requires clean):" >&2
    printf '%s\n' "$out" >&2
    exit 1
fi

if [ "$SELFTEST" = "1" ]; then
    echo "SELFTEST PASS: clean dpkg -V accepted, non-empty output rejected by construction"
    exit 0
fi

dpkg -r proximadb
test ! -e /usr/bin/proximadb-server
echo "DEB verify PASSED (install, payload, TD-PKG-1 ownership/mode, md5-clean, version, erase)"
