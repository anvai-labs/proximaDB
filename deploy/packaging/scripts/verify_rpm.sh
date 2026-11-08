#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
#
# MODES:
#   verify_rpm.sh <version>   the REAL verify branch (container path): dnf
#     install the .rpm from /pkg, assert payload + version, run the rpm -V
#     TD-PKG-1 gate, erase.
#   verify_rpm.sh selftest    host rehearsal (wired into build-rpm before
#     the container dispatch): replays the exact v0.6.0 failure output as
#     the fixture; VERIFY_RPM_DEV_FIXTURE overrides it for negative-path
#     testing (empty = the dead-%post simulation, which must be REJECTED).
#
# The TD-PKG-1 contract lives in postinstall.sh's CHOWN_DIRS (single
# source of truth, parsed by the shared read_contract_dirs helper).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=verify_payload_common.sh
. "$SCRIPT_DIR/verify_payload_common.sh"
POSTINSTALL_SRC="${POSTINSTALL_SRC:-$SCRIPT_DIR/postinstall.sh}"

MODE="${1:?usage: verify_rpm.sh <version>|selftest}"
SELFTEST=0
VER="$MODE"
case "$MODE" in
    selftest) SELFTEST=1 ;;
esac

read_contract_dirs "$POSTINSTALL_SRC"
contract_dirs="$CONTRACT_DIRS"

# rpm -V exits nonzero on ANY deviation. A nonzero rc with NO verify
# output is a hard failure (not-installed package, rpmdb error).
rc=0
devs=""
if [ "$SELFTEST" = "1" ]; then
    # Fixture GENERATED from the contract (CHOWN_DIRS edits cannot break
    # the selftest); VERIFY_RPM_DEV_FIXTURE overrides for negative paths.
    rc=1
    devs="${VERIFY_RPM_DEV_FIXTURE-$(for dir in $contract_dirs; do printf '.M...UG..    %s\n' "$dir"; done)}"
else
    dnf install -y /pkg/proximadb-*.rpm
    assert_common_payload /usr/lib/systemd/system/proximadb.service
    assert_version
    devs="$(rpm -V proximadb)" || true
    # An EMPTY output on a real install is suspicious (a dead %post, or a
    # hard rpm error) — the gate below fails it as 0-matched vs n_expected,
    # and this log line explains why.
fi

# Gate: EVERY contract dir must deviate with the TD-PKG-1 stateline
# (chmod 750 vs payload 0755 → M; chown user+group → U+G; 9-column
# rpm-4.14 stateline → trailing '.'), and NOTHING else may deviate. A
# dead %post (no chown) reports no deviations and fails here — that
# failure mode was the production incident F4 exists to prevent.
# (DESIGN NOTE: the gate mandates the deviations because CHOWN_DIRS is
# the switch — if the payload ever sets the dirs' attrs itself and
# postinstall drops the chown loop, CHOWN_DIRS disappears and this gate
# requires a CLEAN rpm -V instead.)
n_expected=0
n_matched=0
unexpected=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    matched=0
    for dir in $contract_dirs; do
        case "$line" in
            .M...UG..*"$dir") matched=1; break ;;
        esac
    done
    if [ "$matched" -eq 1 ]; then
        n_matched=$((n_matched + 1))
    else
        unexpected="$unexpected$line"$'\n'
    fi
done <<EOF3
$devs
EOF3
set -- $contract_dirs
n_expected=$#

if [ "$n_matched" -ne "$n_expected" ] || [ -n "$unexpected" ]; then
    echo "rpm -V deviations wrong: expected=$n_expected matched=$n_matched" >&2
    [ -n "$unexpected" ] && { echo "unexpected deviations:" >&2; printf '%s' "$unexpected" >&2; }
    exit 1
fi

if [ "$SELFTEST" = "1" ]; then
    echo "SELFTEST PASS: fixture tolerated ($n_matched/$n_expected contract dirs), no foreign deviations"
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, TD-PKG-1 tolerance, erase)"
