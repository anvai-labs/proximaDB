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
#   rehearse_rpm_verify.sh    (sibling script) rehearses the REAL branch
#     on the runner with stubbed dnf/rpm.
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

# rpm -V exits nonzero on ANY deviation (rc=1) — captured below. A hard
# error (not-installed package, rpmdb error) fails via the empty-output
# branch with its own precise message.
devs=""
if [ "$SELFTEST" = "1" ]; then
    # Fixture GENERATED from the contract (CHOWN_DIRS edits cannot break
    # the selftest); VERIFY_RPM_DEV_FIXTURE overrides it for negative
    # paths (empty string = the dead-%post simulation).
    devs="${VERIFY_RPM_DEV_FIXTURE-$(for dir in $contract_dirs; do td_pkg1_stateline_for "$dir"; done)}"
else
    dnf install -y /pkg/proximadb-*.rpm
    assert_common_payload /usr/lib/systemd/system/proximadb.service
    assert_version
    devs="$(rpm -V proximadb)" || true
    # An EMPTY output on a real install is a HARD failure — a dead %post
    # (the production incident F4 exists to catch) or a hard rpm error.
    # Fail with the precise message instead of a confusing count mismatch.
    if [ -z "$devs" ]; then
        echo "rpm -V reported NO deviations on a real install — %post dead or rpmdb error; refusing" >&2
        exit 1
    fi
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
# REQUIRE: each contract dir has EXACTLY one TD-PKG-1 deviation (counting
# per dir, not per line — a duplicated line cannot mask an unmatched dir),
# and every rpm -V line maps to a contract dir (no foreign deviations).
set -- $contract_dirs
n_expected=$#
n_matched=0
unexpected=""
for dir in $contract_dirs; do
    line="$(printf '%s\n' "$devs" | grep -F "    $dir" || true)"
    count="$(printf '%s\n' "$line" | grep -c . || true)"
    if [ "$count" -ne 1 ]; then
        echo "contract dir $dir: expected exactly one TD-PKG-1 deviation, got $count" >&2
        [ -n "$line" ] && printf '%s\n' "$line" >&2
        exit 1
    fi
    path="$(printf '%s' "$line" | sed -E "s/^\.M\.\.\.UG\.\. +//")"
    case "$path" in "$dir") ;; *) echo "deviation for $dir has foreign path: $path" >&2; exit 1 ;; esac
    n_matched=$((n_matched + 1))
done
while IFS= read -r line; do
    [ -n "$line" ] || continue
    path="$(printf '%s' "$line" | sed -E 's/^[.A-Z]+ +//')"
    known=0
    for dir in $contract_dirs; do case "$path" in "$dir") known=1; break ;; esac; done
    [ "$known" -eq 1 ] || { echo "unexpected deviation:" >&2; printf '%s\n' "$line" >&2; unexpected=1; }
done <<EOF3
$devs
EOF3
[ -z "$unexpected" ] || { echo "unexpected deviations present" >&2; exit 1; }

if [ "$SELFTEST" = "1" ]; then
    echo "SELFTEST PASS: fixture tolerated ($n_matched/$n_expected contract dirs), no foreign deviations"
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, TD-PKG-1 tolerance, erase)"
