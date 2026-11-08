#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" rockylinux:8 bash verify_rpm.sh <version>
# (plain `bash` — the script's own `set -euo pipefail` is the strictness
# authority). The .rpm must be at /pkg/proximadb-*.rpm.
#
# MODES:
#   verify_rpm.sh <version>   the REAL verify branch (container path).
#   verify_rpm.sh selftest    host rehearsal: replays the exact v0.6.0
#     failure output as the fixture and requires the gate to tolerate
#     exactly the contract deviations — wired into build-rpm BEFORE the
#     container dispatch, so gate regressions die in CI, not at release.
# VERIFY_RPM_DEV_FIXTURE overrides the selftest fixture (negative-path
# testing: empty = dead-%post simulation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=verify_payload_common.sh
. "$SCRIPT_DIR/verify_payload_common.sh"
POSTINSTALL_SRC="${POSTINSTALL_SRC:-$SCRIPT_DIR/postinstall.sh}"
NEWLINE=$'
'

# The TD-PKG-1 contract, read from its single source of truth. Exactly one
# assignment, optional double quotes — anything else fails loud here
# rather than corrupting the expected-deviation set mid-release.
rc=0
contract="$(grep -E '^CHOWN_DIRS=' "$POSTINSTALL_SRC" || true)"
if [ -z "$contract" ]; then
    echo "cannot read CHOWN_DIRS from $POSTINSTALL_SRC" >&2
    exit 1
fi
[ "$(printf '%s\n' "$contract" | wc -l)" -eq 1 ] || {
    echo "multiple CHOWN_DIRS lines in $POSTINSTALL_SRC" >&2
    exit 1
}
case "$contract" in
    'CHOWN_DIRS="'*'"') ;;
    *) echo "CHOWN_DIRS must be a double-quoted assignment: $contract" >&2; exit 1 ;;
esac
contract_dirs="${contract#CHOWN_DIRS=\"}"; contract_dirs="${contract_dirs%\"}"

# The TD-PKG-1 contract: the %post scriptlet deliberately chowns/chmods the
# state/log dirs (chmod 750 vs payload 0755 → M; chown user+group → U+G;
# 9-column rpm-4.14 stateline → trailing '.'), rendering `.M...UG..` per
# contract dir. The selftest fixture is GENERATED from the contract, so
# CHOWN_DIRS edits cannot break it; VERIFY_RPM_DEV_FIXTURE overrides it
# for negative-path testing (empty = the dead-%post simulation).
if [ "${1:-}" = "selftest" ]; then
    SELFTEST=1
    rc=1
    devs="${VERIFY_RPM_DEV_FIXTURE-$(for dir in $contract_dirs; do printf '.M...UG..    %s\n' "$dir"; done)}"
    devs="${devs%"$NEWLINE"}"
else
    SELFTEST=0
    VER="${1:?real-branch usage: verify_rpm.sh <version>}"
    dnf install -y /pkg/proximadb-*.rpm
    assert_common_payload /usr/lib/systemd/system/proximadb.service
    assert_version
    rc=0
    devs="$(rpm -V proximadb)" || rc=$?
fi

NEWLINE=$'\n'
# rpm -V exits nonzero on ANY deviation. A nonzero rc with NO verify
# output is a hard failure (not-installed package, rpmdb error).
if [ "$rc" -ne 0 ] && [ -z "$devs" ]; then
    echo "rpm -V failed with rc=$rc and no verify output" >&2
    exit 1
fi

# Gate: EVERY contract dir must deviate with the TD-PKG-1 stateline, and
# NOTHING else may deviate. A dead %post reports no deviations and fails
# here — that failure mode was the production incident F4 exists to
# prevent. (DESIGN NOTE: the gate mandates the deviations because
# CHOWN_DIRS is the switch — if the payload ever sets the dirs' attrs and
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
