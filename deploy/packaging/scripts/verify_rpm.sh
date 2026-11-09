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
# The TD-PKG-1 contract lives in deploy/packaging/chown_dirs.env (single
# source of truth, parsed by the shared read_contract_dirs helper; it
# SHIPS as /etc/proximadb/chown_dirs.env — postinstall.sh sources it and
# verify_deb.sh stat-asserts the installed copy).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=verify_payload_common.sh
. "$SCRIPT_DIR/verify_payload_common.sh"

MODE="${1:?usage: verify_rpm.sh <version>|selftest}"
SELFTEST=0
VER="$MODE"
case "$MODE" in
    selftest) SELFTEST=1 ;;
esac

if [ "$SELFTEST" = "1" ]; then
    # host: read the repo contract file
    read_contract_dirs "$SCRIPT_DIR/../chown_dirs.env"
else
    # container: read the SHIPPED contract file
    read_contract_dirs /pkg/deploy/packaging/chown_dirs.env
fi
contract_dirs="$CONTRACT_DIRS"

# rpm -V prints deviations with rc=1 (discarded — the gate below keys on
# the OUTPUT). A hard error (not-installed package, rpmdb error) produces
# empty output and fails via the empty-output branch with its own
# precise message.
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
        echo "rpm -V reported NO deviations on a real install — one of:" >&2
        echo "  - the %post scriptlet died before its chown (dead %post)," >&2
        echo "  - the payload stopped shipping the contract dirs entirely, or" >&2
        echo "  - a hard rpm error. Refusing; check the dnf/install output above." >&2
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
# requires a CLEAN rpm -V instead. Each contract dir must have EXACTLY
# one TD-PKG-1 deviation (counted per dir, not per line — a duplicated
# line cannot mask an unmatched dir), and the total line count must equal
# the contract count (no foreign deviations).
set -- $contract_dirs
n_expected=$#
n_matched=0
unexpected=""
# The expected line, built from the ONE stateline constant (grep -F with
# the full expected line + the end anchor below = no substring games).
stateline_escaped="$(printf '%s' "$TD_PKG1_STATELINE" | sed 's/[.]/[.]/g')"
for dir in $contract_dirs; do
    expected_line="$(td_pkg1_stateline_for "$dir")"
    line="$(printf '%s\n' "$devs" | grep -F "$expected_line" | grep -E " $dir\$" || true)"
    count="$(printf '%s\n' "$line" | grep -c . || true)"
    if [ "$count" -ne 1 ]; then
        echo "contract dir $dir: expected exactly one TD-PKG-1 deviation, got $count" >&2
        [ -n "$line" ] && printf '%s\n' "$line" >&2
        exit 1
    fi
    path="$(printf '%s' "$line" | sed -E "s/^$stateline_escaped +//")"
    case "$path" in "$dir") ;; *) echo "deviation for $dir has foreign path: $path" >&2; exit 1 ;; esac
    n_matched=$((n_matched + 1))
done
# No foreign deviations: each contract dir was validated above to own
# exactly one deviation line, so the TOTAL line count must equal the
# contract count (avoids strip-and-inspect over arbitrary deviation
# shapes — a digest '5' or 'missing' line cannot be mis-parsed here).
total_lines="$(printf '%s\n' "$devs" | grep -c . || true)"
[ "$total_lines" -eq "$n_expected" ] || {
    echo "foreign/unexpected rpm -V deviations present (total=$total_lines, contract=$n_expected):" >&2
    printf '%s\n' "$devs" >&2
    exit 1
}

if [ "$SELFTEST" = "1" ]; then
    echo "SELFTEST PASS: fixture tolerated ($n_matched/$n_expected contract dirs), no foreign deviations"
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, TD-PKG-1 tolerance, erase)"
