#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" rockylinux:8 bash verify_rpm.sh <version>
# (plain `bash` — the script's own `set -euo pipefail` is the strictness
# authority). The .rpm must be at /pkg/proximadb-*.rpm. Self-test mode:
#   VERIFY_RPM_SELFTEST=1 bash verify_rpm.sh <version>
# runs on the HOST (no container): POSTINSTALL_SRC defaults to the
# postinstall.sh next to this script, and the rpm -V fixture replays the
# exact v0.6.0 failure output (run 42496624451). VERIFY_RPM_DEV_FIXTURE
# overrides the fixture for negative-path testing.
set -euo pipefail

MODE="${1:?usage: verify_rpm.sh <version>|selftest}"
case "$MODE" in
    selftest) SELFTEST=1 ;;
    *) [ -n "${VERIFY_RPM_SELFTEST:-}" ] && SELFTEST=1 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POSTINSTALL_SRC="${POSTINSTALL_SRC:-$SCRIPT_DIR/postinstall.sh}"

# The TD-PKG-1 contract, read from its single source of truth. Strict
# parse: exactly one assignment, optional double quotes, no trailing
# comment — anything else is a contract edit this verifier refuses to
# guess at (fail loud at boot, not mid-release).
rc=0
contract="$(grep -cE '^CHOWN_DIRS=' "$POSTINSTALL_SRC" >/dev/null && grep -E '^CHOWN_DIRS=' "$POSTINSTALL_SRC")" || rc=$?
if [ "$rc" -ne 0 ] || [ -z "$contract" ]; then
    echo "cannot read CHOWN_DIRS from $POSTINSTALL_SRC (grep rc=$rc)" >&2
    exit 1
fi
[ "$(printf '%s\n' "$contract" | wc -l)" -eq 1 ] || { echo "multiple CHOWN_DIRS lines in $POSTINSTALL_SRC" >&2; exit 1; }
case "$contract" in
    'CHOWN_DIRS="'*'"'|'CHOWN_DIRS='*) ;;
    *) echo "CHOWN_DIRS is not a plain quoted/plain assignment: $contract" >&2; exit 1 ;;
esac
contract_dirs="${contract#CHOWN_DIRS=}"; contract_dirs="${contract_dirs//\"/}"

if [ "${SELFTEST:-0}" = "1" ]; then
    # Fixture = the exact v0.6.0 failure output (run 42496624451).
    rc=1
    # `-` (not `:-`): an explicitly EMPTY fixture is honored — it is the
    # dead-%post simulation (rc≠0 + no output = hard failure branch).
    devs="${VERIFY_RPM_DEV_FIXTURE-$(printf '.M...UG..    /var/lib/proximadb\n.M...UG..    /var/log/proximadb')}"
else
    dnf install -y /pkg/proximadb-*.rpm
    test -x /usr/bin/proximadb-server
    test -f /etc/proximadb/config.toml
    test -f /usr/lib/systemd/system/proximadb.service
    got="$(/usr/bin/proximadb-server --version)"
    test "$got" = "proximadb-server $VER"
    rc=0
    devs="$(rpm -V proximadb)" || rc=$?
fi

# rpm -V exits nonzero on ANY deviation. A nonzero rc with NO verify
# output is a hard failure (not-installed package, rpmdb error).
if [ "$rc" -ne 0 ] && [ -z "$devs" ]; then
    echo "rpm -V failed with rc=$rc and no verify output" >&2
    exit 1
fi

# Gate: EVERY contract dir must appear among the deviations with the
# TD-PKG-1 stateline (chmod 750 vs payload 0755 → M; chown user+group →
# U+G; 9-column rpm-4.14 stateline → trailing '.'), and NOTHING else may
# deviate. A dead %post reports no deviations and fails here — that
# failure mode was the production incident F4 exists to prevent.
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

if [ "${SELFTEST:-0}" = "1" ]; then
    echo "SELFTEST PASS: fixture tolerated ($n_matched/$n_expected contract dirs), no foreign deviations"
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, TD-PKG-1 tolerance, erase)"
