#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" rockylinux:8 bash verify_rpm.sh <version>
# (plain `bash` — the script's own `set -euo pipefail` is the strictness
# authority). The .rpm must be at /pkg/proximadb-*.rpm. Self-contained
# here (not inlined in release.yml) so it is bash -n testable without
# burning a release dispatch. Self-test mode: VERIFY_RPM_SELFTEST=1
# skips dnf/install and replays the shipped rpm -V fixture.
set -euo pipefail

VER="${1:?usage: verify_rpm.sh <version> — the .rpm must be at /pkg/proximadb-*.rpm}"
POSTINSTALL_SRC="${POSTINSTALL_SRC:-/pkg/deploy/packaging/scripts/postinstall.sh}"

if [ "${VERIFY_RPM_SELFTEST:-0}" != "1" ]; then
    dnf install -y /pkg/proximadb-*.rpm
    test -x /usr/bin/proximadb-server
    test -f /etc/proximadb/config.toml
    test -f /usr/lib/systemd/system/proximadb.service
    got="$(/usr/bin/proximadb-server --version)"
    test "$got" = "proximadb-server $VER"
    rc=0
    devs="$(rpm -V proximadb)" || rc=$?
else
    # Fixture = the exact v0.6.0 failure output (run 42496624451) so the
    # gate below can be exercised without a container or an install.
    rc=1
    devs=$'.M...UG..    /var/lib/proximadb\n.M...UG..    /var/log/proximadb'
fi

# rpm -V exits nonzero on ANY deviation. A nonzero rc with NO verify
# output is a hard failure (not-installed package, rpmdb error).
if [ "$rc" -ne 0 ] && [ -z "$devs" ]; then
    echo "rpm -V failed with rc=$rc and no verify output" >&2
    exit 1
fi

# TD-PKG-1 contract: the %post scriptlet deliberately chowns/chmods the
# state/log dirs (chmod 750 vs the payload's 0755 → mode deviation; chown
# user+group → U+G deviations), which on a 9-column rpm-4.14 stateline
# renders as `.M...UG..` on each chowned dir. The expected dir set is
# derived FROM postinstall.sh — the source of truth — so the two cannot
# drift apart. A dead %post (no chown) reports no deviations and MUST
# fail here: that failure mode was the production incident F4 prevents.
expected_dirs="$(grep -E '^chown ' "$POSTINSTALL_SRC" | grep 'proximadb' | awk '{print $NF}' | sed 's:/*$::' || true)"
if [ -z "$expected_dirs" ]; then
    echo "no chown targets found in $POSTINSTALL_SRC — verifier contract broken" >&2
    exit 1
fi

n_expected=0
n_matched=0
unexpected=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    matched=0
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        case "$line" in
            *"$dir") case "$line" in .M...UG..*) matched=1; break;; esac ;;
        esac
    done <<EOF2
$expected_dirs
EOF2
    if [ "$matched" -eq 1 ]; then
        n_matched=$((n_matched + 1))
    else
        unexpected="$unexpected$line"$'\n'
    fi
done <<EOF3
$devs
EOF3
n_expected="$(printf '%s\n' "$expected_dirs" | wc -l)"

# Require EXACTLY the derived deviations: a dead %post reports none
# (the production incident), a foreign deviation is a packaging defect.
if [ "$n_matched" -ne "$n_expected" ] || [ -n "$unexpected" ]; then
    echo "rpm -V deviations wrong: expected=$n_expected matched=$n_matched" >&2
    [ -n "$unexpected" ] && { echo "unexpected deviations:" >&2; printf '%s' "$unexpected" >&2; }
    exit 1
fi

if [ "${VERIFY_RPM_SELFTEST:-0}" = "1" ]; then
    echo "SELFTEST PASS: the healthy fixture is tolerated (2/2), and the
contract gates are exercised above by the caller."
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, TD-PKG-1 tolerance, erase)"
