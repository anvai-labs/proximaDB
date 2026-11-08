#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
# Run INSIDE the verify container with the repo root mounted at /pkg:
#   docker run ... -v "$PWD:/pkg:ro" rockylinux:8 bash -euo pipefail \
#     /pkg/deploy/packaging/scripts/verify_rpm.sh <version>
# The .rpm must be at /pkg/proximadb-*.rpm (the mount, not the cwd).
# Self-contained here (not inlined in release.yml) so it is bash -n
# testable without burning a release dispatch.
set -euo pipefail

VER="${1:?usage: verify_rpm.sh <version> — the .rpm must be at /pkg/proximadb-*.rpm}"

dnf install -y /pkg/proximadb-*.rpm
test -x /usr/bin/proximadb-server
test -f /etc/proximadb/config.toml
test -f /usr/lib/systemd/system/proximadb.service
got="$(/usr/bin/proximadb-server --version)"
test "$got" = "proximadb-server $VER"

# rpm -V exits nonzero on ANY deviation, INCLUDING the two by-design ones
# (the %post scriptlet deliberately chowns/chmods the state/log dirs,
# TD-PKG-1). rpm 4.14 (el8) emits a 9-column stateline (trailing P for
# file capabilities). So: a nonzero rc with NO verify output is a hard
# failure; among printed deviations, exactly the two expected ones are
# tolerated and anything else fails.
rc=0
devs="$(rpm -V proximadb)" || rc=$?
if [ "$rc" -ne 0 ] && [ -z "$devs" ]; then
    echo "rpm -V failed with rc=$rc and no verify output" >&2
    exit 1
fi
# Derive the expected deviations FROM postinstall.sh (the source of truth
# for the TD-PKG-1 contract) instead of hard-coding the dir list: every
# `chown proximadb... DIR` line means rpm -V will show mode+user+group
# deviation (chmod 750 vs the payload's 0755) for that dir on a 9-column
# rpm-4.14 stateline: `.M...UG..`.
expected_dirs="$(grep -E '^chown proximadb(:proximadb)? ' \
    /pkg/deploy/packaging/scripts/postinstall.sh | awk '{print $NF}' || true)"
if [ -z "$expected_dirs" ]; then
    echo "no chown targets found in postinstall.sh — verifier contract broken" >&2
    exit 1
fi
n_expected=0
n_matched=0
while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    n_expected=$((n_expected + 1))
    line="$(printf '%s\n' "$devs" | grep "[[:space:]]${dir}\$" || true)"
    if [ -n "$line" ] && printf '%s' "$line" | grep -qE '^\.M\.\.\.UG\.\.'; then
        n_matched=$((n_matched + 1))
    else
        echo "expected TD-PKG-1 deviation MISSING for $dir" >&2
    fi
done <<EOF2
$expected_dirs
EOF2
# Anything in rpm -V output that is NOT one of the expected dirs fails.
filter=""
while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    filter="$filter -e ${dir}\$"
done <<EOF2
$expected_dirs
EOF2
unexpected="$(printf '%s\n' "$devs" | grep -v $filter || true)"
if [ "$n_matched" -ne "$n_expected" ] || [ -n "$unexpected" ]; then
    echo "rpm -V deviations wrong: expected=$n_expected matched=$n_matched; unexpected:" >&2
    printf '%s\n' "$unexpected" >&2
    [ -z "$devs" ] || printf '%s\n' "$devs" >&2
    exit 1
fi
n_expected=0
n_matched=0
while IFS= read -r dir; do
    n_expected=$((n_expected + 1))
    line="$(printf '%s\n' "$devs" | grep -E "[[:space:]]${dir}\$" || true)"
    if [ -n "$line" ] && printf '%s' "$line" | grep -qE '^\.M\.\.\.UG\.\.'; then
        n_matched=$((n_matched + 1))
    else
        echo "expected TD-PKG-1 deviation MISSING for $dir" >&2
    fi
done <<EOF2
$expected_dirs
EOF2
unexpected="$(printf '%s\n' "$devs" | grep -vE "$(printf '%s\\n' "$expected_dirs" \
    | sed 's|/|\\/|g; s|^|^[[:space:]]|; s|$|\$|' | paste -sd'|' -)" || true)"
if [ "$n_matched" -ne "$n_expected" ] || [ -n "$unexpected" ]; then
    echo "rpm -V deviations wrong: expected=$n_expected matched=$n_matched; unexpected:" >&2
    printf '%s\n' "$unexpected" >&2
    [ -z "$devs" ] || printf '%s\n' "$devs" >&2
    exit 1
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, metadata tolerance, erase)"
