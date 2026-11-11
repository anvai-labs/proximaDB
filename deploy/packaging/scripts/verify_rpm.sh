#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
#
# MODES:
#   verify_rpm.sh <version>   the REAL verify branch (container path): dnf
#     install the .rpm from /pkg, assert payload + version, run the CLEAN
#     rpm -V gate + the TD-PKG-1 dir stat-asserts, erase.
#   verify_rpm.sh selftest    host rehearsal (wired into build-rpm before
#     the container dispatch): the positive fixture is a CLEAN rpm -V
#     (accepted); VERIFY_RPM_DEV_FIXTURE provides a non-empty output the
#     gate must REJECT (any deviation shape — the gate keys on emptiness).
#
# The TD-PKG-1 contract lives in deploy/packaging/chown_dirs.env (single
# source of truth, parsed by the shared read_contract_dirs helper; it
# SHIPS as /etc/proximadb/chown_dirs.env — postinstall.sh and the
# stat-asserts below read the installed copy).
#
# GATE (build-time attrs, 2026-11-10): the payload carries the contract
# state dirs with their FINAL owner/mode (fpm --rpm-attr, driven from
# chown_dirs.env; the user is created in %pre by preinstall.sh before
# extraction), so a correct install shows a CLEAN rpm -V — ANY deviation
# is a foreign modification of a shipped file (or a broken build that lost
# the attrs) and fails with the deviation list. The dir attrs themselves
# are stat-asserted directly via the shared assert_td_pkg1_dirs — the same
# depth as the deb leg. This retires the deviation-matching gate that
# required the %post chown to have run (the v0.6.0/0.6.1 design); that
# gate's own design note anticipated exactly this flip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=verify_payload_common.sh
. "$SCRIPT_DIR/verify_payload_common.sh"

MODE="${1:?usage: verify_rpm.sh <version>|selftest}"
SELFTEST=0
# shellcheck disable=SC2034  # VER is consumed by assert_version() in verify_payload_common.sh (sourced below)
VER="$MODE"
case "$MODE" in
    selftest) SELFTEST=1 ;;
esac

if [ "$SELFTEST" = "1" ]; then
    # host: read the repo contract file
    read_contract_dirs "$SCRIPT_DIR/../chown_dirs.env"
else
    # container: read the SHIPPED contract file via the /pkg mount (its
    # installed copy at /etc/proximadb is what the stat-asserts check)
    read_contract_dirs /pkg/deploy/packaging/chown_dirs.env
fi
# read_contract_dirs doubles as the CONTRACT PARSE GATE (strict single
# double-quoted assignment; non-empty): a malformed or emptied contract
# file fails the verify before any package is installed. The dirs
# themselves are consumed inside assert_td_pkg1_dirs.

devs=""
if [ "$SELFTEST" = "1" ]; then
    # Positive fixture = CLEAN output (a correct build-time-attrs install).
    # VERIFY_RPM_DEV_FIXTURE overrides for the negative paths: ANY non-empty
    # output — the lost-attrs statelines, a subset, a foreign digest — must
    # be REJECTED (the gate keys on emptiness, no parsing).
    devs="${VERIFY_RPM_DEV_FIXTURE-}"
else
    dnf install -y /pkg/proximadb-*.rpm
    assert_common_payload /usr/lib/systemd/system/proximadb.service
    assert_version
    devs="$(rpm -V proximadb)" || true
fi

# Gate 1: rpm -V must be CLEAN — the payload owns the attrs now, so any
# deviation is foreign modification (or lost --rpm-attr in the build).
if [ -n "$devs" ]; then
    echo "rpm -V reported deviations on a build-time-attrs install —" >&2
    echo "the payload's final owner/mode was not applied (lost --rpm-attr?)," >&2
    echo "or a shipped file was modified post-install. Deviations:" >&2
    printf '%s\n' "$devs" >&2
    exit 1
fi

# Gate 2: the installed contract dirs must carry the TD-PKG-1 attrs
# (owner+group proximadb, mode 750) — stat-asserted directly, the same
# depth as the deb leg (rpm -V cannot see dirs on some rpm builds; the
# stat-assert does not depend on rpm's directory coverage at all).
# Selftest skips this (no install on the host) — the container branch of
# the release job exercises it for real on rockylinux:8.
if [ "$SELFTEST" != "1" ]; then
    assert_td_pkg1_dirs /etc/proximadb/chown_dirs.env
fi

if [ "$SELFTEST" = "1" ]; then
    echo "SELFTEST PASS: clean rpm -V accepted, any non-empty output rejected by construction"
    exit 0
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, clean rpm -V, TD-PKG-1 dir attrs, erase)"
