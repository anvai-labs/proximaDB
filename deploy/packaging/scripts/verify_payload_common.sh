#!/bin/bash
# Sourced by verify_rpm.sh and verify_deb.sh: the payload/version asserts
# shared by both package legs. Kept in ONE place — the last drift between
# the two copies produced an unbound-variable failure on the release path.
# Expects $VER set by the sourcing script.

assert_common_payload() {
    # $1 = the systemd unit path (differs: rpm → /usr/lib, deb → /lib).
    test -x /usr/bin/proximadb-server
    test -f /etc/proximadb/config.toml
    test -f "$1"
}

assert_version() {
    got="$(/usr/bin/proximadb-server --version)"
    test "$got" = "proximadb-server $VER"
}

# The TD-PKG-1 contract dirs, read from postinstall.sh's CHOWN_DIRS with
# the strict parse both legs share (exactly one assignment, double-quoted
# value). Sets CONTRACT_DIRS. $1 = the postinstall.sh path.
read_contract_dirs() {
    local src="${1:?usage: read_contract_dirs <postinstall.sh>}"
    local lines
    lines="$(grep -E '^CHOWN_DIRS=' "$src" || true)"
    [ -n "$lines" ] || { echo "cannot read CHOWN_DIRS from $src" >&2; return 1; }
    [ "$(printf '%s\n' "$lines" | wc -l)" -eq 1 ] || { echo "multiple CHOWN_DIRS lines in $src" >&2; return 1; }
    case "$lines" in
        'CHOWN_DIRS="'*'"') ;;
        *) echo "CHOWN_DIRS must be a double-quoted assignment: $lines" >&2; return 1 ;;
    esac
    CONTRACT_DIRS="${lines#CHOWN_DIRS=\"}"
    CONTRACT_DIRS="${CONTRACT_DIRS%\"}"
    [ -n "$CONTRACT_DIRS" ] || { echo "CHOWN_DIRS is empty — the TD-PKG-1 contract is gone; refuse to verify (a dead %post would pass silently)" >&2; return 1; }
}
