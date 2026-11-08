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
