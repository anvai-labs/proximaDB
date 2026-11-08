#!/bin/bash
# Pre-publish RPM verifier (release-review F4, TD-PKG-1 interaction).
# Run INSIDE the verify container; the cwd must contain proximadb-*.rpm.
# Self-contained here (not inlined in release.yml) so it is bash -n
# testable without burning a release dispatch.
set -euo pipefail

VER="${1:?usage: verify_rpm.sh <version> — cwd must hold the .rpm}"

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
unexpected="$(printf '%s\n' "$devs" \
    | grep -vE '^\.M\.\.\.UG\.\.[[:space:]]+/var/(lib|log)/proximadb$' || true)"
if [ -n "$unexpected" ]; then
    echo "unexpected rpm -V deviations:" >&2
    printf '%s\n' "$unexpected" >&2
    exit 1
fi

rpm -e proximadb
test ! -e /usr/bin/proximadb-server
echo "RPM verify PASSED (install, payload, version, metadata tolerance, erase)"
