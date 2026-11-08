#!/bin/bash
# Rehearsal of verify_rpm.sh's REAL branch (release-review F4): stubbed
# dnf/rpm execute every line the container will run — install, payload,
# version, the TD-PKG-1 gate, erase — on the runner, BEFORE the release
# dispatch. The stub rpm -V output is GENERATED from postinstall.sh's
# CHOWN_DIRS (single source of truth; no second copy of the dir list).
# Expects the repo checkout as cwd. Runs the payload writes via sudo
# (root-owned system paths, GitHub runners are unprivileged).
set -euo pipefail

VERSION="${1:?usage: rehearse_rpm_verify.sh <version>}"
REPO="$(pwd)"

# --- stubs -------------------------------------------------------------
REHEARSE_DIR=$(mktemp -d /tmp/rehearse.XXXXXXX)
trap 'rm -rf "$REHEARSE_DIR"' EXIT
cat > $REHEARSE_DIR/dnf <<'DNF'
#!/bin/bash
sudo mkdir -p /usr/bin /etc/proximadb /usr/lib/systemd/system
printf '#!/bin/sh\necho "proximadb-server $VERIFY_REHEARSE_VERSION"\n' | sudo tee /usr/bin/proximadb-server >/dev/null
sudo chmod +x /usr/bin/proximadb-server
sudo touch /etc/proximadb/config.toml /usr/lib/systemd/system/proximadb.service
DNF
cat > $REHEARSE_DIR/rpm <<'RPM'
#!/bin/bash
case "$1" in
  -V)
    # The contract via the SAME strict parser the real gate uses (the repo
    # checkout is the cwd) — a malformed CHOWN_DIRS fails the rehearsal
    # exactly as it would fail the container.
    . deploy/packaging/scripts/verify_payload_common.sh
    read_contract_dirs deploy/packaging/scripts/postinstall.sh
    for d in $CONTRACT_DIRS; do printf '.M...UG..    %s\n' "$d"; done
    ;;
  -e) sudo rm -f /usr/bin/proximadb-server ;;
esac
RPM
chmod +x $REHEARSE_DIR/dnf $REHEARSE_DIR/rpm

# --- real-branch rehearsal ---------------------------------------------
VERIFY_REHEARSE_VERSION="$VERSION" PATH="/tmp/rehearse:$PATH" \
  bash "$REPO/deploy/packaging/scripts/verify_rpm.sh" "$VERSION"
test ! -e /usr/bin/proximadb-server
echo "REHEARSAL PASS: the real verify branch executed end-to-end"
