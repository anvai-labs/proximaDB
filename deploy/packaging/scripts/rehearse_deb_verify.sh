#!/bin/bash
# Rehearsal of verify_deb.sh's REAL branch (F4 symmetry with the RPM
# rehearsal): stubbed apt-get/dpkg execute every line the container will
# run — install, payload, the TD-PKG-1 stat assertions (dirs created with
# the contract's owner/mode), clean dpkg -V, version, erase — on the
# runner, BEFORE the release dispatch. Expects the repo checkout as cwd.
set -euo pipefail

VERSION="${1:?usage: rehearse_deb_verify.sh <version>}"
REPO="$(pwd)"

mkdir -p /tmp/rehearse-deb
cat > /tmp/rehearse-deb/apt-get <<'APT'
#!/bin/bash
# "install": create the payload AND the TD-PKG-1 state dirs with the
# contract's owner/mode (mirrors postinstall.sh's %post behavior).
. deploy/packaging/scripts/verify_payload_common.sh
read_contract_dirs deploy/packaging/scripts/postinstall.sh
dirs="$CONTRACT_DIRS"
# the contract owner does not exist on a fresh runner
id proximadb &>/dev/null || sudo useradd --system --user-group --home-dir /var/lib/proximadb --shell /usr/sbin/nologin proximadb
mkdir -p /usr/bin /etc/proximadb /lib/systemd/system
printf '#!/bin/sh\necho "proximadb-server $VERIFY_REHEARSE_VERSION"\n' | sudo tee /usr/bin/proximadb-server >/dev/null
sudo chmod +x /usr/bin/proximadb-server
sudo touch /etc/proximadb/config.toml /lib/systemd/system/proximadb.service
for d in $dirs; do
    sudo mkdir -p "$d"
    sudo chown proximadb:proximadb "$d"
    sudo chmod 750 "$d"
done
APT
cat > /tmp/rehearse-deb/dpkg <<'DPKG'
#!/bin/bash
case "$1" in
  -V) exit 0 ;;   # the happy state: clean verify, no output
  -r) sudo rm -f /usr/bin/proximadb-server ;;
esac
DPKG
chmod +x /tmp/rehearse-deb/apt-get /tmp/rehearse-deb/dpkg

VERIFY_REHEARSE_VERSION="$VERSION" PATH="/tmp/rehearse-deb:$PATH" \
  bash "$REPO/deploy/packaging/scripts/verify_deb.sh" "$VERSION"
test ! -e /usr/bin/proximadb-server
echo "DEB REHEARSAL PASS: the real verify branch executed end-to-end"
