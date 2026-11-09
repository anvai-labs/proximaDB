#!/bin/bash
# Post-install script for ProximaDB package

set -e

# Create proximadb system user if it doesn't exist
if ! id proximadb &>/dev/null; then
    useradd --system --user-group --home-dir /var/lib/proximadb --shell /sbin/nologin proximadb
    echo "Created system user 'proximadb'"
fi

# TD-PKG-1: chown UNCONDITIONALLY. dpkg/rpm create the payload's state/log
# dirs as root before postinst runs, so the old "only when absent" guard
# left them root-owned and the service user could not write to its own
# state directories.
# TD-PKG-1: the contract lives in the SHIPPED data file /etc/proximadb/
# chown_dirs.env (deploy/packaging/chown_dirs.env in the repo — the single
# source of truth the pre-publish verifiers also read). chown/chmod stay
# unconditional: dpkg/rpm lay the payload dirs down root-owned.
# (BUILD-TIME follow-up: shipping the dirs with final owner/mode in the
# staging tree would retire this mutation contract — the chown loop then
# serves only as the upgrade path for pre-existing root-owned dirs.)
if [ -f /etc/proximadb/chown_dirs.env ]; then
    # shellcheck disable=SC1091  # shipped by this package
    . /etc/proximadb/chown_dirs.env
else
    echo "chown_dirs.env missing from the package — packaging defect" >&2
    exit 1
fi

for dir in $CHOWN_DIRS; do
    mkdir -p "$dir"
    chown proximadb:proximadb "$dir"
    chmod 750 "$dir"
    echo "State dir $dir ready (owned by proximadb)"
done

# Reload systemd and enable service
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload
    systemctl enable proximadb.service
    echo "Enabled proximadb systemd service"
fi

echo "ProximaDB installation complete!"
echo "Start the service with: systemctl start proximadb"
