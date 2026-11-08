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
# CHOWN_DIRS is the single source of truth for this contract: the RPM
# pre-publish verifier derives its expected rpm -V deviations from this
# variable (the DEB leg instead requires a CLEAN dpkg -V, which ignores
# dirs entirely). Edits here update the RPM packaging contract.
CHOWN_DIRS="/var/lib/proximadb /var/log/proximadb"

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
