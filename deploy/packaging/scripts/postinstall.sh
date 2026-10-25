#!/bin/bash
# Post-install script for ProximaDB package

set -e

# Create proximadb system user if it doesn't exist
if ! id proximadb &>/dev/null; then
    useradd --system --user-group --home-dir /var/lib/proximadb --shell /sbin/nologin proximadb
    echo "Created system user 'proximadb'"
fi

# TD-PKG-1: chown UNCONDITIONALLY. dpkg/rpm create the payload's
# /var/lib/proximadb and /var/log/proximadb as root before postinst runs, so
# the old "only when absent" guard left them root-owned and the service user
# could not write to its own state directories.
mkdir -p /var/lib/proximadb
chown proximadb:proximadb /var/lib/proximadb
chmod 750 /var/lib/proximadb
echo "Data directory /var/lib/proximadb ready"

mkdir -p /var/log/proximadb
chown proximadb:proximadb /var/log/proximadb
chmod 750 /var/log/proximadb
echo "Log directory /var/log/proximadb ready"

# Reload systemd and enable service
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload
    systemctl enable proximadb.service
    echo "Enabled proximadb systemd service"
fi

echo "ProximaDB installation complete!"
echo "Start the service with: systemctl start proximadb"
