#!/bin/bash
# Pre-install script for ProximaDB package (build-time attrs, TD-PKG-1).
#
# Creates the proximadb system user BEFORE payload extraction. The package
# payload now carries the contract state dirs with their FINAL owner/mode
# (rpm --rpm-attr), and rpm resolves the recorded names at extract time —
# without %pre the user does not exist yet, the dirs fall back to root, and
# the clean rpm -V gate can never pass. Wired as --before-install on BOTH
# legs (rpm %pre and deb preinst both run before unpack). Idempotent:
# on upgrade the user already exists and nothing happens.
set -e

if ! id proximadb &>/dev/null; then
    useradd --system --user-group --home-dir /var/lib/proximadb --shell /sbin/nologin proximadb
    echo "Created system user 'proximadb'"
fi
