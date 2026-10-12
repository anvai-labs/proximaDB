#!/usr/bin/env bash
# Copyright (C) 2025 ProximaDB
# SPDX-License-Identifier: Apache-2.0
#
# Data-engineering demo: warehouse analytics + the zero-ETL cross-modal join, driven
# entirely over the PostgreSQL wire protocol against a REAL running ProximaDB.
#
# Prerequisites — a ProximaDB built from THIS code, run LOCALLY (not in a container):
#   cargo build --release -p proximadb-server        # or: make build-server
#   ./target/release/proximadb-server -c config/config.toml
#   # pgwire is already on at 127.0.0.1:5433 (`[api] enable_pgwire = true`); you do NOT
#   # need to set `pg_port` to enable it, and you should not: setting it re-derives the
#   # pgwire bind from `[server] bind_address`, which is only safe while that is loopback.
#   #
#   # Docker will NOT work for this demo. The published image reaches pgwire on the
#   # container's own loopback only, so `-p 5433:5433` accepts no connections, and
#   # setting `pg_port` to move it off loopback makes the server fail closed at boot
#   # (TD-PGWIRE-CONTAINER-1). Run the binary on the host, or drive psql from inside the
#   # container's network namespace. (A container also has to be built from THIS code —
#   # the cross-modal UDTFs + pgwire OLAP route only exist in a same-code build.)
#
# The cross-modal UDTFs read under the pgwire connection's tenant, and the catalog
# (the psql `dbname`) IS the tenant boundary — so we seed the vector/timeseries/graph
# data via REST under tenant "de_demo" and connect psql with dbname=de_demo. They MUST
# match (TD-XMODAL-6).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROXIMADB_REST="${PROXIMADB_REST:-http://127.0.0.1:5678}"
export PROXIMADB_TENANT="${PROXIMADB_TENANT:-de_demo}"
PG_HOST="${PROXIMADB_PG_HOST:-127.0.0.1}"
PG_PORT="${PROXIMADB_PG_PORT:-5433}"
CONN="host=${PG_HOST} port=${PG_PORT} user=postgres dbname=${PROXIMADB_TENANT} sslmode=disable"

echo "▶ ProximaDB data-engineering demo (pgwire ${PG_HOST}:${PG_PORT}, REST ${PROXIMADB_REST}, tenant ${PROXIMADB_TENANT})"

# 1) Seed the non-relational modalities via the v2 REST API under the tenant.
bash "${HERE}/seed.sh"

# 2) Warehouse analytics — CREATE / INSERT / MATERIALIZE / GROUP BY over pgwire.
echo; echo "── warehouse.sql : relational analytics (routes to the DataFusion OLAP engine) ──"
psql "${CONN}" -f "${HERE}/warehouse.sql"

# 3) The moat: relational ⋈ vector ⋈ timeseries ⋈ graph in ONE data-local SQL plan.
echo; echo "── crossmodal.sql : the zero-ETL four-way cross-modal join ──"
psql "${CONN}" -f "${HERE}/crossmodal.sql"

echo; echo "✓ done — one engine, one query plane, zero ETL."
