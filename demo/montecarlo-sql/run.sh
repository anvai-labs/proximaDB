#!/usr/bin/env bash
# Copyright (C) 2025 ProximaDB
# SPDX-License-Identifier: Apache-2.0
#
# Monte Carlo option pricing as an in-database compute workload — driven two ways over
# the PostgreSQL wire protocol against a REAL ProximaDB:
#   1. the SQL editor  (montecarlo.sql,          via psql)
#   2. a notebook      (montecarlo_notebook.py,  via psycopg2)
#
# Prerequisites — a ProximaDB built from THIS code, with pgwire enabled:
#   cargo build --release -p proximadb-server        # or: make build-server
#   ./target/release/proximadb-server -c config/config.toml   # pgwire on 127.0.0.1:5433
#   # pgwire is on by default (`[api] enable_pgwire`); do NOT set `pg_port` to enable it.
#   # Run the binary locally: in the published container pgwire is loopback-only, so a
#   # published 5433 accepts nothing and moving it off loopback fails closed at boot
#   # (TD-PGWIRE-CONTAINER-1).
#
# Unlike the cross-modal demo this needs ONLY pgwire — no REST seeding. The `mc_price` UDF
# lives on the DataFusion compute engine, so a query must engage OLAP (an aggregate over a
# MATERIALIZE'd table); the SQL/notebook do exactly that.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_HOST="${PROXIMADB_PG_HOST:-127.0.0.1}"
PG_PORT="${PROXIMADB_PG_PORT:-5433}"
DB="${PROXIMADB_DB:-quant}"   # the pgwire database == the tenant/catalog
CONN="host=${PG_HOST} port=${PG_PORT} user=postgres dbname=${DB} sslmode=disable"

echo "▶ ProximaDB Monte Carlo compute demo (pgwire ${PG_HOST}:${PG_PORT}, db ${DB})"

# The two surfaces run the SAME workload independently; give each its own pgwire database
# (the database == the tenant/catalog) so they don't share the `option_book` table.
echo; echo "── montecarlo.sql : the SQL-editor workload (price + value + stress a book) ──"
psql "host=${PG_HOST} port=${PG_PORT} user=postgres dbname=${DB} sslmode=disable" -f "${HERE}/montecarlo.sql"

echo; echo "── montecarlo_notebook.py : the notebook workload (same pgwire, + a risk curve) ──"
PROXIMADB_PG_HOST="${PG_HOST}" PROXIMADB_PG_PORT="${PG_PORT}" PROXIMADB_DB="${DB}_nb" \
    python3 "${HERE}/montecarlo_notebook.py"

echo; echo "✓ done — the Monte Carlo simulation ran in-database (compute-to-data)."
