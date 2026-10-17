#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derive an isolated config for the released server binary from the config/
# it ships with (config/config.toml in the archive).
#
# The shipped config is a local-dev one: every storage URL is under
# /tmp/proximadb and the unified listener is `unified_port` (default 5678) --
# which `-p/--port` does NOT override in unified mode. A smoke that runs it
# unmodified shares /tmp/proximadb with anything else on the machine and binds
# 5678. This rewrites only the locations and ports; everything else is the
# shipped default, so the smoke still exercises the config users get.
#
# Usage: binary_config.sh <shipped_config.toml> <data_dir> <rest_port>
#   gRPC/Flight sidecar ports are rest_port+1/+2, pgwire is rest_port-4247
#   (47680 -> 43433), kept on loopback as the shipped config requires.
set -euo pipefail
SRC="${1:?shipped config}"; DATA="${2:?data dir}"; PORT="${3:?rest port}"
[ -f "$SRC" ] || { echo "::error::no shipped config at ${SRC}"; exit 1; }
case "$DATA" in /*) ;; *) echo "::error::data dir must be absolute"; exit 1 ;; esac
for key in unified_mode rest_port grpc_port arrow_flight_port; do
  grep -q "^${key} = " "$SRC" || { echo "::error::shipped config has no '${key} = ' line; update binary_config.sh"; exit 1; }
done
PG=$((PORT - 4247))
args=(-e "s#/tmp/proximadb#${DATA}#g"
      -e "s/^port = [0-9]*/port = ${PORT}/"
      -e "s/^rest_port = [0-9]*/rest_port = ${PORT}/"
      -e "s/^grpc_port = [0-9]*/grpc_port = $((PORT + 1))/")
# Replace a key the shipped config already sets; add it only when absent, so a
# future config that sets unified_port/pg_port does not get a duplicate key.
if grep -q '^unified_port = ' "$SRC"; then
  args+=(-e "s/^unified_port = [0-9]*/unified_port = ${PORT}/")
else
  args+=(-e "s/^unified_mode = true/unified_mode = true\nunified_port = ${PORT}/")
fi
if grep -q '^pg_port = ' "$SRC"; then
  args+=(-e "s/^pg_port = [0-9]*/pg_port = ${PG}/"
         -e "s/^arrow_flight_port = [0-9]*/arrow_flight_port = $((PORT + 2))/")
else
  args+=(-e "s/^arrow_flight_port = [0-9]*/arrow_flight_port = $((PORT + 2))\npg_port = ${PG}/")
fi
sed "${args[@]}" "$SRC"
