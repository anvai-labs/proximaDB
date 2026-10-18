#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a command up to N times with a fixed pause; fail hard with the last exit
# code. For registry installs (pip/npm/docker pull) whose first attempt can hit
# a CDN that has not caught up with a publish.
#
# Usage: retry.sh <attempts> <sleep_seconds> <command...>
set -uo pipefail
ATTEMPTS="${1:?attempts}"; PAUSE="${2:?sleep}"; shift 2
rc=1
for i in $(seq 1 "$ATTEMPTS"); do
  "$@"
  rc=$?
  [ "$rc" -eq 0 ] && exit 0
  echo "attempt ${i}/${ATTEMPTS} failed (exit ${rc}): $*" >&2
  [ "$i" -lt "$ATTEMPTS" ] && sleep "$PAUSE"
done
echo "::error::gave up after ${ATTEMPTS} attempts: $*" >&2
exit "$rc"
