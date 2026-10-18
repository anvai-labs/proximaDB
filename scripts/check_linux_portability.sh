#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fail unless the Linux release binaries run on every distro the packages
# claim: glibc symbol versions <= MAX_GLIBC (default 2.28 = RHEL/Rocky/Alma 8,
# Debian 10, Ubuntu 20.04) and no dynamic libssl/libcrypto (the OpenSSL major
# differs per distro: 1.1 on EL8, 3 on EL9/Ubuntu 22.04+).
#
# v0.4.1 shipped binaries needing GLIBC_2.35 + libssl.so.3 inside an
# `.el8.rpm` that could not start on Rocky 8 or 9.
#
# Usage: check_linux_portability.sh <dir-with-binaries> [max_glibc]
set -euo pipefail
DIR="${1:?directory holding proximadb-server, -bench, -migrate}"
MAX_GLIBC="${2:-2.28}"
fail=0
for bin in proximadb-server proximadb-bench proximadb-migrate; do
  f="$DIR/$bin"
  [ -x "$f" ] || { echo "::error::$f is missing"; fail=1; continue; }
  need="$(objdump -T "$f" | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
  ssl="$(objdump -p "$f" | awk '$1 == "NEEDED" && $2 ~ /^lib(ssl|crypto)\.so/ {print $2}' | tr '\n' ' ')"
  echo "$bin: needs glibc ${need:-?}; system libssl/libcrypto: ${ssl:-none}"
  if [ -z "$need" ] || [ "$(printf '%s\n%s\n' "$need" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
    echo "::error::$bin needs GLIBC_${need:-?} > ${MAX_GLIBC} -- build with scripts/build_portable_linux.sh"; fail=1
  fi
  if [ -n "$ssl" ]; then
    echo "::error::$bin links the system ${ssl}-- build with the vendored-openssl feature"; fail=1
  fi
done
exit "$fail"
