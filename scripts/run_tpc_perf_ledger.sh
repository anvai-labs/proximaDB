#!/usr/bin/env bash
# Run the advisory TPC ledger in a fail-closed systemd/cgroup memory envelope.
#
# Large scales also require:
#   TPC_PERF_ALLOW_LARGE_SCALE=I_ACCEPT_ISOLATED_OOM
# Override the default command by placing a command after `--`.

set -euo pipefail

memory_high="${TPC_PERF_MEMORY_HIGH:-48G}"
memory_max="${TPC_PERF_MEMORY_MAX:-64G}"
swap_max="${TPC_PERF_SWAP_MAX:-8G}"

validate_size() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*[KMGT]$ ]]; then
    echo "ERROR: $name must be a positive systemd size such as 48G; got '$value'" >&2
    exit 2
  fi
}

validate_size TPC_PERF_MEMORY_HIGH "$memory_high"
validate_size TPC_PERF_MEMORY_MAX "$memory_max"
validate_size TPC_PERF_SWAP_MAX "$swap_max"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "ERROR: systemd-run is required; refusing an unbounded benchmark" >&2
  exit 2
fi
if [[ ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
  echo "ERROR: cgroup v2 is required; refusing an unbounded benchmark" >&2
  exit 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  if (( $# == 0 )); then
    echo "ERROR: -- must be followed by a command" >&2
    exit 2
  fi
  benchmark_command=("$@")
else
  benchmark_command=(
    cargo test --release --test tpc_perf_ledger_e2e -- --ignored --nocapture
  )
fi

echo "TPC ledger envelope: MemoryHigh=$memory_high MemoryMax=$memory_max MemorySwapMax=$swap_max" >&2
exec systemd-run --user --scope --quiet --collect \
  --unit="proximadb-tpc-perf-$$" \
  --property=MemoryAccounting=yes \
  --property="MemoryHigh=$memory_high" \
  --property="MemoryMax=$memory_max" \
  --property="MemorySwapMax=$swap_max" \
  "${benchmark_command[@]}"
