#!/usr/bin/env python3
"""Compare PAX exact/ANN GETs on the same SIFT prefixes and cache states.

Build the sift_pax_recall_ratchet_test binary with io-trace and the backend
feature. Pass its executable path and a TEXMEX dataset directory. A remote
storage URL receives new fixture objects under a unique run prefix.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

SCALE_ROWS = [1_000, 10_000, 33_000, 100_000, 333_000, 1_000_000]
RANGE_CAPS_MIB = [4, 8, 16, 24, 32]


def run_worker(binary, dataset, spec):
    env = {
        **os.environ,
        "PROXIMADB_PAX_SCALE_COST_RUN": json.dumps(spec),
        "PROXIMADB_SIFT_DATASET_DIR": str(dataset),
        "PROXIMADB_PAX_VECTOR_QUANT": "rabitq",
        "PROXIMADB_PAX_WRITE_RG_LAYOUT": "1",
        "PROXIMADB_PAX_WRITE_A0_TRAIN": "1",
        "PROXIMADB_PAX_COALESCED_RABITQ": "1",
        "PROXIMADB_PAX_COALESCE_RANGE": str(spec["cap_mib"] << 20),
        "PROXIMADB_PAX_COALESCE_GAP": str(1 << 20),
    }
    output = subprocess.run(
        [str(binary), "scale_cost::worker", "--exact", "--ignored", "--nocapture"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if output.returncode:
        raise RuntimeError(output.stdout)
    marker = "PAX_SCALE_COST "
    rows = [
        json.loads(line.split(marker, 1)[1])
        for line in output.stdout.splitlines()
        if marker in line
    ]
    expected = 2 if spec["action"] == "cold_repeat" else 1
    if len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} measurement(s), received {len(rows)}: {output.stdout}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new JSONL file")
    parser.add_argument(
        "--storage-url", help="backend prefix, default is temporary local files"
    )
    parser.add_argument("--rows", type=int, nargs="+", default=SCALE_ROWS)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--caps-mib", type=int, nargs="+", default=RANGE_CAPS_MIB)
    parser.add_argument(
        "--wire-markers",
        action="store_true",
        help="bracket queries with separate PUTs for an isolated emulator access log",
    )
    args = parser.parse_args()
    if not 1 <= args.queries <= 10_000 or not all(
        10 <= n <= 1_000_000 for n in args.rows
    ):
        parser.error("SIFT1M supports 10..1M base rows and 1..10K queries")
    if len(set(args.rows)) != len(args.rows):
        parser.error("row counts must be distinct")
    if (
        not args.caps_mib
        or min(args.caps_mib) < 1
        or len(set(args.caps_mib)) != len(args.caps_mib)
    ):
        parser.error("range caps must be distinct positive MiB values")
    with (
        tempfile.TemporaryDirectory(prefix="pax-scale-") as temp,
        args.output.open("x") as out,
    ):
        root = (
            (args.storage_url or Path(temp).as_uri()).rstrip("/")
            + "/"
            + uuid.uuid4().hex
        )

        def measure(spec, action, **params):
            run = dict(spec, action=action, **params)
            for row in run_worker(args.binary.resolve(), args.dataset.resolve(), run):
                row["cap_mib"] = run["cap_mib"]
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                out.flush()

        for n in args.rows:
            scratch = Path(temp) / str(n)
            scratch.mkdir()
            spec = {
                "storage": f"{root}/{n}",
                "scratch": str(scratch),
                "rows": n,
                "queries": args.queries,
                "query": 0,
                "route": "prepare",
                "budget_mib": 0,
                "cap_mib": args.caps_mib[0],
                "wire_markers": args.wire_markers,
            }

            measure(spec, "prepare")
            for q in range(args.queries):
                for route in ("exact", "ann") if q % 2 == 0 else ("ann", "exact"):
                    caps = args.caps_mib if route == "ann" else args.caps_mib[:1]
                    for cap in caps:
                        measure(spec, "cold_repeat", query=q, route=route, cap_mib=cap)
                        measure(spec, "reopen", query=q, route=route, cap_mib=cap)
                        shutil.rmtree(scratch / f"{route}-{q}")
            for budget in dict.fromkeys([0, *args.caps_mib]):
                measure(spec, "policy", route="adaptive", budget_mib=budget)
                shutil.rmtree(scratch / "adaptive-0")
            print(f"finished {n:,} rows", flush=True)


if __name__ == "__main__":
    main()
