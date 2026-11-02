#!/usr/bin/env python3
"""Measure per-test LLVM code-region overlap for a selected Rust function.

Build the test binary with cargo llvm-cov first. This reports coverage only;
assertions and input cases must also be compared before pruning a test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def covered_regions(document: dict, source: Path, line: int) -> set[tuple[int, ...]]:
    regions = set()
    found = False
    for block in document["data"]:
        for function in block.get("functions", []):
            files = function["filenames"]
            own = [
                r for r in function["regions"] if Path(files[r[5]]).resolve() == source
            ]
            if not any(r[0] == line and r[7] == 0 for r in own):
                continue
            found = True
            regions.update(tuple(r[:4]) for r in own if r[4] > 0 and r[7] == 0)
    if not found or not regions:
        raise ValueError(
            "selected function has no executed code regions in this profile"
        )
    return regions


def run_test(binary: Path, test: str, raw: Path) -> None:
    env = dict(os.environ, LLVM_PROFILE_FILE=str(raw))
    result = subprocess.run(
        [str(binary), "--exact", test],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    # libtest succeeds even when a misspelled filter matches zero tests.
    if not re.search(r"test result: ok\. 1 passed; 0 failed;", result.stdout):
        raise ValueError(f"expected exactly one passing test: {test}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.test) < 2 or len(set(args.test)) != len(args.test):
        parser.error("provide at least two distinct --test names")
    source = args.source.resolve()
    declarations = [
        n
        for n, text in enumerate(source.read_text().splitlines(), 1)
        if re.search(r"\bfn\s+" + re.escape(args.function) + r"\s*(?:<|\()", text)
    ]
    if len(declarations) != 1:
        parser.error("function name must identify exactly one declaration in --source")
    sysroot = subprocess.check_output(
        ["rustc", "--print", "sysroot"], text=True
    ).strip()
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(
        line.removeprefix("host: ")
        for line in version.splitlines()
        if line.startswith("host: ")
    )
    llvm = Path(sysroot) / "lib/rustlib" / host / "bin"
    measurements = {}
    with tempfile.TemporaryDirectory(prefix="proximadb-test-overlap-") as temp:
        for index, test in enumerate(args.test):
            raw = Path(temp) / f"{index}.profraw"
            profile = Path(temp) / f"{index}.profdata"
            run_test(args.binary.resolve(), test, raw)
            subprocess.run(
                [
                    str(llvm / "llvm-profdata"),
                    "merge",
                    "-sparse",
                    str(raw),
                    "-o",
                    str(profile),
                ],
                check=True,
            )
            exported = subprocess.check_output(
                [
                    str(llvm / "llvm-cov"),
                    "export",
                    str(args.binary.resolve()),
                    f"-instr-profile={profile}",
                    str(source),
                ]
            )
            measurements[test] = covered_regions(
                json.loads(exported), source, declarations[0]
            )
    pairs = []
    for first, second in itertools.combinations(measurements, 2):
        left, right = measurements[first], measurements[second]
        pairs.append(
            {
                "first": first,
                "second": second,
                "shared_regions": len(left & right),
                "first_only_regions": len(left - right),
                "second_only_regions": len(right - left),
            }
        )
    result = {
        "function": args.function,
        "source": str(args.source),
        "tests": {name: sorted(regions) for name, regions in measurements.items()},
        "pairs": pairs,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
