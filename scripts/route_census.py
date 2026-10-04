#!/usr/bin/env python3
"""ADR-094 wire-surface census: extract (method, path) pairs from axum
`.route(...)` registrations across both REST trees.

The snapshot committed at docs/openapi/route-census.txt is the reviewable
wire-surface instrument for the REST-tree convergence PRs: a convergence PR
is wire-identical iff re-running this script produces the same snapshot
(paths + methods; handlers may move between trees — the `source` column
records where each route lived at snapshot time).

Textual extraction is intentionally approximate (a few dynamic router
extensions may be missed); determinism is what matters — the same tree yields
the same census. Run: `python3 scripts/route_census.py` (writes stdout).
"""

import re
import sys
from pathlib import Path

TREES = [
    "src/network/rest",
    "crates/platform/proximadb-api/src/rest",
]

# `.route("path", get(handler).post(h2))` — path + every method chained.
ROUTE_RE = re.compile(
    r"\.route\(\s*\"(?P<path>[^\"]+)\"\s*,\s*(?P<body>[^;]*?)\)",
    re.DOTALL,
)
METHOD_RE = re.compile(
    r"\b(get|post|put|delete|patch|head|options)\s*\(", re.IGNORECASE
)


def census() -> list[str]:
    rows: set[str] = set()
    for tree in TREES:
        for rs_file in Path(tree).rglob("*.rs"):
            text = rs_file.read_text(encoding="utf-8", errors="replace")
            for m in ROUTE_RE.finditer(text):
                path = m.group("path")
                methods = sorted({mm.group(1).lower() for mm in METHOD_RE.finditer(m.group("body"))})
                for method in methods:
                    rows.add(f"{method}\t{path}\t{tree}")
    return sorted(rows)


def main() -> int:
    rows = census()
    for row in rows:
        print(row)
    print(f"# {len(rows)} route registrations", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
