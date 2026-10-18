#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the PUBLISHED embedded engine (`pip install proximadb_embedded==X.Y.Z`):
version, then the graph + typed-property roundtrip release-python.yml runs on
the freshly built wheel BEFORE publishing -- here it runs on the wheel PyPI
actually serves AFTER publishing.

Usage: embedded_smoke.py <expected_version> [--out evidence.json]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import pathlib
import sys
import tempfile


def run(version: str) -> dict:
    installed = importlib.metadata.version("proximadb_embedded")
    assert installed == version, f"installed proximadb_embedded {installed} != {version}"
    import proximadb_embedded as pe

    db = pe.ProximaDB(data_dirs=tempfile.mkdtemp(prefix="rs-embedded-"))
    db.create_graph("smoke", None)
    db.create_nodes(
        "smoke",
        [
            pe.GraphNode("a", ["Fn"], {"name": "alpha"}),
            pe.GraphNode("b", ["Fn"], {"name": "beta"}),
        ],
    )
    db.create_edges("smoke", [pe.GraphEdge("a", "b", "CALLS", id="e1")])
    out = db.get_outgoing_edges("smoke", "a", None)
    assert len(out) == 1 and out[0].to_node_id == "b", out
    inc = db.get_incoming_edges("smoke", "b", None)
    assert len(inc) == 1 and inc[0].from_node_id == "a", inc
    stats = db.graph_stats("smoke")
    assert stats.total_edges == 1, stats

    db.create_nodes(
        "smoke",
        [pe.GraphNode("typed", ["Fn"], {
            "line": 42,
            "score": 0.5,
            "flag": True,
            "name": "alpha",
            "tags": ["a", "b"],
        })],
    )
    got = db.get_node("smoke", "typed").properties
    assert got["line"] == 42 and isinstance(got["line"], int), got
    assert got["score"] == 0.5 and isinstance(got["score"], float), got
    assert got["flag"] is True, got
    assert got["name"] == "alpha", got
    assert got["tags"] == ["a", "b"], got
    hits = db.query_nodes("smoke", None, {"line": 42}, None, None)
    assert [n.id for n in hits] == ["typed"], [n.id for n in hits]
    return {
        "artifact": "pypi:proximadb_embedded",
        "installed_version": installed,
        "python": sys.version.split()[0],
        "module_path": pe.__file__,
        "steps": ["graph_roundtrip", "typed_properties", "query_nodes"],
        "passed": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    try:
        report = run(args.version)
    except AssertionError as exc:
        print(f"::error::embedded smoke FAILED: {exc}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
