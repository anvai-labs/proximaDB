#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the PUBLISHED Python SDK (`pip install proximadb==X.Y.Z`) against a
live server: version, health, create, insert, search, delete, drop.

Release evidence, part (B). Run it from a venv that holds only the registry
install -- never with the repo's `clients/python/src` on sys.path -- so what is
tested is what users download. Hard-fails (exit 1) on any deviation.

Usage: pypi_sdk_smoke.py <base_url> <expected_version> [--out evidence.json]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import pathlib
import sys
import time
import uuid


def _retry(fn, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return fn()
        except AssertionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def run(base_url: str, version: str) -> dict:
    installed = importlib.metadata.version("proximadb")
    assert installed == version, f"installed proximadb {installed} != {version}"
    import proximadb_sdk

    assert proximadb_sdk.__file__ and "site-packages" in proximadb_sdk.__file__, (
        f"proximadb_sdk imported from {proximadb_sdk.__file__}, not an installed wheel"
    )
    assert proximadb_sdk.__version__ == version, (
        f"proximadb_sdk.__version__ {proximadb_sdk.__version__} != {version}"
    )
    client = proximadb_sdk.ProximaDBClient(url=base_url, protocol="rest")
    steps: list[str] = []
    try:
        health = client.health()
        assert health.status == "healthy", health
        assert health.version == version, f"server reports {health.version}, expected {version}"
        steps.append("health")

        name = f"rs_py_{uuid.uuid4().hex[:12]}"
        client.create_collection(name, dimension=4, distance_metric="cosine")
        steps.append("create_collection")

        resp = client.insert(
            name,
            vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ids=["a", "b", "c"],
            metadata=[{"k": "x"}, {"k": "y"}, {"k": "z"}],
        )
        assert sorted(resp.vector_ids or []) == ["a", "b", "c"], resp
        steps.append("insert")

        def top_is_a():
            hits = client.search(name, [1.0, 0.0, 0.0, 0.0], top_k=3)
            ids = [h.id for h in hits]
            assert ids and ids[0] == "a", f"search top-1 should be 'a', got {ids}"
            assert set(ids) == {"a", "b", "c"}, f"search should return all 3, got {ids}"

        _retry(top_is_a)
        steps.append("search")

        resp = client.delete(name, ["a"])
        assert resp.success, resp
        steps.append("delete")

        def a_is_gone():
            ids = [h.id for h in client.search(name, [1.0, 0.0, 0.0, 0.0], top_k=3)]
            assert "a" not in ids, f"deleted id 'a' still returned by search: {ids}"
            assert set(ids) == {"b", "c"}, f"expected b,c after delete, got {ids}"

        _retry(a_is_gone)
        steps.append("search_after_delete")

        assert client.delete_collection(name), "delete_collection returned False"
        steps.append("delete_collection")
    finally:
        client.close()
    return {
        "artifact": "pypi:proximadb",
        "installed_version": installed,
        "python": sys.version.split()[0],
        "module_path": proximadb_sdk.__file__,
        "steps": steps,
        "passed": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("version")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    try:
        report = run(args.base_url, args.version)
    except AssertionError as exc:
        print(f"::error::python SDK smoke FAILED: {exc}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
