# Copyright (C) 2026 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/coverage_ratchet.py (2026-11-20 CI review hardening).

The false-greens these tests pin were reproduced against the real gate:

  * a current coverage report of `{}` exited ZERO against the committed
    baseline — "nothing regressed" is not coverage evidence, it is no
    measurement;
  * a compile-only / zero-denominator report normalized to a number;
  * Rust file attribution used the "any path containing src/" heuristic, so
    `apps/*/src` and `clients/rust/src` would have been swallowed into the
    root bucket had the scope expanded;
  * baseline components absent from a current report only warned — an
    invocation scoped to one surface must be able to REQUIRE its component.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_RATCHET_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "coverage_ratchet.py"
)
_spec = importlib.util.spec_from_file_location("coverage_ratchet", _RATCHET_PATH)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def write_json(tmp, name: str, doc) -> Path:
    root = Path(tmp.name) if isinstance(tmp, tempfile.TemporaryDirectory) else Path(tmp)
    path = root / name
    if isinstance(doc, str):
        path.write_text(doc, encoding="utf-8")
    else:
        path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def llvm_cov_doc(files: list[tuple[str, int, int]]) -> dict:
    """files: (filename, covered, total)."""
    return {
        "data": [
            {
                "files": [
                    {
                        "filename": name,
                        "summary": {"lines": {"covered": covered, "count": total}},
                    }
                    for name, covered, total in files
                ]
            }
        ]
    }


def run_check(baseline: dict, current: dict, require: str = "") -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "baseline.json"
        cur = Path(d) / "current.json"
        base.write_text(json.dumps(baseline), encoding="utf-8")
        cur.write_text(json.dumps(current), encoding="utf-8")
        args = argparse.Namespace(
            baseline=str(base), current=str(cur), require_components=require
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ratchet.cmd_check(args)
    return code, buf.getvalue()


class NormalizeRust(unittest.TestCase):
    def test_attribution_by_repo_relative_paths(self):
        # llvm-cov emits ABSOLUTE paths under the workspace (as CI does); the
        # relative_to(workspace_root) step is what makes attribution safe.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = llvm_cov_doc(
                [
                    (str(root / "src" / "lib.rs"), 900, 1000),
                    (
                        str(
                            root
                            / "crates"
                            / "storage"
                            / "proximadb-storage"
                            / "src"
                            / "inner.rs"
                        ),
                        40,
                        50,
                    ),
                    (
                        str(
                            root
                            / "apps"
                            / "proximadb-server"
                            / "src"
                            / "main.rs"
                        ),
                        80,
                        100,
                    ),
                ]
            )
            path = write_json(tmp=d, name="cov.json", doc=doc)
            out = ratchet.normalize_rust(path, root)
        self.assertEqual(out["rust:proximadb"], 90.0)
        self.assertEqual(out["rust:proximadb-storage"], 80.0)
        self.assertEqual(out["rust:proximadb-server"], 80.0)

    def test_absolute_checkout_prefix_does_not_become_a_crate(self):
        # llvm-cov emits absolute paths under the checkout; after making them
        # repo-relative the checkout dir (…/proximaDB/) must not be mistaken
        # for a `<crate-dir>/src` owner of the root crate. This mirrors CI,
        # where --workspace-root IS the checkout the report was produced in.
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d) / "work" / "proximaDB" / "proximaDB"
            src = ws / "src"
            src.mkdir(parents=True)
            (src / "lib.rs").write_text("fn main() {}\n", encoding="utf-8")
            doc = llvm_cov_doc([(str(src / "lib.rs"), 500, 1000)])
            path = write_json(tmp=d, name="cov.json", doc=doc)
            out = ratchet.normalize_rust(path, ws)
        self.assertEqual(out, {"rust:proximadb": 50.0})

    def test_apps_and_clients_keep_their_own_buckets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = llvm_cov_doc(
                [
                    (str(root / "src" / "lib.rs"), 100, 100),
                    (str(root / "clients" / "rust" / "src" / "genrest.rs"), 10, 100),
                    (
                        str(root / "apps" / "proximadb-bench" / "src" / "bench.rs"),
                        5,
                        100,
                    ),
                ]
            )
            path = write_json(tmp=d, name="cov.json", doc=doc)
            out = ratchet.normalize_rust(path, root)
        self.assertNotIn("rust:proximadb:clients", out)
        self.assertEqual(out["rust:rust"], 10.0)
        self.assertEqual(out["rust:proximadb-bench"], 5.0)

    def test_malformed_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(tmp=d, name="cov.json", doc="{not json")
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_rust(path)

    def test_empty_data_block_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(tmp=d, name="cov.json", doc={"data": []})
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_rust(path)

    def test_no_files_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(tmp=d, name="cov.json", doc={"data": [{"files": []}]})
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_rust(path)

    def test_no_attributable_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            doc = llvm_cov_doc([("/somewhere/else/other.rs", 1, 10)])
            path = write_json(tmp=d, name="cov.json", doc=doc)
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_rust(path)

    def test_out_of_workspace_files_are_skipped_not_attributed(self):
        # A registry/vendored path must not mint a spurious component (e.g.
        # `rust:registry`) whose below-target coverage would fail the gate as
        # a phantom "new component below target".
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = llvm_cov_doc(
                [
                    (str(root / "src" / "lib.rs"), 100, 100),
                    (
                        "/home/runner/.cargo/registry/src/index.crates.io/somecrate/lib.rs",
                        1,
                        10,
                    ),
                ]
            )
            path = write_json(tmp=d, name="cov.json", doc=doc)
            out = ratchet.normalize_rust(path, root)
        self.assertEqual(out, {"rust:proximadb": 100.0})

    def test_missing_report_file_is_usage_error_not_traceback(self):
        with self.assertRaises(ratchet.MalformedReport):
            ratchet.normalize_rust(Path("/nonexistent/llvm-cov.json"))
        with self.assertRaises(ratchet.MalformedReport):
            ratchet.normalize_sdk(Path("/nonexistent/sdk-cov.json"))


class NormalizeSdk(unittest.TestCase):
    def test_percent_covered(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(
                tmp=d,
                name="sdk.json",
                doc={"totals": {"percent_covered": 81.234}},
            )
            out = ratchet.normalize_sdk(path)
        self.assertEqual(out, {"sdk:proximadb-python": 81.23})

    def test_derived_from_covered_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(
                tmp=d,
                name="sdk.json",
                doc={"totals": {"covered_lines": 800, "missing_lines": 200}},
            )
            out = ratchet.normalize_sdk(path)
        self.assertEqual(out, {"sdk:proximadb-python": 80.0})

    def test_zero_denominator_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(
                tmp=d,
                name="sdk.json",
                doc={"totals": {"covered_lines": 0, "missing_lines": 0}},
            )
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_sdk(path)

    def test_missing_totals_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(tmp=d, name="sdk.json", doc={})
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_sdk(path)

    def test_malformed_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(tmp=d, name="sdk.json", doc="[")
            with self.assertRaises(ratchet.MalformedReport):
                ratchet.normalize_sdk(path)


class CheckGate(unittest.TestCase):
    def test_empty_current_report_fails(self):
        # THE false-green: `{}` used to exit zero against any baseline.
        code, out = run_check({"sdk:proximadb-python": 80.0}, {})
        self.assertEqual(code, 1)
        self.assertIn("EMPTY", out)

    def test_required_component_absent_fails(self):
        code, out = run_check(
            {"sdk:proximadb-python": 80.0, "rust:proximadb": 70.0},
            {"rust:proximadb": 70.0},
            require="sdk:proximadb-python",
        )
        self.assertEqual(code, 1)
        self.assertIn("REQUIRED component(s) absent", out)

    def test_required_component_present_passes(self):
        code, _ = run_check(
            {"sdk:proximadb-python": 80.0},
            {"sdk:proximadb-python": 80.0},
            require="sdk:proximadb-python",
        )
        self.assertEqual(code, 0)

    def test_regression_fails(self):
        code, out = run_check(
            {"sdk:proximadb-python": 80.0}, {"sdk:proximadb-python": 75.0}
        )
        self.assertEqual(code, 1)
        self.assertIn("REGRESSIONS", out)

    def test_within_tolerance_passes(self):
        code, _ = run_check(
            {"sdk:proximadb-python": 80.0}, {"sdk:proximadb-python": 79.7}
        )
        self.assertEqual(code, 0)

    def test_new_component_below_target_fails(self):
        code, out = run_check({}, {"rust:newcrate": 50.0})
        self.assertEqual(code, 1)
        self.assertIn("NEW component(s) below", out)

    def test_missing_baseline_component_warns_without_requirement(self):
        code, out = run_check(
            {"sdk:proximadb-python": 80.0, "rust:legacy": 90.0},
            {"sdk:proximadb-python": 80.0},
        )
        self.assertEqual(code, 0)
        self.assertIn("WARNING", out)

    def test_malformed_current_is_usage_error(self):
        with tempfile.TemporaryDirectory() as d:
            cur = Path(d) / "current.json"
            cur.write_text("not json", encoding="utf-8")
            args = argparse.Namespace(
                baseline=str(Path(d) / "b.json"), current=str(cur),
                require_components="",
            )
            (Path(d) / "b.json").write_text("{}", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = ratchet.cmd_check(args)
        self.assertEqual(code, 2)

    def test_write_refuses_empty_current(self):
        with tempfile.TemporaryDirectory() as d:
            cur = Path(d) / "current.json"
            cur.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                current=str(cur), baseline=str(Path(d) / "baseline.json")
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = ratchet.cmd_write(args)
        self.assertEqual(code, 2)
        self.assertFalse((Path(d) / "baseline.json").exists())


class CommandLine(unittest.TestCase):
    """End-to-end through main() so the exit codes CI observes are pinned."""

    def _main(self, argv: list[str]) -> int:
        buf = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            with mock.patch.object(sys, "argv", ["coverage_ratchet.py", *argv]):
                return ratchet.main()

    def test_normalize_rejects_zero_denominator_sdk_report(self):
        with tempfile.TemporaryDirectory() as d:
            sdk = write_json(
                tmp=d, name="sdk.json", doc={"totals": {"percent_covered": None}}
            )
            code = self._main(
                ["normalize", "--sdk", str(sdk), "-o", str(Path(d) / "out.json")]
            )
        self.assertEqual(code, 2)

    def test_normalize_requires_at_least_one_input(self):
        with tempfile.TemporaryDirectory() as d:
            code = self._main(["normalize", "-o", str(Path(d) / "out.json")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
