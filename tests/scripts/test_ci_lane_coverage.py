# Copyright (C) 2026 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/check_ci_lane_coverage.py (release-review F2 guard).

The evasions these tests pin are the ones the first draft of the guard got
wrong, found the hard way on the real tree:

  * a path FILTER (`- 'tests/**'` under a changes/filters step) must NOT
    certify a test as executed — the first draft certified all 234 targets
    that way, exactly the vacuous-green shape;
  * coverage means a reference inside a `run:` block (--test, bare for-loop
    name, literal path, execution glob), nothing else;
  * the baseline is a two-way ratchet: a NEW uncovered test fails, and a
    baseline entry that has since been covered also fails.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import by path, not `from scripts import ...`: on some developer machines an
# anaconda editable install shadows the repo's `scripts` package (the known
# work-commit-check shadowing). CI is unaffected either way; this form works
# in both.
_GUARD_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_ci_lane_coverage.py"
)
_spec = importlib.util.spec_from_file_location("check_ci_lane_coverage", _GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

WORKFLOW_WITH_FILTER_ONLY = """
changes:
  filters:
    everything:
      - 'tests/**'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: cargo build
"""

WORKFLOW_WITH_RUN_LIST = """
jobs:
  lane:
    runs-on: ubuntu-latest
    steps:
      - name: run tests
        run: |
          for test_name in \\
            alpha_e2e \\
            beta_e2e; do
            cargo test --test "$test_name" -- --test-threads=1
          done
          cargo test --test gamma_wal_e2e
          for f in tests/*delta*.rs; do
            cargo test --test "$(basename "$f" .rs)"
          done
"""


class GuardTree:
    """A throwaway repo shape for the guard: tests/, workflows/, baseline."""

    def __init__(self, test_names, workflow_texts, baseline_text=""):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        tests = root / "tests"
        tests.mkdir()
        for name in test_names:
            (tests / name).write_text("// integration test\n", encoding="utf-8")
        workflows = root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        for index, text in enumerate(workflow_texts):
            (workflows / f"w{index}.yml").write_text(text, encoding="utf-8")
        scripts = root / "scripts"
        scripts.mkdir()
        self.baseline = scripts / "ci_lane_baseline.txt"
        self.baseline.write_text(baseline_text, encoding="utf-8")

    def patch(self):
        return mock.patch.multiple(
            guard,
            TESTS_DIR=self.root / "tests",
            WORKFLOWS_DIR=self.root / ".github" / "workflows",
            BASELINE=self.baseline,
        )


class RunBlockExtraction(unittest.TestCase):
    def test_filter_mention_does_not_certify(self):
        tree = GuardTree(["alpha_e2e.rs", "beta_e2e.rs"], [WORKFLOW_WITH_FILTER_ONLY])
        with tree.patch():
            text = guard.run_block_text()
            self.assertNotIn("tests/**", text)
            self.assertFalse(guard.is_covered("alpha_e2e", "alpha_e2e.rs", text, []))

    def test_run_block_forms_certify(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "beta_e2e.rs", "gamma_wal_e2e.rs", "delta_e2e.rs"],
            [WORKFLOW_WITH_RUN_LIST],
        )
        with tree.patch():
            text = guard.run_block_text()
            patterns = sorted(set(__import__("re").findall(r"(tests/[^\s'\";)]+)", text)))
            for stem, name in [
                ("alpha_e2e", "alpha_e2e.rs"),  # bare for-loop name
                ("beta_e2e", "beta_e2e.rs"),  # bare for-loop name
                ("gamma_wal_e2e", "gamma_wal_e2e.rs"),  # --test invocation
                ("delta_e2e", "delta_e2e.rs"),  # execution glob tests/*delta*.rs
            ]:
                self.assertTrue(
                    guard.is_covered(stem, name, text, patterns),
                    f"{stem} should be covered by the run list",
                )


class BaselineRatchet(unittest.TestCase):
    def test_new_uncovered_fails(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="alpha_missing.rs\n",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            self.assertEqual(guard.main(), 1)

    def test_stale_baseline_fails(self):
        # alpha_e2e is executed by the run list AND still on the baseline.
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="alpha_e2e.rs\norphan_e2e.rs\n",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            self.assertEqual(guard.main(), 1)

    def test_consistent_baseline_passes(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="orphan_e2e.rs\n",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            self.assertEqual(guard.main(), 0)

    def test_update_baseline_records_exactly_the_uncovered(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_a_e2e.rs", "orphan_b_e2e.rs"],
            [WORKFLOW_WITH_RUN_LIST],
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check", "--update-baseline"]):
            self.assertEqual(guard.main(), 0)
        self.assertEqual(
            tree.baseline.read_text(encoding="utf-8"),
            "orphan_a_e2e.rs\norphan_b_e2e.rs\n",
        )


if __name__ == "__main__":
    unittest.main()
