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

import contextlib
import importlib.util
import io
import re
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
            ROOT=self.root,
            TESTS_DIR=self.root / "tests",
            WORKFLOWS_DIR=self.root / ".github" / "workflows",
            BASELINE=self.baseline,
        )


class RunBlockExtraction(unittest.TestCase):
    def test_filter_mention_does_not_certify(self):
        tree = GuardTree(["alpha_e2e.rs", "beta_e2e.rs"], [WORKFLOW_WITH_FILTER_ONLY])
        with tree.patch():
            text, _ = guard.run_block_text()
            self.assertNotIn("tests/**", text)
            self.assertFalse(guard.is_covered("alpha_e2e", "alpha_e2e.rs", text, []))

    def test_run_block_forms_certify(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "beta_e2e.rs", "gamma_wal_e2e.rs", "delta_e2e.rs"],
            [WORKFLOW_WITH_RUN_LIST],
        )
        with tree.patch():
            text, _ = guard.run_block_text()
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


class InlineForLoop(unittest.TestCase):
    """`for t in a b; do` on one line must certify every listed stem (it was
    invisible: the api-surface lane's targets were baselined as never-run)."""

    def test_inline_for_list_certifies_each_word(self):
        text = (
            "for test_name in tst_integration_test external_collection_index_in_place_test; do\n"
            '  cargo test --test "$test_name"\n'
            "done\n"
        )
        for stem in ("tst_integration_test", "external_collection_index_in_place_test"):
            self.assertTrue(guard.is_covered(stem, f"{stem}.rs", text, []), stem)

    def test_word_outside_a_for_header_does_not_certify(self):
        text = 'echo "skipping tst_integration_test for now"\nfor t in other_test; do\n'
        self.assertFalse(guard.is_covered("tst_integration_test", "tst_integration_test.rs", text, []))
        self.assertFalse(guard.is_covered("other", "other.rs", text, []))


class BaselineRatchet(unittest.TestCase):
    def test_new_uncovered_fails(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.main(), 1)

    def test_moved_embedded_target_remains_in_the_ratchet(self):
        tree = GuardTree(["alpha_e2e.rs"], [WORKFLOW_WITH_RUN_LIST])
        self.addCleanup(tree.tmp.cleanup)
        embedded = tree.root / "crates/binding/proximadb-embedded/tests"
        embedded.mkdir(parents=True)
        target = embedded / "new_embedded_e2e.rs"
        target.write_text("// integration test\n", encoding="utf-8")
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertIn(target, guard.integration_targets())
                self.assertEqual(guard.main(), 1)
                workflow = tree.root / ".github/workflows/w0.yml"
                workflow.write_text(
                    WORKFLOW_WITH_RUN_LIST
                    + "          cargo test -p proximadb-embedded --test new_embedded_e2e\n",
                    encoding="utf-8",
                )
                self.assertEqual(guard.main(), 0)

    def test_stale_baseline_fails(self):
        # alpha_e2e is executed by the run list AND still on the baseline.
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="alpha_e2e.rs\norphan_e2e.rs\n",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.main(), 1)

    def test_consistent_baseline_passes(self):
        tree = GuardTree(
            ["alpha_e2e.rs", "orphan_e2e.rs"], [WORKFLOW_WITH_RUN_LIST],
            baseline_text="orphan_e2e.rs\n",
        )
        with tree.patch(), mock.patch.object(sys, "argv", ["check"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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


class EvasionClasses(unittest.TestCase):
    """The evasions the adversarial review demonstrated — pinned so they
    cannot return: shell comments must not certify, and neither may a
    comment-embedded glob."""

    COMMENT_ONLY_WORKFLOW = """
jobs:
  lane:
    runs-on: ubuntu-latest
    steps:
      - run: |
          # FIXME: tests/alpha_e2e.rs is not wired into any lane yet
          # the convergence suite (tests/*conv*.rs) still needs a lane
          cargo build
"""

    def test_comment_mentions_do_not_certify(self):
        tree = GuardTree(["alpha_e2e.rs", "conv_e2e.rs"], [self.COMMENT_ONLY_WORKFLOW])
        with tree.patch():
            text, _ = guard.run_block_text()
            patterns = sorted(set(re.findall(guard.TESTS_TOKEN_RE, text))) if hasattr(
                guard, "TESTS_TOKEN_RE") else sorted(set(re.findall(r"(tests/[^\s'\";)]+)", text)))
            self.assertFalse(guard.is_covered("alpha_e2e", "alpha_e2e.rs", text, patterns))
            self.assertFalse(guard.is_covered("conv_e2e", "conv_e2e.rs", text, patterns))


class CompileOnlyDoesNotCertify(unittest.TestCase):
    """2026-11-20 CI review: a COMPILE-ONLY invocation certified a target as
    executed. `cargo test --test example_e2e --no-run` (and `--list` /
    `nextest list` enumeration) proves the target is wired, never that it
    RAN — the exact shape that let a target sit on the burn-down baseline
    while the guard called it covered. The lines are dropped and counted."""

    WORKFLOW = """
jobs:
  lane:
    runs-on: ubuntu-latest
    steps:
      - name: compile check
        run: |
          cargo test --test compile_only_e2e --no-run
          cargo test --test list_only_e2e -- --list
          cargo nextest list --test nextest_list_only_e2e
      - name: actually run one
        run: cargo test --test real_run_e2e -- --test-threads=1
"""

    def test_compile_only_invocations_do_not_certify(self):
        tree = GuardTree(
            [
                "compile_only_e2e.rs",
                "list_only_e2e.rs",
                "nextest_list_only_e2e.rs",
                "real_run_e2e.rs",
            ],
            [self.WORKFLOW],
        )
        with tree.patch():
            text, excluded = guard.run_block_text()
            self.assertEqual(
                excluded, 3, "all three compile-only lines must be counted"
            )
            for stem, name in [
                ("compile_only_e2e", "compile_only_e2e.rs"),
                ("list_only_e2e", "list_only_e2e.rs"),
                ("nextest_list_only_e2e", "nextest_list_only_e2e.rs"),
            ]:
                self.assertFalse(
                    guard.is_covered(stem, name, text, []), name
                )
            # A real invocation on a nearby line still certifies its target.
            self.assertTrue(guard.is_covered("real_run_e2e", "real_run_e2e.rs", text, []))

    def test_script_bodies_are_scan_for_compile_only_too(self):
        # Coverage routed through a repo script counts, but a compile-only
        # invocation inside that script must equally not certify.
        workflow = """
jobs:
  lane:
    runs-on: ubuntu-latest
    steps:
      - name: drive the script
        run: |
          bash scripts/run_emulator.sh
"""
        tree = GuardTree(["scripted_compile_only_e2e.rs"], [workflow])
        with tree.patch():
            (tree.root / "scripts" / "run_emulator.sh").write_text(
                "cargo test --test scripted_compile_only_e2e --no-run\n",
                encoding="utf-8",
            )
            text, excluded = guard.run_block_text()
            self.assertEqual(excluded, 1)
            self.assertFalse(
                guard.is_covered("scripted_compile_only_e2e", "scripted_compile_only_e2e.rs", text, [])
            )


if __name__ == "__main__":
    unittest.main()
