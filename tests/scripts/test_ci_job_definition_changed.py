"""Tests for scripts/ci_job_definition_changed.py (TD-CITIER-1).

The helper decides whether a path-scoped heavy CI suite must run because the PR
edited that suite's own definition. Its two failure modes are asymmetric:

* a false NEGATIVE lets a PR that weakens a gate skip that gate and report
  green -- the defect class the helper exists to prevent;
* a false POSITIVE only costs one extra run.

So the cases below pin every way the job's definition can change to `True`,
pin the unrelated edits (the whole point: ci.yml is touched by a third of PRs)
to `False`, and check the real workflow still parses, because a block the
parser cannot find degrades to "always run" -- safe, but it would silently
undo the speed-up.
"""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ci_job_definition_changed",
    REPO_ROOT / "scripts" / "ci_job_definition_changed.py",
)
helper = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(helper)

BASE = """\
name: CI
on:
  pull_request:
env:
  CARGO_INCREMENTAL: 0

jobs:
  changes:
    name: Detect Changes
    outputs:
      test_tier: ${{ github.base_ref == 'develop' }}
      medium_tier: ${{ github.base_ref == 'qa' }}
      heavy_changes: ${{ steps.filter.outputs.heavy_changes == 'true' || steps.selfdef.outputs.heavy_changes == 'true' }}
      other_changes: ${{ steps.filter.outputs.other_changes }}
    steps:
      - uses: dorny/paths-filter@v4
        id: filter
        with:
          filters: |
            heavy_changes:
              - 'src/heavy/**'
              # a comment inside the block
              - 'Cargo.lock'
            other_changes:
              - 'src/other/**'
      - name: Detect edits to own definitions
        id: selfdef
        run: |
          python3 scripts/ci_job_definition_changed.py --base "$base" \\
            --job heavy=heavy_changes \\
            --job other=other_changes

  heavy:
    name: Heavy suite
    needs: changes
    if: |
      needs.changes.outputs.test_tier == 'true' &&
      (needs.changes.outputs.medium_tier == 'true' ||
       needs.changes.outputs.heavy_changes == 'true')
    steps:
      - run: cargo test --heavy

  other:
    name: Other
    needs: changes
    if: needs.changes.outputs.other_changes == 'true'
    steps:
      - run: echo other
"""


def changed(head: str, base: str | None = BASE) -> bool:
    return helper.changed(base, head, "heavy", "heavy_changes")


class DefinitionChangedTest(unittest.TestCase):
    def test_identical_is_unchanged(self) -> None:
        self.assertFalse(changed(BASE))

    def test_unrelated_job_edit_is_unchanged(self) -> None:
        self.assertFalse(changed(BASE.replace("echo other", "echo other2")))

    def test_unrelated_filter_edit_is_unchanged(self) -> None:
        # The filter right AFTER ours: proves the block boundary is correct.
        self.assertFalse(changed(BASE.replace("'src/other/**'", "'src/else/**'")))

    def test_any_changes_output_edit_is_changed(self) -> None:
        # The whole `changes` job outside the filter BODIES is part of every
        # suite's definition: deliberately conservative (an output line is
        # cheap to over-trigger on, expensive to miss).
        self.assertTrue(
            changed(
                BASE.replace(
                    "other_changes: ${{ steps.filter.outputs.other_changes }}",
                    "other_changes: ${{ steps.filter.outputs.other_changes == 'true' }}",
                )
            )
        )

    def test_own_step_edit_is_changed(self) -> None:
        self.assertTrue(changed(BASE.replace("cargo test --heavy", "cargo test")))

    def test_own_if_edit_is_changed(self) -> None:
        self.assertTrue(
            changed(BASE.replace("needs.changes.outputs.medium_tier == 'true' ||\n", ""))
        )

    def test_own_filter_glob_narrowed_is_changed(self) -> None:
        self.assertTrue(changed(BASE.replace("              - 'Cargo.lock'\n", "")))

    def test_own_filter_comment_edit_is_changed(self) -> None:
        # Textual on purpose: errs towards running.
        self.assertTrue(changed(BASE.replace("a comment inside", "a comment in")))

    def test_tier_output_redefined_is_changed(self) -> None:
        self.assertTrue(
            changed(BASE.replace("github.base_ref == 'qa' }}", "github.base_ref == 'main' }}"))
        )

    def test_own_output_redefined_is_changed(self) -> None:
        self.assertTrue(
            changed(BASE.replace(" || steps.selfdef.outputs.heavy_changes == 'true'", ""))
        )

    def test_job_now_reads_another_output_is_changed(self) -> None:
        # Switching the job onto a different filter is a definition change even
        # though neither filter's text moved.
        self.assertTrue(
            changed(
                BASE.replace(
                    "needs.changes.outputs.heavy_changes == 'true')",
                    "needs.changes.outputs.other_changes == 'true')",
                )
            )
        )

    def test_selfdef_job_pair_removed_is_changed(self) -> None:
        # Review finding (#2043): deleting the suite's own `--job` pair from the
        # selfdef step disarms the self-check, and was invisible to it.
        self.assertTrue(changed(BASE.replace("            --job heavy=heavy_changes \\\n", "")))

    def test_paths_filter_option_added_is_changed(self) -> None:
        # Review finding (#2043): a non-`filters` key on the paths-filter step
        # (predicate-quantifier: every) changes every filter's semantics.
        self.assertTrue(
            changed(
                BASE.replace(
                    "        with:\n          filters: |",
                    "        with:\n          predicate-quantifier: every\n          filters: |",
                )
            )
        )

    def test_other_filter_body_edit_still_unchanged(self) -> None:
        # The precision that matters: the 15/44 PRs that touch ci.yml mostly
        # edit OTHER filters/jobs; those must not fire the heavy suite.
        self.assertFalse(changed(BASE.replace("              - 'src/other/**'\n", "              - 'src/other/**'\n              - 'src/more/**'\n")))

    def test_header_env_edit_is_changed(self) -> None:
        self.assertTrue(changed(BASE.replace("CARGO_INCREMENTAL: 0", "CARGO_INCREMENTAL: 1")))

    def test_renamed_job_is_changed(self) -> None:
        self.assertTrue(changed(BASE.replace("\n  heavy:\n", "\n  heavy-renamed:\n")))

    def test_removed_filter_is_changed(self) -> None:
        head = BASE.replace(
            "            heavy_changes:\n              - 'src/heavy/**'\n", ""
        )
        self.assertTrue(changed(head))

    def test_unreadable_base_fails_open(self) -> None:
        self.assertTrue(changed(BASE, base=None))

    def test_main_fails_open_on_unknown_rev(self) -> None:
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = helper.main(
                [
                    "--base",
                    "refs/does/not/exist",
                    "--job",
                    "mlflow-conformance=mlflow_changes",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), "mlflow_changes=true")


class RealWorkflowContractTest(unittest.TestCase):
    """The helper is wired to the real ci.yml; keep the two consistent."""

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    def pairs(self) -> list[tuple[str, str]]:
        step = self.workflow.split("id: selfdef", 1)[1].split("\n  # ", 1)[0]
        return re.findall(r"--job ([A-Za-z0-9_-]+)=([A-Za-z0-9_]+)", step)

    def test_every_selfdef_output_is_computed_by_the_step(self) -> None:
        consumed = set(re.findall(r"steps\.selfdef\.outputs\.([A-Za-z0-9_]+)", self.workflow))
        produced = {name for _, name in self.pairs()}
        self.assertTrue(consumed, "no output reads steps.selfdef")
        self.assertEqual(consumed, produced)

    def test_every_pair_resolves_on_the_real_workflow(self) -> None:
        # A block the parser cannot find makes the verdict permanently `true`:
        # safe, but it would silently put the heavy suite back on every PR.
        for job, name in self.pairs():
            names = helper.referenced_outputs(self.workflow, job, name)
            self.assertIn(name, names)
            parts = helper.definition(self.workflow, job, names)
            self.assertNotIn(None, parts, f"{job}/{name} has an unresolvable block")
            self.assertFalse(helper.changed(self.workflow, self.workflow, job, name))

    def test_every_paired_job_reads_its_output(self) -> None:
        for job, name in self.pairs():
            block = helper.job_block(self.workflow, job)
            self.assertIsNotNone(block, job)
            self.assertIn(f"needs.changes.outputs.{name}", block or "", job)


if __name__ == "__main__":
    unittest.main()
