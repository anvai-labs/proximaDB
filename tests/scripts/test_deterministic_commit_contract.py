#!/usr/bin/env python3
"""Negative controls for nextest and CI runner selection in the commit contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_deterministic_commit_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[2]


class NextestCommitContractTest(unittest.TestCase):
    def findings_for(self, config: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / ".config"
            config_dir.mkdir()
            (config_dir / "nextest.toml").write_text(config, encoding="utf-8")
            original_root = contract.ROOT
            try:
                contract.ROOT = root
                findings: list[contract.Finding] = []
                contract.check_nextest_contract(findings)
                return findings
            finally:
                contract.ROOT = original_root

    def test_checked_in_retry_budget_is_accepted(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        self.assertEqual(self.findings_for(config), [])

    def test_unit_retry_budget_drift_is_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        unit_marker = "[profile.unit]"
        before, unit = config.split(unit_marker, maxsplit=1)
        mutated = before + unit_marker + unit.replace("retries = 2", "retries = 3", 1)
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("profile.unit.retries must stay at 2" in finding.message for finding in findings),
            findings,
        )

    def test_default_retry_budget_drift_is_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config.replace("retries = 2", "retries = 3", 1)
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("profile.default.retries must stay at 2" in finding.message for finding in findings),
            findings,
        )

    def test_integration_retry_budget_drift_is_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        marker = "[profile.integration]"
        before, integration = config.split(marker, maxsplit=1)
        mutated = before + marker + integration.replace("retries = 1", "retries = 0", 1)
        findings = self.findings_for(mutated)
        self.assertTrue(
            any(
                "profile.integration.retries must stay at 1" in finding.message
                for finding in findings
            ),
            findings,
        )

    def test_unit_cannot_hide_flaky_survivors(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config.replace(
            "[profile.unit]\n",
            '[profile.unit]\nfinal-status-level = "pass"\n',
            1,
        )
        findings = self.findings_for(mutated)
        self.assertTrue(
            any(
                'profile.unit.final-status-level override must be "flaky"'
                in finding.message
                for finding in findings
            ),
            findings,
        )

    def test_integration_cannot_hide_flaky_survivors(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config.replace(
            "[profile.integration]\n",
            '[profile.integration]\nfinal-status-level = "none"\n',
            1,
        )
        findings = self.findings_for(mutated)
        self.assertTrue(
            any(
                'profile.integration.final-status-level override must be "flaky"'
                in finding.message
                for finding in findings
            ),
            findings,
        )

    def test_unit_retry_overrides_are_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config + '\n[[profile.unit.overrides]]\nfilter = "all()"\nretries = 9\n'
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("profile.unit.overrides must not set retries" in finding.message for finding in findings),
            findings,
        )

    def test_default_retry_overrides_are_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config + '\n[[profile.default.overrides]]\nfilter = "all()"\nretries = 9\n'
        findings = self.findings_for(mutated)
        self.assertTrue(
            any(
                "profile.default.overrides must not set retries" in finding.message
                for finding in findings
            ),
            findings,
        )

    def test_flaky_survivor_reporting_drift_is_rejected(self) -> None:
        config = (REPO_ROOT / ".config/nextest.toml").read_text(encoding="utf-8")
        mutated = config.replace('final-status-level = "flaky"', 'final-status-level = "pass"', 1)
        findings = self.findings_for(mutated)
        self.assertTrue(
            any(
                'profile.default.final-status-level must be "flaky"' in finding.message
                for finding in findings
            ),
            findings,
        )


class CiRunnerCommitContractTest(unittest.TestCase):
    def findings_for(self, workflow: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / ".github/workflows/ci.yml"
            path.parent.mkdir(parents=True)
            path.write_text(workflow, encoding="utf-8")
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_ci_runner_contract(findings)
                return findings

    def test_checked_in_ci_runner_selection_is_accepted(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertEqual(self.findings_for(workflow), [])

    def test_explicit_image_is_accepted(self) -> None:
        for label in ("ubuntu-24.04", "'ubuntu-24.04'", '"ubuntu-24.04" # pinned'):
            with self.subTest(label=label):
                self.assertEqual(self.findings_for(f"    runs-on: {label}\n"), [])

    def test_generic_overflow_alias_is_rejected(self) -> None:
        for label in ("ubuntu-latest", "'ubuntu-latest'", '"ubuntu-latest"'):
            with self.subTest(label=label):
                findings = self.findings_for(f"    runs-on: {label}\n")
                self.assertTrue(any(f.check == "ci-runner" for f in findings), findings)

    def test_unqualified_runner_selection_is_rejected(self) -> None:
        for label in (
            "[self-hosted, ubuntu-latest]",
            "${{ matrix.runner }}",
            "\n      group: public-overflow",
        ):
            with self.subTest(label=label):
                self.assertTrue(self.findings_for(f"    runs-on: {label}\n"))

    def test_every_runner_selection_is_checked(self) -> None:
        workflow = "    runs-on: ubuntu-24.04\n    runs-on: ubuntu-latest\n"
        findings = self.findings_for(workflow)
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("ci.yml:2", findings[0].message)

    def test_missing_runner_selection_is_rejected(self) -> None:
        self.assertTrue(self.findings_for("# runs-on: ubuntu-24.04\n"))

    def test_quoted_runner_key_is_checked(self) -> None:
        for key in ('"runs-on"', "'runs-on'"):
            with self.subTest(key=key):
                workflow = f"    runs-on: ubuntu-24.04\n    {key}: ubuntu-latest\n"
                self.assertTrue(self.findings_for(workflow))

    def test_inline_job_cannot_bypass_the_block_style_guard(self) -> None:
        for key in ("runs-on", '"runs-on"', "'runs-on'"):
            with self.subTest(key=key):
                workflow = (
                    "    runs-on: ubuntu-24.04\n"
                    f"  other: {{{key}: ubuntu-latest}}\n"
                )
                self.assertTrue(self.findings_for(workflow))

    def test_hash_without_whitespace_is_part_of_the_label(self) -> None:
        self.assertTrue(self.findings_for("    runs-on: ubuntu-24.04#different-label\n"))


if __name__ == "__main__":
    unittest.main()


class PromotionConcurrencyContractTest(unittest.TestCase):
    """A workflow on a promotion branch must not cancel its own promotion run.

    The group key across this repo is `head_ref || ref_name`, and a `qa -> main`
    PR has `head_ref == 'qa'`, so the PR run and the push-to-qa run of the same
    workflow share one group. With cancellation on, one killed the other and
    `gh pr checks` rendered the cancellation as "fail" -- a phantom failure for
    a check that had passed. Misdiagnosed twice during the v0.4.0 cut.
    """

    COLLAPSING_GROUP = "${{ github.workflow }}-${{ github.head_ref || github.ref_name }}"

    def findings_for(self, workflow: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wf_dir = root / ".github" / "workflows"
            wf_dir.mkdir(parents=True)
            (wf_dir / "probe.yml").write_text(workflow, encoding="utf-8")
            original_root = contract.ROOT
            try:
                contract.ROOT = root
                findings: list[contract.Finding] = []
                contract.check_promotion_concurrency(findings)
                return findings
            finally:
                contract.ROOT = original_root

    def _workflow(self, cancel: str, group: str | None = None) -> str:
        return (
            "name: Probe\n"
            "on:\n"
            "  push:\n"
            "    branches: [main, develop, qa]\n"
            "  pull_request:\n"
            "concurrency:\n"
            f"  group: {group if group is not None else self.COLLAPSING_GROUP}\n"
            f"  cancel-in-progress: {cancel}\n"
            "jobs:\n"
            "  probe:\n"
            "    runs-on: ubuntu-24.04\n"
            "    timeout-minutes: 15\n"
            "    steps:\n"
            "      - run: 'true'\n"
        )

    def test_cancelling_on_qa_is_rejected(self) -> None:
        findings = self.findings_for(
            self._workflow("${{ github.ref_name != 'main' && github.ref_name != 'develop' }}")
        )
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("does not exempt qa", findings[0].message)

    def test_literal_false_is_accepted(self) -> None:
        self.assertEqual(self.findings_for(self._workflow("false")), [])

    def test_exempting_every_promotion_ref_is_accepted(self) -> None:
        canonical = (
            "${{ !(github.event_name == 'pull_request' && "
            "contains(fromJson('[\"qa\", \"main\"]'), github.base_ref)) && "
            "!contains(fromJson('[\"main\", \"develop\", \"qa\"]'), github.ref_name) }}"
        )
        self.assertEqual(self.findings_for(self._workflow(canonical)), [])

    def test_a_non_collapsing_group_key_is_not_a_finding(self) -> None:
        """Keyed on `github.ref`, a push and a PR are in DIFFERENT groups.

        docs-site.yml is keyed that way, so it cannot collide and must not be
        flagged. An earlier revision of this check did flag it.
        """
        self.assertEqual(
            self.findings_for(
                self._workflow(
                    "${{ github.ref != 'refs/heads/main' }}",
                    group="docs-site-${{ github.ref }}",
                )
            ),
            [],
        )

    def test_full_ref_spelling_counts_as_an_exemption(self) -> None:
        canonical = (
            "${{ github.ref != 'refs/heads/main' && github.ref != 'refs/heads/develop' "
            "&& github.ref != 'refs/heads/qa' }}"
        )
        self.assertEqual(self.findings_for(self._workflow(canonical)), [])

    def test_the_real_tree_passes(self) -> None:
        findings: list[contract.Finding] = []
        contract.check_promotion_concurrency(findings)
        self.assertEqual(findings, [], [f.message for f in findings])
