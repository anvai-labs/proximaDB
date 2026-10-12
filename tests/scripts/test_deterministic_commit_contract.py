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


class NodeVersionPinContractTest(unittest.TestCase):
    """Every spelling GitHub accepts must reach the check.

    The first revision of `SETUP_NODE_RE` anchored on `- uses:` and silently
    skipped two real steps in this repo; an adversarial review then found two
    more spellings that passed. Each case below corresponds to one of those,
    so a future regex edit that loses a spelling fails here instead of in six
    months when a Node 20 pin reappears unnoticed.
    """

    STEP = "      - uses: actions/setup-node@v7\n        with:\n"

    def findings_for(self, workflow: str, name: str = "ci.yml") -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / ".github/workflows" / name
            path.parent.mkdir(parents=True)
            path.write_text(workflow, encoding="utf-8")
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_node_version_pins(findings)
                return findings

    def test_the_checked_in_workflows_pass(self) -> None:
        for name in ("ci.yml", "release.yml"):
            with self.subTest(workflow=name):
                text = (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertEqual(self.findings_for(text, name), [])

    def test_the_agreed_major_is_accepted_in_every_benign_spelling(self) -> None:
        for pin in ("'24'", '"24"', "24", "24.9.0", "'24.x'", "'24'  # pinned"):
            with self.subTest(pin=pin):
                self.assertEqual(
                    self.findings_for(f"{self.STEP}          node-version: {pin}\n"), []
                )

    def test_another_major_is_rejected(self) -> None:
        for pin in ("'20'", '"20"', "20", "20.11.0", "'18'", "'26'"):
            with self.subTest(pin=pin):
                self.assertTrue(
                    self.findings_for(f"{self.STEP}          node-version: {pin}\n"), pin
                )

    def test_an_unpinned_step_is_rejected(self) -> None:
        self.assertTrue(self.findings_for("      - uses: actions/setup-node@v7\n"))

    def test_uses_under_a_name_or_if_is_checked(self) -> None:
        # The two shapes the `- uses:`-anchored revision skipped -- release.yml
        # writes `- name:` above its `uses:`, the contract-gate matrix `- if:`.
        for bullet in ("- name: Setup Node", "- if: matrix.setup == 'node'"):
            with self.subTest(bullet=bullet):
                workflow = (
                    f"      {bullet}\n"
                    "        uses: actions/setup-node@v7\n"
                    "        with:\n"
                    "          node-version: '20'\n"
                )
                self.assertTrue(self.findings_for(workflow))

    def test_a_quoted_action_ref_is_checked(self) -> None:
        for ref in ('"actions/setup-node@v7"', "'actions/setup-node@v7'"):
            with self.subTest(ref=ref):
                workflow = (
                    f"      - uses: {ref}\n"
                    "        with:\n"
                    "          node-version: '20'\n"
                )
                self.assertTrue(self.findings_for(workflow))

    def test_a_flow_style_step_cannot_bypass_the_block_style_scan(self) -> None:
        for workflow in (
            "      - {uses: actions/setup-node@v7, with: {node-version: '20'}}\n",
            '      - {uses: "actions/setup-node@v7", with: {node-version: 20}}\n',
            "      - {uses: actions/setup-node@v7}\n",
        ):
            with self.subTest(workflow=workflow.strip()):
                self.assertTrue(self.findings_for(workflow))

    def test_a_flow_style_step_on_the_agreed_major_is_accepted(self) -> None:
        self.assertEqual(
            self.findings_for(
                "      - {uses: actions/setup-node@v7, with: {node-version: '24'}}\n"
            ),
            [],
        )

    def test_a_sha_pinned_action_is_checked(self) -> None:
        workflow = (
            "      - uses: actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020\n"
            "        with:\n"
            "          node-version: '20'\n"
        )
        self.assertTrue(self.findings_for(workflow))

    def test_node_version_file_counts_as_unpinned(self) -> None:
        workflow = f"{self.STEP}          node-version-file: '.nvmrc'\n"
        findings = self.findings_for(workflow)
        self.assertTrue(findings)
        self.assertIn("node-version-file", findings[0].message)

    def test_an_expression_is_rejected_because_the_major_is_not_visible(self) -> None:
        for pin in ("${{ matrix.node }}", "${{env.NODE}}", "'lts/*'"):
            with self.subTest(pin=pin):
                self.assertTrue(
                    self.findings_for(f"{self.STEP}          node-version: {pin}\n"), pin
                )

    def test_every_step_is_checked_not_just_the_first(self) -> None:
        workflow = (
            f"{self.STEP}          node-version: '24'\n"
            "      - uses: actions/setup-node@v7\n"
            "        with:\n"
            "          node-version: '20'\n"
        )
        findings = self.findings_for(workflow)
        self.assertEqual(len(findings), 1, findings)
        # Anchored on the offending step's `uses:` line (4), not on its pin
        # line (6) -- the step is what a reader has to go and edit.
        self.assertIn("ci.yml:4", findings[0].message)

    def test_a_pin_in_an_earlier_step_does_not_satisfy_a_later_one(self) -> None:
        # The forward scan must stop at the step boundary, or an unrelated
        # `node-version:` above would launder an unpinned setup-node below it.
        workflow = (
            "      - uses: ./.github/actions/other\n"
            "        with:\n"
            "          node-version: '24'\n"
            "      - uses: actions/setup-node@v7\n"
        )
        self.assertTrue(self.findings_for(workflow))

    def test_a_yaml_extension_workflow_is_not_skipped(self) -> None:
        workflow = f"{self.STEP}          node-version: '20'\n"
        self.assertTrue(self.findings_for(workflow, "zz-drift.yaml"))

    def test_a_composite_action_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / ".github/actions/setup-stuff/action.yml"
            path.parent.mkdir(parents=True)
            path.write_text(
                "runs:\n  using: composite\n  steps:\n"
                "    - uses: actions/setup-node@v7\n"
                "      with:\n        node-version: '20'\n",
                encoding="utf-8",
            )
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_node_version_pins(findings)
        self.assertTrue(findings)

    def test_the_major_the_repo_agrees_on_is_the_one_asserted(self) -> None:
        # Guards against the check passing vacuously if NODE_MAJOR is ever
        # blanked, and ties the constant to what the workflows actually say.
        self.assertTrue(contract.NODE_MAJOR.isdigit())
        text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn(f"node-version: '{contract.NODE_MAJOR}'", text)


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
