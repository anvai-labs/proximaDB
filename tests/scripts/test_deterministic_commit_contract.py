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


class DockerFilterContractTest(unittest.TestCase):
    """Every Dockerfile a workflow `file:`-builds must trip the `docker` filter.

    TD-DOCKCI-1: the old filter (`Dockerfile*`, `docker-compose*.yml`,
    `.dockerignore`) matched top-level paths only, so `deploy/docker/**` --
    the Dockerfile CI actually builds -- SKIPPED docker-build, while root
    `Dockerfile` edits (nothing builds it) burned a ~38-minute job. Each test
    pins one direction of that inversion.
    """

    def findings_for(
        self, ci_text: str, extra_workflows: dict[str, str] | None = None
    ) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github/workflows"
            workflows.mkdir(parents=True)
            (workflows / "ci.yml").write_text(ci_text, encoding="utf-8")
            for name, text in (extra_workflows or {}).items():
                (workflows / name).write_text(text, encoding="utf-8")
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_docker_filter_covers_build_targets(findings)
                return findings

    def test_the_checked_in_workflows_pass(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        publish = (REPO_ROOT / ".github/workflows/publish-image.yml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            self.findings_for(ci, {"publish-image.yml": publish}),
            [],
        )

    def test_the_old_top_level_filter_is_rejected(self) -> None:
        """The exact filter TD-DOCKCI-1 was filed against must stay rejected."""
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        mutated = ci.replace(
            "- 'deploy/docker/**'", "- 'Dockerfile*'", 1
        )
        self.assertNotEqual(mutated, ci, "sanity: the mutation must apply")
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("deploy/docker/Dockerfile" in f.message for f in findings),
            findings,
        )

    def test_a_dockerfile_outside_every_pattern_is_rejected(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        other = (
            "jobs:\n"
            "  demo:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: demo/config/Dockerfile\n"
        )
        findings = self.findings_for(ci, {"demo.yml": other})
        self.assertTrue(
            any("demo/config/Dockerfile" in f.message for f in findings),
            findings,
        )

    def test_a_dockerfile_under_the_glob_directory_is_matched(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        other = (
            "jobs:\n"
            "  nested:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: deploy/docker/sub/Dockerfile\n"
        )
        self.assertEqual(self.findings_for(ci, {"nested.yml": other}), [])

    def test_a_non_dockerfile_file_target_is_not_flagged(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        other = (
            "jobs:\n"
            "  pack:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: demo/config/bundle.tar\n"
        )
        self.assertEqual(self.findings_for(ci, {"pack.yml": other}), [])

    def test_a_missing_docker_filter_is_rejected(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        # Anchor on the filter KEY LINE: a bare `ci.replace("docker:", ...)`
        # would hit the outputs mapping at line 105 instead.
        anchor = "docker:\n              - 'deploy/docker/**'"
        self.assertIn(anchor, ci, "sanity: the filter key line is where expected")
        mutated = ci.replace(anchor, "docke_removed:\n              - 'deploy/docker/**'", 1)
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("no `docker:` path filter" in f.message for f in findings),
            findings,
        )

    def test_the_matcher_keeps_the_top_level_trap_a_trap(self) -> None:
        """`Dockerfile*` must NOT match deploy/docker/Dockerfile.

        picomatch matches slash-less patterns against top-level paths only --
        that IS the bug. If the matcher here starts matching it, the guard has
        stopped modelling the tool and the old filter would pass silently.
        """
        self.assertFalse(
            contract._picomatch_subset_matches("Dockerfile*", "deploy/docker/Dockerfile")
        )
        self.assertTrue(
            contract._picomatch_subset_matches("Dockerfile*", "Dockerfile.prebuilt")
        )
        self.assertTrue(
            contract._picomatch_subset_matches("deploy/docker/**", "deploy/docker/Dockerfile")
        )
        self.assertFalse(
            contract._picomatch_subset_matches("deploy/docker/**", "deploy/config.toml")
        )

    def test_mid_segment_globstar_is_not_a_globstar(self) -> None:
        """A mid-segment `**` must NOT cross `/` (picomatch: only a whole
        segment is a globstar; `a**b` behaves like `a*b`).

        Over-accepting here is the silent-skip class: a filter pattern
        picomatch never matches would pass the guard, resurrecting the exact
        TD-DOCKCI-1 hole this check exists to close.
        """
        self.assertFalse(
            contract._picomatch_subset_matches("a**b", "a/x/b")
        )
        self.assertFalse(
            contract._picomatch_subset_matches("deploy**/Dockerfile", "deploy/docker/Dockerfile")
        )
        self.assertTrue(contract._picomatch_subset_matches("a**b", "ab"))
        self.assertTrue(contract._picomatch_subset_matches("a**b", "axb"))

    def test_leading_globstar_spans_zero_segments(self) -> None:
        """`**/Dockerfile` matches BOTH root Dockerfile and nested ones."""
        self.assertTrue(contract._picomatch_subset_matches("**/Dockerfile", "Dockerfile"))
        self.assertTrue(
            contract._picomatch_subset_matches("**/Dockerfile", "deploy/docker/Dockerfile")
        )
        self.assertTrue(contract._picomatch_subset_matches("dir/**/file", "dir/file"))
        self.assertTrue(contract._picomatch_subset_matches("dir/**/file", "dir/a/b/file"))

    def test_trailing_globstar_covers_the_bare_directory(self) -> None:
        """picomatch matches `x/**` against `x` itself (verified 2.3.1/4.0.5)."""
        self.assertTrue(contract._picomatch_subset_matches("deploy/docker/**", "deploy/docker"))

    def test_wildcard_head_before_trailing_globstar_declines_bare_dir(self) -> None:
        """picomatch grants the bare-directory allowance of a trailing `/**`
        only to heads with no `*`: `a/**` matches `a`, but `*/**` does NOT
        match `Dockerfile` and `**/*/**` does NOT match `a` (verified against
        picomatch 2.3.1 {dot:true}).

        Granting it anyway is the silent-skip class — round 2 of the
        adversarial review demonstrated exactly that hole for `*/**`.
        """
        self.assertFalse(contract._picomatch_subset_matches("*/**", "Dockerfile"))
        self.assertFalse(contract._picomatch_subset_matches("a*/**", "ab"))
        self.assertFalse(contract._picomatch_subset_matches("**/*/**", "a"))
        # ...while the wildcard-headed head still spans everything BELOW it,
        # and `?`-heads and literal heads keep the bare-directory match.
        self.assertTrue(contract._picomatch_subset_matches("*/**", "a/b"))
        self.assertTrue(contract._picomatch_subset_matches("a/**", "a"))
        self.assertTrue(contract._picomatch_subset_matches("a?/**", "ab"))

    def test_filter_block_survives_comments_and_skips_structural_keys(self) -> None:
        """Comments between items must not truncate the pattern list, and a
        structural `docker:` key (mapping child) must not hijack extraction."""
        ci = (
            "jobs:\n"
            "  demo:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: deploy/docker/sub/Dockerfile\n"
        )
        ci_with_filter = (
            "paths-filter:\n"
            "  docker:\n"
            "    # the image inputs\n"
            "    - 'deploy/docker/**'\n"
            "    # and the workflow that builds it\n"
            "    - '.github/workflows/ci.yml'\n"
        ) + ci
        self.assertEqual(self.findings_for(ci_with_filter), [])
        structural = (
            "jobs:\n"
            "  build:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        docker:\n"
            "          - linux\n"
            "  paths-filter:\n"
            "    docker:\n"
            "      - 'deploy/docker/**'\n"
        ) + ci
        self.assertEqual(self.findings_for(structural), [])

    def test_quoted_and_commented_file_targets_are_parsed(self) -> None:
        ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        quoted = (
            "jobs:\n"
            "  quoted:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: \"deploy/docker/Dockerfile\"  # the image\n"
        )
        self.assertEqual(self.findings_for(ci, {"quoted.yml": quoted}), [])
        uncovered = (
            "jobs:\n"
            "  quoted:\n"
            "    steps:\n"
            "      - uses: docker/build-push-action@v7\n"
            "        with:\n"
            "          file: \"demo/Dockerfile\"  # not watched\n"
        )
        findings = self.findings_for(ci, {"quoted.yml": uncovered})
        self.assertTrue(
            any("demo/Dockerfile" in f.message for f in findings),
            findings,
        )
        # The suggestion must be a pattern picomatch actually matches.
        suggestion = next(f for f in findings if "demo/Dockerfile" in f.message)
        self.assertIn("`demo/**`", suggestion.message)

    def test_mid_scalar_hash_stays_in_the_file_value(self) -> None:
        """A `#` starts a YAML comment only after whitespace: in
        `file: deploy/Dockerfile#x` the `#x` is part of the scalar, and the
        guard must test the same path the workflow builds.
        """
        m = contract.DOCKER_FILE_TARGET_RE.match("          file: deploy/docker/Dockerfile#x\n")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("value"), "deploy/docker/Dockerfile#x")
        m = contract.DOCKER_FILE_TARGET_RE.match(
            "          file: deploy/docker/Dockerfile  # real comment\n"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group("value"), "deploy/docker/Dockerfile")
        # Trailing whitespace with NO comment is still a target (round-3
        # review: the comment-tail space requirement must not swallow it).
        m = contract.DOCKER_FILE_TARGET_RE.match(
            "          file: deploy/docker/Dockerfile  \n"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group("value"), "deploy/docker/Dockerfile")


class SerdeJsonPreserveOrderContractTest(unittest.TestCase):
    """`serde_json` must carry `preserve_order` in both root tables.

    TD-SPECORD-1: the generated OpenAPI document's key order rode on the
    feature being enabled transitively; DataFusion 55.2 deliberately stopped
    supplying it, and a drop flips every map to alphabetical -- a 9k-line,
    semantically empty spec + SDK diff. The fix (#2006 + the workspace-level
    entry) is silent, so this check makes removing it loud.
    """

    def findings_for(self, cargo_text: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Cargo.toml").write_text(cargo_text, encoding="utf-8")
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_serde_json_preserve_order(findings)
                return findings

    def test_the_checked_in_manifest_passes(self) -> None:
        cargo = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
        self.assertEqual(self.findings_for(cargo), [])

    def test_dropping_the_feature_from_root_dependencies_is_rejected(self) -> None:
        cargo = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
        needle = 'serde_json = { version = "1.0", features = ["preserve_order"] }'
        self.assertEqual(cargo.count(needle), 2, "sanity: both tables carry the entry")
        # [workspace.dependencies] (line ~201) precedes [dependencies] (line
        # ~402) in file order, so the ROOT-deps entry is the SECOND occurrence.
        first = cargo.find(needle)
        second = cargo.find(needle, first + 1)
        mutated = (
            cargo[:second] + 'serde_json = { version = "1.0" }' + cargo[second + len(needle):]
        )
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("[dependencies].serde_json" in f.message for f in findings),
            findings,
        )
        self.assertFalse(
            any("[workspace.dependencies].serde_json" in f.message for f in findings),
            findings,
        )

    def test_dropping_the_feature_from_the_workspace_table_is_rejected(self) -> None:
        cargo = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
        needle = 'serde_json = { version = "1.0", features = ["preserve_order"] }'
        first = cargo.find(needle)
        mutated = cargo[:first] + 'serde_json = { version = "1.0" }' + cargo[first + len(needle):]
        # The FIRST occurrence is the workspace table (alphabetical file order:
        # [workspace.dependencies] precedes [dependencies]). Assert the split.
        findings = self.findings_for(mutated)
        self.assertTrue(
            any("[workspace.dependencies].serde_json" in f.message for f in findings),
            findings,
        )
        self.assertFalse(
            any("[dependencies].serde_json" in f.message for f in findings),
            findings,
        )

    def test_removing_the_entry_entirely_is_rejected(self) -> None:
        cargo = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
        needle = 'serde_json = { version = "1.0", features = ["preserve_order"] }\n'
        mutated = cargo.replace(needle, "")
        self.assertNotEqual(mutated, cargo, "sanity: the mutation must apply")
        findings = self.findings_for(mutated)
        self.assertEqual(len(findings), 2, findings)
        self.assertTrue(
            any("[dependencies].serde_json" in f.message for f in findings)
            and any("[workspace.dependencies].serde_json" in f.message for f in findings),
            findings,
        )

    def test_a_feature_list_without_preserve_order_is_rejected(self) -> None:
        cargo = "\n".join(
            [
                "[dependencies]",
                'serde_json = { version = "1.0", features = ["std"] }',
                "[workspace.dependencies]",
                'serde_json = { version = "1.0", features = ["std"] }',
            ]
        )
        findings = self.findings_for(cargo)
        self.assertEqual(len(findings), 2, findings)


class RustCachePolicyContractTest(unittest.TestCase):
    """CI redesign PR1: ci.yml rust-cache steps share one key per build shape,
    never key on the job id, and only a develop push (one writer per key) saves.

    Before: per-job keys, saved on every PR ref (readable by nothing else), and
    develop -- the one scope every run can read -- saved almost nothing. 0 of 15
    heavy jobs hit cache on the v0.4.1 release.
    """

    DEVELOP = "${{ github.event_name == 'push' && github.ref == 'refs/heads/develop' }}"

    def findings_for(self, workflow: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wf_dir = root / ".github" / "workflows"
            wf_dir.mkdir(parents=True)
            (wf_dir / "ci.yml").write_text(workflow, encoding="utf-8")
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_rust_cache_policy(findings)
                return findings

    @staticmethod
    def _job(name: str, cache: str, env: str = "", matrix: str = "") -> str:
        return (
            f"  {name}:\n"
            "    runs-on: ubuntu-24.04\n"
            "    timeout-minutes: 15\n"
            f"{env}{matrix}"
            "    steps:\n"
            "      - uses: actions/checkout@v7\n"
            "      - uses: Swatinem/rust-cache@v2\n"
            "        with:\n"
            f"{cache}"
            "      - run: cargo check\n"
        )

    def _cache(self, key: str = '"dev-test"', save: str | None = None, job_id: bool = True) -> str:
        text = '          prefix-key: "v2-rust"\n'
        if key:
            text += f"          shared-key: {key}\n"
        if job_id:
            text += "          add-job-id-key: false\n"
        if save is not None:
            text += f"          save-if: {save}\n"
        return text

    def wf(self, *jobs: str) -> str:
        return "name: CI\non:\n  push:\njobs:\n" + "".join(jobs)

    def test_writer_plus_reader_is_accepted(self) -> None:
        findings = self.findings_for(
            self.wf(
                self._job("writer", self._cache(save=self.DEVELOP)),
                self._job("reader", self._cache(save="false")),
            )
        )
        self.assertEqual(findings, [], [f.message for f in findings])

    def test_missing_save_if_is_rejected(self) -> None:
        # rust-cache's default is save-if: true -- every PR saved.
        findings = self.findings_for(self.wf(self._job("j", self._cache(save=None))))
        self.assertTrue(any("save-if" in f.message for f in findings), findings)

    def test_save_on_every_push_is_rejected(self) -> None:
        findings = self.findings_for(
            self.wf(self._job("j", self._cache(save="${{ github.event_name == 'push' }}")))
        )
        self.assertTrue(any("save-if" in f.message for f in findings), findings)

    def test_widening_the_develop_condition_is_rejected(self) -> None:
        widened = (
            "${{ github.event_name == 'push' && github.ref == 'refs/heads/develop' "
            "|| github.event_name == 'pull_request' }}"
        )
        findings = self.findings_for(self.wf(self._job("j", self._cache(save=widened))))
        self.assertTrue(any("save-if" in f.message for f in findings), findings)

    def test_narrowing_the_develop_condition_is_accepted(self) -> None:
        narrowed = (
            "${{ github.event_name == 'push' && github.ref == 'refs/heads/develop' "
            "&& matrix.shape == 'test' }}"
        )
        self.assertEqual(self.findings_for(self.wf(self._job("j", self._cache(save=narrowed)))), [])

    def test_job_id_key_is_rejected(self) -> None:
        findings = self.findings_for(
            self.wf(self._job("j", self._cache(save="false", job_id=False)))
        )
        self.assertTrue(any("add-job-id-key" in f.message for f in findings), findings)

    def test_missing_shared_key_is_rejected(self) -> None:
        findings = self.findings_for(self.wf(self._job("j", self._cache(key="", save="false"))))
        self.assertTrue(any("shared-key" in f.message for f in findings), findings)

    def test_two_writers_of_one_key_are_rejected(self) -> None:
        findings = self.findings_for(
            self.wf(
                self._job("a", self._cache(save=self.DEVELOP)),
                self._job("b", self._cache(save=self.DEVELOP)),
            )
        )
        self.assertTrue(any("2 writers" in f.message for f in findings), findings)

    def test_matrix_key_writers_of_different_shapes_are_distinct(self) -> None:
        matrix = (
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - cache: dev-test\n"
            "          - cache: release\n"
        )
        findings = self.findings_for(
            self.wf(
                self._job("m", self._cache(key="${{ matrix.cache }}", save=self.DEVELOP), matrix=matrix),
                self._job("r", self._cache(save="false")),
            )
        )
        self.assertEqual(findings, [], [f.message for f in findings])

    def test_matrix_key_writer_colliding_with_another_writer_is_rejected(self) -> None:
        matrix = (
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - cache: dev-test\n"
            "          - cache: release\n"
        )
        findings = self.findings_for(
            self.wf(
                self._job("m", self._cache(key="${{ matrix.cache }}", save=self.DEVELOP), matrix=matrix),
                self._job("w", self._cache(key='"release"', save=self.DEVELOP)),
            )
        )
        self.assertTrue(any("'release' has 2 writers" in f.message for f in findings), findings)

    def test_job_level_rust_env_on_a_shared_key_is_rejected(self) -> None:
        env = '    env:\n      RUST_MIN_STACK: "8388608"\n'
        findings = self.findings_for(
            self.wf(
                self._job("writer", self._cache(save=self.DEVELOP)),
                self._job("reader", self._cache(save="false"), env=env),
            )
        )
        self.assertTrue(any("RUST_MIN_STACK" in f.message for f in findings), findings)

    def test_job_level_non_cargo_env_is_accepted(self) -> None:
        env = '    env:\n      REQUIRE_PROTOC: "1"\n'
        findings = self.findings_for(
            self.wf(
                self._job("writer", self._cache(save=self.DEVELOP), env=env),
                self._job("reader", self._cache(save="false")),
            )
        )
        self.assertEqual(findings, [], [f.message for f in findings])

    def test_step_under_an_if_bullet_is_checked(self) -> None:
        workflow = self.wf(
            "  j:\n"
            "    runs-on: ubuntu-24.04\n"
            "    steps:\n"
            "      - if: matrix.setup == 'rust'\n"
            "        uses: Swatinem/rust-cache@v2\n"
            "        with:\n"
            '          prefix-key: "v2-rust"\n'
            '          shared-key: "sdk"\n'
        )
        findings = self.findings_for(workflow)
        self.assertTrue(any("add-job-id-key" in f.message for f in findings), findings)

    def test_no_rust_cache_step_is_a_finding(self) -> None:
        findings = self.findings_for(self.wf("  j:\n    runs-on: ubuntu-24.04\n    steps:\n      - run: 'true'\n"))
        self.assertTrue(any("vacuously" in f.message for f in findings), findings)

    def test_the_pre_redesign_ci_yml_is_rejected(self) -> None:
        """The ci.yml this policy replaced: per-job keys, saved everywhere."""
        old = self.wf(
            self._job("rust-clippy", '          prefix-key: "v1-clippy-nightly"\n'),
            self._job(
                "rust-test",
                '          prefix-key: "v1-rust-nightly-dev"\n          shared-key: "build"\n',
            ),
            self._job(
                "rust-integration-test-query",
                '          prefix-key: "v1-rust-nightly-dev"\n          shared-key: "build"\n',
                env='    env:\n      RUST_MIN_STACK: "8388608"\n',
            ),
        )
        messages = " ".join(f.message for f in self.findings_for(old))
        for needle in ("add-job-id-key", "shared-key", "save-if"):
            self.assertIn(needle, messages)

    @staticmethod
    def _gated(name: str, job_if: str, cache: str) -> str:
        return (
            f"  {name}:\n"
            "    runs-on: ubuntu-24.04\n"
            f"    if: {job_if}\n"
            "    steps:\n"
            "      - uses: Swatinem/rust-cache@v2\n"
            "        with:\n"
            f"{cache}"
            "      - run: cargo check\n"
        )

    def test_writer_that_never_runs_on_a_develop_push_is_rejected(self) -> None:
        """Review M1: rust-coverage (`extensive`) and the emulator job
        (`test_tier`) each 'wrote' their own key, and neither runs on a
        develop push, so the keys were never written."""
        for gate in (
            "needs.changes.outputs.extensive == 'true'",
            "needs.changes.outputs.test_tier == 'true' && needs.changes.outputs.rust == 'true'",
            "needs.changes.outputs.medium_tier == 'true'",
        ):
            with self.subTest(gate=gate):
                findings = self.findings_for(
                    self.wf(self._gated("cov", gate, self._cache(key='"coverage"', save=self.DEVELOP)))
                )
                self.assertTrue(any("never written" in f.message for f in findings), findings)

    def test_block_scalar_if_is_read(self) -> None:
        cache = self._cache(key='"objstore-fast"', save=self.DEVELOP)
        job = (
            "  cov:\n"
            "    runs-on: ubuntu-24.04\n"
            "    if: |\n"
            "      !inputs.skip_tests &&\n"
            "      needs.changes.outputs.test_tier == 'true'\n"
            "    steps:\n"
            "      - uses: Swatinem/rust-cache@v2\n"
            "        with:\n"
            f"{cache}"
        )
        findings = self.findings_for(self.wf(job))
        self.assertTrue(any("never written" in f.message for f in findings), findings)

    def test_tier_gated_writer_admitting_warm_or_sweep_is_accepted(self) -> None:
        for gate in (
            "(needs.changes.outputs.test_tier == 'true' || needs.changes.outputs.warm == 'true')",
            "needs.changes.outputs.sweep == 'true' || needs.changes.outputs.medium_tier == 'true'",
            "needs.changes.outputs.compile == 'true'",
        ):
            with self.subTest(gate=gate):
                findings = self.findings_for(
                    self.wf(self._gated("w", gate, self._cache(save=self.DEVELOP)))
                )
                self.assertEqual(findings, [], [f.message for f in findings])

    def test_the_real_tree_passes(self) -> None:
        findings: list[contract.Finding] = []
        contract.check_rust_cache_policy(findings)
        self.assertEqual(findings, [], [f.message for f in findings])

    def test_the_tdd_warning_check_is_gone_with_tdd_yml(self) -> None:
        self.assertFalse((REPO_ROOT / ".github/workflows/tdd.yml").exists())
        self.assertFalse(hasattr(contract, "check_tdd_rust_warning_contract"))


class CiConcurrencyPerEventContractTest(unittest.TestCase):
    """CI redesign PR1: a PR must never queue behind the push of its head."""

    def findings_for(self, group: str) -> list[contract.Finding]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wf_dir = root / ".github" / "workflows"
            wf_dir.mkdir(parents=True)
            (wf_dir / "ci.yml").write_text(
                f"name: CI\non:\n  push:\nconcurrency:\n  group: {group}\n"
                "  cancel-in-progress: false\njobs:\n",
                encoding="utf-8",
            )
            with patch.object(contract, "ROOT", root):
                findings: list[contract.Finding] = []
                contract.check_ci_concurrency_per_event(findings)
                return findings

    def test_the_pre_redesign_group_is_rejected(self) -> None:
        findings = self.findings_for("${{ github.workflow }}-${{ github.head_ref || github.ref_name }}")
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("github.event_name", findings[0].message)

    def test_a_per_event_group_is_accepted(self) -> None:
        self.assertEqual(
            self.findings_for(
                "${{ github.workflow }}-${{ github.event_name }}-${{ github.head_ref || github.ref_name }}"
            ),
            [],
        )

    def test_the_real_tree_passes(self) -> None:
        findings: list[contract.Finding] = []
        contract.check_ci_concurrency_per_event(findings)
        self.assertEqual(findings, [], [f.message for f in findings])
