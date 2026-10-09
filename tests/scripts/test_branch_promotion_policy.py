#!/usr/bin/env python3
"""Executable negative controls for the trusted branch-route workflow."""

from __future__ import annotations

import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/branch-route-guard.yml"
REPOSITORY = "anvai-labs/proximaDB"


def policy_script() -> str:
    """Extract the exact shell policy that GitHub Actions executes."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start_marker = "          # branch-route-policy:start\n"
    end_marker = "          # branch-route-policy:end\n"
    start = workflow.index(start_marker) + len(start_marker)
    end = workflow.index(end_marker, start)
    return textwrap.dedent(workflow[start:end])


def run_route(
    base_ref: str,
    head_ref: str,
    head_repository: str = REPOSITORY,
    repository: str = REPOSITORY,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "BASE_REF": base_ref,
            "HEAD_REF": head_ref,
            "HEAD_REPOSITORY": head_repository,
            "REPOSITORY": repository,
        }
    )
    return subprocess.run(
        ["bash", "-c", policy_script()],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


class BranchPromotionPolicyTest(unittest.TestCase):
    def assert_allowed(self, base_ref: str, head_ref: str) -> None:
        result = run_route(base_ref, head_ref)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_rejected(
        self,
        base_ref: str,
        head_ref: str,
        head_repository: str = REPOSITORY,
    ) -> None:
        result = run_route(base_ref, head_ref, head_repository)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_allowed_routes(self) -> None:
        for base_ref, head_ref in (
            ("develop", "feature/vector-filter"),
            ("develop", "fix/auth-failure"),
            ("develop", "debug/query-plan"),
            ("develop", "chore/ci-cleanup"),
            ("develop", "dependabot/cargo/sysinfo-0.39.6"),
            ("qa", "develop"),
            ("main", "qa"),
        ):
            with self.subTest(base_ref=base_ref, head_ref=head_ref):
                self.assert_allowed(base_ref, head_ref)

    def test_wrong_promotion_sources_are_rejected(self) -> None:
        for base_ref, head_ref in (
            ("qa", "feature/vector-filter"),
            ("qa", "promote/develop-to-qa-v030-r10"),
            ("qa", "main"),
            ("main", "develop"),
            ("main", "feature/vector-filter"),
            ("main", "promote/qa-to-main-v030-r10"),
        ):
            with self.subTest(base_ref=base_ref, head_ref=head_ref):
                self.assert_rejected(base_ref, head_ref)

    def test_fork_cannot_spoof_a_promotion_branch_name(self) -> None:
        for base_ref, head_ref in (("qa", "develop"), ("main", "qa")):
            with self.subTest(base_ref=base_ref, head_ref=head_ref):
                self.assert_rejected(
                    base_ref,
                    head_ref,
                    head_repository="untrusted-fork/proximaDB",
                )

    def test_reserved_and_unknown_routes_are_rejected(self) -> None:
        for base_ref, head_ref in (
            ("develop", "develop"),
            ("develop", "qa"),
            ("develop", "main"),
            ("development", "feature/vector-filter"),
            ("feature/integration", "fix/auth-failure"),
            ("", "feature/vector-filter"),
            ("develop", ""),
        ):
            with self.subTest(base_ref=base_ref, head_ref=head_ref):
                self.assert_rejected(base_ref, head_ref)

    def test_workflow_cannot_execute_pull_request_content(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pull_request_target:", workflow)
        self.assertNotIn("    branches:", workflow)
        self.assertNotIn("    branches-ignore:", workflow)
        self.assertIn("permissions: {}", workflow)
        self.assertEqual(workflow.count("permissions:"), 1)
        self.assertEqual(workflow.count("        run: |"), 1)
        self.assertNotIn("uses:", workflow)
        for event_binding in (
            "BASE_REF: ${{ github.event.pull_request.base.ref }}",
            "HEAD_REF: ${{ github.event.pull_request.head.ref }}",
            "HEAD_REPOSITORY: ${{ github.event.pull_request.head.repo.full_name }}",
            "REPOSITORY: ${{ github.repository }}",
        ):
            self.assertIn(event_binding, workflow)
        self.assertNotIn("actions/checkout", workflow)
        self.assertNotIn("github.event.pull_request.head.sha", workflow)
        self.assertNotIn("github.head_ref", workflow)
        self.assertNotIn("github.base_ref", workflow)
        self.assertNotIn("${{", policy_script())
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("curl ", workflow)
        self.assertNotIn("wget ", workflow)
        self.assertNotIn("gh pr checkout", workflow)


if __name__ == "__main__":
    unittest.main()
