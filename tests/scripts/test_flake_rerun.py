"""Unit tests for scripts/flake_rerun.py (the Flake Rerun matcher).

The fixtures are verbatim excerpts of real job logs, so the matcher is tested
against what GitHub actually writes, not against what we assumed it writes --
the assumption (shutdown text in an annotation) is why the workflow never fired.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import flake_rerun  # noqa: E402

# Job 117020211299 (run 38905751066, `Feature Matrix (enterprise-catalogs)`,
# 2026-10-16): the runner was reclaimed mid-compile. Its ONLY annotation is
# "Process completed with exit code 143."
RUNNER_SHUTDOWN_LOG = """\
2026-10-16T14:59:09.5923474Z \x1b[1m\x1b[33mwarning\x1b[0m\x1b[1m: field `filesystem_factory` is never read\x1b[0m
2026-10-16T14:59:09.5924396Z     \x1b[1m\x1b[94m--> \x1b[0msrc/storage/entity_store/../entity_store_legacy.rs:1323:9
2026-10-16T14:59:09.5926662Z
2026-10-16T15:00:59.7899822Z ##[error]The runner has received a shutdown signal. This can happen when the runner service is stopped, or a manually started runner is canceled.
2026-10-16T15:00:59.7901123Z ##[error]Process completed with exit code 143.
2026-10-16T15:00:59.9291152Z Cleaning up orphan processes
"""
RUNNER_SHUTDOWN_ANNOTATIONS = ["Process completed with exit code 143."]

# Job 115940312892 (run 38589401582, `Security Audit`, 2026-10-14): a genuine
# failure -- cargo audit found 4 vulnerabilities.
GENUINE_FAILURE_LOG = """\
2026-10-14T15:31:00.2022495Z \x1b[0m\x1b[0m\x1b[1m\x1b[31mID:       \x1b[0m RUSTSEC-2026-0346
2026-10-14T15:31:00.2023685Z \x1b[0m\x1b[0m\x1b[1m\x1b[31mSolution: \x1b[0m Upgrade to >=0.23.46
2026-10-14T15:31:00.2024580Z \x1b[0m\x1b[0m\x1b[1m\x1b[31merror:\x1b[0m 4 vulnerabilities found!
2026-10-14T15:31:00.2058165Z ##[error]Process completed with exit code 1.
2026-10-14T15:31:00.2197815Z Post job cleanup.
"""
GENUINE_FAILURE_ANNOTATIONS = ["Process completed with exit code 1."]

LOST_COMMUNICATION_ANNOTATION = (
    "The hosted runner: GitHub Actions 1000001234 lost communication with the "
    "server. Anything in your workflow that terminates the runner process, "
    "starves it for CPU/Memory, or blocks its network access can cause this error."
)


class InfraKillMatcherTest(unittest.TestCase):
    def test_real_runner_shutdown_log_is_an_infra_kill(self) -> None:
        self.assertTrue(
            flake_rerun.job_is_infra_kill(RUNNER_SHUTDOWN_LOG, RUNNER_SHUTDOWN_ANNOTATIONS)
        )

    def test_its_annotations_alone_do_not_carry_the_shutdown_text(self) -> None:
        # Pins the root cause: the old annotation-text matcher saw only this.
        self.assertFalse(
            any("shutdown signal" in a for a in RUNNER_SHUTDOWN_ANNOTATIONS)
        )

    def test_real_genuine_failure_is_not_an_infra_kill(self) -> None:
        self.assertFalse(
            flake_rerun.job_is_infra_kill(GENUINE_FAILURE_LOG, GENUINE_FAILURE_ANNOTATIONS)
        )

    def test_lost_communication_annotation_is_an_infra_kill(self) -> None:
        self.assertTrue(
            flake_rerun.job_is_infra_kill("", [LOST_COMMUNICATION_ANNOTATION])
        )

    def test_lost_communication_error_line_in_the_log_is_an_infra_kill(self) -> None:
        log = (
            "2026-10-16T15:00:59.7899822Z ##[error]The hosted runner: GitHub Actions 7 "
            "lost communication with the server.\n"
        )
        self.assertTrue(flake_rerun.job_is_infra_kill(log, []))

    def test_exit_143_annotation_counts_only_when_the_log_is_unreadable(self) -> None:
        self.assertTrue(flake_rerun.job_is_infra_kill(None, RUNNER_SHUTDOWN_ANNOTATIONS))
        # A readable log WITHOUT the signature is evidence against an infra kill.
        log = "2026-10-16T15:00:59Z ##[error]Process completed with exit code 143.\n"
        self.assertFalse(flake_rerun.job_is_infra_kill(log, RUNNER_SHUTDOWN_ANNOTATIONS))

    def test_unreadable_log_with_a_genuine_exit_code_is_not_an_infra_kill(self) -> None:
        self.assertFalse(flake_rerun.job_is_infra_kill(None, GENUINE_FAILURE_ANNOTATIONS))

    def test_a_test_that_prints_the_phrase_does_not_match(self) -> None:
        log = (
            "2026-10-16T15:00:00Z test runner_shutdown ... asserting "
            "'The runner has received a shutdown signal' is handled\n"
            "2026-10-16T15:00:01Z ##[error]Process completed with exit code 101.\n"
        )
        self.assertFalse(flake_rerun.job_is_infra_kill(log, []))


class RerunDecisionTest(unittest.TestCase):
    def test_all_infra_reruns(self) -> None:
        self.assertTrue(flake_rerun.should_rerun([True, True]))

    def test_one_genuine_failure_blocks_the_rerun(self) -> None:
        self.assertFalse(flake_rerun.should_rerun([True, False]))

    def test_no_failures_is_not_a_rerun(self) -> None:
        self.assertFalse(flake_rerun.should_rerun([]))

    def test_aggregator_is_never_evidence(self) -> None:
        jobs = [
            {"id": 1, "name": "CI Success", "conclusion": "failure"},
            {"id": 2, "name": "Feature Matrix (enterprise-catalogs)", "conclusion": "failure"},
            {"id": 3, "name": "Rust Format", "conclusion": "success"},
        ]
        self.assertEqual([j["id"] for j in flake_rerun.failed_jobs(jobs)], [2])


class WorkflowWiringTest(unittest.TestCase):
    WORKFLOW = REPO_ROOT / ".github/workflows/flake-rerun.yml"

    def test_workflow_runs_the_tested_matcher(self) -> None:
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python3 scripts/flake_rerun.py", text)
        # Rerun at most once: first attempts only.
        self.assertIn("github.event.workflow_run.run_attempt == 1", text)

    def test_workflow_no_longer_matches_annotation_text_inline(self) -> None:
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("checks.listAnnotations", text)


if __name__ == "__main__":
    unittest.main()
