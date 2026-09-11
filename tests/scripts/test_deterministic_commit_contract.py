#!/usr/bin/env python3
"""Negative controls for the nextest portion of the commit contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
