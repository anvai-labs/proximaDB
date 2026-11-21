# Copyright (C) 2026 ProximaDB
# SPDX-License-Identifier: Apache-2.0
"""Wiring guard for qa-gate.yml's required aggregate (TD-CI-8 Pilot 4 slice 1).

Branch protection on qa/main requires the context "QA Gate Success" — the
check name of the `qa-gate-summary` job. The failure modes these tests pin
are the ones that would silently disarm the promotion gate:

  * a job added to qa-gate.yml but NOT to the aggregate's `needs:` runs
    outside the required gate forever (the mlflow-conformance shape: a job
    outside the aggregator gates nothing);
  * the aggregate's hard-gate condition missing a job's failure/cancelled
    conjunct;
  * the advisory job (rust-coverage-ratchet) silently becoming required, or
    losing its continue-on-error — both change promotion semantics in ways
    that belong to TD-CI-8 Pilot 3, not a drive-by edit;
  * the context name drifting away from what branch protection requires.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "qa-gate.yml"

# The single context name branch protection on qa/main requires. Changing
# this string WITHOUT updating branch protection blocks every promotion
# forever (the required context is never reported).
REQUIRED_CONTEXT = "QA Gate Success"

# Advisory jobs: recorded in the summary, non-blocking on failure.
ADVISORY_JOBS = {"rust-coverage-ratchet"}


def _load_workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


class QaGateSummaryWiring(unittest.TestCase):
    def setUp(self):
        self.doc = _load_workflow()
        self.jobs = self.doc["jobs"]
        self.summary = self.jobs.get("qa-gate-summary")
        self.other_jobs = {k for k in self.jobs if k != "qa-gate-summary"}

    def test_aggregate_job_exists_with_required_context_name(self):
        self.assertIsNotNone(self.summary, "qa-gate-summary job missing")
        self.assertEqual(self.summary["name"], REQUIRED_CONTEXT)

    def test_needs_covers_every_other_job(self):
        needs = set(self.summary["needs"])
        self.assertEqual(
            needs,
            self.other_jobs,
            "every qa-gate job must be in the aggregate's needs — a job "
            "outside the aggregator gates nothing (the mlflow-conformance "
            "lesson)",
        )

    def test_aggregate_always_runs(self):
        self.assertEqual(self.summary["if"], "always()")

    def _hard_gate_condition(self) -> str:
        steps = self.summary["steps"]
        gate = next(s for s in steps if s["name"].startswith("Require qa-gate jobs"))
        cond = gate["if"]
        self.assertIn("needs.", cond, "hard gate must read needs results")
        return cond

    def test_hard_gate_names_every_required_job(self):
        cond = self._hard_gate_condition()
        for job in self.other_jobs - ADVISORY_JOBS:
            for outcome in ("failure", "cancelled"):
                self.assertIn(
                    f"needs.{job}.result == '{outcome}'",
                    cond,
                    f"the hard gate must fail on {job} {outcome}",
                )

    def test_advisory_job_fails_gate_only_when_cancelled(self):
        cond = self._hard_gate_condition()
        advisory = next(iter(ADVISORY_JOBS))
        self.assertNotIn(
            f"needs.{advisory}.result == 'failure'",
            cond,
            "the advisory coverage ratchet must NOT gate on failure "
            "(TD-CI-8 Pilot 3 decides when it flips required)",
        )
        self.assertIn(
            f"needs.{advisory}.result == 'cancelled'",
            cond,
            "a cancelled advisory run still gates — a cancelled promotion "
            "proves nothing",
        )

    def test_advisory_job_is_job_level_continue_on_error(self):
        for job in ADVISORY_JOBS:
            self.assertTrue(
                self.jobs[job].get("continue-on-error") is True,
                f"{job} is expected to be job-level continue-on-error; if it "
                "flipped required, update ADVISORY_JOBS here AND the "
                "aggregate's hard gate in the same PR",
            )


if __name__ == "__main__":
    unittest.main()
