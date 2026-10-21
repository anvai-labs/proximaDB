"""Contract tests for the duplication ratchet guard (scripts/dup_ratchet.py).

The ratchet must fail when duplication exceeds baseline + tolerance, pass
when comfortably under, and never silently pass on a missing baseline.
These run the real census (a few seconds each) — keep the count small.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "dup_ratchet.py"
BASELINE = REPO / "docs" / "10-quality" / "DUP_RATCHET_BASELINE.json"


def run_check():
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


class DupRatchetContract(unittest.TestCase):
    def setUp(self):
        self.saved = BASELINE.read_text()

    def tearDown(self):
        BASELINE.write_text(self.saved)

    def _write_baseline(self, percentage):
        doc = json.loads(self.saved)
        doc["percentage"] = percentage
        BASELINE.write_text(json.dumps(doc, indent=2) + "\n")

    def test_ratchet_fails_when_baseline_beats_reality(self):
        # A baseline of 0% makes any real duplication a regression.
        self._write_baseline(0.0)
        result = run_check()
        self.assertNotEqual(
            result.returncode, 0, "check must fail when current > baseline + tolerance"
        )
        self.assertIn("ratchet", result.stderr + result.stdout)

    def test_ratchet_passes_when_baseline_is_generous(self):
        # A 99% baseline cannot fail; the guard must pass (and stay fast).
        self._write_baseline(99.0)
        result = run_check()
        self.assertEqual(
            result.returncode, 0, f"check must pass under a generous baseline: {result.stderr}"
        )
        self.assertIn("OK", result.stdout)

    def test_ratchet_fails_on_absolute_lines_even_with_generous_pct(self):
        # The line metric must fire even when percentage is comfortably
        # under (the dilution case: a big clean import shrinks pct).
        self._write_baseline(99.0)
        doc = json.loads(BASELINE.read_text())
        doc["duplicated_lines"] = 1000
        BASELINE.write_text(json.dumps(doc, indent=2) + "\n")
        result = run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicated lines", result.stderr + result.stdout)

    def test_old_format_baseline_without_lines_key_passes(self):
        # Baselines predating the dual metric (no duplicated_lines) fall
        # back to percentage-only.
        self._write_baseline(99.0)
        doc = json.loads(BASELINE.read_text())
        doc.pop("duplicated_lines", None)
        BASELINE.write_text(json.dumps(doc, indent=2) + "\n")
        result = run_check()
        self.assertEqual(result.returncode, 0)

    def test_missing_baseline_fails_closed(self):
        BASELINE.unlink()
        result = run_check()
        self.assertNotEqual(result.returncode, 0, "missing baseline must fail, not pass")


if __name__ == "__main__":
    unittest.main()
