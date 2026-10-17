"""Tests for scripts/await_prerelease_ci.sh (the release Pre-Release CI gate).

`gh` is replaced by a stub on PATH that prints a canned, already-filtered run
list (the script's own `--jq` filter is the API's concern, not ours), so these
pin the script's decisions and its GITHUB_OUTPUT contract with release.yml:
on success it hands `run_id` of the newest successful run to the release, which
downloads that run's binaries instead of rebuilding them (CI redesign PR1).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "await_prerelease_ci.sh"


@unittest.skipUnless(shutil.which("jq") and shutil.which("bash"), "needs bash + jq")
class AwaitPrereleaseCiTest(unittest.TestCase):
    def run_script(self, runs: list[dict], github_output: bool = True):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            stub = tmp / "bin" / "gh"
            stub.parent.mkdir()
            stub.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{json.dumps(runs)}\nEOF\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            out = tmp / "github_output"
            env = dict(os.environ)
            env["PATH"] = f"{stub.parent}{os.pathsep}{env['PATH']}"
            env["PRERELEASE_POLL_SECONDS"] = "0"
            env.pop("GITHUB_OUTPUT", None)
            if github_output:
                env["GITHUB_OUTPUT"] = str(out)
            proc = subprocess.run(
                ["bash", str(SCRIPT), "o/r", "deadbeef", "1", "0"],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            written = out.read_text() if out.exists() else ""
            return proc, written

    def test_success_writes_the_newest_successful_run_id(self) -> None:
        proc, written = self.run_script(
            [
                {"id": 300, "status": "in_progress", "conclusion": None},
                {"id": 200, "status": "completed", "conclusion": "success"},
                {"id": 100, "status": "completed", "conclusion": "success"},
            ]
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(written, "run_id=200\n")

    def test_red_newest_run_fails_and_writes_nothing(self) -> None:
        proc, written = self.run_script(
            [{"id": 300, "status": "completed", "conclusion": "failure"}]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(written, "")

    def test_without_github_output_it_still_succeeds(self) -> None:
        proc, written = self.run_script(
            [{"id": 200, "status": "completed", "conclusion": "success"}],
            github_output=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(written, "")


if __name__ == "__main__":
    unittest.main()
