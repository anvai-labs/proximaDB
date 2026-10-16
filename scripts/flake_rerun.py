#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rerun a failed workflow run ONCE when every failure is a runner infra kill.

Called by `.github/workflows/flake-rerun.yml` on `workflow_run: completed`.

Why this is a script, not an inline github-script matcher: the previous inline
version searched check-run ANNOTATIONS for "The runner has received a shutdown
signal". That text never appears in an annotation -- the annotation GitHub
records for a reclaimed runner is just "Process completed with exit code 143."
The shutdown text exists only in the job LOG, e.g. (job 117020211299, run
38905751066, 2026-10-16):

    ##[error]The runner has received a shutdown signal. This can happen when ...
    ##[error]Process completed with exit code 143.

so the workflow never fired. The matcher now reads the log
(`GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs`, the endpoint behind
`github.rest.actions.downloadJobLogsForWorkflowRun`) and is unit-tested against
real log excerpts (tests/scripts/test_flake_rerun.py).

Safety limits (unchanged): rerun at most once (the workflow only fires on
attempt 1), and only when ALL failed jobs carry the infra signature -- a single
genuine failure means no rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# The runner writes these as `##[error]` lines in the job log. Anchoring on the
# error marker keeps a test that merely PRINTS the phrase from matching.
INFRA_LOG_RE = re.compile(
    r"##\[error\].*(?:The runner has received a shutdown signal|lost communication with the server)"
)
# Annotations can carry the lost-communication text directly.
INFRA_ANNOTATION_RE = re.compile(
    r"The runner has received a shutdown signal|lost communication with the server"
)
# SIGTERM (128 + 15): what the runner's shutdown delivers to the step. Only
# trusted when the log cannot be read -- with a readable log, the explicit
# signature above is the evidence.
EXIT_143_RE = re.compile(r"Process completed with exit code 143\b")

# Aggregator jobs fail whenever any dependency fails; they are never evidence.
AGGREGATOR_JOBS = frozenset({"CI Success"})


def job_is_infra_kill(log_text: str | None, annotations: list[str]) -> bool:
    """True when this failed job was killed by runner infrastructure."""
    if log_text is not None and INFRA_LOG_RE.search(log_text):
        return True
    if any(INFRA_ANNOTATION_RE.search(a or "") for a in annotations):
        return True
    if log_text is None:
        # Fallback: no log (expired, or the runner died before uploading it).
        return any(EXIT_143_RE.search(a or "") for a in annotations)
    return False


def failed_jobs(jobs: list[dict]) -> list[dict]:
    return [
        j
        for j in jobs
        if j.get("conclusion") == "failure" and j.get("name") not in AGGREGATOR_JOBS
    ]


def should_rerun(verdicts: list[bool]) -> bool:
    """Rerun only when there is at least one failure and ALL are infra kills."""
    return bool(verdicts) and all(verdicts)


def _gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True)


def _gh_json(args: list[str]) -> object:
    proc = _gh(args)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = _gh_json(
        [
            "--paginate",
            "--slurp",
            f"repos/{args.repo}/actions/runs/{args.run_id}/jobs?filter=latest&per_page=100",
        ]
    )
    all_jobs = [j for page in jobs for j in page.get("jobs", [])]  # type: ignore[union-attr]
    failed = failed_jobs(all_jobs)
    if not failed:
        print("No non-aggregator failed jobs; nothing to do.")
        return 0

    verdicts: list[bool] = []
    for job in failed:
        log = _gh([f"repos/{args.repo}/actions/jobs/{job['id']}/logs"])
        log_text = log.stdout if log.returncode == 0 else None
        try:
            pages = _gh_json(
                ["--paginate", "--slurp", f"repos/{args.repo}/check-runs/{job['id']}/annotations"]
            )
            annotations = [a.get("message", "") for page in pages for a in page]  # type: ignore[union-attr]
        except RuntimeError as exc:
            print(f"::warning::{exc}")
            annotations = []
        verdict = job_is_infra_kill(log_text, annotations)
        source = "log" if log_text is not None else "annotations (log unavailable)"
        print(f"{job['name']}: infra-kill={verdict} [{source}]")
        verdicts.append(verdict)

    if not should_rerun(verdicts):
        print("At least one failure is not an infra kill; not rerunning.")
        return 0

    print(
        f"::notice::All {len(failed)} failed job(s) show the runner-shutdown "
        "signature; rerunning failed jobs once."
    )
    if args.dry_run:
        return 0
    proc = _gh(["-X", "POST", f"repos/{args.repo}/actions/runs/{args.run_id}/rerun-failed-jobs"])
    if proc.returncode != 0:
        print(f"::error::rerun request failed: {proc.stderr.strip()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
