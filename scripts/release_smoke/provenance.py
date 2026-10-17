#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove a release `vX.Y.Z` was cut from a QA-proven tree.

Release evidence, part (A). Every fact comes from the GitHub API (`gh api`);
nothing is taken from the release workflow's own say-so. The release is
QA-based only when ALL of these hold:

  tag_resolves        `vX.Y.Z` dereferences (annotated or lightweight) to a commit
  on_main             that commit is reachable from `main`
  qa_merge            it is the merge commit of a merged PR `qa` -> `main`
  qa_tree_equal       the PR's `qa` head has the SAME git tree as the tag, so the
                      bytes QA tested are the bytes released (a merge that pulled
                      in a main-only change would differ here)
  prerelease_ci       a building Pre-Release CI run (prerelease-ci.yml) passed on
                      the tag commit -- the query is scripts/await_prerelease_ci.sh
                      itself, not a copy of it
  qa_push_ci          `CI Success` passed in the CI run for the push to `qa` of
                      that head (the medium tier that gates develop -> qa)
  qa_pr_ci            `CI Success` passed in the newest CI run of the qa -> main
                      PR (the full tier). Branch protection requires it; a merge
                      over a red run is exactly what this evidence must surface.

Writes a JSON evidence document (always, also on failure) and a Markdown table
to --summary. Exit 1 when any check fails.

Usage:
  provenance.py --repo OWNER/NAME --version X.Y.Z --out evidence.json [--summary FILE]
Needs `gh` authenticated (GH_TOKEN) with contents/actions/pull-requests read.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
AWAIT_PRERELEASE = ROOT / "scripts" / "await_prerelease_ci.sh"
CI_WORKFLOW = "ci.yml"
CI_SUCCESS_JOB = "CI Success"


class GhError(RuntimeError):
    pass


def gh_json(path: str) -> Any:
    proc = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GhError(f"gh api {path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- facts


def _ci_success_job(repo: str, run_id: int) -> dict[str, Any] | None:
    jobs = gh_json(f"repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100")
    for job in jobs.get("jobs", []):
        if job.get("name") == CI_SUCCESS_JOB:
            return {
                "conclusion": job.get("conclusion"),
                "url": job.get("html_url"),
                "completed_at": job.get("completed_at"),
            }
    return None


def _newest(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return max(runs, key=lambda r: (r.get("created_at") or "", r.get("id") or 0))


def _ci_runs(repo: str, head_sha: str) -> dict[str, Any]:
    """Newest CI run for the qa push and for the qa -> main PR, with CI Success."""
    runs = gh_json(
        f"repos/{repo}/actions/workflows/{CI_WORKFLOW}/runs?head_sha={head_sha}&per_page=50"
    ).get("workflow_runs", [])
    out: dict[str, Any] = {}
    for key, pick in (
        ("push", lambda r: r.get("event") == "push" and r.get("head_branch") == "qa"),
        ("pull_request", lambda r: r.get("event") == "pull_request" and r.get("head_branch") == "qa"),
    ):
        run = _newest([r for r in runs if pick(r)])
        if run is None:
            out[key] = None
            continue
        out[key] = {
            "run_id": run.get("id"),
            "url": run.get("html_url"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "run_attempt": run.get("run_attempt"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "ci_success": _ci_success_job(repo, int(run["id"])),
        }
    return out


def _prerelease(repo: str, sha: str) -> dict[str, Any]:
    """Ask scripts/await_prerelease_ci.sh itself, with a short deadline.

    By the time a release exists its Pre-Release CI has long concluded, so a
    1-minute deadline and no grace window only ever answer "passed" or "not".
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_file = pathlib.Path(tmp) / "out"
        env = dict(os.environ, GITHUB_OUTPUT=str(out_file), PRERELEASE_POLL_SECONDS="15")
        proc = subprocess.run(
            ["bash", str(AWAIT_PRERELEASE), repo, sha, "1", "0"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        written = out_file.read_text() if out_file.exists() else ""
    run_id = None
    for line in written.splitlines():
        if line.startswith("run_id="):
            run_id = line.split("=", 1)[1].strip()
    info: dict[str, Any] = {
        "gate_exit_code": proc.returncode,
        "gate_output": (proc.stdout + proc.stderr).strip()[-2000:],
        "run_id": run_id,
    }
    if run_id:
        run = gh_json(f"repos/{repo}/actions/runs/{run_id}")
        info.update(
            url=run.get("html_url"),
            event=run.get("event"),
            head_branch=run.get("head_branch"),
            head_sha=run.get("head_sha"),
            conclusion=run.get("conclusion"),
            created_at=run.get("created_at"),
            updated_at=run.get("updated_at"),
        )
    return info


def _qa_pr(repo: str, sha: str) -> dict[str, Any] | None:
    """The merged qa -> main PR whose merge commit is `sha`.

    The commit's associated PRs first; then the merged qa -> main PR list, for
    the case where the association index has not caught up.
    """
    def match(pr: dict[str, Any]) -> bool:
        return (
            pr.get("merged_at") is not None
            and (pr.get("base") or {}).get("ref") == "main"
            and (pr.get("head") or {}).get("ref") == "qa"
            and pr.get("merge_commit_sha") == sha
        )

    candidates = [pr for pr in gh_json(f"repos/{repo}/commits/{sha}/pulls") if match(pr)]
    if not candidates:
        owner = repo.split("/", 1)[0]
        listed = gh_json(
            f"repos/{repo}/pulls?state=closed&base=main&head={owner}:qa&per_page=100"
        )
        candidates = [pr for pr in listed if match(pr)]
    if not candidates:
        return None
    # The list endpoints omit `merged_by`; the single-PR read has it.
    pr = gh_json(f"repos/{repo}/pulls/{candidates[0]['number']}")
    return {
        "number": pr.get("number"),
        "url": pr.get("html_url"),
        "head_sha": (pr.get("head") or {}).get("sha"),
        "base_sha": (pr.get("base") or {}).get("sha"),
        "merged_at": pr.get("merged_at"),
        "merged_by": (pr.get("merged_by") or {}).get("login"),
    }


def gather(repo: str, version: str) -> dict[str, Any]:
    tag = f"v{version}"
    facts: dict[str, Any] = {"repo": repo, "version": version, "tag": tag}
    commit = gh_json(f"repos/{repo}/commits/{tag}")
    sha = commit["sha"]
    facts.update(
        sha=sha,
        tree=commit["commit"]["tree"]["sha"],
        commit_date=commit["commit"]["committer"]["date"],
        commit_subject=commit["commit"]["message"].splitlines()[0],
        parents=[p["sha"] for p in commit.get("parents", [])],
    )
    ref = gh_json(f"repos/{repo}/git/ref/tags/{tag}")
    facts["tag_object_type"] = ref["object"]["type"]
    compare = gh_json(f"repos/{repo}/compare/{sha}...main")
    facts["main_relation"] = {
        "status": compare.get("status"),
        "ahead_by": compare.get("ahead_by"),
        "behind_by": compare.get("behind_by"),
    }
    facts["qa_pr"] = _qa_pr(repo, sha)
    head = (facts["qa_pr"] or {}).get("head_sha")
    if head:
        facts["qa_head_tree"] = gh_json(f"repos/{repo}/commits/{head}")["commit"]["tree"]["sha"]
        facts["qa_ci"] = _ci_runs(repo, head)
    else:
        facts["qa_head_tree"] = None
        facts["qa_ci"] = {"push": None, "pull_request": None}
    facts["prerelease_ci"] = _prerelease(repo, sha)
    try:
        rel = gh_json(f"repos/{repo}/releases/tags/{tag}")
        facts["github_release"] = {
            "url": rel.get("html_url"),
            "published_at": rel.get("published_at"),
            "prerelease": rel.get("prerelease"),
        }
    except GhError as exc:
        facts["github_release"] = {"error": str(exc)}
    return facts


# --------------------------------------------------------------------------- verdict


def _ci_ok(run: dict[str, Any] | None) -> tuple[bool, str]:
    if run is None:
        return False, "no CI run found"
    job = run.get("ci_success")
    if job is None:
        return False, f"run {run['run_id']} has no `{CI_SUCCESS_JOB}` job"
    ok = job.get("conclusion") == "success" and run.get("conclusion") == "success"
    return ok, (
        f"run {run['run_id']} conclusion={run.get('conclusion')}, "
        f"`{CI_SUCCESS_JOB}`={job.get('conclusion')} ({run.get('url')})"
    )


def evaluate(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Pure: facts -> named checks. Kept separate so it is unit-testable."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("tag_resolves", bool(facts.get("sha")), f"{facts.get('tag')} -> {facts.get('sha')}")
    rel = facts.get("main_relation") or {}
    add(
        "on_main",
        rel.get("status") in ("identical", "ahead"),
        f"compare {str(facts.get('sha'))[:12]}...main: {rel.get('status')} (main ahead by {rel.get('ahead_by')})",
    )
    pr = facts.get("qa_pr")
    add(
        "qa_merge",
        pr is not None,
        f"PR #{pr['number']} qa -> main, merged {pr['merged_at']} by {pr['merged_by']}"
        if pr
        else "tag commit is not the merge commit of a merged qa -> main PR",
    )
    qa_tree = facts.get("qa_head_tree")
    add(
        "qa_tree_equal",
        qa_tree is not None and qa_tree == facts.get("tree"),
        f"qa head tree {qa_tree} vs tag tree {facts.get('tree')}",
    )
    pre = facts.get("prerelease_ci") or {}
    add(
        "prerelease_ci",
        pre.get("gate_exit_code") == 0 and bool(pre.get("run_id")),
        f"run {pre.get('run_id')} {pre.get('conclusion')} ({pre.get('url')})"
        if pre.get("run_id")
        else f"await_prerelease_ci.sh exit {pre.get('gate_exit_code')}: {pre.get('gate_output', '')[-300:]}",
    )
    ci = facts.get("qa_ci") or {}
    ok, detail = _ci_ok(ci.get("push"))
    add("qa_push_ci", ok, detail)
    ok, detail = _ci_ok(ci.get("pull_request"))
    add("qa_pr_ci", ok, detail)
    return checks


def render_summary(doc: dict[str, Any]) -> str:
    lines = [
        f"### Release provenance: {doc['facts'].get('tag')}",
        "",
        f"Commit `{doc['facts'].get('sha')}` / tree `{doc['facts'].get('tree')}`",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for c in doc["checks"]:
        mark = "PASS" if c["ok"] else "**FAIL**"
        lines.append(f"| `{c['name']}` | {mark} | {c['detail'].replace('|', '/')} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--summary", type=pathlib.Path)
    args = ap.parse_args()

    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        facts = gather(args.repo, args.version)
        checks = evaluate(facts)
    except (GhError, KeyError, ValueError) as exc:
        facts = {"repo": args.repo, "version": args.version, "tag": f"v{args.version}"}
        checks = [{"name": "gather", "ok": False, "detail": f"could not collect facts: {exc}"}]
    doc = {
        "schema_version": 1,
        "kind": "release-provenance",
        "generated_at": started,
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
        "facts": facts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = render_summary(doc)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(summary + "\n")
    print(summary)
    if not doc["passed"]:
        failed = ", ".join(c["name"] for c in checks if not c["ok"])
        print(f"::error::release provenance FAILED for v{args.version}: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
