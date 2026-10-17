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

A failed CI verdict (qa_push_ci / qa_pr_ci only) is accepted when
ci/release-overrides.toml documents it (approver, reason, evidence URL naming
the failed run). It is then reported as OVERRIDDEN -- `passed` stays false,
`accepted` is true -- never as a pass.

Writes a JSON evidence document (always, also on failure) and a Markdown table
to --summary. Exit 1 when any check fails without a valid override.

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
import re
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


def pick_push_run(runs: list[dict[str, Any]], repo: str, head_sha: str) -> dict[str, Any] | None:
    """Pure: the newest CI run for the push to THIS repo's `qa` at head_sha."""
    return _newest([
        r for r in runs
        if r.get("event") == "push"
        and r.get("head_branch") == "qa"
        and r.get("head_sha") == head_sha
        and ((r.get("head_repository") or {}).get("full_name") == repo)
    ])


def pick_pr_run(runs: list[dict[str, Any]], repo: str, pr: dict[str, Any]) -> dict[str, Any] | None:
    """Pure: the newest CI run of the qa -> main PR ITSELF.

    Not "any pull_request run whose head branch is called qa at this sha": a
    fork's branch named `qa`, or another PR from the same head, could carry a
    green run. A run counts only when its head is THIS repository's `qa` at the
    PR's head sha, it was created while the PR was open (opened <= created <=
    merged), and -- when GitHub lists the run's PRs -- that list names this PR.
    """
    opened, merged = pr.get("created_at") or "", pr.get("merged_at") or ""

    def belongs(r: dict[str, Any]) -> bool:
        if r.get("event") != "pull_request" or r.get("head_branch") != "qa":
            return False
        if r.get("head_sha") != pr.get("head_sha"):
            return False
        if (r.get("head_repository") or {}).get("full_name") != repo:
            return False
        created = r.get("created_at") or ""
        if not (opened and merged and opened <= created <= merged):
            return False
        listed = [x.get("number") for x in (r.get("pull_requests") or [])]
        return not listed or pr.get("number") in listed

    return _newest([r for r in runs if belongs(r)])


def _run_record(repo: str, run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": run.get("id"),
        "url": run.get("html_url"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "run_attempt": run.get("run_attempt"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "head_repository": (run.get("head_repository") or {}).get("full_name"),
        "ci_success": _ci_success_job(repo, int(run["id"])),
    }


def _ci_runs(repo: str, pr: dict[str, Any]) -> dict[str, Any]:
    """CI Success of the qa push run and of the qa -> main PR's own run."""
    head_sha = pr["head_sha"]
    runs = gh_json(
        f"repos/{repo}/actions/workflows/{CI_WORKFLOW}/runs?head_sha={head_sha}&per_page=100"
    ).get("workflow_runs", [])
    return {
        "push": _run_record(repo, pick_push_run(runs, repo, head_sha)),
        "pull_request": _run_record(repo, pick_pr_run(runs, repo, pr)),
    }


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
            and ((pr.get("head") or {}).get("repo") or {}).get("full_name") == repo
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
        "created_at": pr.get("created_at"),
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
        facts["qa_ci"] = _ci_runs(repo, facts["qa_pr"])
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
    for name, key in (("qa_push_ci", "push"), ("qa_pr_ci", "pull_request")):
        ok, detail = _ci_ok(ci.get(key))
        add(name, ok, detail)
        checks[-1]["run_id"] = (ci.get(key) or {}).get("run_id")
    return checks


# ---------------------------------------------------------------------- overrides

OVERRIDES_FILE = ROOT / "ci" / "release-overrides.toml"
# Only CI verdicts can be overridden (an infrastructure kill can turn one red
# without saying anything about the code). The structural facts -- the tag is
# on main, it is a qa->main merge, the tree is the tested tree, Pre-Release CI
# passed -- are never negotiable.
OVERRIDABLE = frozenset({"qa_push_ci", "qa_pr_ci"})
OVERRIDE_FIELDS = ("version", "check", "reason", "approver", "evidence", "followup", "recorded")


class OverrideError(ValueError):
    pass


def load_overrides(path: pathlib.Path = OVERRIDES_FILE) -> list[dict[str, str]]:
    """Parse and validate ci/release-overrides.toml. A malformed entry is an
    error, never ignored: a typo must not silently turn an override off (or on)."""
    if not path.exists():
        return []
    import tomllib

    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = doc.get("override", [])
    if not isinstance(entries, list):
        raise OverrideError(f"{path.name}: `override` must be an array of tables")
    approvers = doc.get("approvers")
    if entries and (not isinstance(approvers, list) or not approvers
                    or not all(isinstance(a, str) and a.strip() for a in approvers)):
        raise OverrideError(f"{path.name}: a non-empty top-level `approvers` list (GitHub logins) is required")
    seen: set[tuple[str, str]] = set()
    for i, e in enumerate(entries):
        where = f"{path.name} override #{i + 1}"
        for field in OVERRIDE_FIELDS:
            if not isinstance(e.get(field), str) or not e[field].strip():
                raise OverrideError(f"{where}: `{field}` is required and must be a non-empty string")
        if e["approver"] not in approvers:
            raise OverrideError(f"{where}: approver `{e['approver']}` is not in `approvers` {approvers}")
        if e["check"] not in OVERRIDABLE:
            raise OverrideError(f"{where}: check `{e['check']}` is not overridable (only {sorted(OVERRIDABLE)})")
        if evidence_run_id(e["evidence"]) is None:
            raise OverrideError(
                f"{where}: evidence must be a run URL https://github.com/<owner>/<repo>/actions/runs/<id>[/job/<id>]"
            )
        if not _dt.date.fromisoformat(e["recorded"]):
            raise OverrideError(f"{where}: recorded must be YYYY-MM-DD")
        key = (e["version"], e["check"])
        if key in seen:
            raise OverrideError(f"{where}: duplicate override for {key}")
        seen.add(key)
    return entries


EVIDENCE_RE = re.compile(
    r"^https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/actions/runs/(?P<run>[0-9]+)"
    r"(?:/attempts/[0-9]+)?(?:/job/[0-9]+)?/?$"
)


def evidence_run_id(url: str, repo: str | None = None) -> int | None:
    """The run id an evidence URL names, parsed (not substring-matched); None
    when it is not a run URL, or not one of `repo` when given."""
    m = EVIDENCE_RE.match(url.strip())
    if m is None or (repo is not None and m.group("repo") != repo):
        return None
    return int(m.group("run"))


def apply_overrides(checks: list[dict[str, Any]], version: str, entries: list[dict[str, str]],
                    repo: str | None = None, merged_by: str | None = None) -> list[str]:
    """Pure: mark failed checks that a valid entry covers as overridden.

    An override applies only to a check that FAILED, only when its evidence URL
    names exactly the run id that failed (in `repo`), and -- for `qa_pr_ci` --
    only when the approver is the person who merged the qa -> main PR over the
    red run. Returns problems (stale or mismatched entries for this version),
    which the caller reports as failures.
    """
    problems: list[str] = []
    by_name = {c["name"]: c for c in checks}
    for e in entries:
        if e["version"] != version:
            continue
        c = by_name.get(e["check"])
        if c is None:
            problems.append(f"override for `{e['check']}`: no such check ran")
        elif c["ok"]:
            problems.append(f"override for `{e['check']}` is stale: the check passes -- delete the entry")
        elif not c.get("run_id") or evidence_run_id(e["evidence"], repo) != int(c["run_id"]):
            problems.append(
                f"override for `{e['check']}`: evidence {e['evidence']} does not name the failed run {c.get('run_id')}"
                + (f" of {repo}" if repo else "")
            )
        elif e["check"] == "qa_pr_ci" and merged_by is not None and e["approver"] != merged_by:
            problems.append(
                f"override for `qa_pr_ci`: approver `{e['approver']}` is not who merged the qa -> main PR (`{merged_by}`)"
            )
        else:
            c["overridden"] = {k: e[k] for k in ("reason", "approver", "evidence", "followup", "recorded")}
    return problems


def status(c: dict[str, Any]) -> str:
    return "pass" if c["ok"] else ("overridden" if c.get("overridden") else "fail")


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
        mark = {"pass": "PASS", "fail": "**FAIL**", "overridden": "**OVERRIDDEN**"}[status(c)]
        lines.append(f"| `{c['name']}` | {mark} | {c['detail'].replace('|', '/')} |")
    lines.append("")
    for c in doc["checks"]:
        o = c.get("overridden")
        if o:
            lines += [
                f"> **OVERRIDDEN `{c['name']}`** -- this check FAILED and is accepted only by a documented override "
                f"(`ci/release-overrides.toml`).",
                f"> Approver: `{o['approver']}` ({o['recorded']}). Evidence: {o['evidence']}",
                f"> Reason: {' '.join(o['reason'].split())}",
                f"> Follow-up: {o['followup']}",
                "",
            ]
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
    try:
        problems = apply_overrides(checks, args.version, load_overrides(), repo=args.repo,
                                   merged_by=(facts.get("qa_pr") or {}).get("merged_by"))
    except (OverrideError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
        problems = [f"ci/release-overrides.toml is invalid: {exc}"]
    for msg in problems:
        checks.append({"name": "overrides", "ok": False, "detail": msg})
    overridden = [c["name"] for c in checks if status(c) == "overridden"]
    doc = {
        "schema_version": 1,
        "kind": "release-provenance",
        "generated_at": started,
        # A release with overrides is NOT reported as clean: `passed` is false
        # whenever any check failed, and `accepted` says whether every failure
        # is covered by a documented override.
        "passed": all(c["ok"] for c in checks),
        "accepted": all(status(c) != "fail" for c in checks),
        "overridden": overridden,
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
    if not doc["accepted"]:
        failed = ", ".join(c["name"] for c in checks if status(c) == "fail")
        print(f"::error::release provenance FAILED for v{args.version}: {failed}")
        return 1
    if overridden:
        print(f"::warning::release provenance for v{args.version} is OVERRIDDEN, not passed: "
              f"{', '.join(overridden)} (ci/release-overrides.toml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
