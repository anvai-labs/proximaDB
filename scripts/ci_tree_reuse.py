#!/usr/bin/env python3
"""Reuse CI results proven on a byte-identical tree at promotion (TD-CITIER-2).

A release crosses four `ci.yml` runs -- PR develop->qa, push qa, PR qa->main,
push main -- and in the 2026-10-15 promotion all four tested the SAME git tree
(318f14450b69). The expensive suites (unit + doc tests, the feature matrix,
seven integration lanes, the recall ratchet, the restart / MLflow / OpenAPI
suites, the Python SDK) are pure functions of the tree and the tier, so once
one promotion-tier run has passed them on a tree, re-running them on the same
tree measures nothing new and sits on the release critical path.

This script answers one question in the `changes` job: which reusable job
groups were already proven GREEN on exactly this tree, by a trusted run? The
answer is a `reused` output (`,key,key,` -- empty when nothing is reused) that
each reusable job ANDs into its `if:` as
`!contains(needs.changes.outputs.reused, ',<key>,')`.

How a proof is recorded and found
---------------------------------
Every promotion-tier run (medium_tier) uploads a tiny artifact named
`ci-tree-proof-<tree>` from its `changes` job. A later run lists artifacts by
that exact name (one API call), then for each producing run checks, through
the API, the run itself and its latest-attempt jobs. The artifact is only an
INDEX: the claim "these jobs passed" always comes from the jobs API, never
from artifact content.

Guarantees (every one is a guard clause below, and each has a mutation test)
----------------------------------------------------------------------------
* Context: reuse is considered only in a promotion context -- a push to qa or
  main, or a same-repository PR whose base is qa or main, at medium tier.
  feat->develop PRs, develop pushes, workflow_dispatch and forks never reuse.
* Identity: only the exact tree hash is looked up; a push proof is
  re-verified by resolving the proving run's head commit to a tree locally.
* Trust: the proving run must be this workflow file, in this repository (not
  a fork), and either a push to qa/main (the pushed commit's ci.yml, on a
  protected branch) or a PR on an exact promotion route, develop->qa or
  qa->main. A pull_request run executes the MERGE ref's ci.yml, which
  includes the BASE branch, so the head alone proves nothing: the PR is
  resolved through the pulls API (head SHA match; `run.pull_requests` is empty
  on real promotion runs), must be unique, same-repository, and have the
  route's base. Both sides of a route are protected, so the definition that
  wrote the proof is reviewed code.
* Evidence: a group is proven only if at least one of its jobs ran and EVERY
  one of its jobs concluded `success` in that run. A run with any failed,
  cancelled or still-running reusable job is not a proof. A FAILED or
  timed-out reusable job on the identical tree in ANY trusted run vetoes all
  reuse: two runs of one tree disagreeing is a flake or an environment
  dependency, and the safe reading of conflicting evidence is "run it".
* Freshness: proofs older than --max-age-hours (default 168) are ignored.
* Dependency safety: a group that other jobs `needs:` (rust-build) is reused
  only when every dependent is reused too, and never at the FULL tier, where
  the main-only extras (integration-test, benchmarks) consume its artifact.
  Main-only extras (coverage, docker-build, integration-test, CodeQL in its
  own workflow) are never in the reusable set.
* Fail open: ANY exception -- API/network error, missing permission, malformed
  JSON, git failure -- yields "reuse nothing". The script exits 0 whenever it
  can write its outputs, so it can never fail the `changes` job (whose failure
  must, and still does, fail CI Success).

Dependency-free on purpose (urllib + subprocess only): it runs in `changes`
before anything is installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

WORKFLOW_PATH = ".github/workflows/ci.yml"
PROOF_PREFIX = "ci-tree-proof-"
PROMOTION_BASES = ("qa", "main")
# The only routes branch-route-guard.yml admits into qa / main: head -> base.
# A PR proof must be one of these EXACT pairs. The head alone is not enough:
# a pull_request run executes the MERGE ref's ci.yml, which includes the BASE
# branch's definition, so a PR from develop into an unprotected base could
# carry a forged workflow (review of #2051).
PROMOTION_ROUTES = {"develop": "qa", "qa": "main"}
# Upper bound on paginated reads; exceeding it raises => fail open.
MAX_PAGES = 20
TRUSTED_PUSH_BRANCHES = ("qa", "main")
CHANGES_JOB_NAME = "Detect Changes"
DEFAULT_MAX_AGE_HOURS = 168.0
_TREE_RE = re.compile(r"^[0-9a-f]{40}$")
# Conclusions that are evidence AGAINST the tree (not merely "no evidence").
_RED = frozenset({"failure", "timed_out", "startup_failure"})


@dataclass(frozen=True)
class Group:
    """One reusable unit: a ci.yml job id and the job names it produces."""

    key: str
    exact: tuple[str, ...] = ()
    prefix: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    # Groups that `needs:` this one; reuse only if all of them are reused.
    dependents: tuple[str, ...] = ()
    # Never reuse at the FULL tier (main-only extras consume this job).
    not_at_full_tier: bool = False

    def matches(self, name: str) -> bool:
        if name in self.exclude:
            return False
        return name in self.exact or any(name.startswith(p) for p in self.prefix)


_LANES = ("storage", "query", "graph", "cluster", "api")

GROUPS: tuple[Group, ...] = (
    Group("rust-test", prefix=("Rust Tests (",)),
    Group(
        "feature-matrix",
        prefix=("Feature Matrix (",),
        exclude=("Feature Matrix (enterprise-catalogs)",),
    ),
    Group(
        "feature-matrix-enterprise-catalogs",
        exact=("Feature Matrix (enterprise-catalogs)",),
    ),
    Group("rust-compile-benches", exact=("Rust Compile (benches)",)),
    Group(
        "rust-build",
        prefix=("Rust Build (",),
        dependents=tuple(f"rust-integration-test-{lane}" for lane in _LANES),
        not_at_full_tier=True,
    ),
    Group("rust-integration-test-storage", exact=("Rust Integration Tests (storage)",)),
    Group("rust-integration-test-authz", exact=("Rust Integration Tests (authz e2e)",)),
    Group("rust-integration-test-query", exact=("Rust Integration Tests (query)",)),
    Group(
        "rust-integration-test-pgwire-accuracy",
        exact=("Rust Integration Tests (pgwire accuracy)",),
    ),
    Group("rust-integration-test-graph", exact=("Rust Integration Tests (graph)",)),
    Group("rust-integration-test-cluster", exact=("Rust Integration Tests (cluster)",)),
    Group("rust-integration-test-api", exact=("Rust Integration Tests (api surface)",)),
    Group("object-store-emulator-tests", exact=("Object-store Cool tier (emulators)",)),
    Group(
        "object-store-restart-recovery",
        exact=("Object-store restart recovery (Azurite)",),
    ),
    Group("mlflow-conformance", exact=("MLflow conformance (real client)",)),
    Group("sift-recall-ratchet", exact=("SIFT PAX recall ratchet (N=10k)",)),
    Group("python-test", prefix=("Python SDK Tests (",)),
    Group("python-sdk-round-trip", exact=("Python SDK round-trip (ADR-068 D5)",)),
    Group("openapi-spec-drift", exact=("OpenAPI Spec From Code (TD-126)",)),
)


@dataclass(frozen=True)
class Context:
    """The run asking the question."""

    repo: str
    run_id: int
    event: str
    ref: str
    base_ref: str
    head_repo: str
    medium_tier: bool
    full_tier: bool
    tree: str


@dataclass
class Decision:
    reused: dict[str, int] = field(default_factory=dict)  # group key -> proving run id
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def output_value(self) -> str:
        if not self.reused:
            return ""
        return "," + ",".join(sorted(self.reused)) + ","


class Api:
    """The three GitHub reads the decision needs. Swapped for a fake in tests."""

    def __init__(self, repo: str, token: str, base_url: str = "https://api.github.com"):
        self.repo = repo
        self.token = token
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/repos/{self.repo}/{path}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
            return json.loads(resp.read().decode("utf-8"))

    def proof_artifacts(self, name: str) -> list[dict]:
        return list(
            self._get(f"actions/artifacts?name={name}&per_page=100").get(
                "artifacts", []
            )
        )

    def run(self, run_id: int) -> dict:
        return self._get(f"actions/runs/{run_id}")

    def _paged(self, path: str, key: str | None) -> list[dict]:
        """Every page of a list endpoint. Stops on a short page; never trusts a
        missing total_count to mean "done"; more than MAX_PAGES raises."""
        items: list[dict] = []
        sep = "&" if "?" in path else "?"
        for page in range(1, MAX_PAGES + 1):
            body = self._get(f"{path}{sep}per_page=100&page={page}")
            batch = body if key is None else body.get(key, [])
            if not isinstance(batch, list):
                raise ValueError(f"{path}: expected a list")
            items.extend(batch)
            if len(batch) < 100:
                return items
            if (
                key is not None
                and "total_count" in body
                and len(items) >= int(body["total_count"])
            ):
                return items
        raise RuntimeError(f"{path}: more than {MAX_PAGES} pages")

    def jobs(self, run_id: int) -> list[dict]:
        return self._paged(f"actions/runs/{run_id}/jobs?filter=latest", "jobs")

    def pulls_from(self, owner: str, branch: str) -> list[dict]:
        """Every PR (any state, any base) whose head is owner:branch."""
        return self._paged(f"pulls?state=all&head={owner}:{branch}", None)


def git_tree_of(sha: str) -> str | None:
    """Resolve a commit to its tree, fetching it once if absent. None on failure."""

    def resolve() -> str | None:
        out = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{tree}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        value = out.stdout.strip()
        return value if out.returncode == 0 and _TREE_RE.match(value) else None

    tree = resolve()
    if tree is None:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", sha],
            capture_output=True,
            check=False,
        )
        tree = resolve()
    return tree


def promotion_context(ctx: Context) -> str | None:
    """None if this run may reuse; otherwise why not."""
    if not _TREE_RE.match(ctx.tree):
        return f"tree hash {ctx.tree!r} is not a 40-hex git tree"
    if not ctx.medium_tier:
        return "not a promotion-tier run (medium_tier=false)"
    if ctx.event == "push":
        if ctx.ref not in tuple(f"refs/heads/{b}" for b in TRUSTED_PUSH_BRANCHES):
            return f"push to {ctx.ref} is not a promotion branch"
        return None
    if ctx.event == "pull_request":
        if ctx.base_ref not in PROMOTION_BASES:
            return f"PR base {ctx.base_ref!r} is not qa/main"
        if ctx.head_repo != ctx.repo:
            return f"PR head repository {ctx.head_repo!r} is a fork"
        return None
    return f"event {ctx.event!r} never reuses"


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def trusted_run(
    run: dict, ctx: Context, now: dt.datetime, max_age_hours: float
) -> str | None:
    """None if `run` may serve as a proof source; otherwise why not."""
    if int(run.get("id", -1)) == ctx.run_id:
        return "is the current run"
    if run.get("path") != WORKFLOW_PATH:
        return f"workflow {run.get('path')!r} is not {WORKFLOW_PATH}"
    repo = (run.get("repository") or {}).get("full_name")
    head_repo = (run.get("head_repository") or {}).get("full_name")
    if repo != ctx.repo or head_repo != ctx.repo:
        return f"head repository {head_repo!r} is not {ctx.repo} (fork)"
    event = run.get("event")
    branch = run.get("head_branch")
    if event == "push":
        if branch not in TRUSTED_PUSH_BRANCHES:
            return f"push to {branch!r} is not a promotion branch"
    elif event == "pull_request":
        pass  # head AND base are checked against the PR itself: pr_route_reason
    else:
        return f"event {event!r} is not a proof source"
    if run.get("status") != "completed":
        return f"status {run.get('status')!r} (not completed)"
    created = run.get("created_at")
    if not created:
        return "no created_at"
    age_h = (now - _parse_time(created)).total_seconds() / 3600.0
    if age_h > max_age_hours or age_h < -1.0:
        return f"age {age_h:.1f}h outside 0..{max_age_hours:.0f}h"
    return None


def pr_route_reason(run: dict, ctx: Context, api: Api) -> str | None:
    """None if a pull_request proof run is a promotion PR on an allowed route.

    `run.pull_requests` is EMPTY on real promotion runs, so the PR is resolved
    by listing every PR whose head is this repo's `head_branch` and keeping
    those whose head SHA is the run's head SHA. Exactly one may match, and it
    must be a same-repository PR on an exact route pair (develop->qa or
    qa->main). Any other PR from the same head commit -- e.g. develop into an
    unprotected base -- makes the run ambiguous, because that PR's run has the
    same head_branch and head_sha and is indistinguishable through the API.
    """
    head_branch = str(run.get("head_branch", ""))
    head_sha = str(run.get("head_sha", ""))
    want_base = PROMOTION_ROUTES.get(head_branch)
    if want_base is None or not head_sha:
        return f"PR head {head_branch!r} is not a promotion source"
    owner = ctx.repo.split("/", 1)[0]
    matches = [
        pr
        for pr in api.pulls_from(owner, head_branch)
        if (pr.get("head") or {}).get("sha") == head_sha
    ]
    if len(matches) != 1:
        bases = sorted(str((pr.get("base") or {}).get("ref")) for pr in matches)
        return f"head {head_sha[:10]} maps to {len(matches)} PR(s) {bases}; need exactly one"
    pr = matches[0]
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    if (head.get("repo") or {}).get("full_name") != ctx.repo:
        return "PR head repository is not this repository"
    if (base.get("repo") or {}).get("full_name") != ctx.repo:
        return "PR base repository is not this repository"
    if head.get("ref") != head_branch:
        return f"PR head ref {head.get('ref')!r} != run head {head_branch!r}"
    if base.get("ref") != want_base:
        return f"PR #{pr.get('number')} {head_branch}->{base.get('ref')} is not a promotion route"
    return None


def group_verdicts(jobs: Sequence[dict]) -> tuple[set[str], set[str], set[str]]:
    """(green groups, groups with a RED job, groups with any other non-success)."""
    green: set[str] = set()
    red: set[str] = set()
    other: set[str] = set()
    for group in GROUPS:
        mine = [j for j in jobs if group.matches(str(j.get("name", "")))]
        if not mine:
            continue
        conclusions = [
            (j.get("conclusion") if j.get("status") == "completed" else "incomplete")
            for j in mine
        ]
        if any(c in _RED for c in conclusions):
            red.add(group.key)
        elif all(c == "success" for c in conclusions):
            green.add(group.key)
        elif any(c not in ("success", "skipped") for c in conclusions):
            other.add(group.key)
    return green, red, other


def decide(
    ctx: Context,
    api: Api,
    tree_of: Callable[[str], str | None] = git_tree_of,
    now: dt.datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> Decision:
    """Pure decision over the API + git; any exception means reuse nothing."""
    try:
        return _decide(ctx, api, tree_of, now or dt.datetime.now(dt.UTC), max_age_hours)
    except Exception as exc:  # noqa: BLE001 - fail open on ANY error, by design
        return Decision(reason=f"fail-open: {type(exc).__name__}: {exc}")


def _decide(
    ctx: Context,
    api: Api,
    tree_of: Callable[[str], str | None],
    now: dt.datetime,
    max_age_hours: float,
) -> Decision:
    why_not = promotion_context(ctx)
    if why_not:
        return Decision(reason=why_not)

    decision = Decision()
    artifacts = api.proof_artifacts(PROOF_PREFIX + ctx.tree)
    # proven[group] -> newest proving run id
    proven: dict[str, tuple[str, int]] = {}
    seen: set[int] = set()
    for art in artifacts:
        if art.get("name") != PROOF_PREFIX + ctx.tree:
            continue
        wr = art.get("workflow_run") or {}
        run_id = int(wr.get("id", -1))
        if run_id < 0 or run_id in seen:
            continue
        seen.add(run_id)
        run = api.run(run_id)
        reason = trusted_run(run, ctx, now, max_age_hours)
        if reason:
            decision.notes.append(f"run {run_id} ignored: {reason}")
            continue
        if run.get("event") == "pull_request":
            route = pr_route_reason(run, ctx, api)
            if route:
                decision.notes.append(f"run {run_id} ignored: {route}")
                continue
        if run.get("event") == "push":
            head_tree = tree_of(str(run.get("head_sha", "")))
            if head_tree != ctx.tree:
                decision.notes.append(
                    f"run {run_id} ignored: head {run.get('head_sha')} has tree {head_tree}"
                )
                continue
        jobs = api.jobs(run_id)
        changes_ok = any(
            j.get("name") == CHANGES_JOB_NAME and j.get("conclusion") == "success"
            for j in jobs
        )
        green, red, other = group_verdicts(jobs)
        if red:
            return Decision(
                reason=(
                    f"veto: run {run_id} on the identical tree has failed reusable "
                    f"job group(s) {sorted(red)}; running everything"
                ),
                notes=decision.notes,
            )
        if not changes_ok:
            decision.notes.append(
                f"run {run_id} ignored: '{CHANGES_JOB_NAME}' did not succeed"
            )
            continue
        if other:
            decision.notes.append(
                f"run {run_id} ignored: reusable group(s) {sorted(other)} cancelled/incomplete"
            )
            continue
        created = str(run.get("created_at"))
        for key in green:
            if key not in proven or created > proven[key][0]:
                proven[key] = (created, run_id)

    keys = {key: run_id for key, (_, run_id) in proven.items()}
    by_key = {g.key: g for g in GROUPS}
    for key in list(keys):
        group = by_key[key]
        if group.not_at_full_tier and ctx.full_tier:
            decision.notes.append(
                f"{key} not reused: its artifact feeds full-tier extras"
            )
            del keys[key]
    for key in list(keys):
        group = by_key[key]
        missing = [d for d in group.dependents if d not in keys]
        if missing:
            decision.notes.append(f"{key} not reused: dependents {missing} must run")
            del keys[key]
    decision.reused = keys
    decision.reason = (
        f"reusing {len(keys)} group(s) proven on tree {ctx.tree}"
        if keys
        else "no trusted green proof for this tree"
    )
    return decision


def summary_markdown(ctx: Context, decision: Decision) -> str:
    lines = [
        "## CI tree-hash reuse (TD-CITIER-2)",
        "",
        f"Tree under test: `{ctx.tree}`",
        "",
    ]
    if decision.reused:
        for key, run_id in sorted(decision.reused.items()):
            url = f"https://github.com/{ctx.repo}/actions/runs/{run_id}"
            lines.append(
                f"- `{key}`: reused results from run [{run_id}]({url}) (tree `{ctx.tree}`)"
            )
    else:
        lines.append(f"No job reused -- everything runs. Reason: {decision.reason}")
    if decision.notes:
        lines += ["", "<details><summary>Candidate notes</summary>", ""]
        lines += [f"- {n}" for n in decision.notes]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _env_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tree", required=True)
    p.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    args = p.parse_args(list(argv) if argv is not None else None)

    env = os.environ
    decision: Decision
    ctx = Context(
        repo=env.get("GITHUB_REPOSITORY", ""),
        run_id=int(env.get("GITHUB_RUN_ID", "0") or 0),
        event=env.get("GITHUB_EVENT_NAME", ""),
        ref=env.get("GITHUB_REF", ""),
        base_ref=env.get("GITHUB_BASE_REF", ""),
        head_repo=env.get("PR_HEAD_REPO", ""),
        medium_tier=_env_bool(env.get("MEDIUM_TIER", "")),
        full_tier=_env_bool(env.get("FULL_TIER", "")),
        tree=args.tree.strip(),
    )
    token = env.get("GITHUB_TOKEN", "")
    if not token:
        decision = Decision(reason="fail-open: no GITHUB_TOKEN")
    else:
        decision = decide(ctx, Api(ctx.repo, token), max_age_hours=args.max_age_hours)

    out = f"reused={decision.output_value()}\n"
    sys.stdout.write(out)
    sys.stdout.write(f"# {decision.reason}\n")
    for note in decision.notes:
        sys.stdout.write(f"#   {note}\n")
    if env.get("GITHUB_OUTPUT"):
        with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(out)
    if env.get("GITHUB_STEP_SUMMARY"):
        with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(summary_markdown(ctx, decision))
    for key, run_id in sorted(decision.reused.items()):
        sys.stdout.write(
            f"::notice title=CI reuse::{key}: reused results from run {run_id} (tree {ctx.tree})\n"
        )
    return 0


def safe_main() -> int:
    """Entry point: even an argument/IO crash prints an empty `reused`."""
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - must never fail the changes job
        sys.stdout.write("reused=\n")
        sys.stdout.write(f"# fail-open: {type(exc).__name__}: {exc}\n")
        return 0


if __name__ == "__main__":
    sys.exit(safe_main())
