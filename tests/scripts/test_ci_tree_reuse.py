"""Tests for scripts/ci_tree_reuse.py (TD-CITIER-2).

The helper lets a promotion run skip expensive suites that a trusted run
already passed on the byte-identical git tree. Its failure modes are
asymmetric:

* a false POSITIVE (reuse when it should not) lets a release ship on gating
  that never ran against its code -- the defect this file exists to prevent;
* a false NEGATIVE only costs one redundant run (today's behavior).

So every guard clause gets a scenario where it is the ONLY thing standing
between the input and a reuse, and the mutation tests at the bottom delete
each guard from the source and assert that scenario then flips to "reuse".
The wiring tests pin ci.yml to the script: every reusable group is a real job
whose `if:` carries the guard, and nothing that `needs:` a reusable job can be
skipped by accident when that job is reused.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci_tree_reuse.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(source: str | None = None, name: str = "ci_tree_reuse") -> ModuleType:
    """Import the script (or a mutated copy of its source) as a module.

    Registered in sys.modules because @dataclass resolves annotations there."""
    mod = ModuleType(name)
    mod.__file__ = str(SCRIPT)
    sys.modules[name] = mod
    text = SCRIPT.read_text(encoding="utf-8") if source is None else source
    exec(compile(text, str(SCRIPT), "exec"), mod.__dict__)  # noqa: S102 - test-only loader
    return mod


reuse = _load()

REPO = "anvai-labs/proximaDB"
TREE = "318f14450b69e351269fd5833d96791cf6225f1f"
OTHER_TREE = "7e3bdaeb585d9d979199573d9d576068ba8efb19"
NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.UTC)
FRESH = "2026-10-15T05:04:50Z"

JOB_NAMES = [
    "Detect Changes",
    "Rust Clippy",
    "Rust Compile (lib)",
    "Rust Compile (test)",
    "Rust Tests (unit)",
    "Rust Tests (doc)",
    "Feature Matrix (cluster)",
    "Feature Matrix (embedded)",
    "Feature Matrix (enterprise-catalogs)",
    "Rust Compile (benches)",
    "Rust Build (dev)",
    "Rust Build (release)",
    "Rust Integration Tests (storage)",
    "Rust Integration Tests (authz e2e)",
    "Rust Integration Tests (query)",
    "Rust Integration Tests (pgwire accuracy)",
    "Rust Integration Tests (graph)",
    "Rust Integration Tests (cluster)",
    "Rust Integration Tests (api surface)",
    "Object-store Cool tier (emulators)",
    "Object-store restart recovery (Azurite)",
    "MLflow conformance (real client)",
    "SIFT PAX recall ratchet (N=10k)",
    "Python SDK Tests (3.11)",
    "Python SDK Tests (3.12)",
    "Python SDK round-trip (ADR-068 D5)",
    "OpenAPI Spec From Code (TD-126)",
    "Code Coverage",
    "Docker Build",
    "CI Success",
]
ALL_KEYS = {g.key for g in reuse.GROUPS}


def jobs(overrides: dict[str, str] | None = None) -> list[dict]:
    overrides = overrides or {}
    out = []
    for name in JOB_NAMES:
        conclusion = overrides.get(name, "success")
        status = "in_progress" if conclusion == "in_progress" else "completed"
        out.append(
            {
                "name": name,
                "status": status,
                "conclusion": None if status != "completed" else conclusion,
            }
        )
    return out


def run(run_id: int = 100, **kw) -> dict:
    base = {
        "id": run_id,
        "path": ".github/workflows/ci.yml",
        "repository": {"full_name": REPO},
        "head_repository": {"full_name": REPO},
        "event": "pull_request",
        "head_branch": "develop",
        "head_sha": "2166670a39eef8dbe9ccd67300750eaf2d3982e4",
        "status": "completed",
        "created_at": FRESH,
    }
    base.update(kw)
    return base


class FakeApi:
    def __init__(
        self,
        runs: list[dict],
        jobs_by_run: dict[int, list[dict]],
        tree: str = TREE,
        fail: str = "",
    ):
        self.runs = {r["id"]: r for r in runs}
        self.jobs_by_run = jobs_by_run
        self.tree = tree
        self.fail = fail

    def proof_artifacts(self, name: str) -> list[dict]:
        if self.fail == "artifacts":
            raise reuse.urllib.error.URLError("boom")
        return [
            {"name": reuse.PROOF_PREFIX + self.tree, "workflow_run": {"id": rid}}
            for rid in self.runs
        ]

    def run(self, run_id: int) -> dict:
        if self.fail == "run":
            raise reuse.urllib.error.HTTPError(
                "u", 403, "Resource not accessible", {}, None
            )
        return self.runs[run_id]

    def jobs(self, run_id: int) -> list[dict]:
        if self.fail == "jobs":
            raise ValueError("malformed JSON")
        return self.jobs_by_run[run_id]


def ctx(module: ModuleType = reuse, **kw):
    base = {
        "repo": REPO,
        "run_id": 999,
        "event": "pull_request",
        "ref": "refs/pull/2039/merge",
        "base_ref": "main",
        "head_repo": REPO,
        "medium_tier": True,
        "full_tier": True,
        "tree": TREE,
    }
    base.update(kw)
    return module.Context(**base)


PUSH_QA = {
    "event": "push",
    "ref": "refs/heads/qa",
    "base_ref": "",
    "head_repo": "",
    "full_tier": False,
}


def decide(api, c=None, module: ModuleType = reuse, tree_of=lambda sha: TREE, **kw):
    return module.decide(c or ctx(module), api, tree_of=tree_of, now=NOW, **kw)


def green_api(**run_kw) -> FakeApi:
    r = run(**run_kw)
    return FakeApi([r], {r["id"]: jobs()})


# ---------------------------------------------------------------------------
# Decision behavior
# ---------------------------------------------------------------------------


class Reuses(unittest.TestCase):
    def test_exact_tree_all_green_reuses_every_group_at_medium_tier(self):
        d = decide(green_api(), ctx(**PUSH_QA))
        self.assertEqual(set(d.reused), ALL_KEYS, d.reason)
        self.assertTrue(all(v == 100 for v in d.reused.values()))

    def test_full_tier_reuses_all_but_rust_build(self):
        d = decide(green_api())
        self.assertEqual(set(d.reused), ALL_KEYS - {"rust-build"})

    def test_output_value_is_comma_delimited_for_exact_contains(self):
        d = decide(green_api(), ctx(**PUSH_QA))
        value = d.output_value()
        self.assertTrue(value.startswith(",") and value.endswith(","))
        # `feature-matrix` must not be found inside `feature-matrix-enterprise-catalogs`
        # only: both are listed separately and delimited.
        self.assertIn(",feature-matrix,", value)
        self.assertIn(",feature-matrix-enterprise-catalogs,", value)
        self.assertEqual(reuse.Decision().output_value(), "")

    def test_newest_proving_run_is_named(self):
        old = run(100, created_at="2026-10-15T01:00:00Z")
        new = run(
            200, created_at="2026-10-15T06:00:00Z", event="push", head_branch="qa"
        )
        d = decide(FakeApi([old, new], {100: jobs(), 200: jobs()}))
        self.assertEqual(set(d.reused.values()), {200})

    def test_groups_combine_across_proving_runs(self):
        # push qa ran only openapi (rust filter false); PR develop->qa ran the rest.
        a = run(100)
        b = run(200, event="push", head_branch="qa", created_at="2026-10-15T07:15:10Z")
        only_openapi = [
            j
            if j["name"] in ("Detect Changes", "OpenAPI Spec From Code (TD-126)")
            else {**j, "conclusion": "skipped"}
            for j in jobs()
        ]
        d = decide(FakeApi([a, b], {100: jobs(), 200: only_openapi}))
        self.assertEqual(d.reused["openapi-spec-drift"], 200)
        self.assertEqual(d.reused["rust-test"], 100)

    def test_summary_names_run_and_tree(self):
        c = ctx(**PUSH_QA)
        d = decide(green_api(), c)
        md = reuse.summary_markdown(c, d)
        self.assertIn(
            f"reused results from run [100](https://github.com/{REPO}/actions/runs/100) (tree `{TREE}`)",
            md,
        )


class RunsEverything(unittest.TestCase):
    # -- identity ---------------------------------------------------------
    def test_tree_mismatch_artifact_is_ignored(self):
        api = green_api()
        api.tree = OTHER_TREE  # proof exists only for a different tree
        self.assertEqual(decide(api).reused, {})

    def test_push_proof_whose_head_has_another_tree_is_ignored(self):
        api = green_api(event="push", head_branch="qa")
        d = decide(api, tree_of=lambda sha: OTHER_TREE)
        self.assertEqual(d.reused, {})

    def test_push_proof_whose_head_cannot_be_resolved_is_ignored(self):
        api = green_api(event="push", head_branch="main")
        self.assertEqual(decide(api, tree_of=lambda sha: None).reused, {})

    def test_malformed_tree_never_reuses(self):
        self.assertEqual(decide(green_api(), ctx(tree="HEAD")).reused, {})

    # -- evidence ---------------------------------------------------------
    def test_failed_reusable_job_vetoes_everything(self):
        r = run(100)
        d = decide(FakeApi([r], {100: jobs({"Rust Tests (unit)": "failure"})}))
        self.assertEqual(d.reused, {})
        self.assertIn("veto", d.reason)

    def test_failure_in_any_trusted_run_vetoes_even_with_a_green_run(self):
        green = run(100)
        red = run(200, event="push", head_branch="qa")
        d = decide(
            FakeApi(
                [green, red],
                {100: jobs(), 200: jobs({"Feature Matrix (cluster)": "failure"})},
            )
        )
        self.assertEqual(d.reused, {})

    def test_timed_out_reusable_job_vetoes(self):
        r = run(100)
        d = decide(
            FakeApi([r], {100: jobs({"SIFT PAX recall ratchet (N=10k)": "timed_out"})})
        )
        self.assertEqual(d.reused, {})

    def test_cancelled_reusable_job_makes_run_ineligible(self):
        r = run(100)
        d = decide(
            FakeApi(
                [r],
                {100: jobs({"Object-store restart recovery (Azurite)": "cancelled"})},
            )
        )
        self.assertEqual(d.reused, {})

    def test_incomplete_reusable_job_makes_run_ineligible(self):
        r = run(100)
        d = decide(FakeApi([r], {100: jobs({"Rust Tests (doc)": "in_progress"})}))
        self.assertEqual(d.reused, {})

    def test_failed_non_reusable_job_does_not_veto(self):
        # Docker Build is a main-only extra, not a reusable group.
        r = run(100)
        d = decide(
            FakeApi([r], {100: jobs({"Docker Build": "failure"})}), ctx(**PUSH_QA)
        )
        self.assertEqual(set(d.reused), ALL_KEYS)

    def test_skipped_group_is_not_proven(self):
        r = run(100)
        d = decide(
            FakeApi(
                [r],
                {
                    100: jobs(
                        {"Rust Tests (unit)": "skipped", "Rust Tests (doc)": "skipped"}
                    )
                },
            )
        )
        self.assertNotIn("rust-test", d.reused)
        self.assertIn("feature-matrix", d.reused)

    def test_partially_skipped_group_is_not_proven(self):
        r = run(100)
        d = decide(FakeApi([r], {100: jobs({"Python SDK Tests (3.11)": "skipped"})}))
        self.assertNotIn("python-test", d.reused)

    def test_changes_job_not_green_makes_run_ineligible(self):
        r = run(100)
        d = decide(FakeApi([r], {100: jobs({"Detect Changes": "skipped"})}))
        self.assertEqual(d.reused, {})

    def test_proof_older_than_max_age_is_ignored(self):
        api = green_api(created_at="2026-10-01T00:00:00Z")
        self.assertEqual(decide(api).reused, {})

    def test_running_proof_is_ignored(self):
        self.assertEqual(decide(green_api(status="in_progress")).reused, {})

    # -- trust ------------------------------------------------------------
    def test_fork_proving_run_is_ignored(self):
        api = green_api(head_repository={"full_name": "mallory/proximaDB"})
        self.assertEqual(decide(api).reused, {})

    def test_feature_branch_pr_proof_is_ignored(self):
        # A PR from a feature branch into qa runs a workflow definition nobody reviewed.
        self.assertEqual(decide(green_api(head_branch="feat/x")).reused, {})

    def test_develop_push_proof_is_ignored(self):
        self.assertEqual(
            decide(green_api(event="push", head_branch="develop")).reused, {}
        )

    def test_other_workflow_is_ignored(self):
        self.assertEqual(
            decide(green_api(path=".github/workflows/qa-gate.yml")).reused, {}
        )

    def test_other_event_is_ignored(self):
        self.assertEqual(
            decide(green_api(event="workflow_dispatch", head_branch="qa")).reused, {}
        )

    def test_current_run_is_not_its_own_proof(self):
        self.assertEqual(decide(green_api(run_id=999)).reused, {})

    # -- context ----------------------------------------------------------
    def test_feat_to_develop_pr_never_reuses(self):
        c = ctx(base_ref="develop", medium_tier=False, full_tier=False)
        self.assertEqual(decide(green_api(), c).reused, {})

    def test_develop_push_never_reuses(self):
        c = ctx(
            event="push",
            ref="refs/heads/develop",
            base_ref="",
            medium_tier=False,
            full_tier=False,
        )
        self.assertEqual(decide(green_api(), c).reused, {})

    def test_push_to_qa_without_medium_tier_never_reuses(self):
        c = ctx(**{**PUSH_QA, "medium_tier": False})
        self.assertEqual(decide(green_api(), c).reused, {})

    def test_fork_pr_never_reuses(self):
        c = ctx(head_repo="mallory/proximaDB")
        self.assertEqual(decide(green_api(), c).reused, {})

    def test_pr_into_develop_with_medium_flag_still_never_reuses(self):
        c = ctx(base_ref="develop")
        self.assertEqual(decide(green_api(), c).reused, {})

    def test_workflow_dispatch_never_reuses(self):
        c = ctx(event="workflow_dispatch", ref="refs/heads/qa", base_ref="")
        self.assertEqual(decide(green_api(), c).reused, {})

    # -- fail-open --------------------------------------------------------
    def test_api_errors_fail_open(self):
        for where in ("artifacts", "run", "jobs"):
            with self.subTest(where=where):
                api = green_api()
                api.fail = where
                d = decide(api)
                self.assertEqual(d.reused, {})
                self.assertTrue(d.reason.startswith("fail-open"), d.reason)

    def test_main_without_token_fails_open_and_exits_zero(self):
        env = {
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/qa",
            "MEDIUM_TIER": "true",
        }
        buf = io.StringIO()
        with mock.patch.dict(reuse.os.environ, env, clear=True), redirect_stdout(buf):
            self.assertEqual(reuse.main(["--tree", TREE]), 0)
        self.assertIn("reused=\n", buf.getvalue())

    def test_safe_main_swallows_crashes(self):
        buf = io.StringIO()
        with (
            mock.patch.object(reuse, "main", side_effect=RuntimeError("x")),
            redirect_stdout(buf),
        ):
            self.assertEqual(reuse.safe_main(), 0)
        self.assertIn("reused=\n", buf.getvalue())

    # -- dependency safety --------------------------------------------------
    def test_rust_build_needs_every_dependent_reused(self):
        r = run(100)
        d = decide(
            FakeApi([r], {100: jobs({"Rust Integration Tests (graph)": "skipped"})}),
            ctx(**PUSH_QA),
        )
        self.assertNotIn("rust-build", d.reused)
        self.assertNotIn("rust-integration-test-graph", d.reused)
        self.assertIn("rust-integration-test-storage", d.reused)

    def test_rust_build_never_reused_at_full_tier(self):
        self.assertNotIn("rust-build", decide(green_api()).reused)


# ---------------------------------------------------------------------------
# Mutation tests: delete each guard; its scenario must flip to "reuse".
# ---------------------------------------------------------------------------

SOURCE = SCRIPT.read_text(encoding="utf-8")

# (label, exact source text, replacement, scenario -> (api, ctx_kwargs, tree_of))
MUTANTS = [
    (
        "context: medium_tier",
        "    if not ctx.medium_tier:\n",
        "    if False:\n",
        lambda: (green_api(), {"base_ref": "main", "medium_tier": False}, None),
    ),
    (
        "context: fork PR",
        "        if ctx.head_repo != ctx.repo:\n",
        "        if False:\n",
        lambda: (green_api(), {"head_repo": "mallory/proximaDB"}, None),
    ),
    (
        "context: PR base",
        "        if ctx.base_ref not in PROMOTION_BASES:\n",
        "        if False:\n",
        lambda: (green_api(), {"base_ref": "develop"}, None),
    ),
    (
        "context: push branch",
        '        if ctx.ref not in tuple(f"refs/heads/{b}" for b in TRUSTED_PUSH_BRANCHES):\n',
        "        if False:\n",
        lambda: (
            green_api(),
            {
                "event": "push",
                "ref": "refs/heads/develop",
                "base_ref": "",
                "full_tier": False,
            },
            None,
        ),
    ),
    (
        "context: other events",
        '    return f"event {ctx.event!r} never reuses"\n',
        "    return None\n",
        lambda: (
            green_api(),
            {"event": "workflow_dispatch", "ref": "refs/heads/qa", "base_ref": ""},
            None,
        ),
    ),
    (
        "context: tree format",
        "    if not _TREE_RE.match(ctx.tree):\n",
        "    if False:\n",
        lambda: (_api_for_tree("HEAD"), {"tree": "HEAD"}, None),
    ),
    (
        "identity: artifact name must equal the tree",
        '        if art.get("name") != PROOF_PREFIX + ctx.tree:\n            continue\n',
        "",
        lambda: (_api_for_tree(OTHER_TREE), {}, None),
    ),
    (
        "identity: push head tree re-verified",
        "            if head_tree != ctx.tree:\n",
        "            if False:\n",
        lambda: (green_api(event="push", head_branch="qa"), {}, lambda sha: OTHER_TREE),
    ),
    (
        "trust: current run",
        '    if int(run.get("id", -1)) == ctx.run_id:\n',
        "    if False:\n",
        lambda: (green_api(run_id=999), {}, None),
    ),
    (
        "trust: workflow path",
        '    if run.get("path") != WORKFLOW_PATH:\n',
        "    if False:\n",
        lambda: (green_api(path=".github/workflows/x.yml"), {}, None),
    ),
    (
        "trust: fork proving run",
        "    if repo != ctx.repo or head_repo != ctx.repo:\n",
        "    if False:\n",
        lambda: (
            green_api(head_repository={"full_name": "mallory/proximaDB"}),
            {},
            None,
        ),
    ),
    (
        "trust: PR head must be protected",
        "        if branch not in TRUSTED_PR_HEADS:\n",
        "        if False:\n",
        lambda: (green_api(head_branch="feat/x"), {}, None),
    ),
    (
        "trust: push branch must be protected",
        "        if branch not in TRUSTED_PUSH_BRANCHES:\n",
        "        if False:\n",
        lambda: (green_api(event="push", head_branch="develop"), {}, None),
    ),
    (
        "trust: other events",
        '        return f"event {event!r} is not a proof source"\n',
        "        pass\n",
        lambda: (green_api(event="schedule", head_branch="qa"), {}, None),
    ),
    (
        "trust: completed",
        '    if run.get("status") != "completed":\n',
        "    if False:\n",
        lambda: (green_api(status="in_progress"), {}, None),
    ),
    (
        "freshness: max age",
        "    if age_h > max_age_hours or age_h < -1.0:\n",
        "    if False:\n",
        lambda: (green_api(created_at="2026-10-01T00:00:00Z"), {}, None),
    ),
    (
        "evidence: red veto",
        "        if red:\n",
        "        if False:\n",
        lambda: (
            FakeApi(
                [run(100), run(200, event="push", head_branch="qa")],
                {100: jobs(), 200: jobs({"Rust Tests (unit)": "failure"})},
            ),
            {},
            None,
        ),
    ),
    (
        "evidence: changes job green",
        "        if not changes_ok:\n",
        "        if False:\n",
        lambda: (
            FakeApi([run(100)], {100: jobs({"Detect Changes": "skipped"})}),
            {},
            None,
        ),
    ),
    (
        "evidence: cancelled run ineligible",
        "        if other:\n",
        "        if False:\n",
        lambda: (
            FakeApi([run(100)], {100: jobs({"Rust Tests (unit)": "cancelled"})}),
            {},
            None,
        ),
    ),
    (
        "evidence: every job of a group must succeed",
        '        elif all(c == "success" for c in conclusions):\n',
        '        elif any(c == "success" for c in conclusions):\n',
        lambda: (
            FakeApi([run(100)], {100: jobs({"Python SDK Tests (3.11)": "skipped"})}),
            {},
            None,
        ),
    ),
    (
        "dependency: full tier",
        "        if group.not_at_full_tier and ctx.full_tier:\n",
        "        if False:\n",
        lambda: (green_api(), {}, None),
    ),
    (
        "dependency: dependents must be reused",
        "        if missing:\n",
        "        if False:\n",
        lambda: (
            FakeApi(
                [run(100)], {100: jobs({"Rust Integration Tests (graph)": "skipped"})}
            ),
            dict(PUSH_QA),
            None,
        ),
    ),
]


def _api_for_tree(tree: str) -> FakeApi:
    api = green_api()
    api.tree = tree
    return api


class Mutations(unittest.TestCase):
    def _reused_keys(self, module: ModuleType, scenario) -> set[str]:
        api, ckw, tree_of = scenario()
        c = ctx(module, **ckw)
        return set(
            module.decide(c, api, tree_of=tree_of or (lambda sha: TREE), now=NOW).reused
        )

    def test_each_guard_is_load_bearing(self):
        for i, (label, old, new, scenario) in enumerate(MUTANTS):
            with self.subTest(guard=label):
                self.assertEqual(
                    SOURCE.count(old), 1, f"mutation anchor drifted: {old!r}"
                )
                baseline = self._reused_keys(reuse, scenario)
                mutant = _load(
                    SOURCE.replace(old, new), name=f"ci_tree_reuse_mutant_{i}"
                )
                mutated = self._reused_keys(mutant, scenario)
                self.assertNotEqual(
                    baseline,
                    mutated,
                    f"deleting guard '{label}' changed nothing -- it is not tested",
                )
                self.assertLess(
                    len(baseline),
                    len(mutated),
                    f"guard '{label}' should be the one blocking reuse",
                )


# ---------------------------------------------------------------------------
# Wiring: ci.yml and the script must agree.
# ---------------------------------------------------------------------------

WF = WORKFLOW.read_text(encoding="utf-8")
_JOB_KEY = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$", re.MULTILINE)


def job_blocks() -> dict[str, str]:
    jobs_at = WF.index("\njobs:\n")
    keys = [(m.group(1), m.start()) for m in _JOB_KEY.finditer(WF, jobs_at)]
    out = {}
    for i, (key, start) in enumerate(keys):
        end = keys[i + 1][1] if i + 1 < len(keys) else len(WF)
        out[key] = WF[start:end]
    return out


BLOCKS = job_blocks()


def job_if(block: str) -> str:
    m = re.search(r"^    if: ?(.*)$", block, re.MULTILINE)
    assert m, block[:200]
    first = m.group(1).strip()
    if first in ("|", ">", ">-", "|-"):
        body = []
        for line in block[m.end() + 1 :].splitlines():
            if not line.startswith("      "):
                break
            body.append(line.strip())
        return " ".join(body)
    return first


def job_needs(block: str) -> list[str]:
    m = re.search(r"^    needs: *(.*)$", block, re.MULTILINE)
    if not m:
        return []
    v = m.group(1).strip()
    if v.startswith("["):
        return [x.strip() for x in v.strip("[]").split(",") if x.strip()]
    if v:
        return [v]
    return re.findall(r"^      - ([A-Za-z0-9_-]+)", block[m.end() :], re.MULTILINE)


def job_name(block: str) -> str:
    m = re.search(r"^    name: (.*)$", block, re.MULTILINE)
    assert m
    return m.group(1).strip()


class Wiring(unittest.TestCase):
    def test_every_group_is_a_job_carrying_the_guard(self):
        for g in reuse.GROUPS:
            with self.subTest(group=g.key):
                self.assertIn(g.key, BLOCKS, "group key must be the ci.yml job id")
                guard = f"!contains(needs.changes.outputs.reused, ',{g.key},')"
                cond = job_if(BLOCKS[g.key])
                self.assertIn(guard, cond)
                # The guard must be ANDed at top level, not inside an `||` arm.
                self.assertTrue(cond.rstrip(" }").endswith(guard), cond)

    def test_group_patterns_match_the_job_name(self):
        for g in reuse.GROUPS:
            with self.subTest(group=g.key):
                name = job_name(BLOCKS[g.key])
                if g.exact:
                    self.assertIn(name, g.exact)
                else:
                    self.assertTrue(any(name.startswith(p) for p in g.prefix), name)

    def test_no_other_job_name_is_captured_by_a_group(self):
        for key, block in BLOCKS.items():
            name = job_name(block)
            owners = [g.key for g in reuse.GROUPS if g.matches(name)]
            with self.subTest(job=key):
                self.assertIn(owners, ([], [key]))

    def test_main_only_extras_and_hard_gates_are_never_reusable(self):
        for key in (
            "rust-coverage",
            "docker-build",
            "integration-test",
            "benchmarks",
            "rust-compile",
            "changes",
            "ci-success",
            "promotion-freshness",
        ):
            self.assertNotIn(key, ALL_KEYS)

    def test_no_dependent_is_silently_skipped_by_reuse(self):
        """A job that `needs:` a reusable job is implicitly skipped when that job
        is skipped. Each one must be (a) a declared dependent that is reused
        together with it, (b) handle the skip explicitly via
        `needs.<dep>.result` + the reuse marker, or (c) only run in contexts
        where its dependency is never reused (rust-build at the FULL tier)."""
        by_key = {g.key: g for g in reuse.GROUPS}
        # job -> (dependency, a clause in its `if:` proving it only runs where
        # that dependency is never reused: the FULL tier, a develop push, or
        # workflow_dispatch -- none of which reuse rust-build).
        only_where_dep_never_reused = {
            "integration-test": (
                "rust-build",
                "needs.changes.outputs.extensive == 'true'",
            ),
            "benchmarks": ("rust-build", "github.ref == 'refs/heads/main'"),
            "tpc-perf-ledger": ("rust-build", "github.ref == 'refs/heads/develop'"),
        }
        for key, block in BLOCKS.items():
            for dep in job_needs(block):
                if dep not in by_key:
                    continue
                with self.subTest(job=key, needs=dep):
                    if key in by_key[dep].dependents:
                        continue
                    if key == "ci-success":
                        # always(): treats skipped as pass by design (asserted below).
                        self.assertEqual(job_if(block), "always()")
                        continue
                    if (
                        key in only_where_dep_never_reused
                        and only_where_dep_never_reused[key][0] == dep
                    ):
                        self.assertTrue(by_key[dep].not_at_full_tier)
                        self.assertIn(
                            only_where_dep_never_reused[key][1], job_if(block)
                        )
                        continue
                    cond = job_if(block)
                    self.assertIn(f"needs.{dep}.result == 'skipped'", cond)
                    self.assertIn(
                        f"contains(needs.changes.outputs.reused, ',{dep},')", cond
                    )
                    self.assertIn("!cancelled()", cond)

    def test_declared_dependents_really_need_the_group(self):
        for g in reuse.GROUPS:
            for dep in g.dependents:
                with self.subTest(group=g.key, dependent=dep):
                    self.assertIn(g.key, job_needs(BLOCKS[dep]))

    def test_changes_job_wiring(self):
        changes = BLOCKS["changes"]
        self.assertIn("reused: ${{ steps.reuse.outputs.reused }}", changes)
        self.assertRegex(
            changes, r"permissions:\n      contents: read\n      actions: read\n"
        )
        step = changes[changes.index("id: reuse") :]
        step = step[: step.index("\n      - name:")]
        self.assertIn("continue-on-error: true", step)
        self.assertIn(
            "github.event.pull_request.head.repo.full_name == github.repository", step
        )
        self.assertIn("refs/heads/qa", step)
        self.assertIn("github.base_ref == 'qa' || github.base_ref == 'main'", step)
        self.assertIn("ci-tree-proof-${{ steps.reuse.outputs.tree }}", changes)
        self.assertEqual(reuse.PROOF_PREFIX, "ci-tree-proof-")

    def test_ci_success_still_needs_changes_and_treats_failure_as_red(self):
        # #2043: a failed `changes` must still fail CI Success.
        block = BLOCKS["ci-success"]
        self.assertIn("      - changes\n", block)
        self.assertIn(
            "contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled')",
            block,
        )
        self.assertIn("if: always()", block)


if __name__ == "__main__":
    unittest.main()
