# SPDX-License-Identifier: Apache-2.0
"""Negative controls for scripts/check_release_smoke.py, plus the pure decision
functions of the release-evidence scripts in scripts/release_smoke/.

A guard that passes on both the broken and the fixed tree is worthless, so every
clause gets a mutation that must make it fail. The script tests pin the
verdicts on the facts measured for the real v0.4.1 release (2026-10-17): the
qa->main PR #2053 was merged over a red `CI Success`, the tap has no proximadb
formula, and the formula release.yml generates carries no sha256.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GUARD = "scripts/check_release_smoke.py"
SMOKE = ".github/workflows/release-smoke.yml"
COPY = (
    GUARD,
    SMOKE,
    ".github/workflows/release.yml",
    ".github/workflows/release-python.yml",
    ".github/workflows/publish-image.yml",
    "scripts/release_smoke/provenance.py",
    "scripts/release_smoke/summarize.py",
    "scripts/release_smoke/await_url.sh",
    "scripts/release_smoke/retry.sh",
    "scripts/release_smoke/wait_healthy.sh",
    "scripts/release_smoke/start_image_server.sh",
    "scripts/release_smoke/binary_config.sh",
)


def _tree() -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="relsmoke-"))
    for rel in COPY:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    return tmp


def _run(tree: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tree / GUARD)], capture_output=True, text=True, check=False
    )


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(pathlib.Path(rel).stem, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReleaseSmokeGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _tree()
        self.addCleanup(shutil.rmtree, self.tree, ignore_errors=True)

    def _edit(self, rel: str, old: str, new: str) -> None:
        p = self.tree / rel
        text = p.read_text(encoding="utf-8")
        self.assertIn(old, text, f"mutation anchor missing from {rel}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")

    def _fails_with(self, needle: str) -> None:
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stdout)

    def test_passes_on_the_real_tree(self) -> None:
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_copy_list_holds_every_smoke_helper(self) -> None:
        """A helper added later must be copied, or its fail-hard scan is vacuous."""
        helpers = {p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts/release_smoke").glob("*.sh")}
        self.assertLessEqual(helpers, set(COPY))

    def test_catches_a_missing_workflow(self) -> None:
        (self.tree / SMOKE).unlink()
        self._fails_with("release-smoke.yml: missing")

    def test_catches_or_true(self) -> None:
        self._edit(SMOKE, 'bash scripts/release_smoke/start_image_server.sh "$IMAGE_REPO" "$VERSION" "$SERVER_PORT"\n',
                   'bash scripts/release_smoke/start_image_server.sh "$IMAGE_REPO" "$VERSION" "$SERVER_PORT" || true\n')
        self._fails_with("must fail hard")

    def test_catches_or_colon(self) -> None:
        self._edit(SMOKE, 'sha256sum -c sums.one\n', 'sha256sum -c sums.one || :\n')
        self._fails_with("must fail hard")

    def test_catches_continue_on_error(self) -> None:
        self._edit(SMOKE, "    timeout-minutes: 15\n    permissions:",
                   "    timeout-minutes: 15\n    continue-on-error: true\n    permissions:")
        self._fails_with("must fail hard")

    # ---- review M2: every soft-fail the first guard missed --------------------
    def test_catches_or_exit_0(self) -> None:
        self._edit(SMOKE, 'sha256sum -c sums.one\n', 'sha256sum -c sums.one || exit 0\n')
        self._fails_with("must fail hard")

    def test_catches_or_echo(self) -> None:
        self._edit(SMOKE, 'sha256sum -c sums.one\n', 'sha256sum -c sums.one || echo "checksum mismatch"\n')
        self._fails_with("must fail hard")

    def test_catches_set_plus_e(self) -> None:
        self._edit(SMOKE, "          set -euo pipefail\n          mkdir -p evidence\n          IMAGE=",
                   "          set +e\n          mkdir -p evidence\n          IMAGE=")
        self._fails_with("must fail hard")

    def test_catches_exit_0_after_summarize(self) -> None:
        self._edit(SMOKE, '--sha "${{ github.sha }}"\n', '--sha "${{ github.sha }}"; exit 0\n')
        self._fails_with("must fail hard")

    def test_catches_or_exit_0_in_a_helper(self) -> None:
        self._edit("scripts/release_smoke/wait_healthy.sh", 'sleep 2\n', 'sleep 2 || exit 0\n')
        self._fails_with("smoke helpers must fail hard")

    def test_catches_if_false_on_the_esm_step(self) -> None:
        self._edit(SMOKE, "      - name: ESM entry -- import + health\n",
                   "      - name: ESM entry -- import + health\n        if: false\n")
        self._fails_with("has `if: false`")

    def test_catches_failure_only_on_a_smoke_step(self) -> None:
        self._edit(SMOKE, "      - name: ESM entry -- import + health\n",
                   "      - name: ESM entry -- import + health\n        if: failure()\n")
        self._fails_with("has `if: failure()`")

    def test_catches_if_false_on_the_npm_job(self) -> None:
        self._edit(SMOKE, "    name: npm client\n", "    name: npm client\n    if: false\n")
        self._fails_with("job `npm` has `if: false`")

    def test_catches_if_false_on_the_aggregate_step(self) -> None:
        self._edit(SMOKE, "      - name: Aggregate (fails unless every job succeeded)\n",
                   "      - name: Aggregate (fails unless every job succeeded)\n        if: false\n")
        self._fails_with("has `if: false`")

    def test_catches_a_changed_summary_condition(self) -> None:
        self._edit(SMOKE, "    if: always() && needs.resolve.result != 'skipped'\n",
                   "    if: needs.resolve.result == 'never'\n")
        self._fails_with("job `summary` has `if:")

    def test_catches_local_path_pip_install(self) -> None:
        self._edit(SMOKE, '"proximadb==${VERSION}" "proximadb_embedded==${VERSION}"',
                   './clients/python "proximadb_embedded==${VERSION}"')
        self._fails_with("argument `./clients/python` is not a registry spec")

    def test_catches_editable_pip_install(self) -> None:
        self._edit(SMOKE, '"proximadb==${VERSION}" "proximadb_embedded==${VERSION}"',
                   '-e clients/python "proximadb_embedded==${VERSION}"')
        self._fails_with("argument `-e` is not a registry spec")

    def test_catches_unpinned_pip_install(self) -> None:
        self._edit(SMOKE, '"proximadb==${VERSION}" "proximadb_embedded==${VERSION}"',
                   'proximadb "proximadb_embedded==${VERSION}"')
        self._fails_with("argument `proximadb` is not a registry spec")

    def test_catches_local_npm_install(self) -> None:
        self._edit(SMOKE, '--no-audit --no-fund --prefer-online "${PKG}@${VERSION}"',
                   '--no-audit --no-fund --prefer-online ./clients/nodejs-embedded')
        self._fails_with("argument `./clients/nodejs-embedded` is not a registry spec")

    def test_catches_if_false_on_the_release_yml_caller(self) -> None:
        text = (self.tree / ".github/workflows/release.yml").read_text(encoding="utf-8")
        start = text.index("  release-smoke:")
        line = next(l for l in text[start:].splitlines() if l.startswith("    if:"))
        self._edit(".github/workflows/release.yml", line + "\n", "    if: false\n")
        self._fails_with("must carry exactly `if:")

    def test_catches_caller_dropping_always(self) -> None:
        self._edit(".github/workflows/release.yml", "    if: always() && needs.create-release.result == 'success' && ",
                   "    if: needs.create-release.result == 'success' && ")
        self._fails_with("must carry exactly `if:")

    def test_catches_or_true_in_a_helper_script(self) -> None:
        self._edit("scripts/release_smoke/wait_healthy.sh", 'sleep 2\n', 'sleep 2 || true\n')
        self._fails_with("smoke helpers must fail hard")

    def test_a_commented_or_true_is_not_a_finding(self) -> None:
        self._edit(SMOKE, "jobs:\n", "jobs:\n  # never write `cmd || true` here\n")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_catches_an_unsmoked_npm(self) -> None:
        self._edit(SMOKE, "node npm_esm_smoke.mjs", "node npm_esm_smoke_removed.mjs")
        self._fails_with("no job smokes `npm`")

    def test_catches_an_unsmoked_embedded_wheel(self) -> None:
        self._edit(SMOKE, 'scripts/release_smoke/embedded_smoke.py', 'scripts/release_smoke/other.py')
        self._fails_with("no job smokes `pypi-embedded`")

    def test_catches_a_binary_smoke_without_checksum(self) -> None:
        self._edit(SMOKE, 'sha256sum -c "${ASSET}.sha256"', 'echo skip')
        self._edit(SMOKE, 'sha256sum -c sums.one', 'echo skip')
        self._fails_with("no job smokes `github-release-binary`")

    def test_catches_a_dropped_rpm_smoke(self) -> None:
        self._edit(SMOKE, 'dnf install -y -q "/rel/${RPM}"', 'rpm -qp "/rel/${RPM}"')
        self._fails_with("no job smokes `github-release-packages`")

    def test_catches_a_partial_checksum(self) -> None:
        self._edit(SMOKE, "          sha256sum -c SHA256SUMS.txt\n", "          sha256sum -c sums.one\n")
        self._fails_with("no job smokes `github-release-packages`")

    def test_catches_a_dropped_msi_smoke(self) -> None:
        self._edit(SMOKE, "$p = Start-Process msiexec.exe", "$p = Start-Process notepad.exe")
        self._fails_with("no job smokes `github-release-windows`")

    def test_catches_a_dropped_homebrew_smoke(self) -> None:
        self._edit(SMOKE, "python3 scripts/release_smoke/homebrew_check.py", "echo")
        self._fails_with("no job smokes `homebrew`")

    def test_catches_a_new_unsmoked_publish_channel(self) -> None:
        """A publish the guard cannot map is an unsmoked channel, not silence."""
        self._edit(".github/workflows/release.yml", "  cleanup:\n",
                   "  publish-helm:\n    timeout-minutes: 5\n    runs-on: ubuntu-latest\n"
                   "    steps:\n      - run: helm push chart.tgz oci://example\n\n  cleanup:\n")
        self._fails_with("job `publish-helm` publishes")

    def test_testpypi_advisory_is_not_a_channel(self) -> None:
        text = (self.tree / ".github/workflows/release-python.yml").read_text(encoding="utf-8")
        self.assertIn("repository-url: https://test.pypi.org/legacy/", text)
        self.assertEqual(_run(self.tree).returncode, 0)

    def test_catches_a_missing_provenance_job(self) -> None:
        self._edit(SMOKE, "python3 scripts/release_smoke/provenance.py", "python3 scripts/release_smoke/nothing.py")
        self._fails_with("nothing proves the release is QA-based")

    def test_catches_provenance_not_reusing_the_prerelease_gate(self) -> None:
        p = self.tree / "scripts/release_smoke/provenance.py"
        p.write_text(p.read_text(encoding="utf-8").replace("await_prerelease_ci.sh", "my_copy.sh"), encoding="utf-8")
        self._fails_with("must reuse scripts/await_prerelease_ci.sh")

    def test_catches_summary_not_needing_a_job(self) -> None:
        self._edit(SMOKE, "needs: [resolve, provenance, docker, binary, packages, windows, python, npm, rust, homebrew]",
                   "needs: [resolve, provenance, docker, binary, packages, windows, python, npm, homebrew]")
        self._fails_with("does not need ['rust']")

    def test_catches_summary_without_always(self) -> None:
        self._edit(SMOKE, "    if: always() && needs.resolve.result != 'skipped'\n", "")
        self._fails_with("must run `if: always()`")

    def test_catches_release_yml_not_calling_the_smoke(self) -> None:
        self._edit(".github/workflows/release.yml", "    uses: ./.github/workflows/release-smoke.yml\n",
                   "    runs-on: ubuntu-latest\n")
        self._fails_with("a red smoke would leave the release green")

    def test_catches_a_lost_dispatch_trigger(self) -> None:
        self._edit(SMOKE, "  workflow_dispatch:\n", "  workflow_dispatch_disabled:\n")
        self._fails_with("lost its `workflow_dispatch:` trigger")


class SmokeScriptsAreShipped(unittest.TestCase):
    """Every scripts/release_smoke/ path the workflow runs must exist and must
    not be git-ignored. The repo's blanket .gitignore drops `python_*.py`, and
    the first draft's `python_sdk_smoke.py` was silently left out of the commit
    that way -- the workflow would have failed on a missing file at release time.
    """

    def test_referenced_scripts_exist_and_are_not_ignored(self) -> None:
        import re as _re
        text = (ROOT / SMOKE).read_text(encoding="utf-8")
        refs = sorted(set(_re.findall(r"scripts/release_smoke/[A-Za-z0-9_./-]+\.(?:py|sh|cjs|mjs)", text)))
        self.assertGreater(len(refs), 5, refs)
        for ref in refs:
            self.assertTrue((ROOT / ref).is_file(), ref)
        if shutil.which("git") and (ROOT / ".git").exists():
            proc = subprocess.run(["git", "-C", str(ROOT), "check-ignore", *refs],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.stdout.strip(), "", f"git-ignored smoke scripts: {proc.stdout}")


class ProvenanceVerdicts(unittest.TestCase):
    """provenance.evaluate on facts shaped like the real API answers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load("scripts/release_smoke/provenance.py")

    def _facts(self, **over):
        ok_run = lambda rid: {  # noqa: E731
            "run_id": rid, "url": f"u/{rid}", "conclusion": "success",
            "ci_success": {"conclusion": "success", "url": "j"},
        }
        facts = {
            "tag": "v0.4.1", "sha": "c4043fb3", "tree": "2cff0b6d",
            "main_relation": {"status": "identical", "ahead_by": 0},
            "qa_pr": {"number": 2053, "merged_at": "t", "merged_by": "x", "head_sha": "9fc3ec20"},
            "qa_head_tree": "2cff0b6d",
            "prerelease_ci": {"gate_exit_code": 0, "run_id": "38934957068", "conclusion": "success", "url": "u"},
            "qa_ci": {"push": ok_run(38905730109), "pull_request": ok_run(38905751066)},
        }
        facts.update(over)
        return facts

    def _failed(self, facts) -> list[str]:
        return [c["name"] for c in self.mod.evaluate(facts) if not c["ok"]]

    def test_all_green_is_qa_based(self) -> None:
        self.assertEqual(self._failed(self._facts()), [])

    def test_v041_pr_merged_over_red_ci_success(self) -> None:
        """The measured v0.4.1 state: qa push CI green, the qa->main PR's CI red."""
        facts = self._facts()
        facts["qa_ci"]["pull_request"]["conclusion"] = "failure"
        facts["qa_ci"]["pull_request"]["ci_success"]["conclusion"] = "failure"
        self.assertEqual(self._failed(facts), ["qa_pr_ci"])

    def test_tag_not_a_qa_merge(self) -> None:
        failed = self._failed(self._facts(qa_pr=None, qa_head_tree=None,
                                          qa_ci={"push": None, "pull_request": None}))
        self.assertEqual(failed, ["qa_merge", "qa_tree_equal", "qa_push_ci", "qa_pr_ci"])

    def test_tree_differs_from_what_qa_tested(self) -> None:
        self.assertEqual(self._failed(self._facts(qa_head_tree="deadbeef")), ["qa_tree_equal"])

    def test_tag_not_on_main(self) -> None:
        self.assertEqual(self._failed(self._facts(main_relation={"status": "diverged", "ahead_by": 3})), ["on_main"])

    def test_prerelease_ci_red(self) -> None:
        facts = self._facts(prerelease_ci={"gate_exit_code": 1, "run_id": None, "gate_output": "concluded 'failure'"})
        self.assertEqual(self._failed(facts), ["prerelease_ci"])

    def test_ci_run_without_ci_success_job(self) -> None:
        facts = self._facts()
        facts["qa_ci"]["push"]["ci_success"] = None
        self.assertEqual(self._failed(facts), ["qa_push_ci"])


class CiRunSelection(unittest.TestCase):
    """The qa_pr_ci / qa_push_ci verdicts come from THIS repo's runs of THIS PR."""

    REPO = "anvai-labs/proximaDB"
    PR = {"number": 2053, "head_sha": "9fc3ec20", "created_at": "2026-10-16T13:09:00Z",
          "merged_at": "2026-10-16T16:49:34Z"}

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load("scripts/release_smoke/provenance.py")

    def _run(self, rid, event="pull_request", repo=REPO, created="2026-10-16T13:09:52Z",
             sha="9fc3ec20", prs=None, branch="qa"):
        return {"id": rid, "event": event, "head_branch": branch, "head_sha": sha,
                "head_repository": {"full_name": repo}, "created_at": created,
                "pull_requests": prs or []}

    def test_the_prs_own_red_run_wins_over_decoys(self) -> None:
        own_red = self._run(1, created="2026-10-16T13:09:52Z")
        decoys = [
            self._run(2, repo="mallory/proximaDB", created="2026-10-16T15:00:00Z"),   # fork named qa
            self._run(3, created="2026-10-16T17:30:00Z"),                            # after the merge
            self._run(4, created="2026-10-16T12:00:00Z"),                            # before the PR opened
            self._run(5, created="2026-10-16T15:00:00Z", prs=[{"number": 9999}]),   # another PR
            self._run(6, created="2026-10-16T15:00:00Z", sha="deadbeef"),            # other head
        ]
        picked = self.mod.pick_pr_run([own_red, *decoys], self.REPO, self.PR)
        self.assertEqual(picked["id"], 1)

    def test_no_own_run_means_no_verdict(self) -> None:
        decoy = self._run(2, repo="mallory/proximaDB", created="2026-10-16T15:00:00Z")
        self.assertIsNone(self.mod.pick_pr_run([decoy], self.REPO, self.PR))

    def test_rerun_within_the_pr_window_is_used(self) -> None:
        runs = [self._run(1), self._run(7, created="2026-10-16T16:00:00Z", prs=[{"number": 2053}])]
        self.assertEqual(self.mod.pick_pr_run(runs, self.REPO, self.PR)["id"], 7)

    def test_push_run_must_be_this_repo_and_sha(self) -> None:
        runs = [self._run(1, event="push"), self._run(2, event="push", repo="mallory/proximaDB",
                                                       created="2026-10-17T00:00:00Z")]
        self.assertEqual(self.mod.pick_push_run(runs, self.REPO, "9fc3ec20")["id"], 1)
        self.assertIsNone(self.mod.pick_push_run(runs, self.REPO, "other"))


class ProvenanceOverrides(unittest.TestCase):
    """ci/release-overrides.toml: strict by default, a documented exception only
    for a CI verdict, tied to the exact failed run, and never shown as a pass."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load("scripts/release_smoke/provenance.py")

    def _checks(self, pr_ok: bool = False, run_id: int = 38905751066):
        return [
            {"name": "qa_merge", "ok": True, "detail": "", },
            {"name": "qa_push_ci", "ok": True, "detail": "", "run_id": 38905730109},
            {"name": "qa_pr_ci", "ok": pr_ok, "detail": "", "run_id": run_id},
        ]

    def _entry(self, **over):
        e = {"version": "0.4.1", "check": "qa_pr_ci", "reason": "runner OOM", "approver": "owner",
             "evidence": "https://github.com/o/r/actions/runs/38905751066/job/1", "followup": "#2055",
             "recorded": "2026-10-17"}
        e.update(over)
        return e

    def _write(self, text: str) -> pathlib.Path:
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = d / "release-overrides.toml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_the_real_file_is_valid_and_covers_v041(self) -> None:
        entries = self.mod.load_overrides(ROOT / "ci/release-overrides.toml")
        v041 = [e for e in entries if e["version"] == "0.4.1"]
        self.assertEqual([e["check"] for e in v041], ["qa_pr_ci"])
        self.assertIn("38905751066", v041[0]["evidence"])
        checks = self._checks()
        self.assertEqual(self.mod.apply_overrides(checks, "0.4.1", entries, repo="anvai-labs/proximaDB",
                                                  merged_by="vjsingh1984"), [])
        self.assertEqual(self.mod.status(checks[2]), "overridden")
        self.assertFalse(checks[2]["ok"], "an override must not rewrite the failure into a pass")

    def test_no_entry_means_fail(self) -> None:
        checks = self._checks()
        self.assertEqual(self.mod.apply_overrides(checks, "0.4.1", []), [])
        self.assertEqual(self.mod.status(checks[2]), "fail")

    def test_other_version_does_not_apply(self) -> None:
        checks = self._checks()
        self.mod.apply_overrides(checks, "0.4.2", [self._entry()])
        self.assertEqual(self.mod.status(checks[2]), "fail")

    def test_evidence_must_name_the_failed_run(self) -> None:
        checks = self._checks(run_id=99999)
        problems = self.mod.apply_overrides(checks, "0.4.1", [self._entry()])
        self.assertEqual(self.mod.status(checks[2]), "fail")
        self.assertTrue(any("does not name the failed run" in p for p in problems), problems)

    def test_stale_override_on_a_passing_check_is_a_problem(self) -> None:
        problems = self.mod.apply_overrides(self._checks(pr_ok=True), "0.4.1", [self._entry()])
        self.assertTrue(any("stale" in p for p in problems), problems)

    def test_structural_checks_are_not_overridable(self) -> None:
        p = self._write('approvers=["a"]\n[[override]]\nversion="0.4.1"\ncheck="qa_tree_equal"\nreason="r"\napprover="a"\n'
                        'evidence="https://github.com/o/r/actions/runs/1"\nfollowup="f"\nrecorded="2026-10-17"\n')
        with self.assertRaisesRegex(self.mod.OverrideError, "not overridable"):
            self.mod.load_overrides(p)

    def test_missing_field_is_an_error_not_a_silent_skip(self) -> None:
        p = self._write('approvers=["a"]\n[[override]]\nversion="0.4.1"\ncheck="qa_pr_ci"\nreason="r"\n'
                        'evidence="https://github.com/o/r/actions/runs/1"\nfollowup="f"\nrecorded="2026-10-17"\n')
        with self.assertRaisesRegex(self.mod.OverrideError, "`approver` is required"):
            self.mod.load_overrides(p)

    def _valid_toml(self, approver: str = "a", approvers: str = '["a"]', evidence: str =
                    "https://github.com/o/r/actions/runs/1") -> str:
        return (f'approvers={approvers}\n[[override]]\nversion="0.4.1"\ncheck="qa_pr_ci"\nreason="r"\n'
                f'approver="{approver}"\nevidence="{evidence}"\nfollowup="f"\nrecorded="2026-10-17"\n')

    def test_approver_must_be_allowlisted(self) -> None:
        self.assertEqual(len(self.mod.load_overrides(self._write(self._valid_toml()))), 1)
        with self.assertRaisesRegex(self.mod.OverrideError, "not in `approvers`"):
            self.mod.load_overrides(self._write(self._valid_toml(approver="mallory")))
        with self.assertRaisesRegex(self.mod.OverrideError, "approvers"):
            self.mod.load_overrides(self._write(self._valid_toml(approvers="[]")))

    def test_evidence_must_be_a_run_url(self) -> None:
        with self.assertRaisesRegex(self.mod.OverrideError, "run URL"):
            self.mod.load_overrides(self._write(self._valid_toml(evidence="https://example.com/38905751066")))

    def test_run_id_is_matched_exactly_not_by_substring(self) -> None:
        # 3890575106 is a prefix of the failed run 38905751066.
        for ev in ("https://github.com/o/r/actions/runs/3890575106",
                   "https://github.com/o/r/actions/runs/389057510661",
                   "https://github.com/o/r/actions/runs/1/job/38905751066"):
            checks = self._checks()
            problems = self.mod.apply_overrides(checks, "0.4.1", [self._entry(evidence=ev)])
            self.assertEqual(self.mod.status(checks[2]), "fail", ev)
            self.assertTrue(problems, ev)

    def test_evidence_must_be_this_repos_run(self) -> None:
        checks = self._checks()
        self.mod.apply_overrides(checks, "0.4.1", [self._entry()], repo="anvai-labs/proximaDB")
        self.assertEqual(self.mod.status(checks[2]), "fail")

    def test_qa_pr_ci_approver_must_be_who_merged(self) -> None:
        checks = self._checks()
        problems = self.mod.apply_overrides(checks, "0.4.1", [self._entry()], merged_by="someone-else")
        self.assertEqual(self.mod.status(checks[2]), "fail")
        self.assertTrue(any("not who merged" in p for p in problems), problems)
        checks = self._checks()
        self.assertEqual(self.mod.apply_overrides(checks, "0.4.1", [self._entry()], merged_by="owner"), [])
        self.assertEqual(self.mod.status(checks[2]), "overridden")

    def test_summary_reports_overridden_not_passed(self) -> None:
        summ = _load("scripts/release_smoke/summarize.py")
        needs = {j: {"result": "success"} for j, _ in summ.ROWS}
        prov = {"overridden": ["qa_pr_ci"], "checks": [
            {"name": "qa_pr_ci", "ok": False, "overridden": {"approver": "owner", "evidence": "https://e/1"}}]}
        doc = summ.build("0.4.1", needs, {"provenance": [{"file": "provenance.json", "evidence": prov}]}, None, None)
        self.assertEqual(doc["verdict"], "OVERRIDDEN")
        self.assertFalse(doc["passed"])
        self.assertTrue(doc["accepted"])
        row = next(j for j in doc["jobs"] if j["job"] == "provenance")
        self.assertEqual(row["result"], "overridden")
        table = summ.render(doc)
        self.assertIn("OVERRIDDEN, not passed", table)
        self.assertIn("| **OVERRIDDEN** |", table)
        # An override never rescues a job that actually failed.
        needs["docker"] = {"result": "failure"}
        self.assertEqual(summ.build("0.4.1", needs, {}, None, None)["verdict"], "FAILED")


class HomebrewVerdicts(unittest.TestCase):
    REPO = "anvai-labs/proximaDB"
    ASSET = "proximadb-0.4.1-x86_64-unknown-linux-gnu.tar.gz"
    SHA = "13ddc6628408fe8a88157b12fe53af4275a4c697460490ebe45b9c62b4510bc6"

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load("scripts/release_smoke/homebrew_check.py")

    def _formula(self, sha: str | None, tag: str = "v0.4.1") -> str:
        url = f"https://github.com/{self.REPO}/releases/download/{tag}/{self.ASSET}"
        sha_line = f'\n    sha256 "{sha}"' if sha else "\n    # sha256 will be updated after release"
        return f'class Proximadb < Formula\n  on_linux do\n    url "{url}"{sha_line}\n  end\nend\n'

    def _failed(self, formula):
        checks = self.mod.evaluate(self.REPO, "0.4.1", formula, {self.ASSET: self.SHA}, {self.ASSET, "SHA256SUMS.txt"})
        return [c["name"] for c in checks if not c["ok"]]

    def test_missing_formula_fails(self) -> None:
        """v0.4.1: anvai-labs/homebrew-tap has no Formula/proximadb.rb."""
        self.assertEqual(self._failed(None), ["formula_exists"])

    def test_matching_formula_passes(self) -> None:
        self.assertEqual(self._failed(self._formula(self.SHA)), [])

    def test_formula_without_sha256_fails(self) -> None:
        """What release.yml's update-homebrew generates today."""
        self.assertEqual(self._failed(self._formula(None)), [f"sha256:{self.ASSET}"])

    def test_wrong_sha256_fails(self) -> None:
        self.assertEqual(self._failed(self._formula("0" * 64)), [f"sha256:{self.ASSET}"])

    def test_stale_version_fails(self) -> None:
        self.assertIn(f"url:{self.ASSET}", self._failed(self._formula(self.SHA, tag="v0.4.0")))

    def test_formula_rendered_by_release_yml_passes_the_smoke(self) -> None:
        """scripts/release_homebrew_formula.py (release.yml update-homebrew)
        produces exactly what the strict smoke accepts -- from the tap's current
        formula and from the fresh template."""
        render = _load("scripts/release_homebrew_formula.py")
        assets = {self.ASSET, "SHA256SUMS.txt", "proximadb-0.4.1-x86_64-pc-windows-msvc.zip"}
        asset, sha = render.select_asset("0.4.1", assets, {self.ASSET: self.SHA})
        self.assertEqual((asset, sha), (self.ASSET, self.SHA))
        # The tap's shape (top-level url/sha256), still at the previous release.
        old_base = render.TEMPLATE.format(
            repo=self.REPO, sha="1" * 64,
            url=f"https://github.com/{self.REPO}/releases/download/v0.4.0/proximadb-0.4.0-x86_64-unknown-linux-gnu.tar.gz")
        for base in (None, old_base):
            formula = render.rewrite(base, self.REPO, "0.4.1", asset, sha)
            self.assertEqual(self._failed(formula), [], formula)

    def test_renderer_refuses_unbuilt_or_extra_platforms_and_downgrades(self) -> None:
        render = _load("scripts/release_homebrew_formula.py")
        with self.assertRaises(render.RenderError):
            render.select_asset("0.4.1", {"SHA256SUMS.txt"}, {})
        with self.assertRaises(render.RenderError):  # a second unix archive needs per-OS blocks
            render.select_asset("0.4.1", {self.ASSET, "proximadb-0.4.1-aarch64-apple-darwin.tar.gz"},
                                {self.ASSET: self.SHA})
        with self.assertRaises(render.RenderError):
            render.select_asset("0.4.1", {self.ASSET}, {})
        current = render.rewrite(None, self.REPO, "0.4.1", self.ASSET, self.SHA)
        with self.assertRaises(render.RenderError):  # downgrade
            render.rewrite(current, self.REPO, "0.4.0", self.ASSET, self.SHA)

    def test_release_yml_no_longer_ships_an_unpinned_formula(self) -> None:
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertNotIn("sha256 will be updated after release", text)
        self.assertIn("scripts/release_homebrew_formula.py", text)


GH_STUB = r"""#!/usr/bin/env python3
import base64, json, os, sys
state_path = os.environ["GH_STUB_STATE"]
st = json.load(open(state_path))
args = sys.argv[1:]
open(os.environ["GH_STUB_LOG"], "a").write(" ".join(args) + "\n")
def save(): json.dump(st, open(state_path, "w"))
def fail(code): sys.stderr.write(f"gh: error (HTTP {code})\n"); sys.exit(1)
if args[0] == "api":
    path = next(a for a in args[1:] if a.startswith("repos/"))
    method = args[args.index("-X") + 1] if "-X" in args else "GET"
    if method != "GET":
        if "contents/" in path: st["branch_file"] = True
        save(); print("{}"); sys.exit(0)
    if path.endswith("/git/ref/heads/main"):
        shas = st["main_shas"]; sha = shas.pop(0) if len(shas) > 1 else shas[0]; save(); print(sha); sys.exit(0)
    if "/git/ref/heads/" in path:
        if st.get("branch_exists"): print("{}"); sys.exit(0)
        fail(404)
    if "/contents/" in path:
        ref = path.split("ref=", 1)[1]
        if ref.startswith("automation/"):
            if st.get("branch_file"): print("blobsha"); sys.exit(0)
            fail(404)
        main = st["main_formula"]
        if main in ("404", "500"): fail(int(main))
        print(base64.b64encode(main.encode()).decode()); sys.exit(0)
if args[:2] == ["pr", "list"]:
    print(st.get("pr", "")); sys.exit(0)
if args[:2] == ["pr", "create"]:
    st["pr"] = "17"; save(); sys.exit(0)
if args[:2] == ["pr", "view"]:
    print(st["ci"]); sys.exit(0)
if args[:2] == ["pr", "merge"]:
    r = st["merges"].pop(0); save(); sys.exit(0 if r else 1)
sys.exit(3)
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("base64"), "needs bash + base64")
class HomebrewPublish(unittest.TestCase):
    """scripts/release_homebrew_publish.sh against a stub `gh` (no network)."""

    def _run(self, state: dict, token: str = "t"):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "bin").mkdir()
        gh = tmp / "bin" / "gh"
        gh.write_text(GH_STUB)
        gh.chmod(0o755)
        (tmp / "state.json").write_text(json.dumps(state))
        (tmp / "new.rb").write_text("NEW FORMULA\n")
        env = dict(os.environ, PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", GH_TOKEN=token,
                   GH_STUB_STATE=str(tmp / "state.json"), GH_STUB_LOG=str(tmp / "log"),
                   HOMEBREW_POLL_SECONDS="0", HOMEBREW_CI_WAIT_MINUTES="1", GITHUB_STEP_SUMMARY=str(tmp / "sum"))
        proc = subprocess.run(["bash", str(ROOT / "scripts/release_homebrew_publish.sh"), "o/tap",
                               "Formula/proximadb.rb", "0.4.2", str(tmp / "new.rb"), "https://run"],
                              capture_output=True, text=True, env=env, cwd=tmp, timeout=60)
        log = (tmp / "log").read_text() if (tmp / "log").exists() else ""
        return proc, log

    def _state(self, **over):
        st = {"main_shas": ["m1"], "main_formula": "OLD FORMULA\n", "ci": "SUCCESS", "merges": [True]}
        st.update(over)
        return st

    def test_no_token_is_reported_not_published(self) -> None:
        proc, log = self._run(self._state(), token="")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("NOT updated", proc.stdout)
        self.assertEqual(log, "")

    def test_happy_path_opens_pr_waits_and_merges(self) -> None:
        proc, log = self._run(self._state())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("pr create", log)
        self.assertIn("pr merge", log)

    def test_main_already_serving_it_is_done(self) -> None:
        proc, log = self._run(self._state(main_formula="NEW FORMULA\n"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("pr create", log)

    def test_main_moved_during_ci_is_retried_from_new_main(self) -> None:
        proc, log = self._run(self._state(main_shas=["m1", "m2", "m2", "m2"], merges=[False, True]))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("main moved", proc.stdout)
        self.assertEqual(log.count("pr merge"), 2)

    def test_unmergeable_without_race_fails(self) -> None:
        proc, _ = self._run(self._state(merges=[False]))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("main did not move", proc.stdout)

    def test_red_tap_ci_fails(self) -> None:
        proc, log = self._run(self._state(ci="FAILURE"))
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("pr merge", log)

    def test_non_404_read_error_is_fatal_not_absent(self) -> None:
        proc, log = self._run(self._state(main_formula="500"))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertNotIn("contents/Formula/proximadb.rb -f", log.replace("-X PUT ", ""))
        self.assertNotIn("PUT", log)

    def test_absent_formula_404_is_created(self) -> None:
        proc, log = self._run(self._state(main_formula="404"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("-X PUT", log)


class SummaryVerdicts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load("scripts/release_smoke/summarize.py")

    def _needs(self, **over):
        needs = {j: {"result": "success"} for j, _ in self.mod.ROWS}
        needs.update({k: {"result": v} for k, v in over.items()})
        return needs

    def test_all_success_passes(self) -> None:
        self.assertTrue(self.mod.build("0.4.1", self._needs(), {}, None, None)["passed"])

    def test_skipped_job_is_not_a_pass(self) -> None:
        self.assertFalse(self.mod.build("0.4.1", self._needs(rust="skipped"), {}, None, None)["passed"])

    def test_unknown_job_still_counts(self) -> None:
        needs = self._needs()
        needs["brand-new"] = {"result": "failure"}
        doc = self.mod.build("0.4.1", needs, {}, None, None)
        self.assertFalse(doc["passed"])

    def test_cli_exit_code_and_evidence_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            (t / "ev").mkdir()
            (t / "ev" / "python-sdk-3.12.json").write_text('{"passed": true}')
            out = t / "release-evidence-v0.4.1.json"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/release_smoke/summarize.py"), "--version", "0.4.1",
                 "--needs-json", json.dumps(self._needs(docker="failure")), "--evidence-dir", str(t / "ev"),
                 "--out", str(out)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 1, proc.stdout)
            doc = json.loads(out.read_text())
            python = next(j for j in doc["jobs"] if j["job"] == "python")
            self.assertEqual(python["evidence"][0]["file"], "python-sdk-3.12.json")


@unittest.skipUnless(shutil.which("bash") and shutil.which("sed"), "needs bash + sed")
class BinaryConfig(unittest.TestCase):
    SHIPPED = ROOT / "config/config.toml"

    def _render(self, src: pathlib.Path) -> dict:
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts/release_smoke/binary_config.sh"), str(src), "/tmp/rs-data", "47680"],
            capture_output=True, text=True, check=True,
        )
        return tomllib.loads(proc.stdout)

    def test_shipped_config_is_relocated_and_reported(self) -> None:
        cfg = self._render(self.SHIPPED)
        self.assertEqual(cfg["api"]["unified_port"], 47680)
        self.assertEqual(cfg["api"]["rest_port"], 47680)
        self.assertEqual(cfg["api"]["pg_port"], 43433)
        self.assertTrue(cfg["server"]["data_dir"].startswith("/tmp/rs-data"))
        self.assertNotIn("/tmp/proximadb", json.dumps(cfg))

    def test_existing_keys_are_replaced_not_duplicated(self) -> None:
        text = self.SHIPPED.read_text(encoding="utf-8")
        text = text.replace("unified_mode = true", "unified_mode = true\nunified_port = 5678", 1)
        text = text.replace("arrow_flight_port = 5680", "arrow_flight_port = 5680\npg_port = 5433", 1)
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "c.toml"
            src.write_text(text, encoding="utf-8")
            cfg = self._render(src)  # tomllib rejects a duplicate key
        self.assertEqual(cfg["api"]["unified_port"], 47680)
        self.assertEqual(cfg["api"]["pg_port"], 43433)


if __name__ == "__main__":
    unittest.main()
