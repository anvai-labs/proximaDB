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
        self._edit(SMOKE, "needs: [resolve, provenance, docker, binary, python, npm, rust, homebrew]",
                   "needs: [resolve, provenance, docker, binary, python, npm, homebrew]")
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
