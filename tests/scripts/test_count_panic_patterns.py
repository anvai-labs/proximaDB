"""Black-box tests for the panic-policy scanner's scopes.

The scanner's workspace scope exists because the guard predates the workspace
decomposition: production code that moved into ``crates/`` was invisible to it
(found via PR #2238's review — four new panic sites merged green). These tests
pin the two properties that must hold:

* the ``legacy`` scope's behavior is unchanged (it still ignores ``crates/``);
* the ``workspace`` scope ratchets every crate and app, including a crate that
  lands *after* the baseline was generated (its absent entry means 0).
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "count_panic_patterns.sh"


def run_scanner(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "count_panic_patterns.sh"), *args],
        capture_output=True,
        text=True,
        cwd=repo,
    )


class FixtureRepo:
    """A minimal repo with one legacy module, one crate, and one app."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, root / "scripts" / "count_panic_patterns.sh")
        # A legacy-scope module with one site.
        legacy = root / "src" / "core"
        legacy.mkdir(parents=True)
        (legacy / "mod.rs").write_text(
            "pub fn a() -> u8 { let x: Option<u8> = None; x.unwrap() }\n"
        )
        # A crate with one site, plus a cfg(test) site that must NOT count.
        crate = root / "crates" / "storage" / "crate-one" / "src"
        crate.mkdir(parents=True)
        (crate.parent / "Cargo.toml").write_text("[package]\nname = \"crate-one\"\n")
        (crate / "lib.rs").write_text(
            "pub fn b() -> u8 { let x: Option<u8> = None; x.unwrap() }\n"
            "#[cfg(test)]\nmod t {\n    #[test]\n    fn f() { let x: Option<u8> = None; x.unwrap() }\n}\n"
        )
        # A separate test file: excluded by name in every scope.
        (crate / "lib_tests.rs").write_text(
            "pub fn c() -> u8 { let x: Option<u8> = None; x.unwrap() }\n"
        )
        # A src/ file OUTSIDE the seven legacy modules: legacy totals were
        # always src-wide, so its site must count (round-1 review caught the
        # membership filter silently dropping such files).
        outside = root / "src" / "database.rs"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("pub fn h() -> u8 { let x: Option<u8> = None; x.unwrap() }\n")

        # An app with one expect.
        app = root / "apps" / "app-one" / "src"
        app.mkdir(parents=True)
        (app.parent / "Cargo.toml").write_text("[package]\nname = \"app-one\"\n")
        (app / "main.rs").write_text(
            'pub fn d() { let x: Option<u8> = None; x.expect("boom"); }\n'
        )

    def baseline(self, scope: str, path: str) -> None:
        out = self.root / path
        proc = run_scanner(self.root, "--scope", scope, "--mode", "report", "--format", "json")
        payload = json.loads(proc.stdout)
        trimmed = {
            "generated_at_utc": payload["generated_at_utc"],
            "scope": scope,
            "totals": payload["totals"],
            "modules": payload["modules"],
        }
        out.write_text(json.dumps(trimmed, indent=2, sort_keys=True) + "\n")


class PanicScopeTests(unittest.TestCase):
    def ratchet(self, *scope_args: str) -> subprocess.CompletedProcess:
        """no-regression against the fixture-generated baseline."""
        return run_scanner(
            self.repo, *scope_args, "--mode", "no-regression", "--baseline", "baseline.json"
        )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = FixtureRepo(Path(self._tmp.name) / "repo")
        self.repo = self.fixture.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legacy_scope_ignores_crates_and_apps(self):
        proc = run_scanner(self.repo, "--scope", "legacy", "--mode", "report", "--format", "json")
        payload = json.loads(proc.stdout)
        # src/core (1) + src/database.rs (1): src-wide totals, and crates/
        # and apps/ are invisible to the legacy scope.
        self.assertEqual(payload["totals"]["total"], 2)
        self.assertNotIn("crate-one", payload["modules"])

    def test_workspace_scope_counts_every_crate_and_app(self):
        proc = run_scanner(self.repo, "--scope", "workspace", "--mode", "report", "--format", "json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["modules"]["crate-one"]["total"], 1)  # cfg(test) + _tests.rs excluded
        self.assertEqual(payload["modules"]["app-one"]["total"], 1)
        self.assertEqual(payload["totals"]["total"], 2)
        self.assertNotIn("core", payload["modules"])  # src/ is NOT workspace scope

    def test_workspace_ratchet_fails_on_a_new_site(self):
        self.fixture.baseline("workspace", "baseline.json")
        crate_file = self.repo / "crates" / "storage" / "crate-one" / "src" / "lib.rs"
        crate_file.write_text(crate_file.read_text() + "pub fn e() { None::<u8>.unwrap(); }\n")
        proc = self.ratchet("--scope", "workspace")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("increased by 1", proc.stdout)

    def test_workspace_ratchet_fails_on_a_new_crate_with_a_site(self):
        """A crate absent from the baseline counts against an implicit 0."""
        self.fixture.baseline("workspace", "baseline.json")
        new_crate = self.repo / "crates" / "horizontal" / "crate-two" / "src"
        new_crate.mkdir(parents=True)
        (new_crate.parent / "Cargo.toml").write_text("[package]\nname = \"crate-two\"\n")
        (new_crate / "lib.rs").write_text("pub fn f() { None::<u8>.unwrap(); }\n")
        proc = self.ratchet("--scope", "workspace")
        self.assertNotEqual(proc.returncode, 0)

    def test_legacy_ratchet_still_works_against_its_own_baseline(self):
        self.fixture.baseline("legacy", "legacy.json")
        proc = run_scanner(
            self.repo, "--scope", "legacy", "--mode", "no-regression", "--baseline", "legacy.json"
        )
        self.assertEqual(proc.returncode, 0)
        # A new site in crates/ does not fail the LEGACY ratchet — that
        # blindness is exactly what the workspace scope exists to close.
        crate_file = self.repo / "crates" / "storage" / "crate-one" / "src" / "lib.rs"
        crate_file.write_text(crate_file.read_text() + "pub fn g() { None::<u8>.unwrap(); }\n")
        proc = run_scanner(
            self.repo, "--scope", "legacy", "--mode", "no-regression", "--baseline", "legacy.json"
        )
        self.assertEqual(proc.returncode, 0)

    def test_legacy_totals_are_src_wide(self):
        """Legacy totals must include src files outside the seven modules."""
        proc = run_scanner(self.repo, "--scope", "legacy", "--mode", "report", "--format", "json")
        payload = json.loads(proc.stdout)
        # src/core (1) + src/database.rs (1), attributed to no module.
        self.assertEqual(payload["totals"]["total"], 2)

    def test_new_module_is_a_regression_even_when_the_sum_is_compensated(self):
        """A new crate's site must fail even if another crate got cleaner."""
        self.fixture.baseline("workspace", "baseline.json")
        crate_file = self.repo / "crates" / "storage" / "crate-one" / "src" / "lib.rs"
        # Remove crate-one's only site (sum stays 2 -> 2)...
        crate_file.write_text("pub fn b() -> u8 { 0 }\n")
        # ...and add a NEW crate with one site.
        new_crate = self.repo / "crates" / "horizontal" / "crate-two" / "src"
        new_crate.mkdir(parents=True)
        (new_crate.parent / "Cargo.toml").write_text("[package]\nname = \"crate-two\"\n")
        (new_crate / "lib.rs").write_text("pub fn f() { None::<u8>.unwrap(); }\n")
        proc = self.ratchet("--scope", "workspace")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("absent from the baseline", proc.stdout)

    def test_invalid_scope_is_rejected(self):
        proc = run_scanner(self.repo, "--scope", "everything", "--mode", "report")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --scope", proc.stderr)


if __name__ == "__main__":
    unittest.main()
