# SPDX-License-Identifier: Apache-2.0
"""Negative controls for scripts/check_release_build_features.py.

A guard that passes on both the broken and the fixed tree is worthless, so each
defect class it catches gets a mutation here. Measured against the real pre-fix
tree (`ea710ae05`, v0.4.0 prep): **25** findings -- 4 feature-less release
targets + 21 missing job timeouts (14 release.yml + 7 prerelease-ci.yml) + **0**
image findings, because `deploy/docker/Dockerfile` already built with
`cloud-full` (it landed in `ed91af6c4`). An earlier draft of this docstring said
26 with 1 image finding; that counted the repo-root Dockerfile, which is the demo
image and is deliberately not guarded.

Those 25 were measured with the timeout check scoped to the two release
workflows. The scope later widened to the pipeline workflows, which on
`6b25150ff` held **41** further unbounded jobs (39 in ci.yml, 2 in qa-gate.yml),
all bounded in the same change. The two counts are against different trees and
are deliberately not added together.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GUARD = "scripts/check_release_build_features.py"
COPY = (
    GUARD,
    ".github/workflows/release.yml",
    ".github/workflows/prerelease-ci.yml",
    # The PUBLISHED image (publish-image.yml `file:`), not the repo-root
    # Dockerfile -- that one is the demo-compose image and is not guarded.
    "deploy/docker/Dockerfile",
    # In TIMED_WORKFLOWS. These MUST be copied, though not for the reason an
    # earlier comment here claimed: the mutating tests read the tree copy
    # BEFORE invoking the guard, so dropping these yields a loud
    # `FileNotFoundError`, not a vacuous pass. The real hazard is the guard's
    # own `if not path.exists(): continue` -- a non-mutating assertion against
    # an absent workflow would pass for the wrong reason.
    ".github/workflows/ci.yml",
    ".github/workflows/qa-gate.yml",
    ".github/workflows/publish-image.yml",
    # The rest of TIMED_WORKFLOWS. Adding a workflow to that tuple WITHOUT
    # adding it here is the mistake this file has already made once: the guard
    # skips a file it cannot find (`if not path.exists(): continue`), so the new
    # scope asserts nothing and the suite stays green. Kept in lockstep by
    # `test_the_copy_list_covers_every_timed_workflow` below, which reads the
    # tuple out of the guard rather than restating it.
    ".github/workflows/_platform-packages.yml",
    ".github/workflows/_shared-build.yml",
    ".github/workflows/branch-route-guard.yml",
    ".github/workflows/capability-contract-tests.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/docs-site.yml",
    ".github/workflows/flake-rerun.yml",
    ".github/workflows/infrastructure-deploy.yml",
    ".github/workflows/layering-check.yml",
    ".github/workflows/mirror-ci-images.yml",
    ".github/workflows/objstore-nightly.yml",
    ".github/workflows/prerelease-ci-v0.2.1-macos.yml",
    ".github/workflows/public-overflow-runner-smoke.yml",
    ".github/workflows/release-python.yml",
    ".github/workflows/tdd.yml",
    ".github/workflows/tier-config-fingerprint.yml",
    ".github/workflows/windows-compile-ratchet.yml",
)


def _tree() -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="relfeat-"))
    for rel in COPY:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    return tmp


def _run(tree: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tree / GUARD)],
        capture_output=True,
        text=True,
        check=False,
    )


class ReleaseBuildFeaturesGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _tree()
        self.addCleanup(shutil.rmtree, self.tree, ignore_errors=True)

    def test_passes_on_the_real_tree(self) -> None:
        """Positive control: the shipped configuration satisfies the guard."""
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_catches_a_feature_less_release_target(self) -> None:
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        self.assertIn('features: "cloud-full"', text)
        p.write_text(text.replace('features: "cloud-full"', 'features: ""', 1), encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must include `cloud-full`", result.stdout)

    def test_catches_a_partial_feature_list(self) -> None:
        """`aws` alone is not `cloud-full`: Azure and GCS would still be absent."""
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        p.write_text(text.replace('features: "cloud-full"', 'features: "aws"', 1), encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must include `cloud-full`", result.stdout)

    def test_catches_a_missing_timeout_on_the_last_job(self) -> None:
        """The final job in the file is the one a naive scan misses."""
        p = self.tree / ".github/workflows/release.yml"
        lines = p.read_text(encoding="utf-8").splitlines()
        last = max(
            i for i, l in enumerate(lines) if l.strip().startswith("timeout-minutes:")
        )
        del lines[last]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("has no `timeout-minutes`", result.stdout)

    def test_catches_a_missing_timeout_in_the_pipeline_workflow(self) -> None:
        """The bound is not release-only: ci.yml jobs are in scope too.

        `Docker Build` capped at 45 min let a 43-min cold build cancel
        `Test Container Health`. An unbounded job is the same defect against
        GitHub's 6h default, so ci.yml and qa-gate.yml are guarded as well.
        """
        for rel in (".github/workflows/ci.yml", ".github/workflows/qa-gate.yml"):
            with self.subTest(workflow=rel):
                self.setUp()
                p = self.tree / rel
                lines = p.read_text(encoding="utf-8").splitlines()
                last = max(
                    i
                    for i, l in enumerate(lines)
                    if l.strip().startswith("timeout-minutes:")
                )
                del lines[last]
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1)
                self.assertIn("has no `timeout-minutes`", result.stdout)
                self.assertIn(rel, result.stdout)

    def test_catches_a_missing_timeout_in_every_newly_scoped_workflow(self) -> None:
        """Every workflow added to TIMED_WORKFLOWS must really be asserted on.

        Not a restatement of the ci.yml control above. The widening left 31 jobs
        across 10 files unbounded, and the way a widening goes wrong here is
        silent: the guard skips a workflow it cannot find in the tree, so naming
        a file in TIMED_WORKFLOWS without copying it into the fixture makes the
        new assertions vacuous. This deletes one cap per newly scoped workflow
        and requires the guard to exit 1 AND to name that file.
        """
        newly_scoped = [
            ".github/workflows/_platform-packages.yml",
            ".github/workflows/_shared-build.yml",
            ".github/workflows/branch-route-guard.yml",
            ".github/workflows/capability-contract-tests.yml",
            ".github/workflows/codeql.yml",
            ".github/workflows/docs-site.yml",
            ".github/workflows/flake-rerun.yml",
            ".github/workflows/infrastructure-deploy.yml",
            ".github/workflows/layering-check.yml",
            ".github/workflows/mirror-ci-images.yml",
            ".github/workflows/objstore-nightly.yml",
            ".github/workflows/prerelease-ci-v0.2.1-macos.yml",
            ".github/workflows/public-overflow-runner-smoke.yml",
            ".github/workflows/release-python.yml",
            ".github/workflows/tdd.yml",
            ".github/workflows/tier-config-fingerprint.yml",
            ".github/workflows/windows-compile-ratchet.yml",
        ]
        for rel in newly_scoped:
            with self.subTest(workflow=rel):
                self.setUp()
                p = self.tree / rel
                lines = p.read_text(encoding="utf-8").splitlines()
                caps = [
                    i
                    for i, l in enumerate(lines)
                    if l.startswith("    timeout-minutes:")
                ]
                self.assertTrue(caps, f"{rel} has no job-level cap to delete")
                del lines[caps[-1]]
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("has no `timeout-minutes`", result.stdout)
                self.assertIn(rel, result.stdout)

    def test_the_copy_list_covers_every_timed_workflow(self) -> None:
        """COPY must not drift behind TIMED_WORKFLOWS.

        Read out of the guard, not restated, because restating it is how the two
        lists diverge. A workflow in the tuple but absent from the fixture is
        skipped by the guard, which is the vacuous-green failure mode.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("_guard", ROOT / GUARD)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(
            sorted(set(mod.TIMED_WORKFLOWS) - set(COPY)),
            [],
            "TIMED_WORKFLOWS names a workflow the fixture does not copy; the "
            "guard skips a missing file, so those assertions would be vacuous",
        )

    def test_an_unlisted_workflow_with_jobs_is_a_finding(self) -> None:
        """A NEW workflow must not escape the guard by not being listed.

        The scope was opt-in before: bounding the release and pipeline tiers
        left every other file out of the tuple and therefore unchecked. A
        workflow added tomorrow would be unchecked the same way, so the guard
        now reads the directory and fails on anything with a `jobs:` block that
        the tuple does not name.
        """
        p = self.tree / ".github/workflows/zz-newcomer.yml"
        p.write_text(
            "name: Newcomer\non:\n  push:\njobs:\n"
            "  newcomer:\n    runs-on: ubuntu-latest\n"
            "    timeout-minutes: 15\n    steps:\n      - run: true\n",
            encoding="utf-8",
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("not in TIMED_WORKFLOWS", result.stdout)
        self.assertIn("zz-newcomer.yml", result.stdout)

    def test_a_jobless_workflow_is_not_a_finding(self) -> None:
        """The completeness check must not fire on a file with no jobs.

        Otherwise the only way to keep it green is to list every stray YAML in
        the directory, and a guard that cries wolf gets the listing padded until
        it means nothing.
        """
        p = self.tree / ".github/workflows/zz-jobless.yml"
        p.write_text("name: Jobless\non:\n  workflow_dispatch:\n", encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_job_key_with_a_trailing_comment_cannot_evade(self) -> None:
        """The confirmed evasion: a comment on the job KEY hid a missing cap.

        `JOB_RE` ended in `\\s*$`, so `rust-clippy:  # the lint gate` did not
        match and the entire job body was absorbed into the previous job --
        whose own cap then satisfied the check for both. The guard exited 0
        with `rust-clippy` carrying no `timeout-minutes` at all. GitHub accepts
        the comment: it is valid YAML and the job runs.
        """
        p = self.tree / ".github/workflows/ci.yml"
        lines = p.read_text(encoding="utf-8").splitlines()
        out, killed = [], False
        for line in lines:
            if line == "  rust-clippy:":
                out.append("  rust-clippy:  # the lint gate")
                continue
            if not killed and line.strip().startswith("timeout-minutes:") and out and out[-1].startswith("    runs-on:"):
                if any(l == "  rust-clippy:  # the lint gate" for l in out):
                    killed = True
                    continue
            out.append(line)
        self.assertTrue(killed, "mutation did not remove rust-clippy's cap")
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("rust-clippy", result.stdout)

    def test_an_expression_valued_timeout_is_not_a_missing_one(self) -> None:
        """`${{ }}` is legal on jobs.<id>.timeout-minutes; rejecting it is a
        false positive on a workflow GitHub would happily run."""
        p = self.tree / ".github/workflows/ci.yml"
        text = p.read_text(encoding="utf-8")
        # Keyed on a REGEX, not a comment literal. An earlier revision searched
        # for "    timeout-minutes: 30  # measured max 13 min"; a later pass in
        # the same PR relabelled every such comment, the replace became a no-op,
        # and both subTests silently re-asserted the baseline. Verified then by
        # narrowing TIMEOUT_VALUE_RE and seeing 0 failures.
        cap_re = re.compile(r"^    timeout-minutes: \d+.*$", re.MULTILINE)
        self.assertIsNotNone(cap_re.search(text), "no job-level cap found to mutate")
        for spelling in (
            "    timeout-minutes: ${{ fromJSON(vars.CLIPPY_CAP) }}",
            "    timeout-minutes: 30.5",
        ):
            with self.subTest(spelling=spelling):
                mutated, n = cap_re.subn(spelling, text, count=1)
                self.assertEqual(n, 1, "mutation did not apply")
                self.assertNotEqual(mutated, text, "mutation was a no-op")
                p.write_text(mutated, encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_quoted_or_underscored_job_key_cannot_evade(self) -> None:
        """Same class as the trailing-comment evasion, two more legal spellings.

        A quoted key is valid YAML and GitHub reads the job id as `rust-clippy`;
        a leading underscore is explicitly allowed in a job id. Either one, left
        unmatched, folds the job body into the previous job.
        """
        for key in ('  "rust-clippy":', "  _rustclippy:"):
            with self.subTest(key=key):
                self.setUp()
                p = self.tree / ".github/workflows/ci.yml"
                lines = p.read_text(encoding="utf-8").splitlines()
                out, seen, killed = [], False, False
                for line in lines:
                    if line == "  rust-clippy:":
                        out.append(key)
                        seen = True
                        continue
                    if seen and not killed and line.startswith("    timeout-minutes:"):
                        killed = True
                        continue
                    out.append(line)
                self.assertTrue(killed, "mutation did not remove the cap")
                p.write_text("\n".join(out) + "\n", encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1, result.stdout)

    def test_a_reusable_workflow_job_is_exempt(self) -> None:
        """GitHub REJECTS job-level `timeout-minutes` on a `uses:` job, so
        demanding one there would make the guard unsatisfiable."""
        p = self.tree / ".github/workflows/ci.yml"
        text = p.read_text(encoding="utf-8")
        p.write_text(
            text.replace(
                "jobs:",
                "jobs:\n"
                "  called-elsewhere:\n"
                "    uses: ./.github/workflows/qa-gate.yml\n",
                1,
            ),
            encoding="utf-8",
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("called-elsewhere", result.stdout)

    def test_catches_a_runtime_config_copied_without_chown(self) -> None:
        """The defect that shipped: an unreachable published image.

        `chown -R ... /config` runs BEFORE the config COPY, so without
        `--chown` the file is root-owned while the server runs as uid 999.
        entrypoint.sh then cannot rewrite bind_address, the server binds
        127.0.0.1, and `-p` cannot reach it. Reproduced in a container: the
        same minimal image built without `--chown` lands the config `root:root`;
        with it, `proximadb:proximadb`.

        `Docker Build` runs only on the extensive tier, so without this guard a
        regression is invisible until a release.
        """
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        needle = "COPY --from=builder --chown=proximadb:proximadb"
        self.assertIn(needle, text, "guard has lost its target")
        p.write_text(
            text.replace(needle, "COPY --from=builder", 1), encoding="utf-8"
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("has no `--chown=`", result.stdout)

    # Every spelling below defeated an earlier `startswith("COPY") and
    # endswith("/config/config.toml")` check while leaving the image
    # unreachable. The last entry is a legal form that must NOT be flagged.
    CONFIG_COPY = (
        "COPY --from=builder --chown=proximadb:proximadb "
        "/build/config/config.toml /config/config.toml"
    )
    OWNERSHIP_EVASIONS = {
        "line continuation": (
            "COPY --from=builder \\\n    /build/config/config.toml \\\n"
            "    /config/config.toml",
            True,
        ),
        "exec/JSON form": (
            'COPY --from=builder ["/build/config/config.toml", '
            '"/config/config.toml"]',
            True,
        ),
        "quoted destination": (
            'COPY --from=builder /build/config/config.toml "/config/config.toml"',
            True,
        ),
        "trailing comment": (
            "COPY --from=builder /build/config/config.toml "
            "/config/config.toml  # default cfg",
            True,
        ),
        "RUN mv into place": (
            "COPY --from=builder /build/config/config.toml /tmp/config.toml\n"
            "RUN mv /tmp/config.toml /config/config.toml",
            True,
        ),
        "directory copy": ("COPY --from=builder /build/config/ /config/", True),
        "wrong owner": (
            "COPY --from=builder --chown=root:root /build/config/config.toml "
            "/config/config.toml",
            True,
        ),
        "legal exec form WITH chown": (
            "COPY --from=builder --chown=proximadb:proximadb "
            '["/build/config/config.toml", "/config/config.toml"]',
            False,
        ),
    }

    def test_ownership_guard_resists_every_known_evasion(self) -> None:
        for name, (spelling, should_fail) in self.OWNERSHIP_EVASIONS.items():
            with self.subTest(spelling=name):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                self.assertIn(self.CONFIG_COPY, text, "guard has lost its target")
                p.write_text(
                    text.replace(self.CONFIG_COPY, spelling, 1), encoding="utf-8"
                )
                result = _run(self.tree)
                self.assertEqual(
                    result.returncode,
                    1 if should_fail else 0,
                    result.stdout,
                )

    RUNTIME_COPY = (
        "COPY --from=builder --chown=proximadb:proximadb "
        "/build/config/config.toml /config/config.toml"
    )
    FULL_COPY = (
        "COPY --from=builder-full --chown=proximadb:proximadb "
        "/build/config/config.toml /config/config.toml"
    )

    def test_a_serving_stage_with_no_config_at_all_is_caught(self) -> None:
        """The tenth evasion: delete ONE stage's COPY, not the ownership.

        "Any stage whose name starts with runtime" meant deleting the `runtime`
        stage's COPY was masked by `runtime-full` keeping its own. Built that
        image: with nothing at /config/config.toml the entrypoint reports "no
        --config file found" and the server binds 127.0.0.1 on every port, so
        `docker run -p` reaches nothing -- on the one tier where no test runs.
        """
        for label, needle in (("runtime", self.RUNTIME_COPY), ("runtime-full", self.FULL_COPY)):
            with self.subTest(stage=label):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                self.assertIn(needle, text, "guard has lost its target")
                p.write_text(text.replace(needle + "\n", "", 1), encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("unreachable", result.stdout)

    OWNER_SPELLINGS = {
        "--chown=proximadb:proximadb": False,
        "--chown=proximadb": False,
        # groupadd -r / useradd -r on the shipped bases yield uid/gid 999, so the
        # numeric form is the SAME owner; rejecting it was a false positive.
        "--chown=999:999": False,
        "--chown=999": False,
        # An earlier prefix test (`startswith("proximadb")`) accepted these.
        "--chown=proximadbX:proximadbX": True,
        "--chown=proximadb-build": True,
        "--chown=root:root": True,
    }

    def test_ownership_guard_matches_the_owner_exactly(self) -> None:
        for spelling, should_fail in self.OWNER_SPELLINGS.items():
            with self.subTest(owner=spelling):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                p.write_text(
                    text.replace("--chown=proximadb:proximadb", spelling, 1),
                    encoding="utf-8",
                )
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1 if should_fail else 0, result.stdout)

    WRITE_PATHS = {
        "ADD instead of COPY": (
            "ADD --from=builder /build/config/config.toml /config/config.toml",
            True,
        ),
        "RUN cat redirect": (
            "COPY --from=builder /build/config/config.toml /tmp/c.toml\n"
            "RUN cat /tmp/c.toml > /config/config.toml",
            True,
        ),
        "RUN tee": (
            "COPY --from=builder /build/config/config.toml /tmp/c.toml\n"
            "RUN tee /config/config.toml < /tmp/c.toml",
            True,
        ),
        "dir copy, source not named config": (
            "COPY --from=builder /build/defaults/ /config/",
            True,
        ),
        "a RUN that only fixes perms is legitimate": (
            "COPY --from=builder --chown=proximadb:proximadb "
            "/build/config/config.toml /config/config.toml\n"
            "RUN chown proximadb:proximadb /config/config.toml",
            False,
        ),
    }

    def test_ownership_guard_covers_every_way_the_config_is_written(self) -> None:
        """`mv|cp|install|ln` was an enumerated verb list; `cat >`, `tee` and
        `sed -i` all wrote the config past it, and ADD was not inspected."""
        for label, (spelling, should_fail) in self.WRITE_PATHS.items():
            with self.subTest(path=label):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                p.write_text(text.replace(self.RUNTIME_COPY, spelling, 1), encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1 if should_fail else 0, result.stdout)

    def test_an_inherited_serving_stage_is_not_a_finding(self) -> None:
        """`FROM runtime AS variant` inherits the parent's config and user.

        The per-serving-stage rule initially treated any added stage carrying
        `USER proximadb` as config-less, which false-failed the idiomatic way to
        add an image variant with a message asserting something untrue.
        """
        for extra in (
            "FROM runtime AS runtime-slim\nUSER proximadb\n",
            "FROM runtime AS dbg\nUSER proximadb\nRUN echo hi\n",
        ):
            with self.subTest(stage=extra.splitlines()[0]):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                p.write_text(p.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_numeric_user_is_still_a_serving_stage(self) -> None:
        """`USER 999:999` is the same user as `USER proximadb`.

        The guard accepted `--chown=999` as that owner while requiring the
        literal string `proximadb` in USER, so switching to the numeric form and
        deleting the config passed with exit 0 -- the cheapest route back to an
        image that binds loopback on every port.
        """
        p = self.tree / "deploy/docker/Dockerfile"
        lines = p.read_text(encoding="utf-8").splitlines()
        # Must mutate the SAME stage whose config is removed. An earlier version
        # replaced the FIRST `USER proximadb`, which lives in `runtime-full`,
        # while deleting `runtime`'s COPY -- so the assertion held for the wrong
        # reason and the control passed even with the fix reverted.
        start = next(
            i for i, l in enumerate(lines) if l.strip().upper().endswith("AS RUNTIME")
        )
        out, dropped, switched = [], False, False
        for i, line in enumerate(lines):
            if i > start and not dropped and line == self.RUNTIME_COPY:
                dropped = True
                continue
            if i > start and not switched and line.strip() == "USER proximadb":
                out.append("USER 999:999")
                switched = True
                continue
            out.append(line)
        self.assertTrue(dropped and switched, "mutation did not apply to the runtime stage")
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("unreachable", result.stdout)

    PERM_RUNS = {
        "chown to root reproduces the defect": ("RUN chown root:root /config/config.toml", True),
        "chmod 000 is worse than the defect": ("RUN chmod 000 /config/config.toml", True),
        "chmod 0444 drops the owner's write": ("RUN chmod 0444 /config/config.toml", True),
        "chown to the runtime user is legitimate": (
            "RUN chown proximadb:proximadb /config/config.toml",
            False,
        ),
        "chmod 0640 keeps the owner's write": ("RUN chmod 0640 /config/config.toml", False),
    }

    def test_a_perms_fixing_run_is_judged_on_owner_and_mode(self) -> None:
        """Checking only the VERB let `chown root:root` and `chmod 000` pass."""
        for label, (extra, should_fail) in self.PERM_RUNS.items():
            with self.subTest(case=label):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                p.write_text(
                    text.replace(self.RUNTIME_COPY, self.RUNTIME_COPY + "\n" + extra, 1),
                    encoding="utf-8",
                )
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1 if should_fail else 0, result.stdout)

    def test_legal_spellings_of_a_correct_copy_are_not_findings(self) -> None:
        """Quoted owners and non-normalised destinations are both legal."""
        cases = {
            'quoted --chown': ('--chown=proximadb:proximadb', '--chown="proximadb:proximadb"'),
            "dotted destination": ("/config/config.toml\n", "/config/./config.toml\n"),
        }
        for label, (needle, repl) in cases.items():
            with self.subTest(case=label):
                self.setUp()
                p = self.tree / "deploy/docker/Dockerfile"
                text = p.read_text(encoding="utf-8")
                self.assertIn(needle, text)
                p.write_text(text.replace(needle, repl, 1), encoding="utf-8")
                result = _run(self.tree)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_ownership_guard_is_checked_per_stage(self) -> None:
        """Breaking ONE runtime stage must fail even though the other is fine.

        An earlier revision counted matches across the whole file, so a
        regression in `runtime` was masked by `runtime-full` still being right.
        """
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        full = (
            "COPY --from=builder-full --chown=proximadb:proximadb "
            "/build/config/config.toml /config/config.toml"
        )
        self.assertIn(full, text)
        p.write_text(
            text.replace(full, "COPY --from=builder-full /build/config/config.toml /config/config.toml", 1),
            encoding="utf-8",
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime-full", result.stdout)

    def test_every_config_copy_removed_names_the_serving_stages(self) -> None:
        """Removing every config lands on the per-stage finding, not vacuity.

        The guard reports the serving stages by name here rather than the
        generic lost-target message, because `USER proximadb` is still present
        and so the target has NOT been lost -- the config has.
        """
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        lines = [
            line for line in text.splitlines() if "/config/config.toml" not in line
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("run the server", result.stdout)
        self.assertIn("unreachable", result.stdout)

    def test_ownership_guard_reports_a_lost_target(self) -> None:
        """If the guard can no longer find a serving stage it must say so.

        This is the self-protection branch: a Dockerfile refactor that drops
        `USER proximadb` leaves the guard with nothing to check, and passing
        silently would retire the protection without anyone noticing.
        """
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        self.assertIn("USER proximadb", text)
        p.write_text(text.replace("USER proximadb", "USER root"), encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("lost its target", result.stdout)

    def test_rejects_a_zero_timeout(self) -> None:
        """`timeout-minutes: 0` cancels instantly; it is not a bound."""
        p = self.tree / ".github/workflows/ci.yml"
        text = p.read_text(encoding="utf-8")
        cap_re = re.compile(r"^    timeout-minutes: \d+.*$", re.MULTILINE)
        mutated, n = cap_re.subn("    timeout-minutes: 0", text, count=1)
        self.assertEqual(n, 1)
        p.write_text(mutated, encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("cancels", result.stdout)

    def test_catches_the_published_image_built_without_cloud(self) -> None:
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        self.assertIn("--features cloud-full", text)
        p.write_text(text.replace("--features cloud-full", "", 1), encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("omits `--features cloud-full`", result.stdout)

    def test_a_feature_named_only_in_a_comment_does_not_satisfy_the_image_check(
        self,
    ) -> None:
        """An earlier revision accepted any mention of the feature on the line."""
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        p.write_text(
            text.replace(
                "--features cloud-full", "# was --features cloud-full", 1
            ),
            encoding="utf-8",
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)

    def _mutate_one_features_entry(self, replacement: str) -> None:
        """Rewrite ONE of the two live targets, leaving the other intact.

        Mutating both would trip the `found == 0` safety net instead of the
        value check, which is how an earlier revision looked guarded while
        letting a single reverted target through.
        """
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        old = '            features: "cloud-full"\n'
        self.assertEqual(text.count(old), 2)
        p.write_text(text.replace(old, replacement + "\n", 1), encoding="utf-8")

    def test_rejects_every_empty_features_spelling(self) -> None:
        """Each of these is legal YAML that ships a feature-less binary.

        All three passed an earlier revision, whose regex required a
        double-quoted value with nothing after it.
        """
        for spelling in (
            "            features: ''",
            "            features:",
            '            features: ""  # TODO re-enable',
            "            features: aws",
            "            features: 'aws,azure'",
        ):
            with self.subTest(spelling=spelling):
                self.setUp()
                self._mutate_one_features_entry(spelling)
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1, spelling)
                self.assertIn("must include `cloud-full`", result.stdout)

    def test_accepts_every_legal_spelling_of_the_right_value(self) -> None:
        for spelling in (
            "            features: cloud-full",
            '            features: "cloud-full"  # keep',
            "            features: 'cloud-full'",
            '            features: "cloud-full,onnx"',
        ):
            with self.subTest(spelling=spelling):
                self.setUp()
                self._mutate_one_features_entry(spelling)
                result = _run(self.tree)
                self.assertEqual(
                    result.returncode, 0, spelling + "\n" + result.stdout
                )

    def _rewrite_image_features(self, flag: str) -> None:
        p = self.tree / "deploy/docker/Dockerfile"
        text = p.read_text(encoding="utf-8")
        old = "--features cloud-full,onnx"
        self.assertIn(old, text)
        p.write_text(text.replace(old, flag), encoding="utf-8")

    def test_image_feature_list_is_parsed_not_substring_matched(self) -> None:
        """Order and `=` spelling must not decide the verdict.

        An earlier revision tested `"--features cloud-full" in command`, which
        rejected the correct `--features onnx,cloud-full` purely on ordering --
        and the real line is `cloud-full,onnx`, one reorder from a false
        failure -- while accepting the bogus `cloud-fullish`.
        """
        for flag in (
            "--features onnx,cloud-full",
            "--features=cloud-full,onnx",
            "--features onnx,cloud-full,aws",
        ):
            with self.subTest(flag=flag, expect="pass"):
                self.setUp()
                self._rewrite_image_features(flag)
                result = _run(self.tree)
                self.assertEqual(result.returncode, 0, flag + "\n" + result.stdout)

        for flag in (
            "--features onnx",
            "--features cloud-fullish",
            "--features cloud-full-nope",
        ):
            with self.subTest(flag=flag, expect="fail"):
                self.setUp()
                self._rewrite_image_features(flag)
                result = _run(self.tree)
                self.assertEqual(result.returncode, 1, flag)
                self.assertIn("omits `--features cloud-full`", result.stdout)

    def test_a_documented_timeout_value_is_not_a_missing_one(self) -> None:
        """A trailing comment on the value must not read as absence."""
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        p.write_text(
            text.replace(
                "    timeout-minutes: 150",
                "    timeout-minutes: 150  # Windows leg measured 66 min",
                1,
            ),
            encoding="utf-8",
        )
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_commented_out_target_is_not_a_finding(self) -> None:
        """The disabled musl/macOS matrix entries keep `features: ""` in comments."""
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        p.write_text(text + '\n          #   features: ""\n', encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
