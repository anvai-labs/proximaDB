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
"""
from __future__ import annotations

import pathlib
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
