# SPDX-License-Identifier: Apache-2.0
"""Negative controls for scripts/check_release_build_features.py.

A guard that passes on both the broken and the fixed tree is worthless, so each
of the three defect classes it exists to catch gets a mutation here. The real
pre-fix tree (v0.4.0 prep, before this guard landed) produced 26 findings:
14 + 7 missing job timeouts, 4 feature-less release targets, 1 image build.
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
    "Dockerfile",
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

    def test_catches_an_image_built_without_cloud(self) -> None:
        p = self.tree / "Dockerfile"
        text = p.read_text(encoding="utf-8")
        p.write_text(text.replace("--features cloud-full ", ""), encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("omits `--features cloud-full`", result.stdout)

    def test_a_commented_out_target_is_not_a_finding(self) -> None:
        """The disabled musl/macOS matrix entries keep `features: ""` in comments."""
        p = self.tree / ".github/workflows/release.yml"
        text = p.read_text(encoding="utf-8")
        p.write_text(text + '\n          #   features: ""\n', encoding="utf-8")
        result = _run(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
