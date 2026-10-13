"""Tests for the Cargo.lock downgrade guard (TD-AUDITGATE-1).

The guard exists because a real security fix was silently reverted and passed
every check, so the cases below are the ones that actually matter: a downgrade
must fail, and the legitimate shapes (dedup, addition, removal, forward move)
must stay clean. A guard that reddens on a dedup gets disabled within a week,
which is the same end state as not having one.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_lockfile_no_downgrade",
    REPO_ROOT / "scripts" / "check_lockfile_no_downgrade.py",
)
guard = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(guard)


def lock(*pairs: tuple[str, str]) -> str:
    """Build a minimal Cargo.lock from (name, version) pairs."""
    out = ['version = 4\n']
    for name, version in pairs:
        out.append(f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n')
    return "".join(out)


class LockfileDowngradeGuardTest(unittest.TestCase):
    def test_a_downgrade_is_reported(self) -> None:
        findings = guard.find_downgrades(
            lock(("rustls", "0.23.45")), lock(("rustls", "0.23.44"))
        )
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("rustls: 0.23.45 -> 0.23.44", findings[0])

    def test_the_real_regression_shape_is_reported(self) -> None:
        # #2004 reverted #2003: the lock kept BOTH majors, and only the 0.23
        # line moved back. Comparing whole-package maxima would have missed it,
        # because 0.23.45 -> 0.23.44 leaves the overall max unchanged at 0.21/0.23
        # ordering if compared naively.
        findings = guard.find_downgrades(
            lock(("rustls", "0.21.12"), ("rustls", "0.23.45")),
            lock(("rustls", "0.21.12"), ("rustls", "0.23.44")),
        )
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("0.23.45 -> 0.23.44", findings[0])

    def test_a_forward_move_is_clean(self) -> None:
        self.assertEqual(
            guard.find_downgrades(
                lock(("rustls", "0.23.44")), lock(("rustls", "0.23.45"))
            ),
            [],
        )

    def test_dropping_a_stale_duplicate_major_is_not_a_downgrade(self) -> None:
        # The TD-RUSTDEP-1 fix (#2011): rustls 0.21 + 0.23 -> 0.23 only. This is
        # the single most important false positive to avoid.
        self.assertEqual(
            guard.find_downgrades(
                lock(("rustls", "0.21.12"), ("rustls", "0.23.45")),
                lock(("rustls", "0.23.45")),
            ),
            [],
        )

    def test_removal_and_addition_are_clean(self) -> None:
        self.assertEqual(
            guard.find_downgrades(lock(("sct", "0.7.1")), lock(("other", "1.0.0"))), []
        )

    def test_zero_x_minor_counts_as_its_own_line(self) -> None:
        # Cargo treats 0.x as breaking per minor, so 0.26 and 0.24 are separate
        # lines: dropping 0.24 is a dedup, but 0.26.6 -> 0.26.5 is a downgrade.
        self.assertEqual(
            guard.find_downgrades(
                lock(("tokio-rustls", "0.24.1"), ("tokio-rustls", "0.26.6")),
                lock(("tokio-rustls", "0.26.6")),
            ),
            [],
        )
        findings = guard.find_downgrades(
            lock(("tokio-rustls", "0.26.6")), lock(("tokio-rustls", "0.26.5"))
        )
        self.assertEqual(len(findings), 1, findings)

    def test_non_numeric_version_metadata_does_not_crash(self) -> None:
        # Both of these are real entries in this repo's lock.
        for a, b in (
            ("0.9.34+deprecated", "0.9.34+deprecated"),
            ("1.10505.0", "1.10506.0"),
        ):
            with self.subTest(pair=(a, b)):
                self.assertEqual(
                    guard.find_downgrades(lock(("p", a)), lock(("p", b))), []
                )
        findings = guard.find_downgrades(
            lock(("p", "1.10506.0")), lock(("p", "1.10505.0"))
        )
        self.assertEqual(len(findings), 1, findings)

    def test_the_committed_lock_passes_against_itself(self) -> None:
        text = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
        self.assertEqual(guard.find_downgrades(text, text), [])

    def test_parse_handles_the_real_lock(self) -> None:
        parsed = guard.parse_lock((REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8"))
        self.assertGreater(len(parsed), 500, "real lock should parse to many packages")
        self.assertIn("rustls", parsed)


if __name__ == "__main__":
    unittest.main()
