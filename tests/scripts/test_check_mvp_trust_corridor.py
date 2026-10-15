#!/usr/bin/env python3
"""Regression tests for check_mvp_trust_corridor.py's release-note target scan.

TD-SECSCAN-1 §8 (`py/redos`): the target regex was
`([a-zA-Z0-9_-]+(?:-[a-zA-Z0-9_-]+)+)` — nested quantifiers over a class that
contains the separator, so a long `-`-heavy run that does not end in a
`.tar.gz`/`.zip` suffix backtracks exponentially. These tests pin both the
semantics (same targets as before on the repo's real release workflow) and the
linear bound. Run: python3 -m unittest discover -s tests/scripts -p test_check_mvp_trust_corridor.py
"""

import importlib.util
import multiprocessing
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_mvp_trust_corridor", ROOT / "scripts" / "check_mvp_trust_corridor.py"
)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

PREFIX = "proximadb-${{ needs.prepare.outputs.version }}-"


def _scan(text, queue):
    queue.put(sorted(checker.release_note_targets(text)))


class ReleaseNoteTargetsTest(unittest.TestCase):
    def test_extracts_multi_segment_targets(self):
        notes = (
            f"| `{PREFIX}x86_64-unknown-linux-gnu.tar.gz` |\n"
            f"| `{PREFIX}aarch64-apple-darwin.tar.gz` |\n"
            f"| `{PREFIX}x86_64-pc-windows-msvc.zip` |\n"
        )
        self.assertEqual(
            checker.release_note_targets(notes),
            {"x86_64-unknown-linux-gnu", "aarch64-apple-darwin", "x86_64-pc-windows-msvc"},
        )

    def test_rejects_single_segment_and_edge_dash_names(self):
        # The original pattern required an interior `-` (at least two segments).
        notes = f"{PREFIX}linux.tar.gz {PREFIX}a-.zip {PREFIX}-.zip {PREFIX}x-y.zip"
        self.assertEqual(checker.release_note_targets(notes), {"x-y"})

    def test_real_release_workflow_matches_contract(self):
        # The live check over the repo must stay green with the rewritten scan.
        import tomllib

        data = tomllib.loads(checker.CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(checker.validate_repository(data), [])

    def test_dash_heavy_input_is_linear(self):
        # 40 dashes and no suffix: the old regex takes ~2^40 steps here.
        hostile = PREFIX + "a" + "-a" * 40 + "!"
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_scan, args=(hostile * 50, queue))
        proc.start()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            self.fail("release_note_targets did not finish in 5s (catastrophic backtracking)")
        self.assertEqual(queue.get(timeout=1), [])


if __name__ == "__main__":
    unittest.main()
