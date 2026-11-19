"""Pins for the rules check_td_adr_unique.py gained in the ADR-098 reconciliation.

Each test builds a throwaway register (legacy table, archive, per-file dir, ADR dir)
and points the module at it, so the real docs tree is never read or written.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("ctau", ROOT / "scripts" / "check_td_adr_unique.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Register(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.td_dir = base / "td"
        self.adr_dir = base / "adr"
        self.td_dir.mkdir()
        self.adr_dir.mkdir()
        self.table = base / "TECHNICAL_DEBT.adoc"
        self.table.write_text("| TD-LIVE-1 | live row | HIGH | Open\n", encoding="utf-8")
        (base / "TECHNICAL_DEBT_CLOSED_2026_06_24.adoc").write_text(
            "| TD-007 | archived | Mar 2026 | done\n", encoding="utf-8"
        )
        self.mod = load()
        self.mod.TD_DIR, self.mod.TD_TABLE = self.td_dir, self.table
        self.mod.ADR_DIR, self.mod.ADR_README = self.adr_dir, self.adr_dir / "README.adoc"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def td(self, name: str, heading: str) -> None:
        (self.td_dir / name).write_text(heading + "\n\n|===\n| Status | Open\n|===\n", encoding="utf-8")

    def run_main(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch("sys.argv", ["check_td_adr_unique.py"]):
                code = self.mod.main()
        return code, err.getvalue()

    def test_clean_register_passes(self) -> None:
        self.td("TD-NEW-1-x.adoc", "= TD-NEW-1: fine")
        self.assertEqual(self.run_main()[0], 0)

    def test_slug_polluted_heading_fails(self) -> None:
        self.td("TD-NEW-1-some-slug.adoc", "= TD-NEW-1-some-slug: title")
        code, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("filename slug", err)

    def test_reusing_an_archived_id_fails(self) -> None:
        self.td("TD-007-again.adoc", "= TD-007: reused")
        code, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("reuses an id closed into", err)

    def test_legacy_row_pointing_at_its_own_file_is_not_a_duplicate(self) -> None:
        self.table.write_text(
            "| TD-LIVE-1 | moved — see `td/TD-LIVE-1-moved.adoc` | HIGH | Partial\n", encoding="utf-8"
        )
        self.td("TD-LIVE-1-moved.adoc", "= TD-LIVE-1: moved")
        self.assertEqual(self.run_main()[0], 0)

    def test_legacy_row_without_pointer_is_a_duplicate(self) -> None:
        self.td("TD-LIVE-1-dup.adoc", "= TD-LIVE-1: dup")
        code, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("declared 2", err)

    def test_markdown_hash_heading_declares_an_id(self) -> None:
        self.td("TD-NEW-2-a.adoc", "= TD-NEW-2: first")
        self.td("TD-NEW-2-b.md", "# TD-NEW-2: duplicate in markdown")
        code, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("TD-NEW-2", err)

    def test_markdown_companion_without_colon_is_not_a_declaration(self) -> None:
        self.td("TD-NEW-3-a.adoc", "= TD-NEW-3: parent")
        self.td("TD-NEW-3-S0-AUDIT.md", "# TD-NEW-3 S0 Audit — companion")
        self.assertEqual(self.run_main()[0], 0)


import unittest.mock  # noqa: E402  (kept after the class for readability of the fixtures)

if __name__ == "__main__":
    unittest.main()
