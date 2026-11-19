"""Pins for scripts/check_doc_references.py (dead links + orphan TD/ADR ids)."""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_doc_references", ROOT / "scripts" / "check_doc_references.py"
)
assert _spec is not None and _spec.loader is not None
cdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdr)

TD_ID_RE = re.compile(r"\bTD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
ADR_DIR = ROOT / "docs" / "12-design" / "adr"


class BrokenLinks(unittest.TestCase):
    files = {
        "docs/12-design/adr/ADR-097-x.adoc",
        "docs/10-quality/td/TD-SPINE-1-y.adoc",
    }

    def broken(self, text: str) -> list[str]:
        return cdr.broken_links(ADR_DIR / "ADR-098-z.adoc", text, self.files)

    def test_existing_sibling_and_relative_targets_resolve(self) -> None:
        text = (
            "link:ADR-097-x.adoc[ADR-097] "
            "link:../../10-quality/td/TD-SPINE-1-y.adoc[TD-SPINE-1] "
            "xref:ADR-097-x.adoc#_d0[D0]"
        )
        self.assertEqual(self.broken(text), [])

    def test_guessed_filename_is_reported(self) -> None:
        # The exact failure shape from the ADR-098 filing: a plausible but wrong slug.
        self.assertEqual(
            self.broken("link:ADR-097-guessed-slug.adoc[ADR-097]"),
            ["ADR-097-guessed-slug.adoc"],
        )

    def test_urls_anchors_and_attributes_are_ignored(self) -> None:
        text = (
            "link:https://example.com/a.adoc[x] xref:_d6_supersessions[D6] "
            "link:{docs-root}/a.adoc[y] link:mailto:a@b.c[z]"
        )
        self.assertEqual(self.broken(text), [])

    def test_markdown_relative_link(self) -> None:
        md = ROOT / "docs" / "12-design" / "adr" / "NOTE.md"
        self.assertEqual(cdr.broken_links(md, "[x](ADR-097-x.adoc)", self.files), [])
        self.assertEqual(cdr.broken_links(md, "[x](missing.adoc)", self.files), ["missing.adoc"])


class ProseOnly(unittest.TestCase):
    def test_markdown_link_with_title_is_checked(self) -> None:
        md = ROOT / "docs" / "12-design" / "adr" / "NOTE.md"
        self.assertEqual(cdr.broken_links(md, '[x](missing.adoc "Title")', set()), ["missing.adoc"])

    def test_tagged_fence_and_post_block_prose(self) -> None:
        # Round-3 finding 1: the tagged opener must toggle the fence (content
        # hidden) AND the bare closer must restore scanning (prose after the
        # block stays visible).
        text = "TD-ZZ-1\n```rust\nTD-ZZ-2\n```\nTD-ZZ-3\n"
        body = cdr.prose(text)
        self.assertIn("TD-ZZ-1", body)
        self.assertIn("TD-ZZ-3", body)
        self.assertNotIn("TD-ZZ-2", body)

    def test_double_backtick_inline_code_is_hidden(self) -> None:
        self.assertNotIn("TD-ZZ-9", cdr.prose("``TD-ZZ-9``"))

    def test_code_blocks_and_inline_code_are_not_references(self) -> None:
        text = "real TD-ZZ-1\n----\nlink:ghost.adoc[x] TD-ZZ-2\n----\n`TD-ZZ-3` inline\n```\nTD-ZZ-4\n```\n"
        body = cdr.prose(text)
        self.assertIn("TD-ZZ-1", body)
        for hidden in ("ghost.adoc", "TD-ZZ-2", "TD-ZZ-3", "TD-ZZ-4"):
            self.assertNotIn(hidden, body)


class UnresolvedIds(unittest.TestCase):
    td_known = {"TD-SPINE-1", "TD-PAXDIR-1", "TD-007"}
    adr_known = {"98", "83"}

    def bad(self, text: str) -> list[str]:
        return cdr.unresolved_ids(text, self.td_known, self.adr_known, TD_ID_RE)

    def test_known_ids_and_file_slugs_resolve(self) -> None:
        self.assertEqual(self.bad("TD-SPINE-1 and TD-PAXDIR-1-collection-global-centroid-directory"), [])

    def test_sub_id_must_exist_but_numeric_slug_words_trim(self) -> None:
        known = {"TD-SEAM-12", "TD-DEPS-1"}
        bad = cdr.unresolved_ids(
            "TD-SEAM-12-F99 TD-DEPS-1-bincode-2x-upgrade TD-SEAM-12", known, set(), TD_ID_RE
        )
        self.assertEqual(bad, ["TD-SEAM-12-F99"])

    def test_hallucinated_id_is_reported(self) -> None:
        self.assertEqual(self.bad("see TD-SPINE-9"), ["TD-SPINE-9"])

    def test_family_mentions_without_a_number_are_not_ids(self) -> None:
        self.assertEqual(self.bad("the TD-IOTRACE family and TD-SEAM-* rows"), [])

    def test_adr_leading_zeros_compare_equal(self) -> None:
        self.assertEqual(self.bad("ADR-098, ADR-0083, ADR-083"), [])
        self.assertEqual(self.bad("ADR-0099"), ["ADR-0099"])

    def test_adr_key_normalization(self) -> None:
        self.assertEqual(cdr._adr_key("0053"), "53")
        self.assertEqual(cdr._adr_key("010a"), "10a")


class ArchivedIdsResolve(unittest.TestCase):
    def test_archive_register_ids_are_known(self) -> None:
        uniq_spec = importlib.util.spec_from_file_location(
            "check_td_adr_unique", ROOT / "scripts" / "check_td_adr_unique.py"
        )
        assert uniq_spec is not None and uniq_spec.loader is not None
        uniq = importlib.util.module_from_spec(uniq_spec)
        uniq_spec.loader.exec_module(uniq)
        archived = uniq.archived_td_ids()
        # TD-007 lives only in TECHNICAL_DEBT_CLOSED_*.adoc; it must resolve, never count as an orphan.
        self.assertIn("TD-007", archived)
        self.assertNotIn("TD-007", uniq.td_ids())


if __name__ == "__main__":
    unittest.main()
