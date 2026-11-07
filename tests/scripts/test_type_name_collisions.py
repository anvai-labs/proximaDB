"""Tests for scripts/count_type_name_collisions.py (TD-NAMING-1).

Each case pins one rule, including the evasions the #2445 review found in
the first cut (comment-hidden cfg markers, single-item cfg(test), cfg(all
(test, ..)) modules, same-file duplicates, unsafe trait / union / same-line
attributes, re-exports, trade-off evasion, silent baseline drift, unreadable
files).
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "count_type_name_collisions.py"
spec = importlib.util.spec_from_file_location("count_type_name_collisions", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PROTO = "crates/foundation/proximadb-proto/src/proto/proximadb.v1.rs"


class TypeNameCollisionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text, raw=False):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.write_bytes if raw else path.write_text)(text)

    def defs(self):
        return mod.collect(self.root)[0]

    def reexports(self):
        return mod.collect(self.root)[1]

    # ---- definitions ----------------------------------------------------

    def test_cross_file_collision_counts(self):
        self.write("src/a.rs", "pub struct StorageAssignment;\n")
        self.write("crates/x/src/b.rs", "pub struct StorageAssignment { x: u8 }\n")
        self.assertEqual(self.defs(), {"StorageAssignment": 1})

    def test_same_file_duplicates_in_different_modules_count(self):
        self.write("src/a.rs", "pub mod v1 { pub struct Request; }\npub mod v2 { pub struct Request; }\n")
        self.assertEqual(self.defs(), {"Request": 1})

    def test_cfg_alternatives_in_one_file_are_one_definition(self):
        self.write(
            "src/a.rs",
            '#[cfg(feature = "x")]\npub struct Backend;\n#[cfg(not(feature = "x"))]\npub struct Backend;\n',
        )
        self.assertEqual(self.defs(), {})

    def test_unsafe_trait_union_and_same_line_attributes_count(self):
        self.write("src/a.rs", "pub unsafe trait Engine {}\n#[repr(C)] pub union Cell { a: u8 }\n")
        self.write("src/b.rs", "pub trait Engine {}\npub union Cell { b: u16 }\n")
        self.assertEqual(self.defs(), {"Engine": 1, "Cell": 1})

    def test_type_aliases_and_generated_proto_are_not_definitions(self):
        self.write("src/a.rs", "pub type Result<T> = std::result::Result<T, E>;\npub struct Collection;\n")
        self.write("src/b.rs", "pub type Result<T> = std::result::Result<T, F>;\n")
        self.write(PROTO, "pub struct Collection;\n")
        self.assertEqual(self.defs(), {})

    def test_test_trees_files_and_modules_are_excluded(self):
        self.write("src/a.rs", "pub struct Widget;\n")
        self.write("tests/it.rs", "pub struct Widget;\n")
        self.write("src/thing_tests.rs", "pub struct Widget;\n")
        self.write("src/b.rs", "#[cfg(test)]\nmod tests { pub struct Widget; }\n")
        self.write("src/c.rs", '#[cfg(all(test, feature = "aws"))]\nmod t { pub struct Widget; }\n')
        self.assertEqual(self.defs(), {})

    def test_cfg_any_test_module_is_NOT_test_only(self):
        self.write("src/a.rs", "pub struct Shared;\n")
        self.write("src/b.rs", "#[cfg(any(test, feature = \"x\"))]\npub mod m { pub struct Shared; }\n")
        self.assertEqual(self.defs(), {"Shared": 1})

    def test_comment_hidden_cfg_marker_does_not_hide_later_items(self):
        self.write("src/a.rs", "pub struct Config;\n")
        self.write("src/b.rs", "// #[cfg(test)]\npub struct Other;\npub struct Config;\n")
        self.assertEqual(self.defs(), {"Config": 1})

    def test_single_item_cfg_test_skips_only_that_item(self):
        self.write("src/a.rs", "pub struct Fixture;\npub struct Real;\n")
        self.write("src/b.rs", "#[cfg(test)]\npub struct Fixture;\npub struct Real;\n")
        self.assertEqual(self.defs(), {"Real": 1})

    # ---- re-exports ----------------------------------------------------

    def test_reexport_of_a_multiply_defined_name_is_a_collision(self):
        # The #2443 footgun: proto + ports both define it; a module re-exports one.
        self.write(PROTO, "pub struct StorageAssignment;\n")
        self.write("crates/ports/src/lib.rs", "pub struct StorageAssignment;\n")
        self.write("src/path_resolver.rs", "pub use ports::{CollectionPathResolver, StorageAssignment};\n")
        self.assertEqual(self.reexports(), {"StorageAssignment": 1})

    def test_reexport_of_a_unique_name_is_fine_and_renames_are_honoured(self):
        self.write("crates/ports/src/lib.rs", "pub struct Unique;\npub struct Twice;\n")
        self.write("src/other.rs", "pub struct Twice;\n")
        self.write("src/a.rs", "pub use ports::Unique;\npub use ports::Twice as PortsTwice;\n")
        self.assertEqual(self.reexports(), {})

    # ---- ratchet behaviour ---------------------------------------------

    def _run(self, *args):
        return mod.main(["--root", str(self.root), *args])

    def _baseline(self):
        path = self.root / "baseline.json"
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(path)), 0)
        return path

    def test_equal_passes_and_a_new_name_fails(self):
        self.write("src/a.rs", "pub struct Dup;\n")
        self.write("src/b.rs", "pub struct Dup;\n")
        base = self._baseline()
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 0)
        self.write("src/c.rs", "pub struct Other;\n")
        self.write("src/d.rs", "pub struct Other;\n")
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 1)

    def test_trade_off_evasion_fails(self):
        # Remove one collision, add a different one: total flat, still fails.
        self.write("src/a.rs", "pub struct X;\n")
        self.write("src/b.rs", "pub struct X;\n")
        base = self._baseline()
        (self.root / "src/b.rs").unlink()
        self.write("src/c.rs", "pub struct Y;\n")
        self.write("src/d.rs", "pub struct Y;\n")
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 1)

    def test_burn_down_without_lowering_the_baseline_fails(self):
        self.write("src/a.rs", "pub struct X;\n")
        self.write("src/b.rs", "pub struct X;\n")
        base = self._baseline()
        (self.root / "src/b.rs").unlink()
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 1)
        self._baseline()  # lowered in the same PR
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 0)

    def test_missing_baseline_and_unreadable_source_fail_closed(self):
        self.write("src/a.rs", "pub struct Dup;\n")
        self.assertEqual(
            self._run("--mode", "no-regression", "--baseline", str(self.root / "absent.json")), 1
        )
        base = self._baseline()
        self.write("src/latin1.rs", b"pub struct Caf\xe9;\n", raw=True)
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 1)


if __name__ == "__main__":
    unittest.main()
