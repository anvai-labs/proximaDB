"""Tests for scripts/count_type_name_collisions.py (TD-NAMING-1).

Each case pins one rule, including the evasions and false positives found in
the #2445 review rounds: comment/string lexing, cfg(test) variants, module
scopes, restricted visibility, nested comments, the both-ways per-name
ratchet, and baseline-write safety.
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
        return mod.collect(self.root)

    # ---- what counts --------------------------------------------------

    def test_cross_file_collision_counts(self):
        self.write("src/a.rs", "pub struct StorageAssignment;\n")
        self.write("crates/x/src/b.rs", "pub struct StorageAssignment { x: u8 }\n")
        self.assertEqual(self.defs(), {"StorageAssignment": 1})

    def test_enum_trait_generics_and_restricted_visibility_all_count(self):
        # Restricted visibility counts: duplicating a primitive harms within
        # the root crate as much as across crates (TD-NAMING-1).
        self.write("src/a.rs", "pub enum Mode { A }\npub trait Port<T> {}\npub(crate) struct Local;\n")
        self.write("src/b.rs", "pub enum Mode<T> { B(T) }\npub trait Port: Send {}\npub(crate) struct Local;\n")
        self.assertEqual(sorted(self.defs()), ["Local", "Mode", "Port"])

    def test_unsafe_trait_union_and_same_line_attributes_count(self):
        self.write("src/a.rs", "pub unsafe trait Engine {}\n#[repr(C)] pub union Cell { a: u8 }\n")
        self.write("src/b.rs", "pub trait Engine {}\npub union Cell { b: u16 }\n")
        self.assertEqual(self.defs(), {"Engine": 1, "Cell": 1})

    def test_same_file_duplicates_in_different_modules_count(self):
        self.write("src/a.rs", "pub mod v1 { pub struct Request; }\npub mod v2 { pub struct Request; }\n")
        self.assertEqual(self.defs(), {"Request": 1})

    def test_always_true_cfg_cannot_hide_a_collision_across_modules(self):
        self.write(
            "src/a.rs",
            "pub mod v1 { #[cfg(all())] pub struct Req; }\npub mod v2 { #[cfg(all())] pub struct Req; }\n",
        )
        self.assertEqual(self.defs(), {"Req": 1})

    def test_differently_named_cfg_modules_are_not_alternatives(self):
        self.write(
            "src/a.rs",
            '#[cfg(feature = "aws")]\npub mod s3 { pub struct Client; }\n'
            '#[cfg(feature = "gcp")]\npub mod gcs { pub struct Client; }\n',
        )
        self.assertEqual(self.defs(), {"Client": 1})

    def test_cfg_any_test_module_ships_and_counts(self):
        self.write("src/a.rs", "pub struct Shared;\n")
        self.write("src/b.rs", '#[cfg(any(test, feature = "x"))]\npub mod m { pub struct Shared; }\n')
        self.assertEqual(self.defs(), {"Shared": 1})

    # ---- what does not count -----------------------------------------

    def test_cfg_alternatives_in_one_scope_are_one_definition(self):
        self.write(
            "src/a.rs",
            '#[cfg(feature = "x")]\npub struct Backend;\n#[cfg(not(feature = "x"))]\npub struct Backend;\n',
        )
        self.assertEqual(self.defs(), {})

    def test_module_level_cfg_alternatives_are_one_definition(self):
        self.write(
            "src/a.rs",
            '#[cfg(feature = "aws")]\nmod imp { pub struct Factory; }\n'
            '#[cfg(not(feature = "aws"))]\nmod imp { pub struct Factory; }\n',
        )
        self.assertEqual(self.defs(), {})

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

    def test_test_anywhere_inside_cfg_all_is_test_only(self):
        self.write("src/a.rs", "pub struct Fixture;\n")
        self.write("src/b.rs", '#[cfg(all(feature = "abac", test))]\nmod t { pub struct Fixture; }\n')
        self.write("src/c.rs", '#[cfg(all(feature = "x", not(test)))]\npub mod live { pub struct Fixture; }\n')
        self.assertEqual(self.defs(), {"Fixture": 1})  # only the not(test) copy ships

    def test_restricted_visibility_test_module_is_still_stripped(self):
        self.write("src/a.rs", "pub struct Helper;\n")
        self.write("src/b.rs", "#[cfg(test)]\npub(crate) mod tests { pub struct Helper; }\n")
        self.assertEqual(self.defs(), {})

    def test_single_item_cfg_test_skips_only_that_item(self):
        self.write("src/a.rs", "pub struct Fixture;\npub struct Real;\n")
        self.write("src/b.rs", "#[cfg(test)]\npub struct Fixture;\npub struct Real;\n")
        self.assertEqual(self.defs(), {"Real": 1})

    def test_inner_cfg_test_file_is_excluded(self):
        self.write("src/a.rs", "pub struct Mock;\n")
        self.write("src/testing.rs", "#![cfg(test)]\npub struct Mock;\n")
        self.assertEqual(self.defs(), {})

    def test_out_of_line_cfg_test_module_file_is_excluded(self):
        self.write("src/storage/common/mod.rs", "#[cfg(test)]\npub mod test_utils;\n")
        self.write("src/storage/common/test_utils.rs", "pub struct MockEngine;\n")
        self.write("src/engine.rs", "pub struct MockEngine;\n")
        self.assertEqual(self.defs(), {})

    # ---- lexical robustness ------------------------------------------

    def test_comment_hidden_cfg_marker_does_not_hide_later_items(self):
        self.write("src/a.rs", "pub struct Config;\n")
        self.write("src/b.rs", "// #[cfg(test)]\npub struct Other;\npub struct Config;\n")
        self.assertEqual(self.defs(), {"Config": 1})

    def test_brace_inside_a_string_in_a_test_module_does_not_hide_later_items(self):
        self.write("src/a.rs", "pub struct Catalog;\n")
        self.write(
            "src/b.rs",
            '#[cfg(test)]\nmod tests {\n    fn t() { assert!(s.contains("Decimal:{")); }\n}\npub struct Catalog;\n',
        )
        self.assertEqual(self.defs(), {"Catalog": 1})

    def test_comment_markers_inside_strings_are_not_comments(self):
        self.write("src/a.rs", "pub struct Hidden;\n")
        self.write(
            "src/b.rs",
            'const URL: &str = "s3://bucket/*";\npub struct Hidden;\nconst END: &str = "*/";\n',
        )
        self.assertEqual(self.defs(), {"Hidden": 1})

    def test_nested_block_comments_are_fully_masked(self):
        self.write("src/a.rs", "pub struct W;\n")
        self.write("src/b.rs", "/* outer /* inner */ pub struct W; */\n")
        self.assertEqual(self.defs(), {})

    # ---- ratchet behaviour -------------------------------------------

    def _run(self, *args):
        return mod.main(["--root", str(self.root), *args])

    def _baseline(self):
        path = self.root / "baseline.json"
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(path), "--init"), 0)
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
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(base)), 0)
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 0)

    def test_baseline_mode_refuses_to_raise(self):
        self.write("src/a.rs", "pub struct X;\n")
        self.write("src/b.rs", "pub struct X;\n")
        base = self._baseline()
        self.write("src/c.rs", "pub struct X;\n")
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(base)), 1)
        self.assertEqual(json.loads(base.read_text())["definitions"], {"X": 1})

    def test_baseline_mode_needs_init_when_no_baseline_exists(self):
        self.write("src/a.rs", "pub struct X;\n")
        self.write("src/b.rs", "pub struct X;\n")
        absent = self.root / "fresh.json"
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(absent)), 1)
        self.assertFalse(absent.exists())

    def test_missing_baseline_and_unreadable_source_fail_closed(self):
        self.write("src/a.rs", "pub struct Dup;\n")
        self.assertEqual(
            self._run("--mode", "no-regression", "--baseline", str(self.root / "absent.json")), 1
        )
        base = self._baseline()
        self.write("src/latin1.rs", b"pub struct Caf\xe9;\n", raw=True)
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(base)), 1)

    # ---- cfg predicate evaluation (#2445 round 4) -----------------------

    def test_nested_any_test_inside_all_ships(self):
        self.write("src/a.rs", "pub struct F;\n")
        self.write("src/b.rs", '#[cfg(all(unix, any(test, feature = "y")))]\npub mod t { pub struct F; }\n')
        self.assertEqual(self.defs(), {"F": 1})

    def test_all_with_not_and_test_is_test_only(self):
        self.write("src/a.rs", "pub struct F;\n")
        self.write("src/b.rs", '#[cfg(all(not(feature = "aws"), test))]\nmod t { pub struct F; }\n')
        self.assertEqual(self.defs(), {})

    def test_attribute_with_nested_brackets_does_not_hide_cfg_test(self):
        self.write("src/a.rs", "pub struct F;\n")
        self.write("src/b.rs", "#[cfg(test)]\n#[schema(example = json!([1]))]\npub struct F;\n")
        self.assertEqual(self.defs(), {})

    def test_inner_cfg_all_test_file_is_excluded(self):
        self.write("src/a.rs", "pub struct F;\n")
        self.write("src/helpers.rs", "#![cfg(all(test, unix))]\npub struct F;\n")
        self.assertEqual(self.defs(), {})


if __name__ == "__main__":
    unittest.main()
