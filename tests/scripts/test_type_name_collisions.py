"""Tests for scripts/count_type_name_collisions.py (TD-NAMING-1).

Each case pins one rule of what does and does not count, including the
evasions a guard like this tends to miss (restricted visibility, generics,
inline test modules) and the deliberate exclusions (type aliases, generated
protobuf, test trees).
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


class TypeNameCollisionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def collisions(self):
        return mod.collect(self.root)

    def test_same_struct_name_in_two_files_collides(self):
        self.write("src/a.rs", "pub struct StorageAssignment;\n")
        self.write("crates/x/src/b.rs", "pub struct StorageAssignment { x: u8 }\n")
        self.assertEqual(list(self.collisions()), ["StorageAssignment"])
        self.assertEqual(mod.redundant_count(self.collisions()), 1)

    def test_enum_trait_restricted_visibility_and_generics_all_count(self):
        self.write("src/a.rs", "pub(crate) enum Mode { A }\npub trait Port<T> {}\n")
        self.write("src/b.rs", "pub enum Mode<T> { B(T) }\npub(super) trait Port: Send {}\n")
        self.assertEqual(sorted(self.collisions()), ["Mode", "Port"])

    def test_same_name_twice_in_one_file_is_not_a_cross_file_collision(self):
        self.write("src/a.rs", "pub struct Foo;\nmod inner { pub struct Foo; }\n")
        self.assertEqual(self.collisions(), {})

    def test_type_aliases_are_excluded(self):
        self.write("src/a.rs", "pub type Result<T> = std::result::Result<T, E>;\n")
        self.write("src/b.rs", "pub type Result<T> = std::result::Result<T, F>;\n")
        self.assertEqual(self.collisions(), {})

    def test_generated_proto_is_excluded(self):
        self.write(
            "crates/foundation/proximadb-proto/src/proto/proximadb.v1.rs",
            "pub struct Collection;\n",
        )
        self.write("src/domain.rs", "pub struct Collection;\n")
        self.assertEqual(self.collisions(), {})

    def test_test_trees_and_test_files_are_excluded(self):
        self.write("src/a.rs", "pub struct Widget;\n")
        self.write("tests/fixtures.rs", "pub struct Widget;\n")
        self.write("crates/x/tests/it.rs", "pub struct Widget;\n")
        self.write("src/thing_tests.rs", "pub struct Widget;\n")
        self.write("src/engine/tests.rs", "pub struct Widget;\n")
        self.assertEqual(self.collisions(), {})

    def test_definitions_inside_inline_cfg_test_modules_are_excluded(self):
        self.write("src/a.rs", "pub struct Real;\n#[cfg(test)]\nmod tests { pub struct Fake; }\n")
        self.write("src/b.rs", "pub struct Fake;\n")
        self.assertEqual(self.collisions(), {})

    def _run(self, *args):
        return mod.main(["--root", str(self.root), *args])

    def test_no_regression_fails_on_increase_and_passes_otherwise(self):
        baseline = self.root / "baseline.json"
        self.write("src/a.rs", "pub struct Dup;\n")
        self.write("src/b.rs", "pub struct Dup;\n")
        self.assertEqual(self._run("--mode", "baseline", "--baseline", str(baseline)), 0)
        self.assertEqual(json.loads(baseline.read_text())["redundant_definitions"], 1)
        # Equal: passes.
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(baseline)), 0)
        # A new collision: fails.
        self.write("src/c.rs", "pub struct Dup;\n")
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(baseline)), 1)
        # Burned down below the baseline: passes (and asks for a lower baseline).
        (self.root / "src/b.rs").unlink()
        (self.root / "src/c.rs").unlink()
        self.assertEqual(self._run("--mode", "no-regression", "--baseline", str(baseline)), 0)

    def test_missing_baseline_fails_closed(self):
        self.write("src/a.rs", "pub struct Dup;\n")
        self.assertEqual(
            self._run("--mode", "no-regression", "--baseline", str(self.root / "absent.json")),
            1,
        )


if __name__ == "__main__":
    unittest.main()
