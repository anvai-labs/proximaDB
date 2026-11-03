"""Negative controls for the per-test coverage report's evidence selection."""

import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "test_coverage_overlap",
    Path(__file__).resolve().parents[2] / "scripts/measure_test_coverage_overlap.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CoverageSelection(unittest.TestCase):
    def document(self, regions, filenames=None):
        return {
            "data": [
                {
                    "functions": [
                        {
                            "filenames": filenames
                            or [str(Path("selected.rs").resolve())],
                            "regions": regions,
                        }
                    ]
                }
            ]
        }

    def test_only_executed_code_regions_from_selected_source_are_counted(self):
        document = self.document(
            [
                [10, 1, 15, 1, 2, 0, 0, 0],
                [11, 1, 11, 3, 0, 0, 0, 0],
                [12, 1, 12, 3, 2, 0, 0, 3],
                [10, 1, 10, 9, 3, 1, 0, 0],
            ],
            [str(Path("selected.rs").resolve()), str(Path("other.rs").resolve())],
        )
        self.assertEqual(
            MODULE.covered_regions(document, Path("selected.rs").resolve(), 10),
            {(10, 1, 15, 1)},
        )

    def test_missing_or_uncovered_function_cannot_produce_pruning_evidence(self):
        for line, count in [(10, 0), (20, 1)]:
            with self.subTest(line=line, count=count), self.assertRaises(ValueError):
                MODULE.covered_regions(
                    self.document([[line, 1, line + 1, 1, count, 0, 0, 0]]),
                    Path("selected.rs").resolve(),
                    10,
                )

    @patch.object(MODULE.subprocess, "run")
    def test_zero_matching_tests_is_rejected_even_when_libtest_exits_zero(self, run):
        run.return_value = subprocess.CompletedProcess(
            [], 0, "test result: ok. 0 passed; 0 failed;", ""
        )
        with self.assertRaises(ValueError):
            MODULE.run_test(Path("binary"), "misspelled", Path("profile"))

    @patch.object(MODULE.subprocess, "run")
    def test_one_passing_test_is_accepted(self, run):
        run.return_value = subprocess.CompletedProcess(
            [], 0, "test result: ok. 1 passed; 0 failed;", ""
        )
        MODULE.run_test(Path("binary"), "selected", Path("profile"))
