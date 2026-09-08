"""Unit tests for the MLflow conformance runner's ratchet contract."""

import contextlib
import io
import unittest

import mlflow_conformance as harness


class ConformanceHarnessTest(unittest.TestCase):
    def setUp(self):
        harness.PASSED.clear()
        harness.FAILED.clear()

    def test_more_than_the_current_ratchet_can_pass(self):
        for index in range(13):
            harness.step(f"step-{index}", lambda: None)

        self.assertEqual(len(harness.PASSED), 13)
        self.assertEqual(harness.result_code(), 0)

    def test_any_failed_step_fails_the_run(self):
        def fail():
            raise RuntimeError("expected test failure")

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            harness.step("broken", fail)

        self.assertEqual(harness.FAILED, ["broken"])
        self.assertEqual(harness.result_code(), 1)


if __name__ == "__main__":
    unittest.main()
