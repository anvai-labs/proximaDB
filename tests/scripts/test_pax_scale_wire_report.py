"""The scale ledger treats absent wire evidence as an error, including a lost query."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/bench"))
from pax_scale_wire_report import read_windows, validate_matrix


class TestPaxScaleWireReport(unittest.TestCase):
    marker = "az://fixture/run/1000/_meter/q0"
    prefix = "/account/fixture/run/1000/_meter/q0"

    def access(self, method, path, status, length="-"):
        return f'127.0.0.1 - - [time] "{method} {path} HTTP/1.1" {status} {length}'

    def test_counts_wire_attempts_and_splits_list_and_range(self):
        row = {"marker": self.marker}
        lines = [
            self.access("GET", "/account/fixture/other", 200, 1000),
            self.access("PUT", self.prefix + ".start", 201),
            self.access("GET", "/account/fixture/obj", 206, 4),
            self.access("GET", "/account/fixture/obj", 200, 9),
            self.access("GET", "/account/fixture/absent", 404),
            self.access("GET", "/account/fixture?restype=container&comp=list", 200),
            self.access("HEAD", "/account/fixture/obj", 200, 9),
            self.access("PUT", self.prefix + ".end", 201),
        ]
        read_windows([row], lines)
        self.assertEqual(
            row["wire"],
            {
                "get": 3,
                "object_bytes": 13,
                "partial_get": 1,
                "errors": 1,
                "list": 1,
                "head": 1,
            },
        )

    def test_missing_or_overlapping_markers(self):
        row = {"marker": self.marker}
        start = self.access("PUT", self.prefix + ".start", 201)
        for lines in ([], [start], [start, start]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                read_windows([row], lines)

    def test_missing_query_rejected_even_when_other_windows_complete(self):
        with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
            validate_matrix([{"rows": 1000}], [], [1000], [4], 1)


if __name__ == "__main__":
    unittest.main()
