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
        row = {"marker": self.marker, "get": 2, "list": 1, "head": 1, "bytes": 13}
        lines = [
            self.access("GET", "/account/fixture/other", 200, 1000),
            self.access("PUT", self.prefix + ".start", 201),
            self.access("GET", "/account/fixture/run/1000/obj", 206, 4),
            self.access("GET", "/account/fixture/run/1000/obj", 200, 9),
            self.access("GET", "/account/fixture/run/1000/absent", 404),
            self.access(
                "GET",
                "/account/fixture?restype=container&comp=list&prefix=run%2F1000%2F",
                200,
            ),
            self.access("HEAD", "/account/fixture/run/1000/obj", 200, 9),
            self.access("PUT", self.prefix + ".end", 201),
        ]
        read_windows([row], lines)
        self.assertEqual(
            row["wire"],
            {
                "get": 3,
                "object_bytes": 13,
                "partial_get": 1,
                "max_range_bytes": 4,
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

    def test_missing_wire_attempt_and_unrelated_traffic(self):
        start = self.access("PUT", self.prefix + ".start", 201)
        end = self.access("PUT", self.prefix + ".end", 201)
        row = {"marker": self.marker, "get": 1, "list": 0, "head": 0, "bytes": 4}
        with self.assertRaisesRegex(ValueError, "incomplete wire get"):
            read_windows([row], [start, end])
        for url in (
            "/account/fixture/run/10000/obj",
            "/account/fixture?restype=container&comp=list&prefix=other",
        ):
            with (
                self.subTest(url=url),
                self.assertRaisesRegex(ValueError, "unrelated request"),
            ):
                read_windows([row], [start, self.access("GET", url, 200, 4), end])

    def test_missing_query_rejected_even_when_other_windows_complete(self):
        with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
            validate_matrix([{"rows": 1000}], [], [1000], [4], 1)


if __name__ == "__main__":
    unittest.main()
