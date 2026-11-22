"""The scale ledger treats absent wire evidence as an error, including a lost query."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/bench"))
from pax_scale_wire_report import (
    DFS_READ_UNIT,
    RATE_CARD,
    estimate_request_cost,
    load_rate_card,
    read_windows,
    validate_matrix,
)


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
                "dfs_read_units": 2,
                "unpriced_responses": 1,
                "max_range_bytes": 4,
                "errors": 1,
                "list": 1,
                "head": 1,
            },
        )

        with RATE_CARD.open() as f:
            costs = estimate_request_cost([row], list(csv.DictReader(f)))
        self.assertEqual(costs["cost_status"], "unpriced_response")
        self.assertIsNone(costs["dfs_total_usd_per_m_queries"])

    def test_pricing_per_response_and_per_operation(self):
        sizes = (1, DFS_READ_UNIT, DFS_READ_UNIT + 1)
        row = {
            "marker": self.marker,
            "get": 3,
            "list": 1,
            "head": 1,
            "bytes": sum(sizes),
        }
        obj = "/account/fixture/run/1000/object"
        listing = "/account/fixture?comp=list&prefix=run%2F1000%2F"
        read_windows(
            [row],
            [
                self.access("PUT", self.prefix + ".start", 201),
                *(self.access("GET", obj, 206, n) for n in sizes),
                self.access("HEAD", obj, 200, sum(sizes)),
                self.access("GET", listing, 200),
                self.access("PUT", self.prefix + ".end", 201),
            ],
        )
        self.assertEqual(row["wire"]["dfs_read_units"], 4)
        with RATE_CARD.open() as f:
            costs = estimate_request_cost([row], list(csv.DictReader(f)))
        self.assertEqual(costs["s3_total_usd_per_m_queries"], 6.6)
        self.assertEqual(costs["blob_total_usd_per_m_queries"], 6.6)
        self.assertEqual(costs["dfs_get_usd_per_m_queries"], 2.08)
        self.assertEqual(costs["dfs_head_usd_per_m_queries"], 0.52)
        self.assertEqual(costs["dfs_total_usd_per_m_queries"], 9.1)

    def test_rate_card_and_retrieval_units(self):
        rates = load_rate_card()
        wire = {"get": 1, "list": 1, "dfs_read_units": 256, "object_bytes": 1 << 30}
        prices = estimate_request_cost([{"wire": wire}], rates)
        self.assertEqual(prices["s3_ia_retrieval_usd_per_m_queries"], 10000)
        self.assertEqual(prices["s3_ia_total_usd_per_m_queries"], 10006)
        with tempfile.TemporaryDirectory() as tmp:
            card = Path(tmp) / "rates.csv"
            for broken in (
                [*rates, rates[0]],
                [dict(r, counter="unknown") for r in rates],
                [r for r in rates if r["operation"] != "list"],
            ):
                with card.open("w") as out:
                    writer = csv.DictWriter(out, fieldnames=rates[0])
                    writer.writeheader()
                    writer.writerows(broken)
                with self.assertRaises(ValueError):
                    load_rate_card(card)

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
