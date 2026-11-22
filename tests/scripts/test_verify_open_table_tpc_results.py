# SPDX-License-Identifier: Apache-2.0
"""Regression checks for benchmark output ordering."""

import unittest

try:
    import duckdb
    import pyarrow as pa
except ModuleNotFoundError:
    duckdb = pa = None
else:
    from scripts.bench.verify_open_table_tpc_results import check_order


@unittest.skipIf(
    duckdb is None or pa is None, "requires the open-table fixture dependencies"
)
class OutputOrderTest(unittest.TestCase):
    def setUp(self):
        self.connection = duckdb.connect(
            config={"threads": 1, "enable_external_access": False}
        )
        self.addCleanup(self.connection.close)
        self.schema = pa.schema([("key", pa.int32()), ("value", pa.int32())])

    def check(self, sql, rows):
        return check_order(self.connection, sql, self.schema, rows)

    def test_ties_can_swap_but_inverted_keys_fail(self):
        sql = "SELECT key, value FROM source ORDER BY 1"
        a, b, c = (1, 10), (1, 20), (2, 30)
        self.assertEqual(self.check(sql, [b, a, a, c]), (True, None))
        self.assertEqual(
            self.check(sql, [b, a, c, a]),
            (True, "ORDER BY inversion between rows 3 and 4"),
        )

    def test_nulls_follow_explicit_or_postgres_default_placement(self):
        rows = [(None, 0), (2, 1), (1, 2)]
        self.assertEqual(
            self.check("SELECT * FROM source ORDER BY key DESC", rows), (True, None)
        )
        self.assertEqual(
            self.check("SELECT * FROM source ORDER BY key DESC NULLS LAST", rows),
            (True, "ORDER BY inversion between rows 1 and 2"),
        )

    def test_alias_precedes_matching_input_name(self):
        self.assertEqual(
            self.check(
                "SELECT measure AS key, key AS value FROM source ORDER BY key",
                [(1, 20), (2, 10)],
            ),
            (True, None),
        )

    def test_expression_over_projected_values(self):
        sql = "SELECT * FROM source ORDER BY value - key, 1"
        self.assertEqual(self.check(sql, [(4, 1), (2, 4), (3, 5)]), (True, None))
        self.assertEqual(
            self.check(sql, [(4, 1), (3, 5), (2, 4)]),
            (True, "ORDER BY inversion between rows 2 and 3"),
        )

    def test_only_top_level_order_is_checked(self):
        self.assertEqual(
            self.check(
                "SELECT * FROM (SELECT * FROM source ORDER BY key) t",
                [(2, 20), (1, 10)],
            ),
            (False, None),
        )

    def test_hidden_key_fails_verification(self):
        with self.assertRaisesRegex(ValueError, "hidden is unavailable"):
            self.check("SELECT key, value FROM source ORDER BY hidden", [(1, 2)])
