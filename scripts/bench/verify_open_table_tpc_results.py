#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check typed answers and SQL ordering for the pinned-source pgwire sweep."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path

import duckdb
import pyarrow as pa


def parse_value(value, data_type):
    if (
        value is None
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
    ):
        return value
    if pa.types.is_integer(data_type):
        return int(value)
    if pa.types.is_decimal(data_type):
        return Decimal(value)
    if pa.types.is_floating(data_type):
        return float(value)
    if pa.types.is_date(data_type):
        return date.fromisoformat(value)
    raise ValueError(f"unsupported answer type {data_type}")


def group_rows(rows, schema):
    float_columns = [
        i for i, field in enumerate(schema) if pa.types.is_floating(field.type)
    ]
    groups = defaultdict(list)
    for row in rows:
        if len(row) != len(schema):
            raise ValueError(f"expected {len(schema)} columns, received {len(row)}")
        # Match exact columns (including decimals) and float null positions before
        # applying tolerances. Preserve multiplicity for duplicate rows.
        key = tuple(
            value is None if i in float_columns else value
            for i, value in enumerate(row)
        )
        groups[key].append(tuple(row[i] for i in float_columns if row[i] is not None))
    return groups


def compare_answer(reader, rows, relative_tolerance, absolute_tolerance):
    schema = reader.schema
    expected = group_rows(
        (
            row
            for i in range(reader.num_record_batches)
            for row in zip(
                *(column.to_pylist() for column in reader.get_batch(i).columns),
                strict=True,
            )
        ),
        schema,
    )
    actual = group_rows(rows, schema)
    if expected.keys() != actual.keys():
        return "rows differ in exact values or float nullability"
    for key, floats in expected.items():
        if len(floats) != len(actual[key]):
            return "row multiplicities differ"
        for expected_values, actual_values in zip(
            sorted(floats), sorted(actual[key]), strict=True
        ):
            if not all(
                math.isclose(
                    a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance
                )
                for a, b in zip(expected_values, actual_values, strict=True)
            ):
                return "float values differ"
    return None


def parse_sql(connection, sql):
    tree = json.loads(
        connection.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    )
    if tree["error"]:
        raise ValueError(tree["error_message"])
    if len(tree["statements"]) != 1:
        raise ValueError("expected one SQL statement")
    return tree


def expression_identity(expression):
    if isinstance(expression, dict):
        return {
            key: expression_identity(value)
            for key, value in expression.items()
            if key not in ("alias", "query_location")
        }
    if isinstance(expression, list):
        return [expression_identity(value) for value in expression]
    return expression


def output_column(index):
    return {
        "class": "COLUMN_REF",
        "type": "COLUMN_REF",
        "alias": "",
        "column_names": [f"wire_{index}"],
    }


def bind_order_expression(expression, projections, names):
    """Bind a sort expression using only values visible in the result."""
    if isinstance(expression, list):
        return [
            bind_order_expression(value, projections, names) for value in expression
        ]
    if not isinstance(expression, dict):
        return expression
    if "class" in expression:
        identity = expression_identity(expression)
        if identity in projections:
            return output_column(projections.index(identity))
        if expression["class"] == "COLUMN_REF":
            # A qualified reference must match a projected expression above.
            parts = expression["column_names"]
            if len(parts) == 1 and names.count(parts[0].lower()) == 1:
                return output_column(names.index(parts[0].lower()))
            raise ValueError(
                f"ORDER BY column {'.'.join(parts)} is unavailable in the result"
            )
        if expression["class"] == "SUBQUERY":
            raise ValueError(
                "ORDER BY subquery cannot be evaluated from result columns"
            )
    return {
        key: bind_order_expression(value, projections, names)
        for key, value in expression.items()
    }


def check_order(connection, sql, schema, rows):
    source = parse_sql(connection, sql)["statements"][0]["node"]
    orders = next(
        (
            entry["orders"]
            for entry in source["modifiers"]
            if entry["type"] == "ORDER_MODIFIER"
        ),
        [],
    )
    if not orders:
        return False, None
    names = [name.lower() for name in schema.names]
    select_list = source.get("select_list", [])
    projections = (
        [expression_identity(entry) for entry in select_list]
        if len(select_list) == len(schema)
        and all(entry["class"] != "STAR" for entry in select_list)
        else []
    )
    aliases = [entry["alias"].lower() for entry in select_list] if projections else []
    for order in orders:
        expression = order["expression"]
        value = expression.get("value", {})
        parts = expression.get("column_names", [])
        if (
            value.get("type", {}).get("id") in ("INTEGER", "BIGINT")
            and not value["is_null"]
        ):
            position = value["value"]
            if not 1 <= position <= len(schema):
                raise ValueError(f"ORDER BY position {position} is outside the result")
            order["expression"] = output_column(position - 1)
        elif len(parts) == 1 and parts[0].lower() in aliases:
            # A bare sort name can resolve to a SELECT alias before an input
            # column with the same name.
            name = parts[0].lower()
            if names.count(name) != 1:
                raise ValueError(f"ORDER BY alias {name} is ambiguous")
            order["expression"] = output_column(aliases.index(name))
        else:
            order["expression"] = bind_order_expression(expression, projections, names)
        if order["null_order"] == "ORDER_DEFAULT":
            order["null_order"] = (
                "NULLS FIRST" if order["type"] == "DESCENDING" else "NULLS LAST"
            )

    # A dense rank assigns equal sort keys the same rank. Check ranks in wire
    # order so peers can appear in either order, even when their rows differ.
    rank_query = parse_sql(
        connection,
        "SELECT dense_rank() OVER (ORDER BY 1) FROM wire_rows ORDER BY wire_position",
    )
    rank_query["statements"][0]["node"]["select_list"][0]["orders"] = orders
    rank_sql = connection.execute(
        "SELECT json_deserialize_sql(?)", [json.dumps(rank_query)]
    ).fetchone()[0]
    table = pa.table(
        {
            **{
                f"wire_{index}": pa.array((row[index] for row in rows), type=field.type)
                for index, field in enumerate(schema)
            },
            "wire_position": pa.array(range(len(rows)), type=pa.int64()),
        }
    )
    connection.register("wire_rows", table)
    try:
        ranks = connection.execute(rank_sql).fetchall()
    finally:
        connection.unregister("wire_rows")
    for position, (left, right) in enumerate(pairwise(ranks), start=1):
        if left > right:
            return (
                True,
                f"ORDER BY inversion between rows {position} and {position + 1}",
            )
    return True, None


def verify_query(connection, fixture, query, record, tolerances):
    sql_bytes = (fixture / query["sql"]).read_bytes()
    if hashlib.sha256(sql_bytes).hexdigest() != query["sql_sha256"]:
        raise ValueError("fixture SQL digest differs from the manifest")
    with pa.ipc.open_file(str(fixture / query["answer"])) as reader:
        schema = reader.schema
        if record["columns"] != len(schema):
            raise ValueError(
                f"expected {len(schema)} result columns, received {record['columns']}"
            )
        rows = [
            tuple(
                parse_value(value, field.type)
                for value, field in zip(row, schema, strict=True)
            )
            for row in record["rows"]
        ]
        row_error = compare_answer(reader, rows, *tolerances)
    try:
        ordered, order_error = check_order(
            connection, sql_bytes.decode("utf-8"), schema, rows
        )
    except (ValueError, duckdb.Error) as exc:
        return row_error, True, f"cannot verify ORDER BY: {exc}"
    return row_error, ordered, order_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "results", type=Path, help="JSONL output from pinned_external_full_tpc_sql"
    )
    parser.add_argument("--float-relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--float-absolute-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    tolerances = (args.float_relative_tolerance, args.float_absolute_tolerance)
    if any(value < 0 or not math.isfinite(value) for value in tolerances):
        parser.error("float tolerances must be finite and nonnegative")
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    with args.results.open(encoding="utf-8") as output:
        records = [json.loads(line) for line in output]
    suite = manifest["suite"]
    queries = manifest["queries"]
    expected_count = {"tpch": 22, "tpcds": 99}[suite]
    expected_ids = list(range(1, expected_count + 1))
    if [query["id"] for query in queries] != expected_ids:
        parser.error("unexpected fixture query inventory")
    if [record["id"] for record in records] != expected_ids:
        parser.error("unexpected result query inventory")

    rows_passed = order_count = order_passed = 0
    with duckdb.connect(
        config={"threads": 1, "enable_external_access": False}
    ) as connection:
        for query, record in zip(queries, records, strict=True):
            error = record.get("error")
            if record["suite"] != suite or record["sql_sha256"] != query["sql_sha256"]:
                error = "query identity differs from the fixture"
            if error is not None:
                print(f"{suite} Q{query['id']:02d}: {error}")
                continue
            try:
                row_error, ordered, order_error = verify_query(
                    connection, args.fixture, query, record, tolerances
                )
            except (ValueError, InvalidOperation, duckdb.Error) as exc:
                print(f"{suite} Q{query['id']:02d}: {exc}")
                continue
            rows_passed += row_error is None
            order_count += ordered
            order_passed += ordered and order_error is None
            for error in (row_error, order_error):
                if error is not None:
                    print(f"{suite} Q{query['id']:02d}: {error}")
    print(
        f"{suite}: {rows_passed}/{len(queries)} typed row multisets match "
        f"(float rel_tol={tolerances[0]:g}, abs_tol={tolerances[1]:g})"
    )
    print(f"{suite}: {order_passed}/{order_count} checked ORDER BY clauses match")
    return int(rows_passed != len(queries) or order_passed != order_count)


if __name__ == "__main__":
    raise SystemExit(main())
