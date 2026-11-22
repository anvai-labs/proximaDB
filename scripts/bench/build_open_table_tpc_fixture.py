#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build pinned open-format inputs and DuckDB reference answers for TPC SQL."""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import deltalake
import duckdb
import pyarrow as pa
import pyiceberg
from pyiceberg.catalog import load_catalog

SUITES = {"tpch": ("dbgen", 8, 22), "tpcds": ("dsdgen", 24, 99)}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def write_ndjson_manifest(path):
    """Pin contiguous, newline-aligned ranges for the ProximaDB NDJSON reader."""
    limit = 8 * 1024 * 1024
    ranges = []
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(limit):
            end = chunk.rfind(b"\n") + 1
            if not end:
                raise ValueError(
                    f"{path}: NDJSON record exceeds 8 MiB or lacks a newline"
                )
            size += end
            ranges.append(
                {
                    "end": size,
                    "sha256": hashlib.sha256(memoryview(chunk)[:end]).hexdigest(),
                }
            )
            stream.seek(size)
    manifest = json.dumps({"version": 1, "size": size, "ranges": ranges}).encode(
        "utf-8"
    )
    if len(manifest) > 1024 * 1024:
        raise ValueError(f"{path}: NDJSON range manifest exceeds 1 MiB")
    path.with_suffix(".jsonl.manifest.json").write_bytes(manifest)
    return hashlib.sha256(manifest).hexdigest()


def export_table(connection, catalog, out, name, batch_rows):
    identifier = name.replace('"', '""')
    table_sql = f'SELECT * FROM "{identifier}"'
    reader = connection.execute(table_sql).fetch_record_batch(batch_rows)
    schema = reader.schema
    delta_path = out / "delta" / name
    deltalake.write_deltalake(str(delta_path), reader)
    delta = deltalake.DeltaTable(str(delta_path))

    iceberg_path = out / "iceberg" / name
    iceberg = catalog.create_table(
        ("benchmark", name),
        schema=schema,
        location=iceberg_path.as_uri(),
        properties={"format-version": "2"},
    )
    row_count = 0
    for batch in connection.execute(table_sql).fetch_record_batch(batch_rows):
        iceberg.append(pa.Table.from_batches([batch]))
        row_count += batch.num_rows
    snapshot = iceberg.current_snapshot()

    json_path = out / "json" / f"{name}.jsonl"
    connection.execute(f"COPY ({table_sql}) TO {sql_string(json_path)} (FORMAT JSON)")
    columns = ", ".join(
        f"{sql_string(column)}: {sql_string(data_type)}"
        for column, data_type, *_ in connection.execute(
            f"DESCRIBE {table_sql}"
        ).fetchall()
    )
    rows_json = connection.execute(
        f"SELECT count(*) FROM read_json({sql_string(json_path)}, "
        f"columns = {{{columns}}}, format = 'newline_delimited')"
    ).fetchone()[0]

    if not rows_json == delta.count() == iceberg.scan().count() == row_count:
        raise RuntimeError(
            f"{name}: NDJSON, Delta or Iceberg read-back row count differs from source"
        )
    return {
        "name": name,
        "rows": row_count,
        "schema_ipc_base64": base64.b64encode(schema.serialize()).decode("ascii"),
        "json": {
            "path": str(json_path.relative_to(out)),
            "manifest_sha256": write_ndjson_manifest(json_path),
        },
        "delta": {"path": str(delta_path.relative_to(out)), "version": delta.version()},
        "iceberg": {
            "location": iceberg.location(),
            "metadata_location": iceberg.metadata_location,
            "snapshot_id": None if snapshot is None else snapshot.snapshot_id,
        },
    }


def export_queries(connection, suite, query_count, out, batch_rows):
    queries = connection.execute(
        f"SELECT query_nr, query FROM {suite}_queries() ORDER BY query_nr"
    ).fetchall()
    if [number for number, _ in queries] != list(range(1, query_count + 1)):
        raise RuntimeError(
            f"{suite}: unexpected SQL corpus, expected every ID 1–{query_count}"
        )
    manifest = []
    for number, sql in queries:
        if len(connection.extract_statements(sql)) != 1:
            raise RuntimeError(f"{suite} Q{number}: expected one statement")
        query_path = out / "queries" / f"q{number:02d}.sql"
        sql_bytes = sql.encode("utf-8")
        query_path.write_bytes(sql_bytes)
        answer_path = out / "answers" / f"q{number:02d}.arrow"
        answer = connection.execute(sql).fetch_record_batch(batch_rows)
        rows = 0
        with pa.ipc.new_file(str(answer_path), answer.schema) as writer:
            for batch in answer:
                writer.write_batch(batch)
                rows += batch.num_rows
        manifest.append(
            {
                "id": number,
                "sql": str(query_path.relative_to(out)),
                "sql_sha256": hashlib.sha256(sql_bytes).hexdigest(),
                "answer": str(answer_path.relative_to(out)),
                "answer_rows": rows,
            }
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument(
        "output", type=Path, help="New output directory at its final location"
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-rows", type=int, default=262_144)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GiB", help="DuckDB memory limit")
    parser.add_argument(
        "--extension", type=Path, help="Local signed DuckDB extension file"
    )
    args = parser.parse_args()
    generate, table_count, query_count = SUITES[args.suite]
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be positive and finite")
    if args.suite == "tpcds" and args.scale < 1:
        parser.error("TPC-DS fixture scale must be at least 1")
    if min(args.batch_rows, args.threads) < 1:
        parser.error("--batch-rows and --threads must be positive")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("json", "delta", "iceberg", "queries", "answers"):
        (out / directory).mkdir()
    catalog = load_catalog(
        "fixture", type="in-memory", warehouse=(out / "iceberg").as_uri()
    )
    catalog.create_namespace("benchmark")
    config = {"threads": args.threads, "memory_limit": args.memory_limit}
    with TemporaryDirectory(dir=out) as scratch:
        with duckdb.connect(
            str(Path(scratch) / "source.duckdb"), config=config
        ) as source:
            source.load_extension(
                str(args.extension.resolve()) if args.extension else args.suite
            )
            source.execute(f"CALL {generate}(sf = {args.scale})")
            tables = [name for (name,) in source.execute("SHOW TABLES").fetchall()]
            if len(tables) != table_count:
                raise RuntimeError(
                    f"{args.suite}: expected {table_count} tables, found {tables}"
                )
            manifest = {
                "suite": args.suite,
                "scale": args.scale,
                "duckdb": duckdb.__version__,
                "pyarrow": pa.__version__,
                "deltalake": deltalake.__version__,
                "pyiceberg": pyiceberg.__version__,
                "tables": [
                    export_table(source, catalog, out, name, args.batch_rows)
                    for name in tables
                ],
                "queries": export_queries(
                    source, args.suite, query_count, out, args.batch_rows
                ),
            }
    # The manifest is the completion marker. Iceberg references the final absolute location.
    pending = out / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pending.replace(out / "manifest.json")


if __name__ == "__main__":
    main()
