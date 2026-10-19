#!/usr/bin/env python3
"""External Iceberg read-back gate (ADR-097 D3, TD-SEAM-1).

Reads a ProximaDB-published Iceberg table with engines that are NOT ProximaDB and
compares every row against the rows ProximaDB itself returned over pgwire:

  * pyiceberg-rest   — PyIceberg through ProximaDB's Iceberg REST catalog
  * pyiceberg-static — PyIceberg straight from the metadata.json file
  * duckdb           — DuckDB `iceberg_scan` over the metadata.json file

Values are compared as text (the pgwire text protocol is the reference), rows in
order of the first column. Driven by tests/iceberg_interop_e2e.rs:

  python3 scripts/iceberg_readback.py --rest-uri http://127.0.0.1:PORT/iceberg \
      --namespace default --table t --metadata-location file:///.../v...json \
      --expected expected.json [--require pyiceberg-rest,pyiceberg-static,duckdb]

A reader that is not installed (module or DuckDB extension missing) is SKIPPED
unless named in --require; a reader that runs and errors always FAILS. Exit 0 =
at least one reader ran and every reader that ran matched; 2 = none ran.
"""

import argparse
import datetime
import decimal
import json
import sys


def text(v):
    """Render a value the way the pgwire text protocol does."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat(sep=" ") if isinstance(v, datetime.datetime) else v.isoformat()
    if isinstance(v, decimal.Decimal):
        return format(v, "f")
    return str(v)


def first_key_sort(rows):
    def key(r):
        k = r[0]
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(rows, key=key)


class Unavailable(Exception):
    """The reader is not installed here (module or DuckDB extension missing)."""


def _require_module(name):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise Unavailable(f"{name} not installed: {e}") from e


def read_pyiceberg_rest(args, columns):
    _require_module("pyiceberg")
    from pyiceberg.catalog import load_catalog

    props = {"type": "rest", "uri": args.rest_uri}
    if args.tenant:
        # The REST catalog resolves within the request tenant (X-Tenant-ID).
        props["header.X-Tenant-ID"] = args.tenant
    catalog = load_catalog("proximadb", **props)
    table = catalog.load_table((*args.namespace.split("."), args.table))
    return table.scan(selected_fields=tuple(columns)).to_arrow().to_pylist(), table.metadata_location


def read_pyiceberg_static(args, columns):
    _require_module("pyiceberg")
    from pyiceberg.table import StaticTable

    table = StaticTable.from_metadata(args.metadata_location)
    return table.scan(selected_fields=tuple(columns)).to_arrow().to_pylist(), None


def read_duckdb(args, columns):
    duckdb = _require_module("duckdb")
    con = duckdb.connect()
    try:
        con.sql("INSTALL iceberg")
        con.sql("LOAD iceberg")
    except duckdb.Error as e:
        raise Unavailable(f"DuckDB iceberg extension unavailable: {e}") from e
    path = args.metadata_location.removeprefix("file://")
    cols = ", ".join(f'"{c}"' for c in columns)
    rel = con.sql(f"SELECT {cols} FROM iceberg_scan('{path}')")
    return [dict(zip(columns, r)) for r in rel.fetchall()], None


READERS = {
    "pyiceberg-rest": read_pyiceberg_rest,
    "pyiceberg-static": read_pyiceberg_static,
    "duckdb": read_duckdb,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rest-uri", required=True)
    p.add_argument("--namespace", required=True, help="dot-separated, e.g. proximadb.default")
    p.add_argument("--tenant", default="", help="X-Tenant-ID for the REST catalog")
    p.add_argument("--table", required=True)
    p.add_argument("--metadata-location", required=True)
    p.add_argument("--expected", required=True, help="JSON {columns: [...], rows: [[text|null]]}")
    p.add_argument("--require", default="", help="comma-separated engines that must run")
    args = p.parse_args()

    expected = json.load(open(args.expected))
    columns = expected["columns"]
    want = first_key_sort(expected["rows"])
    required = {e for e in args.require.split(",") if e}

    report, failed = {}, False
    for name, reader in READERS.items():
        try:
            rows, served_location = reader(args, columns)
        except Unavailable as e:
            # Only a reader that is not installed may be skipped, and only when
            # it is not required.
            status = "ERROR" if name in required else "SKIP"
            report[name] = {"status": status, "reason": str(e)}
            failed |= name in required
            continue
        except Exception as e:  # noqa: BLE001 — a reader that ran and failed is a failure
            report[name] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
            failed = True
            continue
        got = first_key_sort([[text(r[c]) for c in columns] for r in rows])
        entry = {"status": "OK" if got == want else "MISMATCH", "rows": len(got)}
        if served_location is not None:
            entry["metadata_location"] = served_location
            if served_location != args.metadata_location:
                entry["status"] = "MISMATCH"
                entry["error"] = f"REST served {served_location}, expected {args.metadata_location}"
        if got != want:
            entry["got"], entry["want"] = got, want
        if entry["status"] != "OK":
            failed = True
        report[name] = entry

    ran = [n for n, r in report.items() if r["status"] != "SKIP"]
    print(json.dumps(report, indent=2))
    if not ran:
        print("no reader ran", file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
