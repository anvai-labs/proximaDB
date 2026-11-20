#!/usr/bin/env python3
"""Bench driver: external parquet tables on S3-compatible storage over pgwire.

TD-OLAP-4 external-table protocol (see docs/12-design/HOW_TO_FILE_TD_AND_ADR.adoc
for the TD; numbers in the PR discussion). Times the native route (proximaDB's
ObjectStoreParquetTable scan + DataFusion compute) and the DuckDB wrapper
(PROXIMADB_DUCKDB_ROUTE=1) over pgwire; row counts are asserted against
duckdb goldens externally.

Usage:
  # start a release-server with the route env first, e.g.
  #   AWS_ENDPOINT=http://localhost:9000 ... PROXIMADB_EXTERNAL_TABLE_ROOTS=s3://warehouse \
  #     ./target/release-server/proximadb-server -c bench-config.toml
  python3 scripts/bench/external_parquet_s3_bench.py --route native --queries 1 2 3 --runs 3
Requires the SeaweedFS smoke fixture (bucket warehouse, path-style) on :9000
and the server's pgwire on 15433 (edit PG below to taste).
"""
import argparse
import json
import statistics
import time

import psycopg2

PG = "host=127.0.0.1 port=15433 dbname=proximadb user=proximadb"

LOCS = {
    "store_returns": "01a1f6c8-b15f-7799-9d52-73f57c309323",
    "date_dim": "01a1f6c8-af13-74a9-bc5a-96fb7846dd60",
    "store": "01a1f6c8-b123-764a-82cb-0e2b9b5541a2",
    "customer": "01a1f6c8-aba8-72fc-8ebc-d6531a708138",
    "store_sales": "01a1f6c8-e619-77b6-84ff-0cffc3b86af7",
    "item": "01a1f6c8-affe-74a9-a79b-09a0a302e870",
    "catalog_sales": "01a1f6c8-92f2-7275-9b16-36b22be8bb48",
    "web_sales": "01a1f6c8-d60c-734c-9781-300bd536aecd",
}

DDL = {
    "store_returns": "(sr_returned_date_sk BIGINT, sr_item_sk BIGINT, sr_customer_sk BIGINT, sr_store_sk BIGINT, sr_return_amt DECIMAL(7,2))",
    "date_dim": "(d_date_sk BIGINT, d_year INT, d_moy INT, d_week_seq INT, d_day_name VARCHAR)",
    "store": "(s_store_sk BIGINT, s_state VARCHAR)",
    "customer": "(c_customer_sk BIGINT, c_customer_id VARCHAR)",
    "store_sales": "(ss_sold_date_sk BIGINT, ss_item_sk BIGINT, ss_ext_discount_amt DECIMAL(7,2))",
    "item": "(i_item_sk BIGINT, i_brand_id INT, i_brand VARCHAR, i_manufact_id INT)",
    "catalog_sales": "(cs_sold_date_sk BIGINT, cs_ext_sales_price DECIMAL(7,2))",
    "web_sales": "(ws_sold_date_sk BIGINT, ws_ext_sales_price DECIMAL(7,2))",
}


def register(cur):
    for name in LOCS:
        cur.execute(f"DROP TABLE IF EXISTS {name}")
    for name, uuid in LOCS.items():
        loc = f"s3://warehouse/{uuid}/data"
        cur.execute(
            f"CREATE TABLE {name} {DDL[name]} "
            f"WITH (format='parquet', external_location='{loc}', authority='external')"
        )
        print(f"registered {name}")


def load_query(qn):
    sql = open(f"/tmp/tpcds-queries/query{qn}.sql").read()
    return "\n".join(
        l for l in sql.splitlines() if not l.lstrip().startswith("--")
    )


def run_query(cur, sql):
    t0 = time.perf_counter()
    cur.execute(sql)
    rows = cur.fetchall()
    return time.perf_counter() - t0, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["native", "duckdb"], required=True)
    ap.add_argument("--queries", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    conn = psycopg2.connect(PG)
    conn.autocommit = True
    cur = conn.cursor()
    register(cur)

    results = {}
    for qn in args.queries:
        sql = load_query(qn)
        try:
            el, rows = run_query(cur, sql)
            print(f"q{qn} warmup: rows={rows} elapsed={el:.2f}s")
            times = []
            for _ in range(args.runs):
                el, rows = run_query(cur, sql)
                times.append(el)
                print(f"q{qn} run: rows={rows} elapsed={el:.2f}s")
            results[f"q{qn}"] = {
                "rows": rows,
                "median_s": statistics.median(times),
                "all_s": [round(t, 3) for t in times],
            }
        except Exception as e:
            results[f"q{qn}"] = {"error": str(e)[:300]}
            print(f"q{qn} ERROR: {str(e)[:300]}")

    print(json.dumps({"route": args.route, **results}, indent=2))


if __name__ == "__main__":
    main()
