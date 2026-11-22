#!/usr/bin/env python3
"""Join a PAX scale ledger (--wire-markers) to a dedicated Azurite access log."""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from urllib.parse import parse_qs, unquote, urlsplit

from pax_scale_cost import RANGE_CAPS_MIB, SCALE_ROWS

RATE_CARD = (
    Path(__file__).resolve().parents[2]
    / "bench_results/pax_sift_scale_20261122/request_rates.csv"
)
DFS_READ_UNIT = 4 * 1024 * 1024


def load_rate_card(path=RATE_CARD):
    allowed = {
        "get": {"get", "dfs_read_units"},
        "list": {"list"},
        "head": {"head"},
        "retrieval": {"object_bytes"},
    }
    with path.open(newline="") as card:
        rates = list(csv.DictReader(card))
    operations = defaultdict(set)
    for rate in rates:
        profile, op = rate["profile"], rate["operation"]
        if op in operations[profile] or rate["counter"] not in allowed.get(op, set()):
            raise ValueError(
                f"duplicate or unknown rate component: {profile}/{op}/{rate['counter']}"
            )
        usd = Decimal(rate["usd"])
        if not usd.is_finite() or usd < 0 or int(rate["per_units"]) <= 0:
            raise ValueError(f"invalid rate: {profile}/{op}")
        operations[profile].add(op)
    if not rates or any(ops != allowed.keys() for ops in operations.values()):
        raise ValueError("each profile needs a GET, LIST, HEAD and retrieval rate")
    return rates


def estimate_request_cost(rows, rates):
    # Quote the complete request mix only when all observed responses succeeded.
    # An S3/DFS charge for a failed Blob request cannot be inferred from its verb.
    quoted = all(not r["wire"].get("unpriced_responses", 0) for r in rows)
    costs = {"cost_status": "modeled" if quoted else "unpriced_response"}
    totals = {}
    for rate in rates:
        profile = rate["profile"]
        units = sum(r["wire"].get(rate["counter"], 0) for r in rows)
        price = (
            Decimal(rate["usd"])
            * 1_000_000
            * units
            / (int(rate["per_units"]) * len(rows))
        )
        totals[profile] = totals.get(profile, Decimal(0)) + price
        costs[f"{profile}_{rate['operation']}_usd_per_m_queries"] = (
            float(round(price, 6)) if quoted else None
        )
    for profile, total in totals.items():
        costs[f"{profile}_total_usd_per_m_queries"] = (
            float(round(total, 6)) if quoted else None
        )
    return costs


def read_windows(rows, lines):
    # Azurite paths start with /account/container/. Client URLs start with az://container/.
    windows = {}
    for row in rows:
        uri = urlsplit(row["marker"])
        key = f"/{uri.netloc}{uri.path}"
        if key in windows:
            raise ValueError(f"duplicate query window {key}")
        windows[key] = row
    started, ended = set(), set()
    active = None
    requests = defaultdict(Counter)
    for line in lines:
        match = re.search(r'"(\w+) (\S+) HTTP/[\d.]+" (\d{3}) (\d+|-)', line)
        if not match:
            if active is not None and " HTTP/" in line:
                raise ValueError(f"unrecognized access log record in {active}")
            continue
        verb, url, status, length = match.groups()
        parsed = urlsplit(url)
        path = "/" + unquote(parsed.path).lstrip("/").partition("/")[2]
        if "/_meter/" in path:
            key, _, edge = path.rpartition(".")
            if key in windows and verb == "PUT" and status == "201":
                if edge == "start" and active is None and key not in started:
                    active = key
                    started.add(key)
                elif edge == "end" and key == active:
                    ended.add(key)
                    active = None
                else:
                    raise ValueError(f"overlapping or invalid marker {path}")
            elif active is not None:
                raise ValueError(f"unrelated or failed marker inside {active}: {path}")
            continue
        if active is None:
            continue
        if verb not in {"GET", "HEAD"}:
            raise ValueError(
                f"unexpected {verb} inside read-only query window {active}"
            )
        counts = requests[active]
        query = parse_qs(parsed.query)
        is_list = verb == "GET" and query.get("comp") == ["list"]
        target = path + "/" + query.get("prefix", [""])[0] if is_list else path
        fixture = active.rsplit("/_meter/", 1)[0] + "/"
        if not target.startswith(fixture):
            raise ValueError(f"unrelated request inside {active}: {url}")
        if is_list:
            counts["list"] += 1
        else:
            counts[verb.lower()] += 1
            if verb == "GET" and status in {"200", "206"}:
                if length == "-":
                    raise ValueError(f"missing object response length for {url}")
                response_bytes = int(length)
                counts["object_bytes"] += response_bytes
                # DFS pricing projection, calculated per successful data response.
                counts["dfs_read_units"] += max(
                    1, (response_bytes + DFS_READ_UNIT - 1) // DFS_READ_UNIT
                )
                if status == "206":
                    counts["partial_get"] += 1
                    counts["max_range_bytes"] = max(
                        counts["max_range_bytes"], int(length)
                    )
        if status not in {"200", "206"}:
            counts["unpriced_responses"] += 1
        if int(status) >= 400:
            counts["errors"] += 1
    if ended != set(windows) or active is not None:
        raise ValueError(
            f"incomplete access log: {len(ended)} of {len(windows)} windows"
        )
    for key, row in windows.items():
        counts = requests[key]
        # This adapter meters physical reads. Wire attempts can exceed the
        # client counters (e.g. a retry). A deficit needs an investigation
        # before this log can serve as billing evidence.
        for app, wire in (
            ("get", "get"),
            ("list", "list"),
            ("head", "head"),
            ("bytes", "object_bytes"),
        ):
            if counts[wire] < row[app]:
                raise ValueError(
                    f"incomplete wire {wire} in {key}: {counts[wire]} < {row[app]}"
                )
        row["wire"] = dict(counts)


def validate_matrix(fixtures, rows, sizes, caps, queries):
    if sorted(f["rows"] for f in fixtures) != sorted(sizes):
        raise ValueError("fixture inventory differs from requested sizes")
    expected = set()
    for n in sizes:
        for q in range(queries):
            for route, route_caps in (("exact", caps[:1]), ("ann", caps)):
                for cap in route_caps:
                    for phase in ("cold", "l1_repeat", "l2_reopen"):
                        expected.add((n, q, route, cap, 0, phase))
        for budget in [0, *caps]:
            expected.add((n, 0, "adaptive", caps[0], budget, "policy"))
    seen = {
        (r["rows"], r["query"], r["route"], r["cap_mib"], r["budget_mib"], r["phase"])
        for r in rows
    }
    if seen != expected or len(seen) != len(rows):
        raise ValueError(
            f"incomplete or duplicate matrix: missing={len(expected - seen)}, "
            f"extra={len(seen - expected)}, duplicates={len(rows) - len(seen)}"
        )


def summarize(raw, log, sizes, caps, queries):
    records = [json.loads(line) for line in raw if line.strip()]
    fixtures = [r for r in records if "segment_bytes" in r]
    rows = [r for r in records if "phase" in r]
    if not rows or not fixtures:
        raise ValueError("empty query ledger or fixture inventory")
    validate_matrix(fixtures, rows, sizes, caps, queries)
    read_windows(rows, log)
    baseline = {
        (r["rows"], r["query"]): set(r["ids"])
        for r in rows
        if r["route"] == "ann" and r["phase"] == "cold" and r["cap_mib"] == caps[0]
    }
    groups = defaultdict(list)
    for r in rows:
        groups[r["rows"], r["route"], r["cap_mib"], r["budget_mib"], r["phase"]].append(
            r
        )
    rates = load_rate_card()
    summary = []
    for (n, route, cap, budget, phase), rs in sorted(groups.items()):
        wire = {
            key: round(mean(r["wire"].get(key, 0) for r in rs), 3)
            for key in ("get", "partial_get", "list", "head", "errors", "object_bytes")
        }
        wire["max_range_bytes"] = max(r["wire"].get("max_range_bytes", 0) for r in rs)
        stats = {
            key: round(mean(r[key] for r in rs), 3)
            for key in (
                "get",
                "bytes",
                "footer_hit",
                "footer_miss",
                "l1_hit",
                "l1_miss",
                "l2_hit",
                "l2_miss",
                "cells_probed",
                "rows_probed",
                "whole_region_fallback",
            )
        }
        summary.append(
            dict(
                rows=n,
                route=route,
                cap_mib=cap,
                budget_mib=budget,
                phase=phase,
                actual=sorted({r["actual"] for r in rs}),
                queries=len(rs),
                recall_at_10=mean(r["overlap"] / 10 for r in rs),
                median_ms=round(median(r["us"] for r in rs) / 1000, 3),
                ann_top10_set_differences=(
                    sum(set(r["ids"]) != baseline[n, r["query"]] for r in rs)
                    if route == "ann"
                    else None
                ),
                modeled_dfs_read_units=round(
                    mean(r["wire"].get("dfs_read_units", 0) for r in rs), 6
                ),
                wire=wire,
                **estimate_request_cost(rs, rates),
                **stats,
            )
        )
    return {"fixtures": fixtures, "groups": summary}


def open_text(path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("azurite_log", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=SCALE_ROWS)
    parser.add_argument("--caps-mib", type=int, nargs="+", default=RANGE_CAPS_MIB)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--csv", action="store_true", help="write one row per group")
    args = parser.parse_args()
    with open_text(args.raw) as raw, open_text(args.azurite_log) as log:
        report = summarize(raw, log, args.rows, args.caps_mib, args.queries)
    if args.csv:
        flat = [
            {
                **{k: v for k, v in r.items() if k not in {"wire", "actual"}},
                "actual": ",".join(r["actual"]),
                **{f"wire_{k}": v for k, v in r["wire"].items()},
            }
            for r in report["groups"]
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=flat[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
