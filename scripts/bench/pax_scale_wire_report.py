#!/usr/bin/env python3
"""Join a PAX scale ledger (--wire-markers) to a dedicated Azurite access log."""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from urllib.parse import parse_qs, unquote, urlsplit

from pax_scale_cost import RANGE_CAPS_MIB, SCALE_ROWS


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
            continue
        if active is None:
            continue
        if verb not in {"GET", "HEAD"}:
            raise ValueError(
                f"unexpected {verb} inside read-only query window {active}"
            )
        counts = requests[active]
        if verb == "GET" and parse_qs(parsed.query).get("comp") == ["list"]:
            counts["list"] += 1
        else:
            counts[verb.lower()] += 1
            if verb == "GET" and status in {"200", "206"}:
                if length == "-":
                    raise ValueError(f"missing object response length for {url}")
                counts["object_bytes"] += int(length)
                if status == "206":
                    counts["partial_get"] += 1
        if int(status) >= 400:
            counts["errors"] += 1
    if ended != set(windows) or active is not None:
        raise ValueError(
            f"incomplete access log: {len(ended)} of {len(windows)} windows"
        )
    for key, row in windows.items():
        row["wire"] = dict(requests[key])


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
    summary = []
    for (n, route, cap, budget, phase), rs in sorted(groups.items()):
        wire = {
            key: round(mean(r["wire"].get(key, 0) for r in rs), 3)
            for key in ("get", "partial_get", "list", "head", "errors", "object_bytes")
        }
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
                wire=wire,
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
