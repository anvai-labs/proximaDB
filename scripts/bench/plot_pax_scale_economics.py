#!/usr/bin/env python3
"""Render cost-ledger figures from the reconciled query groups."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pax_scale_wire_report import (
    DFS_READ_UNIT,
    RATE_CARD,
    estimate_request_cost,
    load_rate_card,
)

ROOT = Path(__file__).resolve().parents[2]
HOT_LABEL = "S3 Standard / Azure Blob Hot (abfs://) / GCS Standard"
COLORS = ["#086b71", "#be5d0b", "#5756a4", "#a13560"]
QPS = 100


def ops(r):
    return sum(float(r["wire_" + k]) for k in ("get", "head", "list"))


def size_label(n):
    return "1M" if n == 1_000_000 else f"{n // 1000}K"


def save(fig, dest, name):
    path = dest / f"pax-{name}.svg"
    fig.savefig(path, metadata={"Date": None}, bbox_inches="tight")
    # Matplotlib leaves trailing spaces in multiline SVG paths.
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def recall_plot(anns, exact, dest):
    fig, ax = plt.subplots(figsize=(8.7, 4.8), layout="constrained")
    for r in anns:
        x, y = float(r["recall_at_10"]), QPS * ops(r)
        ax.scatter(x, y, s=60, color=COLORS[0], zorder=3)
        ax.annotate(
            f"{size_label(int(r['rows']))} · {x:.2f}"
            if int(r["rows"]) in (33_000, 333_000)
            else size_label(int(r["rows"])),
            (x, y),
            xytext=(7, 3),
            textcoords="offset points",
        )
    # The six forced-exact controls coincide on this collection shape.
    points = {(float(r["recall_at_10"]), QPS * ops(r)) for r in exact}
    if len(points) != 1:
        raise ValueError("exact control points no longer coincide")
    pt = points.pop()
    ax.scatter(*pt, marker="D", s=55, color=COLORS[1], zorder=3)
    ax.annotate(
        "Exact · all sizes", pt, xytext=(-9, -2), ha="right", textcoords="offset points"
    )
    ax.set(
        xlim=(0.953, 1.008),
        ylim=(0, 2100),
        xlabel="Recall@10 against exact top ten",
        ylabel="Object API operations / second",
        title="Cold search · object IOPS versus recall",
    )
    ax.set_xticks([0.96, 0.97, 0.98, 0.99, 1.0])
    fig.text(
        0.5,
        -0.02,
        "100 searches/s assumed   ·   ANN labels give corpus size   ·   128d SIFT, ten queries per size",
        ha="center",
        size=9,
    )
    save(fig, dest, "iops-recall")


def hot_costs(rows):
    """Draw the shared price only while the three serving endpoints agree."""

    def series(key):
        return [float(r[key + "_total_usd_per_m_queries"]) for r in rows]

    flat = series("s3")
    if any(series(k) != flat for k in ("blob", "gcs")):
        raise ValueError("hot endpoint prices diverged; show individual curves")
    return flat


def tier_plot(r, dest):
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.1), layout="constrained")
    costs = [float(r["s3_" + op + "_usd_per_m_queries"]) for op in ("get", "list")]
    counts = [QPS * float(r["wire_" + op]) for op in ("get", "list")]
    total = hot_costs([r])[0]
    for ax, values, title in zip(
        axes,
        [counts, costs],
        ["Object operations / second", "USD / million searches"],
        strict=True,
    ):
        bars = ax.bar(["GET", "LIST"], values, color=COLORS[:2], width=0.55)
        ax.bar_label(
            bars, labels=[f"{v:,.0f}" if ax is axes[0] else f"${v:.2f}" for v in values]
        )
        ax.set(title=title, ylim=(0, max(values) * 1.22))
    fig.suptitle(
        f"1M vectors · {sum(counts):,.0f} object IOPS · ${total:.2f}/M searches",
        weight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        HOT_LABEL + " · cold ANN · 100 searches/s assumed · 8 MiB merge cap",
        ha="center",
        size=9,
    )
    save(fig, dest, "iops-tier-cost")


def scale_plot(anns, exact, dest):
    fig, ax = plt.subplots(figsize=(8.3, 4.2), layout="constrained")
    xs = [int(r["rows"]) for r in anns]
    for i, (route, rows) in enumerate((("ANN", anns), ("Exact · one full GET", exact))):
        values = hot_costs(rows)
        ax.plot(xs, values, marker=("o", "s")[i], color=COLORS[i], label=route)
        ax.annotate(
            f"${values[-1]:.2f}",
            (xs[-1], values[-1]),
            xytext=(-6, 7),
            ha="right",
            textcoords="offset points",
        )
    ax.set(
        xscale="log",
        xlabel="Vectors in one collection",
        ylabel="USD / million searches",
        title="Cold ANN and exact read cost · " + HOT_LABEL,
    )
    ax.set_xticks(xs, [size_label(n) for n in xs])
    ax.margins(y=0.18)
    ax.legend()
    fig.text(
        0.5,
        -0.10,
        "Hot/Standard minimizes modeled read cost for active PAX segments.\n128d SIFT · 8 MiB ANN merge cap · GET and LIST priced from the Blob trace",
        ha="center",
        size=9,
    )
    save(fig, dest, "cost-corpus")


DIMENSIONS = tuple(range(128, 1537, 128))


def dimension_model(exact, rates):
    """Keep 128d file overhead fixed; grow one f32, SQ8 and 1-bit code per coordinate."""
    result = []
    for e in exact:
        n = int(e["rows"])
        base = int(float(e["wire_object_bytes"]))
        for d in DIMENSIONS:
            body = base + n * (d - 128) * 41 // 8
            for plan, gets in (
                ("whole", 1),
                ("8MiB_ranges", (body + (8 << 20) - 1) // (8 << 20)),
            ):
                # 8 MiB boundaries align with the 4 MiB DFS meter.
                dfs_reads = (body + DFS_READ_UNIT - 1) // DFS_READ_UNIT
                wire = {
                    "get": gets,
                    "list": 1,
                    "head": 0,
                    "dfs_read_units": dfs_reads,
                    "object_bytes": body,
                }
                costs = estimate_request_cost([{"wire": wire}], rates)
                result.append(
                    dict(
                        rows=n,
                        dimensions=d,
                        plan=plan,
                        body_bytes=body,
                        wire_get=gets,
                        wire_list=1,
                        wire_head=0,
                        modeled_dfs_read_units=dfs_reads,
                        object_iops_at_100qps=QPS * (gets + 1),
                        **costs,
                    )
                )
    return result


def dimension_plots(model, dest):
    plans = {"whole": "Single GET", "8MiB_ranges": "8 MiB windows"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for i, (plan, label) in enumerate(plans.items()):
        rs = [r for r in model if r["rows"] == 1_000_000 and r["plan"] == plan]
        left.plot(
            DIMENSIONS,
            [r["object_iops_at_100qps"] for r in rs],
            marker=".",
            color=COLORS[i],
            label=label,
        )
        right.plot(DIMENSIONS, hot_costs(rs), color=COLORS[i], marker=".", label=label)
    left.set(
        title="HTTP IOPS at 100 searches/s", yscale="log", ylabel="Operations / second"
    )
    right.set(
        title="Read cost per million searches", yscale="log", ylabel="USD (log scale)"
    )
    for ax in (left, right):
        ax.set(xlabel="Embedding dimensions", xticks=DIMENSIONS)
        ax.tick_params(axis="x", labelrotation=32, labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(
        "1M vectors · two exact read plans on hot object storage", weight="bold"
    )
    fig.text(
        0.5,
        -0.025,
        HOT_LABEL + " · one LIST · both plans read identical data",
        ha="center",
        size=9,
    )
    save(fig, dest, "cost-iops-dimension")

    fig, ax = plt.subplots(figsize=(9.4, 4.9), layout="constrained")
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=128, vmax=1536)
    for d in DIMENSIONS:
        rows = [r for r in model if r["dimensions"] == d and r["plan"] == "8MiB_ranges"]
        xs = [r["rows"] for r in rows]
        ax.plot(xs, hot_costs(rows), marker=".", linewidth=1.3, color=cmap(norm(d)))
    whole = [r for r in model if r["plan"] == "whole"]
    totals = set(hot_costs(whole))
    if len(totals) != 1:
        raise ValueError("whole-read prices vary by dimension or corpus; split curves")
    price = totals.pop()
    ax.axhline(
        price,
        linestyle="--",
        color="#97420d",
        lw=2,
        label=f"Single GET · ${price:.2f} at every width",
    )
    ax.legend(loc="upper left")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Vectors in one collection",
        ylabel="USD / million searches (log scale)",
        title="Exact scan cost by corpus size and dimension · " + HOT_LABEL,
    )
    ax.set_xticks(xs, [size_label(n) for n in xs])
    ax.set_ylim(bottom=4.0)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        ticks=DIMENSIONS,
        label="Dimensions · 8 MiB range plan",
    )
    fig.text(
        0.5,
        -0.055,
        "One cold LIST per search. Both plans read the entire object. More 8 MiB GETs raise request cost.",
        ha="center",
        size=9,
    )
    save(fig, dest, "cost-corpus-dimension")


def cache_plot(groups, dest):
    rows = {
        r["phase"]: r
        for r in groups
        if r["rows"] == "1000000" and r["route"] == "ann" and r["cap_mib"] == "8"
    }
    fig, ax = plt.subplots(figsize=(8.3, 4.0), layout="constrained")
    cold = hot_costs([rows["cold"]])[0]
    for i, (phase, title) in enumerate(
        (("l1_repeat", "DRAM repeat"), ("l2_reopen", "Disk-cache reopen"))
    ):
        warm = hot_costs([rows[phase]])[0]
        ax.plot([0, 100], [warm, cold], color=COLORS[i], marker="o", label=title)
        ax.annotate(
            f"${warm:.2f}", (0, warm), xytext=(5, 7), textcoords="offset points"
        )
    ax.annotate(
        f"${cold:.2f}",
        (100, cold),
        xytext=(-5, 6),
        ha="right",
        textcoords="offset points",
    )
    ax.set(
        title="1M vectors · cost as the share of cold searches grows",
        xlabel="Searches starting cold (%)",
        ylabel="USD / million searches",
        xticks=[0, 25, 50, 75, 100],
        xlim=(-3, 104),
        ylim=(0, 14),
    )
    ax.legend(loc="upper left")
    fig.text(
        0.5,
        -0.05,
        HOT_LABEL + " · remaining searches repeat in DRAM or reopen from disk",
        ha="center",
        size=9,
    )
    save(fig, dest, "cost-cache-mix")


def publication_plot(dest, rates):
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    card = {
        r["operation"]: float(r["usd"]) * 1_000_000 / int(r["per_units"])
        for r in rates
        if r["profile"] == "s3"
    }
    # S3 Standard PUT and LIST share a class. Proposed pointer is a single GET.
    saved = card["list"] - card["get"]
    for i, writes in enumerate((1, 2)):
        write_cost = writes * card["list"]
        xs = [0, 1, 2, 3, 4, 5, 6]
        ax.plot(
            xs,
            [saved * x - write_cost for x in xs],
            color=COLORS[i],
            label=f"{writes} additional PUT(s) per publish",
        )
        knee = write_cost / saved
        ax.scatter(knee, 0, color=COLORS[i], zorder=3)
        ax.annotate(f"{knee:.2f}", (knee, 0), xytext=(4, 7), textcoords="offset points")
    ax.axhline(0, color="#36424a", lw=1)
    ax.set(
        xlabel="Cold discoveries per publish",
        ylabel="Net USD saved / million publishes",
        title="Would replacing a LIST with a versioned head pay for itself?",
    )
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        -0.03,
        "S3 Standard thought experiment   ·   one LIST replaced by one GET   ·   publish and recovery costs must still be measured",
        ha="center",
        size=9,
    )
    save(fig, dest, "manifest-break-even")


def storage_break_even_plot(ann, exact, dest):
    gib = float(exact["wire_object_bytes"]) / (1 << 30)
    with (RATE_CARD.parent / "s3_capacity_rates.csv").open() as f:
        storage = {
            r["profile"]: gib * float(r["usd_per_gib_month"]) for r in csv.DictReader(f)
        }
    reads = {k: float(ann[k + "_total_usd_per_m_queries"]) / 1_000_000 for k in storage}
    # Standard becomes the cheapest when it crosses every cheaper-capacity alternative.
    knee = max(
        (storage["s3"] - monthly) / (reads[k] - reads["s3"])
        for k, monthly in storage.items()
        if k != "s3"
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout="constrained")
    xs = [0, 35]
    ax.plot(
        xs,
        [100 * (storage["s3"] + x * reads["s3"]) for x in xs],
        label="S3 Standard · storage + cold reads",
        color=COLORS[0],
        lw=2.2,
    )
    ax.axvspan(knee, xs[-1], color=COLORS[0], alpha=0.09)
    ax.axvline(knee, linestyle="--", color=COLORS[0])
    ax.text(
        knee + 1,
        0.85,
        f"Standard is cheapest above\n{knee:.1f} cold searches/month",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
    )
    ax.set(
        xlabel="Cold ANN searches per month",
        ylabel="US cents / month",
        xlim=xs,
        title="S3 Standard · monthly cost for one 1M-vector segment",
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.5,
        -0.035,
        "712.90 MiB · us-east-1 · cold-read mix fixed · retained past alternative tiers’ minimum durations",
        ha="center",
        size=9,
    )
    save(fig, dest, "tier-break-even")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        type=Path,
        default=ROOT / "bench_results/pax_sift_scale_20261122/groups.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/benchmarks")
    parser.add_argument(
        "--dimension-csv",
        type=Path,
        default=ROOT / "bench_results/pax_sift_scale_20261122/dimension_costs.csv",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "pax-sift-scale",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.axisbelow": True,
        }
    )
    with args.groups.open() as f:
        groups = list(csv.DictReader(f))
        cold = sorted(
            (r for r in groups if r["phase"] == "cold"),
            key=lambda r: int(r["rows"]),
        )
    ann = [r for r in cold if r["route"] == "ann" and r["cap_mib"] == "8"]
    exact = [r for r in cold if r["route"] == "exact"]
    if [int(r["rows"]) for r in ann] != [1000, 10000, 33000, 100000, 333000, 1000000]:
        raise ValueError("missing or duplicate ANN scale points")
    if [r["rows"] for r in exact] != [r["rows"] for r in ann]:
        raise ValueError("missing or duplicate exact controls")
    args.output.mkdir(parents=True, exist_ok=True)
    recall_plot(ann, exact, args.output)
    tier_plot(ann[-1], args.output)
    scale_plot(ann, exact, args.output)
    rates = load_rate_card()
    model = dimension_model(exact, rates)
    with args.dimension_csv.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=model[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(model)
    dimension_plots(model, args.output)
    cache_plot(groups, args.output)
    publication_plot(args.output, rates)
    storage_break_even_plot(ann[-1], exact[-1], args.output)


if __name__ == "__main__":
    main()
