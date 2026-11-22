#!/usr/bin/env python3
"""Render cost-ledger figures from the reconciled query groups."""

import argparse
import csv
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pax_scale_wire_report import (
    DFS_READ_UNIT,
    RATE_CARD,
    estimate_request_cost,
    load_rate_card,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "AWS S3": [
        ("s3", "Standard"),
        ("s3_ia", "Standard-IA"),
        ("s3_gir", "Glacier Instant Retrieval"),
    ],
    "Azure Blob": [("blob", "Hot"), ("blob_cool", "Cool"), ("blob_cold", "Cold")],
    "Azure DFS": [("dfs", "Hot"), ("dfs_cool", "Cool"), ("dfs_cold", "Cold")],
    "Google Cloud Storage": [
        ("gcs", "Standard"),
        ("gcs_nearline", "Nearline"),
        ("gcs_coldline", "Coldline"),
        ("gcs_archive", "Archive"),
    ],
}
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
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def recall_plot(anns, exact, dest):
    fig, ax = plt.subplots(figsize=(8.7, 4.8), layout="constrained")
    for r in anns:
        x, y = float(r["recall_at_10"]), QPS * ops(r)
        ax.scatter(x, y, s=60, color=COLORS[0], zorder=3)
        ax.annotate(
            size_label(int(r["rows"])),
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


def tier_plot(r, dest):
    fig, (left, right) = plt.subplots(
        1, 2, sharey=True, figsize=(11.4, 6.6), layout="constrained"
    )
    labels = []
    for backend, tiers in PROFILES.items():
        for key, name in tiers:
            y = len(labels)
            labels.append(f"{backend}  ·  {name}")
            cost = float(r[key + "_total_usd_per_m_queries"])
            wire = ops(r) * QPS
            read_units = float(
                r["modeled_dfs_read_units"] if backend == "Azure DFS" else r["wire_get"]
            )
            billable = QPS * (
                read_units + float(r["wire_list"]) + float(r["wire_head"])
            )
            left.scatter(
                wire, y, s=56, facecolors="none", edgecolors=COLORS[0], zorder=4
            )
            left.scatter(billable, y, marker="x", color=COLORS[1], zorder=3)
            right.scatter(cost, y, color=COLORS[0], s=36)
            right.annotate(
                f"${cost:,.2f}",
                (cost, y),
                xytext=(7, -3),
                textcoords="offset points",
                size=8,
            )
    left.set_yticks(range(len(labels)), labels)
    left.invert_yaxis()
    left.set(xlabel="Operations / second at 100 searches/s", xlim=(1770, 2370))
    left.set_xticks([1900, 2100, 2240])
    right.set(xscale="log", xlim=(8, 8500), xlabel="USD / million searches (log scale)")
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.suptitle("1M vectors · cold ANN by storage tier", weight="bold")
    fig.text(
        0.5,
        -0.02,
        "○  Wire IOPS     ×  Modeled billed units/s     •  USD from Blob trace replay, including retrieval\n8 MiB merge cap   ·   Hot Azure account / Standard-default GCS bucket",
        ha="center",
        size=9,
    )
    save(fig, dest, "iops-tier-cost")


def scale_plot(anns, dest):
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.1), sharex=True, sharey=True, layout="constrained"
    )
    xs = [int(r["rows"]) for r in anns]
    for ax, (provider, tiers) in zip(axes.flat, PROFILES.items(), strict=True):
        for i, (key, name) in enumerate(tiers):
            values = [float(r[key + "_total_usd_per_m_queries"]) for r in anns]
            ax.plot(
                xs,
                values,
                color=COLORS[i],
                marker=["o", "s", "^", "D"][i],
                markersize=4,
                label=name,
            )
        ax.set(xscale="log", yscale="log", title=provider, ylim=(3.5, 4500))
        ax.set_xticks(xs, [size_label(n) for n in xs])
        ax.set_yticks([5, 20, 100, 500, 2500], ["$5", "$20", "$100", "$500", "$2,500"])
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Corpus size and cold ANN read cost", weight="bold")
    fig.supxlabel("Vectors in one collection (log scale)")
    fig.supylabel("USD / million searches (log scale)")
    fig.text(
        0.5,
        -0.02,
        "Blob trace replay with tier rates   ·   128d, 8 MiB cap   ·   Hot Azure account / Standard-default GCS bucket",
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
        for j, (profile, tier) in enumerate(
            (("s3", "S3 Standard"), ("s3_ia", "S3 Standard-IA"), ("dfs", "DFS Hot"))
        ):
            if i and profile == "dfs":
                continue  # Aligned windows have the same billed units at every width.
            right.plot(
                DIMENSIONS,
                [r[profile + "_total_usd_per_m_queries"] for r in rs],
                color=COLORS[j],
                linestyle=("--" if i else "-"),
                marker=".",
                label=(tier + " · both" if profile == "dfs" else tier + " · " + label),
            )
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
        "1M vectors · dimension sensitivity of two exact read plans", weight="bold"
    )
    fig.text(
        0.5,
        -0.025,
        "Model at 128d steps   ·   single cold segment, one LIST   ·   ranges cover identical data   ·   128d file size measured",
        ha="center",
        size=9,
    )
    save(fig, dest, "cost-iops-dimension")

    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.6), sharex=True, sharey=True, layout="constrained"
    )
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=128, vmax=1536)
    examples = (
        ("s3", "S3 Standard"),
        ("dfs", "Azure DFS Hot"),
        ("gcs_coldline", "GCS Coldline"),
    )
    for ax, (profile, name) in zip(axes, examples, strict=True):
        for d in DIMENSIONS:
            rows = [r for r in model if r["dimensions"] == d and r["plan"] == "whole"]
            xs = [r["rows"] for r in rows]
            ax.plot(
                xs,
                [r[profile + "_total_usd_per_m_queries"] for r in rows],
                marker=".",
                linewidth=1.2,
                color=cmap(norm(d)),
            )
        ax.set(xscale="log", yscale="log", title=name)
        ax.set_xticks(xs, [size_label(n) for n in xs], fontsize=8)
    axes[0].set_ylabel("USD / million searches (log scale)")
    axes[0].text(0.04, 0.12, "All widths coincide", transform=axes[0].transAxes, size=9)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        ticks=DIMENSIONS,
        label="Dimensions",
    )
    fig.supxlabel("Vectors in one collection (log scale)")
    fig.suptitle("Whole-object exact · corpus size and dimension", weight="bold")
    fig.text(
        0.5,
        -0.03,
        "One cold LIST + one object GET   ·   128–1536d, step 128   ·   f32 + SQ8 + 1-bit tiers   ·   fixed non-vector overhead per corpus",
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
    fig, axes = plt.subplots(
        1, 2, figsize=(9.8, 4.2), layout="constrained", sharey=True
    )
    for ax, (profile, label) in zip(
        axes, (("s3", "S3 Standard"), ("dfs", "Azure DFS Hot")), strict=True
    ):
        key = profile + "_total_usd_per_m_queries"
        cold = float(rows["cold"][key])
        for i, (phase, title) in enumerate(
            (("l1_repeat", "DRAM repeat"), ("l2_reopen", "Disk-cache reopen"))
        ):
            warm = float(rows[phase][key])
            ax.plot([0, 100], [warm, cold], color=COLORS[i], marker="o", label=title)
            ax.annotate(
                f"${warm:.2f}",
                (0, warm),
                xytext=(5, 5),
                textcoords="offset points",
                size=8,
            )
        ax.set(
            title=label, xlabel="Cold queries in mix (%)", xlim=(-3, 104), ylim=(0, 20)
        )
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("USD / million searches")
    fig.suptitle("1M vectors · what a process restart costs", weight="bold")
    fig.text(
        0.5,
        -0.03,
        "Azurite request counts and bytes   ·   prices and mixes modeled   ·   fresh discovery adds one LIST at reopen",
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
    plans = []
    with (RATE_CARD.parent / "s3_capacity_rates.csv").open() as f:
        for rate, (key, label) in zip(
            csv.DictReader(f), PROFILES["AWS S3"], strict=True
        ):
            if rate["profile"] != key:
                raise ValueError("capacity and transaction profiles differ")
            monthly = gib * float(rate["usd_per_gib_month"])
            per_search = float(ann[key + "_total_usd_per_m_queries"]) / 1_000_000
            plans.append((label, monthly, per_search))
    fig, ax = plt.subplots(figsize=(8.4, 4.3), layout="constrained")
    for i, (label, monthly, per_search) in enumerate(plans):
        ax.plot(
            [0, 35],
            [100 * monthly, 100 * (monthly + 35 * per_search)],
            label=label,
            color=COLORS[i],
        )
    for left, right in pairwise(plans):
        knee = (left[1] - right[1]) / (right[2] - left[2])
        cost = 100 * (left[1] + knee * left[2])
        ax.scatter(knee, cost, color="#344454", zorder=3)
        ax.annotate(
            f"{knee:.1f} searches",
            (knee, cost),
            xytext=(3, 8),
            textcoords="offset points",
            size=9,
        )
    ax.set(
        xlabel="Cold ANN searches per month",
        ylabel="US cents / month",
        title="S3 at 1M vectors · storage savings versus read frequency",
    )
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        -0.03,
        "One 712.90 MiB object   ·   128d, 8 MiB cap   ·   capacity + LIST + GET + retrieval   ·   retained beyond tier minimums",
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
    scale_plot(ann, args.output)
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
