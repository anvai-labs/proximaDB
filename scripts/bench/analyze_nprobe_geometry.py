#!/usr/bin/env python3
"""Analyze and visualize measured PAX nprobe/recall/cost geometry.

The input is auditable JSON emitted by :mod:`nprobe_sweep`. The analyzer keeps
two decisions separate:

* ``economy_profile`` is the unconstrained diminishing-return knee on a
  measured recall/cost Pareto frontier. Physical GET/query is the primary
  objective for Azure/Azurite evidence because Azure bills hot-tier reads per
  operation.
* ``quality_profile`` is the least measured nprobe satisfying an explicit
  recall target. It reports ``unattained`` instead of redefining that target.

Power laws are descriptive fits over the independently measured knees/floors.
They are not permission to extrapolate outside the measured cell-count range.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def sorted_points(points: list[dict]) -> list[dict]:
    ordered = sorted(points, key=lambda point: int(point["nprobe"]))
    if not ordered:
        raise RuntimeError("nprobe curve has no points")
    if len({int(point["nprobe"]) for point in ordered}) != len(ordered):
        raise RuntimeError("nprobe curve contains duplicate probes")
    return ordered


def metric_value(point: dict, key: str) -> float:
    """Read a numeric point metric, including dotted nested keys."""
    value: object = point
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise RuntimeError(f"point is missing cost metric {key!r}")
        value = value[component]
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"point cost metric {key!r} is not numeric") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"point cost metric {key!r} is not finite")
    return numeric


def quality_profile(points: list[dict], target_recall: float) -> dict:
    """Report whether, and where, the measured curve attains the SLA."""
    ordered = sorted_points(points)
    candidates = [
        point for point in ordered if float(point["recall_at_k"]) >= target_recall
    ]
    if not candidates:
        maximum = max(
            ordered,
            key=lambda point: (float(point["recall_at_k"]), int(point["nprobe"])),
        )
        return {
            "status": "unattained",
            "target_recall": target_recall,
            "point": None,
            "max_measured_recall": float(maximum["recall_at_k"]),
            "max_measured_nprobe": int(maximum["nprobe"]),
        }
    # nprobe is the stable work coordinate. GETs can be non-monotone because
    # adjacent ranges coalesce and are therefore a secondary tie-break only.
    point = min(
        candidates,
        key=lambda candidate: (
            int(candidate["nprobe"]),
            float(candidate["gets_per_query"]),
        ),
    )
    return {
        "status": "attained",
        "target_recall": target_recall,
        "point": point,
        "max_measured_recall": max(
            float(candidate["recall_at_k"]) for candidate in ordered
        ),
        "max_measured_nprobe": max(int(candidate["nprobe"]) for candidate in ordered),
    }


def pareto_frontier(points: list[dict], cost_key: str) -> list[dict]:
    """Return points not dominated by a cheaper point with equal recall."""
    ordered = sorted(
        sorted_points(points),
        key=lambda point: (
            metric_value(point, cost_key),
            -float(point["recall_at_k"]),
            int(point["nprobe"]),
        ),
    )
    if any(metric_value(point, cost_key) <= 0.0 for point in ordered):
        raise RuntimeError(f"curve bend requires positive {cost_key}")
    frontier = []
    best_recall = -math.inf
    for point in ordered:
        recall = float(point["recall_at_k"])
        if recall > best_recall:
            frontier.append(point)
            best_recall = recall
    return frontier


def curve_bend(points: list[dict], cost_key: str) -> dict:
    """Return the Kneedle elbow for a measured recall/cost Pareto frontier.

    X is log(cost), because probe sweeps and their physical costs are geometric.
    Dominated samples are removed before the selected interior point maximizes
    normalized ``y - x``.
    """
    ordered = pareto_frontier(points, cost_key)
    if len(ordered) < 3:
        raise RuntimeError("curve bend requires at least three points")
    recalls = [float(point["recall_at_k"]) for point in ordered]
    x_values = [math.log(metric_value(point, cost_key)) for point in ordered]
    x_span = x_values[-1] - x_values[0]
    y_span = recalls[-1] - recalls[0]
    if x_span <= 0.0 or y_span <= 0.0:
        raise RuntimeError("curve bend requires increasing cost and recall ranges")
    distances = [
        (
            (recalls[index] - recalls[0]) / y_span
            - (x_values[index] - x_values[0]) / x_span,
            index,
        )
        for index in range(1, len(ordered) - 1)
    ]
    _, best_index = max(distances, key=lambda candidate: (candidate[0], candidate[1]))
    return ordered[best_index]


def natural_knee_profile(points: list[dict], cost_key: str) -> dict:
    """Return a structured knee so sparse curves remain valid evidence."""
    frontier = pareto_frontier(points, cost_key)
    try:
        point = curve_bend(points, cost_key)
    except RuntimeError as error:
        return {
            "status": "unavailable",
            "cost_metric": cost_key,
            "point": None,
            "frontier_point_count": len(frontier),
            "reason": str(error),
        }
    return {
        "status": "measured",
        "cost_metric": cost_key,
        "point": point,
        "frontier_point_count": len(frontier),
    }


def _ordinary_log_fit(samples: list[dict]) -> tuple[float, float]:
    xs = [math.log(float(sample["coarse_cells"])) for sample in samples]
    ys = [math.log(float(sample["nprobe"])) for sample in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_variance = sum((value - x_mean) ** 2 for value in xs)
    if x_variance <= 0.0:
        raise RuntimeError("power-law fit requires distinct coarse-cell counts")
    exponent = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(xs, ys, strict=True)
        )
        / x_variance
    )
    coefficient = math.exp(y_mean - exponent * x_mean)
    return coefficient, exponent


def fit_power_law(samples: list[dict]) -> dict:
    """Fit ``nprobe = coefficient * coarse_cells ** exponent`` in log space."""
    if len(samples) < 3:
        raise RuntimeError("power-law fit requires at least three scales")
    if any(
        float(sample["coarse_cells"]) <= 0.0 or float(sample["nprobe"]) <= 0.0
        for sample in samples
    ):
        raise RuntimeError("power-law samples must be positive")
    coefficient, exponent = _ordinary_log_fit(samples)
    observed = [math.log(float(sample["nprobe"])) for sample in samples]
    predicted = [
        math.log(coefficient) + exponent * math.log(float(sample["coarse_cells"]))
        for sample in samples
    ]
    mean = sum(observed) / len(observed)
    total = sum((value - mean) ** 2 for value in observed)
    residual = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(observed, predicted, strict=True)
    )
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    errors = []
    for withheld in range(len(samples)):
        training = [sample for index, sample in enumerate(samples) if index != withheld]
        fold_coefficient, fold_exponent = _ordinary_log_fit(training)
        sample = samples[withheld]
        estimate = fold_coefficient * (float(sample["coarse_cells"]) ** fold_exponent)
        errors.append(abs(estimate - float(sample["nprobe"])) / sample["nprobe"])
    return {
        "formula": "nprobe = coefficient * coarse_cells ^ exponent",
        "coefficient": coefficient,
        "exponent": exponent,
        "r_squared_log": r_squared,
        "loocv_mape": sum(errors) / len(errors),
        "measured_cell_range": [
            min(int(sample["coarse_cells"]) for sample in samples),
            max(int(sample["coarse_cells"]) for sample in samples),
        ],
    }


def load_matrix(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("protocol") != "pax_nprobe_topk_matrix":
        raise RuntimeError(f"{path}: unexpected protocol {result.get('protocol')!r}")
    geometry = result.get("settled_geometry", {})
    segments = geometry.get("segments", [])
    if geometry.get("segment_count") != 1 or len(segments) != 1:
        raise RuntimeError(f"{path}: fit requires exactly one settled PAX segment")
    if geometry.get("row_count") != result.get("dataset", {}).get("corpus_rows"):
        raise RuntimeError(f"{path}: settled rows differ from corpus rows")
    status = result.get("status")
    if status == "running":
        raise RuntimeError(f"{path}: matrix checkpoint is still running")
    if status not in {"pass", "fail", "incomplete"}:
        raise RuntimeError(f"{path}: unexpected matrix status {status!r}")
    measurement_failures = result.get("measurement_failures")
    if measurement_failures is None:
        # Read-only evidence compatibility for checkpoints emitted before
        # quality outcomes were separated from measurement failures.
        measurement_failures = result.get("failures", [])
    if measurement_failures:
        raise RuntimeError(f"{path}: matrix has measurement failures")
    checkpoint = result.get("checkpoint", {})
    matrix = result.get("matrix", {})
    points = matrix.get("points", [])
    configured_point_count = len(matrix.get("nprobes", [])) * len(
        matrix.get("top_k_values", [])
    )
    if status == "incomplete":
        if checkpoint.get("state") != "incomplete":
            raise RuntimeError(f"{path}: inconsistent incomplete checkpoint")
        if checkpoint.get("completed_points") != len(points) or not points:
            raise RuntimeError(f"{path}: incomplete checkpoint point count differs")
        if checkpoint.get("expected_points") != configured_point_count:
            raise RuntimeError(f"{path}: incomplete checkpoint plan differs")
    if status in {"pass", "fail"}:
        if checkpoint:
            if checkpoint.get("state") != status:
                raise RuntimeError(f"{path}: terminal checkpoint state differs")
            if checkpoint.get("completed_points") != len(points):
                raise RuntimeError(f"{path}: terminal checkpoint point count differs")
            if checkpoint.get("expected_points") != len(points):
                raise RuntimeError(
                    f"{path}: terminal matrix is missing measured points"
                )
        elif status == "pass":
            # Read-only evidence compatibility for complete matrices emitted
            # before atomic checkpoints were added to this benchmark harness.
            if configured_point_count != len(points):
                raise RuntimeError(
                    f"{path}: terminal matrix is missing measured points"
                )
    if status == "fail":
        if "measurement_failures" not in result:
            raise RuntimeError(f"{path}: legacy failed matrix cannot attribute failure")
        quality_outcomes = result.get("quality_outcomes", [])
        if result.get("matrix", {}).get("quality_policy") != "require" or not any(
            outcome.get("status") == "unattained" for outcome in quality_outcomes
        ):
            raise RuntimeError(f"{path}: failed matrix has no attributable failure")
    return result


KNEE_COST_METRICS = (
    "nprobe",
    "gets_per_query",
    "bytes_per_query",
    "latency_ms.p50",
    "latency_ms.p95",
)


def summarize_matrix(
    result: dict,
    target_recall: float,
    sources: list[dict],
    measurement_coverage: dict,
) -> dict:
    points = result["matrix"]["points"]
    top_k_values = sorted({int(point["top_k"]) for point in points})
    curves = {}
    for top_k in top_k_values:
        curve = [point for point in points if int(point["top_k"]) == top_k]
        natural_knees = {
            metric: natural_knee_profile(curve, metric) for metric in KNEE_COST_METRICS
        }
        curves[str(top_k)] = {
            "natural_knees": natural_knees,
            "economy_profile": {
                "objective": "gets_per_query",
                **natural_knees["gets_per_query"],
            },
            "quality_profile": quality_profile(curve, target_recall),
        }
    segment = result["settled_geometry"]["segments"][0]
    corpus_rows = int(result["dataset"]["corpus_rows"])
    dimension = int(result["dataset"]["dimension"])
    coarse_cells = int(segment["coarse_cells"])
    return {
        "sources": sources,
        "measurement_coverage": measurement_coverage,
        "corpus_rows": corpus_rows,
        "dimension": dimension,
        "coarse_cells": coarse_cells,
        "rows_per_cell_mean": corpus_rows / coarse_cells,
        "sq8_bytes_per_cell_mean": corpus_rows * dimension / coarse_cells,
        "rabitq_bytes_per_cell_mean": (
            corpus_rows * (8 + math.ceil(dimension / 8)) / coarse_cells
        ),
        "cell_row_summary": segment.get("cell_row_summary"),
        "cell_row_max_to_mean": segment.get("cell_row_max_to_mean"),
        "empty_cell_fraction": segment.get("empty_cell_fraction"),
        "radius_summary": segment.get("radius_summary"),
        "curves": curves,
        "points": sorted(
            points,
            key=lambda point: (int(point["top_k"]), int(point["nprobe"])),
        ),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _segment_provenance(segment: dict) -> dict:
    return {
        key: segment.get(key)
        for key in (
            "path",
            "blob_etag",
            "bytes",
            "layout_version",
            "coarse_cells",
            "coarse_seed",
        )
    }


def matrix_provenance(result: dict) -> dict:
    """Return immutable identity; intentionally exclude snapshot timestamps."""
    dataset = result.get("dataset", {})
    filesystem = result.get("filesystem_profile", {})
    geometry = result.get("settled_geometry", {})
    return {
        "git_revision": result.get("git_revision"),
        "collection_id": result.get("collection_id"),
        "binary": {
            "sha256": result.get("binary", {}).get("sha256"),
            "source_revision": result.get("binary", {}).get("source_revision"),
        },
        "bed_config_sha256": result.get("bed_config", {}).get("sha256"),
        "dataset": {
            key: dataset.get(key)
            for key in (
                "base",
                "base_format",
                "corpus_rows",
                "dimension",
                "queries_path",
                "query_format",
                "groundtruth",
                "groundtruth_format",
                "groundtruth_scope_rows",
                "groundtruth_width",
                "query_range",
            )
        },
        "filesystem": {
            "storage_url": filesystem.get("storage_url"),
            "azurite": filesystem.get("azurite"),
            "persistent_local_tier": filesystem.get("persistent_local_tier"),
        },
        "compute_profile": result.get("compute_profile"),
        "settled_geometry": {
            "segment_count": geometry.get("segment_count"),
            "row_count": geometry.get("row_count"),
            "segments": [
                _segment_provenance(segment) for segment in geometry.get("segments", [])
            ],
        },
    }


def _fit_series(samples: list[dict]) -> dict:
    ordered = sorted(samples, key=lambda sample: int(sample["corpus_rows"]))
    if len(ordered) < 3:
        return {
            "status": "insufficient_samples",
            "sample_count": len(ordered),
            "samples": ordered,
        }
    fit = fit_power_law(ordered)
    return {
        "status": "fit",
        "evidence_status": (
            "complete"
            if all(sample["measurement_coverage"] == "complete" for sample in ordered)
            else "provisional_partial_input"
        ),
        **fit,
        "sample_count": len(ordered),
        "samples": ordered,
        "comparison": [
            {
                **sample,
                "fitted_nprobe": fit["coefficient"]
                * sample["coarse_cells"] ** fit["exponent"],
                "current_2sqrt_nprobe": math.ceil(
                    2.0 * math.sqrt(sample["coarse_cells"])
                ),
            }
            for sample in ordered
        ],
    }


def build_analysis(matrices: list[Path], target_recall: float) -> dict:
    if not matrices:
        raise RuntimeError("analysis requires at least one matrix")
    grouped: dict[int, list[tuple[Path, dict]]] = {}
    for path in matrices:
        result = load_matrix(path)
        corpus_rows = int(result["dataset"]["corpus_rows"])
        grouped.setdefault(corpus_rows, []).append((path, result))

    scales = []
    for corpus_rows, entries in grouped.items():
        expected_provenance = matrix_provenance(entries[0][1])
        if any(
            matrix_provenance(result) != expected_provenance
            for _, result in entries[1:]
        ):
            raise RuntimeError(
                f"corpus scale {corpus_rows}: checkpoint provenance differs"
            )
        merged_points = []
        identities: set[tuple[int, int]] = set()
        expected_identities: set[tuple[int, int]] = set()
        sources = []
        statuses = []
        for path, result in entries:
            statuses.append(str(result["status"]))
            sources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "status": result["status"],
                    "checkpoint": result.get("checkpoint"),
                }
            )
            expected_identities.update(
                (int(nprobe), int(top_k))
                for nprobe in result["matrix"].get("nprobes", [])
                for top_k in result["matrix"].get("top_k_values", [])
            )
            for point in result["matrix"]["points"]:
                identity = (int(point["nprobe"]), int(point["top_k"]))
                if identity in identities:
                    raise RuntimeError(
                        f"corpus scale {corpus_rows}: duplicate measured point "
                        f"nprobe={identity[0]} top_k={identity[1]}"
                    )
                identities.add(identity)
                merged_points.append(point)
        merged = dict(entries[0][1])
        merged["matrix"] = dict(merged["matrix"])
        merged["matrix"]["points"] = merged_points
        coverage_status = (
            "complete"
            if all(status in {"pass", "fail"} for status in statuses)
            else "partial"
        )
        expected_point_count = len(expected_identities)
        if expected_point_count == 0:
            raise RuntimeError(f"corpus scale {corpus_rows}: empty measurement plan")
        coverage = {
            "status": coverage_status,
            "source_count": len(entries),
            "source_statuses": statuses,
            "measured_point_count": len(merged_points),
            "expected_point_count": expected_point_count,
            "completion_fraction": len(merged_points) / expected_point_count,
        }
        scales.append(summarize_matrix(merged, target_recall, sources, coverage))

    scales.sort(key=lambda scale: scale["corpus_rows"])
    top_k_values = sorted({int(value) for scale in scales for value in scale["curves"]})
    fits = {"natural_knee": {}, "quality_floor": {}}
    for top_k in top_k_values:
        natural_samples = []
        quality_samples = []
        for scale in scales:
            curve = scale["curves"].get(str(top_k))
            if curve is None:
                continue
            economy = curve["economy_profile"]
            if economy["status"] == "measured":
                natural_samples.append(
                    {
                        "corpus_rows": scale["corpus_rows"],
                        "coarse_cells": scale["coarse_cells"],
                        "nprobe": int(economy["point"]["nprobe"]),
                        "measurement_coverage": scale["measurement_coverage"]["status"],
                    }
                )
            quality = curve["quality_profile"]
            if quality["status"] == "attained":
                quality_samples.append(
                    {
                        "corpus_rows": scale["corpus_rows"],
                        "coarse_cells": scale["coarse_cells"],
                        "nprobe": int(quality["point"]["nprobe"]),
                        "measurement_coverage": scale["measurement_coverage"]["status"],
                    }
                )
        fits["natural_knee"][str(top_k)] = _fit_series(natural_samples)
        fits["quality_floor"][str(top_k)] = _fit_series(quality_samples)
    return {
        "protocol": "pax_nprobe_geometry_analysis_v2",
        "target_recall": target_recall,
        "scale_count": len(scales),
        "scales": scales,
        "fits": fits,
        "interpretation": {
            "coarse_cell_rule": (
                "k_c = clamp(floor(corpus_rows * dimension / 4MiB), 2, 4096); "
                "therefore the fitted deployment form is coefficient * "
                "(corpus_rows * dimension / 4MiB) ^ exponent"
            ),
            "economy_rule": (
                "maximum normalized Kneedle distance on log(GET/query) versus "
                "recall after removing cost-dominated points"
            ),
            "quality_rule": (
                "minimum measured nprobe whose recall meets the explicit target; "
                "report unattained rather than lowering the target"
            ),
            "extrapolation": (
                "descriptive only within measured_cell_range; clamp to "
                "[configured_min, coarse_cells]"
            ),
            "cluster_radii": (
                "persisted and reported as shape evidence, but the current "
                "probe ranks centroid distance only; radii do not select cells "
                "and cannot cause a recall or GET change in this sweep"
            ),
        },
    }


def write_svg(analysis: dict, destination: Path) -> None:
    width, height = 1200, 760
    left, right, top, bottom = 95, 55, 70, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    scales = analysis["scales"]
    all_points = [point for scale in scales for point in scale["points"]]
    min_recall = min(float(point["recall_at_k"]) for point in all_points)
    y_min = max(0.0, math.floor((min_recall - 0.02) * 20.0) / 20.0)
    # Reserve visual headroom so exact-recall bubbles do not clip at the frame.
    y_max = 1.01
    x_logs = [math.log10(scale["corpus_rows"]) for scale in scales]
    x_min, x_max = min(x_logs), max(x_logs)
    max_gets = max(float(point["gets_per_query"]) for point in all_points)
    max_probe = max(int(point["nprobe"]) for point in all_points)

    def x_position(rows: int, nprobe: int, top_k: int) -> float:
        base = (math.log10(rows) - x_min) / max(x_max - x_min, 1e-12)
        # Small deterministic jitter exposes each vertical sweep without
        # changing the corpus-size encoding.
        probe_fraction = math.log(nprobe) / max(math.log(max_probe), 1e-12)
        top_offset = -0.010 if top_k == 10 else 0.010
        return left + plot_width * (
            0.025 + 0.95 * base + (probe_fraction - 0.5) * 0.035 + top_offset
        )

    def y_position(recall: float) -> float:
        fraction = (recall - y_min) / max(y_max - y_min, 1e-12)
        return top + plot_height * (1.0 - fraction)

    def color(gets: float) -> str:
        fraction = min(max(gets / max(max_gets, 1e-12), 0.0), 1.0)
        red = round(48 + 200 * fraction)
        green = round(150 - 90 * fraction)
        blue = round(210 - 145 * fraction)
        return f"rgb({red},{green},{blue})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#18212b}",
        ".grid{stroke:#d9e1e8;stroke-width:1}.axis{stroke:#536272;stroke-width:1.5}",
        ".natural-knee{stroke:#111827;stroke-width:2.5;fill:none}",
        ".quality-floor{stroke:#111827;stroke-width:2.5;fill:none;"
        "stroke-dasharray:5 4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        '<text x="95" y="34" font-size="22" font-weight="700">'
        "PAX nprobe scale geometry — release/Azurite evidence</text>",
        f'<text x="95" y="56" font-size="13">Bubble area scales with nprobe; '
        f"color scales with physical GET/query; solid ring = GET knee; dashed "
        f"ring = first recall ≥ {analysis['target_recall']:.3f} when attained; "
        "x jitter only separates bubbles.</text>",
    ]
    for tick in range(6):
        recall = y_min + (1.0 - y_min) * tick / 5
        y = y_position(recall)
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{width - right}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="12">{recall:.3f}</text>',
            ]
        )
    target_y = y_position(float(analysis["target_recall"]))
    svg.append(
        f'<line x1="{left}" y1="{target_y:.2f}" x2="{width - right}" '
        f'y2="{target_y:.2f}" stroke="#111827" stroke-dasharray="7 5"/>'
    )
    for scale in scales:
        x = left + plot_width * (
            0.025
            + 0.95
            * (math.log10(scale["corpus_rows"]) - x_min)
            / max(x_max - x_min, 1e-12)
        )
        label = (
            f"{scale['corpus_rows'] / 1_000_000:.1f}M"
            if scale["corpus_rows"] >= 1_000_000
            else f"{scale['corpus_rows'] // 1_000}K"
        )
        svg.extend(
            [
                f'<line class="axis" x1="{x:.2f}" y1="{height - bottom}" '
                f'x2="{x:.2f}" y2="{height - bottom + 6}"/>',
                f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle" '
                f'font-size="13">{label}</text>',
                f'<text x="{x:.2f}" y="{height - bottom + 42}" text-anchor="middle" '
                f'font-size="11">k_c={scale["coarse_cells"]}</text>',
            ]
        )
        for point in scale["points"]:
            top_k = int(point["top_k"])
            nprobe = int(point["nprobe"])
            gets = float(point["gets_per_query"])
            cx = x_position(scale["corpus_rows"], nprobe, top_k)
            cy = y_position(float(point["recall_at_k"]))
            radius = 3.5 + 13.0 * math.sqrt(nprobe / max_probe)
            fill = color(gets)
            opacity = "0.72" if top_k == 10 else "0.38"
            dash = "" if top_k == 10 else ' stroke-dasharray="3 2"'
            tooltip = html.escape(
                f"N={scale['corpus_rows']:,}; top_k={top_k}; "
                f"nprobe={nprobe}; recall={point['recall_at_k']:.5f}; "
                f"GET/q={gets:.2f}"
            )
            svg.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
                f'fill="{fill}" fill-opacity="{opacity}" stroke="{fill}"'
                f"{dash}><title>{tooltip}</title></circle>"
            )
        for top_k_text, curve in scale["curves"].items():
            marker_profiles = (
                ("natural-knee", curve["economy_profile"]),
                ("quality-floor", curve["quality_profile"]),
            )
            for marker_class, profile in marker_profiles:
                knee = profile.get("point")
                if knee is None:
                    continue
                cx = x_position(
                    scale["corpus_rows"], int(knee["nprobe"]), int(top_k_text)
                )
                cy = y_position(float(knee["recall_at_k"]))
                radius = 6.0 + 13.0 * math.sqrt(int(knee["nprobe"]) / max_probe)
                svg.append(
                    f'<circle class="{marker_class}" cx="{cx:.2f}" '
                    f'cy="{cy:.2f}" r="{radius:.2f}"/>'
                )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" '
            f'text-anchor="middle" font-size="14">Corpus vectors (log scale)</text>',
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            f'font-size="14" transform="rotate(-90 24 '
            f'{top + plot_height / 2:.2f})">Recall</text>',
            '<circle cx="980" cy="40" r="7" fill="#308fd2" fill-opacity=".72"/>',
            '<text x="994" y="44" font-size="12">top-10</text>',
            '<circle cx="1060" cy="40" r="7" fill="#308fd2" '
            'fill-opacity=".38" stroke="#308fd2" stroke-dasharray="3 2"/>',
            '<text x="1074" y="44" font-size="12">top-20</text>',
            "</svg>",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n")


SCALE_COLORS = ("#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed")


def scale_label(rows: int) -> str:
    if rows >= 1_000_000:
        return f"{rows / 1_000_000:g}M"
    return f"{rows / 1_000:g}K"


def write_dual_axis_svg(analysis: dict, destination: Path) -> None:
    """Plot recall and physical GETs against normalized probe fraction.

    A dual axis is useful for seeing both curves, but their visual crossing is
    not treated as a decision rule: the explicit recall and GET thresholds are.
    """
    width, height = 1380, 820
    left, right, top, bottom = 92, 100, 110, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    scales = analysis["scales"]
    all_points = [point for scale in scales for point in scale["points"]]
    min_recall = min(float(point["recall_at_k"]) for point in all_points)
    recall_min = max(0.0, math.floor((min_recall - 0.02) * 20.0) / 20.0)
    max_gets = max(float(point["gets_per_query"]) for point in all_points) * 1.05

    def x_position(fraction: float) -> float:
        return left + plot_width * fraction

    recall_max = 1.005

    def recall_y(value: float) -> float:
        fraction = (value - recall_min) / max(recall_max - recall_min, 1e-12)
        return top + plot_height * (1.0 - fraction)

    def gets_y(value: float) -> float:
        return top + plot_height * (1.0 - value / max(max_gets, 1e-12))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#18212b}",
        ".grid{stroke:#d9e1e8;stroke-width:1}.axis{stroke:#536272;stroke-width:1.5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        f'<text x="{left}" y="35" font-size="22" font-weight="700">'
        "Recall and GET cost by normalized PAX probe fraction</text>",
        f'<text x="{left}" y="58" font-size="13">'
        "Solid = recall (left axis); dashed = physical GET/query (right axis); "
        "circle = top-10; square = top-20. Axis crossings are not optima.</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        x = x_position(fraction)
        svg.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                f'x2="{x:.2f}" y2="{height - bottom}"/>',
                f'<text x="{x:.2f}" y="{height - bottom + 24}" '
                f'text-anchor="middle" font-size="12">{fraction:.1f}</text>',
            ]
        )
        recall = recall_min + (1.0 - recall_min) * fraction
        recall_tick_y = recall_y(recall)
        get_value = max_gets * fraction
        get_tick_y = gets_y(get_value)
        svg.extend(
            [
                f'<line class="grid" x1="{left}" y1="{recall_tick_y:.2f}" '
                f'x2="{width - right}" y2="{recall_tick_y:.2f}"/>',
                f'<text x="{left - 10}" y="{recall_tick_y + 4:.2f}" '
                f'text-anchor="end" font-size="12">{recall:.3f}</text>',
                f'<text x="{width - right + 10}" y="{get_tick_y + 4:.2f}" '
                f'font-size="12">{get_value:.1f}</text>',
            ]
        )
    target_y = recall_y(float(analysis["target_recall"]))
    svg.append(
        f'<line x1="{left}" y1="{target_y:.2f}" x2="{width - right}" '
        f'y2="{target_y:.2f}" stroke="#111827" stroke-dasharray="9 5">'
        f"<title>recall target {analysis['target_recall']:.3f}</title></line>"
    )
    if max_gets >= 10.0:
        budget_y = gets_y(10.0)
        svg.append(
            f'<line x1="{left}" y1="{budget_y:.2f}" x2="{width - right}" '
            f'y2="{budget_y:.2f}" stroke="#b91c1c" stroke-dasharray="3 5">'
            "<title>10 GET/query budget</title></line>"
        )
    for scale_index, scale in enumerate(scales):
        color = SCALE_COLORS[scale_index % len(SCALE_COLORS)]
        coarse_cells = int(scale["coarse_cells"])
        for top_k in sorted({int(point["top_k"]) for point in scale["points"]}):
            points = sorted(
                (point for point in scale["points"] if int(point["top_k"]) == top_k),
                key=lambda point: int(point["nprobe"]),
            )
            recall_path = " ".join(
                f"{x_position(int(point['nprobe']) / coarse_cells):.2f},"
                f"{recall_y(float(point['recall_at_k'])):.2f}"
                for point in points
            )
            gets_path = " ".join(
                f"{x_position(int(point['nprobe']) / coarse_cells):.2f},"
                f"{gets_y(float(point['gets_per_query'])):.2f}"
                for point in points
            )
            opacity = 1.0 if top_k == 10 else 0.55
            svg.extend(
                [
                    f'<polyline points="{recall_path}" fill="none" stroke="{color}" '
                    f'stroke-width="2.4" opacity="{opacity}"/>',
                    f'<polyline points="{gets_path}" fill="none" stroke="{color}" '
                    f'stroke-width="2" stroke-dasharray="8 5" opacity="{opacity}"/>',
                ]
            )
            for point in points:
                fraction = int(point["nprobe"]) / coarse_cells
                tooltip = html.escape(
                    f"N={scale['corpus_rows']:,}; k_c={coarse_cells}; "
                    f"top_k={top_k}; nprobe={point['nprobe']}; "
                    f"recall={point['recall_at_k']:.5f}; "
                    f"GET/q={point['gets_per_query']:.3f}"
                )
                x = x_position(fraction)
                for y in (
                    recall_y(float(point["recall_at_k"])),
                    gets_y(float(point["gets_per_query"])),
                ):
                    if top_k == 10:
                        svg.append(
                            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" '
                            f'fill="{color}" opacity="{opacity}">'
                            f"<title>{tooltip}</title></circle>"
                        )
                    else:
                        svg.append(
                            f'<rect x="{x - 3.5:.2f}" y="{y - 3.5:.2f}" '
                            f'width="7" height="7" fill="{color}" opacity="{opacity}">'
                            f"<title>{tooltip}</title></rect>"
                        )
        legend_x = left + scale_index * 185
        legend_y = 82
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 30}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="3"/>',
                f'<text x="{legend_x + 39}" y="{legend_y + 4}" font-size="12">'
                f"{scale_label(int(scale['corpus_rows']))} "
                f"(k_c={coarse_cells})</text>",
            ]
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}"/>',
            f'<line class="axis" x1="{width - right}" y1="{top}" '
            f'x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" '
            'text-anchor="middle" font-size="14">Normalized probe fraction '
            "(nprobe / k_c)</text>",
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            f'font-size="14" transform="rotate(-90 24 {top + plot_height / 2:.2f})">'
            "Recall</text>",
            f'<text x="{width - 24}" y="{top + plot_height / 2:.2f}" '
            f'text-anchor="middle" font-size="14" transform="rotate(90 '
            f'{width - 24} {top + plot_height / 2:.2f})">Physical GET/query</text>',
            "</svg>",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n")


def write_pareto_svg(analysis: dict, destination: Path) -> None:
    """Plot measured GET cost against recall with probed-row bubble area."""
    width, height = 1280, 820
    left, right, top, bottom = 95, 55, 110, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    scales = analysis["scales"]
    all_points = [point for scale in scales for point in scale["points"]]
    min_recall = min(float(point["recall_at_k"]) for point in all_points)
    x_min = max(0.0, math.floor((min_recall - 0.02) * 20.0) / 20.0)
    max_gets = max(float(point["gets_per_query"]) for point in all_points) * 1.05

    recall_max = 1.005

    def x_position(recall: float) -> float:
        return left + plot_width * (recall - x_min) / max(recall_max - x_min, 1e-12)

    def y_position(gets: float) -> float:
        return top + plot_height * (1.0 - gets / max(max_gets, 1e-12))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#18212b}",
        ".grid{stroke:#d9e1e8;stroke-width:1}.axis{stroke:#536272;stroke-width:1.5}",
        ".natural-knee{stroke:#111827;stroke-width:2.5;fill:none}",
        ".quality-floor{stroke:#111827;stroke-width:2.5;fill:none;"
        "stroke-dasharray:5 4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        f'<text x="{left}" y="34" font-size="22" font-weight="700">'
        "Recall → GET/query Pareto frontier</text>",
        f'<text x="{left}" y="57" font-size="13">'
        "Bubble area scales with actual rows probed / corpus rows; solid ring = GET "
        "knee; dashed ring = first point at "
        f"recall ≥ {analysis['target_recall']:.3f} when attained; circle = top-10; "
        "square = top-20.</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        recall = x_min + (1.0 - x_min) * fraction
        x = x_position(recall)
        gets = max_gets * fraction
        y = y_position(gets)
        svg.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                f'x2="{x:.2f}" y2="{height - bottom}"/>',
                f'<text x="{x:.2f}" y="{height - bottom + 24}" '
                f'text-anchor="middle" font-size="12">{recall:.3f}</text>',
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{width - right}" y2="{y:.2f}"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="12">{gets:.1f}</text>',
            ]
        )
    target_x = x_position(float(analysis["target_recall"]))
    svg.append(
        f'<line x1="{target_x:.2f}" y1="{top}" x2="{target_x:.2f}" '
        f'y2="{height - bottom}" stroke="#111827" stroke-dasharray="9 5"/>'
    )
    if max_gets >= 10.0:
        budget_y = y_position(10.0)
        svg.append(
            f'<line x1="{left}" y1="{budget_y:.2f}" x2="{width - right}" '
            f'y2="{budget_y:.2f}" stroke="#b91c1c" stroke-dasharray="3 5"/>'
        )
    for scale_index, scale in enumerate(scales):
        color = SCALE_COLORS[scale_index % len(SCALE_COLORS)]
        corpus_rows = int(scale["corpus_rows"])
        for top_k_text, curve_summary in scale["curves"].items():
            top_k = int(top_k_text)
            points = sorted(
                (point for point in scale["points"] if int(point["top_k"]) == top_k),
                key=lambda point: int(point["nprobe"]),
            )
            path = " ".join(
                f"{x_position(float(point['recall_at_k'])):.2f},"
                f"{y_position(float(point['gets_per_query'])):.2f}"
                for point in points
            )
            opacity = 1.0 if top_k == 10 else 0.55
            svg.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" '
                f'stroke-width="2" opacity="{opacity}"/>'
            )
            for point in points:
                probed_fraction = min(
                    float(point["ivf"]["probed_rows_per_query"]) / corpus_rows,
                    1.0,
                )
                radius = 3.5 + 11.0 * math.sqrt(probed_fraction)
                x = x_position(float(point["recall_at_k"]))
                y = y_position(float(point["gets_per_query"]))
                tooltip = html.escape(
                    f"N={corpus_rows:,}; top_k={top_k}; nprobe={point['nprobe']}; "
                    f"recall={point['recall_at_k']:.5f}; "
                    f"GET/q={point['gets_per_query']:.3f}; rows probed="
                    f"{point['ivf']['probed_rows_per_query']:.0f}"
                )
                if top_k == 10:
                    svg.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                        f'fill="{color}" fill-opacity=".68" stroke="{color}">'
                        f"<title>{tooltip}</title></circle>"
                    )
                else:
                    svg.append(
                        f'<rect x="{x - radius:.2f}" y="{y - radius:.2f}" '
                        f'width="{2 * radius:.2f}" height="{2 * radius:.2f}" '
                        f'fill="{color}" fill-opacity=".38" stroke="{color}">'
                        f"<title>{tooltip}</title></rect>"
                    )
            marker_profiles = (
                ("natural-knee", curve_summary["economy_profile"]),
                ("quality-floor", curve_summary["quality_profile"]),
            )
            for marker_class, profile in marker_profiles:
                knee = profile.get("point")
                if knee is None:
                    continue
                knee_x = x_position(float(knee["recall_at_k"]))
                knee_y = y_position(float(knee["gets_per_query"]))
                svg.append(
                    f'<circle class="{marker_class}" cx="{knee_x:.2f}" '
                    f'cy="{knee_y:.2f}" r="17"/>'
                )
        legend_x = left + scale_index * 185
        legend_y = 82
        svg.extend(
            [
                f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color}"/>',
                f'<text x="{legend_x + 12}" y="{legend_y + 4}" font-size="12">'
                f"{scale_label(corpus_rows)} (k_c={scale['coarse_cells']})</text>",
            ]
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" '
            'text-anchor="middle" font-size="14">Recall</text>',
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            f'font-size="14" transform="rotate(-90 24 {top + plot_height / 2:.2f})">'
            "Physical GET/query</text>",
            "</svg>",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.975)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-dual-svg", type=Path)
    parser.add_argument("--output-pareto-svg", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        raise RuntimeError("--target-recall must be in (0, 1]")
    outputs = [
        output
        for output in (
            args.output_json,
            args.output_svg,
            args.output_dual_svg,
            args.output_pareto_svg,
        )
        if output is not None
    ]
    if any(output.exists() for output in outputs):
        raise RuntimeError("refusing to overwrite analysis output")
    analysis = build_analysis(args.matrix, args.target_recall)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    write_svg(analysis, args.output_svg)
    if args.output_dual_svg is not None:
        write_dual_axis_svg(analysis, args.output_dual_svg)
    if args.output_pareto_svg is not None:
        write_pareto_svg(analysis, args.output_pareto_svg)
    print(f"analysis: {args.output_json}")
    print(f"chart: {args.output_svg}")
    if args.output_dual_svg is not None:
        print(f"dual-axis chart: {args.output_dual_svg}")
    if args.output_pareto_svg is not None:
        print(f"pareto chart: {args.output_pareto_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
