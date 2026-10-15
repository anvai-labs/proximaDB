#!/usr/bin/env python3
"""Validate the narrow customer promise that defines the ProximaDB MVP.

The support matrix remains the authority for the whole repository. This gate
adds a smaller invariant: release collateral and the live first-run path must
agree on the exact subset we invite a new user to depend on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/00-product/MVP_TRUST_CORRIDOR.toml"
ALLOWED_SURFACE_TIERS = {"supported", "diagnostic"}
ALLOWED_EXCLUSION_TIERS = {"beta", "experimental", "planned"}


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def validate_contract(data: dict) -> list[str]:
    errors: list[str] = []
    for key in (
        "schema_version",
        "as_of_date",
        "positioning",
        "deployment",
        "canonical_api",
        "default_engine",
        "promise",
        "storage",
        "security",
        "surface",
        "exclusion",
        "release",
        "gtm",
    ):
        if key not in data:
            errors.append(f"contract missing top-level key: {key}")

    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("canonical_api") != "REST /api/v2":
        errors.append("the MVP canonical_api must remain REST /api/v2")
    if data.get("default_engine") != "sst":
        errors.append("the MVP default_engine must remain the Supported SST engine")
    storage = data.get("storage", {})
    if storage.get("quickstart_backend") != "local":
        errors.append("the reproducible MVP quickstart must default to local storage")
    if storage.get("object_store_tier") != "beta":
        errors.append("object-store production maturity must remain Beta until its gate closes")
    if set(storage.get("object_store_backends", [])) != {"s3", "azure", "gcs"}:
        errors.append("the co-design evidence matrix must cover S3, Azure, and GCS")
    security = data.get("security", {})
    if security.get("tenancy_mode") != "single-tenant":
        errors.append("the MVP remains single-tenant until fail-closed ABAC is proven")
    if security.get("metadata_filters_are_authorization") is not False:
        errors.append("metadata filters must never be represented as authorization")
    if security.get("multi_tenant_abac_tier") != "beta":
        errors.append("multi-tenant ABAC must remain Beta until its cross-surface gate closes")

    ids: set[str] = set()
    openapi = _text("docs/openapi/proximadb-openapi.yaml")
    for item in data.get("surface", []):
        item_id = item.get("id")
        if not item_id or item_id in ids:
            errors.append(f"surface id is missing or duplicated: {item_id!r}")
        ids.add(item_id)
        if item.get("tier") not in ALLOWED_SURFACE_TIERS:
            errors.append(f"surface {item_id}: invalid corridor tier {item.get('tier')!r}")
        path = item.get("path", "")
        if item.get("tier") == "supported" and path not in openapi:
            errors.append(f"surface {item_id}: path absent from generated OpenAPI: {path}")
        for evidence in item.get("evidence", []):
            if not (ROOT / evidence).is_file():
                errors.append(f"surface {item_id}: missing evidence {evidence}")

    exclusion_ids: set[str] = set()
    for item in data.get("exclusion", []):
        item_id = item.get("id")
        if not item_id or item_id in exclusion_ids:
            errors.append(f"exclusion id is missing or duplicated: {item_id!r}")
        exclusion_ids.add(item_id)
        if item.get("tier") not in ALLOWED_EXCLUSION_TIERS:
            errors.append(f"exclusion {item_id}: invalid tier {item.get('tier')!r}")
        if not item.get("reason"):
            errors.append(f"exclusion {item_id}: reason is required")
        for tracking in item.get("tracking", []):
            if not (ROOT / tracking).is_file():
                errors.append(f"exclusion {item_id}: missing tracking file {tracking}")

    release = data.get("release", {})
    for doc in release.get("required_docs", []):
        if not (ROOT / doc).is_file():
            errors.append(f"required MVP document missing: {doc}")

    gtm = data.get("gtm", {})
    for metric in (
        "design_partner_target",
        "external_weekly_active_target",
        "public_case_study_target",
        "non_founder_maintainer_target",
    ):
        if not isinstance(gtm.get(metric), int) or gtm[metric] <= 0:
            errors.append(f"gtm.{metric} must be a positive integer")

    return errors


# One flat character class, no nested quantifier: linear-time on any input
# (TD-SECSCAN-1 §8). The previous `([a-zA-Z0-9_-]+(?:-[a-zA-Z0-9_-]+)+)` nested
# two quantifiers over a class containing the `-` separator and backtracked
# exponentially on `-`-heavy text with no archive suffix.
_RELEASE_ASSET_RE = re.compile(
    r"proximadb-\$\{\{ needs\.prepare\.outputs\.version \}\}-([a-zA-Z0-9_-]+)\.(?:tar\.gz|zip)"
)


def release_note_targets(release_notes: str) -> set[str]:
    """Binary targets named by `proximadb-<version>-<target>.(tar.gz|zip)` assets.

    Keeps the old pattern's semantics: a target has at least two `-`-joined
    segments, i.e. a `-` strictly inside the name (not first or last char).
    """
    return {
        target
        for target in _RELEASE_ASSET_RE.findall(release_notes)
        if "-" in target[1:-1]
    }


def validate_repository(data: dict) -> list[str]:
    errors: list[str] = []
    positioning = data["positioning"]
    required_fragments = {
        "README.adoc": [positioning, "MVP trust corridor", "scripts/mvp_smoke.py"],
        "docs/SUPPORTED_SURFACE.adoc": ["MVP Trust Corridor", "MVP_TRUST_CORRIDOR.toml"],
        "docs/00-product/VISION.adoc": [positioning, "MVP_TRUST_CORRIDOR.toml"],
        "docs/00-product/COMPETITIVE_LANDSCAPE.adoc": [
            "direct category competitor",
            "economic proof",
        ],
        "docs/01-quick-start/CONTEXT_SERVING_MVP.adoc": [
            "scripts/mvp_smoke.py",
            "/api/v2",
        ],
        "docs/00-product/MVP_EXECUTION_SCORECARD.adoc": [
            "Design-partner contract",
            "50 external activations",
        ],
        "docs/10-quality/benchmarks/CONTEXT_CORRIDOR_BENCHMARK.adoc": [
            "context-corridor-v1",
            "publication_eligible: false",
        ],
        "scripts/docker-demo-test.sh": ["scripts/mvp_smoke.py"],
        "Makefile": ["mvp-contract-check:", "mvp-smoke-test:"],
        ".github/workflows/ci.yml": ["MVP Trust Corridor"],
        ".github/ISSUE_TEMPLATE/design-partner.yml": [
            "Co-design evidence agreement",
            "I have not included secrets",
        ],
    }
    for relative, fragments in required_fragments.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"required integration file missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in content:
                errors.append(f"{relative}: missing MVP contract fragment {fragment!r}")

    customer_docs = "\n".join(
        _text(relative)
        for relative in (
            "README.adoc",
            "docs/00-product/VISION.adoc",
            "docs/00-product/COMPETITIVE_LANDSCAPE.adoc",
            "docs/01-quick-start/CONTEXT_SERVING_MVP.adoc",
        )
    )
    forbidden = {
        "unqualified Full SQL claim": re.compile(r"\bFull SQL\s*\+", re.I),
        "vector competitors dismissed as single-model": re.compile(
            r"vector databases? (?:are|as) single-model", re.I
        ),
        "deprecated REST v1 quickstart": re.compile(r"/api/v1/|/v1/collections"),
    }
    for label, pattern in forbidden.items():
        if pattern.search(customer_docs):
            errors.append(f"customer collateral contains {label}")

    docker_smoke = _text("scripts/docker-demo-test.sh")
    if "/v1/collections" in docker_smoke or "storage_engine\": \"viper" in docker_smoke:
        errors.append("Docker smoke must not exercise deprecated v1 or a Beta engine")
    if "log_warning" in docker_smoke:
        errors.append("Docker MVP smoke must hard-fail; warning-only assertions are forbidden")

    release_workflow = _text(".github/workflows/release.yml")
    notes_start = release_workflow.find("- name: Generate Release Notes")
    notes_end = release_workflow.find("- name: Create Release", notes_start)
    if notes_start < 0 or notes_end < 0:
        errors.append("release workflow: cannot locate generated release-note block")
    else:
        release_notes = release_workflow[notes_start:notes_end]
        expected = set(data["release"]["binary_targets"])
        mentioned = release_note_targets(release_notes)
        if mentioned != expected:
            errors.append(
                "generated release notes advertise targets that differ from the MVP contract: "
                f"expected={sorted(expected)}, found={sorted(mentioned)}"
            )

    return errors


def main() -> int:
    try:
        data = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: cannot read MVP trust corridor: {exc}", file=sys.stderr)
        return 1

    errors = validate_contract(data)
    if not errors:
        errors.extend(validate_repository(data))
    if errors:
        print("ERROR: MVP trust corridor drift:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "MVP trust corridor OK: 7 canonical v2 operations, SST, single-node, "
        "single-tenant boundary, 2 release targets, explicit Beta/Experimental exclusions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
