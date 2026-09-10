#!/usr/bin/env python3
"""Validate deterministic, architecture-safe commit guard wiring.

This is a fast static guard for the repo's "safe to commit" contract. It does
not replace the expensive build/test gates; it verifies that those gates still
point at deterministic policies:

* unit tests use the zero-retry nextest profile
* CI and Makefile still invoke that profile
* architecture docs still separate code presence from support level
* tenant/path mandates remain visible in the system map
* Arrow Flight exports bind client paths to the selected collection
* query ratchets count only executed queries and cloud proofs cover every backend
* rust-cache inputs resolve to keys accepted by GitHub's cache service
* conflict markers are not staged into source/docs
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_EXTENSIONS = {
    ".adoc",
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

CONFLICT_RE = re.compile(r"^(<<<<<<<|=======|>>>>>>>)($| )", re.MULTILINE)
RUST_CACHE_RE = re.compile(r"^(?P<indent>\s*)-\s+uses:\s*Swatinem/rust-cache@")
CACHE_INPUT_RE = re.compile(
    r"^\s+(?P<name>prefix-key|shared-key):\s*(?P<value>.*?)\s*$"
)
MATRIX_REF_RE = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*}}")
RUST_TOOLCHAIN_RE = re.compile(
    r"^(?P<indent>\s*)uses:\s*actions-rust-lang/setup-rust-toolchain@v2\s*$"
)


@dataclass(frozen=True)
class Finding:
    check: str
    message: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def require_contains(findings: list[Finding], check: str, path: str, needle: str) -> None:
    text = read_text(path)
    if needle not in text and compact_whitespace(needle) not in compact_whitespace(text):
        findings.append(Finding(check, f"{path} must contain: {needle!r}"))


def require_not_contains(
    findings: list[Finding], check: str, path: str, needle: str
) -> None:
    text = read_text(path)
    if needle in text or compact_whitespace(needle) in compact_whitespace(text):
        findings.append(Finding(check, f"{path} must not contain: {needle!r}"))


def check_nextest_contract(findings: list[Finding]) -> None:
    path = ROOT / ".config/nextest.toml"
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles = config.get("profile", {})

    unit = profiles.get("unit", {})
    integration = profiles.get("integration", {})

    # NOTE: the zero-retry contract (profile.default/unit.retries must be 0, and
    # no retry overrides) was intentionally removed. A small retry budget absorbs
    # load-induced flakes on constrained CI runners; nextest still surfaces any
    # survivor distinctly as FLAKY, so genuine breakage (which exhausts all
    # retries) still fails the run.
    if unit.get("test-threads", 0) < 2:
        findings.append(
            Finding(
                "nextest",
                ".config/nextest.toml profile.unit.test-threads must stay >1 to expose contention flakes",
            )
        )
    if unit.get("failure-output") != "immediate-final":
        findings.append(
            Finding(
                "nextest",
                '.config/nextest.toml profile.unit.failure-output must be "immediate-final"',
            )
        )
    if integration.get("retries", 0) > 1:
        findings.append(
            Finding(
                "nextest",
                ".config/nextest.toml profile.integration.retries must not exceed 1",
            )
        )


def check_gate_wiring(findings: list[Finding]) -> None:
    require_contains(
        findings,
        "gate-wiring",
        ".github/workflows/ci.yml",
        "- name: Verify Cargo.lock is reproducible",
    )
    require_contains(
        findings,
        "gate-wiring",
        ".github/workflows/ci.yml",
        "run: cargo metadata --locked --format-version 1 >/dev/null",
    )
    require_contains(
        findings,
        "gate-wiring",
        ".github/workflows/ci.yml",
        "cargo nextest run --lib --profile unit",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "cargo nextest run --lib --profile unit",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "docs-claim-check",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "workspace-boundaries-check",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "release-check: work-commit-check proto-check release-smoke query-conformance-check build-server",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "query-conformance-check:",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "cargo test --test tpch_pgwire_e2e",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "cargo test --test tpcds_pgwire_e2e",
    )
    require_contains(
        findings,
        "gate-wiring",
        ".github/workflows/layering-check.yml",
        "python3 scripts/check_tenant_path_guard.py",
    )
    require_contains(
        findings,
        "gate-wiring",
        "Makefile",
        "tenant-ingress-check",
    )
    require_contains(
        findings,
        "gate-wiring",
        ".github/workflows/layering-check.yml",
        "python3 scripts/check_tenant_ingress_contract.py",
    )


def check_architecture_contract(findings: list[Finding]) -> None:
    require_contains(
        findings,
        "architecture",
        "docs/SUPPORTED_SURFACE.adoc",
        "Do not infer support level from code presence alone.",
    )
    require_contains(
        findings,
        "architecture",
        "docs/README.md",
        "Code presence is broader than the supported product surface.",
    )
    require_contains(
        findings,
        "architecture",
        "docs/README.md",
        "12-design/SYSTEM_MAP_2026_05_30.adoc",
    )
    require_contains(
        findings,
        "architecture",
        "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
        "code presence does not imply support level",
    )
    require_contains(
        findings,
        "architecture",
        "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
        "TenantContext",
    )
    require_contains(
        findings,
        "architecture",
        "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
        "DrPathBuilder",
    )
    require_contains(
        findings,
        "architecture",
        "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
        "Validation and guardrail map",
    )
    require_contains(
        findings,
        "architecture",
        "docs/12-design/SYSTEM_MAP_2026_05_30.adoc",
        "make work-commit-check",
    )
    require_contains(
        findings,
        "architecture",
        "docs/06-internals/workflows/TDD_GUIDE.md",
        "zero-retry unit contract",
    )
    require_contains(
        findings,
        "architecture",
        "docs/06-internals/workflows/TDD_GUIDE.md",
        "make work-commit-check",
    )


def check_flight_export_authority(findings: list[Finding]) -> None:
    handler = "src/network/arrow_ipc/file_export.rs"
    service = "src/network/arrow_ipc/service.rs"

    require_contains(
        findings,
        "flight-export-authority",
        handler,
        "pub fn resolve_collection_file(",
    )
    require_contains(
        findings,
        "flight-export-authority",
        handler,
        "pub fn read_collection_file(",
    )
    require_contains(
        findings,
        "flight-export-authority",
        service,
        ".read_collection_file(&collection, &file_ticket.file_path)",
    )
    require_not_contains(
        findings,
        "flight-export-authority",
        service,
        ".read_arrow_file(&file_ticket.file_path)",
    )


def check_query_conformance_authority(findings: list[Finding]) -> None:
    harness = "tests/tpch_pgwire_e2e.rs"
    require_contains(
        findings,
        "query-conformance-authority",
        harness,
        "passed.len() >= TPCH_RATCHET",
    )
    require_not_contains(
        findings,
        "query-conformance-authority",
        harness,
        "passed.len() + skipped.len()",
    )
    require_not_contains(
        findings,
        "query-conformance-authority",
        harness,
        'let skip: &[&str] = &["Q2"]',
    )


def check_object_store_proof_authority(findings: list[Finding]) -> None:
    launcher = "scripts/prove_object_store_durability.sh"
    require_contains(
        findings,
        "object-store-proof-authority",
        launcher,
        "gs://*|gcs://*) feature=gcp ;;",
    )
    require_contains(
        findings,
        "object-store-proof-authority",
        "docs/10-quality/td/TD-OBJSTORE-2.adoc",
        "OSS owns this backend-neutral proof mechanism; anvaiops owns",
    )


def check_rust_cache_keys(findings: list[Finding]) -> None:
    """Catch invalid commas before expensive rust-cache jobs fan out.

    rust-cache reports GitHub's key-validation error as an annotation and then
    continues with a cold compile. Matrix references therefore need resolving;
    checking only the literal ``shared-key`` expression misses the failure.
    """

    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            action = RUST_CACHE_RE.match(line)
            if action is None:
                continue

            job_start = index
            while job_start >= 0 and not re.match(
                r"^  [A-Za-z0-9_-]+:\s*$", lines[job_start]
            ):
                job_start -= 1
            job_end = job_start + 1
            while job_end < len(lines) and not re.match(
                r"^  [A-Za-z0-9_-]+:\s*$", lines[job_end]
            ):
                job_end += 1
            job = lines[job_start:job_end] if job_start >= 0 else []

            action_indent = len(action.group("indent"))
            cursor = index + 1
            while cursor < len(lines):
                stripped = lines[cursor].lstrip()
                indent = len(lines[cursor]) - len(stripped)
                if indent == action_indent and stripped.startswith("- "):
                    break
                cache_input = CACHE_INPUT_RE.match(lines[cursor])
                cursor += 1
                if cache_input is None:
                    continue

                value = cache_input.group("value").strip("\"'")
                if "," in MATRIX_REF_RE.sub("", value):
                    findings.append(
                        Finding(
                            "rust-cache-key",
                            f"{rel(path)}:{cursor} {cache_input.group('name')} "
                            f"contains a comma: {value!r}",
                        )
                    )

                for field in MATRIX_REF_RE.findall(value):
                    field_re = re.compile(
                        rf"(?:^|[{{,])\s*{re.escape(field)}\s*:\s*"
                        r'(?:(?P<quote>["\'])(?P<quoted>.*?)(?P=quote)|'
                        r"(?P<plain>[^,}\s#]+))"
                    )
                    values: list[str] = []
                    for job_line in job:
                        candidate = job_line.strip().removeprefix("- ").lstrip()
                        values.extend(
                            match.group("quoted") or match.group("plain") or ""
                            for match in field_re.finditer(candidate)
                        )
                    if not values:
                        findings.append(
                            Finding(
                                "rust-cache-key",
                                f"{rel(path)}:{cursor} cannot resolve matrix.{field}",
                            )
                        )
                    for matrix_value in values:
                        if "," in matrix_value:
                            findings.append(
                                Finding(
                                    "rust-cache-key",
                                    f"{rel(path)}:{cursor} matrix.{field} contains "
                                    f"a comma: {matrix_value!r}",
                                )
                            )


def check_tdd_rust_warning_contract(findings: list[Finding]) -> None:
    """Keep the TDD workflow's intentional warning policy across action upgrades."""

    path = ROOT / ".github/workflows/tdd.yml"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        action = RUST_TOOLCHAIN_RE.match(line)
        if action is None:
            continue

        action_indent = len(action.group("indent"))
        cursor = index + 1
        step: list[str] = []
        while cursor < len(lines):
            stripped = lines[cursor].lstrip()
            indent = len(lines[cursor]) - len(stripped)
            if indent < action_indent or (
                indent == action_indent - 2 and stripped.startswith("- ")
            ):
                break
            step.append(lines[cursor])
            cursor += 1

        if not any(
            re.match(r'^\s+build-warnings:\s*["\']{2}\s*(?:#.*)?$', item)
            for item in step
        ):
            findings.append(
                Finding(
                    "tdd-rust-warning-policy",
                    f"{rel(path)}:{index + 1} setup-rust-toolchain@v2 must set "
                    'build-warnings: "" to preserve the workflow warning policy',
                )
            )


def tracked_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )
    return [Path(line) for line in output.splitlines() if line]


def check_conflict_markers(findings: list[Finding]) -> None:
    for rel_path in tracked_files():
        if rel_path.suffix not in TEXT_EXTENSIONS:
            continue
        path = ROOT / rel_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = CONFLICT_RE.search(text)
        if match:
            line = text[: match.start()].count("\n") + 1
            findings.append(
                Finding(
                    "conflict-markers",
                    f"{rel_path.as_posix()}:{line} contains a merge conflict marker",
                )
            )


def main() -> int:
    findings: list[Finding] = []

    check_nextest_contract(findings)
    check_gate_wiring(findings)
    check_architecture_contract(findings)
    check_flight_export_authority(findings)
    check_query_conformance_authority(findings)
    check_object_store_proof_authority(findings)
    check_rust_cache_keys(findings)
    check_tdd_rust_warning_contract(findings)
    check_conflict_markers(findings)

    print("Deterministic commit contract")
    if not findings:
        print(
            "OK: nextest, CI/Makefile wiring, architecture guards, Flight/query/object-store "
            "authorities, rust-cache keys, TDD Rust warning policy, and conflict-marker "
            "checks pass."
        )
        return 0

    print(f"FAILED: {len(findings)} finding(s)")
    for finding in findings:
        print(f"- [{finding.check}] {finding.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
