#!/usr/bin/env python3
"""Validate deterministic, architecture-safe commit guard wiring.

This is a fast static guard for the repo's "safe to commit" contract. It does
not replace the expensive build/test gates; it verifies that those gates still
point at deterministic policies:

* unit tests use the bounded-retry nextest profile and report survivors as flaky
* CI and Makefile still invoke that profile
* the main CI workflow uses its qualified, explicit Ubuntu image label
* architecture docs still separate code presence from support level
* tenant/path mandates remain visible in the system map
* Arrow Flight exports bind client paths to the selected collection
* query ratchets count only executed queries and cloud proofs cover every backend
* rust-cache inputs resolve to keys accepted by GitHub's cache service
* every Dockerfile a workflow `file:`-builds is visible to the `docker` path filter
* `serde_json` is declared with `preserve_order` where the OpenAPI doc is serialised
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

    # A fixed, small retry budget absorbs load-induced flakes on constrained CI
    # runners. It is part of the contract: increasing it can hide defects, while
    # removing flaky reporting makes retry survivors invisible.
    if profiles.get("default", {}).get("retries") != 2:
        findings.append(
            Finding(
                "nextest",
                ".config/nextest.toml profile.default.retries must stay at 2",
            )
        )
    if unit.get("retries") != 2:
        findings.append(
            Finding(
                "nextest",
                ".config/nextest.toml profile.unit.retries must stay at 2",
            )
        )
    if profiles.get("default", {}).get("final-status-level") != "flaky":
        findings.append(
            Finding(
                "nextest",
                '.config/nextest.toml profile.default.final-status-level must be "flaky"',
            )
        )
    unit_final_status = unit.get("final-status-level")
    if unit_final_status is not None and unit_final_status != "flaky":
        findings.append(
            Finding(
                "nextest",
                '.config/nextest.toml profile.unit.final-status-level override must be "flaky"',
            )
        )
    integration_final_status = integration.get("final-status-level")
    if integration_final_status is not None and integration_final_status != "flaky":
        findings.append(
            Finding(
                "nextest",
                '.config/nextest.toml profile.integration.final-status-level override must be "flaky"',
            )
        )
    for profile_name in ("default", "unit", "integration"):
        for override in profiles.get(profile_name, {}).get("overrides", []):
            if "retries" in override:
                findings.append(
                    Finding(
                        "nextest",
                        f".config/nextest.toml profile.{profile_name}.overrides must not set retries",
                    )
                )
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
    if integration.get("retries") != 1:
        findings.append(
            Finding(
                "nextest",
                ".config/nextest.toml profile.integration.retries must stay at 1",
            )
        )


def check_ci_runner_contract(findings: list[Finding]) -> None:
    """Guard this workflow's explicit block-style routing, without a YAML dependency."""
    path = ".github/workflows/ci.yml"
    selections = 0
    key = r'''(?:runs-on|'runs-on'|"runs-on")'''
    for line_number, line in enumerate(read_text(path).splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if re.search(rf"[{{,]\s*{key}\s*:", line):
            findings.append(
                Finding(
                    "ci-runner",
                    f"{path}:{line_number} must use block-style runner selection",
                )
            )
            continue
        match = re.match(rf"^\s*{key}\s*:\s*(.*?)\s*$", line)
        if match is None:
            continue
        selections += 1
        label = re.split(r"\s+#", match.group(1), maxsplit=1)[0].strip()
        if label not in ("ubuntu-24.04", "'ubuntu-24.04'", '"ubuntu-24.04"'):
            findings.append(
                Finding(
                    "ci-runner",
                    f"{path}:{line_number} must use explicit ubuntu-24.04; "
                    "ubuntu-latest aliases unqualified overflow runners",
                )
            )
    if selections == 0:
        findings.append(Finding("ci-runner", f"{path} has no runner selections"))


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
        "retry-budgeted unit contract",
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


PROMOTION_REFS = ("main", "develop", "qa")

# This module is deliberately dependency-free: `layering-check.yml` runs it on a
# runner with no PyYAML, so an `import yaml` here fails the job outright (it did
# -- `ModuleNotFoundError: No module named 'yaml'`). Parse the two blocks we need
# from the text, the same way the rest of this contract reads workflows.
_TOP_KEY_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<rest>.*)$")


def _top_level_block(text: str, key: str) -> list[str]:
    """Lines of a top-level mapping `key:`, excluding the key line itself."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        match = _TOP_KEY_RE.match(line)
        if match:
            if inside:
                break
            inside = match.group("key") == key
            continue
        if inside:
            out.append(line)
    return out


def _scalar_after(block: list[str], key: str, indent: int) -> str | None:
    """Value of `key:` at exactly `indent` spaces within `block`."""
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:\s*(?P<value>.*)$")
    for line in block:
        found = pattern.match(line)
        if found:
            return found.group("value").strip()
    return None


def _push_branches(text: str) -> list[str]:
    """Branches the workflow triggers on for `push`.

    Returns every promotion ref when `push` is present but unqualified, since an
    unqualified push trigger fires on all of them.
    """
    block = _top_level_block(text, "on")
    if not block:
        return []
    push_at: int | None = None
    for i, line in enumerate(block):
        if re.match(r"^  push:\s*$", line):
            push_at = i
            break
        if re.match(r"^  push:\s*\S", line) or re.match(r"^\s*-\s*push\s*$", line):
            return list(PROMOTION_REFS)
    if push_at is None:
        return []
    branches: list[str] = []
    in_branches = False
    for line in block[push_at + 1 :]:
        if re.match(r"^  \S", line):  # next key under `on:`
            break
        if re.match(r"^    branches:\s*$", line):
            in_branches = True
            continue
        inline = re.match(r"^    branches:\s*\[(?P<items>.*)\]\s*$", line)
        if inline:
            return [i.strip().strip("'\"") for i in inline.group("items").split(",") if i.strip()]
        if in_branches:
            item = re.match(r"^      -\s*(?P<name>\S+)\s*$", line)
            if item:
                branches.append(item.group("name").strip("'\""))
                continue
            if re.match(r"^    \S", line):
                in_branches = False
    return branches


# The single Node major this repo builds and publishes on. Raise it here and in
# every workflow in the same commit -- the check below is what keeps the two in
# step. Node 20 went to end-of-life maintenance in April 2026; 24 is the active
# LTS line.
NODE_MAJOR = "24"

# Matched anywhere in the step and in any spelling GitHub accepts, because each
# spelling it did NOT match was a silent pass:
#
#   * `uses:` on the bullet line (`ci.yml`), under a `- name:` (`release.yml`),
#     or under a `- if:` (the contract-gate matrix). Anchoring on `- uses:`
#     skipped the last two -- including the pin that builds the npm package the
#     release publishes.
#   * `- {uses: actions/setup-node@v7, with: {node-version: '20'}}` -- flow
#     style, where there is no bullet-plus-key line at all.
#   * `uses: "actions/setup-node@v7"` -- a quoted action ref.
#
# The sibling runner guard in this file is already tested against the last two
# (`test_quoted_runner_key_is_checked`, `test_inline_job_cannot_bypass_the_
# block_style_guard`); this check now is too.
SETUP_NODE_RE = re.compile(
    r"""^(?P<indent>\s*)(-\s*)?\{?\s*(?P<k>['"]?)uses(?P=k)\s*:\s*"""
    r"""['"]?actions/setup-node@"""
)
STEP_BULLET_RE = re.compile(r"^(?P<indent>\s*)-\s")
NODE_VERSION_RE = re.compile(
    r"""^[\s{,]*(?P<k>['"]?)node-version(?P=k)\s*:\s*"""
    r"""(?P<quote>['"]?)(?P<value>[^'"#\s,}]+)(?P=quote)\s*[,}]?\s*(#.*)?$"""
)
# `node-version-file:` points at an `.nvmrc`-style file instead of naming a
# version, so the major is invisible here and drifts with that file. Treated as
# unpinned rather than quietly accepted.
NODE_VERSION_FILE_RE = re.compile(r"""^[\s{,]*(?P<k>['"]?)node-version-file(?P=k)\s*:""")

# Unanchored twins, for the single line of a flow-style step: the anchored
# patterns above can only match a line's FIRST key, so on
# `- {uses: …, with: {node-version: '20'}}` they match nothing at all.
INLINE_NODE_VERSION_RE = re.compile(
    r"""(?P<k>['"]?)node-version(?P=k)\s*:\s*"""
    r"""(?P<quote>['"]?)(?P<value>[^'"#\s,}]+)(?P=quote)"""
)
INLINE_NODE_VERSION_FILE_RE = re.compile(r"""(?P<k>['"]?)node-version-file(?P=k)\s*:""")


def check_node_version_pins(findings: list[Finding]) -> None:
    """Every `actions/setup-node` step pins the one agreed Node major.

    Two failure modes this closes, both observed:

    1. **Drift back to an EOL major.** Four `ci.yml` jobs sat on Node 20 while
       `release.yml` had moved to 24, so the SDK the release published was never
       built on the runtime CI tested.
    2. **A lockfile that only resolves under one npm major.** `npm ci` fails
       closed when the lock is MISSING an entry the installing npm needs, and the
       majors hoist differently -- npm 11 prunes
       `node_modules/tailwindcss/node_modules/yaml`, which npm 10 requires. A
       lock regenerated on a developer's Node 24 therefore broke a Node 20 job
       with `Missing: yaml@2.9.1 from lock file`. Pinning one major everywhere
       means the lock is only ever resolved by the npm that generated it.

    An unpinned step (no `node-version:`) is a finding too: it silently tracks
    whatever the runner image ships, which is the same drift by another route.
    """
    # Both extensions, and composite actions too. GitHub honours `.yaml` as
    # readily as `.yml`, and a `uses: ./.github/actions/x` step runs whatever
    # that action's own `action.yml` sets up -- so globbing only
    # `workflows/*.yml` left a pin one `git mv` or one new directory away from
    # being unchecked. Neither gap is reachable in the tree today; both were
    # reachable in the first revision of this check.
    roots = [ROOT / ".github" / "workflows", ROOT / ".github" / "actions"]
    paths: list[Path] = []
    for directory in roots:
        if not directory.is_dir():
            continue
        for pattern in ("*.yml", "*.yaml", "**/*.yml", "**/*.yaml"):
            paths.extend(directory.glob(pattern))
    for path in sorted(set(paths)):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = SETUP_NODE_RE.match(line)
            if not match:
                continue
            # `with:` belongs to this step, so scan forward only while the
            # indentation stays deeper than the step's own bullet -- found by
            # walking back to the nearest `- ` at or above this line's indent.
            bullet = len(match.group("indent"))
            for back in reversed(lines[:index]):
                hit = STEP_BULLET_RE.match(back)
                if hit and len(hit.group("indent")) <= bullet:
                    bullet = len(hit.group("indent"))
                    break
            found: str | None = None
            for follow in lines[index + 1 :]:
                if not follow.strip():
                    continue
                if len(follow) - len(follow.lstrip()) <= bullet:
                    break
                if NODE_VERSION_FILE_RE.match(follow):
                    found = "<file>"
                    break
                version = NODE_VERSION_RE.match(follow)
                if version:
                    found = version.group("value")
                    break
            if found is None:
                # A flow-style step is a single line, so the forward scan above
                # never reaches its `with:` -- read the pin off the same line.
                if INLINE_NODE_VERSION_FILE_RE.search(line):
                    found = "<file>"
                else:
                    inline = INLINE_NODE_VERSION_RE.search(line)
                    if inline:
                        found = inline.group("value")
            if found is None:
                findings.append(
                    Finding(
                        "node-version-pin",
                        f"{rel(path)}:{index + 1}: `actions/setup-node` does not "
                        f"pin `node-version` -- it tracks whatever the runner "
                        f"image ships, so a runner bump changes the npm major "
                        f"and a lockfile generated under the other major fails "
                        f"`npm ci` closed. Pin `node-version: "
                        f"'{NODE_MAJOR}'`.",
                    )
                )
                continue
            if found == "<file>":
                findings.append(
                    Finding(
                        "node-version-pin",
                        f"{rel(path)}:{index + 1}: `actions/setup-node` resolves "
                        f"its version from `node-version-file`, so the Node "
                        f"major is not visible here and drifts with that file. "
                        f"Pin `node-version: '{NODE_MAJOR}'` instead.",
                    )
                )
                continue
            if found.split(".")[0] != NODE_MAJOR:
                findings.append(
                    Finding(
                        "node-version-pin",
                        f"{rel(path)}:{index + 1}: pins Node {found}, but this "
                        f"repo builds and publishes on Node {NODE_MAJOR} "
                        f"(NODE_MAJOR in scripts/"
                        f"check_deterministic_commit_contract.py). Mixed majors "
                        f"mean the lockfile is generated by one npm and "
                        f"installed by another, which fails `npm ci` closed.",
                    )
                )


def check_promotion_concurrency(findings: list[Finding]) -> None:
    """A workflow that runs on a promotion branch must not cancel its own run.

    The group key across this repo is
    `${{ github.workflow }}-${{ github.head_ref || github.ref_name }}`. For a
    `qa -> main` promotion the PR's `head_ref` IS `qa`, so the PR run and the
    push-to-qa run of the SAME workflow land in one group. With
    `cancel-in-progress` truthy for `qa`, whichever started second killed the
    other -- and `gh pr checks` renders a cancelled run as "fail", so the
    promotion PR displayed a phantom failure for a check that had passed on its
    own run. That was misdiagnosed twice during the v0.4.0 cut.

    Accepted spellings are a literal `false` (qa-gate.yml) or an expression that
    exempts every promotion ref it pushes on (ci.yml). The check applies ONLY
    when the group key collapses a push and a PR into one value, i.e. when it
    contains `head_ref`: a group keyed on `github.ref` alone is inherently safe,
    because a push is `refs/heads/<b>` and a PR is `refs/pull/<n>/merge`. An
    earlier revision omitted that condition and flagged `docs-site.yml`, which
    cannot collide.
    """
    workflow_dir = ROOT / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return
    for path in sorted(workflow_dir.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        touched = [b for b in _push_branches(text) if b in PROMOTION_REFS]
        if not touched:
            continue
        block = _top_level_block(text, "concurrency")
        if not block:
            continue
        cancel = _scalar_after(block, "cancel-in-progress", indent=2)
        if cancel is None:
            continue
        group = _scalar_after(block, "group", indent=2) or ""
        if "head_ref" not in group:
            continue
        if cancel.strip().lower() in {"false", "'false'", '"false"'}:
            continue
        missing = [
            ref
            for ref in touched
            if not any(
                token in cancel
                for token in (
                    f"'{ref}'",
                    f'"{ref}"',
                    f"'refs/heads/{ref}'",
                    f'"refs/heads/{ref}"',
                )
            )
        ]
        if missing:
            findings.append(
                Finding(
                    "promotion-concurrency",
                    f"{rel(path)}: runs on push to {', '.join(touched)} but "
                    f"`cancel-in-progress` does not exempt {', '.join(missing)} "
                    f"-- the push run and a promotion PR run share one "
                    f"concurrency group, so one cancels the other and the PR "
                    f"shows a phantom failure. Use `false`, or exempt every "
                    f"promotion ref as ci.yml does.",
                )
            )


# A `#` starts a YAML comment only after whitespace (or at line start); a
# mid-scalar `#` (`file: deploy/Dockerfile#x`) is literal and must stay in the
# value, so the comment tail requires at least one space. Trailing whitespace
# with no comment is still a match (`(?:\s+(#.*)?)?` — the space is part of
# the optional group).
DOCKER_FILE_TARGET_RE = re.compile(r"^\s*file:\s*(?P<value>\S+?)(?:\s+(#.*)?)?$")
DOCKER_FILTER_KEY_RE = re.compile(r"^(?P<indent>\s*)docker:\s*$")
DOCKER_FILTER_ITEM_RE = re.compile(r"^\s*-\s*(?P<pattern>\S+)\s*$")
CI_WORKFLOW = ".github/workflows/ci.yml"


def _compile_glob_segment(segment: str) -> str:
    """Compile one path segment: ``*``/``?`` stay within the segment.

    A mid-segment ``**`` (``a**b``) is NOT a picomatch globstar -- only a
    whole-segment ``**`` crosses ``/`` -- so here it degrades to ``*``
    (segment-local), the fail-closed direction for a subset matcher.
    """
    out = ""
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            out += "[^/]*"
            index += 1
        elif char == "?":
            out += "[^/]"
            index += 1
        else:
            out += re.escape(char)
            index += 1
    return out


def _compile_glob(segments: list[str]) -> str:
    """Compile split segments to a regex picomatch-subset.

    Whole-segment ``**`` is a globstar: as the last segment it covers the
    directory itself and any depth under it; elsewhere it spans zero or more
    whole segments (``**/Dockerfile`` matches root ``Dockerfile``).
    """
    if not segments:
        return ""
    head, tail = segments[0], segments[1:]
    if head == "**":
        if not tail:
            return ".*"
        return f"(?:[^/]+/)*{_compile_glob(tail)}"
    head_expr = _compile_glob_segment(head)
    if tail == ["**"]:
        if "*" in head:
            # picomatch (2.3.1 and 4.0.5 alike) declines the bare-directory
            # allowance when the segment before a trailing globstar ENDS in a
            # wildcard: `*/**` does NOT match `Dockerfile`, while `a/**`,
            # `a?/**` and literal heads do. We decline for any head CONTAINING
            # a `*` — slightly stricter than picomatch for star-leading heads
            # (`*a/**` vs `ba`), which is the fail-closed direction for a
            # subset matcher. Granting the wildcard case would be an
            # over-match — the one direction a subset matcher must never take.
            return f"{head_expr}/.+"
        return f"{head_expr}(?:/.*)?"
    if not tail:
        return head_expr
    return f"{head_expr}/{_compile_glob(tail)}"


def _picomatch_subset_matches(pattern: str, path: str) -> bool:
    """Match a dorny/paths-filter glob against a repo-relative path.

    Supports the subset this repo's filters use: literal segments, ``*``
    (within one segment) and whole-segment ``**`` (across segments, including
    zero segments). picomatch matches bare patterns against top-level paths
    only when they contain no ``/`` -- the exact trap TD-DOCKCI-1 documents --
    so ``Dockerfile*`` deliberately does NOT match ``deploy/docker/Dockerfile``
    here either. This is a deliberate SUBSET: when in doubt it declines, so a
    filter the guard accepts is one picomatch certainly matches.
    """
    pattern = pattern.strip().strip("\"'")
    # Collapse consecutive globstars (`**/**` == `**`).
    collapsed: list[str] = []
    for segment in pattern.split("/"):
        if segment == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(segment)
    return re.fullmatch(_compile_glob(collapsed), path) is not None


def _ci_docker_filter_patterns() -> list[str] | None:
    """Extract the `docker:` filter patterns from ci.yml, or None if absent."""
    text = read_text(CI_WORKFLOW)
    lines = text.splitlines()

    def is_noise(line: str) -> bool:
        stripped = line.strip()
        return not stripped or stripped.startswith("#")

    for index, line in enumerate(lines):
        key = DOCKER_FILTER_KEY_RE.match(line)
        if key is None:
            continue
        # A structural `docker:` key (mapping child or empty) is not the
        # filter; the filter block's first content line is a list item.
        first = index + 1
        while first < len(lines) and is_noise(lines[first]):
            first += 1
        if first >= len(lines) or DOCKER_FILTER_ITEM_RE.match(lines[first]) is None:
            continue
        key_indent = len(key.group("indent"))
        patterns: list[str] = []
        cursor = first
        while cursor < len(lines):
            if is_noise(lines[cursor]):
                cursor += 1
                continue
            item = DOCKER_FILTER_ITEM_RE.match(lines[cursor])
            item_indent = len(lines[cursor]) - len(lines[cursor].lstrip())
            if item is None or item_indent <= key_indent:
                break
            patterns.append(item.group("pattern"))
            cursor += 1
        # A matrix axis named `docker` (`- linux`) is structurally identical
        # to the filter; path filters always carry a glob, slash or quote.
        # Misclassifying the real filter degenerates to the missing-filter
        # finding (fail-closed), never to silence.
        if patterns and not any(any(c in p for c in '/*"\'') for p in patterns):
            continue
        return patterns
    return None


def check_docker_filter_covers_build_targets(findings: list[Finding]) -> None:
    """Every `file:` Dockerfile a workflow builds must trip the `docker` filter.

    TD-DOCKCI-1: the old filter watched three top-level paths CI never builds
    and could not see `deploy/docker/**` -- the Dockerfile `docker-build` and
    `publish-image.yml` actually build -- so an image-breaking edit skipped the
    only job that validates it. This is the machine-checkable form: any
    `file:` target under .github/workflows that no `docker` filter pattern
    matches is a finding. Note the reverse leak (a watched path nothing
    builds) is a waste, not a correctness hole, and is left to review.
    """
    patterns = _ci_docker_filter_patterns()
    if patterns is None:
        findings.append(
            Finding(
                "docker-filter",
                f"{CI_WORKFLOW} has no `docker:` path filter, but "
                f"docker-build gates on it -- the gate cannot see the change "
                f"that defines it.",
            )
        )
        return

    workflow_dir = ROOT / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return
    for path in sorted(workflow_dir.glob("*.y*ml")):
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            target = DOCKER_FILE_TARGET_RE.match(line)
            if target is None:
                continue
            value = target.group("value").strip().strip("\"'")
            if "Dockerfile" not in value:
                continue
            if not any(_picomatch_subset_matches(p, value) for p in patterns):
                suggested = (
                    f"{value.rsplit('/', 1)[0]}/**" if "/" in value else f"**/{value}"
                )
                findings.append(
                    Finding(
                        "docker-filter",
                        f"{rel(path)}:{index} builds `{value}` but no `docker` "
                        f"filter pattern in {CI_WORKFLOW} matches it -- an edit "
                        f"confined to that Dockerfile skips docker-build "
                        f"(TD-DOCKCI-1). Add a pattern covering it, e.g. "
                        f"`{suggested}`.",
                    )
                )


def check_serde_json_preserve_order(findings: list[Finding]) -> None:
    """`serde_json` must be declared with `preserve_order` where the spec is built.

    TD-SPECORD-1: the published OpenAPI document's key order rode on
    serde_json/preserve_order being enabled TRANSITIVELY (DataFusion supplied
    it until 55.2 deliberately stopped). When the feature drops, every map in
    the generated spec serialises alphabetically and the drift gate fails with
    a 9k-line, semantically empty diff. Both declarations are checked: the
    root `[dependencies]` entry (what the serialising binary actually
    compiles) and `[workspace.dependencies]` (what the 68 members inheriting
    `serde_json.workspace = true` read).
    """
    manifest = ROOT / "Cargo.toml"
    if not manifest.is_file():
        return
    with open(manifest, "rb") as handle:
        try:
            parsed = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            findings.append(
                Finding("serde-json-preserve-order", f"Cargo.toml does not parse: {error}")
            )
            return

    for section_path, where in (
        (("dependencies",), "[dependencies]"),
        (("workspace", "dependencies"), "[workspace.dependencies]"),
    ):
        table: object = parsed
        for key in section_path:
            table = table.get(key) if isinstance(table, dict) else None
        entry = table.get("serde_json") if isinstance(table, dict) else None
        features = entry.get("features") if isinstance(entry, dict) else None
        if features is None or "preserve_order" not in features:
            findings.append(
                Finding(
                    "serde-json-preserve-order",
                    f"Cargo.toml {where}.serde_json must be declared with "
                    f'`features = ["preserve_order"]` (TD-SPECORD-1: without '
                    f"it the generated OpenAPI document serialises "
                    f"alphabetically and every SDK regenerates as a 9k-line "
                    f"diff). Found: {entry!r}",
                )
            )


def main() -> int:
    findings: list[Finding] = []

    check_nextest_contract(findings)
    check_ci_runner_contract(findings)
    check_gate_wiring(findings)
    check_architecture_contract(findings)
    check_flight_export_authority(findings)
    check_query_conformance_authority(findings)
    check_object_store_proof_authority(findings)
    check_rust_cache_keys(findings)
    check_tdd_rust_warning_contract(findings)
    check_conflict_markers(findings)
    check_promotion_concurrency(findings)
    check_node_version_pins(findings)
    check_docker_filter_covers_build_targets(findings)
    check_serde_json_preserve_order(findings)

    print("Deterministic commit contract")
    if not findings:
        print(
            "OK: nextest, CI runner/Makefile wiring, architecture guards, Flight/query/object-store "
            "authorities, rust-cache keys, TDD Rust warning policy, promotion concurrency, "
            "Node version pins, docker-filter build targets, serde_json preserve_order, and "
            "conflict-marker checks pass."
        )
        return 0

    print(f"FAILED: {len(findings)} finding(s)")
    for finding in findings:
        print(f"- [{finding.check}] {finding.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
