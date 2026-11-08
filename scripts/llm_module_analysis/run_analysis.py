"""Driver for per-module LLM analysis over victor graph.

Reads precomputed dossiers from `report/llm_module_analysis/dossiers/`,
renders a prompt template per module, calls `victor run --json -f <prompt-file>`,
and collects findings into `report/llm_module_analysis/<module>.{md,json}`.

Resume via `state.json` — tracks which modules are done, in-flight, or pending.
Concurrency 2 (configurable).
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROMPT_TEMPLATE = """\
You are analyzing the ProximaDB module `{module_path}` for code quality issues.

## Module Dossier

The dossier (JSON) below carries precomputed signals from the victor graph:
- hotspot_score, pagerank, afferent/efferent coupling, instability, LCOM4 cohesion, TDD priority
- top-degree public symbols (most-referenced)
- cross-layer suspect edges (layer violations)
- structural clone families intersecting this module (from jscpd/AST analysis)
- zero-reference public symbols (dead code candidates)
- complexity tops (top-5 functions by signature length / statement count)

```json
{dossier_json}
```

## Your Task

Identify concrete, actionable findings in these categories:
1. **duplicate-suspect**: functions/blocks that look like duplicates of other code (named symbols required)
2. **coupling-violation**: layer violations or tight coupling that should be refactored
3. **dead-code**: public symbols with zero references (verify they're truly unused)
4. **complexity**: functions that are too complex and should be split
5. **td-candidate**: patterns that warrant a Technical Design doc (architectural debt)
6. **exempt-clone**: generated code, test scaffolding, or deliberate duplication

For each finding:
- **category**: one of the above
- **title**: short label (≤60 chars)
- **summary**: one-sentence description
- **file**: repo-relative path
- **line**: 1-indexed line number (if known)
- **confidence**: 0.0–1.0 (only report ≥0.6)
- **evidence**: named symbols or file:line references
- **recommendation**: what to do (refactor, delete, file TD, etc.)

Output format: JSON object with a `findings` array (max 8 per category).

## Constraints

- Do NOT rediscover what the scanner already found — triage the clone families as
  `exempt-clone` (generated/test) or `duplicate-suspect` (real), with evidence.
- Do NOT report findings you cannot name (require file:line or symbol names).
- Do NOT report low-confidence speculation (≥0.6 only).
- If the module is clean, return an empty findings array.

Respond with ONLY the JSON object (no markdown fences, no commentary).
"""


@dataclass
class AnalysisState:
    module_path: str
    status: str  # pending | in-flight | done | failed
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    findings_count: int = 0
    error: Optional[str] = None


@dataclass
class RunState:
    modules: dict[str, AnalysisState] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunState:
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text())
        modules = {k: AnalysisState(**v) for k, v in data.get("modules", {}).items()}
        return cls(modules=modules, started_at=data.get("started_at", time.time()))


def render_prompt(dossier_path: Path) -> str:
    """Render the prompt template for one module."""
    dossier = json.loads(dossier_path.read_text())
    module_path = dossier["module_path"]
    return PROMPT_TEMPLATE.format(
        module_path=module_path,
        dossier_json=json.dumps(dossier, indent=2),
    )


_BANNER_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d+ - ")


def _balance_braces(s: str, start: int) -> int:
    """Index of the '}' matching the '{' at s[start], string-aware; -1 if unbalanced."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _try_parse(candidate: str) -> dict:
    """Parse a JSON candidate, tolerating hard line-wraps (newline -> space)."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(candidate.replace("\n", " "))


def _extract_json(output_text: str) -> dict:
    """Extract the findings JSON from a `victor run --json` stdout transcript.

    Observed envelope (verified empirically 2026-11-04): banner lines
    (``✓ …``), indented tool-output chrome (``[File: …]``, ``… N more
    lines``), then the model's reply LAST — hard-wrapped at ~80 columns, so a
    wrap can even land inside the JSON. The reply may also carry markdown
    fences or leading prose. Tool chatter can contain braces, so anchoring on
    the FIRST '{' mis-anchors; anchor on the LAST relevant token instead.
    """
    content_lines = []
    for line in output_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("✓", "✗", "❌", "💡", "•")):
            continue
        if _BANNER_RE.match(stripped):
            continue
        content_lines.append(line)
    content = "\n".join(content_lines).strip()

    # 1. Direct parse (clean envelope: banners + reply only)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. LAST ```json fence (fenced replies sit after tool chatter)
    if "```json" in content:
        try:
            block = content.rsplit("```json", 1)[1].split("```", 1)[0]
            return _try_parse(block.strip())
        except (IndexError, json.JSONDecodeError):
            pass

    # 3. Anchor on the LAST "findings" key: nearest '{' before it, balance forward
    anchor = content.rfind('"findings"')
    if anchor != -1:
        start = content.rfind("{", 0, anchor)
        if start != -1:
            end = _balance_braces(content, start)
            if end > start:
                try:
                    return _try_parse(content[start : end + 1])
                except json.JSONDecodeError:
                    pass

    # 4. Last resort: last '{' whose balanced object reaches the content tail
    pos = len(content)
    while (pos := content.rfind("{", 0, pos)) != -1:
        end = _balance_braces(content, pos)
        if end != -1 and end >= len(content) - 5:
            try:
                return _try_parse(content[pos : end + 1])
            except json.JSONDecodeError:
                pass

    # 5. Python-literal reply: models sometimes emit {'findings': ...} with
    # single quotes / True / None. Balance from the last 'findings' anchor
    # and evaluate as a Python literal, then require it to be dict-shaped.
    anchor = content.rfind("'findings'")
    if anchor == -1:
        anchor = content.rfind('"findings"')
    if anchor != -1:
        start = max(content.rfind("{", 0, anchor), 0)
        end = _balance_braces(content, start)
        if end > start:
            try:
                obj = ast.literal_eval(content[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except (ValueError, SyntaxError):
                pass

    raise ValueError(f"no JSON found in victor output: {content[:300]!r}")


async def analyze_module(
    module_path: str,
    dossier_path: Path,
    output_dir: Path,
    state: RunState,
    state_path: Path,
    *,
    victor_bin: str = "victor",
    profile: str = "homelab-zai-coding",
    timeout: int = 600,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the LLM analysis for one module."""
    if module_path not in state.modules:
        state.modules[module_path] = AnalysisState(
            module_path=module_path, status="pending"
        )
    entry = state.modules[module_path]

    if entry.status == "done":
        return  # already done

    entry.status = "in-flight"
    entry.started_at = time.time()
    state.save(state_path)

    # Render prompt
    prompt = render_prompt(dossier_path)
    prompt_file = output_dir / f"{module_path.replace('/', '__')}.prompt.md"
    prompt_file.write_text(prompt)

    # Call victor run with profile
    cmd = [
        victor_bin,
        "run",
        "analyze",
        "-f",
        str(prompt_file),
        "--json",
        "-p",
        profile,
    ]
    cwd = repo_root or Path.cwd()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            entry.status = "failed"
            entry.error = (
                f"victor run exited {proc.returncode}: {stderr.decode()[:500]}"
            )
            state.save(state_path)
            return

        # Parse output (strip victor banner envelope, tolerate fences/prose)
        output_text = stdout.decode()
        try:
            result = _extract_json(output_text)
        except ValueError as e:
            entry.status = "failed"
            entry.error = f"JSON extraction failed: {e}"
            state.save(state_path)
            return

        # Extract findings
        findings = result.get("findings", [])
        findings_count = len(findings)

        # Write output
        md_path = output_dir / f"{module_path.replace('/', '__')}.md"
        json_path = output_dir / f"{module_path.replace('/', '__')}.json"

        # Markdown report
        md_lines = [f"# LLM Analysis: {module_path}\n"]
        md_lines.append(f"**Findings**: {findings_count}\n")
        for i, f in enumerate(findings, 1):
            md_lines.append(
                f"## {i}. [{f.get('category', 'unknown')}] {f.get('title', 'untitled')}\n"
            )
            md_lines.append(f"**Summary**: {f.get('summary', '')}\n")
            md_lines.append(f"**File**: {f.get('file', '?')}:{f.get('line', '?')}\n")
            md_lines.append(f"**Confidence**: {f.get('confidence', 0.0)}\n")
            md_lines.append(f"**Evidence**: {f.get('evidence', '')}\n")
            md_lines.append(f"**Recommendation**: {f.get('recommendation', '')}\n")
        md_path.write_text("\n".join(md_lines))

        # JSON output
        json_path.write_text(json.dumps(result, indent=2))

        entry.status = "done"
        entry.finished_at = time.time()
        entry.findings_count = findings_count
        state.save(state_path)

    except asyncio.TimeoutError:
        # Kill the wedged victor agent: without this the orphaned subprocess
        # keeps spending the LLM profile after the driver moves on.
        proc.kill()
        await proc.wait()
        entry.status = "failed"
        entry.error = f"Timeout after {timeout}s"
        state.save(state_path)
    except Exception as e:
        entry.status = "failed"
        entry.error = str(e)
        state.save(state_path)


def _module_path_of(dossier_path: Path) -> str:
    """Authoritative module path: read the dossier's own module_path field.

    Filename-decoding is lossy for ``__init__.py`` (its underscores collide
    with the ``__``-as-separator encoding), so the dossier JSON is the source
    of truth; the filename is only a fallback.
    """
    try:
        return json.loads(dossier_path.read_text())["module_path"]
    except (json.JSONDecodeError, KeyError, OSError):
        return dossier_path.stem.replace("__", "/")


async def run_analysis(
    dossiers_dir: Path,
    output_dir: Path,
    state_path: Path,
    *,
    concurrency: int = 2,
    victor_bin: str = "victor",
    profile: str = "homelab-zai-coding",
    timeout: int = 600,
    repo_root: Optional[Path] = None,
) -> None:
    """Run analysis for all pending modules with the given concurrency."""
    state = RunState.load(state_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pending modules
    for dossier_path in sorted(dossiers_dir.glob("*.json")):
        module_path = _module_path_of(dossier_path)
        if module_path not in state.modules:
            state.modules[module_path] = AnalysisState(
                module_path=module_path, status="pending"
            )
    state.save(state_path)

    # Run with concurrency
    sem = asyncio.Semaphore(concurrency)

    async def worker(module_path: str, dossier_path: Path) -> None:
        async with sem:
            await analyze_module(
                module_path,
                dossier_path,
                output_dir,
                state,
                state_path,
                victor_bin=victor_bin,
                profile=profile,
                timeout=timeout,
                repo_root=repo_root,
            )

    tasks = []
    for dossier_path in sorted(dossiers_dir.glob("*.json")):
        module_path = _module_path_of(dossier_path)
        entry = state.modules.get(module_path)
        if entry and entry.status != "done":
            tasks.append(worker(module_path, dossier_path))

    print(f"Analyzing {len(tasks)} modules (concurrency={concurrency})")
    await asyncio.gather(*tasks)

    # Write rollup
    rollup = {
        "total_modules": len(state.modules),
        "done": sum(1 for m in state.modules.values() if m.status == "done"),
        "failed": sum(1 for m in state.modules.values() if m.status == "failed"),
        "pending": sum(1 for m in state.modules.values() if m.status == "pending"),
        "total_findings": sum(m.findings_count for m in state.modules.values()),
        "elapsed_seconds": time.time() - state.started_at,
    }
    rollup_path = output_dir / "rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2))
    print(
        f"Done. {rollup['done']} done, {rollup['failed']} failed, {rollup['total_findings']} findings"
    )


def selftest(dossiers_dir: Path, output_dir: Path) -> bool:
    """Run selftest checks.

    - catalog get_statistics ×5 MUST be found (A1 dedup)
    - generated families classified exempt
    - 3 sampled dead-code findings grep-verified
    """
    print("Running selftest...")
    checks_passed = 0
    checks_total = 3

    # 1. catalog get_statistics ×5 found
    catalog_dossier = dossiers_dir / "crates__control__proximadb-catalog.json"
    if catalog_dossier.is_file():
        dossier = json.loads(catalog_dossier.read_text())
        clone_families = dossier.get("clone_families", [])
        # Check if any family has 5+ members from the catalog crate
        found_5x = False
        for fam_id, sim, members in clone_families:
            catalog_members = [m for m in members if "proximadb-catalog" in m[0]]
            if len(catalog_members) >= 5:
                found_5x = True
                break
        if found_5x:
            print("✓ catalog get_statistics ×5 clone family detected")
            checks_passed += 1
        else:
            print("✗ catalog get_statistics ×5 NOT found in dossier")
    else:
        print("✗ catalog dossier not found")

    # 2. generated families classified exempt (check a sample output)
    sample_output = output_dir / "clients__python__src__proximadb_sdk__codegen.json"
    if sample_output.is_file():
        result = json.loads(sample_output.read_text())
        findings = result.get("findings", [])
        exempt_count = sum(1 for f in findings if f.get("category") == "exempt-clone")
        if exempt_count > 0:
            print(
                f"✓ generated code classified as exempt-clone ({exempt_count} findings)"
            )
            checks_passed += 1
        else:
            print("✗ no exempt-clone findings in generated code output")
    else:
        # A missing output file is a FAILED precondition, not a pass — a
        # selftest that exits green without verifying anything is exactly the
        # gate-that-cannot-see failure shape.
        print("✗ sample output not found — run the analysis before selftesting")

    # 3. dead-code findings grep-verified (sample 3): the named symbol must
    # appear in the named file's CONTENT (existence alone proves nothing).
    dead_code_verified = 0
    for output_file in output_dir.glob("*.json"):
        if output_file.name == "rollup.json":
            continue
        try:
            result = json.loads(output_file.read_text())
        except json.JSONDecodeError:
            continue
        for f in result.get("findings", []):
            if f.get("category") == "dead-code" and dead_code_verified < 3:
                file_path = Path(f.get("file", ""))
                symbol = (f.get("title", "") or "").split()[:1]
                if file_path.is_file() and symbol:
                    content = file_path.read_text(errors="replace")
                    if symbol[0] in content:
                        dead_code_verified += 1
    if dead_code_verified >= 3:
        print(f"✓ {dead_code_verified} dead-code findings grep-verified")
        checks_passed += 1
    else:
        print(f"✗ only {dead_code_verified}/3 dead-code findings content-verified")

    print(f"Selftest: {checks_passed}/{checks_total} checks passed")
    return checks_passed == checks_total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dossiers-dir", default="report/llm_module_analysis/dossiers")
    parser.add_argument("--output-dir", default="report/llm_module_analysis")
    parser.add_argument("--state-file", default="report/llm_module_analysis/state.json")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--victor-bin", default="victor")
    parser.add_argument("--profile", default="homelab-zai-coding")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--repo-root", default=None, help="Repo root for victor tool context"
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        ok = selftest(Path(args.dossiers_dir), Path(args.output_dir))
        sys.exit(0 if ok else 1)

    repo_root = Path(args.repo_root) if args.repo_root else None
    asyncio.run(
        run_analysis(
            Path(args.dossiers_dir),
            Path(args.output_dir),
            Path(args.state_file),
            concurrency=args.concurrency,
            victor_bin=args.victor_bin,
            profile=args.profile,
            timeout=args.timeout,
            repo_root=repo_root,
        )
    )
