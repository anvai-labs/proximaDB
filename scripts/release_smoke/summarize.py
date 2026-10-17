#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate release-smoke.yml's per-job evidence into ONE document + table.

Inputs: the workflow's `needs` context as JSON (job -> {result, outputs}) and a
directory of per-job evidence files (`*.json`, one or more per job, uploaded as
`evidence-*` artifacts). Output: `release-evidence-vX.Y.Z.json` and a Markdown
table appended to --summary. Exit 1 unless EVERY job's result is `success` --
a skipped or cancelled job is missing evidence, not a pass.

Usage: summarize.py --version X.Y.Z --needs-json '<json>' --evidence-dir DIR
                    --out release-evidence-vX.Y.Z.json [--summary FILE]
                    [--run-url URL] [--sha SHA]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any

# Job id -> what it proves. The summary table is ordered by this.
ROWS = (
    ("provenance", "(A) tag is a qa -> main merge, QA-tested tree, Pre-Release CI + CI Success green"),
    ("docker", "(B) Docker Hub image: pull, provenance label, health, MVP smoke"),
    ("binary", "(B) GitHub Release linux x86_64 binary: sha256, --version, MVP smoke"),
    ("python", "(B) PyPI proximadb + proximadb_embedded: install, version, SDK + embedded roundtrip"),
    ("npm", "(B) npm @anvailabs/proximadb-client: install, version, CJS + ESM against server"),
    ("rust", "(B) crates.io proximadb-sdk: build from registry, version, calls against server"),
    ("homebrew", "(B) Homebrew tap formula: version, url + sha256 match the release"),
)


def load_evidence(directory: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    """Group evidence files by job: a file `<job>[-...].json` belongs to <job>."""
    out: dict[str, list[dict[str, Any]]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.rglob("*.json")):
        job = path.stem.split("-", 1)[0]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            doc = {"unreadable": str(exc)}
        out.setdefault(job, []).append({"file": path.name, "evidence": doc})
    return out


def build(version: str, needs: dict[str, Any], evidence: dict[str, list[dict[str, Any]]],
          run_url: str | None, sha: str | None) -> dict[str, Any]:
    jobs = []
    for job, proves in ROWS:
        result = (needs.get(job) or {}).get("result", "absent")
        jobs.append({"job": job, "proves": proves, "result": result, "evidence": evidence.get(job, [])})
    # A job in `needs` that this table does not know is still counted.
    for job, info in needs.items():
        if job not in {j for j, _ in ROWS}:
            jobs.append({"job": job, "proves": "", "result": (info or {}).get("result", "absent"),
                         "evidence": evidence.get(job, [])})
    return {
        "schema_version": 1,
        "kind": "release-evidence",
        "version": version,
        "tag": f"v{version}",
        "workflow_run": run_url,
        "workflow_sha": sha,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "passed": all(j["result"] == "success" for j in jobs),
        "jobs": jobs,
    }


def render(doc: dict[str, Any]) -> str:
    verdict = "PASSED" if doc["passed"] else "FAILED"
    lines = [
        f"## Release evidence: {doc['tag']} -- {verdict}",
        "",
        "| Artifact / claim | Result | Evidence files |",
        "|---|---|---|",
    ]
    for j in doc["jobs"]:
        mark = "PASS" if j["result"] == "success" else f"**{j['result'].upper()}**"
        files = ", ".join(f"`{e['file']}`" for e in j["evidence"]) or "--"
        lines.append(f"| `{j['job']}`: {j['proves']} | {mark} | {files} |")
    lines += ["", f"Evidence document: `release-evidence-{doc['tag']}.json`", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--needs-json", required=True)
    ap.add_argument("--evidence-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--summary", type=pathlib.Path)
    ap.add_argument("--run-url")
    ap.add_argument("--sha")
    args = ap.parse_args()

    needs = json.loads(args.needs_json)
    doc = build(args.version, needs, load_evidence(args.evidence_dir), args.run_url, args.sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = render(doc)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(table + "\n")
    print(table)
    if not doc["passed"]:
        bad = ", ".join(f"{j['job']}={j['result']}" for j in doc["jobs"] if j["result"] != "success")
        print(f"::error::release evidence for v{args.version} is NOT complete: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
