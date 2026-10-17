#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guard the release-evidence contract: `.github/workflows/release-smoke.yml`.

Every release must evidence (A) it was built from a QA-proven tree and (B) every
published artifact installs and works. Before release-smoke.yml existed,
`verify-release` only asked crates.io and npm whether they LIST the version, and
v0.4.1 shipped with defects no check could see (the npm ESM entry points at a
file the package does not contain; the advertised Homebrew formula does not
exist in the tap; the qa->main PR was merged over a red `CI Success`).

This guard pins the contract so it cannot quietly erode:

  exists        release-smoke.yml is present.
  coverage      every channel release.yml / release-python.yml / publish-image.yml
                publishes to has a smoke job that installs from THAT channel:
                PyPI SDK, PyPI embedded, crates.io, npm, Docker Hub, GitHub
                Release binary, Homebrew. A publish step the guard cannot map to
                a channel fails too -- a new registry must arrive with its smoke.
  fail-hard     no `|| true`, `|| :` or `continue-on-error` in the workflow or
                in scripts/release_smoke/*.sh: a smoke that cannot fail is not
                evidence.
  provenance    a job runs scripts/release_smoke/provenance.py, which reuses
                scripts/await_prerelease_ci.sh rather than a copy of its query.
  aggregate     the `summary` job needs EVERY other job, runs `if: always()`,
                and runs scripts/release_smoke/summarize.py.
  hooked        release.yml calls release-smoke.yml as a job (`uses:`), so a red
                smoke makes the release run red; release-smoke.yml keeps its
                `workflow_call` and `workflow_dispatch` triggers.

A text guard on purpose (no PyYAML): it must run without a toolchain, and
mirrors scripts/check_release_build_features.py.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE = ".github/workflows/release-smoke.yml"
SMOKE_SCRIPTS = "scripts/release_smoke"
PUBLISHERS = (
    ".github/workflows/release.yml",
    ".github/workflows/release-python.yml",
    ".github/workflows/publish-image.yml",
)
CALLER = ".github/workflows/release.yml"
PROVENANCE = "scripts/release_smoke/provenance.py"
SUMMARIZE = "scripts/release_smoke/summarize.py"
AWAIT_PRERELEASE = "scripts/await_prerelease_ci.sh"

# channel -> (publish markers in the publishing workflows, markers that MUST all
# appear in one job of release-smoke.yml: the install from that channel and the
# exercise of what it installed).
CHANNELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "pypi-sdk": (
        ("https://pypi.org/project/proximadb/",),
        ("pypi.org/pypi/", "pip install", '"proximadb==${VERSION}"', "pypi_sdk_smoke.py"),
    ),
    "pypi-embedded": (
        ("https://pypi.org/project/proximadb-embedded/",),
        ("pypi.org/pypi/", "pip install", '"proximadb_embedded==${VERSION}"', "embedded_smoke.py"),
    ),
    "crates.io": (
        ("cargo publish",),
        ("crates.io/api/v1/crates/proximadb-sdk/", "render_manifest.sh", "cargo run"),
    ),
    "npm": (
        ("npm publish",),
        ("registry.npmjs.org/", "npm install", "node npm_smoke.cjs", "node npm_esm_smoke.mjs"),
    ),
    "docker-hub": (
        ("imagetools create", "push-by-digest=true"),
        ("hub.docker.com/v2/repositories/", "start_image_server.sh", "org.opencontainers.image.revision", "mvp_smoke.py"),
    ),
    "github-release-binary": (
        ("softprops/action-gh-release",),
        ("gh release download", "sha256sum -c", "SHA256SUMS.txt", "--version", "mvp_smoke.py"),
    ),
    "github-release-packages": (
        ("softprops/action-gh-release",),
        ("sha256sum -c SHA256SUMS.txt", "package_smoke.sh deb", "package_smoke.sh rpm", "rockylinux:8",
         "pypi.org/pypi/proximadb_embedded/"),
    ),
    "github-release-windows": (
        ("softprops/action-gh-release",),
        ("proximadb-server.exe", "Start-Process msiexec.exe", "Get-FileHash"),
    ),
    "homebrew": (
        ("scripts/release_homebrew_formula.py", "< Formula"),
        ("homebrew_check.py",),
    ),
}
# Anything that looks like a publish. A JOB containing one must also contain a
# channel's publish marker (the PyPI jobs are told apart by their environment
# url, since both use the same action); otherwise it is an unsmoked channel.
PUBLISH_LIKE = re.compile(
    r"cargo publish|npm publish|pypi-publish|twine upload|action-gh-release|gh release (?:create|upload)"
    r"|imagetools create|docker push|push:\s*true|push=true|< Formula|release_homebrew_formula\.py"
    r"|helm push|oras push"
)
# TestPyPI is advisory (continue-on-error in release-python.yml) and is not a
# channel users install from.
ADVISORY = "repository-url: https://test.pypi.org/legacy/"
# Ways a failing command is turned into a passing step. In the WORKFLOW a bare
# `exit 0` is also banned (nothing there needs it, and `; exit 0` after the
# aggregation step would whitewash the verdict); the helper scripts exit 0 on
# success by design, so for them only the `|| exit 0` form is banned.
SOFT_FAIL = re.compile(
    r"\|\|\s*(?:true|:|exit\s+0|return\s+0|echo\b)(?:\s|;|$|\))"
    r"|continue-on-error|\bset\s+\+[a-z]*e"
)
WORKFLOW_SOFT_FAIL = re.compile(SOFT_FAIL.pattern + r"|\bexit\s+0\b")
# The only `if:` conditions a smoke job or step may carry. Anything else
# (`if: false`, an event filter, `failure()` on a smoke step) can skip the
# smoke while the run stays green.
JOB_IF_ALLOWED = {
    "resolve": "github.event_name != 'release' || !github.event.release.prerelease",
    "summary": "always() && needs.resolve.result != 'skipped'",
}
# release.yml's caller job: pinned exactly.
CALLER_IF = (
    "always() && needs.create-release.result == 'success' && "
    "needs.prepare.outputs.dry_run != 'true' && needs.prepare.outputs.prerelease != 'true'"
)
PIP_OPTS = {"--no-cache-dir", "--upgrade", "-U", "--pre"}
NPM_OPTS = {"--no-audit", "--no-fund", "--prefer-online"}
PIP_SPEC = re.compile(r'^"?(?:pip|[A-Za-z0-9_.-]+==\$\{VERSION\})"?$')
NPM_SPEC = re.compile(r'^"?\$\{PKG\}@\$\{VERSION\}"?$')
JOB_RE = re.compile(r"^  (?P<job>[_A-Za-z][A-Za-z0-9_-]*):\s*(?:#.*)?$")


def _live(line: str) -> bool:
    return not line.lstrip().startswith("#")


def _jobs(lines: list[str]) -> dict[str, list[str]]:
    try:
        start = next(i for i, l in enumerate(lines) if l.rstrip() == "jobs:")
    except StopIteration:
        return {}
    jobs: dict[str, list[str]] = {}
    job = None
    for line in lines[start + 1 :]:
        m = JOB_RE.match(line)
        if m is not None:
            job = m.group("job")
            jobs[job] = []
        elif job is not None and _live(line):
            jobs[job].append(line)
    return jobs


def _needs(body: list[str]) -> set[str]:
    for i, line in enumerate(body):
        m = re.match(r"^    needs:\s*(?P<rest>.*)$", line)
        if not m:
            continue
        rest = m.group("rest").split("#", 1)[0].strip()
        if rest.startswith("["):
            return {x.strip().strip("\"'") for x in rest.strip("[]").split(",") if x.strip()}
        if rest:
            return {rest.strip("\"'")}
        out = set()
        for nxt in body[i + 1 :]:
            if re.match(r"^      - ", nxt):
                out.add(nxt.strip()[2:].strip().strip("\"'"))
            else:
                break
        return out
    return set()


def _strip_expr(cond: str) -> str:
    cond = cond.split(" #", 1)[0].strip()
    m = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", cond)
    return m.group(1) if m else cond


def _step_blocks(body: list[str]) -> list[list[str]]:
    steps: list[list[str]] = []
    for line in body:
        if re.match(r"^      - ", line):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    return steps


def _logical_lines(body: list[str]) -> list[str]:
    """Shell lines with backslash continuations joined."""
    out: list[str] = []
    buf = ""
    for line in body:
        s = line.strip()
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        out.append(buf + s)
        buf = ""
    if buf:
        out.append(buf)
    return out


def check(msgs: list[str]) -> None:
    smoke_path = ROOT / SMOKE
    if not smoke_path.exists():
        msgs.append(f"{SMOKE}: missing -- releases have no artifact-level evidence")
        return
    lines = smoke_path.read_text(encoding="utf-8").splitlines()
    live = [l for l in lines if _live(l)]
    jobs = _jobs(lines)

    # ---- fail-hard
    for n, line in enumerate(lines, 1):
        if _live(line) and WORKFLOW_SOFT_FAIL.search(line):
            msgs.append(f"{SMOKE}:{n}: `{line.strip()}` -- a smoke step must fail hard (no `|| true`, `|| :`, `continue-on-error`)")
    scripts_dir = ROOT / SMOKE_SCRIPTS
    for script in sorted(scripts_dir.glob("*.sh")) if scripts_dir.is_dir() else []:
        for n, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
            if _live(line) and SOFT_FAIL.search(line):
                rel = script.relative_to(ROOT).as_posix()
                msgs.append(f"{rel}:{n}: `{line.strip()}` -- smoke helpers must fail hard")

    # ---- no conditional skipping
    for job, body in jobs.items():
        for b in body:
            m = re.match(r"^    if:\s*(?P<cond>.*?)\s*$", b)
            if m and JOB_IF_ALLOWED.get(job) != _strip_expr(m.group("cond")):
                msgs.append(f"{SMOKE}: job `{job}` has `if: {m.group('cond')}` -- only {JOB_IF_ALLOWED} are allowed")
        if job in JOB_IF_ALLOWED and not any(re.match(r"^    if:", b) for b in body):
            msgs.append(f"{SMOKE}: job `{job}` lost its pinned `if:` ({JOB_IF_ALLOWED[job]})")
        for step in _step_blocks(body):
            cond = next((_strip_expr(m.group(1)) for s in step
                         if (m := re.match(r"^\s+(?:- )?if:\s*(.*?)\s*$", s))), None)
            if cond is None:
                continue
            name = next((s.split("name:", 1)[1].strip() for s in step if re.match(r"^\s+(?:- )?name:", s)), "")
            uploads = any("actions/upload-artifact@" in s for s in step)
            if cond == "always()" and uploads:
                continue
            if cond == "failure()" and name.endswith("(on failure)"):
                continue
            msgs.append(
                f"{SMOKE}: job `{job}` step `{name or step[0].strip()}` has `if: {cond}` -- steps may only be "
                "`if: always()` on an upload-artifact step or `if: failure()` on a '(on failure)' log step"
            )

    # ---- installs come from the registry, never the checkout
    for job, body in jobs.items():
        for cmd in _logical_lines(body):
            for tool, opts, spec in (("pip", PIP_OPTS, PIP_SPEC), ("npm", NPM_OPTS, NPM_SPEC)):
                m = re.search(rf"\b{tool} install\b(?P<args>.*)$", cmd)
                if not m:
                    continue
                args = m.group("args").split()
                i = 0
                while i < len(args):
                    a = args[i]
                    if tool == "npm" and a == "--prefix":
                        i += 2
                        continue
                    if a in opts or (tool == "pip" and a.startswith("--only-binary=")):
                        i += 1
                        continue
                    if not spec.match(a):
                        msgs.append(
                            f"{SMOKE}: job `{job}` `{tool} install` argument `{a}` is not a registry spec "
                            f"pinned to ${{VERSION}} (local paths, -e, git/file URLs are not allowed)"
                        )
                    i += 1

    # ---- coverage: every published channel has a smoke job
    published: set[str] = set()
    for rel in PUBLISHERS:
        path = ROOT / rel
        if not path.exists():
            msgs.append(f"{rel}: missing (this guard reads it to learn what is published)")
            continue
        for job, body in _jobs(path.read_text(encoding="utf-8").splitlines()).items():
            publish_lines = [b.strip() for b in body if PUBLISH_LIKE.search(b)]
            if not publish_lines or any(b.strip() == ADVISORY for b in body):
                continue
            hits = [c for c, (pub, _) in CHANNELS.items() if any(p in b for p in pub for b in body)]
            published.update(hits)
            if not hits:
                msgs.append(
                    f"{rel}: job `{job}` publishes (`{publish_lines[0]}`) to a channel {SMOKE} does not "
                    f"smoke -- add a smoke job and a CHANNELS entry in scripts/check_release_smoke.py"
                )
    for channel, (_, markers) in CHANNELS.items():
        if channel not in published:
            msgs.append(f"{channel}: no publish step found in {', '.join(PUBLISHERS)} -- update CHANNELS if it moved")
        covering = [j for j, body in jobs.items() if all(any(m in b for b in body) for m in markers)]
        if not covering:
            msgs.append(
                f"{SMOKE}: no job smokes `{channel}` -- one job must contain all of {list(markers)}"
            )

    # ---- provenance
    if not any(PROVENANCE in b for body in jobs.values() for b in body):
        msgs.append(f"{SMOKE}: no job runs {PROVENANCE} -- nothing proves the release is QA-based")
    prov = ROOT / PROVENANCE
    if not prov.exists():
        msgs.append(f"{PROVENANCE}: missing")
    elif "await_prerelease_ci.sh" not in prov.read_text(encoding="utf-8"):
        msgs.append(f"{PROVENANCE}: must reuse {AWAIT_PRERELEASE} for the Pre-Release CI verdict, not a copy of its query")

    # ---- aggregate
    summary = jobs.get("summary")
    if summary is None:
        msgs.append(f"{SMOKE}: no `summary` job -- there is no single evidence table")
    else:
        missing = set(jobs) - {"summary"} - _needs(summary)
        if missing:
            msgs.append(f"{SMOKE}: `summary` does not need {sorted(missing)} -- their failures would not fail the evidence")
        if not any(re.match(r"^    if:.*\balways\(\)", b) for b in summary):
            msgs.append(f"{SMOKE}: `summary` must run `if: always()` so a failed job still yields its evidence table")
        if not any(SUMMARIZE in b for b in summary):
            msgs.append(f"{SMOKE}: `summary` does not run {SUMMARIZE}")

    # ---- hooked
    for trig in ("workflow_call:", "workflow_dispatch:"):
        if not any(l.strip() == trig for l in live):
            msgs.append(f"{SMOKE}: lost its `{trig}` trigger")
    caller = ROOT / CALLER
    if caller.exists():
        caller_jobs = _jobs(caller.read_text(encoding="utf-8").splitlines())
        callers = [
            (j, body) for j, body in caller_jobs.items()
            if any(re.match(r"^    uses:\s*\./\.github/workflows/release-smoke\.yml\s*$", b) for b in body)
        ]
        if not callers:
            msgs.append(f"{CALLER}: no job `uses: ./{SMOKE}` -- a red smoke would leave the release green")
        for j, body in callers:
            conds = [_strip_expr(m.group(1)) for b in body if (m := re.match(r"^    if:\s*(.*?)\s*$", b))]
            if conds != [CALLER_IF]:
                msgs.append(f"{CALLER}: job `{j}` must carry exactly `if: {CALLER_IF}` (found {conds})")


def main() -> int:
    msgs: list[str] = []
    check(msgs)
    if msgs:
        print("release-smoke: FAILED")
        for m in msgs:
            print(f"  - {m}")
        return 1
    print(
        f"release-smoke: OK ({len(CHANNELS)} published channels each smoked from their registry, "
        "provenance proven, no soft-fail steps, summary aggregates every job, hooked into release.yml)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
