#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guard what the *shipped* release artifacts are compiled with.

Two defects this exists to catch, both found while cutting v0.4.0:

1. **Cloud backends absent from every artifact.** `aws`/`azure`/`gcp` are
   opt-in features. The release matrix and the Dockerfile built with none of
   them, so `store_for_url("s3://...")` in a released binary failed with
   `feature for AmazonS3 not enabled` -- while `docs/SUPPORTED_SURFACE.adoc`
   tiers S3/Azure/GCS as Beta, i.e. available. CI tested `cloud-full`; nothing
   shipped it.

2. **No job timeouts in either release workflow.** TD-CI-5 recorded a
   `cloud-full` build leg hanging 121 minutes. With no `timeout-minutes` a hung
   release job runs to GitHub's 6-hour default.

This is a text/YAML guard on purpose: it must hold without a toolchain, and it
is the artifact contract, not behaviour.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RELEASE_WORKFLOWS = (
    ".github/workflows/release.yml",
    ".github/workflows/prerelease-ci.yml",
)
REQUIRED_FEATURE = "cloud-full"
# Matches a matrix `features:` entry that is NOT inside a comment.
FEATURES_RE = re.compile(r'^(?P<indent>\s+)features:\s*"(?P<value>[^"]*)"\s*$')
JOB_RE = re.compile(r"^  (?P<job>[A-Za-z][A-Za-z0-9_-]*):\s*$")


def _fail(msgs: list[str], msg: str) -> None:
    msgs.append(msg)


def check_workflow_features(msgs: list[str]) -> None:
    for rel in RELEASE_WORKFLOWS:
        path = ROOT / rel
        if not path.exists():
            _fail(msgs, f"{rel}: missing (this guard names it explicitly)")
            continue
        found = 0
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = FEATURES_RE.match(line)
            if m is None:
                continue
            found += 1
            value = m.group("value")
            if REQUIRED_FEATURE not in value.split(","):
                _fail(
                    msgs,
                    f"{rel}:{n}: release target builds with features=\"{value}\" -- "
                    f"must include `{REQUIRED_FEATURE}`, or the shipped binary "
                    f"cannot open s3://, az:// or gs:// at all",
                )
        if found == 0:
            _fail(msgs, f"{rel}: no live matrix `features:` entry found")


def check_workflow_timeouts(msgs: list[str]) -> None:
    for rel in RELEASE_WORKFLOWS:
        path = ROOT / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        try:
            start = next(i for i, l in enumerate(lines) if l.rstrip() == "jobs:")
        except StopIteration:
            _fail(msgs, f"{rel}: no `jobs:` block")
            continue
        job, body = None, []
        # Sentinel flushes the LAST job. It must satisfy JOB_RE -- an earlier
        # `__end__` did not (JOB_RE requires a leading [A-Za-z]), so the final
        # job in each file was silently never checked.
        for line in lines[start + 1 :] + ["  zzEndSentinel:"]:
            m = JOB_RE.match(line)
            if m is None:
                body.append(line)
                continue
            if job is not None and not any(
                re.match(r"^    timeout-minutes:\s*\d+\s*$", b) for b in body
            ):
                _fail(
                    msgs,
                    f"{rel}: job `{job}` has no `timeout-minutes` -- a hung "
                    f"release job would run to GitHub's 6h default (TD-CI-5)",
                )
            job, body = m.group("job"), []


def check_dockerfile(msgs: list[str]) -> None:
    path = ROOT / "Dockerfile"
    if not path.exists():
        _fail(msgs, "Dockerfile: missing")
        return
    builds = [
        (n, l)
        for n, l in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"^\s*RUN\s+cargo build\b", l)
    ]
    if not builds:
        _fail(msgs, "Dockerfile: no `RUN cargo build` line found")
        return
    for n, line in builds:
        if f"--features {REQUIRED_FEATURE}" not in line and REQUIRED_FEATURE not in line:
            _fail(
                msgs,
                f"Dockerfile:{n}: image build omits `--features {REQUIRED_FEATURE}` -- "
                f"the container is how operators consume cloud object storage",
            )


def main() -> int:
    msgs: list[str] = []
    check_workflow_features(msgs)
    check_workflow_timeouts(msgs)
    check_dockerfile(msgs)
    if msgs:
        print("release-build-features: FAILED")
        for m in msgs:
            print(f"  - {m}")
        return 1
    print(
        "release-build-features: OK "
        f"(both release workflows build with `{REQUIRED_FEATURE}`, "
        "every job bounded, image build matches)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
