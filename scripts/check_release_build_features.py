#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guard what the *shipped* release artifacts are compiled with.

Two defects this exists to catch, both found while cutting v0.4.0:

1. **Cloud backends absent from the released binaries.** `aws`/`azure`/`gcp`
   are opt-in features and the release matrix built with none of them, so
   `store_for_url("s3://...")` in a released binary failed with
   `feature for AmazonS3 not enabled` -- while `docs/SUPPORTED_SURFACE.adoc`
   tiers S3/Azure/GCS as Beta, i.e. available. The published container image
   (`deploy/docker/Dockerfile`) always had them; the standalone binaries did
   not, which is the asymmetry this guard now pins on both sides.

2. **No job timeouts.** TD-CI-5 recorded a
   `Feature Matrix (cloud-full+onnx)` leg hanging 121 minutes -- that TD judged
   it a transient runner problem, and `cloud-full` alone builds clean, so the
   lesson taken here is only the missing bound: with no `timeout-minutes` a hung
   job runs to GitHub's 6-hour default. The converse is just as real and was
   measured on the v0.4.0 promotion: a bound set too tight silently disarms the
   step behind it -- `Docker Build` at 45 min let a 43-min cold build cancel
   `Test Container Health`, the only check that proves the published image
   accepts connections. So the scope is every job in the release AND pipeline
   workflows, and the caps there are set from measured durations.

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
# The timeout bound is not a release-only concern: `Docker Build` in ci.yml was
# capped at 45 min, a cold image build took 43, and `Test Container Health` --
# the only check that starts the published image and connects to its published
# port -- was cancelled 1 minute in. An unbounded job is the same defect with
# GitHub's 6h default instead of a cap, so every job in the pipeline workflows
# carries a bound too.
TIMED_WORKFLOWS = RELEASE_WORKFLOWS + (
    ".github/workflows/ci.yml",
    ".github/workflows/qa-gate.yml",
)
REQUIRED_FEATURE = "cloud-full"
# The image the publish workflow actually ships (publish-image.yml `file:`).
# The repo-root Dockerfile is the demo-compose image and is NOT checked here.
PUBLISHED_DOCKERFILE = "deploy/docker/Dockerfile"
# Deliberately exempt: a documented ultra-minimal (~80MB) variant that no
# workflow publishes. Cloud backends would defeat its stated purpose. Listed so
# the omission reads as a decision rather than an oversight.
EXEMPT_DOCKERFILES = {"deploy/docker/Dockerfile.alpine"}
# `--features X` or `--features=X` on a shell command line.
CARGO_FEATURES_RE = re.compile(r"--features[= ]+(?P<value>[^\s]+)")
# A matrix `features:` entry, in any spelling YAML allows. Deliberately NOT
# anchored on a double-quoted value: `features: \'\'`, bare `features: x`, a
# bare `features:` (null) and a trailing `# comment` are all legal YAML and an
# earlier revision of this guard passed every one of them while the target
# shipped with no features at all.
FEATURES_RE = re.compile(r"^\s+features:(?P<rest>.*)$")
# A job-level `timeout-minutes`. `${{ ... }}` is accepted because GitHub allows
# an expression there (github/needs/vars/inputs/matrix contexts), and a bare
# `\d+` match rejected it as missing -- a false positive on legal YAML. A float
# is likewise legal per the schema.
TIMEOUT_VALUE_RE = re.compile(
    r"^    timeout-minutes:\s*(?:\d+(?:\.\d+)?|\$\{\{.*\}\})\s*(?:#.*)?$"
)
# Three spellings, all legal, each of which previously let a job with NO cap
# through with exit 0, because an unmatched key folds the job's body into the
# PREVIOUS job -- whose cap then satisfies the check for both, and the unmatched
# job is never even named:
#   * `rust-clippy:  # the lint gate`  -- a trailing comment on the key
#   * `"rust-clippy":`                 -- a quoted key (GitHub job id rust-clippy)
#   * `_rustclippy:`                   -- GitHub job ids may start with `_`
# The quoted form also compounds with the `uses:` exemption below: folded into a
# reusable job, the merged body inherits that exemption too.
#
# `(#.*)?` is load-bearing, not tidiness: without it a job key written
# `rust-clippy:  # the lint gate` does not match, so the whole job body is
# absorbed into the PREVIOUS job's body. That job's own `timeout-minutes` then
# satisfies the check for both and the commented job is never even named --
# verified: the guard exited 0 with `rust-clippy` carrying no cap at all.
JOB_RE = re.compile(
    r"^  (?P<job>\"[^\"]+\"|'[^']+'|[_A-Za-z][A-Za-z0-9_-]*):\s*(#.*)?$"
)


def _scalar(rest: str) -> str:
    """The effective string value of a YAML scalar after `features:`.

    Strips a trailing comment, then one layer of matching quotes. A bare
    `features:` (null) and `features: ""` both yield "" -- which is exactly the
    state that ships a feature-less binary, so both must fail.
    """
    text = rest.strip()
    if text.startswith(("\"", "\'")):
        quote = text[0]
        end = text.find(quote, 1)
        if end != -1:
            return text[1:end]
        return text[1:]
    # Unquoted: a `#` begins a comment only when preceded by whitespace.
    text = re.split(r"(?:^|\s)#", text, maxsplit=1)[0]
    return text.strip()


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
            value = _scalar(m.group("rest"))
            if REQUIRED_FEATURE not in [v.strip() for v in value.split(",")]:
                _fail(
                    msgs,
                    f"{rel}:{n}: release target builds with features=\"{value}\" -- "
                    f"must include `{REQUIRED_FEATURE}`, or the shipped binary "
                    f"cannot open s3://, az:// or gs:// at all",
                )
        if found == 0:
            _fail(msgs, f"{rel}: no live matrix `features:` entry found")


def check_workflow_timeouts(msgs: list[str]) -> None:
    for rel in TIMED_WORKFLOWS:
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
            reusable = any(re.match(r"^    uses:", b) for b in body)
            bounded = any(TIMEOUT_VALUE_RE.match(b) for b in body)
            if job is not None and not bounded and not reusable:
                _fail(
                    msgs,
                    f"{rel}: job `{job}` has no `timeout-minutes` -- a hung "
                    f"job would run to GitHub's 6h default (TD-CI-5)",
                )
            job, body = m.group("job"), []


def check_runtime_config_ownership(msgs: list[str]) -> None:
    """Every `COPY ... config.toml /config/config.toml` must carry --chown.

    Not style. In each runtime stage the Dockerfile does
    `chown -R proximadb:proximadb ... /config` and only AFTER that copies the
    config in, so without `--chown` the file lands root-owned 0644 while the
    server runs as uid 999. entrypoint.sh then cannot rewrite `bind_address`,
    the server binds 127.0.0.1, `-p` cannot reach it, and the container is
    unreachable on its published ports.

    That shipped, and cost a full promotion round: `Test Container Health`
    failed on a tree that already contained the loopback fix, and the warning
    explaining it was scrolled away by `docker logs | tail -100`.

    Guarded here because `Docker Build` only runs on the extensive tier
    (`main`-targeted PRs), so nothing else catches a regression until a release.
    """
    path = ROOT / PUBLISHED_DOCKERFILE
    if not path.exists():
        _fail(msgs, f"{PUBLISHED_DOCKERFILE}: missing")
        return
    found = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("COPY"):
            continue
        if not stripped.endswith("/config/config.toml"):
            continue
        found += 1
        if "--chown=" not in stripped:
            _fail(
                msgs,
                f"{PUBLISHED_DOCKERFILE}:{i}: `{stripped}` has no `--chown=` -- "
                "the `chown -R ... /config` above runs BEFORE this COPY, so the "
                "config lands root-owned and entrypoint.sh cannot rewrite "
                "bind_address; the image binds 127.0.0.1 and is unreachable on "
                "its published ports",
            )
    if found == 0:
        _fail(
            msgs,
            f"{PUBLISHED_DOCKERFILE}: no `COPY ... /config/config.toml` found -- "
            "this guard has lost its target; re-point it rather than deleting it",
        )


def check_dockerfiles(msgs: list[str]) -> None:
    """Every image that compiles the server must compile the cloud backends.

    Scans the whole image directory rather than one named file, so a new
    published variant is covered the day it is added. Variants that do not
    compile the server (`.prebuilt`, `.wheel`) have no build line and are
    therefore silently fine; `EXEMPT_DOCKERFILES` carries the ones that compile
    it deliberately without cloud.
    """
    published = ROOT / PUBLISHED_DOCKERFILE
    if not published.exists():
        _fail(
            msgs,
            f"{PUBLISHED_DOCKERFILE}: missing (the publish workflow names this file)",
        )
        return

    checked_any = False
    for path in sorted((ROOT / "deploy/docker").glob("Dockerfile*")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT_DOCKERFILES:
            continue
        builds = [
            (n, l)
            for n, l in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"^\s*RUN\b.*\bcargo build\b", l)
        ]
        if not builds:
            continue  # does not compile the server at all
        checked_any = True
        for n, line in builds:
            # Strip a trailing comment so a mention of the feature in prose
            # cannot satisfy the check -- the flag must be on the command.
            command = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]
            # Parse the list rather than substring-matching it, symmetrically
            # with check_workflow_features. A raw `"--features cloud-full" in`
            # test accepted `cloud-fullish` AND rejected the correct
            # `--features onnx,cloud-full` purely on ordering -- and line 79
            # here is `cloud-full,onnx`, one reorder away from a false failure.
            listed: list[str] = []
            for fm in CARGO_FEATURES_RE.finditer(command):
                listed.extend(v.strip() for v in fm.group("value").split(","))
            if REQUIRED_FEATURE not in listed:
                _fail(
                    msgs,
                    f"{rel}:{n}: image build omits `--features {REQUIRED_FEATURE}` "
                    f"-- the container is how operators consume cloud object "
                    f"storage",
                )
    if not checked_any:
        _fail(
            msgs,
            "deploy/docker: no non-exempt Dockerfile compiles the server -- "
            "this guard would pass vacuously",
        )


def main() -> int:
    msgs: list[str] = []
    check_workflow_features(msgs)
    check_workflow_timeouts(msgs)
    check_dockerfiles(msgs)
    check_runtime_config_ownership(msgs)
    if msgs:
        print("release-build-features: FAILED")
        for m in msgs:
            print(f"  - {m}")
        return 1
    print(
        "release-build-features: OK "
        f"(both release workflows and {PUBLISHED_DOCKERFILE} build with "
        f"`{REQUIRED_FEATURE}`, every job in {len(TIMED_WORKFLOWS)} workflows "
        "bounded, runtime config copied --chown)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
