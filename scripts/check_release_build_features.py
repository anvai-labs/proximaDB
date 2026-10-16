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
import posixpath
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
    # Builds AND publishes the shipped image, so an unbounded job here is on the
    # release path.
    ".github/workflows/publish-image.yml",
    # Everything else with a `jobs:` block. The first revision of this guard
    # covered only the release + pipeline workflows, which left 31 jobs across
    # 10 files on GitHub's 6h default -- including `release-python.yml`, whose
    # jobs hold the `pypi` deployment environment while they run, and the two
    # `workflow_call` libraries, whose jobs would inherit no bound from their
    # caller. The bound is a property of a job, not of a workflow's tier, so the
    # list is now every workflow in the directory and `check_every_workflow_is_timed`
    # below fails if a new file is added without being named here.
    ".github/workflows/_platform-packages.yml",
    ".github/workflows/_shared-build.yml",
    ".github/workflows/branch-route-guard.yml",
    ".github/workflows/capability-contract-tests.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/docs-site.yml",
    ".github/workflows/flake-rerun.yml",
    ".github/workflows/infrastructure-deploy.yml",
    ".github/workflows/layering-check.yml",
    ".github/workflows/mirror-ci-images.yml",
    ".github/workflows/objstore-nightly.yml",
    ".github/workflows/prerelease-ci-v0.2.1-macos.yml",
    ".github/workflows/public-overflow-runner-smoke.yml",
    ".github/workflows/release-python.yml",
    ".github/workflows/tdd.yml",
    ".github/workflows/tier-config-fingerprint.yml",
    ".github/workflows/windows-compile-ratchet.yml",
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
# A literal cap must be a real bound. `0` matched TIMEOUT_VALUE_RE and read as
# "bounded" while cancelling the job instantly.
TIMEOUT_ZERO_RE = re.compile(r"^    timeout-minutes:\s*0+(?:\.0+)?\s*(?:#.*)?$")
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
            if job is not None and any(TIMEOUT_ZERO_RE.match(b) for b in body):
                _fail(
                    msgs,
                    f"{rel}: job `{job}` has `timeout-minutes: 0`, which cancels "
                    f"the job instantly rather than bounding it",
                )
            if job is not None and not bounded and not reusable:
                _fail(
                    msgs,
                    f"{rel}: job `{job}` has no `timeout-minutes` -- a hung "
                    f"job would run to GitHub's 6h default (TD-CI-5)",
                )
            job, body = m.group("job"), []


def check_every_workflow_is_timed(msgs: list[str]) -> None:
    """`TIMED_WORKFLOWS` must name every workflow that has a `jobs:` block.

    Without this the guard is opt-in: a workflow added tomorrow is simply not
    in the tuple, so its jobs run to GitHub's 6h default and the guard still
    exits 0 -- which is exactly how 31 jobs across 10 files stayed unbounded
    after the first revision bounded the release and pipeline tiers.

    Scanned from the directory rather than trusted from the tuple, so the
    failure mode is a loud "add it" rather than silence.
    """
    wf_dir = ROOT / ".github/workflows"
    if not wf_dir.is_dir():
        _fail(msgs, ".github/workflows: missing (this guard has lost its target)")
        return
    listed = set(TIMED_WORKFLOWS)
    found = False
    for path in sorted(wf_dir.glob("*.y*ml")):
        rel = path.relative_to(ROOT).as_posix()
        if "\njobs:" not in path.read_text(encoding="utf-8"):
            continue  # no jobs to bound
        found = True
        if rel not in listed:
            _fail(
                msgs,
                f"{rel}: has a `jobs:` block but is not in TIMED_WORKFLOWS, so "
                f"its jobs are unchecked and a hung one runs to GitHub's 6h "
                f"default -- add it to the tuple (and to the COPY list in "
                f"tests/scripts/test_release_build_features.py, or the new "
                f"assertions are vacuous)",
            )
    if not found:
        _fail(
            msgs,
            ".github/workflows: no workflow with a `jobs:` block -- this guard "
            "would pass vacuously",
        )


# ---- TD-RELTAG-1: one tag `vX.Y.Z` releases every registry at one version ----------
# The two PUBLISHING workflows. They stay separate files because npm's trusted
# publisher is bound to `release.yml` and PyPI's to `release-python.yml` -- but
# they must fire on the SAME tags and both gate tag == manifests before building.
# Before this, release-python.yml fired on a separate `python-v*` family and PyPI
# drifted to 0.3.0 / 0.2.2 while crates.io shipped 0.4.0.
PUBLISH_WORKFLOWS = (
    ".github/workflows/release.yml",
    ".github/workflows/release-python.yml",
)
VERSION_SYNC = "scripts/version-sync.sh"
VERSION_GATE = "scripts/version-sync.sh check-tag"
# Every manifest whose version a registry publishes. `version-sync.sh check`
# must cover each one, or the gate (tag == every checked manifest) has a hole
# a package could ship through at a different version.
PUBLISHED_MANIFESTS = (
    "Cargo.toml:cargo",  # GitHub Release binaries (proximadb-server etc.)
    "Cargo.toml:cargo_workspace",  # every workspace crate inheriting it
    "clients/rust/Cargo.toml:cargo",  # crates.io proximadb-sdk
    "clients/nodejs-embedded/package.json:package_json",  # npm
    "clients/python/pyproject.toml:pyproject",  # PyPI proximadb
    "pyproject.toml:pyproject",  # PyPI proximadb_embedded (root maturin)
)
LEGACY_TAG_RE = re.compile(r"python-v[0-9*]|tags/python-v")
NEEDS_INLINE_RE = re.compile(r"^    needs:\s*(?P<rest>.*)$")


def _live(line: str) -> bool:
    return not line.lstrip().startswith("#")


def _push_tags(lines: list[str]) -> list[str]:
    """The `on: push: tags:` patterns, read from the text (no PyYAML needed)."""
    tags: list[str] = []
    in_on = in_push = in_tags = False
    for line in lines:
        if not line.strip() or not _live(line):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_on = line.rstrip() in ("on:", '"on":', "'on':")
            in_push = in_tags = False
            continue
        if not in_on:
            continue
        if indent == 2:
            in_push = line.strip() == "push:"
            in_tags = False
        elif indent == 4 and in_push:
            in_tags = line.strip() == "tags:"
        elif in_tags and line.strip().startswith("- "):
            tags.append(_scalar(line.strip()[2:]))
    return tags


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
            job = m.group("job").strip("\"'")
            jobs[job] = []
        elif job is not None:
            jobs[job].append(line)
    return jobs


def _needs(body: list[str]) -> set[str]:
    out: set[str] = set()
    for i, line in enumerate(body):
        m = NEEDS_INLINE_RE.match(line)
        if m is None:
            continue
        # Strip a trailing comment; job ids cannot contain `#` or whitespace.
        rest = re.split(r"(?:^|\s)#", m.group("rest"), maxsplit=1)[0].strip()
        if rest.startswith("["):
            out.update(x.strip().strip("\"'") for x in rest.strip("[]").split(",") if x.strip())
        elif rest:
            out.add(rest.strip("\"'"))
        else:  # block list
            for nxt in body[i + 1 :]:
                if re.match(r"^      - ", nxt):
                    out.add(_scalar(nxt.strip()[2:]))
                elif nxt.strip() and _live(nxt):
                    break
    return out


def check_unified_release_tag(msgs: list[str]) -> None:
    patterns: dict[str, list[str]] = {}
    for rel in PUBLISH_WORKFLOWS:
        path = ROOT / rel
        if not path.exists():
            _fail(msgs, f"{rel}: missing (TD-RELTAG-1 names it explicitly; renaming it also breaks its trusted publisher)")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        tags = _push_tags(lines)
        patterns[rel] = tags
        if not tags:
            _fail(msgs, f"{rel}: no `on: push: tags:` patterns -- a release tag would not trigger it (TD-RELTAG-1)")
        for t in tags:
            if not t.startswith("v[0-9]"):
                _fail(msgs, f"{rel}: tag pattern `{t}` is not the unified `v<semver>` family (TD-RELTAG-1)")
        for n, line in enumerate(lines, 1):
            if _live(line) and LEGACY_TAG_RE.search(line):
                _fail(msgs, f"{rel}:{n}: live reference to the retired per-package `python-v` tag family (TD-RELTAG-1)")
        jobs = _jobs(lines)
        gates = [j for j, body in jobs.items() if any(_live(b) and VERSION_GATE in b for b in body)]
        if not gates:
            _fail(msgs, f"{rel}: no job runs `{VERSION_GATE}` -- a tag could publish packages at a different version (TD-RELTAG-1)")
            continue
        needs = {j: _needs(body) for j, body in jobs.items()}

        def reaches_gate(job: str, seen: frozenset = frozenset()) -> bool:
            if job in gates:
                return True
            return any(reaches_gate(d, seen | {job}) for d in needs.get(job, ()) if d not in seen)

        for job in jobs:
            if not reaches_gate(job):
                _fail(
                    msgs,
                    f"{rel}: job `{job}` does not depend (directly or transitively) on the "
                    f"version-gate job `{gates[0]}` -- it could build or publish before tag == "
                    f"manifests is proven (TD-RELTAG-1)",
                )
    if len(patterns) == len(PUBLISH_WORKFLOWS):
        a, b = (patterns[r] for r in PUBLISH_WORKFLOWS)
        if a != b:
            _fail(
                msgs,
                f"release tag patterns differ: {PUBLISH_WORKFLOWS[0]} {a} vs "
                f"{PUBLISH_WORKFLOWS[1]} {b} -- ONE tag must release every registry (TD-RELTAG-1)",
            )
    vs = ROOT / VERSION_SYNC
    if not vs.exists():
        _fail(msgs, f"{VERSION_SYNC}: missing -- the release version gate has no implementation")
        return
    vs_lines = vs.read_text(encoding="utf-8").splitlines()
    if not any(_live(l) and l.strip().startswith("check-tag)") for l in vs_lines):
        _fail(msgs, f"{VERSION_SYNC}: no `check-tag` command -- both release workflows call it (TD-RELTAG-1)")
    listed = {l.strip().strip('"') for l in vs_lines if _live(l) and l.strip().startswith('"')}
    for entry in PUBLISHED_MANIFESTS:
        if entry not in listed:
            _fail(
                msgs,
                f"{VERSION_SYNC}: `check` no longer covers published manifest `{entry}` -- "
                f"that package could ship at a version the tag gate never compared (TD-RELTAG-1)",
            )


def _dockerfile_instructions(text: str) -> list[tuple[int, str]]:
    """Logical instructions with line continuations joined and comments stripped.

    A guard that reads raw lines is defeated by `COPY --from=builder \\` plus a
    continuation, which is ordinary Dockerfile style.
    """
    out: list[tuple[int, str]] = []
    buf, first = "", 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if not buf:
            first = lineno
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        buf += stripped
        # A trailing comment on the instruction must not hide the target.
        out.append((first, re.sub(r"\s+#.*$", "", buf).strip()))
        buf = ""
    if buf:
        out.append((first, buf.strip()))
    return out


# The runtime user. `groupadd -r proximadb && useradd -r -g proximadb proximadb`
# on the shipped bases yields uid/gid 999, so the numeric spelling is the same
# owner and must not be a false positive. Matched by EQUALITY, not prefix: an
# earlier `startswith("proximadb")` also accepted `proximadbX`.
RUNTIME_OWNERS = frozenset(
    {
        "proximadb",
        "proximadb:proximadb",
        "999",
        "999:999",
        "proximadb:999",
        "999:proximadb",
    }
)


def _lands_runtime_config(instruction: str) -> bool:
    """Does this COPY put a file at /config/config.toml, however spelled?

    Covers the exec/JSON form, a quoted destination, and a directory copy --
    each of which defeated an `endswith("/config/config.toml")` check.
    """
    body = instruction[len("COPY") :]
    # The exec/JSON form `COPY ["src", "dst"]` must normalise to the same
    # argument list as the shell form; leaving the brackets attached made the
    # destination parse as a bare "]" and the whole instruction slip through.
    tokens = re.findall(r'"[^"]*"|\S+', body)
    args = [t.strip('"[], ') for t in tokens if not t.startswith("--")]
    args = [a for a in args if a]
    if not args:
        return False
    dest, sources = posixpath.normpath(args[-1]), args[:-1]
    if dest == "/config/config.toml":
        return True
    if dest.rstrip("/") == "/config" and sources:
        # Any copy into /config may land config.toml; the source need not be
        # named for it. `COPY /build/defaults/ /config/` slipped through a check
        # that required the source to look like a config.
        return True
    return False


def _run_only_fixes_perms(instruction: str) -> bool:
    """Is this RUN purely a legitimate ownership/mode fix on the config?

    An earlier version checked only the VERB, so `RUN chown root:root
    /config/config.toml` passed -- which reproduces the original defect exactly
    -- and `RUN chmod 000` passed too, which is worse: uid 999 then cannot even
    read the file, so neither the rewrite nor the server works.
    """
    body = instruction[len("RUN") :].strip()
    for part in re.split(r"&&|;", body):
        tokens = part.split()
        if not tokens:
            continue
        verb, rest = tokens[0], tokens[1:]
        if verb == "chown":
            owners = [t.strip('"') for t in rest if not t.startswith("-")]
            if not owners or owners[0] not in RUNTIME_OWNERS:
                return False
        elif verb == "chmod":
            modes = [t for t in rest if not t.startswith("-")]
            if not modes:
                return False
            mode = modes[0]
            # The owner must keep write: the entrypoint rewrites this file.
            if re.fullmatch(r"[0-7]?([0-7])[0-7][0-7]", mode):
                user_digit = int(mode[-3])
                if not user_digit & 0o2:
                    return False
            elif not re.search(r"u[+=][^,]*w|a[+=][^,]*w|\+w", mode):
                return False
        else:
            return False
    return True


def check_runtime_config_ownership(msgs: list[str]) -> None:
    """The runtime config must be owned by the user the server runs as.

    Not style. In each runtime stage the Dockerfile does
    `chown -R proximadb:proximadb ... /config` and only AFTER that copies the
    config in, so without `--chown` the file lands root-owned 0644 while the
    server runs as uid 999. entrypoint.sh then cannot rewrite `bind_address`,
    the server binds 127.0.0.1, `-p` cannot reach it, and the image is
    unreachable on its published ports.

    That shipped, and cost three promotion rounds: `Test Container Health`
    failed on a tree that already contained the loopback fix, and the warning
    explaining it was scrolled away by `docker logs | tail -100`.

    Guarded here because `Docker Build` only runs on the extensive tier
    (`main`-targeted PRs), so nothing else catches a regression until a release.
    Checked PER STAGE: an earlier revision counted matches across the whole
    file, so breaking one of the two shipped stages left the other satisfying
    the check.
    """
    path = ROOT / PUBLISHED_DOCKERFILE
    if not path.exists():
        _fail(msgs, f"{PUBLISHED_DOCKERFILE}: missing")
        return
    text = path.read_text(encoding="utf-8")
    # Keyed by a UNIQUE index per FROM, not by name: an un-named `FROM` used to
    # key on its BASE IMAGE, so two un-named `FROM debian:bookworm-slim` stages
    # collided and a config copied in a non-serving stage satisfied a serving
    # one. `parent` carries `FROM <stage> AS <child>` inheritance, because a
    # child really does inherit the parent's config and user -- treating that as
    # config-less was a false failure on the idiomatic way to add a variant.
    stage_key = -1
    names: dict[str, int] = {}
    parent: dict[int, int] = {}
    per_stage: dict[int, int] = {}
    labels: dict[int, str] = {}
    serving_stages: list[int] = []
    for lineno, instruction in _dockerfile_instructions(text):
        upper = instruction.upper()
        if upper.startswith("FROM"):
            stage_key += 1
            tokens = instruction.split()
            base = tokens[1] if len(tokens) > 1 else "<unknown>"
            match = re.search(r"\sAS\s+(\S+)", instruction, re.IGNORECASE)
            name = match.group(1) if match else f"{base}#{stage_key}"
            labels[stage_key] = name
            names[name] = stage_key
            if base in names:
                parent[stage_key] = names[base]
            continue
        # A config moved into place by a RUN defeats a COPY-only check.
        if upper.startswith("USER"):
            who = instruction.split()[1] if len(instruction.split()) > 1 else ""
            if who.strip('"') in RUNTIME_OWNERS:
                if stage_key not in serving_stages:
                    serving_stages.append(stage_key)
            continue
        if upper.startswith("RUN") and "/config/config.toml" in instruction:
            # Anything that WRITES it, not an enumerated verb list: `cat >`,
            # `tee`, `sed -i` and a bare redirect all defeated `mv|cp|install|ln`.
            # chown/chmod are the legitimate exceptions -- they FIX ownership.
            if not _run_only_fixes_perms(instruction):
                _fail(
                    msgs,
                    f"{PUBLISHED_DOCKERFILE}:{lineno}: stage `{labels.get(stage_key, stage_key)}` puts the "
                    "runtime config in place with a RUN rather than a COPY "
                    "--chown; ownership is then unguarded (see this guard's "
                    "docstring for why that makes the image unreachable)",
                )
            continue
        if not (upper.startswith("COPY") or upper.startswith("ADD")) or not _lands_runtime_config(
            instruction
        ):
            continue
        per_stage[stage_key] = per_stage.get(stage_key, 0) + 1
        owner = re.search(r'--chown=("[^"]*"|\S+)', instruction)
        if owner is None:
            _fail(
                msgs,
                f"{PUBLISHED_DOCKERFILE}:{lineno}: stage `{labels.get(stage_key, stage_key)}`: "
                f"`{instruction}` has no `--chown=` -- the `chown -R ... /config` "
                "above runs BEFORE this COPY, so the config lands root-owned and "
                "entrypoint.sh cannot rewrite bind_address; the image binds "
                "127.0.0.1 and is unreachable on its published ports",
            )
        elif owner.group(1).strip('"') not in RUNTIME_OWNERS:
            _fail(
                msgs,
                f"{PUBLISHED_DOCKERFILE}:{lineno}: stage `{labels.get(stage_key, stage_key)}`: "
                f"`--chown={owner.group(1).strip('\"')}` is not the user the server runs as "
                "(`proximadb`), so the config is still unwritable at runtime",
            )
    # Per SERVING stage, not "any stage named runtime*". Deleting the `runtime`
    # stage's COPY while `runtime-full` kept its own satisfied the old check and
    # shipped an image that binds 127.0.0.1 on every port: with no config at
    # /config/config.toml entrypoint.sh reports "no --config file found" and
    # leaves the binary's loopback default. Verified by building it.
    def has_config(key: int) -> bool:
        """A stage has a config if it copies one, or inherits from one that does."""
        seen: set[int] = set()
        while key is not None and key not in seen:
            if per_stage.get(key):
                return True
            seen.add(key)
            key = parent.get(key)  # type: ignore[assignment]
        return False

    serving = [labels.get(st, str(st)) for st in serving_stages if not has_config(st)]
    if serving:
        _fail(
            msgs,
            f"{PUBLISHED_DOCKERFILE}: stage(s) {', '.join(sorted(serving))} run the "
            "server (`USER proximadb`) but no instruction puts a config at "
            "/config/config.toml -- that image binds 127.0.0.1 on every port and "
            "is unreachable through `docker run -p`",
        )
    if not serving_stages:
        _fail(
            msgs,
            f"{PUBLISHED_DOCKERFILE}: found no stage with `USER proximadb` -- this "
            "guard has lost its target; re-point it rather than deleting it",
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


PRERELEASE_GATE = "scripts/await_prerelease_ci.sh"
# Only a building Pre-Release CI run may satisfy the gate: a push to qa also
# triggers prerelease-ci.yml but skips every build, concluding `success` in ~15 s.
PRERELEASE_RUN_FILTER = (
    'select((.event == "push" and .head_branch == "main") or .event == "workflow_dispatch")'
)
# A live line with one of these makes a job a registry publish.
PUBLISH_MARKERS = (
    "pypa/gh-action-pypi-publish",
    "cargo publish",
    "npm publish",
    "softprops/action-gh-release",
)
# TestPyPI is advisory (continue-on-error) and deliberately not gated.
ADVISORY_PUBLISH = "test.pypi.org"


def check_prerelease_gate(msgs: list[str]) -> None:
    """Every registry publish waits for Pre-Release CI on the tagged commit.

    v0.4.1 (2026-10-16): release-python.yml published PyPI with no Pre-Release
    CI gate at all, and release.yml's gate failed instead of waiting.
    """
    gate = ROOT / PRERELEASE_GATE
    if not gate.exists():
        _fail(msgs, f"{PRERELEASE_GATE}: missing -- both release workflows wait on it before publishing")
    elif PRERELEASE_RUN_FILTER not in gate.read_text(encoding="utf-8"):
        _fail(
            msgs,
            f"{PRERELEASE_GATE}: no longer restricts to building runs (push to main or dispatch) -- "
            f"a qa-push run skips every build yet concludes success",
        )
    for rel in PUBLISH_WORKFLOWS:
        path = ROOT / rel
        if not path.exists():
            continue  # check_unified_release_tag already reports it
        jobs = _jobs(path.read_text(encoding="utf-8").splitlines())
        gates = {j for j, body in jobs.items() if any(_live(b) and PRERELEASE_GATE in b for b in body)}
        if not gates:
            _fail(msgs, f"{rel}: no job runs `{PRERELEASE_GATE}` -- it can publish a commit whose Pre-Release CI failed")
            continue
        needs = {j: _needs(body) for j, body in jobs.items()}

        def reaches(job: str, seen: frozenset = frozenset()) -> bool:
            if job in gates:
                return True
            return any(reaches(d, seen | {job}) for d in needs.get(job, ()) if d not in seen)

        for job, body in jobs.items():
            live = [b for b in body if _live(b)]
            publishes = any(m in b for b in live for m in PUBLISH_MARKERS)
            advisory = any(ADVISORY_PUBLISH in b for b in live)
            if publishes and not advisory and not reaches(job):
                _fail(
                    msgs,
                    f"{rel}: job `{job}` publishes but does not wait (directly or transitively) "
                    f"for Pre-Release CI (`{PRERELEASE_GATE}`)",
                )


def main() -> int:
    msgs: list[str] = []
    check_workflow_features(msgs)
    check_workflow_timeouts(msgs)
    check_every_workflow_is_timed(msgs)
    check_dockerfiles(msgs)
    check_runtime_config_ownership(msgs)
    check_unified_release_tag(msgs)
    check_prerelease_gate(msgs)
    if msgs:
        print("release-build-features: FAILED")
        for m in msgs:
            print(f"  - {m}")
        return 1
    print(
        "release-build-features: OK "
        f"(both release workflows and {PUBLISHED_DOCKERFILE} build with "
        f"`{REQUIRED_FEATURE}`, every job in {len(TIMED_WORKFLOWS)} workflows "
        "bounded, runtime config copied --chown, one release tag gated "
        "against every published manifest, every publish behind Pre-Release CI)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
