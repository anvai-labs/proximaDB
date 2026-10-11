#!/usr/bin/env python3
"""Regression checks for immutable cloud-emulator container inputs.

These assert the PROPERTIES the harness must have, not the vendor it currently
uses. The S3 emulator has moved three times (Docker Hub MinIO -> Quay MinIO ->
LocalStack, TD-CI-6), and the previous version of this file pinned the vendor by
name — so a necessary migration failed the guard rather than being checked by it.
What must stay true regardless of vendor:

  * the image is pinned by DIGEST, so a CI run cannot silently change what it
    tests, and a published tag that starts requiring a licence cannot break the
    job by moving under us (which is exactly how the LocalStack 2026.09.1 attempt
    failed: it pulled, then exited 55 on "License activation failed");
  * every emulator binds to LOOPBACK only, because they all run with development
    credentials;
  * readiness is established by an operation the suite depends on, not merely by
    a port being open or a health URL answering;
  * bucket creation is idempotent.
"""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts/run_cloud_emulator_tests.sh"


OBJECT_STORE_SRC = REPO_ROOT / "crates/storage/proximadb-object-store/src/lib.rs"


def _tier_test_filters(source: str) -> list[str]:
    """The positional test-name filters the tier gate's payload runs.

    Extracted from the `cargo test -p proximadb-object-store … -- --ignored`
    invocation: everything after the `--` separator that is not a flag.
    """
    # Anchored at a line start, so a mention in a COMMENT (or a second
    # invocation added later) cannot silently redirect this at the wrong block.
    m = re.search(r"^cargo test -p proximadb-object-store\b", source, re.MULTILINE)
    if m is None:
        return []
    block: list[str] = []
    for line in source[m.start():].splitlines():
        block.append(line)
        if not line.rstrip().endswith("\\"):
            break
    joined = " ".join(ln.rstrip("\\") for ln in block)
    _, sep, after = joined.partition(" -- ")
    if not sep:
        return []
    # Skip the ARGUMENT of flags that take one. `--skip` is already used
    # elsewhere in this same script, so adding it to the tier gate would
    # otherwise be read as a fourth test name — a false failure on a
    # legitimate edit.
    takes_arg = {"--skip", "--test", "--features", "-p", "--exact-filter"}
    out: list[str] = []
    toks = after.split()
    k = 0
    while k < len(toks):
        tok = toks[k]
        if tok in takes_arg:
            k += 2
            continue
        if not tok.startswith("-"):
            out.append(tok)
        k += 1
    return out


# `docker run` flags that consume the following token. Needed to find the IMAGE
# argument positionally. If a flag outside this set ever takes an argument, the
# image detection shifts by one and the guard fails LOUDLY with the offending
# token rather than passing silently — extend this set when that happens.
_DOCKER_ARG_FLAGS = frozenset(
    {
        "-p", "--publish", "--name", "-e", "--env", "-v", "--volume",
        "--network", "--net", "--entrypoint", "-u", "--user", "-w",
        "--workdir", "--label", "-l", "--add-host", "--health-cmd",
        "--memory", "-m", "--cpus", "--restart", "--platform",
    }
)


def _takes_arg(tok: str) -> bool:
    return tok in _DOCKER_ARG_FLAGS


def _tier_gate_features(source: str) -> set[str]:
    """The `--features` the tier gate compiles with."""
    m = re.search(
        r"^cargo test -p proximadb-object-store\s+--features\s+(\S+)", source, re.MULTILINE
    )
    return set(m.group(1).split(",")) if m else set()


def _docker_run_port_bindings(source: str) -> tuple[list[str], int, list[str], list[str]]:
    """Port bindings, container count, and off-host escape hatches.

    Scoped to `docker run` blocks because `-p` is also cargo's package flag: a
    naive scan picks up `cargo test -p proximadb-server` and reports it as an
    unqualified port binding (it did). Comment lines are dropped, because this
    script's stated purpose includes recording the emulator migration history —
    a commented-out old `docker run` would otherwise fail the guard on dead prose.

    Returns the bindings and the number of `--name`d containers, so the count
    assertion can be derived rather than hard-coded.
    """
    bindings: list[str] = []
    publish_all: list[str] = []
    host_network: list[str] = []
    images: list[str | None] = []
    names = 0
    lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        if "docker run" in lines[i]:
            block = []
            while i < len(lines):
                block.append(lines[i])
                if not lines[i].rstrip().endswith("\\"):
                    break
                i += 1
            joined = "\n".join(block)
            image: str | None = None
            # Candidates start AFTER `docker run`, or the literal token `docker`
            # is read as the image (it was).
            # From the SAME token list the walk indexes (continuations joined),
            # not a differently-split copy — otherwise the index is off by the
            # number of `\` tokens.
            _walk_toks = joined.replace("\\\n", " ").split()
            run_at = next((n for n, t in enumerate(_walk_toks) if t == "run" and n > 0), 0)
            # A TOKEN WALK, not a regex. Every form docker accepts has to be
            # covered -- `-p X`, `-p=X`, `-pX`, `--publish X`, `--publish=X` --
            # because the first version matched only `-p X`, so
            # `--publish 0.0.0.0:9000:9000` parsed as NO binding and sailed
            # straight through the loopback check. The regex that covered all
            # five forms backtracked catastrophically on this script (the test
            # hung for over a minute), which is its own argument for tokens.
            toks = joined.replace("\\\n", " ").split()
            k = 0
            while k < len(toks):
                tok = toks[k]
                # `-P` and `--publish-all` publish EVERY exposed port on
                # 0.0.0.0 while contributing no parseable binding, so they slip
                # past the loopback loop entirely and the derived count still
                # balances. Likewise `--network host`, which bypasses port
                # mapping altogether. All three are rejected outright.
                # Flag SHAPE, not token equality: `--publish-all=true` and the
                # combined short form `-dP` are both accepted by Docker and both
                # publish every exposed port on 0.0.0.0, and both passed the
                # equality check. A combined short flag is any single-dash group
                # containing `P` with no `:` (a `-p` binding has one).
                if (
                    tok.split("=")[0] in ("-P", "--publish-all")
                    or (
                        tok.startswith("-")
                        and not tok.startswith("--")
                        and "P" in tok[1:]
                        and ":" not in tok
                    )
                ):
                    publish_all.append(tok)
                if tok in ("--network", "--net") and k + 1 < len(toks):
                    if toks[k + 1].strip("\"'") == "host":
                        host_network.append(tok)
                if tok in ("--network=host", "--net=host"):
                    host_network.append(tok)
                if tok in ("-p", "--publish"):
                    if k + 1 < len(toks):
                        bindings.append(toks[k + 1].strip("\"'"))
                    k += 2
                    continue
                for prefix in ("--publish=", "-p=", "-p"):
                    if not (tok.startswith(prefix) and len(tok) > len(prefix)):
                        continue
                    candidate = tok[len(prefix) :].strip("\"'")
                    # The bare `-pX` form needs a sanity check, because a
                    # container's OWN arguments live in the same block: fake-gcs
                    # is run with `-port 4443 -public-host localhost:4443`, and
                    # a plain prefix match turned `-port` into the binding
                    # `'ort'` and failed the loopback assertion on a correct
                    # script. A real binding always contains a colon.
                    if prefix == "-p" and ":" not in candidate:
                        break
                    bindings.append(candidate)
                    break
                # Both forms: `--name x` and `--name=x`. Counting only the
                # first let a `--name=rogue` container with no binding keep
                # bindings == names, hiding it from the count assertion.
                if tok == "--name" or tok.startswith("--name="):
                    names += 1
                # The IMAGE is the first token that is neither a flag nor a
                # flag's argument. Identified POSITIONALLY rather than by
                # pattern-matching tokens, because the pattern approach missed a
                # bare official image (`redis:7` has no `/`) and an untagged one
                # (`alpine`, which means `:latest`) — both verified unCAUGHT
                # before this change.
                if (
                    image is None
                    and k > run_at
                    and not tok.startswith("-")
                    and not _takes_arg(toks[k - 1])
                ):
                    image = tok
                k += 1
            images.append(image)
        i += 1
    return bindings, names, publish_all, host_network, images


_OVERRIDE_RE = re.compile(r"^\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*):-(?P<default>.+)\}$")


def _image_default(reference: str) -> tuple[str, str | None]:
    """Resolve an image declaration to the reference a default run would use.

    The declarations are overridable -- `${PROXIMADB_CI_X_IMAGE:-vendor/img@sha256:...}`
    -- so CI can point at the GHCR mirror while a local run keeps working against
    the public source with no credentials. The digest-pinning guarantee must still
    hold on the value a default run actually resolves to, which is the default half
    of that form; returning the override variable name too lets a caller assert the
    indirection is the sanctioned one rather than an arbitrary expansion.
    """
    match = _OVERRIDE_RE.match(reference)
    if match is None:
        return reference, None
    return match.group("default"), match.group("var")


class CloudEmulatorHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = HARNESS.read_text(encoding="utf-8")

    def test_tier_gate_filters_name_tests_that_exist(self) -> None:
        """A filter that matches nothing is not an error — so it must be pinned.

        The gate's entire payload is three POSITIONAL `cargo test` filters. A
        name that stops matching (a rename, a typo, a `#[cfg]` change) makes that
        arm a silent no-op: `cargo test … -- --ignored a_name_that_does_not_exist`
        exits **0** and prints `0 passed; 16 filtered out`. The job stays green
        having run nothing, and `ci-success` only sees the exit code.

        This commit RENAMED one of those three filters, which is exactly the
        change that triggers it — so the guard for it belongs here.
        """
        filters = _tier_test_filters(self.source)
        # Non-empty FIRST, so a parse failure is reported as a parse failure
        # rather than as a count mismatch that hides which invariant broke.
        self.assertTrue(
            filters, "could not parse the tier gate's positional test filters"
        )
        self.assertEqual(
            len(filters),
            3,
            f"expected three tier-test filters (azure, s3, gcs); found {filters}",
        )
        src = OBJECT_STORE_SRC.read_text(encoding="utf-8")
        lines = src.splitlines()
        features = _tier_gate_features(self.source)
        self.assertTrue(features, "could not parse the tier gate's --features list")
        for name in filters:
            self.assertIn(
                f"async fn {name}(",
                src,
                f"the tier gate filters on `{name}`, which does not exist in "
                f"{OBJECT_STORE_SRC.name} — that arm would run ZERO tests and "
                "still exit 0",
            )
            # And the FEATURE it is gated on must be in the gate's --features.
            # A feature dropped from that list silently removes the test from the
            # binary, which is the same zero-tests-exit-0 outcome as a rename —
            # and the first version of this guard only checked the name, while
            # its docstring claimed to cover "a `#[cfg]` change". Measured:
            # `--features azure,gcp` + the s3 filter prints
            # `0 passed; 15 filtered out` and exits 0.
            idx = next(i for i, ln in enumerate(lines) if f"async fn {name}(" in ln)
            # The contiguous attribute run directly above the fn, so a
            # neighbouring test's cfg cannot be attributed to this one.
            attrs = []
            for ln in reversed(lines[max(0, idx - 12) : idx]):
                st = ln.strip()
                if st.startswith("#[") or st.startswith("///") or st.startswith("//") or not st:
                    attrs.append(st)
                else:
                    break
            attr_block = "\n".join(attrs)
            gated = [a for a in attrs if a.startswith("#[cfg")]
            # Every `feature = "X"` anywhere in the attribute, so `all(...)`,
            # `any(...)` and `cfg_attr` forms are covered. The first version
            # matched only the exact single-feature spelling and SKIPPED
            # silently otherwise — which is the hole it was added to close.
            feats = re.findall(r'feature\s*=\s*"([^"]+)"', attr_block)
            if gated:
                self.assertTrue(
                    feats,
                    f"`{name}` has a #[cfg] this guard cannot parse ({gated!r}) — "
                    "extend the pattern rather than skipping, or a feature change "
                    "compiles the test out silently",
                )
            for feat in feats:
                self.assertIn(
                    feat,
                    features,
                    f"`{name}` is gated on feature `{feat}`, which is NOT in the "
                    f"tier gate's --features {sorted(features)} — it would be "
                    "compiled out and the filter would match zero tests",
                )

    def test_every_emulator_image_is_pinned_by_digest(self) -> None:
        """All three images, not just the one this migration touched.

        The first version checked only `S3_EMULATOR_IMAGE` while its docstring
        sold digest-pinning as a vendor-independent property — and the Azure
        resident-tier read-back is now a HARD failure on a required check, so it
        depends on an upstream image's `properties.blobTier` shape. A floating
        `:latest` there would reproduce the exact failure TD-CI-6 is about.
        """
        declared = re.findall(r'^([A-Z0-9_]*IMAGE)="([^"]+)"$', self.source, re.MULTILINE)
        # Derived, not hard-coded at 3 — the same criticism that was already
        # applied to the binding count. A legitimate fourth emulator should be
        # CHECKED, not rejected.
        _, _, _, _, used = _docker_run_port_bindings(self.source)
        self.assertGreaterEqual(
            len(declared),
            len(used),
            f"every `docker run` image must come from a declared *_IMAGE variable; "
            f"declared={[n for n, _ in declared]} used={used}",
        )
        for name, declaration in declared:
            reference, override = _image_default(declaration)
            if override is not None:
                # Only the sanctioned override shape, so a declaration cannot be
                # turned into an arbitrary shell expansion that this guard then
                # reads as "pinned".
                self.assertEqual(
                    override,
                    f"PROXIMADB_CI_{name}",
                    f"{name} may only be overridden by PROXIMADB_CI_{name}",
                )
            self.assertRegex(
                reference,
                r"@sha256:[0-9a-f]{64}$",
                f"{name} must be immutable: pinned by digest, never a tag "
                f"(declaration: {declaration})",
            )
        # And every `docker run` must take its image from one of those variables.
        # Checked POSITIONALLY (the image is the first non-flag, non-flag-argument
        # token), not by pattern-matching tokens: the pattern version required a
        # `/` and a `:`, so `redis:7` (official image, no slash) and bare
        # `alpine` (implicit `:latest`) both slipped through — verified before
        # this change.
        _, _, _, _, images = _docker_run_port_bindings(self.source)
        self.assertTrue(images, "no `docker run` image arguments found")
        for image in images:
            self.assertIsNotNone(
                image,
                "a `docker run` block has no identifiable image argument — if a "
                "flag that consumes a token was added, extend _DOCKER_ARG_FLAGS",
            )
            self.assertTrue(
                (image or "").startswith("$") or (image or "").startswith('"$'),
                f"`docker run` uses the literal image '{image}' — declare it as a "
                "digest-pinned *_IMAGE variable so it cannot float",
            )

    def test_s3_emulator_image_is_pinned_by_digest(self) -> None:
        image = re.search(r'^S3_EMULATOR_IMAGE="([^"]+)"$', self.source, re.MULTILINE)
        self.assertIsNotNone(image, "the shared harness must declare S3_EMULATOR_IMAGE")
        reference, _ = _image_default(image.group(1) if image else "")
        self.assertRegex(
            reference,
            r"^[a-z0-9./-]+@sha256:[0-9a-f]{64}$",
            "the S3 emulator must be immutable: pinned by digest, never a tag",
        )
        # A floating tag is the specific failure this guards: `:latest` or any
        # `:tag` form can change under CI, and in TD-CI-6's case a newer tag of
        # the same repository became licence-gated.
        self.assertNotRegex(
            reference,
            r":(latest|[0-9]+(\.[0-9]+)*)$",
            "a tag reference is mutable; pin the digest",
        )

    def test_every_emulator_binds_only_to_loopback(self) -> None:
        # Checks the PROPERTY of every `-p` argument rather than matching port
        # literals: the S3 port is passed as a variable, so a literal-matching
        # assertion fails on a correct script (it did) and would have to be
        # edited for any port change — which is how a security check turns into
        # a maintenance tax and then gets relaxed.
        bindings, containers, publish_all, host_network, _ = _docker_run_port_bindings(
            self.source
        )
        self.assertEqual(
            publish_all,
            [],
            f"{publish_all} publishes every exposed port on all interfaces and "
            "contributes no binding to check",
        )
        self.assertEqual(
            host_network,
            [],
            f"{host_network} bypasses port mapping entirely, so no binding "
            "assertion can constrain it",
        )
        self.assertTrue(bindings, "no port bindings found in the harness")
        for binding in bindings:
            self.assertTrue(
                binding.startswith("127.0.0.1:"),
                f"binding '{binding}' is not loopback-qualified; emulators run "
                "with development credentials and must not be reachable off-host",
            )
        # Derived from the number of `--name`d containers, not hard-coded at 3.
        # A hard-coded count passes when a FOURTH emulator is added with no
        # binding at all (or one the parser misses), because the loop then has
        # nothing bad to inspect — and it hard-fails on a legitimate fourth
        # emulator instead of checking it.
        self.assertEqual(
            len(bindings),
            containers,
            f"every `docker run` must publish its port: {containers} container(s) "
            f"named but {len(bindings)} binding(s) found ({bindings})",
        )

    def test_readiness_is_gated_on_a_real_s3_call(self) -> None:
        # Measured in TD-CI-6: `/_localstack/health` can answer 200 while S3
        # still returns NoSuchBucket for a bucket just created, and S3 can be
        # serving before that endpoint answers at all. So the gate must be an
        # actual S3 operation.
        self.assertIn("wait_s3", self.source, "readiness must be gated on an S3 call")
        self.assertRegex(
            self.source,
            r"wait_s3\(\)\s*\{[^}]*s3api list-buckets",
            "the readiness gate must issue a real S3 request",
        )

    def test_startup_failures_are_diagnosable(self) -> None:
        # A timeout that reports only "did not come up" cannot distinguish "still
        # booting" from "exited on startup" — the licence failure was the latter,
        # and cost a CI round to identify.
        #
        # Asserts the COMMAND IS INVOKED, not that the string appears: the first
        # version checked `"docker logs" in source`, and a mutation replacing the
        # real call with `true` survived it, because the phrase also occurs in the
        # `echo "::group::docker logs $1"` label right above it.
        diag = re.search(r"diagnose\(\)\s*\{([\s\S]*?)\n\}", self.source)
        self.assertIsNotNone(diag, "the harness must define a `diagnose` helper")
        body = diag.group(1) if diag else ""
        self.assertRegex(
            body,
            re.compile(r"^\s*docker logs\b", re.MULTILINE),
            "the diagnose helper must actually run `docker logs`",
        )
        self.assertRegex(
            body,
            re.compile(r"^\s*docker inspect\b", re.MULTILINE),
            "and report container state/exit code, which is what distinguishes "
            "'still booting' from 'exited on startup'",
        )
        # Every wait helper the harness actually defines must use it, discovered
        # rather than listed: the first version hard-coded ("wait_port",
        # "wait_http") and so PINNED `wait_http` in place after its last call
        # site was deleted — a CI check holding 12 lines of dead shell.
        # All the legal definition forms: `f() {`, `f () {` (space before the
        # parens), `function f {`, and any of them indented. The first version
        # required `^wait_\w+\(\)` exactly, so a helper written in any of the
        # other three forms had its missing `diagnose` go unchecked.
        helpers = re.findall(
            r"^\s*(?:function\s+)?(wait_\w+)\s*(?:\(\))?\s*\{", self.source, re.MULTILINE
        )
        self.assertTrue(helpers, "the harness must define at least one wait helper")
        for helper in helpers:
            # One-liners FIRST (`f() { cmd; }`). The multi-line pattern requires
            # a `}` at column 0, so on a one-liner it ran past the end and
            # captured the NEXT helper's body — the assertion then passed on a
            # neighbour's `diagnose`, which is a guard passing while the thing it
            # guards is absent.
            head = rf"^\s*(?:function\s+)?{helper}\s*(?:\(\))?\s*\{{"
            m = re.search(head + r"(.*?)\}\s*$", self.source, re.MULTILINE)
            if m is None:
                m = re.search(head + r"([\s\S]*?)\n\}", self.source, re.MULTILINE)
            self.assertIsNotNone(m, f"could not read the body of `{helper}`")
            self.assertNotRegex(
                m.group(1) if m else "",
                r"wait_\w+\s*(?:\(\))?\s*\{",
                f"the captured body of `{helper}` runs into another helper — the "
                "extraction is borrowing a neighbour's body",
            )
            self.assertIn(
                "diagnose",
                m.group(1) if m else "",
                f"`{helper}` must dump diagnostics on timeout, or its failure is silent",
            )

    def test_bucket_creation_is_idempotent(self) -> None:
        self.assertIn("create_s3_bucket", self.source)
        self.assertRegex(
            self.source,
            r"create_s3_bucket\(\)\s*\{[\s\S]*?head-bucket",
            "bucket creation must tolerate an already-existing bucket",
        )

    def test_no_stale_vendor_wiring_remains(self) -> None:
        # The old harness ran MinIO with an explicit server command and polled a
        # MinIO-specific health path; both are meaningless now and would silently
        # do nothing if left behind.
        #
        # CODE lines only. This script is deliberately a record of the emulator
        # migration history, and TD-CI-6 quotes `/minio/health/ready` — so
        # scanning comments would fail the guard on exactly the prose it is
        # supposed to encourage.
        code = "\n".join(
            ln for ln in self.source.splitlines() if not ln.lstrip().startswith("#")
        )
        for stale in ('"$MINIO_IMAGE"', "/minio/health/ready", "create_minio_bucket"):
            self.assertNotIn(
                stale, code, f"stale MinIO wiring left in the harness: {stale}"
            )


if __name__ == "__main__":
    unittest.main()
