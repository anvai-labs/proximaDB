# Release procedure: one tag `vX.Y.Z`, every registry, one version

| Field | Value |
|---|---|
| Status | **Active.** Adopted 2026-10-15 ([TD-RELTAG-1](../../10-quality/td/TD-RELTAG-1-one-release-tag-for-every-registry.adoc)); first release under it is 0.4.1. |
| Applies to | Every product release: GitHub Release binaries, crates.io, npm, PyPI |
| Enforced by | `scripts/version-sync.sh check-tag` (run by both release workflows before any build); `scripts/check_release_build_features.py` (CI); `scripts/check_release_smoke.py` (CI: the release-evidence contract) |
| Related | [Branch promotion procedure](PROMOTION_PROCEDURE.adoc) — how the release commit reaches `main` (Asciidoc, not rendered on the docs site) |

## The rule

**One tag `vX.Y.Z` releases every artifact, and every package carries the
identical version `X.Y.Z`.** There is no per-registry tag (`python-v*` is
retired; the old `python-v0.2.2` / `python-v0.3.0` tags are history) and no
per-package version hold.

| Registry / artifact | Workflow (trigger) | Version source | Gate |
|---|---|---|---|
| GitHub Release: server binaries, RPM/DEB/MSI, sdist | `release.yml` (`vX.Y.Z`) | `Cargo.toml` `[package]` / `[workspace.package]` | `prepare`: `version-sync.sh check-tag` |
| crates.io `proximadb-sdk` | `release.yml` (`vX.Y.Z`, stable only) | `clients/rust/Cargo.toml` | `prepare` gate + per-job assert before `cargo publish` |
| npm `@anvailabs/proximadb-client` | `release.yml` (`vX.Y.Z`, stable only) | `clients/nodejs-embedded/package.json` | `prepare` gate + per-job assert before `npm publish` |
| PyPI `proximadb` | `release-python.yml` (`vX.Y.Z`, stable only) | `clients/python/pyproject.toml` | `version-gate`: `version-sync.sh check-tag` |
| PyPI `proximadb_embedded` | `release-python.yml` (`vX.Y.Z`, stable only) | root `pyproject.toml` (maturin) | `version-gate`: `version-sync.sh check-tag` |
| TestPyPI (advisory) | `release-python.yml` (any `vX.Y.Z[-pre]`, or dispatch) | as above | as above |

A tag with a hyphen (`v0.5.0-rc.1`) is a pre-release in both workflows: it builds
everything and creates a pre-release GitHub Release and a TestPyPI upload, but
never publishes to crates.io, npm or PyPI.

The two workflow files stay separate on purpose. Trusted publishing (OIDC) is
bound to the **file name**: npm to `release.yml` (environment `npm`), PyPI to
`release-python.yml` (environments `pypi`, `testpypi`). Renaming or merging them
breaks publishing until the owner re-registers on npmjs.com / pypi.org.

## Cutting a release

1. **Bump every manifest in one PR** (to `develop`, promoted per the
   [promotion procedure](PROMOTION_PROCEDURE.adoc)):

   ```bash
   bash scripts/version-sync.sh set X.Y.Z     # all manifests + lockfiles (cargo update -w --offline)
   bash scripts/version-sync.sh check-tag vX.Y.Z
   ```

   Pick a version unused on **every** registry: check crates.io, npm and both
   PyPI projects. A version consumed on any one of them (even by a failed or
   partial release) is burned for all of them; move to the next patch.

2. **Dry-run before tagging** (no publishing):

   ```bash
   gh workflow run release.yml        --ref main -f version=X.Y.Z -f dry_run=true
   gh workflow run release-python.yml --ref main -f dry_run=true
   ```

3. **Push ONE tag** on the promoted `main` commit:

   ```bash
   git tag vX.Y.Z <main-sha> && git push origin vX.Y.Z
   ```

   Both `release.yml` and `release-python.yml` start. If the tag disagrees with
   any manifest, both fail in their gate job with nothing built or published.
   Fix by bumping in a PR and tagging the new commit; never move a tag that
   already produced a GitHub Release.

4. **Verify.** `release.yml`'s `Verify Release` checks crates.io and npm serve
   `X.Y.Z`; its last job, `Release Smoke (evidence)`, then runs
   `release-smoke.yml` (see [Release evidence](#release-evidence)). The release
   is done when that job is green and the run carries the
   `release-evidence-vX.Y.Z` artifact.

## Re-running a partial release

PyPI (`skip-existing`) and npm (a registry probe before `npm publish`) are
idempotent. crates.io is not: a re-run of `publish-crates` after a successful
upload fails on the duplicate, which is harmless (the version is live). Re-run
the failed jobs of the tag's run; do not push a second tag for the same version.

## Release evidence

A release is finished only when it has **evidence** for two claims, produced by
[`.github/workflows/release-smoke.yml`](https://github.com/anvai-labs/proximaDB/blob/develop/.github/workflows/release-smoke.yml)
from the published artifacts themselves (never from the release workflow's own
say-so):

| Claim | Job | What it checks |
|---|---|---|
| (A) built from a QA-proven tree | `provenance` | `vX.Y.Z` -> commit on `main`; it is the merge commit of a merged `qa` -> `main` PR; the `qa` head has the **same git tree** as the tag; a building Pre-Release CI run passed on it (`scripts/await_prerelease_ci.sh`); `CI Success` passed on the `qa` push run **and** on the qa -> main PR run |
| (B) Docker Hub image | `docker` | pull `<image>:X.Y.Z`; its `org.opencontainers.image.revision` has the tag's tree; the pulled digest is the hub's; a surviving `tree-<tree>` tag points at the same manifest; `/health` reports `X.Y.Z`; `scripts/mvp_smoke.py` |
| (B) GitHub Release binary | `binary` | download the linux x86_64 archive; `sha256sum -c` against both the `.sha256` and `SHA256SUMS.txt`; `proximadb-server --version`; start it on a temp data dir (shipped config, relocated); `scripts/mvp_smoke.py` |
| (B) PyPI `proximadb` + `proximadb_embedded` | `python` (3.11, 3.12) | fresh venv, `pip install --no-cache-dir`, `importlib.metadata.version == X.Y.Z`; SDK create / insert / search / delete against the released image; embedded graph + typed-property roundtrip |
| (B) npm `@anvailabs/proximadb-client` | `npm` | `npm install` into a temp project; installed version; health / create / insert / search / delete through the CommonJS entry; health through the ESM (`import`) entry |
| (B) crates.io `proximadb-sdk` | `rust` | temp crate with `proximadb-sdk = "=X.Y.Z"` (lockfile must say crates.io); health / create / insert / search / delete against the released image |
| (B) Homebrew | `homebrew` | the tap formula exists, references `vX.Y.Z`, and every release url's `sha256` equals `SHA256SUMS.txt` |

`summary` writes one table to the run summary and uploads
`release-evidence-vX.Y.Z.json`; it fails unless **every** job succeeded (a
skipped job is missing evidence, not a pass). Every step fails hard (no
`|| true`, no `continue-on-error`) and registry propagation is a bounded wait
(`scripts/release_smoke/await_url.sh`, `retry.sh`).

**How it runs.** `release.yml` calls it as its final job (`workflow_call`), so a
red smoke makes the release run red; a `workflow_run` trigger could not, so
there is none. It also runs on a manually published GitHub Release, and on
demand for any past release:

```bash
gh workflow run release-smoke.yml -f version=X.Y.Z
```

Every smoke script lives in `scripts/release_smoke/` and runs locally against
the real registries with the same arguments. The contract (every published
channel smoked, fail-hard, provenance, aggregation, the `release.yml` hook) is
guarded by `scripts/check_release_smoke.py` (`make release-evidence-contract-check`,
part of `make work-commit-check`, and the CI `Deterministic Commit Contract` job).
A new publish channel in `release.yml`, `release-python.yml` or
`publish-image.yml` fails that guard until it has a smoke job.

**First run, v0.4.1 (2026-10-17, from a workstation).** Red, on real defects:
the qa -> main PR #2053 was merged over a red `CI Success` (a `Feature Matrix`
runner shutdown); the released image and binary both fail `mvp_smoke.py` --
a deleted record is still returned by a **filtered** vector search (any
collection; unfiltered search, filtered scan and get-by-id are correct); the
npm package's ESM entry (`exports.import` -> `dist/index.mjs`) does not exist
in the tarball, so `import` fails for every ESM consumer; and the Homebrew tap
has no `proximadb` formula (`release.yml`'s `update-homebrew` only uploads an
unpinned formula as a workflow artifact). The PyPI, crates.io and npm-CommonJS
smokes passed.

