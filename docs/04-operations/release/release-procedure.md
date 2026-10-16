# Release procedure: one tag `vX.Y.Z`, every registry, one version

| Field | Value |
|---|---|
| Status | **Active.** Adopted 2026-10-15 ([TD-RELTAG-1](../../10-quality/td/TD-RELTAG-1-one-release-tag-for-every-registry.adoc)); first release under it is 0.4.1. |
| Applies to | Every product release: GitHub Release binaries, crates.io, npm, PyPI |
| Enforced by | `scripts/version-sync.sh check-tag` (run by both release workflows before any build); `scripts/check_release_build_features.py` (CI) |
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
   `X.Y.Z`. Check PyPI for both projects:

   ```bash
   curl -s https://pypi.org/pypi/proximadb/json          | python3 -c 'import json,sys;print(json.load(sys.stdin)["info"]["version"])'
   curl -s https://pypi.org/pypi/proximadb_embedded/json | python3 -c 'import json,sys;print(json.load(sys.stdin)["info"]["version"])'
   ```

## Re-running a partial release

PyPI (`skip-existing`) and npm (a registry probe before `npm publish`) are
idempotent. crates.io is not: a re-run of `publish-crates` after a successful
upload fails on the duplicate, which is harmless (the version is live). Re-run
the failed jobs of the tag's run; do not push a second tag for the same version.
