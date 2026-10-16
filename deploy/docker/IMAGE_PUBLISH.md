# Publishing the ProximaDB OSS image to Docker Hub

`.github/workflows/publish-image.yml` builds the OSS server image
(`deploy/docker/Dockerfile`, target `runtime`) and pushes a **multi-arch
manifest (`linux/amd64` + `linux/arm64`)** to **Docker Hub** as
`vjsingh1984/proximadb`.

This is the **public OSS image**. AnvaiOps' *commercial* pipeline
(`anvaiops/.github/workflows/build-commercial-image.yml`) consumes it as a
`FROM` base, bakes pricing tiers in, and pushes the result to **ACR** — the OSS
image itself is on Docker Hub, not ACR.

## Architectures

Built for both **amd64** and **arm64** so the same tag runs on x86 clouds,
Graviton/Ampere, Apple-Silicon dev, and anvaiops' arm64 node pool (ADR-0016 /
`proximadb_node_arch = "arm64"`). Each arch builds **natively** on its own
hosted runner (free for public repos: `ubuntu-latest` = amd64,
`ubuntu-24.04-arm` = arm64), is pushed **by digest**, and a final `merge` job
combines both digests into one multi-arch manifest list that carries the tags.
QEMU emulation is intentionally avoided — the heavy Rust release build would be
far too slow under it.

## What gets published

| Trigger | Tags |
|---|---|
| Push tag `v<X.Y.Z>` | `<X.Y.Z>`, `sha-<short>`, `latest` |
| Push to `qa` (develop→qa promotion) | `qa`, `sha-<short>`, `tree-<git tree hash>` (no `latest`) |
| `workflow_dispatch` | `<version-from-Cargo.toml>`, `sha-<short>`, `latest` (if `tag_latest=true`) |

Every trigger publishes both targets: `runtime` and `runtime-full` (ONNX +
baked bge-small), the latter under a parallel `-full`-suffixed tag namespace.

**A release tag reuses the qa image of the same tree.** The `resolve` job
computes `git rev-parse HEAD^{tree}` for the tagged commit and looks up
`tree-<tree>` and `tree-<tree>-full`. When both exist (qa built exactly this
source), `merge` points the release tags at those manifest lists with
`docker buildx imagetools create` and no image is rebuilt; the image's
`org.opencontainers.image.revision` label then names the qa commit that built
it. When either is missing, the tag builds its images as before.

## One-time setup

Repository **secrets**:

| Secret | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account that owns `vjsingh1984/proximadb` |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Account Settings → Security → New Access Token, Read/Write/Delete) |

Repository **variables** (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `DOCKERHUB_IMAGE` | `vjsingh1984/proximadb` | Override the published repo |
| `DOCKERHUB_PRUNE_KEEP` | `10` | Number of newest `sha-*` tags, and of `tree-*` tags, to retain |

The `DOCKERHUB_TOKEN` needs **delete** scope for the prune job to remove stale
`sha-*` and `tree-*` tags.

## Downstream consumption (AnvaiOps commercial overlay)

Once published, in **anvaiops** set the repo variable
`PROXIMADB_OSS_BASE=vjsingh1984/proximadb` and flip
`OSS_BASE_IMAGE_AVAILABLE=true`, then pin the base **by digest** (ADR-0016) —
the digest is printed in this workflow's run summary — via the
`proximadb_version` dispatch input / build arg in `build-commercial-image.yml`.
AKS pulls the public image directly (no imagePullSecret needed).

## Cost control

The `prune` job runs after every successful push and deletes superseded rolling
`sha-*` and `tree-*` tags via the Docker Hub API, keeping the newest
`DOCKERHUB_PRUNE_KEEP` of each family. It **never** touches semver tags,
`latest`, `qa`, or `develop`.

## Local build (parity check)

```bash
# minimal OSS server (default target)
docker build -f deploy/docker/Dockerfile --target runtime -t proximadb:local .

# full image with ONNX + baked bge-small-en-v1.5
docker build -f deploy/docker/Dockerfile --target runtime-full -t proximadb:local-full .
```

## Fast prebuilt image (decoupled compile vs. bake)

`deploy/docker/Dockerfile.prebuilt` bakes an **already-built** cloud-full server
binary into the same hardened, non-root, Python-free runtime — **no in-Docker
Rust compile**. Compile once (host/CI, with sccache), then bake in seconds:

```bash
# Linux host / CI — the binary is COPY'd, so it must be a Linux binary for the
# target arch (a macOS host binary will NOT run; use the multi-stage Dockerfile
# locally on Mac).
cargo build --release --features cloud-full -p proximadb-server
docker build -f deploy/docker/Dockerfile.prebuilt \
  --build-arg BIN=target/release/proximadb-server -t proximadb:preview .
```

Measured ~13s to bake vs. minutes for the in-Docker compile. This is the
foundation for cheap **per-PR preview images** (build the binary once in CI,
reuse it across the bake + smoke deploy). Wiring per-PR `pr-<n>-<sha>` tags
with auto-prune-on-merge into `publish-image.yml` is a tracked follow-up —
deliberately on-demand rather than auto-building every PR push (4 native Rust
builds per push is too costly, and fork PRs can't access the registry secrets).
