#!/usr/bin/env bash
# Spin up cloud object-store emulators (Azurite / MinIO / fake-gcs-server) via
# Docker, create the test bucket+container, and run the #[ignore]d object-store
# tier integration tests against them — the TD-168 "validate the Cool tier on a
# real cloud API" check. Used by .github/workflows/qa-gate.yml (the develop→qa
# gate), .github/workflows/ci.yml (the develop early-detection job, --fast), and
# `make cloud-emulator-test` (local). Single source of truth so every CI path is
# exactly the locally-runnable path.
#
# Usage: run_cloud_emulator_tests.sh [--fast|--all|--restart|--qa|--nightly]
#   --fast : run ONLY the cheap object-store tier tests (~3-5 min — compiles just
#            the small object-store crate, no main-crate compile, no OOM risk).
#            Used by the develop early-detection job so a tier regression is caught
#            on the introducing feat→develop PR.
#   --all  : (default) also run cold_graph_record_store_round_trips_on_real_azure,
#            which compiles the full main crate and runs under CARGO_BUILD_JOBS=2.
#   --restart : ONLY the full-server object-store RESTARTABILITY recovery proof
#            (TD-OBJSTORE-5 S1, ADR-063 D8 PR tier) — spawns the real
#            `proximadb-server` binary against an Azurite (adls://) prefix, SIGKILLs
#            it, and recovers catalog + WAL-replays collections on a fresh local
#            disk. Compiles the server binary, so it gets its OWN scope/job (kept
#            off --fast/--all's compile budget).
#
# Requires: docker, cargo, aws CLI (MinIO bucket), az CLI (Azurite container),
# curl (fake-gcs bucket). On GitHub ubuntu-latest all are preinstalled.
#
# Azure + S3 are strict (PUT-with-tier must be accepted + round-trip). GCS is
# best-effort: the test itself skips on a fake-gcs/object_store incompatibility,
# so this script never fails on GCS alone.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Scope selection (see header). --fast = object-store tier tests only; --all (default)
# adds the cold-payload azure round-trip, which compiles the full main crate.
SCOPE="all"
for arg in "$@"; do
  case "$arg" in
    --fast) SCOPE="fast" ;;
    --all)  SCOPE="all" ;;
    --restart) SCOPE="restart" ;;
    --qa) SCOPE="qa" ;;
    --nightly) SCOPE="nightly" ;;
    *) echo "::error::unknown argument: $arg"; echo "usage: $0 [--fast|--all|--restart|--qa|--nightly]"; exit 2 ;;
  esac
done
echo "==> Scope: $SCOPE"

CONTAINER_BUCKET="proximadb-test"
AZURITE_CONN="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
# Docker Hub's minio/minio repository stopped serving pulls after the OSS
# project was archived. This is the last official AGPL container image,
# pulled from MinIO's Quay repository and pinned to its multi-architecture
# manifest digest.
MINIO_IMAGE="quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"

cleanup() { docker rm -f azurite minio fake-gcs >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_port() { # $1=port $2=name
  for _ in $(seq 1 30); do
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- ; echo "  $2 (:$1) up"; return 0; }
    sleep 2
  done
  echo "::error::$2 (:$1) did not come up"; return 1
}

wait_http() { # $1=url $2=name
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "$1" >/dev/null; then
      echo "  $2 ready"
      return 0
    fi
    sleep 2
  done
  echo "::error::$2 readiness endpoint did not become healthy"
  return 1
}

minio_aws() {
  AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    AWS_DEFAULT_REGION=us-east-1 \
    aws --endpoint-url http://127.0.0.1:9000 "$@"
}

create_minio_bucket() {
  for _ in $(seq 1 15); do
    if minio_aws s3api head-bucket --bucket "$CONTAINER_BUCKET" >/dev/null 2>&1; then
      echo "  minio bucket already exists"
      return 0
    fi
    if minio_aws s3api create-bucket --bucket "$CONTAINER_BUCKET" >/dev/null 2>&1; then
      echo "  minio bucket created"
      return 0
    fi
    sleep 2
  done
  echo "::error::failed to create or verify MinIO bucket '$CONTAINER_BUCKET'"
  return 1
}

echo "==> Starting emulators (Docker)"
cleanup
docker run -d --name azurite -p 127.0.0.1:10000:10000 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck >/dev/null
docker run -d --name minio -p 127.0.0.1:9000:9000 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  "$MINIO_IMAGE" server /data >/dev/null
docker run -d --name fake-gcs -p 127.0.0.1:4443:4443 \
  fsouza/fake-gcs-server -scheme http -port 4443 -public-host localhost:4443 >/dev/null

echo "==> Waiting for emulators"
wait_port 10000 azurite
wait_port 9000 minio
wait_port 4443 fake-gcs
wait_http http://127.0.0.1:9000/minio/health/ready minio

echo "==> Creating bucket/container '$CONTAINER_BUCKET'"
# MinIO bucket (aws CLI)
create_minio_bucket
# Azurite container (az CLI)
az storage container create --name "$CONTAINER_BUCKET" --connection-string "$AZURITE_CONN" >/dev/null 2>&1 \
  || echo "  (azurite container exists / az CLI missing)"
# fake-gcs bucket (JSON API, no auth)
curl -sf -X POST "http://127.0.0.1:4443/storage/v1/b?project=proximadb" \
  -H "Content-Type: application/json" -d "{\"name\":\"$CONTAINER_BUCKET\"}" >/dev/null 2>&1 \
  || echo "  (fake-gcs bucket exists)"

echo "==> Running object-store tier integration tests (scope: $SCOPE)"
# Azure (Azurite) + S3 (MinIO) through the production from_url + env path; GCS via builder.
# Both emulator env vars are hoisted here (global) so EVERY Azure-touching test in
# every scope honors Azurite — including the --qa backend-contract test (#1129),
# which runs BEFORE the --qa restart section's own PROXIMADB_AZURE_EMULATOR export.
# The engine's azure_config_from_env reads PROXIMADB_AZURE_EMULATOR (not the
# object_store-convention AZURE_STORAGE_USE_EMULATOR); without it, the contract
# test built real Azure → OIDC "Identity not found" (the qa-gate failure).
export PROXIMADB_AZURE_EMULATOR=1 AZURE_STORAGE_USE_EMULATOR=true AZURE_ALLOW_HTTP=true
export AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_REGION=us-east-1
export PROXIMADB_GCS_TEST_ENDPOINT=http://127.0.0.1:4443
# The production root FileSystem GCS backend (contract tests) reads these:
export PROXIMADB_GCS_ENDPOINT=http://127.0.0.1:4443 PROXIMADB_GCS_ANONYMOUS=1 GCP_PROJECT=proximadb

if [ "$SCOPE" = "restart" ]; then
  # TD-OBJSTORE-5 S1 (ADR-063 D8 PR tier = Azurite-strict): prove full-server
  # object-store RESTARTABILITY recovery in the runner. Spawns the real
  # `proximadb-server` three times against ONE Azurite (adls://) prefix with a
  # fresh local `server.data_dir` each time: CREATE+INSERT → SIGKILL → recover
  # catalog from metadata_url + WAL-replay reattaches SST/HELIX collections →
  # SIGINT (flush SST) → cold read on another empty disk. All durable state lives
  # only in the object store (ADR-048 stateless catalog). Isolated scope: it
  # compiles the proximadb-server binary (heavy, CARGO_BUILD_JOBS=1 for OOM
  # safety), so it runs in its own path-gated CI job, not on --fast/--all's budget.
  # The zero-vector-KV restart arm (the TD-OBJSTORE-1 batch-3 gate) is IN the
  # gate: TD-OBJSTORE-4 S1 (#1061 + the reconciled S1 PR) greened it end-to-end.
  echo "==> TD-OBJSTORE-5 S1: full-server restart recovery over Azurite (adls://)"
  # Azurite = ADLS Blob emulator; the production adls:// from_url + emulator env.
  export PROXIMADB_AZURE_EMULATOR=1 AZURE_STORAGE_USE_EMULATOR=true AZURE_ALLOW_HTTP=true
  export AZURE_STORAGE_ACCOUNT=devstoreaccount1 AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1
  export PROXIMADB_OBJECT_STORE_URL="adls://$CONTAINER_BUCKET/objstore-restart-recovery"
  CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features azure \
    --test object_store_restart_recovery \
    -- --ignored --nocapture --test-threads=1
  # TD-CACHE-7: restart-warming manifest write/read round-trip on object storage
  # — the boundary that repeatedly regressed with zero coverage (container-strip
  # / malformed prefix). Reuses the exported PROXIMADB_OBJECT_STORE_URL.
  echo "==> TD-CACHE-7: warm-manifest round-trip over Azurite (adls://)"
  CARGO_BUILD_JOBS=1 cargo test -p proximadb --features azure --lib \
    warm_manifest_round_trips_on_object_store \
    -- --ignored --nocapture --test-threads=1
  # TD-OBJSTORE-5 S2 (backend contract tests) moved to --qa — see below.
  # Keeping it here forced a 3rd cold feature-set compile (aws,azure,gcp)
  # that, combined with the two azure builds above, exceeded the job's timeout
  # budget. In --qa it's a near-free incremental build on the existing
  # cloud-full artifact (cloud-full = aws+azure+gcp — identical feature set).
  echo "==> Restart-recovery validation complete (cleanup trap tears emulators down)"
  exit 0
fi

if [ "$SCOPE" = "qa" ]; then
  # TD-OBJSTORE-5 S3 (ADR-063 D8 QA tier): one cloud-full server build, then the
  # restart proofs per STRICT store — Azure (Azurite) first, then S3 (MinIO) with
  # Azurite stopped so an S3 run that accidentally reaches Azure fails loudly
  # (cross-store isolation is part of the proof). The recall ratchet runs on ONE
  # strict backend (Azure): recall is a ranged-read/footer-fidelity check, not
  # the restart-correctness gate. Build BEFORE exercising emulators; jobs=1 for
  # the 16GB-runner OOM ceiling (same rationale as --all). The QA budget is a
  # measured ratchet: record the cold-cache wall time in TD-OBJSTORE-5 on the
  # first run.
  echo "==> TD-OBJSTORE-5 QA tier: build server (cloud-full) before emulator runs"
  CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features cloud-full \
    --test object_store_restart_recovery --no-run

  # TD-OBJSTORE-5 S2: backend contract tests (moved from --restart to --qa for
  # compile efficiency). cloud-full (= aws+azure+gcp) is already built above;
  # this is a near-free incremental build + run. Emulators (Azurite/MinIO/gcs)
  # are started by the setup trap before this scope runs.
  echo "==> TD-OBJSTORE-5 S2: backend contract tests (Azure/S3 strict, GCS best-effort)"
  CARGO_BUILD_JOBS=1 cargo test -p proximadb --features cloud-full \
    --test objstore_backend_contract_test \
    -- --ignored --nocapture --test-threads=1

  echo "==> QA tier [1/2]: Azure (Azurite, adls://) — restart proofs + recall ratchet"
  export PROXIMADB_AZURE_EMULATOR=1 AZURE_STORAGE_USE_EMULATOR=true AZURE_ALLOW_HTTP=true
  export AZURE_STORAGE_ACCOUNT=devstoreaccount1 AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1
  PROXIMADB_OBJECT_STORE_URL="adls://$CONTAINER_BUCKET/qa-restart-azure" \
    CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features cloud-full \
    --test object_store_restart_recovery \
    -- --ignored --nocapture --test-threads=1

  echo "==> QA tier: stopping Azurite (S3 must not be able to reach Azure)"
  docker rm -f azurite >/dev/null 2>&1 || true

  echo "==> QA tier [2/2]: S3 (MinIO, s3://) — restart proofs"
  export AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false
  export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_REGION=us-east-1
  # Both restart proofs; the recall ratchet runs on ONE strict backend (Azure),
  # so skip it here (cargo test accepts a single positional filter — use --skip).
  PROXIMADB_OBJECT_STORE_URL="s3://$CONTAINER_BUCKET/qa-restart-s3" \
    CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features cloud-full \
    --test object_store_restart_recovery \
    -- --ignored --nocapture --test-threads=1 --skip cold_recall_ratchet

  echo "==> QA tier complete (Azure strict + S3 strict; GCS lives in --nightly)"
  exit 0
fi

if [ "$SCOPE" = "nightly" ]; then
  # TD-OBJSTORE-5 S4 (ADR-063 D8 nightly tier): GCS (fake-gcs) restart + recall,
  # BEST-EFFORT — fake-gcs/object_store incompatibilities are documented, so
  # failures warn and never block promotion (GCS is explicitly NOT called a gate
  # while warn-only). Runs on a schedule, off the promotion path.
  echo "==> TD-OBJSTORE-5 nightly tier: GCS (fake-gcs, gs://) — best-effort"
  export PROXIMADB_GCS_TEST_ENDPOINT=http://127.0.0.1:4443
  export STORAGE_EMULATOR_HOST=http://127.0.0.1:4443
  if PROXIMADB_OBJECT_STORE_URL="gs://$CONTAINER_BUCKET/nightly-restart-gcs" \
    CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features cloud-full \
    --test object_store_restart_recovery \
    -- --ignored --nocapture --test-threads=1; then
    echo "==> GCS nightly restart + recall: PASS"
  else
    echo "::warning::GCS nightly restart/recall failed (best-effort tier — file an issue, do not block)"
  fi
  exit 0
fi

cargo test -p proximadb-object-store --features aws,azure,gcp -- --ignored --nocapture \
  put_with_tier_accepted_by_azurite \
  put_with_tier_accepted_by_minio \
  put_with_tier_against_fake_gcs

if [ "$SCOPE" = "all" ]; then
  # Compiles the full main `proximadb` crate (~8400-test binary). CARGO_BUILD_JOBS=1
  # (serial): a single root-crate rustc peaks at ~12.8GB, so jobs>=2 is a HARD OOM on
  # the 16GB hosted runner (12.8 + concurrent >= 16) — jobs=2 failed at ~26m with
  # "runner received a shutdown signal" on the #992 promotion's cold compile. jobs=1
  # caps peak RSS at one ~12.8GB rustc (fits), trading speed for reliability — this
  # mirrors the rust-test ci.yml job, which also runs jobs=1 for the same reason.
  # The 90m qa-gate budget + sccache absorb the ~35-40m serialized cold compile;
  # warm runs compile far less. Durable fix = root-crate extraction (lower peak RSS).
  CARGO_BUILD_JOBS=1 cargo test -p proximadb --features azure -- --ignored --nocapture \
    cold_graph_record_store_round_trips_on_real_azure
fi

echo "==> Strong read-back: confirm Azurite persisted the Cool tier"
TIER="$(az storage blob show --container-name "$CONTAINER_BUCKET" --name cold/probe-azure.bin \
  --connection-string "$AZURITE_CONN" --query properties.blobTier -o tsv 2>/dev/null || echo "")"
echo "  Azurite blobTier(cold/probe-azure.bin) = '${TIER:-<unknown>}'"
if [ "$TIER" = "Cool" ]; then
  echo "  ✅ Cool tier persisted on a real Azure API"
else
  echo "::warning::Azurite did not report Cool tier (got '${TIER:-<unknown>}') — header accepted but resident tier unverified"
fi

echo "==> Cloud emulator tier validation complete"
