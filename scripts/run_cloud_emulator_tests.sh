#!/usr/bin/env bash
# Spin up cloud object-store emulators (Azurite / LocalStack / fake-gcs-server) via
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
# Requires: docker, cargo, aws CLI (S3 bucket), az CLI (Azurite container),
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
# S3 emulator: LocalStack, pinned to the multi-architecture manifest digest of
# 4.0.3 (amd64 + arm64, verified against the registry).
#
# This is the THIRD source for the S3 emulator, and the history is the reason it
# is written down rather than rediscovered each time (TD-CI-6):
#   1. docker.io/minio/minio  -- stopped serving pulls after the OSS project was
#      archived.
#   2. quay.io/minio/minio    -- the last official AGPL image, pinned by digest.
#      Quay then closed anonymous access too: an unauthenticated manifest fetch
#      of that digest returns 401 UNAUTHORIZED, so `docker run` failed with
#      "unauthorized: access to the requested resource is not authorized" and
#      exit 125 before a single test ran. Deterministic -- re-running could not
#      fix it, and no pin of that repository works from an unauthenticated
#      runner.
#   3. LocalStack 4.0.3 -- anonymously pullable AND unlicensed. Note the version
#      pin is load-bearing, not incidental: LocalStack's current date-based
#      releases (e.g. 2026.09.1) are LICENSED and exit 55 with "License
#      activation failed" unless LOCALSTACK_AUTH_TOKEN is set. 4.0.3 is the
#      newest release verified to start with no credentials.
#
# Capabilities verified by running 4.0.3 locally rather than assumed, because
# this job's purpose is per-object ACCESS TIER behaviour (`x-amz-storage-class`,
# ADR-036) and an emulator that ignored the header would turn the tier
# assertions vacuously green:
#   * `--storage-class STANDARD_IA` accepted AND returned by head-object, so the
#     class actually PERSISTS -- MinIO never did this (see TECHNICAL_DEBT.adoc),
#     so the S3 tier test should stop best-effort-skipping `InvalidStorageClass`
#     and start asserting the round-trip.
#   * conditional create enforced: a second `PUT` with `If-None-Match: *` is
#     refused, which is what `write_if_absent` / `PutMode::Create` relies on.
#   * ranged reads: `Range: bytes=2-5` returns `206` with exactly those bytes --
#     the DEPTH-dominant path (ADR-033).
#
# The S3 surface is reached only through standard AWS env vars (endpoint, keys,
# region) and the `aws` CLI, so nothing outside this block is emulator-specific.
S3_EMULATOR_IMAGE="localstack/localstack@sha256:17c2f79ca4e1f804eb912291a19713d4134806325ef0d21d4c1053161dfa72d0"
# LocalStack serves every service on one edge port.
S3_EMULATOR_PORT=4566
# LocalStack accepts any credentials; these are the conventional placeholders.
S3_ACCESS_KEY=test
S3_SECRET_KEY=test
# Azurite and fake-gcs are pinned by digest for the same reason as the S3
# emulator, and this is not cosmetic symmetry: the Azure resident-tier read-back
# below is now a HARD failure on a required check, so it depends on an upstream
# image's `properties.blobTier` shape. Leaving that image floating on `:latest`
# would reproduce precisely the failure TD-CI-6 is about — an upstream image
# moves and a required check dies in a PR with no storage changes.
# Both verified multi-arch (amd64 + arm64) against their registries.
AZURITE_IMAGE="mcr.microsoft.com/azure-storage/azurite@sha256:830430c1da1a2d537e08f3e6764dd1f5ae00cf0346bcaf625b968ec3f0971fd5"
FAKE_GCS_IMAGE="fsouza/fake-gcs-server@sha256:797ce226d62f947c009dc40246b30cfb456b8473d8241407f9d6f2c04e4d69ef"

cleanup() { docker rm -f azurite s3-emulator fake-gcs >/dev/null 2>&1 || true; }
trap cleanup EXIT

# $3 = attempts (2s apart), default 30. On timeout, dump the container's own log:
# without it a startup failure reports only "did not come up", which is the
# emulator equivalent of a silent failure — the first LocalStack attempt timed
# out and the log was the only way to tell "still booting" from "crashed".
diagnose() { # $1=container
  echo "::group::docker logs $1"
  docker logs --tail 80 "$1" 2>&1 || echo "  (no container '$1')"
  docker inspect -f 'state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$1" 2>/dev/null || true
  echo "::endgroup::"
}

wait_port() { # $1=port $2=name $3=attempts (default 30)
  for _ in $(seq 1 "${3:-30}"); do
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- ; echo "  $2 (:$1) up"; return 0; }
    sleep 2
  done
  echo "::error::$2 (:$1) did not come up"; diagnose "$2"; return 1
}

s3_aws() {
  AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" \
    AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
    AWS_DEFAULT_REGION=us-east-1 \
    aws --endpoint-url "http://127.0.0.1:$S3_EMULATOR_PORT" "$@"
}

create_s3_bucket() {
  for _ in $(seq 1 15); do
    if s3_aws s3api head-bucket --bucket "$CONTAINER_BUCKET" >/dev/null 2>&1; then
      echo "  s3 bucket already exists"
      return 0
    fi
    if s3_aws s3api create-bucket --bucket "$CONTAINER_BUCKET" >/dev/null 2>&1; then
      echo "  s3 bucket created"
      return 0
    fi
    sleep 2
  done
  echo "::error::failed to create or verify S3 bucket '$CONTAINER_BUCKET'"
  return 1
}

echo "==> Starting emulators (Docker)"
cleanup
docker run -d --name azurite -p 127.0.0.1:10000:10000 \
  "$AZURITE_IMAGE" \
  azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck >/dev/null
# SERVICES=s3 is an ALLOW-LIST, not a startup-cost tweak: a service left out is
# REFUSED, not lazily started. Verified — with this set, `sts get-caller-identity`
# fails with "Service 'sts' is not enabled. Please check your 'SERVICES'
# configuration variable." Nothing here needs STS (the credentials are static),
# but a future need for STS/KMS will surface as an opaque failure rather than a
# slow start, so add it here rather than debugging the symptom.
#
# Deliberately NOT EAGER_SERVICE_LOADING: it defers the edge port opening, and
# readiness is established by `wait_s3` below — an actual S3 call.
docker run -d --name s3-emulator -p "127.0.0.1:$S3_EMULATOR_PORT:$S3_EMULATOR_PORT" \
  -e SERVICES=s3 \
  -e AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" -e AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  "$S3_EMULATOR_IMAGE" >/dev/null
docker run -d --name fake-gcs -p 127.0.0.1:4443:4443 \
  "$FAKE_GCS_IMAGE" -scheme http -port 4443 -public-host localhost:4443 >/dev/null

echo "==> Waiting for emulators"
wait_port 10000 azurite
wait_port "$S3_EMULATOR_PORT" s3-emulator 60
wait_port 4443 fake-gcs
# Readiness is an ACTUAL S3 CALL, not `/_localstack/health`. Measured locally:
# the health endpoint can return 200 while S3 still answers `NoSuchBucket` for a
# bucket that was just created, and in one run S3 was already serving before the
# health endpoint ever answered. Gating on it produced exactly the flake this
# check exists to prevent, so the gate is the operation the suite depends on.
wait_s3() {
  for _ in $(seq 1 60); do
    if s3_aws s3api list-buckets >/dev/null 2>&1; then echo "  s3-emulator serving"; return 0; fi
    sleep 2
  done
  echo "::error::s3-emulator never answered an S3 request"; diagnose s3-emulator; return 1
}
wait_s3

echo "==> Creating bucket/container '$CONTAINER_BUCKET'"
# S3 bucket (aws CLI)
create_s3_bucket
# Azurite container (az CLI)
az storage container create --name "$CONTAINER_BUCKET" --connection-string "$AZURITE_CONN" >/dev/null 2>&1 \
  || echo "  (azurite container exists / az CLI missing)"
# fake-gcs bucket (JSON API, no auth)
curl -sf -X POST "http://127.0.0.1:4443/storage/v1/b?project=proximadb" \
  -H "Content-Type: application/json" -d "{\"name\":\"$CONTAINER_BUCKET\"}" >/dev/null 2>&1 \
  || echo "  (fake-gcs bucket exists)"

echo "==> Running object-store tier integration tests (scope: $SCOPE)"
# Azure (Azurite) + S3 (LocalStack) through the production from_url + env path; GCS via builder.
# Both emulator env vars are hoisted here (global) so EVERY Azure-touching test in
# every scope honors Azurite — including the --qa backend-contract test (#1129),
# which runs BEFORE the --qa restart section's own PROXIMADB_AZURE_EMULATOR export.
# The engine's azure_config_from_env reads PROXIMADB_AZURE_EMULATOR (not the
# object_store-convention AZURE_STORAGE_USE_EMULATOR); without it, the contract
# test built real Azure → OIDC "Identity not found" (the qa-gate failure).
export PROXIMADB_AZURE_EMULATOR=1 AZURE_STORAGE_USE_EMULATOR=true AZURE_ALLOW_HTTP=true
export AWS_ENDPOINT="http://127.0.0.1:$S3_EMULATOR_PORT" AWS_ALLOW_HTTP=true AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false
export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" AWS_REGION=us-east-1
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
  # restart proofs per STRICT store — Azure (Azurite) first, then S3 (LocalStack) with
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
  # this is a near-free incremental build + run. Emulators (Azurite/LocalStack/gcs)
  # are started by the setup trap before this scope runs.
  echo "==> TD-OBJSTORE-5 S2: backend contract tests (Azure/S3 strict, GCS best-effort)"
  CARGO_BUILD_JOBS=1 cargo test -p proximadb --features cloud-full \
    --test objstore_backend_contract_test \
    -- --ignored --nocapture --test-threads=1

  # TD-MLOPS-4 S1-remainder: the MLflow artifact seam battery against the
  # REAL S3 protocol (the lane's LocalStack). cloud-full is already built; this
  # is a near-free incremental lib-test run. AWS_ENDPOINT/AWS_* are the
  # global exports pointing at the lane's S3 emulator.
  echo "==> TD-MLOPS-4 S1: MLflow S3 artifact seam conformance (LocalStack)"
  PROXIMADB_MLFLOW_ARTIFACTS_TEST_URL="s3://$CONTAINER_BUCKET/mlflow-seam-conformance" \
    CARGO_BUILD_JOBS=1 cargo test -p proximadb --features cloud-full --lib \
    services::mlflow_artifact_s3::tests::s3_backend_passes_seam_conformance \
    -- --exact --nocapture

  echo "==> QA tier [1/2]: Azure (Azurite, adls://) — restart proofs + recall ratchet"
  export PROXIMADB_AZURE_EMULATOR=1 AZURE_STORAGE_USE_EMULATOR=true AZURE_ALLOW_HTTP=true
  export AZURE_STORAGE_ACCOUNT=devstoreaccount1 AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1
  PROXIMADB_OBJECT_STORE_URL="adls://$CONTAINER_BUCKET/qa-restart-azure" \
    CARGO_BUILD_JOBS=1 cargo test -p proximadb-server --features cloud-full \
    --test object_store_restart_recovery \
    -- --ignored --nocapture --test-threads=1

  echo "==> QA tier: stopping Azurite (S3 must not be able to reach Azure)"
  docker rm -f azurite >/dev/null 2>&1 || true

  echo "==> QA tier [2/2]: S3 (LocalStack, s3://) — restart proofs"
  export AWS_ENDPOINT="http://127.0.0.1:$S3_EMULATOR_PORT" AWS_ALLOW_HTTP=true AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false
  export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY" AWS_REGION=us-east-1
  # Both restart proofs; the recall ratchet runs on ONE strict backend (Azure),
  # so skip it here with --skip. (An older comment here claimed cargo takes
  # only ONE positional filter; it takes several — the tier gate below passes
  # three. --skip is used because it is an EXCLUSION, which a positional
  # filter cannot express.)
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
  put_with_tier_accepted_by_s3_emulator \
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

# ── Resident-tier read-back (the only thing that proves the tier was APPLIED) ──
#
# `object_store` 0.13 cannot surface the tier on read, so the Rust tests can only
# assert that a tiered PUT was ACCEPTED and round-trips — which a backend that
# silently swallows `x-amz-storage-class` also satisfies. These out-of-band reads
# are what close that gap.
#
# THE KEY IS RESOLVED BY LISTING, not spelled out, and that is load-bearing. The
# tests open their store with `from_url("s3://bucket/cold/probe-s3.bin")` and then
# write `Path::from("cold/probe-s3.bin")` into it — and object_store treats the
# URL's path as a PREFIX, so the object actually lands at
# `cold/probe-s3.bin/cold/probe-s3.bin`. The previous read-back queried the
# undoubled name, found nothing, and warned — so the "strong read-back" this job
# advertises had been VACUOUS since it was written. On the only arm that existed:
# Azure. The S3 arm is new, and it failed immediately for the same reason, which
# is how the Azure one came to light. Nobody noticed because it only warned.
# See TD-CI-7.
PROBE_SUFFIX="cold/probe-s3.bin"
echo "==> Strong read-back: confirm the S3 emulator persisted the Cool tier"
S3_KEY="$(s3_aws s3api list-objects-v2 --bucket "$CONTAINER_BUCKET" \
  --query "Contents[?ends_with(Key, '$PROBE_SUFFIX')].Key | [0]" --output text 2>/dev/null || echo "")"
if [ -z "$S3_KEY" ] || [ "$S3_KEY" = "None" ]; then
  echo "::error::no object ending in '$PROBE_SUFFIX' in bucket '$CONTAINER_BUCKET' — the tier \
test did not write its probe, so the tier cannot be verified"
  exit 1
fi
S3_CLASS="$(s3_aws s3api head-object --bucket "$CONTAINER_BUCKET" --key "$S3_KEY" \
  --query StorageClass --output text 2>/dev/null || echo "")"
echo "  s3 StorageClass($S3_KEY) = '${S3_CLASS:-<unset>}'"
if [ "$S3_CLASS" = "STANDARD_IA" ]; then
  echo "  S3 Cool tier persisted (STANDARD_IA)"
else
  # HARD failure: this is the only check on the S3 side that distinguishes "the
  # tier was applied" from "the header was swallowed" (ADR-036).
  echo "::error::the S3 emulator did not persist STANDARD_IA (got '${S3_CLASS:-<unset>}') — \
a tiered PUT that is accepted but not applied makes this job vacuous"
  exit 1
fi

echo "==> Strong read-back: confirm Azurite persisted the Cool tier"
AZ_BLOB="$(az storage blob list --container-name "$CONTAINER_BUCKET" \
  --connection-string "$AZURITE_CONN" --query "[?ends_with(name, 'cold/probe-azure.bin')].name | [0]" \
  -o tsv 2>/dev/null || echo "")"
TIER="$(az storage blob show --container-name "$CONTAINER_BUCKET" --name "${AZ_BLOB:-cold/probe-azure.bin}" \
  --connection-string "$AZURITE_CONN" --query properties.blobTier -o tsv 2>/dev/null || echo "")"
echo "  Azurite blobTier(${AZ_BLOB:-<not found>}) = '${TIER:-<unknown>}'"
if [ "$TIER" = "Cool" ]; then
  echo "  Cool tier persisted on a real Azure API"
elif ! command -v az >/dev/null 2>&1; then
  # A genuine skip, distinguished from a failure. This is the ONLY reason this
  # arm may stay quiet: `az` is not installed everywhere the script runs locally.
  echo "::warning::az CLI not installed — Azure resident tier not verified (install az, or \
run this in CI where it is preinstalled)"
else
  # HARD failure now. This arm was warn-only, and because it was ALSO querying a
  # key that never existed (TD-CI-7), it reported '<unknown>' on every run since
  # it was written — a check that could not fail, looking for a subject it could
  # not find. With the key resolved it genuinely reports `Cool`, so there is no
  # longer any reason for it to be unable to fail.
  echo "::error::Azurite did not report Cool tier (got '${TIER:-<unknown>}') for \
'${AZ_BLOB:-<not found>}' — the header was accepted but the resident tier was not applied"
  exit 1
fi

echo "==> Cloud emulator tier validation complete"
