# TD-168 — Object-store Cool-tier validation against real cloud APIs (emulators)

> **Superseded in part (2026-10-08, TD-CI-6):** the S3 emulator is no longer MinIO on port
> 9000 — it is **LocalStack on 4566**, pinned by digest, and the S3 tier test is now named
> `put_with_tier_accepted_by_s3_emulator`. MinIO's image stopped being anonymously pullable
> from both Docker Hub and Quay. Everything below about the AZURITE arm and the out-of-band
> `blobTier` read-back still holds; treat MinIO/9000/`..._by_minio` references as history.
> `scripts/run_cloud_emulator_tests.sh` is the source of truth for connection details.


**Date:** 2026-06-28 · **Scope:** TD-168 Phase 2 / TD-173 (per-object access tier)

## Why this exists

`put_with_tier` / `ColdGraphRecordStore` unit tests run on `memory://`, where the
access tier is **meaningless and degrades to a plain put** — they cannot prove the
`x-ms-access-tier` / `x-amz-storage-class` / `x-goog-storage-class` header is
*accepted* by a real cloud API. The failure they miss: a native-class mapping
regression (an invalid storage-class string) that a real API would reject with a
4xx. These integration tests run the tier path against **emulators** for all three
clouds — the develop→qa "upfront validation" for the object-store cost lever.

## What runs where

| Cloud | Emulator | Path exercised | Tier behavior | Strictness |
|-------|----------|----------------|---------------|------------|
| Azure | Azurite | **production** `from_url("az://…")` + env | accepts **and persists** `Cool` | strict |
| AWS S3 | MinIO | **production** `from_url("s3://…")` + env | accepts header, ignores value (S3-compat) | strict (accept + round-trip) |
| GCP | fake-gcs-server | builder `with_base_url` (no emulator env key) | header sent; may not honor | best-effort (test self-skips on incompatibility) |

Tests (all `#[cfg(feature = …)] #[ignore]`, run with `-- --ignored`):
- `put_with_tier_accepted_by_azurite` · `put_with_tier_accepted_by_minio` ·
  `put_with_tier_against_fake_gcs` (crate `proximadb-object-store`)
- `cold_graph_record_store_round_trips_on_real_azure` (crate `proximadb`) — the
  feature end-to-end through the production `ColdGraphRecordStore::from_storage_root`.

## Run it

One command (Docker + `aws`/`az` CLIs + `curl` required) — the **same** path CI runs:

```bash
make cloud-emulator-test
#   ├─ docker run Azurite (:10000), MinIO (:9000), fake-gcs (:4443)
#   ├─ create bucket/container `proximadb-test`
#   ├─ cargo test --features aws,azure,gcp -- --ignored <the tier tests>
#   └─ read back Azurite blobTier → assert "Cool"
```

Orchestration lives in `scripts/run_cloud_emulator_tests.sh` (single source of
truth). CI invokes the same script from the `cloud-emulator-object-store` job in
`.github/workflows/qa-gate.yml` (the develop→qa gate; also `workflow_dispatch`).

## The tier read-back

`object_store` 0.13.2 sends the tier on PUT (`azure/client.rs` →
`x-ms-access-tier`) but **does not surface it on read** —
`GetResult`/`ObjectMeta` attributes omit it (`client::get::get_attributes` parses
only CacheControl/Content-*/user-metadata). So the Rust tests assert *acceptance +
round-trip*. For the **strong resident-tier proof** the script/CI reads it back
out-of-band on Azurite (which persists it):

```bash
az storage blob show --container-name proximadb-test --name cold/probe-azure.bin \
  --connection-string "…BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;" \
  --query properties.blobTier -o tsv   # expected: Cool
```

(MinIO/fake-gcs don't persist storage class, so there's no equivalent read-back
there — those backends prove header *acceptance* only.) A strong in-Rust tier
read-back would need the raw Azure SDK (`get_properties().access_tier`) — deferred
(TD-168 residual 6).

## Notes / limitations

- The `qa-gate.yml` job runs on PRs to `qa`/`main` and `workflow_dispatch` — **not**
  on develop-targeting PRs — so its first execution is the develop→qa promotion or a
  manual dispatch. It is not yet a *required* status check (add to branch protection
  on `qa` to gate promotion).
- GCS is the fragile leg: `object_store`'s GCS builder has no clean emulator/anonymous
  mode, and fake-gcs may not honor storage class. The test skips (not fails) on
  incompatibility, so GCS never blocks the gate.
