// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Durable per-tenant `_metering` writer (TD-161).
//!
//! ADR-027 decided the differentiator: per-tenant billing meters are not just
//! scraped from Prometheus (pull-only, ephemeral) but **persisted durably to the
//! tenant's own object-store subtree** under [`DrTenantSystemPath::metering_subprefix`]
//! (`data/{tenant}/_metering/…`), where the control plane (AnvaiOps) reads them
//! and applies its rate card. This module is the OSS-native half of that sink:
//! it serializes the per-tenant resident-storage (KSU) snapshot the
//! storage-snapshot daemon already computes and writes it through the
//! [`FilesystemFactory`] — the *same* object-store abstraction the data path uses,
//! so it lands beside the tenant's data and inherits its isolation and DR rules.
//!
//! **Cadence (co-design).** KSU is an *accrual* (`resident_bytes · time`), so the
//! durable record is a coarse, **coalesced hourly snapshot**, not a per-event
//! write — matching how real clouds meter storage (AWS byte-hours, Snowflake
//! daily averages). Metering the bill must not itself become a material line on
//! the bill: object-store round-trips/egress are *the* dominant cost term this
//! co-design effort exists to minimize (CLAUDE.md co-design mandate), so an hourly
//! cadence keeps this at ~24 PUTs/tenant/day rather than the ~1440 a per-minute
//! poll would cost. The in-memory Prometheus *level gauge* refreshes more often;
//! the *durable* write is decoupled and coarse. The exact-integral flush/compaction
//! event hook is a tracked follow-up.
//!
//! **Isolation (structural, fail-closed).** The write path is resolved through
//! [`DrPathBuilder::build_tenant_system`], which validates the tenant id (rejecting
//! empty / traversal / reserved segments) before it is ever interpolated into an
//! object key. A tenant whose id fails validation is skipped and logged — never
//! written to an unvalidated path. The proto `Collection` carries only
//! `config.owner` (no account id), so the legacy `data/{tenant}/_metering/` render
//! is used; an account-rooted layout falls out automatically once account context
//! reaches the snapshot.

use std::sync::Arc;

use crate::metrics::consumption_metrics::TenantStorageUsage;
use crate::storage::persistence::filesystem::{FileOptions, FilesystemFactory};
use crate::storage::trait_components::path_resolver::DrPathBuilder;

/// On-disk schema version for a metering record. Bump on any incompatible change
/// to [`MeteringRecord`]; readers (AnvaiOps) branch on it (mixed-read-safe).
pub const METERING_RECORD_SCHEMA_VERSION: u32 = 1;

/// One durable per-tenant metering record — the serialized shape written under
/// `…/_metering/storage/`. Kept deliberately flat and self-describing (schema
/// version + kind + unit) so the consuming control plane needs no out-of-band
/// schema. JSON, not bincode: the metering sink is a cross-process contract read
/// by AnvaiOps, so human-readable + language-neutral beats compactness here (the
/// records are tiny and written hourly).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MeteringRecord {
    /// Record schema version ([`METERING_RECORD_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Meter discriminant. `"ksu_storage_snapshot"` for the resident-bytes accrual.
    pub kind: String,
    /// Owning tenant (validated path segment).
    pub tenant_id: String,
    /// Resident bytes at snapshot time (the KSU level being integrated downstream).
    pub resident_bytes: u64,
    /// Unit of `resident_bytes` — always `"bytes"` for this record kind.
    pub unit: String,
    /// Snapshot wall-clock time (unix seconds). The control plane integrates the
    /// level across successive snapshots to byte-seconds → GB-hours.
    pub snapshot_unix_secs: i64,
}

impl MeteringRecord {
    /// Build a KSU storage-snapshot record from one tenant's aggregated usage.
    pub fn ksu_storage(usage: &TenantStorageUsage, snapshot_unix_secs: i64) -> Self {
        Self {
            schema_version: METERING_RECORD_SCHEMA_VERSION,
            kind: "ksu_storage_snapshot".to_string(),
            tenant_id: usage.tenant_id.clone(),
            resident_bytes: usage.resident_bytes,
            unit: "bytes".to_string(),
            snapshot_unix_secs,
        }
    }
}

/// Record kind discriminant for TD-TIER-1 SC3 reservation usage.
pub const RESERVATION_USAGE_KIND: &str = "tier1_reservation_usage";

/// One durable reservation-usage interval (TD-TIER-1 SC3) — the additive
/// second meter kind beside the KSU snapshot.
///
/// The record carries the reserved-byte *level* held across
/// `[interval_start, interval_end]`; the control plane integrates successive
/// intervals into byte-seconds, exactly as it does for KSU snapshots. Dedup
/// identity is `(tenant_id, pool_id, producer_epoch, sequence)` — it is
/// embedded in the object key, so a replayed emission (crash between the
/// authority's ledger checkpoint and this sink) overwrites the same object
/// rather than double-charging. A `producer_epoch` jump between consecutive
/// records for a grant marks an unobserved crash gap: do not fabricate usage
/// across it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReservationUsageRecord {
    pub schema_version: u32,
    pub kind: String,
    pub tenant_id: String,
    /// Owning pool identity (stable operator-declared id).
    pub pool_id: String,
    /// The grant the interval accrues for (audit linkage).
    pub grant_id: String,
    /// Reserved bytes held throughout the interval (the level, not the
    /// integral).
    pub reserved_bytes: u64,
    pub unit: String,
    pub interval_start_unix_secs: i64,
    pub interval_end_unix_secs: i64,
    /// Bumped on every authority reopen; dedup key + crash-gap marker.
    pub producer_epoch: u64,
    /// Monotonic within an epoch; dedup key.
    pub sequence: u64,
}

impl ReservationUsageRecord {
    /// Build a record from one authority-emitted interval.
    pub fn from_interval(
        interval: &crate::services::collection::reservation::ReservationUsageInterval,
    ) -> Self {
        Self {
            schema_version: METERING_RECORD_SCHEMA_VERSION,
            kind: RESERVATION_USAGE_KIND.to_string(),
            tenant_id: interval.tenant_id.clone(),
            pool_id: interval.pool_id.clone(),
            grant_id: interval.grant_id.clone(),
            reserved_bytes: interval.reserved_bytes,
            unit: "bytes".to_string(),
            interval_start_unix_secs: interval.interval_start_unix_secs,
            interval_end_unix_secs: interval.interval_end_unix_secs,
            producer_epoch: interval.producer_epoch,
            sequence: interval.sequence,
        }
    }
}

/// Resolve the durable object URL for one reservation-usage record. The dedup
/// identity `(epoch, sequence)` lives in the key, so replay overwrites:
/// `…/_metering/reservations/{pool}/usage-{epoch}-{sequence}.json`.
pub fn reservation_usage_object_url(
    base_url: &str,
    account_id: Option<&str>,
    tenant_id: &str,
    pool_id: &str,
    producer_epoch: u64,
    sequence: u64,
) -> anyhow::Result<String> {
    let resolved = DrPathBuilder::build_tenant_system(account_id, tenant_id)
        .map_err(|e| anyhow::anyhow!("invalid tenant id for metering path: {e}"))?;
    let prefix = resolved.metering_subprefix();
    let base = base_url.trim_end_matches('/');
    Ok(format!(
        "{base}/{prefix}reservations/{pool_id}/usage-{producer_epoch}-{sequence}.json"
    ))
}

/// Resolve the durable object URL for one tenant's KSU storage snapshot.
///
/// `base_url` is the configured storage root (e.g. `file:///…/data` or
/// `s3://bucket/prefix`). The returned URL is
/// `{base}/data/{tenant}/_metering/storage/snapshot-{secs}.json` — the
/// `data/{tenant}/_metering/` portion comes verbatim from
/// [`DrTenantSystemPath::metering_subprefix`], so the metering sink stays under the
/// same `DrPathBuilder` isolation prefix as the tenant's data. Returns `Err` if
/// the tenant id fails path validation (fail-closed — never write to an
/// unvalidated key).
pub fn ksu_snapshot_object_url(
    base_url: &str,
    account_id: Option<&str>,
    tenant_id: &str,
    snapshot_unix_secs: i64,
) -> anyhow::Result<String> {
    let resolved = DrPathBuilder::build_tenant_system(account_id, tenant_id)
        .map_err(|e| anyhow::anyhow!("invalid tenant id for metering path: {e}"))?;
    let prefix = resolved.metering_subprefix(); // `data/{tenant}/_metering/`
    let base = base_url.trim_end_matches('/');
    Ok(format!(
        "{base}/{prefix}storage/snapshot-{snapshot_unix_secs}.json"
    ))
}

/// Writes the durable per-tenant metering snapshot through the object-store
/// abstraction. Holds an `Arc<FilesystemFactory>` (URL-routed, mirroring
/// `src/metrics/store.rs`) and the configured storage root.
#[derive(Clone)]
pub struct DurableMeteringWriter {
    factory: Arc<FilesystemFactory>,
    base_url: String,
}

impl DurableMeteringWriter {
    /// Construct a writer rooted at `base_url` (the configured storage location).
    pub fn new(factory: Arc<FilesystemFactory>, base_url: impl Into<String>) -> Self {
        Self {
            factory,
            base_url: base_url.into(),
        }
    }

    /// Persist a KSU storage snapshot for every tenant in `usage`.
    ///
    /// Best-effort and **never fatal**: a per-tenant serialize/validate/write
    /// failure is logged and skipped so one bad tenant cannot stall metering for
    /// the rest (KSU must keep accruing). Returns the number of records
    /// successfully written. `account_id` is threaded for the account-rooted
    /// layout; pass `None` for the legacy `data/{tenant}/` render (the current
    /// snapshot source carries no account id).
    pub async fn write_storage_snapshot(
        &self,
        usage: &[TenantStorageUsage],
        account_id: Option<&str>,
        snapshot_unix_secs: i64,
    ) -> usize {
        let mut written = 0usize;
        for u in usage {
            match self.write_one(u, account_id, snapshot_unix_secs).await {
                Ok(()) => written += 1,
                Err(e) => tracing::warn!(
                    tenant = %u.tenant_id,
                    "TD-161: durable metering write skipped for tenant: {e}"
                ),
            }
        }
        if written > 0 {
            tracing::debug!(
                "TD-161: persisted {written}/{} per-tenant metering snapshots",
                usage.len()
            );
        }
        written
    }

    async fn write_one(
        &self,
        usage: &TenantStorageUsage,
        account_id: Option<&str>,
        snapshot_unix_secs: i64,
    ) -> anyhow::Result<()> {
        let url = ksu_snapshot_object_url(
            &self.base_url,
            account_id,
            &usage.tenant_id,
            snapshot_unix_secs,
        )?;
        let record = MeteringRecord::ksu_storage(usage, snapshot_unix_secs);
        let bytes = serde_json::to_vec(&record)
            .map_err(|e| anyhow::anyhow!("serialize metering record: {e}"))?;
        let options = FileOptions {
            overwrite: true,
            create_dirs: true,
            ..Default::default()
        };
        self.factory
            .write(&url, &bytes, Some(options))
            .await
            .map_err(|e| anyhow::anyhow!("write metering record to {url}: {e}"))?;
        Ok(())
    }

    /// Persist one reservation-usage interval (TD-TIER-1 SC3) under the same
    /// per-tenant `_metering/` isolation subtree, keyed for idempotent replay
    /// (dedup identity `(epoch, sequence)` in the object key). Unlike the KSU
    /// snapshot sweep this surfaces per-record errors: reservation billing
    /// correctness must not silently drop intervals.
    pub async fn write_reservation_usage(
        &self,
        record: &ReservationUsageRecord,
        account_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let url = reservation_usage_object_url(
            &self.base_url,
            account_id,
            &record.tenant_id,
            &record.pool_id,
            record.producer_epoch,
            record.sequence,
        )?;
        let bytes = serde_json::to_vec(record)
            .map_err(|e| anyhow::anyhow!("serialize reservation usage record: {e}"))?;
        let options = FileOptions {
            overwrite: true,
            create_dirs: true,
            ..Default::default()
        };
        self.factory
            .write(&url, &bytes, Some(options))
            .await
            .map_err(|e| anyhow::anyhow!("write reservation usage to {url}: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ksu_url_uses_legacy_tenant_metering_prefix() {
        let url = ksu_snapshot_object_url("file:///srv/data", None, "tenant_acme", 1_700_000_000)
            .unwrap();
        assert_eq!(
            url,
            "file:///srv/data/data/tenant_acme/_metering/storage/snapshot-1700000000.json"
        );
    }

    #[test]
    fn ksu_url_account_rooted_when_account_present() {
        let url = ksu_snapshot_object_url(
            "s3://bucket/prefix",
            Some("acct1"),
            "tenant_acme",
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(
            url,
            "s3://bucket/prefix/accounts/acct1/tenant_acme/_metering/storage/snapshot-1700000000.json"
        );
    }

    #[test]
    fn ksu_url_trims_trailing_slash_on_base() {
        let url = ksu_snapshot_object_url("file:///srv/data/", None, "t1", 42).unwrap();
        assert_eq!(
            url,
            "file:///srv/data/data/t1/_metering/storage/snapshot-42.json"
        );
    }

    #[test]
    fn ksu_url_rejects_traversal_tenant_id_fail_closed() {
        // A tenant id that escapes its subtree must never resolve to a path.
        assert!(ksu_snapshot_object_url("file:///srv/data", None, "../escape", 1).is_err());
        assert!(ksu_snapshot_object_url("file:///srv/data", None, "", 1).is_err());
    }

    #[test]
    fn record_is_self_describing_and_versioned() {
        let usage = TenantStorageUsage {
            tenant_id: "t1".to_string(),
            resident_bytes: 4096,
        };
        let rec = MeteringRecord::ksu_storage(&usage, 1_700_000_000);
        assert_eq!(rec.schema_version, METERING_RECORD_SCHEMA_VERSION);
        assert_eq!(rec.kind, "ksu_storage_snapshot");
        assert_eq!(rec.unit, "bytes");
        assert_eq!(rec.resident_bytes, 4096);
        // Round-trips through JSON (the cross-process contract with AnvaiOps).
        let json = serde_json::to_vec(&rec).unwrap();
        let back: MeteringRecord = serde_json::from_slice(&json).unwrap();
        assert_eq!(rec, back);
    }

    /// Runtime-verify: the writer actually persists per-tenant records through the
    /// real `FilesystemFactory` and they read back intact under the
    /// `data/{tenant}/_metering/storage/` key — the durable-sink contract, not just
    /// path arithmetic.
    #[tokio::test]
    async fn durable_writer_round_trips_through_local_filesystem() {
        let dir = tempfile::tempdir().unwrap();
        let base = format!("file://{}", dir.path().display());
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let writer = DurableMeteringWriter::new(factory.clone(), base.clone());

        let usage = vec![
            TenantStorageUsage {
                tenant_id: "tenant_a".to_string(),
                resident_bytes: 1234,
            },
            TenantStorageUsage {
                tenant_id: "tenant_b".to_string(),
                resident_bytes: 5678,
            },
        ];
        let ts = 1_700_000_000;
        let written = writer.write_storage_snapshot(&usage, None, ts).await;
        assert_eq!(written, 2, "both tenants persisted");

        // Read tenant_a's record back via the same factory and assert content.
        let url = ksu_snapshot_object_url(&base, None, "tenant_a", ts).unwrap();
        let bytes = factory.read(&url).await.unwrap();
        let rec: MeteringRecord = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(rec.tenant_id, "tenant_a");
        assert_eq!(rec.resident_bytes, 1234);
        assert_eq!(rec.kind, "ksu_storage_snapshot");
        assert_eq!(rec.snapshot_unix_secs, ts);

        // The key lands under the per-tenant `_metering/` isolation subtree.
        assert!(url.contains("/data/tenant_a/_metering/storage/"));
    }

    // ---- TD-TIER-1 SC3: reservation usage records ----

    fn interval(
        epoch: u64,
        sequence: u64,
    ) -> crate::services::collection::reservation::ReservationUsageInterval {
        crate::services::collection::reservation::ReservationUsageInterval {
            tenant_id: "tenant_a".to_string(),
            pool_id: "nvme-1".to_string(),
            grant_id: "g1".to_string(),
            reserved_bytes: 4096,
            interval_start_unix_secs: 1000,
            interval_end_unix_secs: 2000,
            producer_epoch: epoch,
            sequence,
        }
    }

    #[test]
    fn reservation_record_round_trips_with_dedup_identity() {
        let rec = ReservationUsageRecord::from_interval(&interval(3, 7));
        assert_eq!(rec.kind, RESERVATION_USAGE_KIND);
        assert_eq!(rec.schema_version, METERING_RECORD_SCHEMA_VERSION);
        let json = serde_json::to_vec(&rec).unwrap();
        let back: ReservationUsageRecord = serde_json::from_slice(&json).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn reservation_url_embeds_dedup_identity() {
        let url =
            reservation_usage_object_url("file:///srv/data", None, "t1", "nvme-1", 3, 7).unwrap();
        assert_eq!(
            url,
            "file:///srv/data/data/t1/_metering/reservations/nvme-1/usage-3-7.json"
        );
        // Fail-closed on a traversal tenant id, like the KSU URL.
        assert!(reservation_usage_object_url("file:///srv/data", None, "../x", "p", 1, 1).is_err());
    }

    /// A replayed emission (same epoch + sequence) overwrites the SAME object
    /// key — the overwrite-not-double-charge contract.
    #[tokio::test]
    async fn replayed_reservation_usage_overwrites_same_key() {
        let dir = tempfile::tempdir().unwrap();
        let base = format!("file://{}", dir.path().display());
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let writer = DurableMeteringWriter::new(factory.clone(), base.clone());

        let rec = ReservationUsageRecord::from_interval(&interval(2, 5));
        writer.write_reservation_usage(&rec, None).await.unwrap();
        // Replay: same identity, perhaps after a crash between ledger persist
        // and sink — must land on the same key.
        writer.write_reservation_usage(&rec, None).await.unwrap();

        let url = reservation_usage_object_url(&base, None, "tenant_a", "nvme-1", 2, 5).unwrap();
        let bytes = factory.read(&url).await.unwrap();
        let back: ReservationUsageRecord = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.reserved_bytes, 4096);
        assert_eq!(back.producer_epoch, 2);
        assert_eq!(back.sequence, 5);
    }
}
