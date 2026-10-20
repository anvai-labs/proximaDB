//! Canonical-WAL-backed durable storage for rank profiles.
//!
//! Spec reference: `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md` §R-7c production wiring.
//!
//! Rank profiles are catalog-level metadata that must survive restarts. This
//! store encodes each install/remove as a `CanonicalOperation::RecordUpsert` /
//! `RecordDelete` on a synthetic `__proxima_rank_profiles__` collection and
//! replays the WAL at startup to rebuild the visible profile state. The design
//! reuses the shared `TableWalAppender` boundary instead of extending every
//! `Catalog` backend, mirroring the pattern the graph subsystem uses for its
//! canonical checkpoint records.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::RwLock;
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};

use crate::services::record_store::TableWalAppender;

/// Synthetic collection id used to namespace rank-profile entries inside the
/// shared canonical WAL. Chosen to be unambiguously non-user-collection-shaped
/// so scans never collide with caller-defined collection names.
pub const RANK_PROFILES_COLLECTION_ID: &str = "__proxima_rank_profiles__";

/// Snapshot of a rank profile as it lives in catalog storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRankProfile {
    /// Profile name (catalog key).
    pub name: String,
    /// Monotonically increasing version; bumps on every install for the same name.
    pub version: u32,
    /// Owning tenant. `None` = single-tenant / unscoped.
    pub tenant: Option<String>,
    /// Raw TOML body of the profile, parsed at install time and re-parsed by
    /// the registry hot-reload path.
    pub spec_toml: String,
    /// Wall-clock time the profile was installed (ms since Unix epoch).
    pub created_at_ms: i64,
}

/// Durable rank-profile catalog API.
///
/// Implementations persist installs/removes through the canonical WAL spine so
/// the registry survives restarts. Reads are served from an in-memory snapshot
/// rebuilt at construction.
#[async_trait]
pub trait RankProfileStore: Send + Sync {
    /// Install or update a profile. If `explicit_version` is `None`, the store
    /// assigns the next monotonic version for the given name.
    async fn install(
        &self,
        name: &str,
        spec_toml: String,
        tenant: Option<String>,
        explicit_version: Option<u32>,
    ) -> Result<StoredRankProfile>;

    /// Fetch the currently visible profile for `name`, if any.
    async fn get(&self, name: &str) -> Result<Option<StoredRankProfile>>;

    /// List profiles scoped to a specific tenant (excludes profiles with no
    /// tenant or with a different tenant).
    async fn list_for_tenant(&self, tenant: &str) -> Result<Vec<StoredRankProfile>>;

    /// List every visible profile across tenants. Used by server-startup
    /// recovery to populate the in-process `ProfileRegistry`.
    async fn list_all(&self) -> Result<Vec<StoredRankProfile>>;

    /// Remove a profile by name. Returns `true` if a profile was present.
    async fn remove(&self, name: &str) -> Result<bool>;

    /// Re-append every live profile to the head of the canonical WAL and return
    /// the sequence of the first re-appended entry: this store no longer needs
    /// any WAL entry below it (TD-SEAM-12). `None` when no profile is live, so
    /// nothing constrains truncation on this store's account.
    async fn checkpoint(&self) -> Result<Option<u64>>;
}

/// Canonical-WAL-backed `RankProfileStore`.
///
/// State is an in-memory `HashMap<name, StoredRankProfile>` populated by
/// replaying entries supplied at construction. Subsequent installs/removes
/// append a single canonical operation to the WAL and update the in-memory
/// map; the writer side is last-write-wins keyed by profile name.
pub struct CanonicalWalRankProfileStore {
    appender: Arc<dyn TableWalAppender>,
    state: RwLock<HashMap<String, StoredRankProfile>>,
    /// Serializes installs, removes and [`RankProfileStore::checkpoint`].
    /// Replay is last-write-wins by name, so a checkpoint snapshot re-appended
    /// AFTER a concurrent install would replay the older value over the newer
    /// one; holding this across "read state → append" rules that out.
    write_gate: tokio::sync::Mutex<()>,
}

impl CanonicalWalRankProfileStore {
    /// Build an empty store over a fresh appender. Use this for tests or when
    /// the canonical WAL is known to contain no prior rank-profile entries.
    pub fn new(appender: Arc<dyn TableWalAppender>) -> Self {
        Self {
            appender,
            state: RwLock::new(HashMap::new()),
            write_gate: tokio::sync::Mutex::new(()),
        }
    }

    /// Build a store and replay an existing canonical WAL slice into it. The
    /// caller is expected to read entries from the WAL file (e.g. via
    /// `FramedTableWalAppender::read_entries_from_path`) before instantiating
    /// the store, so the appender backing this store represents the same WAL.
    pub fn from_wal_entries(
        appender: Arc<dyn TableWalAppender>,
        entries: &[CanonicalWalEntry],
    ) -> Self {
        let store = Self::new(appender);
        store.replay_entries(entries);
        store
    }

    /// Apply a slice of canonical WAL entries to the in-memory state. Entries
    /// whose collection_id is not `RANK_PROFILES_COLLECTION_ID` are skipped.
    fn replay_entries(&self, entries: &[CanonicalWalEntry]) {
        let mut state = self.state.write();
        for entry in entries {
            match &entry.operation {
                CanonicalOperation::RecordUpsert {
                    collection_id,
                    record,
                    ..
                } if collection_id == RANK_PROFILES_COLLECTION_ID => {
                    if let Some(profile) = record_to_profile(record.as_ref()) {
                        state.insert(profile.name.clone(), profile);
                    }
                }
                CanonicalOperation::RecordDelete {
                    collection_id, oid, ..
                } if collection_id == RANK_PROFILES_COLLECTION_ID => {
                    state.remove(oid);
                }
                _ => {}
            }
        }
    }
}

fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

fn profile_to_record(profile: &StoredRankProfile) -> ProximaRecord {
    let mut props: HashMap<String, ProximaTreeNode> = HashMap::new();
    props.insert(
        "spec_toml".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(profile.spec_toml.clone())),
    );
    props.insert(
        "version".to_string(),
        ProximaTreeNode::Value(ProximaValue::UInt32(profile.version)),
    );
    props.insert(
        "created_at_ms".to_string(),
        ProximaTreeNode::Value(ProximaValue::Int64(profile.created_at_ms)),
    );
    ProximaRecord {
        oid: profile.name.clone(),
        variation_id: Some(RANK_PROFILES_COLLECTION_ID.to_string()),
        tenant_id: profile.tenant.clone().unwrap_or_default(),
        created_at_ns: now_ns(),
        props,
        ..Default::default()
    }
}

fn record_to_profile(record: &ProximaRecord) -> Option<StoredRankProfile> {
    let spec_toml = match record.props.get("spec_toml")? {
        ProximaTreeNode::Value(ProximaValue::String(s)) => s.clone(),
        _ => return None,
    };
    let version = match record.props.get("version")? {
        ProximaTreeNode::Value(ProximaValue::UInt32(v)) => *v,
        _ => return None,
    };
    let created_at_ms = match record.props.get("created_at_ms")? {
        ProximaTreeNode::Value(ProximaValue::Int64(v)) => *v,
        _ => return None,
    };
    Some(StoredRankProfile {
        name: record.oid.clone(),
        version,
        tenant: if record.tenant_id.is_empty() {
            None
        } else {
            Some(record.tenant_id.clone())
        },
        spec_toml,
        created_at_ms,
    })
}

#[async_trait]
impl RankProfileStore for CanonicalWalRankProfileStore {
    async fn install(
        &self,
        name: &str,
        spec_toml: String,
        tenant: Option<String>,
        explicit_version: Option<u32>,
    ) -> Result<StoredRankProfile> {
        let _gate = self.write_gate.lock().await;
        let profile = {
            let state = self.state.read();
            let version = explicit_version
                .unwrap_or_else(|| state.get(name).map(|p| p.version + 1).unwrap_or(1));
            StoredRankProfile {
                name: name.to_string(),
                version,
                tenant: tenant.clone(),
                spec_toml,
                created_at_ms: proximadb_kernel::time::now_unix_ms(),
            }
        };
        let record = profile_to_record(&profile);

        self.appender
            .append_operations(
                vec![CanonicalOperation::RecordUpsert {
                    collection_id: RANK_PROFILES_COLLECTION_ID.to_string(),
                    record: Box::new(record),
                    projections: Vec::new(),
                }],
                profile.tenant.clone(),
            )
            .await
            .context("appending rank profile install to canonical WAL")?;

        self.state
            .write()
            .insert(profile.name.clone(), profile.clone());
        Ok(profile)
    }

    async fn get(&self, name: &str) -> Result<Option<StoredRankProfile>> {
        Ok(self.state.read().get(name).cloned())
    }

    async fn list_for_tenant(&self, tenant: &str) -> Result<Vec<StoredRankProfile>> {
        Ok(self
            .state
            .read()
            .values()
            .filter(|profile| profile.tenant.as_deref() == Some(tenant))
            .cloned()
            .collect())
    }

    async fn list_all(&self) -> Result<Vec<StoredRankProfile>> {
        Ok(self.state.read().values().cloned().collect())
    }

    async fn remove(&self, name: &str) -> Result<bool> {
        let _gate = self.write_gate.lock().await;
        let prior = self.state.read().get(name).cloned();
        let Some(prior) = prior else {
            return Ok(false);
        };

        self.appender
            .append_operations(
                vec![CanonicalOperation::RecordDelete {
                    collection_id: RANK_PROFILES_COLLECTION_ID.to_string(),
                    oid: name.to_string(),
                    projections: Vec::new(),
                }],
                prior.tenant.clone(),
            )
            .await
            .context("appending rank profile delete to canonical WAL")?;

        self.state.write().remove(name);
        Ok(true)
    }

    async fn checkpoint(&self) -> Result<Option<u64>> {
        let _gate = self.write_gate.lock().await;
        let mut live: Vec<StoredRankProfile> = self.state.read().values().cloned().collect();
        // Deterministic order, so a checkpoint is reproducible in tests and logs.
        live.sort_by(|a, b| a.name.cmp(&b.name));
        let mut first: Option<u64> = None;
        for profile in &live {
            let entries = self
                .appender
                .append_operations(
                    vec![CanonicalOperation::RecordUpsert {
                        collection_id: RANK_PROFILES_COLLECTION_ID.to_string(),
                        record: Box::new(profile_to_record(profile)),
                        projections: Vec::new(),
                    }],
                    profile.tenant.clone(),
                )
                .await
                .context("re-appending a rank profile for a WAL checkpoint")?;
            if first.is_none() {
                first = entries.first().map(|e| e.sequence_number);
            }
        }
        Ok(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::canonical_wal::FramedTableWalAppender;
    use tempfile::tempdir;

    fn sample_toml(_name: &str) -> String {
        // Minimal valid TOML body (matches what `parse_single` expects: no
        // `[profile]` wrapper — top-level keys + `[first_phase]` table).
        r#"
[first_phase]
expression = "1.0"
heap_size = 100
"#
        .to_string()
    }

    #[tokio::test]
    async fn install_roundtrips_through_get() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        let installed = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        assert_eq!(installed.name, "alpha");
        assert_eq!(installed.version, 1);

        let fetched = store
            .get("alpha")
            .await
            .unwrap()
            .expect("profile should exist after install");
        assert_eq!(fetched, installed);
    }

    #[tokio::test]
    async fn install_assigns_monotonic_version_per_name() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        let v1 = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        let v2 = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        let v3 = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();

        assert_eq!(v1.version, 1);
        assert_eq!(v2.version, 2);
        assert_eq!(v3.version, 3);

        let fetched = store.get("alpha").await.unwrap().unwrap();
        assert_eq!(fetched.version, 3, "last write wins for the same name");
    }

    #[tokio::test]
    async fn restart_recovery_replays_canonical_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");

        // First boot: install two profiles and drop the store + appender.
        {
            let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
            let store = CanonicalWalRankProfileStore::new(appender);
            store
                .install("alpha", sample_toml("alpha"), Some("tenant_a".into()), None)
                .await
                .unwrap();
            store
                .install("beta", sample_toml("beta"), None, None)
                .await
                .unwrap();
        }

        // Second boot: read prior WAL entries, then build a fresh store from them.
        let entries = FramedTableWalAppender::read_entries_from_path(&wal_path)
            .await
            .unwrap();
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let recovered = CanonicalWalRankProfileStore::from_wal_entries(appender, &entries);

        let all = recovered.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
        let alpha = recovered.get("alpha").await.unwrap().unwrap();
        assert_eq!(alpha.tenant.as_deref(), Some("tenant_a"));
        let beta = recovered.get("beta").await.unwrap().unwrap();
        assert_eq!(beta.tenant, None);
    }

    #[tokio::test]
    async fn list_for_tenant_scopes_to_owning_tenant() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        store
            .install("a1", sample_toml("a1"), Some("tenant_a".into()), None)
            .await
            .unwrap();
        store
            .install("a2", sample_toml("a2"), Some("tenant_a".into()), None)
            .await
            .unwrap();
        store
            .install("b1", sample_toml("b1"), Some("tenant_b".into()), None)
            .await
            .unwrap();
        store
            .install("global", sample_toml("global"), None, None)
            .await
            .unwrap();

        let mut tenant_a = store
            .list_for_tenant("tenant_a")
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.name)
            .collect::<Vec<_>>();
        tenant_a.sort();
        assert_eq!(tenant_a, vec!["a1".to_string(), "a2".to_string()]);

        let tenant_b = store
            .list_for_tenant("tenant_b")
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.name)
            .collect::<Vec<_>>();
        assert_eq!(tenant_b, vec!["b1".to_string()]);

        let none_scope = store.list_for_tenant("tenant_c").await.unwrap();
        assert!(
            none_scope.is_empty(),
            "tenant_c shouldn't see any other tenant's profile"
        );
    }

    #[tokio::test]
    async fn remove_returns_false_for_missing_and_clears_state() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        assert!(!store.remove("ghost").await.unwrap());

        store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        assert!(store.remove("alpha").await.unwrap());
        assert!(store.get("alpha").await.unwrap().is_none());
        assert!(
            !store.remove("alpha").await.unwrap(),
            "second remove of the same name should report no change"
        );
    }

    #[tokio::test]
    async fn delete_then_install_resurrects_with_fresh_version() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        let v1 = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        assert_eq!(v1.version, 1);
        assert!(store.remove("alpha").await.unwrap());
        let resurrected = store
            .install("alpha", sample_toml("alpha"), None, None)
            .await
            .unwrap();
        assert_eq!(
            resurrected.version, 1,
            "after delete, the next install starts a fresh version sequence"
        );
    }

    #[tokio::test]
    async fn explicit_version_overrides_assigned_version() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("rank_profiles.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalRankProfileStore::new(appender);

        let installed = store
            .install("alpha", sample_toml("alpha"), None, Some(42))
            .await
            .unwrap();
        assert_eq!(installed.version, 42);
    }
}
