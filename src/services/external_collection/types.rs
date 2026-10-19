//! External Collection control-plane types (Phase 8 F5 / TD-090).
//!
//! An *external collection* is a lake table (Parquet first) that ProximaDB
//! indexes **without copying**: the source stays externally governed
//! (`CatalogAuthorityMode::FederatedRead`) while ProximaDB owns and serves a
//! vector index built over it. These are the durable registry records.

use proximadb_kernel::time::now_unix_ms;
use serde::{Deserialize, Serialize};

/// Open-table format of the external source. Parquet is the only format wired
/// in Slice 1; `Iceberg`/`Lance` are reserved for later slices (F5 Slice 2 / F6)
/// so the registry schema does not need to change when they land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExternalFormat {
    /// Apache Parquet file or directory of files.
    #[default]
    Parquet,
}

impl ExternalFormat {
    /// The catalog physical-format this maps to for the storage layout.
    pub fn catalog_format(&self) -> proximadb_catalog::CatalogPhysicalFormat {
        match self {
            ExternalFormat::Parquet => proximadb_catalog::CatalogPhysicalFormat::Parquet,
        }
    }
}

/// Lifecycle of an external collection: register (catalog only) → build (index
/// in place) → ready (served). `Failed` records the build error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExternalCollectionStatus {
    /// Registered in the catalog; no index built yet.
    #[default]
    Registered,
    /// Index build in progress.
    Building,
    /// Index built and queryable over the external snapshot.
    Ready,
    /// Build failed; see `error`.
    Failed,
}

/// Registration parameters for an external collection. The `id_column` /
/// `vector_column` name the columns in the external source that become the
/// record oid and its dense embedding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalCollectionSpec {
    /// Logical collection name (also the catalog table name + index key).
    pub name: String,
    /// Source location (filesystem path or directory of Parquet files).
    pub location: String,
    /// Source open-table format.
    pub format: ExternalFormat,
    /// Column holding the record identifier (Utf8).
    pub id_column: String,
    /// Column holding the dense vector (FixedSizeList/List of Float32).
    pub vector_column: String,
    /// Vector dimensionality (validated against the source on build).
    pub dimension: usize,
    /// Distance metric for the built index (catalog-style string, e.g. "cosine").
    pub distance_metric: String,
    /// Optional Utf8 column to build a BM25 inverted index over (F5 Slice 3).
    /// When set, search supports hybrid (vector + lexical) retrieval. `None` ⇒
    /// vector-only. `#[serde(default)]` keeps older registry records readable.
    #[serde(default)]
    pub text_column: Option<String>,
}

impl ExternalCollectionSpec {
    /// Construct a Parquet external-collection spec with cosine distance.
    pub fn parquet(
        name: impl Into<String>,
        location: impl Into<String>,
        id_column: impl Into<String>,
        vector_column: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            name: name.into(),
            location: location.into(),
            format: ExternalFormat::Parquet,
            id_column: id_column.into(),
            vector_column: vector_column.into(),
            dimension,
            distance_metric: "cosine".to_string(),
            text_column: None,
        }
    }

    /// Set the BM25 text column (builder).
    pub fn with_text_column(mut self, text_column: impl Into<String>) -> Self {
        self.text_column = Some(text_column.into());
        self
    }
}

/// A durable external-collection registry record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalCollection {
    /// Stable identifier (`ext_<uuid>`), never reused.
    pub id: String,
    /// Registration parameters.
    pub spec: ExternalCollectionSpec,
    /// Snapshot fingerprint of the source at register/build time. Recorded as
    /// the projection `source_range`; staleness detection (Slice 2) compares a
    /// freshly computed fingerprint against this.
    pub snapshot_id: String,
    /// Lifecycle state.
    pub status: ExternalCollectionStatus,
    /// Records indexed by the last successful build (0 until built).
    pub indexed_record_count: u64,
    /// Creation time (epoch ms).
    pub created_at_ms: i64,
    /// Last state-change time (epoch ms).
    pub updated_at_ms: i64,
    /// Failure detail when `status == Failed`.
    pub error: Option<String>,
    /// The tenant that registered it (`None` = the default tenant). Only that
    /// tenant sees or operates on it.
    #[serde(default)]
    pub tenant: Option<String>,
}

impl ExternalCollection {
    /// Create a freshly `Registered` external collection for `spec`.
    pub fn new(spec: ExternalCollectionSpec, snapshot_id: impl Into<String>) -> Self {
        let now = now_unix_ms();
        Self {
            id: format!("ext_{}", uuid::Uuid::new_v4().simple()),
            spec,
            snapshot_id: snapshot_id.into(),
            status: ExternalCollectionStatus::Registered,
            indexed_record_count: 0,
            created_at_ms: now,
            updated_at_ms: now,
            error: None,
            tenant: None,
        }
    }

    /// Whether a request acting as `tenant` may see this collection.
    pub fn visible_to(&self, tenant: &str) -> bool {
        proximadb_tenant::resolve_request_tenant(self.tenant.as_deref()) == tenant
    }

    /// Whether the collection has a built, queryable index.
    pub fn is_ready(&self) -> bool {
        self.status == ExternalCollectionStatus::Ready
    }
}

/// Current epoch milliseconds.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_external_collection_is_registered_with_id_prefix() {
        let spec = ExternalCollectionSpec::parquet("docs", "/tmp/docs.parquet", "id", "vector", 8);
        let ec = ExternalCollection::new(spec.clone(), "snap-abc");
        assert!(ec.id.starts_with("ext_"));
        assert_eq!(ec.status, ExternalCollectionStatus::Registered);
        assert_eq!(ec.snapshot_id, "snap-abc");
        assert_eq!(ec.indexed_record_count, 0);
        assert!(!ec.is_ready());
        assert_eq!(ec.spec, spec);
    }

    #[test]
    fn serde_round_trips() {
        let spec = ExternalCollectionSpec::parquet("docs", "/tmp/docs.parquet", "id", "vector", 8);
        let ec = ExternalCollection::new(spec, "snap-1");
        let json = serde_json::to_string(&ec).unwrap();
        let back: ExternalCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(ec, back);
    }

    #[test]
    fn parquet_format_maps_to_catalog_parquet() {
        assert_eq!(
            ExternalFormat::Parquet.catalog_format(),
            proximadb_catalog::CatalogPhysicalFormat::Parquet
        );
    }
}
