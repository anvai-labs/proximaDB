//! xCatalog contract types and the `Catalog` trait.
//!
//! These serializable contracts are shared by catalog implementations, query planning, storage
//! selection, and protocol/API boundaries without requiring the root runtime crate.
//!
//! ## Key exports
//! - `TableIdentifier` — namespace + table name tuple for addressing tables
//! - `Catalog` — the core async trait every catalog backend implements
//! - All `Catalog*` types used in trait method signatures

use proximadb_data_model::ProximaType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

// Proto re-export so modules extracted from the root `src/catalog` keep their
// `crate::proto::proximadb_v1::*` paths (mirrors the observability engine crate).
pub mod proto {
    pub use proximadb_proto::proximadb_v1;
}

pub mod cache;
// Runtime/wiring modules extracted from the root `src/catalog` (decomposition
// Slice 1 — foundation-pure leaves). The root `crate::catalog` re-exports these
// so existing `crate::catalog::*` import paths are unchanged.
pub mod canonical_precision;
pub mod corpus_version_fs_store;
#[cfg(feature = "fc-metamodel")]
pub mod fc_metamodel;
#[cfg(feature = "fc-metamodel")]
pub mod grants;
pub mod internal;
pub mod partition_pruning;
#[cfg(feature = "fc-metamodel")]
pub mod principal_registry;
pub mod recall_probe;
pub mod syscat_cache;
pub mod syscat_warm;
#[cfg(feature = "fc-metamodel")]
pub mod tenant_posture;
// Catalog runtime manager (Slice 2) — `CatalogManager` + `TableOpLockRegistry`,
// moved from the root `src/catalog/mod.rs`. Object-store catalog URLs route
// through the injected `CatalogFilesystemResolver` port (no catalog->storage
// up-edge).
pub mod manager;
pub use manager::{CatalogFilesystemResolver, CatalogManager, TableOpLockRegistry};
/// Typed MLOps/model-registry facet for the unified xCatalog object model.
pub mod mlops;
/// Tenant-scoped model-registry lifecycle application service shared by API adapters.
pub mod model_registry_service;
/// MLflow-compatible tracking store port (TD-MLOPS-1) — experiments, runs,
/// params, metrics, tags; substrate-backed implementations live beside their
/// wiring, not here.
pub mod run_store;
// Catalog federation (Slice 3) — unified view across internal + external
// catalogs, moved from root src/catalog/federation (now that CatalogManager is
// in this crate).
// Collection-level DR / CRR engine contract (P1 of
// COLLECTION_DR_CRR_ENGINE_CONTRACT.adoc).
pub mod collection_dr_policy;
// Global corpus-version registry + store trait (relocated from the root crate's
// src/catalog so storage-side consumers like compaction can depend on it downward).
pub mod corpus_version;
pub use corpus_version::{
    ContentRevisionSnapshot, CorpusVersionDomain, CorpusVersionRegistry,
    DEFAULT_CORPUS_VERSION_TENANT, process_content_revision_incarnation,
};
// Customer-facing DR policy mutation surface (S14 of the same contract).
pub mod dr_policy_store;
// DR reconciler decision logic (P3a of the same contract).
pub mod dr_reconciler;
// DR restore-readiness primitives (P5 of the same contract).
#[cfg(feature = "delta-lake")]
pub mod delta;
pub mod dr_restore;
// Embedding-precision rollout (PR 6 of EMBEDDING_PRECISION_LLD_2026_05_22).
pub mod embedding_precision_policy;
#[cfg(feature = "aws")]
pub mod glue;
pub mod iceberg;
// Iceberg REST catalog service + PAX segment registry (moved from the root
// `src/catalog` — they follow `CatalogManager` into this crate now that the
// `ObjectStoreBridge` contract and `SegmentMeta` live at or below this layer).
#[cfg(test)]
mod authority_seal_tests;
pub mod iceberg_rest_service;
pub mod id_allocator;
pub mod native;
#[cfg(test)]
pub(crate) mod testfs;
// The canonical object-store bridge contract now lives in the foundation
// `proximadb-catalog-schema` crate; re-exported here so historical
// `proximadb_catalog::object_store_bridge::*` paths keep working and catalog
// services (Iceberg REST manifest generation) share one trait with the storage
// plane without a dependency cycle.
pub mod object_store_bridge {
    pub use proximadb_catalog_schema::object_store_bridge::*;
}
#[cfg(feature = "oltp-catalog")]
pub mod oltp;
#[cfg(feature = "polaris-catalog")]
pub mod polaris;
pub mod relational;
pub mod schema;
pub mod segment_registry;
pub use segment_registry::SegmentRegistry;
pub mod system_columns;
#[cfg(feature = "unity-catalog")]
pub mod unity;
// ─── Re-exports from the foundation schema crate (TD-DECOMP ratchet) ───────────
// These contract types were extracted into `proximadb-catalog-schema` so
// `proximadb-storage-common` can depend on them without a storage -> control
// upward edge. They are re-exported here unchanged; all historical
// `proximadb_catalog::*` import paths keep working.
pub use proximadb_catalog_schema::{
    AnnFilteringMode, AnnFilteringPolicy, CatalogAuthorityMode, CatalogBranchMergePolicy,
    CatalogBranchMergeResolution, CatalogColumn, CatalogCompressionRejectedCandidate,
    CatalogCompressionStatsProfile, CatalogEmbeddingConfig, CatalogIndex, CatalogIndexType,
    CatalogPhysicalFormat, CatalogPrimaryPod, CatalogPrimaryPodReason, CatalogProjection,
    CatalogProjectionKind, CatalogStorageConfig, CatalogStorageLayout, CatalogStorageLayoutKind,
    CatalogStorageSpecialization, CatalogTableSchema, CatalogWorkloadProfile, CatalogWriteMode,
    ColumnConstraint, ObservabilityCompressionHint, PrecisionMigrationState, ProjectionFreshness,
    ProjectionFreshnessState, PropsAutoPromotionPolicy, PropsEvaluationCadence,
    RebuildEstimateSource, RebuildRtoSpec, RecallSlo, RecallTargets, ReferentialAction,
    RelationalCapabilities, StoragePoolClass, catalog_arrow_type, same_column_set,
};

/// Typed accessors for the opaque `mlops_asset` payload on [`CatalogTableSchema`].
///
/// The `mlops_asset` field is `Option<serde_json::Value>` in the foundation
/// `proximadb-catalog-schema` crate (so it need not depend on this crate's
/// `mlops::CatalogMlopsAsset` type). These accessors (de)serialize between the
/// opaque value and the typed asset. This is an extension trait because Rust
/// forbids inherent `impl` blocks on a type outside its defining crate (E0116);
/// the methods that need only pure helpers (`validate_model_binding`, `pinned`,
/// `with_unique`) stayed behind as inherent methods in the foundation crate.
pub trait CatalogMlopsAssetExt: Sized {
    /// Deserialize the opaque payload into the typed asset (if any).
    fn mlops_asset_as_typed(&self) -> anyhow::Result<Option<mlops::CatalogMlopsAsset>>;
    /// Serialize a typed asset into the opaque payload.
    fn set_mlops_asset_typed(&mut self, asset: mlops::CatalogMlopsAsset);
    /// Validate a typed asset and attach it to this catalog object.
    fn with_mlops_asset(self, asset: mlops::CatalogMlopsAsset) -> anyhow::Result<Self>;
}

impl CatalogMlopsAssetExt for CatalogTableSchema {
    fn mlops_asset_as_typed(&self) -> anyhow::Result<Option<mlops::CatalogMlopsAsset>> {
        self.mlops_asset
            .as_ref()
            .map(|value| serde_json::from_value(value.clone()).map_err(anyhow::Error::new))
            .transpose()
    }

    fn set_mlops_asset_typed(&mut self, asset: mlops::CatalogMlopsAsset) {
        self.mlops_asset = serde_json::to_value(&asset).ok();
    }

    fn with_mlops_asset(mut self, asset: mlops::CatalogMlopsAsset) -> anyhow::Result<Self> {
        asset.validate().map_err(anyhow::Error::new)?;
        let value = serde_json::to_value(&asset).map_err(anyhow::Error::new)?;
        self.mlops_asset = Some(value);
        Ok(self)
    }
}
/// Plane a catalog instance serves in the two-tier operator/account isolation
/// model (Phase 5).
///
/// * [`Operator`](Self::Operator) — the SaaS provider's **control-plane**
///   registry of all customer accounts (entitlements, storage bindings,
///   billing). It holds metadata-*about*-accounts, never tenant table data,
///   and is stored under the reserved `_operator/` root.
/// * [`Account`](Self::Account) — a customer account's **data-plane** catalog
///   (its tenants → namespaces → objects), stored under
///   `accounts/{account_id}/…`. A per-account catalog is structurally unable to
///   name another account's objects — isolation is structural, not a query
///   predicate.
///
/// Single-deployment default is one `Operator`-roled system catalog (it holds
/// the whole deployment's objects until multi-account provisioning splits
/// data-plane catalogs out per account).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogRole {
    /// Control-plane registry of accounts (under `_operator/`).
    Operator,
    /// Data-plane catalog scoped to one customer account.
    Account {
        /// The owning account ID (roots the catalog under `accounts/{id}/…`).
        account_id: String,
    },
}

impl CatalogRole {
    /// The owning account ID for an [`Account`](Self::Account) catalog, or
    /// `None` for the [`Operator`](Self::Operator) control plane.
    pub fn account_id(&self) -> Option<&str> {
        match self {
            CatalogRole::Operator => None,
            CatalogRole::Account { account_id } => Some(account_id.as_str()),
        }
    }

    /// True for the control-plane operator catalog.
    pub fn is_operator(&self) -> bool {
        matches!(self, CatalogRole::Operator)
    }
}

/// Namespace metadata.
///
/// Serves two roles:
///
/// 1. **Iceberg-REST federation identifier** — `levels`, `properties`,
///    `owner`, `location`, timestamps. Compatible with the Iceberg REST
///    catalog wire format.
/// 2. **Engine multi-tenant authority** — `namespace_id`, `tenant_id`,
///    `region_home`, `default_dr_region_pair_id`, `storage_pool_class`.
///    Drives the physical path layout
///    `data/{tenant_id}/{namespace_id}/{collection_id}/` and the DR
///    policy authority boundary.
///
/// New engine-authoritative fields are `Option<>` for backwards
/// compatibility with the P0.5 migration. The DR-strict path
/// (reconciler, path resolver guard) refuses null `namespace_id` /
/// `tenant_id`. The Iceberg-REST handler projects only the federation
/// fields so external Iceberg clients see the wire format they expect.
///
/// See `docs/12-design/COLLECTION_DR_CRR_ENGINE_CONTRACT.adoc` "Namespace
/// As The DR Authority Boundary".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogNamespace {
    // --- Iceberg-REST federation fields (mutable labels) ---
    /// Namespace hierarchy (e.g., ["database", "schema"])
    pub levels: Vec<String>,
    /// Namespace properties
    pub properties: HashMap<String, String>,
    /// Owner principal
    pub owner: Option<String>,
    /// Storage location
    pub location: Option<String>,
    /// Creation timestamp (millis since epoch)
    pub created_at_ms: i64,
    /// Last update timestamp (millis since epoch)
    pub updated_at_ms: i64,

    // --- Engine multi-tenant authority fields (stable identifiers) ---
    /// ADR-031 / TD-181 P0: stable, immutable `u64` **catalog object identity**,
    /// minted from the single system-wide catalog sequence (globally unique
    /// across all tenants, never reused). This is the catalog-object surrogate
    /// the planner/FK/path layers will key on; it is distinct from
    /// `namespace_id` (the legacy `ns_<uuid>` path token, retired in a later
    /// phase) and from `tenant_id`/`account_id` (externally assigned, **not**
    /// catalog surrogates — ADR-031 reconciliation amendment 2). Additive +
    /// `#[serde(default)]`, so legacy rows and federated (external-catalog)
    /// namespaces load as `None` (mixed-read-safe; only native-minted
    /// namespaces carry one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_id: Option<u64>,
    /// Opaque, stable, server-issued ULID. Never reused, never changes on
    /// rename. Drives physical paths and provider rule filters. `None`
    /// for legacy rows pending P0.5 migration backfill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace_id: Option<String>,
    /// Owning tenant. A namespace cannot be re-parented. `None` for
    /// legacy rows pending migration backfill (target value:
    /// `"tnt_legacy_system"` until operator re-parents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// Owning customer **account** — the SaaS billing/isolation boundary
    /// that sits *above* `tenant_id` (a tenant is a workspace/sub-org
    /// inside an account). Drives the account-rooted physical path
    /// `accounts/{account_id}/{tenant_id}/{namespace_id}/{object_id}/`.
    /// `None` keeps the legacy flat `data/{tenant_id}/...` render
    /// (mixed-safe; the account tier is inert until provisioned).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// Region where authoritative writes land. `None` is allowed only
    /// for namespaces with no DR policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_home: Option<String>,
    /// Operator-curated default that new collections inherit at DR
    /// enablement time. Recommended canonical form
    /// `{provider}:{source_region}:{destination_region}`; engine treats
    /// it as opaque.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_dr_region_pair_id: Option<String>,
    /// Storage pool class. Path resolver refuses writes that target a
    /// bucket/container outside the matching class. Defaults to `Pooled`
    /// for backwards compatibility with legacy rows.
    #[serde(default)]
    pub storage_pool_class: StoragePoolClass,
}

impl CatalogNamespace {
    /// Create a new namespace with only the Iceberg-REST federation
    /// fields populated. Engine multi-tenant fields are `None` /
    /// defaults; use `with_tenant`, `with_namespace_id`, etc. to set
    /// them when constructing an engine-authoritative namespace.
    pub fn new(levels: Vec<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        Self {
            levels,
            properties: HashMap::new(),
            owner: None,
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
            object_id: None,
            namespace_id: None,
            tenant_id: None,
            account_id: None,
            region_home: None,
            default_dr_region_pair_id: None,
            storage_pool_class: StoragePoolClass::default(),
        }
    }

    /// Get fully qualified name
    pub fn fqn(&self) -> String {
        self.levels.join(".")
    }

    /// Set the stable `u64` catalog object identity (ADR-031 / TD-181 P0).
    pub fn with_object_id(mut self, object_id: u64) -> Self {
        self.object_id = Some(object_id);
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set owner
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Set location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set the engine-authoritative namespace ID (server-issued ULID).
    pub fn with_namespace_id(mut self, namespace_id: impl Into<String>) -> Self {
        self.namespace_id = Some(namespace_id.into());
        self
    }

    /// Set the owning tenant.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set the owning customer account (the billing/isolation boundary
    /// above `tenant_id`). Activates the account-rooted physical path;
    /// leaving it unset keeps the legacy flat `data/{tenant_id}/...` render.
    pub fn with_account(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Set the home region for authoritative writes.
    pub fn with_region_home(mut self, region: impl Into<String>) -> Self {
        self.region_home = Some(region.into());
        self
    }

    /// Set the default region-pair ID for DR enablement.
    pub fn with_default_dr_region_pair(mut self, pair_id: impl Into<String>) -> Self {
        self.default_dr_region_pair_id = Some(pair_id.into());
        self
    }

    /// Set the storage pool class.
    pub fn with_storage_pool_class(mut self, class: StoragePoolClass) -> Self {
        self.storage_pool_class = class;
        self
    }

    /// True when both `namespace_id` and `tenant_id` are populated. The
    /// DR path resolver and reconciler require this; legacy namespaces
    /// pending migration backfill return false.
    pub fn is_dr_addressable(&self) -> bool {
        self.namespace_id.is_some() && self.tenant_id.is_some()
    }
}
// ─── Protocol Type Coercion ───────────────────────────────────────────────────

/// Lossy conversion descriptor for a single ProximaType when exposed through
/// a specific protocol. Used by planners and EXPLAIN to surface type fidelity
/// warnings before data is written to or read from an external protocol surface.
///
/// See ADR-013 for the full coercion matrix across pgwire, Arrow Flight,
/// Iceberg REST, and REST/gRPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolTypeCoercionNote {
    /// The ProximaType field or type name being coerced.
    pub proxima_type: String,
    /// Target protocol surface: "pgwire", "arrow_flight", "iceberg_rest", "rest_grpc".
    pub protocol: String,
    /// The wire type used in the protocol.
    pub wire_type: String,
    /// Whether the conversion is lossless.
    pub lossless: bool,
    /// Description of what information is lost or altered.
    pub loss_description: Option<String>,
}

impl ProtocolTypeCoercionNote {
    /// Standard coercion table entries grounded in ADR-007 and ADR-013.
    ///
    /// Returns a reference to a process-lifetime static; do not call this per-query
    /// once wired into EXPLAIN — clone the slice once at plan construction time.
    pub fn standard_notes() -> &'static [Self] {
        static NOTES: OnceLock<Vec<ProtocolTypeCoercionNote>> = OnceLock::new();
        NOTES.get_or_init(|| {
            vec![
                Self {
                    proxima_type: "ProximaValue::Map(string, ProximaValue)".to_string(),
                    protocol: "iceberg_rest".to_string(),
                    wire_type: "map<string, binary> (msgpack)".to_string(),
                    lossless: false,
                    loss_description: Some(
                        "Spark/Trino cannot push predicates into msgpack binary; \
                         use props auto-promotion (PropsAutoPromotionPolicy) to \
                         promote high-frequency keys to typed columns."
                            .to_string(),
                    ),
                },
                Self {
                    proxima_type: "Embedding([f32; dim])".to_string(),
                    protocol: "iceberg_rest".to_string(),
                    wire_type: "list<float> (dimension in table properties)".to_string(),
                    lossless: false,
                    loss_description: Some(
                        "Iceberg list<float> does not carry fixed dimension. \
                         Dimension is stored in table property 'proximadb.dim_<model_id>'. \
                         Clients must read the property to validate vector dimensions."
                            .to_string(),
                    ),
                },
                Self {
                    proxima_type: "Embedding([f32; dim])".to_string(),
                    protocol: "pgwire".to_string(),
                    wire_type: "float4[] (pgvector compatible)".to_string(),
                    lossless: true,
                    loss_description: None,
                },
                Self {
                    proxima_type: "Embedding([f32; dim])".to_string(),
                    protocol: "arrow_flight".to_string(),
                    wire_type: "FixedSizeList<float32>[dim]".to_string(),
                    lossless: true,
                    loss_description: None,
                },
                Self {
                    proxima_type: "ProximaValue::Variant".to_string(),
                    protocol: "iceberg_rest".to_string(),
                    wire_type: "binary (not representable as Iceberg column)".to_string(),
                    lossless: false,
                    loss_description: Some(
                        "Iceberg does not have a union/variant type. \
                         Variant values serialize as opaque binary. \
                         Use a typed column instead."
                            .to_string(),
                    ),
                },
                Self {
                    proxima_type: "Edge { source, target, edge_type, weight }".to_string(),
                    protocol: "pgwire".to_string(),
                    wire_type: "4 separate nullable columns (no native graph type)".to_string(),
                    lossless: true,
                    loss_description: None,
                },
            ]
        })
    }
}
/// Table statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogTableStatistics {
    /// Row count
    pub row_count: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Number of files
    pub file_count: u64,
    /// Last analyze timestamp
    pub last_analyzed_ms: Option<i64>,
    /// Column statistics
    pub column_stats: HashMap<String, CatalogColumnStatistics>,
}

impl CatalogTableStatistics {
    /// Returns true if these statistics should be treated as stale by the planner.
    ///
    /// Stats are stale when:
    /// - `last_analyzed_ms` is `None` (never updated), or
    /// - `now_ms - last_analyzed_ms > ttl_ms` (older than the configured freshness window).
    ///
    /// Planners should fall back to defaults (or trigger a refresh) when this returns true.
    pub fn is_stale(&self, now_ms: i64, ttl_ms: i64) -> bool {
        match self.last_analyzed_ms {
            None => true,
            Some(last_ms) => now_ms.saturating_sub(last_ms) > ttl_ms,
        }
    }
}

/// Column statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogColumnStatistics {
    /// Number of distinct values
    pub distinct_count: Option<u64>,
    /// Number of null values
    pub null_count: Option<u64>,
    /// Min value (as string)
    pub min_value: Option<String>,
    /// Max value (as string)
    pub max_value: Option<String>,
    /// Average size in bytes
    pub avg_size_bytes: Option<f64>,
}

/// Partition specification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogPartitionSpec {
    /// Spec ID
    pub spec_id: i32,
    /// Partition fields
    pub fields: Vec<CatalogPartitionField>,
}

/// Partition field (Iceberg-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogPartitionField {
    /// Source column ID
    pub source_id: i32,
    /// Field ID (assigned by partition spec)
    pub field_id: i32,
    /// Field name
    pub name: String,
    /// Transform type
    pub transform: PartitionTransform,
}

/// Type alias for backwards compatibility
pub type PartitionField = CatalogPartitionField;

/// Partition transforms (Iceberg-compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PartitionTransform {
    /// Identity transform (no transformation)
    #[default]
    Identity,
    /// Bucket transform (hash into N buckets)
    Bucket(u32),
    /// Truncate transform (truncate to width)
    Truncate(u32),
    /// Year transform (extract year from timestamp/date)
    Year,
    /// Month transform (extract month from timestamp/date)
    Month,
    /// Day transform (extract day from timestamp/date)
    Day,
    /// Hour transform (extract hour from timestamp)
    Hour,
    /// Void transform (always produces null)
    Void,
}

impl std::fmt::Display for PartitionTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitionTransform::Identity => write!(f, "identity"),
            PartitionTransform::Bucket(n) => write!(f, "bucket[{}]", n),
            PartitionTransform::Truncate(w) => write!(f, "truncate[{}]", w),
            PartitionTransform::Year => write!(f, "year"),
            PartitionTransform::Month => write!(f, "month"),
            PartitionTransform::Day => write!(f, "day"),
            PartitionTransform::Hour => write!(f, "hour"),
            PartitionTransform::Void => write!(f, "void"),
        }
    }
}

impl PartitionTransform {
    /// Parse from string (Iceberg format)
    pub fn parse_from_iceberg_format(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower == "identity" {
            PartitionTransform::Identity
        } else if lower.starts_with("bucket[") {
            let n: u32 = lower
                .trim_start_matches("bucket[")
                .trim_end_matches(']')
                .parse()
                .unwrap_or(16);
            PartitionTransform::Bucket(n)
        } else if lower.starts_with("truncate[") {
            let w: u32 = lower
                .trim_start_matches("truncate[")
                .trim_end_matches(']')
                .parse()
                .unwrap_or(16);
            PartitionTransform::Truncate(w)
        } else if lower == "year" {
            PartitionTransform::Year
        } else if lower == "month" {
            PartitionTransform::Month
        } else if lower == "day" {
            PartitionTransform::Day
        } else if lower == "hour" {
            PartitionTransform::Hour
        } else if lower == "void" {
            PartitionTransform::Void
        } else {
            PartitionTransform::Identity
        }
    }
}

/// Sort order
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogSortOrder {
    /// Order ID
    pub order_id: i32,
    /// Sort fields
    pub fields: Vec<CatalogSortField>,
}

/// Sort field (Iceberg-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSortField {
    /// Source column ID
    pub source_id: i32,
    /// Transform to apply before sorting
    pub transform: PartitionTransform,
    /// Sort direction
    pub direction: SortDirection,
    /// Null ordering
    pub null_order: NullOrder,
}

/// Type alias for backwards compatibility
pub type SortField = CatalogSortField;

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    /// Sort values from smallest to largest (ASC)
    Ascending,
    /// Sort values from largest to smallest (DESC)
    Descending,
}

/// Null ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullOrder {
    /// NULL values sort before non-NULL values
    NullsFirst,
    /// NULL values sort after non-NULL values
    NullsLast,
}

/// Schema evolution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSchemaEvolution {
    /// Changes to apply
    pub changes: Vec<SchemaChange>,
}

/// Schema change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChange {
    /// Add a new column
    AddColumn {
        /// Name of the new column
        name: String,
        /// Data type of the new column
        data_type: ProximaType,
        /// Whether the column accepts NULL values
        nullable: bool,
        /// Optional SQL expression used as the column default
        default_value: Option<String>,
        /// Optional human-readable description of the column
        comment: Option<String>,
        /// Position: None = end, Some(col) = after column
        after: Option<String>,
    },
    /// Drop a column
    DropColumn {
        /// Name of the column to drop
        name: String,
    },
    /// Rename a column
    RenameColumn {
        /// Current column name
        old_name: String,
        /// Desired new column name
        new_name: String,
    },
    /// Change column type (must be compatible)
    ChangeType {
        /// Name of the column whose type should change
        name: String,
        /// New data type (must be compatible with the existing type)
        new_type: ProximaType,
    },
    /// Update column comment
    UpdateComment {
        /// Name of the column to update
        name: String,
        /// New comment text
        comment: String,
    },
    /// Make column nullable (DROP NOT NULL)
    MakeNullable {
        /// Name of the column to make nullable
        name: String,
    },
    /// Make column NOT NULL (SET NOT NULL)
    MakeNotNullable {
        /// Name of the column to make non-nullable
        name: String,
    },
    /// Add default value
    SetDefault {
        /// Name of the column to update
        name: String,
        /// SQL expression to use as the default value
        default_value: String,
    },
    /// Remove default value
    DropDefault {
        /// Name of the column whose default should be removed
        name: String,
    },
    /// Move column position (FIRST or AFTER another column)
    MoveColumn {
        /// Name of the column to move
        name: String,
        /// Position: None = FIRST, Some(col) = AFTER column
        after: Option<String>,
    },
    /// Add a constraint to a column or table
    AddConstraint {
        /// Constraint name (optional for some DBs)
        constraint_name: Option<String>,
        /// Constraint type
        constraint: ColumnConstraint,
    },
    /// Drop a constraint
    DropConstraint {
        /// Name of the constraint to drop
        constraint_name: String,
    },
    /// Promote a high-frequency props key to a typed PAX/Iceberg column.
    ///
    /// The promoted column name follows the `props__<key>` convention (double
    /// underscore) so it is distinguishable from user-defined schema columns.
    /// The storage layer re-routes the key from the msgpack blob (column ID 8)
    /// into the new typed column during the next compaction cycle.
    PromotePropsKey {
        /// The key inside the `props` msgpack blob to promote.
        key: String,
        /// Target catalog data type for the promoted column.
        column_type: ProximaType,
        /// Optional human-readable description for the new column.
        comment: Option<String>,
    },
    /// Set a table-level option (ALTER TABLE … SET (key = 'value')).
    ///
    /// Recognised option keys:
    /// - `props_auto_promotion`: `'enabled'` / `'disabled'`
    SetTableOption {
        /// Option name (case-insensitive).
        key: String,
        /// Option value.
        value: String,
    },
}
#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_compression_types::CompressionAlgorithm;
    use proximadb_data_model::{TimeUnit, VectorElement};
    use proximadb_distance_types::DistanceMetric;
    use proximadb_quantization_types::QuantizationType;

    #[test]
    fn test_namespace_fqn() {
        let ns = CatalogNamespace::new(vec!["catalog".into(), "database".into()]);
        assert_eq!(ns.fqn(), "catalog.database");
    }

    #[test]
    fn namespace_defaults_are_legacy_compatible() {
        // Legacy callers that build a namespace without DR fields get
        // backwards-compatible defaults: no tenant/namespace ID, no
        // region home, Pooled pool class. The migration backfills these.
        let ns = CatalogNamespace::new(vec!["legacy".into()]);
        assert!(ns.namespace_id.is_none());
        assert!(ns.tenant_id.is_none());
        assert!(ns.region_home.is_none());
        assert!(ns.default_dr_region_pair_id.is_none());
        assert_eq!(ns.storage_pool_class, StoragePoolClass::Pooled);
        assert!(!ns.is_dr_addressable());
    }

    #[test]
    fn catalog_role_accessors_and_serde() {
        let op = CatalogRole::Operator;
        assert!(op.is_operator());
        assert_eq!(op.account_id(), None);

        let acct = CatalogRole::Account {
            account_id: "acct_acme".to_string(),
        };
        assert!(!acct.is_operator());
        assert_eq!(acct.account_id(), Some("acct_acme"));

        // snake_case, round-trips.
        let j = serde_json::to_string(&op).expect("ser operator");
        assert_eq!(j, "\"operator\"");
        let back: CatalogRole = serde_json::from_str(&j).expect("de operator");
        assert_eq!(back, op);
        let j2 = serde_json::to_string(&acct).expect("ser account");
        let back2: CatalogRole = serde_json::from_str(&j2).expect("de account");
        assert_eq!(back2, acct);
    }

    #[test]
    fn namespace_dr_builders_compose() {
        let ns = CatalogNamespace::new(vec!["catalog".into(), "db".into()])
            .with_account("acct_acme")
            .with_tenant("tnt_acme")
            .with_namespace_id("ns_01HX7Q8K2N5R9P3M1B2C3D4E5F")
            .with_region_home("us-east-1")
            .with_default_dr_region_pair("aws:us-east-1:us-west-2")
            .with_storage_pool_class(StoragePoolClass::Standard);

        assert_eq!(ns.account_id.as_deref(), Some("acct_acme"));
        assert_eq!(ns.tenant_id.as_deref(), Some("tnt_acme"));
        assert_eq!(
            ns.namespace_id.as_deref(),
            Some("ns_01HX7Q8K2N5R9P3M1B2C3D4E5F"),
        );
        assert_eq!(ns.region_home.as_deref(), Some("us-east-1"));
        assert_eq!(
            ns.default_dr_region_pair_id.as_deref(),
            Some("aws:us-east-1:us-west-2"),
        );
        assert_eq!(ns.storage_pool_class, StoragePoolClass::Standard);
        assert!(ns.is_dr_addressable());
    }

    #[test]
    fn namespace_serde_round_trips_legacy_rows() {
        // A namespace serialized before this migration has none of the
        // new fields. Deserializing must succeed and leave them at
        // their defaults.
        let legacy_json = r#"{
            "levels": ["db", "schema"],
            "properties": {},
            "owner": null,
            "location": null,
            "created_at_ms": 1000,
            "updated_at_ms": 2000
        }"#;
        let ns: CatalogNamespace =
            serde_json::from_str(legacy_json).expect("legacy namespace JSON must deserialize");
        assert!(ns.namespace_id.is_none());
        assert!(ns.tenant_id.is_none());
        assert!(ns.account_id.is_none());
        assert_eq!(ns.storage_pool_class, StoragePoolClass::Pooled);

        // Re-serializing must skip the None fields so legacy consumers
        // still see only the federation fields.
        let reserialized = serde_json::to_string(&ns).expect("serialize");
        assert!(!reserialized.contains("namespace_id"));
        assert!(!reserialized.contains("tenant_id"));
        assert!(!reserialized.contains("account_id"));
        assert!(!reserialized.contains("region_home"));
        assert!(!reserialized.contains("default_dr_region_pair_id"));
        // `storage_pool_class` is non-Option so it does show up; that's
        // expected because legacy rows backfill to "pooled".
        assert!(reserialized.contains("\"storage_pool_class\":\"pooled\""));
    }

    #[test]
    fn storage_pool_class_serde_uses_snake_case() {
        // Serialization uses the neutral capability-class names; legacy commercial
        // wire values still deserialize via serde aliases (back-compat).
        let classes = [
            (StoragePoolClass::Pooled, "\"pooled\"", None),
            (
                StoragePoolClass::Standard,
                "\"standard\"",
                Some("\"business\""),
            ),
            (
                StoragePoolClass::Premium,
                "\"premium\"",
                Some("\"enterprise\""),
            ),
            (
                StoragePoolClass::Dedicated,
                "\"dedicated\"",
                Some("\"enterprise_dedicated\""),
            ),
        ];
        for (variant, expected_json, legacy_alias) in classes {
            let s = serde_json::to_string(&variant).unwrap();
            assert_eq!(s, expected_json, "variant {variant:?}");
            let back: StoragePoolClass = serde_json::from_str(expected_json).unwrap();
            assert_eq!(back, variant);
            if let Some(alias) = legacy_alias {
                let from_alias: StoragePoolClass = serde_json::from_str(alias).unwrap();
                assert_eq!(from_alias, variant, "legacy alias {alias} must still read");
            }
        }
    }

    #[test]
    fn storage_pool_class_default_is_pooled() {
        assert_eq!(StoragePoolClass::default(), StoragePoolClass::Pooled);
    }

    #[test]
    fn test_column_builder() {
        let col = CatalogColumn::new(1, "id", ProximaType::Int64)
            .nullable(false)
            .with_comment("Primary key");

        assert_eq!(col.name, "id");
        assert!(!col.nullable);
        assert_eq!(col.comment, Some("Primary key".to_string()));
    }

    #[test]
    fn test_table_schema_builder() {
        let schema = CatalogTableSchema::new("users")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "name", ProximaType::String))
            .with_primary_key(vec!["id".into()]);

        assert_eq!(schema.name, "users");
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.primary_key, vec!["id"]);
        assert_eq!(schema.storage_layouts.len(), 1);
        assert_eq!(
            schema.storage_layouts[0].authority,
            CatalogAuthorityMode::InternalCanonical
        );
    }

    #[test]
    fn test_data_type_roundtrip() {
        // Proto-i32 round-trips now live on the canonical type.
        let types = vec![
            ProximaType::Boolean,
            ProximaType::Int64,
            ProximaType::Float64,
            ProximaType::String,
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0,
            },
        ];

        for dt in types {
            let proto = dt.to_proto_i32();
            let back = ProximaType::from_proto_i32(proto).expect("known code");
            assert_eq!(dt, back);
        }
    }

    /// DURABLE catalog-format compatibility: a `CatalogColumn` persisted in the
    /// legacy `CatalogDataType` form (bare unit-string tags) MUST still
    /// deserialize, mapping onto the canonical [`ProximaType`].
    #[test]
    fn catalog_column_deserializes_legacy_catalog_data_type_form() {
        // Legacy JSON: data_type as a bare string (the old CatalogDataType
        // unit-variant encoding), including the tags that DIFFER from the
        // canonical ProximaType encoding (Decimal/Vector/TimestampTz are
        // struct variants in ProximaType but bare strings in the legacy form).
        let legacy = r#"[
                {"id": 1, "name": "id", "data_type": "Int64", "nullable": false,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 2, "name": "balance", "data_type": "Decimal", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 3, "name": "embedding", "data_type": "Vector", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 4, "name": "created", "data_type": "TimestampTz", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 5, "name": "tags", "data_type": "SparseVector", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 6, "name": "bits", "data_type": "BinaryVector", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}},
                {"id": 7, "name": "label", "data_type": "String", "nullable": true,
                 "default_value": null, "comment": null, "properties": {}}
            ]"#;

        let columns: Vec<CatalogColumn> =
            serde_json::from_str(legacy).expect("legacy catalog columns must deserialize");
        let by_name = |n: &str| {
            columns
                .iter()
                .find(|c| c.name == n)
                .unwrap()
                .data_type
                .clone()
        };
        assert_eq!(by_name("id"), ProximaType::Int64);
        assert_eq!(
            by_name("balance"),
            ProximaType::Decimal {
                precision: 38,
                scale: 10
            }
        );
        assert_eq!(
            by_name("embedding"),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0
            }
        );
        assert_eq!(
            by_name("created"),
            ProximaType::TimestampTz(TimeUnit::Nanosecond)
        );
        assert_eq!(
            by_name("tags"),
            ProximaType::SparseVector {
                element: VectorElement::Float32
            }
        );
        assert_eq!(by_name("bits"), ProximaType::BinaryVector { dim: 0 });
        assert_eq!(by_name("label"), ProximaType::String);
    }

    /// The NEW canonical [`ProximaType`] form (struct variants as objects) must
    /// also deserialize, and a serialize → deserialize round-trip must hold.
    #[test]
    fn catalog_column_round_trips_canonical_proxima_type_form() {
        let col = CatalogColumn::new(
            2,
            "balance",
            ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
        );
        let json = serde_json::to_string(&col).unwrap();
        // Struct variant must serialize in the canonical object form.
        assert!(json.contains("\"Decimal\":{"), "got: {json}");
        let back: CatalogColumn = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.data_type,
            ProximaType::Decimal {
                precision: 38,
                scale: 10
            }
        );
    }

    #[test]
    fn workload_and_storage_specialization_options_parse_and_render_stable_names() {
        let workload_cases = [
            ("transactional", CatalogWorkloadProfile::Oltp, "oltp"),
            ("analytics", CatalogWorkloadProfile::Olap, "olap"),
            ("pax", CatalogWorkloadProfile::Htap, "htap"),
            ("ann", CatalogWorkloadProfile::Vector, "vector"),
            ("jsonb", CatalogWorkloadProfile::Document, "document"),
            ("cypher", CatalogWorkloadProfile::Graph, "graph"),
            (
                "time_series",
                CatalogWorkloadProfile::Observability,
                "observability",
            ),
            ("multimodal", CatalogWorkloadProfile::Mixed, "mixed"),
        ];
        for (input, expected, rendered) in workload_cases {
            assert_eq!(CatalogWorkloadProfile::parse(input), Some(expected));
            assert_eq!(expected.as_str(), rendered);
        }
        assert_eq!(CatalogWorkloadProfile::parse("unknown"), None);

        let specialization_cases = [
            (
                "row_record",
                CatalogStorageSpecialization::GenericRelational,
                "generic_relational",
            ),
            (
                "pax",
                CatalogStorageSpecialization::PaxRowFamily,
                "pax_row_family",
            ),
            ("rowdir", CatalogStorageSpecialization::PaxOltp, "pax_oltp"),
            (
                "pax_olap",
                CatalogStorageSpecialization::PaxOlap,
                "pax_olap",
            ),
            (
                "lsm",
                CatalogStorageSpecialization::LsmWriteOptimized,
                "lsm_write_optimized",
            ),
            (
                "analytics",
                CatalogStorageSpecialization::ColumnarAnalytics,
                "columnar_analytics",
            ),
            (
                "hnsw",
                CatalogStorageSpecialization::VectorAnn,
                "vector_ann",
            ),
            (
                "json",
                CatalogStorageSpecialization::DocumentJson,
                "document_json",
            ),
            (
                "csr",
                CatalogStorageSpecialization::GraphTopology,
                "graph_topology",
            ),
            (
                "timeseries",
                CatalogStorageSpecialization::ObservabilityTimeSeries,
                "observability_time_series",
            ),
            (
                "lakehouse",
                CatalogStorageSpecialization::ExternalOpenTable,
                "external_open_table",
            ),
        ];
        for (input, expected, rendered) in specialization_cases {
            assert_eq!(CatalogStorageSpecialization::parse(input), Some(expected));
            assert_eq!(expected.as_str(), rendered);
        }
        assert_eq!(CatalogStorageSpecialization::parse("unknown"), None);
    }

    #[test]
    fn authority_layout_projection_and_compression_contracts_cover_builder_surface() {
        let authority_cases = [
            (
                CatalogAuthorityMode::InternalCanonical,
                true,
                false,
                false,
                "ProximaAuthoritative",
            ),
            (
                CatalogAuthorityMode::ProximaAuthoritative,
                true,
                false,
                false,
                "ProximaAuthoritative",
            ),
            (
                CatalogAuthorityMode::ExternalAuthoritative,
                false,
                true,
                false,
                "ExternalAuthoritative",
            ),
            (
                CatalogAuthorityMode::ImportedSnapshot,
                false,
                false,
                false,
                "ImportedSnapshot",
            ),
            (
                CatalogAuthorityMode::ExportedPublication,
                false,
                false,
                true,
                "ProjectionPublication",
            ),
            (
                CatalogAuthorityMode::ProjectionPublication,
                false,
                false,
                true,
                "ProjectionPublication",
            ),
            (
                CatalogAuthorityMode::RebuildableProjection,
                false,
                false,
                true,
                "RebuildableProjection",
            ),
            (
                CatalogAuthorityMode::FederatedRead,
                false,
                false,
                false,
                "FederatedRead",
            ),
        ];
        for (mode, proxima, external, rebuildable, name) in authority_cases {
            assert_eq!(mode.is_proxima_authoritative(), proxima);
            assert_eq!(mode.is_external_authoritative(), external);
            assert_eq!(mode.is_rebuildable_or_publication(), rebuildable);
            assert_eq!(mode.ownership_mode_name(), name);
        }

        let pax = CatalogStorageLayout::proxima_authoritative_pax("primary");
        assert_eq!(pax.layout_kind, CatalogStorageLayoutKind::Pax);
        assert_eq!(pax.snapshot_semantics.as_deref(), Some("mvcc"));
        assert!(!pax.requires_external_contract());

        let publication = CatalogStorageLayout::projection_publication(
            "iceberg_pub",
            CatalogPhysicalFormat::Iceberg,
            "s3://pub",
        );
        assert_eq!(
            publication.authority,
            CatalogAuthorityMode::ProjectionPublication
        );
        assert_eq!(publication.write_mode, CatalogWriteMode::CopyOnWrite);

        let specialty = CatalogStorageLayout::specialty_projection(
            "hnsw",
            CatalogStorageLayoutKind::VectorAnn,
            CatalogPhysicalFormat::ProximaBlock,
        );
        assert_eq!(
            specialty.authority,
            CatalogAuthorityMode::RebuildableProjection
        );

        let imported = CatalogStorageLayout::imported_snapshot(
            "import",
            CatalogPhysicalFormat::Parquet,
            "s3://import",
        );
        assert_eq!(imported.authority, CatalogAuthorityMode::ImportedSnapshot);

        let federated =
            CatalogStorageLayout::federated_read("fed", CatalogPhysicalFormat::Delta, "s3://delta");
        assert!(federated.requires_external_contract());

        let projection =
            CatalogProjection::rebuildable("col", CatalogProjectionKind::Columnar, "primary")
                .with_bounded_lag(1_000)
                .with_rebuild_rto(RebuildRtoSpec::columnar_estimated())
                .with_freshness_state(ProjectionFreshnessState::Updating)
                .with_lineage("wal:1", "wal:2")
                .with_policy_and_gate("engine", "smoke");
        assert_eq!(projection.max_lag_ms, Some(1_000));
        assert!(
            projection
                .rebuild_rto
                .as_ref()
                .unwrap()
                .supports_incremental_rebuild
        );
        assert_eq!(
            projection.freshness_state,
            ProjectionFreshnessState::Updating
        );
        assert_eq!(projection.source_range.as_deref(), Some("wal:1"));
        assert_eq!(projection.benchmark_gate.as_deref(), Some("smoke"));

        let profile = CatalogCompressionStatsProfile::new("p", "codec", 0, 10, 0, true)
            .with_layout_name("primary")
            .with_projection_id("proj")
            .with_decode_ns_per_value(4.0);
        assert_eq!(profile.measured_ratio, 0.0);
        assert_eq!(profile.bytes_per_value(), 0.0);
        assert_eq!(profile.decode_ns_per_value, Some(4.0));
    }

    #[test]
    fn ann_rto_props_observability_protocol_partition_and_health_helpers_are_covered() {
        for (input, mode, rendered) in [
            ("pre", AnnFilteringMode::PreFilter, "pre_filter"),
            ("intra", AnnFilteringMode::Inline, "inline"),
            ("post", AnnFilteringMode::PostFilter, "post_filter"),
        ] {
            assert_eq!(AnnFilteringMode::parse(input), Some(mode));
            assert_eq!(mode.as_str(), rendered);
        }
        assert_eq!(AnnFilteringMode::parse("bad"), None);

        let policy = AnnFilteringPolicy::default();
        assert_eq!(policy.routing_mode(0.01), AnnFilteringMode::PreFilter);
        assert_eq!(policy.routing_mode(0.20), AnnFilteringMode::Inline);
        assert_eq!(policy.routing_mode(0.90), AnnFilteringMode::PostFilter);
        assert_eq!(policy.effective_top_k_for_post_filter(5), 10);
        let forced = AnnFilteringPolicy {
            force_mode: Some(AnnFilteringMode::Inline),
            ..AnnFilteringPolicy::default()
        };
        assert_eq!(forced.routing_mode(0.0), AnnFilteringMode::Inline);

        let default_rto = RebuildRtoSpec::default();
        assert_eq!(
            default_rto.estimate_source,
            RebuildEstimateSource::Unverified
        );
        assert_eq!(
            RebuildRtoSpec::hnsw_benchmarked().estimate_source,
            RebuildEstimateSource::Benchmarked
        );
        assert!(!RebuildRtoSpec::csr_estimated().degraded_scan_during_rebuild);
        assert_eq!(
            RebuildRtoSpec::columnar_estimated().estimated_rebuild_seconds(10 * 1024 * 1024 * 1024),
            30.0
        );
        assert_eq!(RebuildEstimateSource::Benchmarked.as_str(), "benchmarked");
        assert_eq!(RebuildEstimateSource::Estimated.as_str(), "estimated");
        assert_eq!(RebuildEstimateSource::Unverified.as_str(), "unverified");

        assert_eq!(
            PropsEvaluationCadence::parse("auto"),
            Some(PropsEvaluationCadence::Compaction)
        );
        assert_eq!(
            PropsEvaluationCadence::parse("ddl"),
            Some(PropsEvaluationCadence::Explicit)
        );
        assert_eq!(PropsEvaluationCadence::parse("bad"), None);
        assert_eq!(PropsEvaluationCadence::Compaction.as_str(), "compaction");
        assert_eq!(PropsEvaluationCadence::Explicit.as_str(), "explicit");
        let document_policy = PropsAutoPromotionPolicy::document_default();
        assert!(document_policy.enabled);
        assert!(
            document_policy
                .eligible_types
                .contains(&ProximaType::TimestampTz(TimeUnit::Nanosecond))
        );

        let observability = ObservabilityCompressionHint::default();
        assert_eq!(observability.series_key_column, "series_key");
        assert_eq!(observability.timestamp_ns_column, "__proxima_created_at_ns");
        assert!(observability.sort_by_series_key);
        assert!(observability.xor_float_encoding);

        let notes = ProtocolTypeCoercionNote::standard_notes();
        assert!(
            notes
                .iter()
                .any(|note| note.protocol == "pgwire" && note.lossless)
        );
        assert!(std::ptr::eq(
            notes.as_ptr(),
            ProtocolTypeCoercionNote::standard_notes().as_ptr()
        ));

        for (input, expected, rendered) in [
            ("identity", PartitionTransform::Identity, "identity"),
            ("bucket[32]", PartitionTransform::Bucket(32), "bucket[32]"),
            ("bucket[x]", PartitionTransform::Bucket(16), "bucket[16]"),
            (
                "truncate[8]",
                PartitionTransform::Truncate(8),
                "truncate[8]",
            ),
            (
                "truncate[x]",
                PartitionTransform::Truncate(16),
                "truncate[16]",
            ),
            ("year", PartitionTransform::Year, "year"),
            ("month", PartitionTransform::Month, "month"),
            ("day", PartitionTransform::Day, "day"),
            ("hour", PartitionTransform::Hour, "hour"),
            ("void", PartitionTransform::Void, "void"),
            ("unknown", PartitionTransform::Identity, "identity"),
        ] {
            let parsed = PartitionTransform::parse_from_iceberg_format(input);
            assert_eq!(parsed, expected);
            assert_eq!(parsed.to_string(), rendered);
        }

        assert_eq!(TableIdentifier::parse("table").to_fqn(), "table");
        assert_eq!(
            TableIdentifier::parse("db.schema.table").to_string(),
            "db.schema.table"
        );
        let healthy = CatalogHealth::healthy(12);
        assert!(healthy.is_healthy);
        assert_eq!(healthy.latency_ms, 12);
        let unhealthy = CatalogHealth::unhealthy("down");
        assert!(!unhealthy.is_healthy);
        assert_eq!(unhealthy.error.as_deref(), Some("down"));
    }

    // ----- ADR-024: catalog data_type is now canonical ProximaType -----

    #[test]
    fn test_catalog_column_data_type_is_canonical_proxima_type() {
        // The legacy CatalogDataType → ProximaType bridge is gone; the column's
        // data_type IS already a ProximaType. Exercise the canonical surface.
        let all = [
            ProximaType::Boolean,
            ProximaType::Int64,
            ProximaType::Float64,
            ProximaType::String,
            ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
            ProximaType::TimestampTz(TimeUnit::Nanosecond),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0,
            },
        ];
        for ty in &all {
            // Every variant projects to Arrow + pgwire with no panic.
            let _ = ty.to_arrow_type();
            let _ = ty.pgwire_oid();
        }
    }

    #[test]
    fn test_catalog_timestamptz_pgwire_oid() {
        let oid = ProximaType::TimestampTz(TimeUnit::Nanosecond).pgwire_oid();
        assert_eq!(oid, 1184, "TimestampTz OID must be 1184");
    }

    #[test]
    fn test_catalog_uuid_pgwire_oid() {
        let oid = ProximaType::Uuid.pgwire_oid();
        assert_eq!(oid, 2950, "UUID OID must be 2950");
    }

    #[test]
    fn test_storage_layout_defaults_to_internal_authority() {
        let layout = CatalogStorageLayout::default();

        assert_eq!(layout.authority, CatalogAuthorityMode::InternalCanonical);
        assert_eq!(layout.layout_kind, CatalogStorageLayoutKind::RowRecord);
        assert_eq!(layout.physical_format, CatalogPhysicalFormat::ProximaBlock);
        assert_eq!(layout.write_mode, CatalogWriteMode::Mutable);
        assert!(layout.policy_enforced_in_proxima);
        assert!(layout.lossy_type_mappings.is_empty());
    }

    #[test]
    fn test_external_authoritative_layout_declares_boundary() {
        let layout = CatalogStorageLayout::external_authoritative(
            "iceberg_orders",
            CatalogPhysicalFormat::Iceberg,
            "s3://warehouse/orders",
        );

        assert_eq!(
            layout.authority,
            CatalogAuthorityMode::ExternalAuthoritative
        );
        assert_eq!(layout.layout_kind, CatalogStorageLayoutKind::ExternalTable);
        assert_eq!(layout.physical_format, CatalogPhysicalFormat::Iceberg);
        assert_eq!(layout.write_mode, CatalogWriteMode::ExternalRefresh);
        assert_eq!(layout.location.as_deref(), Some("s3://warehouse/orders"));
        assert_eq!(
            layout.snapshot_semantics.as_deref(),
            Some("external-latest")
        );
        assert!(!layout.policy_enforced_in_proxima);
    }

    #[test]
    fn test_projection_records_rebuild_source_and_freshness() {
        let projection = CatalogProjection::rebuildable(
            "orders_hnsw",
            CatalogProjectionKind::VectorAnn,
            "orders.primary",
        )
        .with_bounded_lag(5_000)
        .with_freshness_state(ProjectionFreshnessState::Stale)
        .with_lineage("wal:100-200", "commit:180")
        .with_policy_and_gate("engine-enforced", "hybrid-vector-smoke");

        assert_eq!(projection.kind, CatalogProjectionKind::VectorAnn);
        assert_eq!(projection.rebuild_source, "orders.primary");
        assert_eq!(projection.freshness, ProjectionFreshness::BoundedLag);
        assert_eq!(projection.freshness_state, ProjectionFreshnessState::Stale);
        assert_eq!(projection.max_lag_ms, Some(5_000));
        assert_eq!(projection.source_range.as_deref(), Some("wal:100-200"));
        assert_eq!(
            projection.last_included_position.as_deref(),
            Some("commit:180")
        );
        assert_eq!(
            projection.policy_boundary.as_deref(),
            Some("engine-enforced")
        );
        assert_eq!(
            projection.benchmark_gate.as_deref(),
            Some("hybrid-vector-smoke")
        );
        assert!(projection.rebuildable);
        assert!(!projection.lossy);
    }

    #[test]
    fn projection_tier_is_capability_tag_that_inherits_or_overrides() {
        // Default: no per-projection tier → inherits the namespace's pool class.
        let base = CatalogProjection::rebuildable(
            "orders_hnsw",
            CatalogProjectionKind::VectorAnn,
            "orders.primary",
        );
        assert_eq!(base.tier, None);
        assert_eq!(
            base.effective_tier(StoragePoolClass::Standard),
            StoragePoolClass::Standard
        );
        assert_eq!(
            base.effective_tier(StoragePoolClass::Pooled),
            StoragePoolClass::Pooled
        );

        // Explicit override wins over the namespace tier (e.g. hot ANN on premium).
        let hot = base.clone().with_tier(StoragePoolClass::Premium);
        assert_eq!(hot.tier, Some(StoragePoolClass::Premium));
        assert_eq!(
            hot.effective_tier(StoragePoolClass::Standard),
            StoragePoolClass::Premium
        );

        // Tier is a capability tag; physical placement still flows through `location`,
        // independent of the tier class.
        let placed = hot.with_location("s3://premium-bucket/idx/ann/");
        assert_eq!(
            placed.location.as_deref(),
            Some("s3://premium-bucket/idx/ann/")
        );
        assert_eq!(
            placed.effective_tier(StoragePoolClass::Standard),
            StoragePoolClass::Premium
        );

        // serde round-trips the tier; an absent tier stays `None` (serde default).
        let json = serde_json::to_string(&placed).unwrap();
        let back: CatalogProjection = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tier, Some(StoragePoolClass::Premium));
        let default_json = serde_json::to_string(&base).unwrap();
        let default_back: CatalogProjection = serde_json::from_str(&default_json).unwrap();
        assert_eq!(default_back.tier, None);
    }

    #[test]
    fn test_relational_capabilities_are_optional() {
        let empty = RelationalCapabilities::default();
        assert!(!empty.has_enforced_semantics());

        let with_pk = RelationalCapabilities {
            primary_key: vec!["id".to_string()],
            ..Default::default()
        };
        assert!(with_pk.has_enforced_semantics());
    }

    #[test]
    fn test_table_schema_persists_layout_projection_and_relational_metadata() {
        let layout = CatalogStorageLayout::internal("pax_hot", CatalogStorageLayoutKind::Pax);
        let projection =
            CatalogProjection::rebuildable("docs_text", CatalogProjectionKind::FullText, "primary");
        let capabilities = RelationalCapabilities {
            primary_key: vec!["id".to_string()],
            ..Default::default()
        };

        let schema = CatalogTableSchema::new("docs")
            .with_storage_layout(layout)
            .with_projection(projection)
            .with_relational_capabilities(capabilities);

        assert_eq!(schema.storage_layouts.len(), 2);
        assert_eq!(
            schema.storage_layouts[1].layout_kind,
            CatalogStorageLayoutKind::Pax
        );
        assert_eq!(schema.projections.len(), 1);
        assert_eq!(schema.projections[0].kind, CatalogProjectionKind::FullText);
        assert!(schema.relational_capabilities.has_enforced_semantics());
    }

    #[test]
    fn test_table_schema_persists_compression_stats_profiles() {
        let profile = CatalogCompressionStatsProfile::new(
            "bench_23/vector/base_xor",
            "VectorBaseXorEntropy",
            1_024,
            256,
            128,
            true,
        )
        .with_layout_name("pax_vector_spatial")
        .with_projection_id("embedding_exact")
        .with_decode_ns_per_value(42.0);

        let schema =
            CatalogTableSchema::new("vectors").with_compression_stats_profile(profile.clone());

        assert_eq!(profile.measured_ratio, 4.0);
        assert_eq!(profile.bytes_per_value(), 2.0);
        assert_eq!(schema.compression_stats_profiles.len(), 1);
        assert_eq!(
            schema.compression_stats_profiles[0]
                .projection_id
                .as_deref(),
            Some("embedding_exact")
        );

        let encoded = serde_json::to_string(&schema).unwrap();
        let decoded: CatalogTableSchema = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.compression_stats_profiles[0], profile);
    }

    #[test]
    fn catalog_branch_merge_policy_defaults_match_adr_012() {
        let policy = CatalogBranchMergePolicy::default();
        assert_eq!(
            policy.node_upsert,
            CatalogBranchMergeResolution::LastWriteWins
        );
        assert_eq!(policy.node_delete, CatalogBranchMergeResolution::DeleteWins);
        assert_eq!(
            policy.edge_upsert,
            CatalogBranchMergeResolution::LastWriteWins
        );
        assert_eq!(policy.edge_delete, CatalogBranchMergeResolution::DeleteWins);
        assert_eq!(
            policy.embedding_update,
            CatalogBranchMergeResolution::LastWriteWins
        );
        assert_eq!(
            policy.label_set,
            CatalogBranchMergeResolution::AddWinsSetUnion
        );
        assert_eq!(
            policy.props_key,
            CatalogBranchMergeResolution::LastWriteWinsPerKey
        );
    }

    #[test]
    fn catalog_table_schema_serde_back_compat_with_pre_branch_policy_json() {
        let legacy_json = serde_json::json!({
            "name": "legacy_graph",
            "columns": [],
            "primary_key": [],
            "indexes": [],
            "schema_version": 1,
            "properties": {},
            "location": null,
            "created_at_ms": 1700000000000_i64,
            "updated_at_ms": 1700000000000_i64,
        });
        let schema: CatalogTableSchema = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(
            schema.branch_merge_policy,
            CatalogBranchMergePolicy::adr_012_default()
        );
    }

    #[test]
    fn catalog_table_schema_round_trips_branch_merge_policy() {
        let mut policy = CatalogBranchMergePolicy::adr_012_default();
        policy
            .properties
            .insert("merge_endpoint".to_string(), "rest-v1".to_string());

        let schema = CatalogTableSchema::new("graph_docs").with_branch_merge_policy(policy.clone());
        let encoded = serde_json::to_string(&schema).unwrap();
        let decoded: CatalogTableSchema = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded.branch_merge_policy, policy);
        assert_eq!(
            decoded
                .branch_merge_policy
                .properties
                .get("merge_endpoint")
                .map(String::as_str),
            Some("rest-v1")
        );
    }

    #[test]
    fn catalog_table_statistics_is_stale_when_never_analyzed() {
        let stats = CatalogTableStatistics::default();
        assert!(
            stats.is_stale(1_000_000, 60_000),
            "stats with no last_analyzed_ms must be considered stale"
        );
    }

    #[test]
    fn catalog_table_statistics_is_fresh_within_ttl_window() {
        let stats = CatalogTableStatistics {
            row_count: 10,
            last_analyzed_ms: Some(950_000),
            ..Default::default()
        };
        // now=1_000_000, last=950_000, ttl=60_000 -> delta=50_000 < 60_000 -> fresh
        assert!(
            !stats.is_stale(1_000_000, 60_000),
            "stats within TTL must NOT be stale"
        );
    }

    #[test]
    fn catalog_table_statistics_is_stale_past_ttl_window() {
        let stats = CatalogTableStatistics {
            row_count: 10,
            last_analyzed_ms: Some(900_000),
            ..Default::default()
        };
        // now=1_000_000, last=900_000, ttl=60_000 -> delta=100_000 > 60_000 -> stale
        assert!(
            stats.is_stale(1_000_000, 60_000),
            "stats older than TTL must be stale"
        );
    }

    // === PR 6b: CatalogTableSchema precision fields ===

    #[test]
    fn catalog_table_schema_default_inherits_global_precision_policy() {
        let schema = CatalogTableSchema::default();
        assert!(schema.embedding_precision_policy_id.is_none());
        assert!(schema.embedding_precision_policy_version.is_none());
        assert_eq!(schema.current_precision_epoch, 0);
        assert_eq!(
            schema.canonical_embedding_precision,
            proximadb_records::EmbeddingScalarType::Fp32
        );
        assert!(schema.allowed_embedding_precisions.is_empty());
        let slo = schema.embedding_recall_slo;
        assert_eq!(slo.cosine.at_10, 0.99);
        assert_eq!(slo.dot.at_10, 0.995);
        assert!(schema.precision_migration_state.is_none());
    }

    #[test]
    fn catalog_table_schema_serde_back_compat_with_pre_pr6_json() {
        let pre_pr6_json = serde_json::json!({
            "name": "legacy_collection",
            "columns": [],
            "primary_key": [],
            "indexes": [],
            "schema_version": 1,
            "properties": {},
            "location": null,
            "created_at_ms": 1700000000000_i64,
            "updated_at_ms": 1700000000000_i64,
        });
        let schema: CatalogTableSchema = serde_json::from_value(pre_pr6_json).unwrap();
        assert_eq!(schema.name, "legacy_collection");
        assert!(schema.embedding_precision_policy_id.is_none());
        assert_eq!(schema.current_precision_epoch, 0);
        assert_eq!(
            schema.canonical_embedding_precision,
            proximadb_records::EmbeddingScalarType::Fp32
        );
    }

    // === ADR-024 Step 4: storage-side ProximaSchema absorbed into CatalogTableSchema ===

    /// DURABLE compat: a `CatalogTableSchema` persisted BEFORE the ADR-024
    /// Step-4 merge (i.e. without the absorbed storage-plane evolution fields
    /// `schema_id`/`version`/`parent_schema_id`/`fingerprint`/
    /// `created_at_ms_schema`/`is_legacy_vector_record`, and with columns that
    /// lack `is_deleted`/`original_id`) MUST still deserialize, with the new
    /// fields defaulting.
    #[test]
    fn catalog_table_schema_serde_back_compat_with_pre_merge_json() {
        let pre_merge_json = serde_json::json!({
            "name": "pre_merge_collection",
            "columns": [
                {
                    "id": 1,
                    "name": "id",
                    "data_type": "Int64",
                    "nullable": false,
                    "default_value": null,
                    "comment": null,
                    "properties": {}
                    // NOTE: no is_deleted / original_id
                }
            ],
            "primary_key": ["id"],
            "indexes": [],
            "schema_version": 1,
            "properties": {},
            "location": null,
            "created_at_ms": 1700000000000_i64,
            "updated_at_ms": 1700000000000_i64
            // NOTE: no schema_id / version / parent_schema_id / fingerprint /
            // created_at_ms_schema / is_legacy_vector_record
        });

        let schema: CatalogTableSchema = serde_json::from_value(pre_merge_json).unwrap();
        assert_eq!(schema.name, "pre_merge_collection");
        // Absorbed schema-level fields default.
        assert_eq!(schema.schema_id, "");
        assert_eq!(schema.version, 0);
        assert!(schema.parent_schema_id.is_none());
        assert_eq!(schema.fingerprint, 0);
        assert_eq!(schema.created_at_ms_schema, 0);
        assert!(!schema.is_legacy_vector_record);
        // Absorbed column-level fields default.
        assert_eq!(schema.columns.len(), 1);
        assert!(!schema.columns[0].is_deleted);
        assert!(schema.columns[0].original_id.is_none());
        // Ported helper still works against the deserialized schema.
        assert!(schema.column_by_name("id").is_some());
        assert_eq!(schema.primary_key_ids(), vec![1]);
    }

    // === Phase 0 (system-catalog redesign): typed vector/storage config ===

    /// All four typed fields survive a JSON serialize → deserialize round-trip
    /// losslessly, exercising the reused foundation enums (`DistanceMetric`,
    /// `QuantizationType`, `CompressionAlgorithm`) and the catalog-native config
    /// structs.
    #[test]
    fn catalog_table_schema_round_trips_typed_vector_storage_config() {
        let schema = CatalogTableSchema::new("typed_vec")
            .with_distance_metric(DistanceMetric::Cosine)
            .with_quantization(QuantizationType::Scalar)
            .with_embedding_config(CatalogEmbeddingConfig {
                model: "bge-small".to_string(),
                dimension: 384,
                native_precision: proximadb_records::EmbeddingScalarType::Fp32,
                normalize: true,
                ..Default::default()
            })
            .with_storage_config(CatalogStorageConfig {
                compression: Some(CompressionAlgorithm::Zstd),
                max_segment_size_mb: Some(128),
                enable_caching: Some(true),
            });

        let json = serde_json::to_string(&schema).unwrap();
        let back: CatalogTableSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(back.distance_metric, Some(DistanceMetric::Cosine));
        assert_eq!(back.quantization, Some(QuantizationType::Scalar));
        assert_eq!(
            back.embedding_config,
            Some(CatalogEmbeddingConfig {
                model: "bge-small".to_string(),
                dimension: 384,
                native_precision: proximadb_records::EmbeddingScalarType::Fp32,
                normalize: true,
                ..Default::default()
            })
        );
        assert_eq!(
            back.storage_config,
            Some(CatalogStorageConfig {
                compression: Some(CompressionAlgorithm::Zstd),
                max_segment_size_mb: Some(128),
                enable_caching: Some(true),
            })
        );
    }

    /// DURABLE compat: the four typed fields are omitted from JSON when unset
    /// (`skip_serializing_if`), and a catalog persisted BEFORE this change — i.e.
    /// JSON lacking the keys entirely — still deserializes, with each field
    /// defaulting to `None`.
    #[test]
    fn catalog_table_schema_omits_and_defaults_typed_fields_when_unset() {
        // Unset → omitted from serialized JSON.
        let schema = CatalogTableSchema::new("plain");
        let value = serde_json::to_value(&schema).unwrap();
        let obj = value.as_object().unwrap();
        assert!(!obj.contains_key("distance_metric"));
        assert!(!obj.contains_key("quantization"));
        assert!(!obj.contains_key("embedding_config"));
        assert!(!obj.contains_key("storage_config"));

        // Pre-Phase-0 JSON (no typed-config keys) still loads, fields default.
        let pre_phase0_json = serde_json::json!({
            "name": "pre_phase0_collection",
            "columns": [],
            "primary_key": [],
            "indexes": [],
            "schema_version": 1,
            "properties": {},
            "location": null,
            "created_at_ms": 1700000000000_i64,
            "updated_at_ms": 1700000000000_i64
        });
        let loaded: CatalogTableSchema = serde_json::from_value(pre_phase0_json).unwrap();
        assert_eq!(loaded.name, "pre_phase0_collection");
        assert!(loaded.distance_metric.is_none());
        assert!(loaded.quantization.is_none());
        assert!(loaded.embedding_config.is_none());
        assert!(loaded.storage_config.is_none());
    }

    /// The ported storage-plane constructors/helpers behave like the original
    /// `ProximaSchema` they replaced: legacy vector schema, by-id/by-name
    /// column lookup, tombstone-aware active count, and id<->name primary-key
    /// resolution.
    #[test]
    fn catalog_table_schema_storage_plane_helpers() {
        let schema = CatalogTableSchema::vector_record_schema(1536);
        assert!(schema.is_legacy_vector_record);
        assert_eq!(schema.version, 0);
        assert_eq!(schema.active_column_count(), 5);
        assert_eq!(schema.vector_dimension(), Some(1536));
        assert_eq!(schema.primary_key, vec!["id".to_string()]);
        assert_eq!(schema.primary_key_ids(), vec![1]);

        // from_columns resolves primary-key ids to names.
        let built = CatalogTableSchema::from_columns(
            "custom",
            vec![
                CatalogColumn::new(1, "pk", ProximaType::Int64).nullable(false),
                CatalogColumn::new(2, "val", ProximaType::String),
            ],
            vec![1],
        );
        assert_eq!(built.primary_key, vec!["pk".to_string()]);
        assert!(built.column_by_id(2).is_some());

        // fingerprint is stable for identical column sets, differs for distinct.
        let a = CatalogTableSchema::vector_record_schema(512);
        let b = CatalogTableSchema::vector_record_schema(512);
        let c = CatalogTableSchema::vector_record_schema(1024);
        assert_eq!(a.fingerprint, b.fingerprint);
        assert_ne!(a.fingerprint, c.fingerprint);
    }

    #[test]
    fn catalog_table_schema_serde_omits_none_policy_fields() {
        let schema = CatalogTableSchema::new("test");
        let json = serde_json::to_value(&schema).unwrap();
        assert!(json.get("embedding_precision_policy_id").is_none());
        assert!(json.get("embedding_precision_policy_version").is_none());
        assert!(json.get("precision_migration_state").is_none());
    }

    #[test]
    fn catalog_table_schema_round_trips_explicit_policy_reference() {
        let mut schema = CatalogTableSchema::new("fp16_collection");
        schema.embedding_precision_policy_id = Some("tenant_business_fp16".to_string());
        schema.embedding_precision_policy_version = Some(7);
        schema.current_precision_epoch = 3;
        schema.canonical_embedding_precision = proximadb_records::EmbeddingScalarType::Fp16;
        schema.allowed_embedding_precisions = vec![
            proximadb_records::EmbeddingScalarType::Fp16,
            proximadb_records::EmbeddingScalarType::Fp32,
        ];
        schema.precision_migration_state =
            Some(embedding_precision_policy::PrecisionMigrationState::ShadowingTarget);

        let json = serde_json::to_string(&schema).unwrap();
        let back: CatalogTableSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.embedding_precision_policy_id.as_deref(),
            Some("tenant_business_fp16")
        );
        assert_eq!(back.embedding_precision_policy_version, Some(7));
        assert_eq!(back.current_precision_epoch, 3);
        assert_eq!(
            back.canonical_embedding_precision,
            proximadb_records::EmbeddingScalarType::Fp16
        );
        assert_eq!(back.allowed_embedding_precisions.len(), 2);
        assert_eq!(
            back.precision_migration_state,
            Some(embedding_precision_policy::PrecisionMigrationState::ShadowingTarget)
        );
    }

    // ── Slice 5: primary_pod catalog field tests ────────────────────

    #[test]
    fn primary_pod_defaults_to_none_for_new_schema() {
        // Legacy schemas + new schemas alike must default to "no
        // binding". The gateway treats `None` as the legacy
        // unbounded case where writes can land on any pod.
        let schema = CatalogTableSchema::new("collection-x");
        assert!(schema.primary_pod.is_none());
    }

    #[test]
    fn with_primary_pod_sets_and_clears() {
        // Builder must support both setting a binding and clearing
        // it back to `None` — operators occasionally unbind to
        // return a collection to default routing during planned
        // maintenance.
        let bound = CatalogPrimaryPod::now("pod-a", CatalogPrimaryPodReason::Create);
        let schema = CatalogTableSchema::new("c").with_primary_pod(Some(bound.clone()));
        assert_eq!(schema.primary_pod.as_ref().unwrap().pod, "pod-a");

        let cleared = schema.with_primary_pod(None);
        assert!(cleared.primary_pod.is_none());
    }

    #[test]
    fn primary_pod_serde_roundtrip_preserves_all_fields() {
        let original = CatalogPrimaryPod {
            pod: "proximadb-write-3".to_string(),
            assigned_at_ms: 1_711_500_000_000,
            reason: CatalogPrimaryPodReason::Failover,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let back: CatalogPrimaryPod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, original);
    }

    #[test]
    fn primary_pod_absent_from_json_when_unset() {
        // `skip_serializing_if = "Option::is_none"` matters for both
        // backwards-compat (old readers ignore the field) and
        // human-readable JSON dumps (no `primary_pod: null` noise on
        // every collection).
        let schema = CatalogTableSchema::new("c");
        let json = serde_json::to_string(&schema).expect("serialize");
        assert!(
            !json.contains("primary_pod"),
            "absent primary_pod must not appear in JSON; got: {}",
            json
        );
    }

    #[test]
    fn legacy_schema_without_primary_pod_deserializes_to_none() {
        // Old catalog records (pre-Slice-5) must deserialize cleanly.
        // We construct a minimal JSON payload missing the field and
        // confirm `primary_pod` defaults to `None` via #[serde(default)].
        let minimal_json = r#"{
            "name": "legacy_collection",
            "columns": [],
            "primary_key": [],
            "indexes": [],
            "schema_version": 1,
            "properties": {},
            "location": null,
            "created_at_ms": 0,
            "updated_at_ms": 0
        }"#;
        let schema: CatalogTableSchema =
            serde_json::from_str(minimal_json).expect("legacy schema must deserialize");
        assert!(schema.primary_pod.is_none(), "missing field → None");
    }

    #[test]
    fn primary_pod_reason_labels_are_stable() {
        // Operators wire dashboards against these strings. Lock them
        // in here so a rename is caught at test time, not at
        // 3am-page time.
        assert_eq!(CatalogPrimaryPodReason::Create.label(), "create");
        assert_eq!(CatalogPrimaryPodReason::Operator.label(), "operator");
        assert_eq!(CatalogPrimaryPodReason::Failover.label(), "failover");
        assert_eq!(CatalogPrimaryPodReason::Rebalance.label(), "rebalance");
        assert_eq!(
            CatalogPrimaryPodReason::CatalogReplay.label(),
            "catalog_replay"
        );
    }

    #[test]
    fn primary_pod_now_uses_current_wall_clock() {
        // Sanity check that `now()` lands somewhere recent. Use a
        // generous lower bound (year 2024 = ~1.7e12 ms) to avoid
        // false failures from clock skew while still catching the
        // "we serialized a zero" regression.
        let p = CatalogPrimaryPod::now("pod-a", CatalogPrimaryPodReason::Operator);
        assert!(
            p.assigned_at_ms > 1_700_000_000_000,
            "assigned_at_ms must be recent wall-clock millis, got {}",
            p.assigned_at_ms
        );
        assert_eq!(p.pod, "pod-a");
        assert_eq!(p.reason, CatalogPrimaryPodReason::Operator);
    }

    #[test]
    fn primary_pod_reason_serde_uses_snake_case() {
        // REST payloads and catalog JSON both speak snake_case for
        // operator-facing enums. Lock in the wire format so a future
        // serde rename doesn't silently break dashboards.
        let json = serde_json::to_string(&CatalogPrimaryPodReason::Failover).unwrap();
        assert_eq!(json, "\"failover\"");
        let json = serde_json::to_string(&CatalogPrimaryPodReason::CatalogReplay).unwrap();
        assert_eq!(json, "\"catalog_replay\"");
    }
}

// ---------------------------------------------------------------------------
// TableIdentifier
// ---------------------------------------------------------------------------

/// Fully-qualified table address: namespace path + table name.
///
/// Mirrors the Iceberg REST catalog `{namespace}/{table}` addressing scheme
/// so that OLTP, lake, and Iceberg REST catalog backends share one type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableIdentifier {
    /// Namespace hierarchy (e.g., `["db", "schema"]`).
    pub namespace: Vec<String>,
    /// Unqualified table name.
    pub name: String,
}

impl TableIdentifier {
    pub fn new(namespace: Vec<String>, name: impl Into<String>) -> Self {
        Self {
            namespace,
            name: name.into(),
        }
    }

    /// Parse from a dot-separated fully-qualified name (e.g., `"db.schema.table"`).
    pub fn parse(s: &str) -> Self {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() == 1 {
            Self::new(vec![], parts[0])
        } else {
            let namespace = parts[..parts.len() - 1]
                .iter()
                .map(|p| p.to_string())
                .collect();
            Self::new(namespace, parts[parts.len() - 1])
        }
    }

    /// Dot-joined fully-qualified name.
    pub fn to_fqn(&self) -> String {
        if self.namespace.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.namespace.join("."), self.name)
        }
    }
}

impl std::fmt::Display for TableIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_fqn())
    }
}

// ---------------------------------------------------------------------------
// CatalogHealth
// ---------------------------------------------------------------------------

/// Catalog backend health and connectivity status.
#[derive(Debug, Clone)]
pub struct CatalogHealth {
    pub is_healthy: bool,
    pub latency_ms: u64,
    pub error: Option<String>,
    pub details: HashMap<String, String>,
}

impl CatalogHealth {
    pub fn healthy(latency_ms: u64) -> Self {
        Self {
            is_healthy: true,
            latency_ms,
            error: None,
            details: HashMap::new(),
        }
    }

    pub fn unhealthy(error: impl Into<String>) -> Self {
        Self {
            is_healthy: false,
            latency_ms: 0,
            error: Some(error.into()),
            details: HashMap::new(),
        }
    }

    /// Attach an additional key-value detail to the health status.
    /// Returns `self` for builder-style chaining. Ported from the
    /// local CatalogHealth as part of Option B consolidation.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Catalog trait
// ---------------------------------------------------------------------------

/// TD-CAT-7: the **identity half** of a catalog, split out so the contract is
/// *honourable*.
///
/// `Catalog` used to demand identity minting from HTTP shims (Unity, Polaris)
/// and external metastores (Hive, Iceberg, Glue) that structurally cannot
/// provide it — they own no allocator. That is precisely why every identity
/// method acquired a silent `Ok(None)` default, and why "a catalog that does
/// not mint identity" became a *representable state*: an implementation that
/// implemented nothing still compiled and answered successfully.
///
/// Identity is not bookkeeping — **authorization depends on it** (relational
/// ABAC gates on `stable_namespace_id` AND `object_id`). So the obligations
/// live here instead, where **every method is required**: a backend that claims
/// authority cannot forget one, because the compiler will not let it.
///
/// External/federated catalogs simply do not implement this trait and leave
/// [`Catalog::identity_authority`] at its `None` default — an honest "I am not
/// an identity authority", distinguishable from "I am, and I minted nothing".
#[async_trait::async_trait]
pub trait CatalogAuthority: Send + Sync {
    /// ADR-031 allocator unification: the highest persisted `object_id` across
    /// all catalog objects. `None` means *nothing has been minted yet* — the
    /// startup floor-raise treats that as "start at 1", which is only sound
    /// because it can no longer also mean "this backend has no allocator".
    async fn max_object_id(&self) -> anyhow::Result<Option<u64>>;

    /// Reserve the next `object_id` from this authority's sequence. An
    /// authority always can, so this returns `u64`, not `Option<u64>`.
    async fn allocate_object_id(&self) -> anyhow::Result<u64>;

    /// ADR-031 Phase 4c: pre-mint the typed identity triple
    /// `(account_u32, namespace_u16, collection_u32)` before storage-dir
    /// creation. `None` here retains its one legitimate meaning: **no account
    /// was supplied**. Idempotent with the later `create_table`.
    async fn mint_collection_typed_identity(
        &self,
        account: &str,
        namespace_key: &str,
    ) -> anyhow::Result<Option<(u32, u16, u32)>>;

    /// Lookup-or-mint the durable account stable id. `None` only for an
    /// empty/absent account.
    async fn account_id_u32(&self, account: &str) -> anyhow::Result<Option<u32>>;

    /// Sync, read-only account lookup for the request-hot resolver — no mint,
    /// no I/O. `None` when unminted (fail-closed deny).
    fn account_id_u32_lookup(&self, account: &str) -> Option<u32>;

    /// Resolve a table by its stable `object_id` — the inverse of
    /// `get_table(...).object_id`. `None` when no table carries that id.
    async fn get_table_by_object_id(
        &self,
        object_id: u64,
    ) -> anyhow::Result<Option<TableIdentifier>>;

    /// Namespace analogue of [`Self::get_table_by_object_id`].
    async fn get_namespace_by_object_id(
        &self,
        object_id: u64,
    ) -> anyhow::Result<Option<Vec<String>>>;
}

/// Core catalog trait — every ProximaDB catalog backend implements this.
///
/// The trait uses only types from this crate (`proximadb-catalog`) so
/// implementations can live in any workspace crate without depending on the
/// root `proximadb` crate.
#[async_trait::async_trait]
pub trait Catalog: Send + Sync {
    fn name(&self) -> &str;
    fn catalog_type(&self) -> &str;

    /// TD-CAT-7 capability probe: the identity authority behind this catalog,
    /// or `None` when it has none (external/federated metastore).
    ///
    /// This is the distinction whose absence let identity vanish silently — a
    /// caller could not tell `None because this backend cannot mint` from
    /// `None because the backend that can, didn't`. Every identity default
    /// below now delegates through here, so an authority answers with its own
    /// implementation or not at all.
    fn identity_authority(&self) -> Option<&dyn CatalogAuthority> {
        None
    }

    // Namespace operations
    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
    ) -> anyhow::Result<CatalogNamespace>;

    /// Create a namespace owned by `tenant` (TD-064/TD-113). Backends that own
    /// ProximaDB physical paths record `tenant_id` so the namespace is
    /// DR-addressable (`is_dr_addressable`) and the warehouse path resolver can
    /// assert tenant ownership / route by storage pool. The default implementation
    /// ignores `tenant` and delegates to [`create_namespace`](Self::create_namespace),
    /// so external/federated catalogs (which manage their own identity) are
    /// unaffected.
    async fn create_namespace_for_tenant(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
        _tenant: Option<&str>,
    ) -> anyhow::Result<CatalogNamespace> {
        self.create_namespace(namespace, properties).await
    }

    async fn drop_namespace(&self, namespace: &[String], cascade: bool) -> anyhow::Result<bool>;
    async fn list_namespaces(
        &self,
        parent: Option<&[String]>,
    ) -> anyhow::Result<Vec<CatalogNamespace>>;
    async fn namespace_exists(&self, namespace: &[String]) -> anyhow::Result<bool>;
    async fn get_namespace(&self, namespace: &[String]) -> anyhow::Result<CatalogNamespace>;
    /// ADR-031 Phase 4b: resolve the numeric account u32 for an account string
    /// (lookup in the durable account registry; mints+persists on first sight).
    /// Returns `None` for an empty/absent account. The root path-resolver uses
    /// this to compose a `CollectionIdentity` for the typed object-store path.
    /// Default `None` — only the native catalog mints; federated/external
    /// catalogs have no typed identity (legacy paths).
    async fn account_id_u32(&self, account: &str) -> anyhow::Result<Option<u32>> {
        match self.identity_authority() {
            Some(authority) => authority.account_id_u32(account).await,
            None => Ok(None),
        }
    }
    /// TD-TENANT-1 item 3: SYNC lookup of the account u32 — no mint, no I/O.
    /// For the request-hot `TenantStableIdResolver` (which is sync). Returns the
    /// already-minted u32 for an account/tenant string, or `None` when unminted
    /// (fail-closed deny). Default `None` — only the native catalog has a
    /// registry; federated/external catalogs stay `None` (legacy-safe).
    fn account_id_u32_lookup(&self, account: &str) -> Option<u32> {
        self.identity_authority()
            .and_then(|authority| authority.account_id_u32_lookup(account))
    }
    /// ADR-031 allocator unification: the highest persisted `object_id` across
    /// all tables (from the durable `object_name_index`). The root startup path
    /// uses this to raise the collection-id allocator floor above every existing
    /// object_id, so a freshly minted collection id can never collide with a
    /// legacy (pre-unification) `schema.object_id` → no oid-index corruption.
    /// Default `None` (federated catalogs have no native object_ids).
    async fn max_object_id(&self) -> anyhow::Result<Option<u64>> {
        match self.identity_authority() {
            Some(authority) => authority.max_object_id().await,
            None => Ok(None),
        }
    }
    /// Reserve the next `object_id` from this catalog's authoritative sequence.
    ///
    /// Lifecycle services that must expose an id before creating the catalog
    /// object use this instead of maintaining a second allocator. The later
    /// `create_table` call adopts the reserved id and raises (but does not
    /// advance past) the same sequence. `None` means this catalog has no native
    /// object-id authority; callers may use a compatibility allocator.
    async fn allocate_object_id(&self) -> anyhow::Result<Option<u64>> {
        match self.identity_authority() {
            Some(authority) => authority.allocate_object_id().await.map(Some),
            None => Ok(None),
        }
    }
    /// ADR-031 Phase 4c: pre-mint the typed identity triple
    /// `(account_u32, namespace_u16, collection_u32)` for a collection being
    /// created under `account`/`namespace_key`, BEFORE storage-dir creation.
    /// The root composes a `CollectionIdentity` from it for the typed DATA path.
    /// `None` when no account is known (legacy, mixed-safe). Idempotent with the
    /// later `create_table`→`mint_stable_identity` (preserves pre-stamped values
    /// via the shared `resolve_typed_triple`). Default `None`.
    async fn mint_collection_typed_identity(
        &self,
        account: &str,
        namespace_key: &str,
    ) -> anyhow::Result<Option<(u32, u16, u32)>> {
        match self.identity_authority() {
            Some(authority) => {
                authority
                    .mint_collection_typed_identity(account, namespace_key)
                    .await
            }
            None => Ok(None),
        }
    }
    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        removals: Vec<String>,
    ) -> anyhow::Result<()>;

    // Table operations
    /// Backends implement table creation **here**; [`Self::create_table`] is the
    /// sealed wrapper callers use.
    ///
    /// TD-CAT-7: minting must be *unforgettable*, not merely implemented. The
    /// gap was hand-fixed on one backend at a time three times over, and each
    /// fix left the next instance alive — so the post-condition lives in one
    /// place every backend necessarily passes through. Overriding
    /// `create_table` would route around it; `scripts/check_catalog_seal.py`
    /// (in `make work-commit-check`) fails the build if any impl does.
    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> anyhow::Result<CatalogTableSchema>;

    /// Create a table, then enforce the identity post-condition.
    ///
    /// An identity **authority** that returns a table with no `object_id` has
    /// silently produced a row that authorization cannot key on. That must fail
    /// loudly at the write, not surface later as an empty introspection column
    /// or a policy that matches nothing. Non-authorities (federated catalogs)
    /// are exempt — they never claimed to mint.
    ///
    /// This is a **post**-condition, and deliberately not a rollback: the id is
    /// minted inside `create_table_inner`, so there is nothing to check before
    /// the write. If it fires, the row may already be persisted and the caller
    /// still sees an error — a visible, operator-recoverable inconsistency,
    /// which is strictly better than the silent one it replaces. For a correct
    /// authority it is unreachable.
    ///
    /// **Do not override.**
    async fn create_table(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> anyhow::Result<CatalogTableSchema> {
        let created = self.create_table_inner(identifier, schema).await?;
        if self.identity_authority().is_some() && created.object_id.is_none() {
            anyhow::bail!(
                "TD-CAT-7: catalog '{}' ({}) is an identity authority but created table {:?} \
                 with no object_id; authorization keys on it, so the row must not be committed \
                 without one",
                self.name(),
                self.catalog_type(),
                identifier,
            );
        }
        Ok(created)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, purge: bool) -> anyhow::Result<bool>;
    async fn list_tables(&self, namespace: &[String]) -> anyhow::Result<Vec<TableIdentifier>>;
    async fn table_exists(&self, identifier: &TableIdentifier) -> anyhow::Result<bool>;
    async fn get_table(&self, identifier: &TableIdentifier) -> anyhow::Result<CatalogTableSchema>;

    /// TD-110 S3: enumerate every table in a namespace subtree (the `scope`),
    /// for cross-namespace FOREIGN KEY child discovery on ON DELETE.
    /// `scope = None` → all top-level namespaces; `scope = Some([tenant])` → the
    /// tenant's subtree (keeps FK child discovery intra-tenant — cross-tenant
    /// children are never enumerated). Default impl BFS-walks `list_namespaces`
    /// + `list_tables`; correct for every backend (a backend-specific global
    /// index override is possible but would miss legacy name-keyed tables, so
    /// the recursive default is the complete path).
    async fn list_all_tables_in_scope(
        &self,
        scope: Option<&[String]>,
    ) -> anyhow::Result<Vec<TableIdentifier>> {
        let mut tables = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut frontier: Vec<Vec<String>> = match scope {
            Some(ns) => vec![ns.to_vec()],
            None => self
                .list_namespaces(None)
                .await?
                .into_iter()
                .map(|ns| ns.levels)
                .collect(),
        };
        while let Some(ns) = frontier.pop() {
            if !visited.insert(ns.clone()) {
                continue;
            }
            tables.extend(self.list_tables(&ns).await?);
            for child in self.list_namespaces(Some(&ns)).await? {
                frontier.push(child.levels);
            }
        }
        Ok(tables)
    }

    /// ADR-031 O1 (dual-read): resolve a table by its stable `object_id` — the
    /// inverse of `get_table(...).object_id`. Returns `None` when no table carries
    /// that id, or when the backend does not allocate object_ids (external
    /// catalogs). Lets `dyn Catalog` consumers (change-feed, recovery, planner)
    /// key on the global id rather than the mutable name. Default: `None`.
    async fn get_table_by_object_id(
        &self,
        object_id: u64,
    ) -> anyhow::Result<Option<TableIdentifier>> {
        match self.identity_authority() {
            Some(authority) => authority.get_table_by_object_id(object_id).await,
            None => Ok(None),
        }
    }

    /// Resolve a namespace's levels from its stable catalog `object_id`
    /// (ADR-031 / TD-181) — the namespace analogue of `get_table_by_object_id`.
    /// Default `Ok(None)` for external/federated catalogs that don't mint
    /// ProximaDB object_ids.
    async fn get_namespace_by_object_id(
        &self,
        object_id: u64,
    ) -> anyhow::Result<Option<Vec<String>>> {
        match self.identity_authority() {
            Some(authority) => authority.get_namespace_by_object_id(object_id).await,
            None => Ok(None),
        }
    }

    /// Resolve a collection's pinned embedding binding to one immutable,
    /// policy-checked xCatalog snapshot.
    ///
    /// Collection creation, embedding workers, lifecycle APIs, and external
    /// compatibility adapters share this query seam. It rejects aliases and
    /// verifies the legacy route name too, so mixed-version readers cannot
    /// execute a different model.
    async fn resolve_embedding_model_binding(
        &self,
        binding: &CatalogEmbeddingConfig,
        policy: &mlops::CatalogModelUsePolicy,
    ) -> anyhow::Result<mlops::CatalogResolvedEmbeddingModel> {
        binding.validate_model_binding()?;
        let asset_id = binding
            .model_asset_id
            .ok_or_else(|| anyhow::anyhow!("embedding model binding is not pinned to an asset"))?;
        let version = binding
            .model_version
            .ok_or_else(|| anyhow::anyhow!("embedding model binding is not pinned to a version"))?;
        let contract_sha256 = binding.contract_sha256.as_deref().ok_or_else(|| {
            anyhow::anyhow!("embedding model binding is not pinned to a contract digest")
        })?;
        let identifier = self
            .get_table_by_object_id(asset_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("embedding model asset {asset_id} was not found"))?;
        let schema = self.get_table(&identifier).await?;
        let asset = schema.mlops_asset_as_typed()?.ok_or_else(|| {
            anyhow::anyhow!("catalog object {asset_id} is not an embedding model asset")
        })?;
        let mlops::CatalogMlopsAsset::EmbeddingModel(registry) = asset;
        if registry.name != binding.model {
            return Err(anyhow::anyhow!(
                "embedding route '{}' does not match registered model '{}' for asset {}",
                binding.model,
                registry.name,
                asset_id
            ));
        }
        let model = registry
            .resolve_use(version, contract_sha256, binding.dimension, policy)
            .map_err(anyhow::Error::new)?
            .clone();
        Ok(mlops::CatalogResolvedEmbeddingModel {
            asset_id,
            registry_name: registry.name,
            registry_revision: registry.revision,
            contract_sha256: contract_sha256.to_string(),
            model,
        })
    }

    async fn rename_table(
        &self,
        from: &TableIdentifier,
        to: &TableIdentifier,
    ) -> anyhow::Result<()>;

    // Schema evolution
    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> anyhow::Result<CatalogTableSchema>;

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> anyhow::Result<i32>;
    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        version: i32,
    ) -> anyhow::Result<CatalogTableSchema>;

    // Index operations
    async fn create_index(
        &self,
        identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> anyhow::Result<CatalogIndex>;

    async fn drop_index(
        &self,
        identifier: &TableIdentifier,
        index_name: &str,
    ) -> anyhow::Result<bool>;
    async fn list_indexes(&self, identifier: &TableIdentifier)
    -> anyhow::Result<Vec<CatalogIndex>>;

    // Statistics
    async fn get_statistics(
        &self,
        identifier: &TableIdentifier,
    ) -> anyhow::Result<CatalogTableStatistics>;
    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> anyhow::Result<()>;

    // Partitioning (default: not supported)
    async fn get_partition_spec(
        &self,
        identifier: &TableIdentifier,
    ) -> anyhow::Result<Option<CatalogPartitionSpec>> {
        let _ = identifier;
        Ok(None)
    }

    async fn update_partition_spec(
        &self,
        identifier: &TableIdentifier,
        spec: CatalogPartitionSpec,
    ) -> anyhow::Result<()> {
        let _ = (identifier, spec);
        Err(anyhow::anyhow!(
            "partitioning not supported by this catalog"
        ))
    }

    // Sort Order (for Iceberg-compatible catalogs; default: not supported)
    async fn get_sort_order(
        &self,
        identifier: &TableIdentifier,
    ) -> anyhow::Result<Option<CatalogSortOrder>> {
        let _ = identifier;
        Ok(None)
    }

    async fn update_sort_order(
        &self,
        identifier: &TableIdentifier,
        order: CatalogSortOrder,
    ) -> anyhow::Result<()> {
        let _ = (identifier, order);
        Err(anyhow::anyhow!(
            "sort order updates not supported by this catalog"
        ))
    }

    // Primary-pod binding (Slice 5b.1 of tenant-pod affinity)
    //
    // Records which pod is the write authority for a (tenant, collection)
    // pair. `Some(_)` binds; `None` clears. Default impl rejects so that
    // any catalog backend not aware of primary_pod bookkeeping surfaces
    // a clear "not supported" rather than silently dropping the field.
    // NativeCatalog (the SharedServices default) overrides this; the
    // lakehouse backends opt in as their write-side affinity story
    // matures.
    async fn set_primary_pod(
        &self,
        identifier: &TableIdentifier,
        primary: Option<CatalogPrimaryPod>,
    ) -> anyhow::Result<()> {
        let _ = (identifier, primary);
        Err(anyhow::anyhow!(
            "primary_pod binding not supported by this catalog"
        ))
    }

    /// Replace the table's `storage_layouts` (the materialization / publication
    /// descriptor set) and persist the change, returning the updated schema.
    ///
    /// This is the catalog side of warehouse materialization: after a table's
    /// rows are published as a Parquet snapshot to object storage, the caller
    /// records a `Parquet` + published-authority layout (with the snapshot
    /// `location`) here, so the OLAP router's `catalog_table_is_parquet_backed`
    /// check passes and SELECTs over the table route to DataFusion. It is a
    /// physical/publication attribute (like [`set_primary_pod`]), not a logical
    /// schema evolution, so it does not bump `schema_version`.
    ///
    /// Default: unsupported (backends that don't own native table metadata).
    ///
    /// [`set_primary_pod`]: Catalog::set_primary_pod
    async fn set_storage_layouts(
        &self,
        identifier: &TableIdentifier,
        layouts: Vec<CatalogStorageLayout>,
    ) -> anyhow::Result<CatalogTableSchema> {
        let _ = (identifier, layouts);
        Err(anyhow::anyhow!(
            "storage layout mutation not supported by this catalog"
        ))
    }

    /// Atomically apply one command to a typed `mlops.*` model-registry asset.
    /// Command-shaped mutation prevents compatibility adapters from replacing
    /// immutable versions or append-only evidence with stale document writes.
    async fn apply_model_registry_mutation(
        &self,
        identifier: &TableIdentifier,
        expected_revision: u64,
        mutation: mlops::CatalogModelRegistryMutation,
    ) -> anyhow::Result<CatalogTableSchema> {
        let _ = (identifier, expected_revision, mutation);
        Err(anyhow::anyhow!(
            "model registry mutation not supported by this catalog"
        ))
    }

    // Note: cache() / invalidate_cache were on the legacy local Catalog
    // trait, but had zero external trait-method callers (CatalogManager::cache()
    // is a separate struct method). Not ported — implementors that want
    // a cache accessor can expose an inherent method instead.

    async fn health_check(&self) -> anyhow::Result<CatalogHealth> {
        Ok(CatalogHealth::healthy(0))
    }

    async fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

// =============================================================================
// LakehouseExtension — table-format-specific operations for lakehouse catalogs
// (Iceberg / Delta / Hudi / Polaris). Ported from the local trait at
// src/catalog/traits.rs as part of Option B consolidation.
// =============================================================================

/// Extension trait for lakehouse table format operations.
///
/// Implemented by `IcebergCatalog`, `DeltaCatalog`, `PolarisCatalog`
/// — anything that exposes snapshot / location / schema-history
/// semantics that are specific to open table formats.
#[async_trait::async_trait]
pub trait LakehouseExtension: Catalog {
    /// Get the table format (Iceberg / Delta / Hudi / etc.).
    fn table_format(&self) -> TableFormat;

    /// Get the table's storage location URI.
    async fn get_table_location(&self, identifier: &TableIdentifier) -> anyhow::Result<String>;

    /// Get the current snapshot id (Iceberg) / version (Delta).
    async fn get_current_snapshot(
        &self,
        identifier: &TableIdentifier,
    ) -> anyhow::Result<Option<i64>>;

    /// List all snapshot / version ids in chronological order.
    async fn list_snapshots(&self, identifier: &TableIdentifier) -> anyhow::Result<Vec<i64>>;

    /// Get the table's schema-version history (list of schema version ids).
    async fn get_schema_history(&self, identifier: &TableIdentifier) -> anyhow::Result<Vec<i32>>;
}

/// Table format for lakehouse tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableFormat {
    /// Native ProximaDB table format.
    ProximaDB,
    /// Apache Iceberg open table format.
    Iceberg,
    /// Linux Foundation Delta Lake format.
    Delta,
    /// Apache Hudi transactional data lake format.
    Hudi,
    /// Raw Apache Parquet files (read-only, no transaction log).
    Parquet,
}

impl std::fmt::Display for TableFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableFormat::ProximaDB => write!(f, "proximadb"),
            TableFormat::Iceberg => write!(f, "iceberg"),
            TableFormat::Delta => write!(f, "delta"),
            TableFormat::Hudi => write!(f, "hudi"),
            TableFormat::Parquet => write!(f, "parquet"),
        }
    }
}
