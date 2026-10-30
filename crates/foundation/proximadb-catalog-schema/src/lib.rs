//! Foundation-layer xCatalog schema contract types.
//!
//! These serializable schema contracts are shared by catalog implementations,
//! query planning, storage selection, and protocol/API boundaries WITHOUT
//! requiring the control-layer `proximadb-catalog` crate. Extracted from
//! `proximadb-catalog` (TD-DECOMP ratchet) so `proximadb-storage-common` can
//! depend on the schema types without a storage -> control upward edge.
//!
//! The control-layer catalog crate re-exports everything here, so historical
//! `proximadb_catalog::*` import paths keep working.

use anyhow::anyhow;
use proximadb_compression_types::CompressionAlgorithm;
use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};
use proximadb_distance_types::DistanceMetric;
use proximadb_quantization_types::QuantizationType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Re-export the canonical object-store bridge contract so storage-common can
/// consume it from this foundation crate (no storage -> control edge).
pub mod object_store_bridge;

/// Storage pool class for a namespace's bytes. The path resolver routes
/// writes to the matching bucket/container and refuses cross-class writes.
/// Tier-to-pool mapping is operator policy and lives in the operator layer;
/// the engine only knows the enum.
///
/// See `docs/12-design/COLLECTION_DR_CRR_ENGINE_CONTRACT.adoc` "Storage Pool
/// Classes".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StoragePoolClass {
    /// Shared pool — lowest operational overhead, no per-collection CRR.
    /// Default for legacy rows backfilled by the P0.5 migration.
    #[default]
    Pooled,
    /// Shared standard pool — prefix-scoped CRR allowed. (Commercial tier names
    /// are an operator/control-plane concern; the OSS engine uses neutral
    /// capability classes and reads legacy wire values via serde aliases.)
    #[serde(alias = "business")]
    Standard,
    /// Shared premium pool — stricter KMS, monitoring, and rule budgeting.
    #[serde(alias = "enterprise")]
    Premium,
    /// Dedicated bucket/storage-account pair per tenant per region pair.
    #[serde(alias = "enterprise_dedicated")]
    Dedicated,
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogColumn {
    /// Column ID — the **physical** field identity: stable across renames and
    /// mapped 1:1 to the Iceberg/Parquet field-id (ADR-010). This is the on-disk
    /// column mapping, NOT the catalog surrogate; see `object_id`.
    pub id: i32,
    /// ADR-031 / TD-181: the **catalog** surrogate identity — a stable, immutable
    /// `u64` `object_id` from the one system-wide catalog sequence (globally
    /// unique, never reused). Distinct role from `id` (the physical field-id):
    /// `object_id` is what catalog→catalog references and the path migration key
    /// on; `id` is the Parquet/PAX field mapping. Both coexist by role (ADR-031
    /// reconciliation amendment 4). Additive + `#[serde(default)]`, so legacy
    /// rows and not-yet-persisted columns load as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_id: Option<u64>,
    /// Column name
    pub name: String,
    /// Data type (canonical logical type — ADR-024).
    ///
    /// Serializes in the [`proximadb_data_model::ProximaType`] form, but
    /// deserializes from BOTH the new form AND the legacy `CatalogDataType`
    /// form (bare unit-string tags like `"Int64"`/`"Decimal"`/`"Vector"`) so
    /// catalogs persisted before ADR-024 keep loading. See
    /// [`deserialize_data_type_compat`].
    #[serde(deserialize_with = "deserialize_data_type_compat")]
    pub data_type: ProximaType,
    /// Is nullable
    pub nullable: bool,
    /// Default value (SQL expression)
    pub default_value: Option<String>,
    /// Column comment
    pub comment: Option<String>,
    /// Column metadata/properties (canonical metadata map — ADR-024 Step 4
    /// absorbed the storage-side `ProximaColumn.metadata` into this field).
    pub properties: HashMap<String, String>,

    // === ADR-024 Step 4: absorbed from storage-side `ProximaColumn` ===
    /// Tombstone flag (soft delete for schema evolution). Additive; legacy
    /// persisted catalogs (without this field) deserialize as `false`.
    #[serde(default)]
    pub is_deleted: bool,
    /// Original field ID (for tracking renames). Additive; legacy persisted
    /// catalogs (without this field) deserialize as `None`.
    #[serde(default)]
    pub original_id: Option<i32>,
}

impl CatalogColumn {
    /// Create a new column
    pub fn new(id: i32, name: impl Into<String>, data_type: ProximaType) -> Self {
        Self {
            id,
            object_id: None,
            name: name.into(),
            data_type,
            nullable: true,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
    }

    /// Set the stable `u64` catalog object identity (ADR-031 / TD-181).
    pub fn with_object_id(mut self, object_id: u64) -> Self {
        self.object_id = Some(object_id);
        self
    }

    /// Set nullable
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }

    /// Set comment
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// True when this column is active (not tombstoned). Ported from the
    /// storage-side `ProximaColumn` (ADR-024 Step 4).
    pub fn is_active(&self) -> bool {
        !self.is_deleted
    }

    /// Convert to an Arrow [`Field`]. Uses the canonical storage-plane Arrow
    /// projection ([`ProximaType::to_arrow_type`]) — matching the storage-side
    /// `ProximaColumn::to_arrow_field` this absorbed (ADR-024 Step 4), where a
    /// dense vector is a fixed-width `FixedSizeBinary` carrier. This is the
    /// projection the data-plane converter (`proxima_arrow`) writes against, so
    /// schema and data agree. (The dimensionless-catalog [`catalog_arrow_type`]
    /// projection remains available for catalog-only Arrow exports.) The stable
    /// column id and comment are carried in the field metadata.
    pub fn to_arrow_field(&self) -> arrow_schema::Field {
        let arrow_type = self.data_type.to_arrow_type();
        let mut field = arrow_schema::Field::new(&self.name, arrow_type, self.nullable);
        let mut meta = HashMap::new();
        meta.insert("proxima_column_id".to_string(), self.id.to_string());
        if let Some(ref comment) = self.comment {
            meta.insert("comment".to_string(), comment.clone());
        }
        field = field.with_metadata(meta);
        field
    }

    /// Create a column from an Arrow [`Field`] with an explicit stable id.
    /// Ported from the storage-side `ProximaColumn::from_arrow_field`.
    pub fn from_arrow_field(field: &arrow_schema::Field, id: i32) -> Self {
        Self {
            id,
            object_id: None,
            name: field.name().clone(),
            data_type: ProximaType::from_arrow_type(field.data_type()),
            nullable: field.is_nullable(),
            default_value: None,
            comment: field.metadata().get("comment").cloned(),
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
    }
}

/// Map a legacy `CatalogDataType` bare-string tag to its canonical
/// [`ProximaType`] (ADR-024). This is the SAME mapping the deleted
/// `CatalogDataType::to_proxima_type` used; dimensionless vectors keep
/// `dim: 0` (the real dimension lives in column properties / collection
/// config). Returns `None` for tags that are not a legacy unit variant
/// (the caller then falls back to the canonical `ProximaType` form).
fn legacy_catalog_data_type_tag(tag: &str) -> Option<ProximaType> {
    let ty = match tag {
        "Boolean" => ProximaType::Boolean,
        "Int8" => ProximaType::Int8,
        "Int16" => ProximaType::Int16,
        "Int32" => ProximaType::Int32,
        "Int64" => ProximaType::Int64,
        "Float32" => ProximaType::Float32,
        "Float64" => ProximaType::Float64,
        "String" => ProximaType::String,
        "Binary" => ProximaType::Binary,
        "Date" => ProximaType::Date,
        "Time" => ProximaType::Time(TimeUnit::Nanosecond),
        "Timestamp" => ProximaType::Timestamp(TimeUnit::Nanosecond),
        "TimestampTz" => ProximaType::TimestampTz(TimeUnit::Nanosecond),
        "Decimal" => ProximaType::Decimal {
            precision: 38,
            scale: 10,
        },
        "Uuid" => ProximaType::Uuid,
        "Json" => ProximaType::Json,
        "Vector" => ProximaType::DenseVector {
            element: VectorElement::Float32,
            dim: 0,
        },
        "SparseVector" => ProximaType::SparseVector {
            element: VectorElement::Float32,
        },
        "BinaryVector" => ProximaType::BinaryVector { dim: 0 },
        _ => return None,
    };
    Some(ty)
}

/// Deserialize [`CatalogColumn::data_type`] accepting BOTH the canonical
/// [`ProximaType`] form (current) AND the legacy `CatalogDataType` form
/// (pre-ADR-024 persisted catalogs). The legacy form encodes every variant as
/// a bare string tag (`"Int64"`, `"Decimal"`, `"Vector"`, …); the canonical
/// form encodes unit variants as bare strings too but struct variants as
/// objects (`{"Decimal":{"precision":38,"scale":10}}`,
/// `{"DenseVector":{...}}`). We therefore: (1) deserialize into an untyped
/// `serde_json::Value`; (2) if it is a bare string that names a legacy unit
/// variant, map it via [`legacy_catalog_data_type_tag`]; (3) otherwise parse it
/// as a canonical [`ProximaType`].
fn deserialize_data_type_compat<'de, D>(deserializer: D) -> Result<ProximaType, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let value = serde_json::Value::deserialize(deserializer)?;
    if let serde_json::Value::String(tag) = &value
        && let Some(ty) = legacy_catalog_data_type_tag(tag)
    {
        return Ok(ty);
    }
    serde_json::from_value(value).map_err(D::Error::custom)
}

/// Arrow projection for a **catalog** column type (ADR-024 Step 3).
///
/// The catalog stores vectors dimensionlessly (`DenseVector`/`BinaryVector` with
/// `dim: 0` — the real dimension lives in column properties / collection config)
/// and represents UUIDs logically. This preserves the catalog's historical Arrow
/// carriers for those types — variable-length `List<Float32>` for dense vectors,
/// `Binary` for packed binary vectors, `Utf8` for UUID — rather than the
/// storage-plane fixed-width layout from [`ProximaType::to_arrow_type`], which
/// would emit a zero-length `FixedSizeBinary(0)` for a dimensionless catalog
/// vector. All other types delegate to the canonical [`ProximaType::to_arrow_type`]
/// (matching the deleted `CatalogDataType::to_arrow_datatype` for those).
pub fn catalog_arrow_type(ty: &ProximaType) -> arrow_schema::DataType {
    use arrow_schema::{DataType, Field};
    match ty {
        ProximaType::DenseVector { .. } => DataType::List(std::sync::Arc::new(Field::new(
            "item",
            DataType::Float32,
            true,
        ))),
        ProximaType::BinaryVector { .. } => DataType::Binary,
        ProximaType::Uuid => DataType::Utf8,
        other => other.to_arrow_type(),
    }
}

/// Catalog-authoritative embedding descriptor for a vector table.
///
/// The system catalog is the system-of-record for *which* model vectorizes a
/// table and the geometry of its output, but it deliberately does NOT depend on
/// the modality engine (`proximadb-embedding` is a `modality` crate, above the
/// control layer). The embedding engine maps this descriptor to its own route
/// type. Precision *policy* (allowed/canonical precisions, recall SLO) is carried
/// separately in the dedicated precision-policy fields; this captures only model
/// identity + output geometry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEmbeddingConfig {
    /// Model/route identifier, opaque to the catalog and resolved by the
    /// embedding engine, e.g. `"bge-small"`, `"openai:text-embedding-3-small"`,
    /// or `"byo:https://…"`.
    pub model: String,
    /// Stable xCatalog object ID of the registered `mlops.*` model. `None`
    /// preserves the legacy opaque route-name behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_asset_id: Option<u64>,
    /// Immutable registered-model version selected for this collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<u64>,
    /// Executable contract digest pinned with `model_version`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_sha256: Option<String>,
    /// Output dimensionality of the model.
    pub dimension: u32,
    /// Native scalar precision of the model output. Legacy/unset → `Fp32`.
    #[serde(default)]
    pub native_precision: proximadb_records::EmbeddingScalarType,
    /// Whether the engine L2-normalizes embeddings before they are stored.
    #[serde(default)]
    pub normalize: bool,
}

impl CatalogEmbeddingConfig {
    /// Build a collection binding that cannot follow a mutable model alias.
    pub fn pinned(
        model: impl Into<String>,
        dimension: u32,
        model_asset_id: u64,
        model_version: u64,
        contract_sha256: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let binding = Self {
            model: model.into(),
            model_asset_id: Some(model_asset_id),
            model_version: Some(model_version),
            contract_sha256: Some(contract_sha256.into()),
            dimension,
            ..Default::default()
        };
        binding.validate_model_binding()?;
        Ok(binding)
    }

    pub fn validate_model_binding(&self) -> anyhow::Result<()> {
        let pinned_fields = [
            self.model_asset_id.is_some(),
            self.model_version.is_some(),
            self.contract_sha256.is_some(),
        ];
        if pinned_fields.iter().any(|set| *set) && !pinned_fields.iter().all(|set| *set) {
            return Err(anyhow::anyhow!(
                "model asset id, model version, and contract sha256 must be set together"
            ));
        }
        if let Some(version) = self.model_version
            && version == 0
        {
            return Err(anyhow::anyhow!("model version must be positive"));
        }
        if self.model_asset_id == Some(0) {
            return Err(anyhow::anyhow!("model asset id must be positive"));
        }
        if let Some(digest) = &self.contract_sha256 {
            validate_contract_digest(digest)?;
        }
        if self.dimension == 0 {
            return Err(anyhow::anyhow!("embedding dimension must be positive"));
        }
        Ok(())
    }
}

/// Catalog-authoritative table-level storage descriptor.
///
/// Catalog-native (not the wire `proto` `StorageConfig`) so the catalog stays
/// decoupled from the transport schema. This is the table-level default;
/// fine-grained per-layout authority/format lives in
/// [`CatalogTableSchema::storage_layouts`] and the selected specialization in
/// [`CatalogTableSchema::storage_specialization`]. Every field is `None` =
/// "inherit the engine default", so the descriptor is inert until explicitly set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogStorageConfig {
    /// Compression codec for cold/warehouse segments. `None` = engine default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<CompressionAlgorithm>,
    /// Target segment/file size in MiB before rotation. `None` = engine default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_segment_size_mb: Option<u32>,
    /// Whether read-through caching is enabled for this table. `None` = inherit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_caching: Option<bool>,
}

/// Table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogTableSchema {
    /// Table name
    pub name: String,
    /// ADR-031 stable, immutable internal object id (per-type `u64`, globally unique
    /// across tenants, never reused) — the rename-safe physical key for the WAL
    /// `collection_id`, memtable partition, and object-store paths. `None` = legacy /
    /// not-yet-allocated; `create_table` assigns it, `rename_table` preserves it.
    /// Additive + `#[serde(default)]` so old persisted schemas deserialize to `None`
    /// (mixed-read-safe). Physical layers still key on `name` until the migration
    /// cuts over (ADR-031 O2).
    #[serde(default)]
    pub object_id: Option<u64>,
    /// ADR-031 Phase 4a per-scope compact numeric identity for the typed
    /// object-store path. `stable_namespace_id` (u16, per-account) and
    /// `stable_collection_id` (u32, per-namespace) are minted by the catalog's
    /// `CatalogIdService` at create time and persisted here. The **account** u32
    /// is NOT stored (it would duplicate `account_id`); it's derived from the
    /// account string via a registry at path-build time. Legacy rows load as
    /// `None` (mixed-read-safe — the typed path is opt-in, env-gated
    /// `PROXIMADB_TYPED_PATHS`). The root crate composes these primitives into a
    /// `CollectionIdentity` at the path boundary (layering: the catalog cannot
    /// import that root type). All values are numeric in-memory; base62 is
    /// applied only to path segments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_namespace_id: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_collection_id: Option<u32>,
    /// Table columns
    pub columns: Vec<CatalogColumn>,
    /// Primary key columns (by name)
    pub primary_key: Vec<String>,
    /// Indexes
    pub indexes: Vec<CatalogIndex>,
    /// Cataloged storage layouts and authority modes.
    #[serde(default)]
    pub storage_layouts: Vec<CatalogStorageLayout>,
    /// Cataloged rebuildable projections/access methods.
    #[serde(default)]
    pub projections: Vec<CatalogProjection>,
    /// Optional relational integrity and transaction capabilities.
    #[serde(default)]
    pub relational_capabilities: RelationalCapabilities,
    /// Intended workload profile for layout/query planning.
    #[serde(default)]
    pub workload_profile: CatalogWorkloadProfile,
    /// Primary storage specialization selected for this table/profile.
    #[serde(default)]
    pub storage_specialization: CatalogStorageSpecialization,
    /// ANN filtering routing policy. Controls selectivity thresholds for
    /// pre-filter vs. inline vs. post-filter mode selection. Applies to all
    /// vector ANN projections registered for this table.
    #[serde(default)]
    pub ann_filtering_policy: AnnFilteringPolicy,
    /// Props auto-promotion policy. Controls whether high-frequency msgpack
    /// props keys are promoted to typed PAX columns during compaction.
    /// Must be enabled for document tables to benefit from column-level pruning
    /// and Iceberg/Spark predicate pushdown into props fields.
    #[serde(default)]
    pub props_auto_promotion: PropsAutoPromotionPolicy,
    /// Observability compression hint. Applies to tables with
    /// `CatalogWorkloadProfile::Observability`. Instructs the PAX block writer
    /// to sort by series key + timestamp and apply delta-delta/XOR encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability_compression: Option<ObservabilityCompressionHint>,
    /// Machine-readable codec/layout profiling feedback for PAX, projection,
    /// and open-format planners.
    #[serde(default)]
    pub compression_stats_profiles: Vec<CatalogCompressionStatsProfile>,
    /// Cataloged ADR-012 graph branch merge policy for this table.
    #[serde(default)]
    pub branch_merge_policy: CatalogBranchMergePolicy,

    /// Slice 5 of tenant-pod-affinity: the catalog-authoritative
    /// primary pod for writes to this collection. `None` means
    /// "unbounded" — the gateway routes by its default policy and
    /// reads on any pod may miss the freshest writes that landed
    /// elsewhere. When set, writes for `(tenant_id, this collection)`
    /// MUST be served by the named pod; misrouted writes are
    /// rejected with HTTP 421 Misdirected Request (see
    /// `src/cluster/primary_pod_registry.rs::consult_for_write`).
    ///
    /// Persisted here so the binding survives process restarts and
    /// is consistent across pods. The local in-memory
    /// `PrimaryPodRegistry` is a write-through cache on top of this
    /// field; in steady state the two are in sync, and on cold
    /// start the registry hydrates from this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_pod: Option<CatalogPrimaryPod>,

    // === Embedding-precision rollout (PR 6 of EMBEDDING_PRECISION_LLD_2026_05_22) ===
    /// Reference to the precision policy row in `embedding_precision_policy`.
    /// `None` = inherit the cluster's `GLOBAL_DEFAULT_POLICY_ID` seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_precision_policy_id: Option<String>,
    /// Locked policy version this collection's writes resolve against. Bumps
    /// when an operator promotes the collection to a new policy revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_precision_policy_version: Option<u64>,
    /// Monotonically-increasing epoch tagged onto every WAL record + segment
    /// header. Starts at 0; bumped each time the canonical precision changes.
    #[serde(default)]
    pub current_precision_epoch: u64,
    /// Canonical (default) precision for new writes. Legacy schemas
    /// deserialize as `Fp32` via `Default`.
    #[serde(default)]
    pub canonical_embedding_precision: proximadb_records::EmbeddingScalarType,
    /// Precisions ingest will accept (subject to `ingest_mismatch` policy).
    /// Empty = inherit from policy.
    #[serde(default)]
    pub allowed_embedding_precisions: Vec<proximadb_records::EmbeddingScalarType>,
    /// Per-metric recall@10/@100 SLO; LLD §Q13 defaults seeded automatically.
    #[serde(default)]
    pub embedding_recall_slo: RecallSlo,
    /// Migration lifecycle state for in-flight precision changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision_migration_state: Option<PrecisionMigrationState>,

    // === ADR-024 Step 4: absorbed from storage-side `ProximaSchema` ===
    // The Arrow-native storage schema evolution fields. All additive
    // `#[serde(default)]` so catalogs persisted before the merge still load.
    /// Storage-plane schema identifier (UUID). Default `""` for catalogs that
    /// never carried a storage-side schema id.
    #[serde(default)]
    pub schema_id: String,
    /// Storage-plane schema version (monotonically increasing).
    #[serde(default)]
    pub version: u32,
    /// Parent schema ID (for inheritance tracking).
    #[serde(default)]
    pub parent_schema_id: Option<String>,
    /// Schema fingerprint for fast comparison (xxhash64-style).
    #[serde(default)]
    pub fingerprint: u64,
    /// Creation timestamp (millis since epoch) of the storage-plane schema.
    #[serde(default)]
    pub created_at_ms_schema: i64,
    /// Flag indicating if this is the legacy VectorRecord schema.
    #[serde(default)]
    pub is_legacy_vector_record: bool,

    // === Phase 0 (system-catalog redesign): typed vector/storage config ===
    // Promote distance metric, quantization codec, embedding model, and storage
    // config out of loose `properties` string carriage into typed fields, so the
    // forthcoming catalog WAL/snapshot format serializes the *final* schema shape
    // from day one. All additive `#[serde(default)]` ⇒ catalogs persisted before
    // this change deserialize with `None`.
    /// Vector distance metric for this table's ANN projections. `None` =
    /// unspecified (the engine/projection default applies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance_metric: Option<DistanceMetric>,
    /// Vector quantization codec for this table. `None` = unspecified (engine
    /// default, typically full-precision / no quantization).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationType>,
    /// Embedding model descriptor (model identity + output geometry). `None` for
    /// non-vector tables or when embeddings are supplied pre-computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_config: Option<CatalogEmbeddingConfig>,
    /// Table-level storage configuration (compression, segment size, caching).
    /// `None` = inherit engine defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_config: Option<CatalogStorageConfig>,

    /// Typed AI/MLOps facet. `None` for ordinary tables and every legacy
    /// catalog row. The field is additive and inert until an `mlops.*` asset is
    /// explicitly created; old readers ignore it and new readers default it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlops_asset: Option<serde_json::Value>,

    /// Schema version
    pub schema_version: i32,
    /// Table properties
    pub properties: HashMap<String, String>,
    /// Storage location
    pub location: Option<String>,
    /// Creation timestamp
    pub created_at_ms: i64,
    /// Last update timestamp
    pub updated_at_ms: i64,
}

impl Default for CatalogTableSchema {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        Self {
            name: String::new(),
            object_id: None,
            stable_namespace_id: None,
            stable_collection_id: None,
            columns: Vec::new(),
            primary_key: Vec::new(),
            indexes: Vec::new(),
            storage_layouts: vec![CatalogStorageLayout::default()],
            projections: Vec::new(),
            relational_capabilities: RelationalCapabilities::default(),
            workload_profile: CatalogWorkloadProfile::default(),
            storage_specialization: CatalogStorageSpecialization::default(),
            ann_filtering_policy: AnnFilteringPolicy::default(),
            props_auto_promotion: PropsAutoPromotionPolicy::default(),
            observability_compression: None,
            compression_stats_profiles: Vec::new(),
            branch_merge_policy: CatalogBranchMergePolicy::default(),
            // Slice 5: no catalog-bound primary by default. Operators
            // opt in by PUT-ing through the REST API, which writes
            // both the in-memory registry and this catalog field.
            primary_pod: None,
            // PR 6: inherit cluster default policy; fp32-only baseline.
            embedding_precision_policy_id: None,
            embedding_precision_policy_version: None,
            current_precision_epoch: 0,
            canonical_embedding_precision: proximadb_records::EmbeddingScalarType::Fp32,
            allowed_embedding_precisions: Vec::new(),
            embedding_recall_slo: RecallSlo::lld_defaults(),
            precision_migration_state: None,
            // ADR-024 Step 4: storage-plane schema evolution defaults.
            schema_id: String::new(),
            version: 0,
            parent_schema_id: None,
            fingerprint: 0,
            created_at_ms_schema: 0,
            is_legacy_vector_record: false,
            // Phase 0: typed vector/storage config, all unspecified by default.
            distance_metric: None,
            quantization: None,
            embedding_config: None,
            storage_config: None,
            mlops_asset: None,
            schema_version: 1,
            properties: HashMap::new(),
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
        }
    }
}

impl CatalogTableSchema {
    /// Create a new table schema
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Add a column
    pub fn with_column(mut self, column: CatalogColumn) -> Self {
        self.columns.push(column);
        self
    }

    /// Set primary key
    pub fn with_primary_key(mut self, columns: Vec<String>) -> Self {
        self.primary_key = columns;
        self
    }

    /// Add an index
    pub fn with_index(mut self, index: CatalogIndex) -> Self {
        self.indexes.push(index);
        self
    }

    /// Declare a `UNIQUE` on `columns` — **the one way to do it** (ADR-077 M1).
    ///
    /// A UNIQUE has a canonical home and a projection, and they must not drift:
    ///
    /// * canonical — `relational_capabilities.constraints`, which is what the
    ///   uniqueness enforcement path reads;
    /// * projection — `relational_capabilities.unique_indexes`, which the pg/JDBC
    ///   introspection surfaces render as an index.
    ///
    /// Setting either field directly is how a UNIQUE ends up cataloged but never
    /// enforced. This maintains both together, so that state is not reachable
    /// through the builder. `validate_schema` rejects it if reached another way.
    ///
    /// Idempotent and order-insensitive: re-declaring the same key — even with the
    /// columns in a different order — does not fence it twice.
    pub fn with_unique(mut self, index_name: impl Into<String>, columns: Vec<String>) -> Self {
        if columns.is_empty() {
            return self;
        }
        let already = self.relational_capabilities.constraints.iter().any(|c| {
            matches!(c, ColumnConstraint::Unique { columns: existing }
                              if same_column_set(existing, &columns))
        });
        if !already {
            self.relational_capabilities
                .constraints
                .push(ColumnConstraint::Unique {
                    columns: columns.clone(),
                });
            self.relational_capabilities
                .unique_indexes
                .push(CatalogIndex::new(index_name, columns, CatalogIndexType::BTree).unique());
        }
        self
    }

    /// Add a storage layout/authority descriptor.
    pub fn with_storage_layout(mut self, layout: CatalogStorageLayout) -> Self {
        self.storage_layouts.push(layout);
        self
    }

    /// Add a rebuildable projection descriptor.
    pub fn with_projection(mut self, projection: CatalogProjection) -> Self {
        self.projections.push(projection);
        self
    }

    /// Set the vector distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = Some(metric);
        self
    }

    /// Set the vector quantization codec.
    pub fn with_quantization(mut self, quantization: QuantizationType) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Set the embedding model descriptor.
    pub fn with_embedding_config(mut self, config: CatalogEmbeddingConfig) -> Self {
        self.embedding_config = Some(config);
        self
    }

    /// Set the table-level storage configuration.
    pub fn with_storage_config(mut self, config: CatalogStorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }

    /// Set optional relational capability metadata.
    pub fn with_relational_capabilities(mut self, capabilities: RelationalCapabilities) -> Self {
        self.relational_capabilities = capabilities;
        self
    }

    /// Set the intended workload profile.
    pub fn with_workload_profile(mut self, profile: CatalogWorkloadProfile) -> Self {
        self.workload_profile = profile;
        self
    }

    /// Set the primary storage specialization.
    pub fn with_storage_specialization(
        mut self,
        specialization: CatalogStorageSpecialization,
    ) -> Self {
        self.storage_specialization = specialization;
        self
    }

    /// Set the ANN filtering routing policy.
    pub fn with_ann_filtering_policy(mut self, policy: AnnFilteringPolicy) -> Self {
        self.ann_filtering_policy = policy;
        self
    }

    /// Enable props auto-promotion for document-heavy tables.
    pub fn with_props_auto_promotion(mut self, policy: PropsAutoPromotionPolicy) -> Self {
        self.props_auto_promotion = policy;
        self
    }

    /// Set observability compression hint for time-series/metrics tables.
    pub fn with_observability_compression(mut self, hint: ObservabilityCompressionHint) -> Self {
        self.observability_compression = Some(hint);
        self
    }

    /// Add codec/layout profiling feedback for planners and EXPLAIN.
    pub fn with_compression_stats_profile(
        mut self,
        profile: CatalogCompressionStatsProfile,
    ) -> Self {
        self.compression_stats_profiles.push(profile);
        self
    }

    /// Set the ADR-012 graph branch merge policy.
    pub fn with_branch_merge_policy(mut self, policy: CatalogBranchMergePolicy) -> Self {
        self.branch_merge_policy = policy;
        self
    }

    /// Slice 5 of tenant-pod-affinity: bind this collection to a
    /// primary pod for write routing. Pass `None` to clear an
    /// existing binding ("unbind"). When set, writes for this
    /// collection MUST be served by the named pod or rejected with
    /// HTTP 421 (see `consult_for_write` in the runtime registry).
    pub fn with_primary_pod(mut self, primary: Option<CatalogPrimaryPod>) -> Self {
        self.primary_pod = primary;
        self
    }

    // === ADR-024 Step 4: storage-plane (ProximaSchema) constructors/helpers ===
    //
    // These were inherent methods on the storage-side `ProximaSchema`, ported
    // verbatim onto the unified `CatalogTableSchema` so the ~19 storage-plane
    // consumers (proxima_arrow, proxima_parquet, schema evolution, the Arrow
    // bridge, the DataFusion adapters, …) keep compiling against the alias.
    //
    // Note on primary keys: the storage-side schema kept the primary key as a
    // `Vec<i32>` of *column ids*, while the catalog keeps it as a `Vec<String>`
    // of *column names* (the canonical form). `from_columns` accepts ids and
    // resolves them to names; `primary_key_ids` is the inverse by-id helper.

    /// Construct a storage-plane schema from explicit columns and a primary key
    /// given by *column id* (the storage-side `ProximaSchema::new` signature).
    /// The ids are resolved to the canonical by-name `primary_key`.
    pub fn from_columns(
        schema_id: impl Into<String>,
        columns: Vec<CatalogColumn>,
        primary_key_ids: Vec<i32>,
    ) -> Self {
        let fingerprint = Self::compute_fingerprint_for_columns(&columns);
        let primary_key = primary_key_ids
            .iter()
            .filter_map(|id| columns.iter().find(|c| c.id == *id).map(|c| c.name.clone()))
            .collect();
        Self {
            schema_id: schema_id.into(),
            version: 1,
            columns,
            primary_key,
            fingerprint,
            created_at_ms_schema: proximadb_kernel::time::now_unix_ms(),
            is_legacy_vector_record: false,
            ..Default::default()
        }
    }

    /// Create the legacy VectorRecord schema (v0) for backward compatibility.
    pub fn vector_record_schema(dimension: u32) -> Self {
        let columns = vec![
            CatalogColumn {
                comment: Some("Vector record ID".to_string()),
                ..CatalogColumn::new(1, "id", ProximaType::String).nullable(false)
            },
            CatalogColumn {
                comment: Some("Embedding vector".to_string()),
                ..CatalogColumn::new(
                    2,
                    "vector",
                    ProximaType::DenseVector {
                        element: VectorElement::Float32,
                        dim: dimension as usize,
                    },
                )
                .nullable(false)
            },
            CatalogColumn {
                comment: Some("JSON metadata".to_string()),
                ..CatalogColumn::new(3, "metadata", ProximaType::Json)
            },
            CatalogColumn {
                comment: Some("Record timestamp".to_string()),
                default_value: Some("CURRENT_TIMESTAMP".to_string()),
                ..CatalogColumn::new(
                    4,
                    "timestamp",
                    ProximaType::Timestamp(TimeUnit::Millisecond),
                )
                .nullable(false)
            },
            CatalogColumn {
                comment: Some("MVCC version".to_string()),
                default_value: Some("1".to_string()),
                ..CatalogColumn::new(5, "version", ProximaType::Int64)
            },
        ];

        let fingerprint = Self::compute_fingerprint_for_columns(&columns);
        Self {
            schema_id: "vector_record_v0".to_string(),
            version: 0,
            columns,
            primary_key: vec!["id".to_string()],
            fingerprint,
            properties: HashMap::from([("legacy".to_string(), "true".to_string())]),
            created_at_ms_schema: 0,
            is_legacy_vector_record: true,
            ..Default::default()
        }
    }

    /// Create a standard vector schema with custom metadata columns.
    pub fn with_metadata_columns(
        schema_id: impl Into<String>,
        dimension: u32,
        metadata_fields: Vec<(String, ProximaType)>,
    ) -> Self {
        let mut columns = vec![
            CatalogColumn::new(1, "id", ProximaType::String).nullable(false),
            CatalogColumn::new(
                2,
                "vector",
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: dimension as usize,
                },
            )
            .nullable(false),
            CatalogColumn {
                default_value: Some("CURRENT_TIMESTAMP".to_string()),
                ..CatalogColumn::new(
                    3,
                    "timestamp",
                    ProximaType::Timestamp(TimeUnit::Millisecond),
                )
                .nullable(false)
            },
        ];

        for (next_id, (name, dtype)) in (4..).zip(metadata_fields) {
            columns.push(CatalogColumn::new(next_id, name, dtype));
        }

        Self::from_columns(schema_id, columns, vec![1])
    }

    /// Convert to an Arrow [`Schema`](arrow_schema::Schema), skipping tombstoned
    /// columns. Uses the catalog Arrow projection per column.
    pub fn to_arrow_schema(&self) -> std::sync::Arc<arrow_schema::Schema> {
        let fields: Vec<arrow_schema::Field> = self
            .columns
            .iter()
            .filter(|c| !c.is_deleted)
            .map(|col| col.to_arrow_field())
            .collect();
        std::sync::Arc::new(arrow_schema::Schema::new(fields))
    }

    /// Create from an Arrow [`Schema`](arrow_schema::Schema) with auto-generated
    /// stable column ids.
    pub fn from_arrow_schema(schema: &arrow_schema::Schema, schema_id: impl Into<String>) -> Self {
        let columns: Vec<CatalogColumn> = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| CatalogColumn::from_arrow_field(field, (idx + 1) as i32))
            .collect();
        Self::from_columns(schema_id, columns, Vec::new())
    }

    /// Compute the schema fingerprint over the active (non-tombstoned) columns.
    pub fn compute_fingerprint_for_columns(columns: &[CatalogColumn]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for col in columns {
            if !col.is_deleted {
                col.id.hash(&mut hasher);
                col.name.hash(&mut hasher);
                format!("{:?}", col.data_type).hash(&mut hasher);
                col.nullable.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Active column by stable id (skips tombstoned columns).
    pub fn column_by_id(&self, id: i32) -> Option<&CatalogColumn> {
        self.columns.iter().find(|c| c.id == id && !c.is_deleted)
    }

    /// Active column by name (skips tombstoned columns).
    pub fn column_by_name(&self, name: &str) -> Option<&CatalogColumn> {
        self.columns
            .iter()
            .find(|c| c.name == name && !c.is_deleted)
    }

    /// Next free column id.
    pub fn next_column_id(&self) -> i32 {
        self.columns.iter().map(|c| c.id).max().unwrap_or(0) + 1
    }

    /// Count of active (non-tombstoned) columns.
    pub fn active_column_count(&self) -> usize {
        self.columns.iter().filter(|c| !c.is_deleted).count()
    }

    /// Dimension of the first dense-vector column, if any.
    pub fn vector_dimension(&self) -> Option<u32> {
        for col in &self.columns {
            if let ProximaType::DenseVector { dim, .. } = &col.data_type {
                return Some(*dim as u32);
            }
        }
        None
    }

    /// By-id view of the primary key, resolving the canonical by-name
    /// `primary_key` back to column ids (the storage-side representation).
    pub fn primary_key_ids(&self) -> Vec<i32> {
        self.primary_key
            .iter()
            .filter_map(|name| self.columns.iter().find(|c| &c.name == name).map(|c| c.id))
            .collect()
    }
}

/// xCatalog feedback record for one measured compression/layout profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogCompressionStatsProfile {
    /// Stable profile id referenced by PAX metadata, projections, and EXPLAIN fixtures.
    pub profile_id: String,
    /// Optional cataloged layout that produced this profile.
    pub layout_name: Option<String>,
    /// Optional projection/access-method that produced this profile.
    pub projection_id: Option<String>,
    /// Selected codec or pilot codec family.
    pub selected_scheme: String,
    /// Raw input bytes measured by the profiler.
    pub raw_bytes: u64,
    /// Encoded payload bytes measured by the profiler.
    pub encoded_bytes: u64,
    /// Number of logical values covered by this profile.
    pub value_count: u64,
    /// Measured raw/encoded ratio.
    pub measured_ratio: f64,
    /// Whether exact visible values can be reconstructed from the encoded payload.
    pub exact_reconstruction: bool,
    /// Measured encode CPU per block, if known.
    pub encode_cpu_ms_per_block: Option<f64>,
    /// Measured decode cost per logical value, if known.
    pub decode_ns_per_value: Option<f64>,
    /// Rejected alternatives and reasons.
    #[serde(default)]
    pub rejected_candidates: Vec<CatalogCompressionRejectedCandidate>,
    /// Extension fields for benchmark or engine-specific metadata.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

impl CatalogCompressionStatsProfile {
    pub fn new(
        profile_id: impl Into<String>,
        selected_scheme: impl Into<String>,
        raw_bytes: u64,
        encoded_bytes: u64,
        value_count: u64,
        exact_reconstruction: bool,
    ) -> Self {
        Self {
            profile_id: profile_id.into(),
            layout_name: None,
            projection_id: None,
            selected_scheme: selected_scheme.into(),
            raw_bytes,
            encoded_bytes,
            value_count,
            measured_ratio: measured_ratio(raw_bytes, encoded_bytes),
            exact_reconstruction,
            encode_cpu_ms_per_block: None,
            decode_ns_per_value: None,
            rejected_candidates: Vec::new(),
            properties: HashMap::new(),
        }
    }

    pub fn with_layout_name(mut self, layout_name: impl Into<String>) -> Self {
        self.layout_name = Some(layout_name.into());
        self
    }

    pub fn with_projection_id(mut self, projection_id: impl Into<String>) -> Self {
        self.projection_id = Some(projection_id.into());
        self
    }

    pub fn with_decode_ns_per_value(mut self, decode_ns_per_value: f64) -> Self {
        self.decode_ns_per_value = Some(decode_ns_per_value);
        self
    }

    pub fn bytes_per_value(&self) -> f64 {
        if self.value_count == 0 {
            0.0
        } else {
            self.encoded_bytes as f64 / self.value_count as f64
        }
    }
}

/// Cataloged ADR-012 branch merge policy for graph-capable tables.
///
/// This is metadata only: xCatalog records the policy the branch merge service
/// must apply, while the graph merge runtime owns conflict detection and WAL
/// write-back. Defaults match `docs/12-design/adr/ADR-012-graph-branch-merge-semantics.adoc`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogBranchMergePolicy {
    #[serde(default = "catalog_branch_merge_policy_lww")]
    pub node_upsert: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_delete_wins")]
    pub node_delete: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_lww")]
    pub edge_upsert: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_delete_wins")]
    pub edge_delete: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_lww")]
    pub embedding_update: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_add_wins")]
    pub label_set: CatalogBranchMergeResolution,
    #[serde(default = "catalog_branch_merge_policy_lww_per_key")]
    pub props_key: CatalogBranchMergeResolution,
    /// Extension point for future branch ancestry/schema-mode knobs.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

impl Default for CatalogBranchMergePolicy {
    fn default() -> Self {
        Self {
            node_upsert: CatalogBranchMergeResolution::LastWriteWins,
            node_delete: CatalogBranchMergeResolution::DeleteWins,
            edge_upsert: CatalogBranchMergeResolution::LastWriteWins,
            edge_delete: CatalogBranchMergeResolution::DeleteWins,
            embedding_update: CatalogBranchMergeResolution::LastWriteWins,
            label_set: CatalogBranchMergeResolution::AddWinsSetUnion,
            props_key: CatalogBranchMergeResolution::LastWriteWinsPerKey,
            properties: HashMap::new(),
        }
    }
}

impl CatalogBranchMergePolicy {
    pub fn adr_012_default() -> Self {
        Self::default()
    }
}

/// Per-conflict resolution mode recorded in xCatalog for ADR-012 merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogBranchMergeResolution {
    LastWriteWins,
    DeleteWins,
    AddWinsSetUnion,
    LastWriteWinsPerKey,
}

fn catalog_branch_merge_policy_lww() -> CatalogBranchMergeResolution {
    CatalogBranchMergeResolution::LastWriteWins
}

fn catalog_branch_merge_policy_delete_wins() -> CatalogBranchMergeResolution {
    CatalogBranchMergeResolution::DeleteWins
}

fn catalog_branch_merge_policy_add_wins() -> CatalogBranchMergeResolution {
    CatalogBranchMergeResolution::AddWinsSetUnion
}

fn catalog_branch_merge_policy_lww_per_key() -> CatalogBranchMergeResolution {
    CatalogBranchMergeResolution::LastWriteWinsPerKey
}

/// Slice 5 of tenant-pod-affinity: catalog-authoritative primary-pod
/// binding for a `(tenant, collection)` pair. Persisted on the
/// owning [`CatalogTableSchema`] and consumed by the in-process
/// [`PrimaryPodRegistry`] as the durable source of truth.
///
/// Why a separate catalog type (rather than re-exporting the
/// in-process `PrimaryPod`):
///
/// * The catalog crate is a foundation layer — depending on the
///   root crate's `cluster::primary_pod_registry::PrimaryPod` would
///   invert the dependency arrow.
/// * The on-disk encoding belongs to the catalog (`assigned_at_ms`
///   millisecond unit matches the rest of `CatalogTableSchema`),
///   while the in-process registry uses `assigned_at_ns` to match
///   `SystemTime::now()` ergonomics. The conversion is one
///   multiplication.
/// * Decoupling means future catalog-only fields can land here
///   without forcing the root crate to recompile.
///
/// Conversion shims live in the root crate where both types are in
/// scope, mirroring the pattern used for other catalog ↔ runtime
/// type pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogPrimaryPod {
    /// Pod identifier — typically a k8s pod name like
    /// `proximadb-write-0`. Free-form; the catalog doesn't validate
    /// reachability or naming convention.
    pub pod: String,
    /// Millis since epoch when this assignment was last set.
    /// Reassignments advance this so dashboards can show
    /// "primary changed N seconds ago" without a separate history
    /// table.
    pub assigned_at_ms: i64,
    /// Why the assignment happened. Lives as an enum so future
    /// variants land via a compile error rather than a silent
    /// dashboard break.
    pub reason: CatalogPrimaryPodReason,
}

impl CatalogPrimaryPod {
    /// Construct with current wall-clock millis. Useful at REST
    /// PUT time where the caller doesn't carry a millisecond
    /// timestamp around.
    pub fn now(pod: impl Into<String>, reason: CatalogPrimaryPodReason) -> Self {
        let assigned_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Self {
            pod: pod.into(),
            assigned_at_ms,
            reason,
        }
    }
}

/// Why a primary-pod assignment was made. Mirrors the in-process
/// `AssignmentReason` enum one-to-one so the conversion shim is
/// trivial. Stable lowercase labels for catalog JSON, REST payloads,
/// and Prometheus alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogPrimaryPodReason {
    /// Initial assignment at collection-create time.
    Create,
    /// Explicit operator decision (REST PATCH or admin tool).
    Operator,
    /// Failover after the previous primary became unreachable.
    Failover,
    /// Planned rebalance — capacity / latency tuning, not a fault.
    Rebalance,
    /// Catalog reconciliation pulled the assignment from xCatalog
    /// after a process restart. Used when the in-process registry
    /// hydrates from the catalog on cold start, NOT when the
    /// catalog itself loads from durable storage.
    CatalogReplay,
}

impl CatalogPrimaryPodReason {
    /// Stable lowercase label, matching the in-process enum's label
    /// surface. Locked in by a roundtrip test in this module.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Operator => "operator",
            Self::Failover => "failover",
            Self::Rebalance => "rebalance",
            Self::CatalogReplay => "catalog_replay",
        }
    }
}

/// xCatalog rejected codec candidate recorded from profiling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogCompressionRejectedCandidate {
    pub scheme: String,
    pub reason: String,
    pub expected_ratio: Option<f32>,
}

fn measured_ratio(raw_bytes: u64, encoded_bytes: u64) -> f64 {
    if raw_bytes == 0 || encoded_bytes == 0 {
        0.0
    } else {
        raw_bytes as f64 / encoded_bytes as f64
    }
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogIndex {
    /// ADR-031 / TD-181 P0: stable, immutable `u64` catalog object identity,
    /// minted from the single system-wide catalog sequence (globally unique,
    /// never reused) — the same sequence that mints table and namespace oids.
    /// Additive + `#[serde(default)]`, so legacy persisted indexes load as
    /// `None` (mixed-read-safe; backfilled on first allocation pass).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_id: Option<u64>,
    /// Index name
    pub name: String,
    /// Indexed columns
    pub columns: Vec<String>,
    /// Index type
    pub index_type: CatalogIndexType,
    /// Is unique index
    pub is_unique: bool,
    /// Index properties
    pub properties: HashMap<String, String>,
}

impl CatalogIndex {
    /// Create a new index
    pub fn new(
        name: impl Into<String>,
        columns: Vec<String>,
        index_type: CatalogIndexType,
    ) -> Self {
        Self {
            object_id: None,
            name: name.into(),
            columns,
            index_type,
            is_unique: false,
            properties: HashMap::new(),
        }
    }

    /// Set unique
    pub fn unique(mut self) -> Self {
        self.is_unique = true;
        self
    }

    /// Set the stable `u64` catalog object identity (ADR-031 / TD-181 P0).
    pub fn with_object_id(mut self, object_id: u64) -> Self {
        self.object_id = Some(object_id);
        self
    }
}

/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatalogIndexType {
    /// B-tree index
    BTree,
    /// Hash index
    Hash,
    /// Full-text index
    FullText,
    /// PostgreSQL-compatible GIN index for JSONB/path-heavy document projections
    Gin,
    /// HNSW vector index
    Hnsw,
    /// IVF vector index
    Ivf,
    /// Product quantization
    Pq,
}

/// Workload profile used by xCatalog, pgwire DDL, and planners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogWorkloadProfile {
    /// Row-oriented point reads/writes and transactional DML.
    Oltp,
    /// Scan-heavy analytical reads and external/open-table publication.
    Olap,
    /// Hybrid transactional/analytical processing over the same record model.
    #[default]
    Htap,
    /// Vector-heavy workload with ANN projections.
    Vector,
    /// Document-heavy workload with JSON/path projections.
    Document,
    /// Graph-heavy workload with adjacency/topology projections.
    Graph,
    /// Observability/time-series workload.
    Observability,
    /// Mixed multimodal table/profile.
    Mixed,
}

impl CatalogWorkloadProfile {
    /// Parse common SQL/catalog option spellings.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "oltp" | "transactional" | "row" => Some(Self::Oltp),
            "olap" | "analytic" | "analytics" | "analytical" | "columnar" => Some(Self::Olap),
            "htap" | "hybrid" | "pax" => Some(Self::Htap),
            "vector" | "ann" => Some(Self::Vector),
            "document" | "json" | "jsonb" => Some(Self::Document),
            "graph" | "pgq" | "cypher" => Some(Self::Graph),
            "observability" | "time_series" | "timeseries" | "metrics" | "traces" => {
                Some(Self::Observability)
            }
            "mixed" | "multimodal" => Some(Self::Mixed),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oltp => "oltp",
            Self::Olap => "olap",
            Self::Htap => "htap",
            Self::Vector => "vector",
            Self::Document => "document",
            Self::Graph => "graph",
            Self::Observability => "observability",
            Self::Mixed => "mixed",
        }
    }
}

/// Primary storage specialization. Specialization is a physical/access-method
/// hint, not a separate semantic authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogStorageSpecialization {
    /// Generic durable relational record layout.
    GenericRelational,
    /// PAX row-family with row directory plus column stripes.
    #[default]
    PaxRowFamily,
    /// PAX OLTP mode, row-directory optimized.
    PaxOltp,
    /// PAX OLAP mode, column-stripe optimized.
    PaxOlap,
    /// SST/LSM write-optimized family.
    LsmWriteOptimized,
    /// Columnar analytics family.
    ColumnarAnalytics,
    /// Vector ANN specialty projection family.
    VectorAnn,
    /// Document JSON/path specialty projection family.
    DocumentJson,
    /// Graph adjacency/topology specialty projection family.
    GraphTopology,
    /// Observability/time-series specialty projection family.
    ObservabilityTimeSeries,
    /// External/open-table mapping.
    ExternalOpenTable,
}

impl CatalogStorageSpecialization {
    /// Parse SQL/catalog layout or specialization option values.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "generic" | "relational" | "row" | "rowrecord" | "row_record" => {
                Some(Self::GenericRelational)
            }
            "pax" | "hybrid" | "htap" | "pax_row_family" => Some(Self::PaxRowFamily),
            "oltp" | "pax_oltp" | "rowdir" | "row_directory" => Some(Self::PaxOltp),
            "olap" | "pax_olap" => Some(Self::PaxOlap),
            "sst" | "lsm" | "lsm_record" => Some(Self::LsmWriteOptimized),
            "columnar" | "column" | "analytics" => Some(Self::ColumnarAnalytics),
            "vector" | "ann" | "hnsw" | "ivf" | "pq" => Some(Self::VectorAnn),
            "document" | "json" | "jsonb" => Some(Self::DocumentJson),
            "graph" | "adjacency" | "csr" | "coo" => Some(Self::GraphTopology),
            "observability" | "time_series" | "timeseries" => Some(Self::ObservabilityTimeSeries),
            "external" | "open_table" | "opentable" | "lakehouse" => Some(Self::ExternalOpenTable),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::GenericRelational => "generic_relational",
            Self::PaxRowFamily => "pax_row_family",
            Self::PaxOltp => "pax_oltp",
            Self::PaxOlap => "pax_olap",
            Self::LsmWriteOptimized => "lsm_write_optimized",
            Self::ColumnarAnalytics => "columnar_analytics",
            Self::VectorAnn => "vector_ann",
            Self::DocumentJson => "document_json",
            Self::GraphTopology => "graph_topology",
            Self::ObservabilityTimeSeries => "observability_time_series",
            Self::ExternalOpenTable => "external_open_table",
        }
    }
}

/// Source-of-truth mode for a cataloged table, stream, projection, or external format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogAuthorityMode {
    /// ProximaRecord plus WAL/log/manifest own durable truth.
    #[default]
    InternalCanonical,
    /// Explicit name for ProximaDB WAL + ProximaRecord authority.
    ProximaAuthoritative,
    /// An external table/source owns durable truth; ProximaDB maps and governs access.
    ExternalAuthoritative,
    /// Point-in-time import from an external source.
    ImportedSnapshot,
    /// Publication generated from canonical records.
    ExportedPublication,
    /// Explicit name for a publication generated from canonical records.
    ProjectionPublication,
    /// Rebuildable structure derived from canonical records or events.
    RebuildableProjection,
    /// Read-only federation over an external source without importing record authority.
    FederatedRead,
}

impl CatalogAuthorityMode {
    /// Returns true when ProximaDB WAL + ProximaRecord storage own durable truth.
    pub fn is_proxima_authoritative(self) -> bool {
        matches!(
            self,
            CatalogAuthorityMode::InternalCanonical | CatalogAuthorityMode::ProximaAuthoritative
        )
    }

    /// Returns true when durable truth is outside ProximaDB's canonical record store.
    pub fn is_external_authoritative(self) -> bool {
        matches!(self, CatalogAuthorityMode::ExternalAuthoritative)
    }

    /// Returns true when the layout/projection is rebuildable from another source.
    pub fn is_rebuildable_or_publication(self) -> bool {
        matches!(
            self,
            CatalogAuthorityMode::ExportedPublication
                | CatalogAuthorityMode::ProjectionPublication
                | CatalogAuthorityMode::RebuildableProjection
        )
    }

    /// Stable xCatalog ownership-mode string used by adapters and docs.
    pub fn ownership_mode_name(self) -> &'static str {
        match self {
            CatalogAuthorityMode::InternalCanonical
            | CatalogAuthorityMode::ProximaAuthoritative => "ProximaAuthoritative",
            CatalogAuthorityMode::ExternalAuthoritative => "ExternalAuthoritative",
            CatalogAuthorityMode::ImportedSnapshot => "ImportedSnapshot",
            CatalogAuthorityMode::ExportedPublication
            | CatalogAuthorityMode::ProjectionPublication => "ProjectionPublication",
            CatalogAuthorityMode::RebuildableProjection => "RebuildableProjection",
            CatalogAuthorityMode::FederatedRead => "FederatedRead",
        }
    }
}

/// Physical layout family selected for a cataloged object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogStorageLayoutKind {
    /// Mutable record/row-oriented layout.
    #[default]
    RowRecord,
    /// Log-structured merge layout for write-heavy data.
    LsmRecord,
    /// Partition Attributes Across hybrid row/column layout.
    Pax,
    /// Columnar segment layout.
    Columnar,
    /// Append-only event stream layout.
    AppendOnlyEvent,
    /// Vector ANN projection or fragment layout.
    VectorAnn,
    /// Graph topology projection such as adjacency, CSR, or COO.
    GraphTopology,
    /// Time-series compression or partition block.
    TimeSeriesBlock,
    /// Externally owned or externally formatted table.
    ExternalTable,
}

/// Physical format used by a storage layout or projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogPhysicalFormat {
    /// Native ProximaDB record/block format.
    #[default]
    ProximaBlock,
    /// Sorted string table or SST-family layout.
    Sst,
    /// Apache Arrow in-memory or IPC format.
    Arrow,
    /// Delimited text, usually comma-separated values.
    Csv,
    /// JSON or JSON Lines.
    Json,
    /// XML documents or rows.
    Xml,
    /// Apache Avro object container or schema-registry payloads.
    Avro,
    /// Apache Parquet file format.
    Parquet,
    /// Apache ORC file format.
    Orc,
    /// Apache Iceberg table format.
    Iceberg,
    /// Delta Lake table format.
    Delta,
    /// Apache Hudi table format.
    Hudi,
    /// GraphAr-style graph lake format.
    GraphAr,
    /// Format not yet modeled as a first-class enum variant.
    External(String),
}

/// Write and refresh behavior for a cataloged layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogWriteMode {
    /// In-place or logically mutable writes are accepted.
    #[default]
    Mutable,
    /// Writes append immutable records/events.
    AppendOnly,
    /// Writes create new snapshots or file sets.
    CopyOnWrite,
    /// Data is refreshed from an external authority.
    ExternalRefresh,
    /// Data arrives through a stream/CDC connector.
    StreamingIngest,
    /// Read-only mapping.
    ReadOnly,
}

/// Freshness semantics for a rebuildable projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionFreshness {
    /// Projection is updated in the same write transaction as canonical records.
    Synchronous,
    /// Projection may lag but must stay within a declared bound.
    BoundedLag,
    /// Projection is refreshed lazily on demand or by background work.
    #[default]
    Lazy,
    /// Projection is refreshed only by explicit command.
    Manual,
}

/// Runtime freshness state for a rebuildable projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionFreshnessState {
    /// Projection includes all committed records required by its freshness SLA.
    #[default]
    Fresh,
    /// Projection update is in progress.
    Updating,
    /// Projection is known to lag canonical WAL/record state.
    Stale,
    /// Projection cannot be incrementally repaired and must be rebuilt.
    RebuildRequired,
    /// External table/publication snapshot has been registered in xCatalog.
    ExternalSnapshotRegistered,
    /// Physical projection is not usable.
    Unavailable,
}

/// Projection or access-method family cataloged below xCatalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CatalogProjectionKind {
    /// JSON path or generated-column projection.
    JsonPath,
    /// Full-text or lexical search projection.
    FullText,
    /// Dense/sparse/binary vector ANN projection.
    VectorAnn,
    /// Graph adjacency, CSR, COO, or graph algorithm projection.
    GraphTopology,
    /// Columnar materialization.
    Columnar,
    /// Time-series rollup or compression projection.
    TimeSeries,
    /// Trace assembly, service map, or observability correlation projection.
    Observability,
    /// Materialized SQL/logical view.
    MaterializedView,
    /// Generic projection until a more specific family is introduced.
    #[default]
    Other,
}

/// Catalog metadata shared by internal layouts and external table mappings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogStorageLayout {
    /// Stable layout identifier inside the table/profile/collection.
    pub name: String,
    /// Source-of-truth role for this layout.
    pub authority: CatalogAuthorityMode,
    /// Physical layout family.
    pub layout_kind: CatalogStorageLayoutKind,
    /// Concrete physical format.
    pub physical_format: CatalogPhysicalFormat,
    /// Write/refresh behavior.
    pub write_mode: CatalogWriteMode,
    /// Optional URI/path/catalog identifier for the physical data.
    pub location: Option<String>,
    /// Snapshot or isolation semantics, e.g. "mvcc", "iceberg-snapshot", "external-latest".
    pub snapshot_semantics: Option<String>,
    /// Whether policy/RLS enforcement happens inside ProximaDB before rows leave this layout.
    pub policy_enforced_in_proxima: bool,
    /// Names of ProximaType fields that require lossy conversion in this format, if any.
    pub lossy_type_mappings: Vec<String>,
    /// Additional implementation-specific metadata.
    pub properties: HashMap<String, String>,
}

impl Default for CatalogStorageLayout {
    fn default() -> Self {
        Self {
            name: "primary".to_string(),
            authority: CatalogAuthorityMode::InternalCanonical,
            layout_kind: CatalogStorageLayoutKind::RowRecord,
            physical_format: CatalogPhysicalFormat::ProximaBlock,
            write_mode: CatalogWriteMode::Mutable,
            location: None,
            snapshot_semantics: None,
            policy_enforced_in_proxima: true,
            lossy_type_mappings: Vec::new(),
            properties: HashMap::new(),
        }
    }
}

impl CatalogStorageLayout {
    /// Create an internal canonical layout with default policy enforcement.
    pub fn internal(name: impl Into<String>, layout_kind: CatalogStorageLayoutKind) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::ProximaAuthoritative,
            layout_kind,
            ..Default::default()
        }
    }

    /// Create the recommended ProximaDB-authoritative PAX relational layout.
    pub fn proxima_authoritative_pax(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::ProximaAuthoritative,
            layout_kind: CatalogStorageLayoutKind::Pax,
            physical_format: CatalogPhysicalFormat::ProximaBlock,
            write_mode: CatalogWriteMode::Mutable,
            snapshot_semantics: Some("mvcc".to_string()),
            policy_enforced_in_proxima: true,
            ..Default::default()
        }
    }

    /// Create an external authoritative table mapping.
    pub fn external_authoritative(
        name: impl Into<String>,
        format: CatalogPhysicalFormat,
        location: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::ExternalAuthoritative,
            layout_kind: CatalogStorageLayoutKind::ExternalTable,
            physical_format: format,
            write_mode: CatalogWriteMode::ExternalRefresh,
            location: Some(location.into()),
            snapshot_semantics: Some("external-latest".to_string()),
            policy_enforced_in_proxima: false,
            ..Default::default()
        }
    }

    /// Create a rebuildable external/open-format publication from ProximaDB-owned records.
    pub fn projection_publication(
        name: impl Into<String>,
        format: CatalogPhysicalFormat,
        location: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::ProjectionPublication,
            layout_kind: CatalogStorageLayoutKind::ExternalTable,
            physical_format: format,
            write_mode: CatalogWriteMode::CopyOnWrite,
            location: Some(location.into()),
            snapshot_semantics: Some("published-snapshot".to_string()),
            policy_enforced_in_proxima: true,
            ..Default::default()
        }
    }

    /// Create a rebuildable internal specialty projection/access-method layout.
    pub fn specialty_projection(
        name: impl Into<String>,
        layout_kind: CatalogStorageLayoutKind,
        physical_format: CatalogPhysicalFormat,
    ) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::RebuildableProjection,
            layout_kind,
            physical_format,
            write_mode: CatalogWriteMode::ReadOnly,
            snapshot_semantics: Some("rebuildable-projection".to_string()),
            policy_enforced_in_proxima: true,
            ..Default::default()
        }
    }

    /// Create a point-in-time imported snapshot. ProximaDB owns records after import.
    pub fn imported_snapshot(
        name: impl Into<String>,
        format: CatalogPhysicalFormat,
        location: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::ImportedSnapshot,
            layout_kind: CatalogStorageLayoutKind::ExternalTable,
            physical_format: format,
            write_mode: CatalogWriteMode::ReadOnly,
            location: Some(location.into()),
            snapshot_semantics: Some("imported-snapshot".to_string()),
            policy_enforced_in_proxima: true,
            ..Default::default()
        }
    }

    /// Create a federated read mapping without importing or owning record state.
    pub fn federated_read(
        name: impl Into<String>,
        format: CatalogPhysicalFormat,
        location: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            authority: CatalogAuthorityMode::FederatedRead,
            layout_kind: CatalogStorageLayoutKind::ExternalTable,
            physical_format: format,
            write_mode: CatalogWriteMode::ReadOnly,
            location: Some(location.into()),
            snapshot_semantics: Some("external-snapshot".to_string()),
            policy_enforced_in_proxima: false,
            ..Default::default()
        }
    }

    /// Whether this layout needs an explicit external ownership/policy contract.
    pub fn requires_external_contract(&self) -> bool {
        matches!(
            self.authority,
            CatalogAuthorityMode::ExternalAuthoritative | CatalogAuthorityMode::FederatedRead
        )
    }
}

/// Catalog metadata for a rebuildable projection/access method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogProjection {
    /// Projection name.
    pub name: String,
    /// Projection family.
    pub kind: CatalogProjectionKind,
    /// Physical format backing the projection.
    pub physical_format: CatalogPhysicalFormat,
    /// Canonical layout/table/stream used to rebuild this projection.
    pub rebuild_source: String,
    /// Freshness semantics.
    pub freshness: ProjectionFreshness,
    /// Runtime freshness state used by planners and EXPLAIN.
    #[serde(default)]
    pub freshness_state: ProjectionFreshnessState,
    /// Maximum accepted lag in milliseconds for bounded-lag projections.
    /// Required when freshness == BoundedLag. None means "no bound declared"
    /// which the planner treats as Stale-OK. For ANN freshness, this directly
    /// impacts recall: new records not yet in the HNSW index are missed results.
    pub max_lag_ms: Option<i64>,
    /// Source WAL range, snapshot, or manifest used by this projection.
    #[serde(default)]
    pub source_range: Option<String>,
    /// Last commit, snapshot, or manifest included by this projection.
    #[serde(default)]
    pub last_included_position: Option<String>,
    /// Whether the projection can be rebuilt without data loss.
    pub rebuildable: bool,
    /// Quantitative rebuild time objective. None = unverified / not benchmarked.
    /// Planners should reject routing through this projection for latency-sensitive
    /// queries if an active rebuild has exceeded `rto.max_rebuild_wait_seconds`.
    #[serde(default)]
    pub rebuild_rto: Option<RebuildRtoSpec>,
    /// Invalidation policy, e.g. synchronous, enqueue, mark-stale, rebuild-required.
    #[serde(default)]
    pub invalidation_policy: Option<String>,
    /// Policy/RLS boundary for this projection.
    #[serde(default)]
    pub policy_boundary: Option<String>,
    /// Whether using this projection can change recall or ranking quality.
    pub lossy: bool,
    /// Benchmark or smoke gate required before supported/default use.
    #[serde(default)]
    pub benchmark_gate: Option<String>,
    /// Human-readable support status: experimental, beta, supported, deprecated.
    pub support_status: String,
    /// Catalog-resolved physical location (URI/path prefix) of this projection's
    /// materialized bytes — the index/MV files. `None` means "not yet resolved";
    /// the engine then derives the default path from `DrPathBuilder`
    /// (`data/{tenant}/{namespace}/{collection}/indexes/{name}/`). When set, it is
    /// authoritative, so a projection can be relocated/tiered independently. This
    /// is the catalog-resolution of index/MV addressing (CATALOG_OBJECT_MODEL P1).
    #[serde(default)]
    pub location: Option<String>,
    /// Storage tier (capability class) for this projection's materialized bytes —
    /// the per-projection tiering knob (CATALOG_OBJECT_MODEL #4). A *capability*
    /// tag (cost/CRR/KMS/monitoring class), NOT a path selector: physical placement
    /// is carried by [`location`](Self::location), which an operator/control-plane
    /// sets to the chosen tier's bucket (e.g. hot RaBitQ codes → premium, cold f32
    /// rerank → standard). `None` inherits the owning namespace's
    /// [`StoragePoolClass`] (see [`effective_tier`](Self::effective_tier)); `Some`
    /// overrides it for this projection. Automated tier→bucket resolution is a
    /// deferred follow-up.
    #[serde(default)]
    pub tier: Option<StoragePoolClass>,
    /// Additional implementation-specific metadata.
    pub properties: HashMap<String, String>,
}

impl CatalogProjection {
    /// Create a rebuildable projection with lazy freshness by default.
    pub fn rebuildable(
        name: impl Into<String>,
        kind: CatalogProjectionKind,
        rebuild_source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            physical_format: CatalogPhysicalFormat::ProximaBlock,
            rebuild_source: rebuild_source.into(),
            freshness: ProjectionFreshness::Lazy,
            freshness_state: ProjectionFreshnessState::Fresh,
            max_lag_ms: None,
            source_range: None,
            last_included_position: None,
            rebuildable: true,
            rebuild_rto: None,
            invalidation_policy: Some("mark-stale".to_string()),
            policy_boundary: Some("engine-enforced".to_string()),
            lossy: false,
            benchmark_gate: None,
            support_status: "experimental".to_string(),
            location: None,
            tier: None,
            properties: HashMap::new(),
        }
    }

    /// Declare bounded-lag freshness for planner and EXPLAIN decisions.
    pub fn with_bounded_lag(mut self, max_lag_ms: i64) -> Self {
        self.freshness = ProjectionFreshness::BoundedLag;
        self.max_lag_ms = Some(max_lag_ms);
        self
    }

    /// Set the catalog-resolved physical location (URI/path prefix) of this
    /// projection's materialized bytes. Authoritative when set; otherwise the
    /// engine derives the default `DrPathBuilder` index path.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set this projection's storage tier (capability class) — the per-projection
    /// tiering override (CATALOG_OBJECT_MODEL #4). `None` (the default) inherits the
    /// owning namespace's tier; see [`effective_tier`](Self::effective_tier).
    pub fn with_tier(mut self, tier: StoragePoolClass) -> Self {
        self.tier = Some(tier);
        self
    }

    /// The effective storage tier for this projection: its own `tier` when set,
    /// otherwise the owning namespace's `StoragePoolClass`. This is the single
    /// rule for resolving a projection's tier (placement still flows through
    /// [`location`](Self::location); the tier is a capability/cost-accounting class).
    pub fn effective_tier(&self, namespace_tier: StoragePoolClass) -> StoragePoolClass {
        self.tier.unwrap_or(namespace_tier)
    }

    /// Attach a quantitative rebuild RTO specification.
    /// Without this, planners must treat the projection as having an
    /// unknown rebuild duration and may refuse to use it in SLA-critical paths.
    pub fn with_rebuild_rto(mut self, rto: RebuildRtoSpec) -> Self {
        self.rebuild_rto = Some(rto);
        self
    }

    /// Update runtime freshness state.
    pub fn with_freshness_state(mut self, state: ProjectionFreshnessState) -> Self {
        self.freshness_state = state;
        self
    }

    /// Record projection lineage used for repair and freshness checks.
    pub fn with_lineage(
        mut self,
        source_range: impl Into<String>,
        last_included_position: impl Into<String>,
    ) -> Self {
        self.source_range = Some(source_range.into());
        self.last_included_position = Some(last_included_position.into());
        self
    }

    /// Record policy boundary and benchmark gate metadata.
    pub fn with_policy_and_gate(
        mut self,
        policy_boundary: impl Into<String>,
        benchmark_gate: impl Into<String>,
    ) -> Self {
        self.policy_boundary = Some(policy_boundary.into());
        self.benchmark_gate = Some(benchmark_gate.into());
        self
    }
}

// ─── ANN Filtering ───────────────────────────────────────────────────────────

/// Selectivity-based routing mode for approximate nearest-neighbor (ANN) queries
/// that carry scalar filter predicates. Determined by the planner from catalog
/// statistics and exposed in EXPLAIN output (ADR-004 FilterDiagnostics).
///
/// Literature basis: FAVOR (arXiv:2605.07770), Filtered ANN Survey
/// (arXiv:2602.11443), Learning-based Filtered-ANN planning (arXiv:2602.17914).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AnnFilteringMode {
    /// Run the scalar filter first, then ANN over the surviving candidate set.
    /// Correct when selectivity is very low (< `pre_filter_threshold`).
    /// Avoids expensive ANN traversal over the full corpus.
    PreFilter,
    /// Interleave scalar predicate evaluation during HNSW/IVF graph traversal.
    /// Correct for moderate selectivity (`pre_filter_threshold`..`post_filter_threshold`).
    /// Requires predicate-aware graph index support (AXIS primitive).
    #[default]
    Inline,
    /// Run ANN first for a larger candidate set, then filter results.
    /// Correct when selectivity is high (> `post_filter_threshold`) so that
    /// nearly all ANN neighbors survive the filter anyway.
    PostFilter,
}

impl AnnFilteringMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PreFilter => "pre_filter",
            Self::Inline => "inline",
            Self::PostFilter => "post_filter",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pre_filter" | "prefilter" | "pre" => Some(Self::PreFilter),
            "inline" | "intra_filter" | "intra" => Some(Self::Inline),
            "post_filter" | "postfilter" | "post" => Some(Self::PostFilter),
            _ => None,
        }
    }
}

/// Catalog-level policy controlling how the planner selects `AnnFilteringMode`.
///
/// Thresholds are selectivity fractions in [0.0, 1.0] where lower = fewer
/// records pass the filter.  The planner reads `estimated_selectivity` from
/// `FilterDiagnostics` (ADR-004) and routes as follows:
///
/// ```text
/// selectivity < pre_filter_threshold   → PreFilter
/// selectivity > post_filter_threshold  → PostFilter
/// otherwise                            → Inline
/// ```
///
/// Defaults are empirically grounded in the FAVOR paper (2605.07770) and the
/// Filtered ANN Survey (2602.11443):
///   pre_filter_threshold  = 0.05  (< 5 % of corpus passes filter → pre-filter)
///   post_filter_threshold = 0.50  (> 50 % passes filter → post-filter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnFilteringPolicy {
    /// Selectivity below which PreFilter is preferred. Default 0.05.
    pub pre_filter_threshold: f64,
    /// Selectivity above which PostFilter is preferred. Default 0.50.
    pub post_filter_threshold: f64,
    /// Override: force a specific mode regardless of selectivity.
    pub force_mode: Option<AnnFilteringMode>,
    /// Minimum candidate oversample factor used in PostFilter mode to compensate
    /// for filter rejection. Default 2.0 (fetch 2× top-k before filtering).
    pub post_filter_oversample_factor: f64,
}

impl Default for AnnFilteringPolicy {
    fn default() -> Self {
        Self {
            pre_filter_threshold: 0.05,
            post_filter_threshold: 0.50,
            force_mode: None,
            post_filter_oversample_factor: 2.0,
        }
    }
}

impl AnnFilteringPolicy {
    /// Determine the routing mode from an estimated selectivity fraction.
    pub fn routing_mode(&self, estimated_selectivity: f64) -> AnnFilteringMode {
        if let Some(forced) = self.force_mode {
            return forced;
        }
        if estimated_selectivity < self.pre_filter_threshold {
            AnnFilteringMode::PreFilter
        } else if estimated_selectivity > self.post_filter_threshold {
            AnnFilteringMode::PostFilter
        } else {
            AnnFilteringMode::Inline
        }
    }

    /// Effective top-k to request from ANN in PostFilter mode (before scalar filtering).
    pub fn effective_top_k_for_post_filter(&self, top_k: usize) -> usize {
        ((top_k as f64) * self.post_filter_oversample_factor).ceil() as usize
    }
}

// ─── Projection Rebuild RTO ───────────────────────────────────────────────────

/// Quantitative rebuild time objective for a rebuildable L2 projection.
///
/// These bounds are operational commitments, not aspirational. A missing RTO
/// means the projection has not been benchmarked and MUST NOT be relied upon
/// for latency-sensitive query paths after a rebuild event.
///
/// Background: ADR-010 mandates that PAX blocks are Layer 2 (rebuildable
/// projections). Without an RTO bound, "rebuildable" is an architectural
/// claim, not an operational guarantee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildRtoSpec {
    /// Estimated rebuild duration in seconds per 10 GB of L1 (ProximaRecord) data.
    /// Planner uses this to estimate total rebuild time from `size_bytes`.
    pub rebuild_seconds_per_10gb: f64,
    /// Whether queries fall back to full L1 scans while this projection rebuilds.
    /// If false, queries on this projection are rejected or degrade to `Unavailable`.
    pub degraded_scan_during_rebuild: bool,
    /// Maximum tolerated rebuild lag at which point queries must route around this
    /// projection. If the rebuild has been running longer than this, the planner
    /// marks the projection `Unavailable` for planning purposes.
    pub max_rebuild_wait_seconds: f64,
    /// Whether this projection supports incremental rebuild from a WAL checkpoint
    /// (faster) or requires a full L1 scan (slower).
    pub supports_incremental_rebuild: bool,
    pub estimate_source: RebuildEstimateSource,
}

const BYTES_PER_10_GIB: f64 = 10.0 * 1024.0 * 1024.0 * 1024.0;

impl Default for RebuildRtoSpec {
    fn default() -> Self {
        Self {
            rebuild_seconds_per_10gb: 60.0,
            degraded_scan_during_rebuild: true,
            max_rebuild_wait_seconds: 300.0,
            supports_incremental_rebuild: false,
            estimate_source: RebuildEstimateSource::Unverified,
        }
    }
}

impl RebuildRtoSpec {
    /// Estimate total rebuild duration in seconds given a projected data volume.
    pub fn estimated_rebuild_seconds(&self, size_bytes: u64) -> f64 {
        let size_gb = (size_bytes as f64) / BYTES_PER_10_GIB;
        size_gb * self.rebuild_seconds_per_10gb
    }

    /// Return an RTO spec for a benchmarked HNSW projection over L1 PAX blocks.
    /// Grounded in FGIM (arXiv:2603.21710) graph-index merge benchmarks.
    pub fn hnsw_benchmarked() -> Self {
        Self {
            rebuild_seconds_per_10gb: 45.0,
            degraded_scan_during_rebuild: true,
            max_rebuild_wait_seconds: 600.0,
            supports_incremental_rebuild: true,
            estimate_source: RebuildEstimateSource::Benchmarked,
        }
    }

    /// Return an RTO spec for a CSR graph topology projection.
    /// CSR rebuild requires a full sorted adjacency scan from L1 edge records.
    pub fn csr_estimated() -> Self {
        Self {
            rebuild_seconds_per_10gb: 90.0,
            degraded_scan_during_rebuild: false,
            max_rebuild_wait_seconds: 900.0,
            supports_incremental_rebuild: false,
            estimate_source: RebuildEstimateSource::Estimated,
        }
    }

    /// Return an RTO spec for a columnar analytics projection (PAX → Parquet).
    pub fn columnar_estimated() -> Self {
        Self {
            rebuild_seconds_per_10gb: 30.0,
            degraded_scan_during_rebuild: true,
            max_rebuild_wait_seconds: 300.0,
            supports_incremental_rebuild: true,
            estimate_source: RebuildEstimateSource::Estimated,
        }
    }
}

// ─── Props Auto-Promotion ─────────────────────────────────────────────────────

/// When props auto-promotion is evaluated: during background compaction or only
/// on explicit DDL (`ALTER TABLE ... PROMOTE PROPS KEY`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PropsEvaluationCadence {
    /// Evaluate promotion candidates automatically during compaction.
    #[default]
    Compaction,
    /// Only promote when the operator issues an explicit DDL command.
    Explicit,
}

impl PropsEvaluationCadence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compaction => "compaction",
            Self::Explicit => "explicit",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "compaction" | "auto" => Some(Self::Compaction),
            "explicit" | "manual" | "ddl" => Some(Self::Explicit),
            _ => None,
        }
    }
}

/// Source of a `RebuildRtoSpec` estimate, distinguishing measured from assumed values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RebuildEstimateSource {
    /// Measured in a controlled benchmark against this deployment's hardware.
    Benchmarked,
    /// Derived from published benchmarks or analogous system measurements.
    Estimated,
    /// No measurement; the value is a placeholder that must not be used for SLA commitments.
    #[default]
    Unverified,
}

impl RebuildEstimateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Benchmarked => "benchmarked",
            Self::Estimated => "estimated",
            Self::Unverified => "unverified",
        }
    }
}

/// Policy for automatically promoting high-frequency document props keys to
/// typed PAX/Iceberg columns. Without promotion, props are stored as opaque
/// msgpack blobs (column ID 8) and cannot be pruned at column or block level,
/// making document queries over `props` effectively full-block scans.
///
/// Promotion creates a new user-defined column (IDs 100+) and reindexes the
/// promoted key into that column during the next compaction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropsAutoPromotionPolicy {
    pub enabled: bool,
    /// Fraction of records that must contain a key before it is promoted.
    /// Default 0.50 — key must appear in ≥ 50 % of records.
    pub frequency_threshold: f64,
    /// Minimum number of records required before promotion eligibility is evaluated.
    /// Prevents premature promotion on sparse datasets. Default 10_000.
    pub min_record_count: u64,
    /// Maximum number of auto-promoted props columns per table. Default 32.
    pub max_promoted_columns: u32,
    /// When promotion candidates are evaluated.
    pub evaluation_cadence: PropsEvaluationCadence,
    /// [`ProximaType`] variants eligible for promotion. Keys whose observed
    /// values do not match an eligible type are retained in the msgpack blob.
    pub eligible_types: Vec<ProximaType>,
    /// Keys that have already been promoted, mapping props key → `props__<key>` column name.
    /// Written by `SchemaChange::PromotePropsKey` and read by the compaction writer
    /// to know which msgpack keys to route into typed columns.
    #[serde(default)]
    pub promoted_keys: HashMap<String, String>,
}

impl Default for PropsAutoPromotionPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency_threshold: 0.50,
            min_record_count: 10_000,
            max_promoted_columns: 32,
            evaluation_cadence: PropsEvaluationCadence::Compaction,
            eligible_types: vec![
                ProximaType::String,
                ProximaType::Int64,
                ProximaType::Float64,
                ProximaType::Boolean,
                ProximaType::TimestampTz(TimeUnit::Nanosecond),
            ],
            promoted_keys: HashMap::new(),
        }
    }
}

impl PropsAutoPromotionPolicy {
    /// Return the default policy for document-heavy tables.
    /// Enables promotion with standard thresholds.
    pub fn document_default() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
}

// ─── Observability Compression Hint ──────────────────────────────────────────

/// Compression strategy hint for observability / time-series ProximaRecords.
///
/// A ProximaRecord per metric sample loses Gorilla-style (Pelkonen et al.,
/// VLDB 2015) cross-sample compression. This hint instructs the storage layer
/// to apply series-aware compression during compaction when PAX blocks are
/// written for observability tables.
///
/// The hint does NOT change the semantic record model — each write remains a
/// canonical ProximaRecord. Compression is applied by the PAX block writer
/// when sorting by (series_key_column, timestamp_column) and encoding the
/// resulting column stripe with delta-delta + varint encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityCompressionHint {
    /// Column name that identifies the time-series key (e.g., metric name + labels hash).
    pub series_key_column: String,
    /// Column name of the timestamp in nanoseconds (usually `__proxima_created_at_ns`).
    pub timestamp_ns_column: String,
    /// Column name of the numeric value to delta-delta encode.
    pub value_column: String,
    /// Target block sort order: sort by (series_key, timestamp_ns) before encoding.
    pub sort_by_series_key: bool,
    /// Delta-delta bit width for timestamp encoding. 0 = auto-select.
    pub timestamp_delta_bits: u8,
    /// Gorilla-style XOR encoding for float value column.
    pub xor_float_encoding: bool,
}

impl Default for ObservabilityCompressionHint {
    fn default() -> Self {
        Self {
            // "series_key" is the conventional user-defined column; operator must
            // set this to the actual metric-name+labels hash column for the table.
            series_key_column: "series_key".to_string(),
            timestamp_ns_column: "__proxima_created_at_ns".to_string(),
            value_column: "value".to_string(),
            sort_by_series_key: true,
            timestamp_delta_bits: 0,
            xor_float_encoding: true,
        }
    }
}

/// Optional relational semantics for a table/profile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationalCapabilities {
    /// Primary key columns, if enforced.
    pub primary_key: Vec<String>,
    /// Unique indexes or constraints.
    pub unique_indexes: Vec<CatalogIndex>,
    /// Non-unique secondary indexes.
    pub secondary_indexes: Vec<CatalogIndex>,
    /// Foreign-key and check constraints.
    pub constraints: Vec<ColumnConstraint>,
    /// Materialized view names derived from this table/profile.
    pub materialized_views: Vec<String>,
    /// Transaction/isolation profile name.
    pub transaction_profile: Option<String>,
    /// Schema evolution policy name.
    pub schema_evolution_policy: Option<String>,
}

impl RelationalCapabilities {
    /// Returns true when this table opts into any relational integrity semantics.
    pub fn has_enforced_semantics(&self) -> bool {
        !self.primary_key.is_empty()
            || !self.unique_indexes.is_empty()
            || !self.secondary_indexes.is_empty()
            || !self.constraints.is_empty()
            || !self.materialized_views.is_empty()
            || self.transaction_profile.is_some()
            || self.schema_evolution_policy.is_some()
    }
}

/// Column constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnConstraint {
    /// UNIQUE constraint on one or more columns
    Unique {
        /// Column names included in the uniqueness constraint
        columns: Vec<String>,
    },
    /// CHECK constraint with SQL expression
    Check {
        /// SQL boolean expression that rows must satisfy
        expression: String,
    },
    /// FOREIGN KEY constraint
    ForeignKey {
        /// Local column names that form the foreign key
        columns: Vec<String>,
        /// Name of the referenced table
        references_table: String,
        /// Column names in the referenced table
        references_columns: Vec<String>,
        /// Action to take when the referenced row is deleted
        on_delete: Option<ReferentialAction>,
        /// Action to take when the referenced row is updated
        on_update: Option<ReferentialAction>,
    },
}

/// Referential actions for foreign key constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferentialAction {
    /// CASCADE - propagate changes to referencing rows
    Cascade,
    /// SET NULL - set referencing columns to NULL
    SetNull,
    /// SET DEFAULT - set referencing columns to their default values
    SetDefault,
    /// RESTRICT - prevent the action if there are referencing rows
    Restrict,
    /// NO ACTION - similar to RESTRICT but deferred
    NoAction,
}

/// Where a collection is in its precision-migration lifecycle. A migration
/// from fp32 → fp16 typically goes `Stable → ShadowingTarget →
/// CutoverPending → Stable` over multiple compactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMigrationState {
    Stable,
    ShadowingTarget,
    CutoverPending,
    RollingBack,
}

/// Per-metric recall@K target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecallTargets {
    pub at_10: f32,
    pub at_100: f32,
}

/// Per-distance-metric recall SLO. LLD §Q13 locks the defaults that ship
/// with the global default policy — operators can override per policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecallSlo {
    pub cosine: RecallTargets,
    pub l2: RecallTargets,
    pub dot: RecallTargets,
}

impl Default for RecallSlo {
    fn default() -> Self {
        Self::lld_defaults()
    }
}

impl RecallSlo {
    /// LLD §Q13 per-metric recall defaults. Cosine + L2 share fp16-noise
    /// tolerance (normalized magnitudes); dot product needs tighter recall
    /// because raw magnitude affects ranking.
    pub const fn lld_defaults() -> Self {
        Self {
            cosine: RecallTargets {
                at_10: 0.99,
                at_100: 0.995,
            },
            l2: RecallTargets {
                at_10: 0.99,
                at_100: 0.995,
            },
            dot: RecallTargets {
                at_10: 0.995,
                at_100: 0.998,
            },
        }
    }
}

/// Do two column lists denote the same key? Order-insensitive: `UNIQUE(a,b)` and
/// `UNIQUE(b,a)` fence the same tuple, so they are one key, not two.
pub fn same_column_set(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a: Vec<&str> = a.iter().map(String::as_str).collect();
    let mut b: Vec<&str> = b.iter().map(String::as_str).collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

/// Validate a content-addressed `sha256:<64 lowercase hex chars>` digest (the
/// model-contract digest pinned on [`CatalogEmbeddingConfig`]). Pure format
/// check — duplicated from the catalog mlops module so this foundation crate
/// need not depend on the control-layer catalog for a string validation.
pub(crate) fn validate_contract_digest(value: &str) -> anyhow::Result<()> {
    let field = "model contract digest";
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(anyhow!(
            "{field} must be a sha256 digest in the form sha256:<64 lowercase hex chars>"
        ));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(anyhow!(
            "{field} must be a sha256 digest in the form sha256:<64 lowercase hex chars>"
        ));
    }
    Ok(())
}
