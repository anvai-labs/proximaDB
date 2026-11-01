//! Unified API Handlers for the platform runtime layer.
//!
//! ## Architecture
//!
//! `UnifiedHandlers` is the composition root that wires together injected service
//! ports into a coherent API surface.  Each service dependency is expressed as a
//! port trait (`CollectionPort`, `VectorOpsPort`, `QueryAdapterPort`) so no
//! root-crate concrete type crosses the crate boundary.
//!
//! ## Migration status
//!
//! This is the **sole production** `ApiHandlersPort` implementation. The legacy
//! root-crate twin (`src/api_handlers/request_handlers.rs`) was retired in
//! TD-104 S3-f — every REST/gRPC/Arrow/pgwire/embedded surface now routes
//! through this handler, which delegates to injected service ports.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_data_model::{ProximaType, ProximaValue, TimeUnit};
use proximadb_proto::v1::{
    Collection, CollectionConfig, CollectionOperation, CollectionRequest, CollectionResponse,
    ExecuteQueryResponse, FilterableColumnSpec, FilterableDataType, HybridSearchRequest,
    HybridSearchResponse, RecordSchemaConfig, SqlRow, SqlRowField, TextStorageConfig,
    VectorBatchRequest, VectorOperationResponse, VectorSearchRequest,
};
use proximadb_records::conversions::proxima_to_sql_value;

use crate::port::{
    ApiHandlersPort, CollectionSchemaColumn, CollectionSchemaEnforcement, CollectionSchemaMetadata,
    CollectionSchemaUpdate, CollectionTextStorage,
};
use crate::service_ports::{
    CollectionPort, PortIdentity, QueryAdapterPort, SqlExecutionResult, VectorOpsPort,
};

/// Global request counter for generating unique request IDs.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a 16-char hex request ID (8 chars timestamp + 8 chars counter).
pub fn generate_request_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u32)
        .unwrap_or(0);
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed) as u32;
    format!("{:08x}{:08x}", timestamp, counter)
}

// ── Collection ID cache ───────────────────────────────────────────────────────

const COLLECTION_ID_CACHE_TTL_SECS: u64 = 300;
const COLLECTION_ID_CACHE_MAX_SIZE: usize = 1000;

#[derive(Clone)]
struct CollectionIdCacheEntry {
    collection_id: String,
    cached_at: Instant,
}

/// Thread-safe TTL-based cache for collection ID resolution.
///
/// Reduces metadata backend lookups from ~5 ms/request to ~0.1 ms on cache hits.
pub struct CollectionIdCache {
    cache: std::sync::RwLock<HashMap<String, CollectionIdCacheEntry>>,
    ttl: Duration,
    max_size: usize,
}

impl CollectionIdCache {
    pub fn new() -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(COLLECTION_ID_CACHE_TTL_SECS),
            max_size: COLLECTION_ID_CACHE_MAX_SIZE,
        }
    }

    pub fn get(&self, collection_name: &str) -> Option<String> {
        let cache = self.cache.read().ok()?;
        let entry = cache.get(collection_name)?;
        if entry.cached_at.elapsed() > self.ttl {
            return None;
        }
        Some(entry.collection_id.clone())
    }

    pub fn put(&self, collection_name: String, collection_id: String) {
        if let Ok(mut cache) = self.cache.write() {
            let now = Instant::now();
            cache.retain(|_, entry| now.duration_since(entry.cached_at) < self.ttl);
            while cache.len() >= self.max_size {
                let oldest_key = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.cached_at)
                    .map(|(key, _)| key.clone());
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                } else {
                    break;
                }
            }
            cache.insert(
                collection_name,
                CollectionIdCacheEntry {
                    collection_id,
                    cached_at: now,
                },
            );
        }
    }
}

impl Default for CollectionIdCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Placeholder for hybrid runtime config ─────────────────────────────────────

/// Placeholder for hybrid runtime configuration (weights, seeding).
pub struct HybridRuntimeConfig;

// ── UnifiedHandlers ───────────────────────────────────────────────────────────

/// Canonical runtime-native API handler — the seam new schema/port work targets.
///
/// This is one of two same-named handler types. Prefer this one: it delegates
/// to the runtime ports (`CollectionPort` / `VectorOpsPort`) and, on the schema
/// surface, speaks the runtime-native contract (`CollectionSchemaMetadata`,
/// built on canonical `ProximaType`) with no v1 envelope. Note the trait still
/// carries legacy v1-proto methods (`handle_collection_operation_for_tenant`,
/// `handle_vector_search_v1_*`, `execute_sql_v1`, …) that this impl bridges to
/// the ports; those retire with the TD-123 v1→v2 message migration.
///
/// The legacy root-crate twin (`api_handlers::UnifiedHandlers`) was retired in
/// TD-104 S3-f: its orchestration was extracted onto runtime ports / concrete
/// services, and every network surface (REST/gRPC/Arrow/pgwire/embedded) now
/// routes through THIS handler. There is no longer a root-crate `ApiHandlersPort`
/// impl — this is the sole production handler.
///
/// Composition root that wires service ports into the API surface.
///
/// Hold `Arc<dyn CollectionPort>`, `Arc<dyn VectorOpsPort>`, and optionally an
/// `Arc<dyn QueryAdapterPort>` so the actual business logic can be injected at
/// server startup (root crate's concrete services) without `proximadb-runtime`
/// knowing about their implementations.
pub struct UnifiedHandlers {
    /// Collection lifecycle operations.
    pub collection: Arc<dyn CollectionPort>,
    /// Vector CRUD and search operations.
    pub vector_ops: Arc<dyn VectorOpsPort>,
    /// Optional unified query facade for SQL / hybrid routing.
    pub query_adapter: Option<Arc<dyn QueryAdapterPort>>,
    /// Optional hybrid runtime configuration.
    pub hybrid_runtime: Arc<std::sync::RwLock<Option<HybridRuntimeConfig>>>,
    /// Cache for collection ID resolution.
    pub collection_id_cache: CollectionIdCache,
}

impl UnifiedHandlers {
    /// Construct from injected service port implementations.
    pub fn new(
        collection: Arc<dyn CollectionPort>,
        vector_ops: Arc<dyn VectorOpsPort>,
        query_adapter: Option<Arc<dyn QueryAdapterPort>>,
    ) -> Self {
        Self {
            collection,
            vector_ops,
            query_adapter,
            hybrid_runtime: Arc::new(std::sync::RwLock::new(None)),
            collection_id_cache: CollectionIdCache::new(),
        }
    }
}

/// TD-COLLTIME-1: `Collection.created_at`/`updated_at` are Unix epoch
/// MILLISECONDS, but collections created before the unit fix carry
/// MICROseconds (the manager and engine paths wrote `timestamp_micros()`).
/// A millisecond epoch is ~1.7-2.0e12 today; a microsecond epoch is ~1000x
/// that — anything above 1e14 is unambiguously microseconds, and normalizing
/// on read makes stored pre-fix collections render correctly without a
/// migration (pre-GA: no back-compat gate, per the 2026-07-14 directive).
pub fn normalize_epoch_ms(value: i64) -> i64 {
    if value > 100_000_000_000_000 {
        value / 1000
    } else {
        value
    }
}

fn schema_metadata_from_collection(collection: &Collection) -> CollectionSchemaMetadata {
    let Some(config) = collection.config.as_ref() else {
        return CollectionSchemaMetadata {
            collection_id: collection.id.clone(),
            created_at_ms: normalize_epoch_ms(collection.created_at),
            updated_at_ms: normalize_epoch_ms(collection.updated_at),
            ..CollectionSchemaMetadata::default()
        };
    };

    let mut columns = config
        .text_columns
        .iter()
        .map(|name| CollectionSchemaColumn {
            name: name.clone(),
            data_type: ProximaType::String,
            nullable: true,
            indexed: false,
            filterable: true,
            text_storage: Some(CollectionTextStorage::Inline),
            max_length: None,
        })
        .collect::<Vec<_>>();

    for text_config in &config.text_storage_configs {
        if let Some(existing) = columns
            .iter_mut()
            .find(|column| column.name == text_config.column_name)
        {
            existing.text_storage = Some(CollectionTextStorage::Large);
            existing.max_length = Some(text_config.chunk_size);
        } else {
            columns.push(CollectionSchemaColumn {
                name: text_config.column_name.clone(),
                data_type: ProximaType::String,
                nullable: true,
                indexed: false,
                filterable: false,
                text_storage: Some(CollectionTextStorage::Large),
                max_length: Some(text_config.chunk_size),
            });
        }
    }

    let existing_column_names = columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<std::collections::HashSet<_>>();
    columns.extend(config.filterable_columns.iter().filter_map(|column| {
        if existing_column_names.contains(&column.name) {
            return None;
        }
        filterable_type_to_proxima_type(column.data_type).map(|data_type| CollectionSchemaColumn {
            name: column.name.clone(),
            data_type,
            nullable: true,
            indexed: column.indexed,
            filterable: true,
            text_storage: None,
            max_length: None,
        })
    }));

    let record_schema = config.record_schema.as_ref();
    CollectionSchemaMetadata {
        collection_id: collection.id.clone(),
        created_at_ms: normalize_epoch_ms(collection.created_at),
        updated_at_ms: normalize_epoch_ms(collection.updated_at),
        schema_id: record_schema.map(|schema| schema.schema_id.clone()),
        schema_version: record_schema.map(|schema| schema.schema_version.clone()),
        enforcement: record_schema.map(|schema| enforcement_from_i32(schema.enforcement)),
        auto_evolve: record_schema.is_none_or(|schema| schema.auto_evolve),
        // TD-GRPCSCHEMA-2: `enabled` is the enforcement answer — key it on the
        // flag alone, exactly like every write-path gate (sst/flush shred spec,
        // props promotion). The old `|| record_schema.is_some()` disjunct was
        // the heuristic for the era when the flag never persisted; now a
        // pinned-OFF create can carry a record_schema block, and it made this
        // read ON (contradicting REST's flag-derived view) for a collection the
        // caller explicitly created OFF.
        enabled: config.enable_proxima_record.unwrap_or(false),
        columns,
    }
}

fn apply_schema_update(
    config: &mut CollectionConfig,
    update: &CollectionSchemaUpdate,
) -> Result<()> {
    config.record_schema = Some(RecordSchemaConfig {
        schema_id: update.schema_id.clone(),
        schema_version: update.schema_version.clone(),
        enforcement: enforcement_to_i32(update.enforcement),
        auto_evolve: update.auto_evolve,
        columns: Vec::new(),
    });
    config.enable_proxima_record = Some(true);
    config.text_columns = update
        .columns
        .iter()
        .filter(|column| matches!(column.data_type, ProximaType::String))
        .map(|column| column.name.clone())
        .collect();
    config.text_storage_configs = update
        .columns
        .iter()
        .filter(|column| {
            matches!(column.data_type, ProximaType::String)
                && matches!(column.text_storage, Some(CollectionTextStorage::Large))
        })
        .map(|column| TextStorageConfig {
            column_name: column.name.clone(),
            strategy: 1,
            inline_threshold: 4096,
            chunked_threshold: 1_048_576,
            chunk_size: column.max_length.unwrap_or(512),
            ..Default::default()
        })
        .collect();
    config.filterable_columns = update
        .columns
        .iter()
        .filter(|column| column.filterable)
        .filter_map(|column| {
            proxima_type_to_filterable_type(&column.data_type).map(|data_type| {
                FilterableColumnSpec {
                    name: column.name.clone(),
                    data_type,
                    indexed: column.indexed,
                    supports_range: supports_range_filter(&column.data_type),
                    estimated_cardinality: None,
                }
            })
        })
        .collect();

    // ADR-047 / TD-TBL-1: the narrow v1 projection above is best-effort — it can
    // only carry text + the `FilterableDataType` vocabulary. Columns it cannot
    // represent (UInt/Struct/Map/Sparse/BinaryVector …) are preserved verbatim
    // in the canonical sidecar written by `set_collection_schema_columns`; they
    // are no longer rejected, so the canonical authority is never lossy.
    for column in &update.columns {
        let storable_text = matches!(column.data_type, ProximaType::String);
        let storable_filterable =
            column.filterable && proxima_type_to_filterable_type(&column.data_type).is_some();
        if !storable_text && !storable_filterable {
            tracing::debug!(
                "schema column '{}' (type {:?}) preserved only in canonical sidecar \
                 (not representable in the narrow v1 collection config)",
                column.name,
                column.data_type
            );
        }
    }
    Ok(())
}

fn enforcement_from_i32(value: i32) -> CollectionSchemaEnforcement {
    match value {
        1 => CollectionSchemaEnforcement::Strict,
        2 => CollectionSchemaEnforcement::Flexible,
        _ => CollectionSchemaEnforcement::Hybrid,
    }
}

fn enforcement_to_i32(value: CollectionSchemaEnforcement) -> i32 {
    match value {
        CollectionSchemaEnforcement::Strict => 1,
        CollectionSchemaEnforcement::Flexible => 2,
        CollectionSchemaEnforcement::Hybrid => 3,
    }
}

fn filterable_type_to_proxima_type(value: i32) -> Option<ProximaType> {
    match FilterableDataType::try_from(value).ok()? {
        FilterableDataType::FilterableInteger => Some(ProximaType::Int64),
        FilterableDataType::FilterableFloat => Some(ProximaType::Float64),
        FilterableDataType::FilterableDecimal => Some(ProximaType::Decimal {
            precision: 38,
            scale: 18,
        }),
        FilterableDataType::FilterableBoolean => Some(ProximaType::Boolean),
        FilterableDataType::FilterableDatetime => {
            Some(ProximaType::Timestamp(TimeUnit::Nanosecond))
        }
        FilterableDataType::FilterableTimestampTz => {
            Some(ProximaType::TimestampTz(TimeUnit::Nanosecond))
        }
        FilterableDataType::FilterableDate => Some(ProximaType::Date),
        FilterableDataType::FilterableTime => Some(ProximaType::Time(TimeUnit::Nanosecond)),
        FilterableDataType::FilterableUuid => Some(ProximaType::Uuid),
        FilterableDataType::FilterableJson => Some(ProximaType::Json),
        FilterableDataType::FilterableArrayString => {
            Some(ProximaType::Array(Box::new(ProximaType::String)))
        }
        FilterableDataType::FilterableArrayInteger => {
            Some(ProximaType::Array(Box::new(ProximaType::Int64)))
        }
        FilterableDataType::FilterableArrayFloat => {
            Some(ProximaType::Array(Box::new(ProximaType::Float64)))
        }
        FilterableDataType::FilterableArrayBoolean => {
            Some(ProximaType::Array(Box::new(ProximaType::Boolean)))
        }
        _ => None,
    }
}

/// The v1 filterable type a canonical column narrows to (`None` when it has no
/// filterable projection). Shared so the catalog can tell whether a declared
/// column is compatible with the filterable column already stored for it.
/// TD-CATREPLACE-1: the ONE mapping lives in the foundation data-model crate;
/// this re-export keeps the historical path working.
pub use proximadb_data_model::proxima_type_to_filterable_type;

fn supports_range_filter(value: &ProximaType) -> bool {
    matches!(
        value,
        ProximaType::Int8
            | ProximaType::Int16
            | ProximaType::Int32
            | ProximaType::Int64
            | ProximaType::Float32
            | ProximaType::Float64
            | ProximaType::Decimal { .. }
            | ProximaType::Timestamp(_)
            | ProximaType::TimestampTz(_)
            | ProximaType::Date
            | ProximaType::Time(_)
    )
}

// ── ApiHandlersPort implementation ───────────────────────────────────────────
// CANONICAL impl — and the sole production one. New schema/port methods belong
// here (runtime-native shape). (The legacy root-crate twin that used to bridge
// to v1 CollectionRequest/CollectionOperation was deleted in TD-104 S3-f.)

#[async_trait]
impl ApiHandlersPort for UnifiedHandlers {
    async fn handle_collection_operation_for_tenant(
        &self,
        request: CollectionRequest,
        tenant_id: Option<&str>,
    ) -> Result<CollectionResponse> {
        let op = CollectionOperation::try_from(request.operation)
            .unwrap_or(CollectionOperation::Unspecified);
        let collection_id = request.collection_id.as_deref().unwrap_or("");
        let start = Instant::now();

        let mut resp = CollectionResponse {
            operation: request.operation,
            ..Default::default()
        };

        match op {
            CollectionOperation::CollectionCreate => {
                let config = request
                    .collection_config
                    .ok_or_else(|| anyhow!("collection_config required for CREATE"))?;
                let col = self.collection.create_collection(config, tenant_id).await?;
                resp.success = true;
                resp.collection = Some(col);
            }
            CollectionOperation::CollectionUpdate => {
                let config = request
                    .collection_config
                    .ok_or_else(|| anyhow!("collection_config required for UPDATE"))?;
                let col = self
                    .collection
                    .update_collection(collection_id, config, tenant_id)
                    .await?;
                resp.success = true;
                resp.collection = Some(col);
            }
            CollectionOperation::CollectionGet => {
                let mut col = self
                    .collection
                    .get_collection(collection_id, tenant_id)
                    .await?;
                if let Some(collection) = col.as_mut() {
                    self.collection.enrich_collection_status(collection).await;
                }
                resp.success = col.is_some();
                resp.collection = col;
            }
            CollectionOperation::CollectionList => {
                let cols = self.collection.list_collections(tenant_id).await?;
                resp.success = true;
                resp.total_count = cols.len() as i64;
                resp.collections = cols;
            }
            CollectionOperation::CollectionDelete => {
                let deleted = self
                    .collection
                    .delete_collection(collection_id, tenant_id)
                    .await?;
                resp.success = deleted;
                resp.affected_count = if deleted { 1 } else { 0 };
            }
            CollectionOperation::CollectionGetIdByName => {
                // Resolved within the request tenant, like CollectionGet.
                let resolved = self
                    .collection
                    .get_collection(collection_id, tenant_id)
                    .await?
                    .map(|collection| collection.id);
                resp.success = resolved.is_some();
                if let Some(id) = resolved {
                    resp.metadata.insert("collection_id".to_string(), id);
                }
            }
            CollectionOperation::CollectionMigrate | CollectionOperation::Unspecified => {
                return Err(anyhow!("collection operation {:?} not implemented", op));
            }
        }

        resp.processing_time_us = start.elapsed().as_micros() as i64;
        Ok(resp)
    }

    async fn get_collection_schema_metadata(
        &self,
        collection_id: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<CollectionSchemaMetadata>> {
        let collection = self
            .collection
            .get_collection(collection_id, tenant_id)
            .await?;
        let Some(collection) = collection else {
            return Ok(None);
        };
        let mut metadata = schema_metadata_from_collection(&collection);
        // ADR-047 / TD-TBL-1: the canonical sidecar (if present) is authoritative;
        // otherwise keep the narrow-derived view (legacy collection fallback).
        if let Some(canonical) = self
            .collection
            .get_collection_schema_columns(collection_id, tenant_id)
            .await?
        {
            metadata.columns = canonical;
        }
        Ok(Some(metadata))
    }

    async fn update_collection_schema_metadata(
        &self,
        collection_id: &str,
        update: CollectionSchemaUpdate,
        tenant_id: Option<&str>,
    ) -> Result<CollectionSchemaMetadata> {
        // A schema UPDATE turns record mode on (the SDK semantic): delegate
        // with the mode unpinned. TD-GRPCSCHEMA-2.
        self.update_collection_schema_metadata_with_record_mode(
            collection_id,
            update,
            tenant_id,
            None,
        )
        .await
    }

    async fn update_collection_schema_metadata_with_record_mode(
        &self,
        collection_id: &str,
        update: CollectionSchemaUpdate,
        tenant_id: Option<&str>,
        record_mode: Option<bool>,
    ) -> Result<CollectionSchemaMetadata> {
        let collection = self
            .collection
            .get_collection(collection_id, tenant_id)
            .await?
            .ok_or_else(|| anyhow!("collection not found: {collection_id}"))?;
        let mut config = collection.config.clone().unwrap_or_default();
        // Narrow projection (best-effort); the canonical columns are persisted
        // verbatim via the sidecar below so rich ProximaType variants survive.
        apply_schema_update(&mut config, &update)?;
        if let Some(mode) = record_mode {
            config.enable_proxima_record = Some(mode);
        }
        // TD-CATREPLACE-1: canonical FIRST. The canonical write validates the
        // whole update (including the fold against the stored filterable
        // columns); a rejection here leaves the narrow config untouched — the
        // old narrow-first order persisted it before the rejection and the
        // collection came back half-applied (new filterable columns, old
        // canonical schema). Only when the catalog accepted the schema is the
        // narrow config replaced.
        self.collection
            .set_collection_schema_columns(collection_id, &update.columns, tenant_id)
            .await?;
        let updated = self
            .collection
            .update_collection(collection_id, config, tenant_id)
            .await?;
        // Return the canonical columns as the authoritative view (the narrow
        // config above cannot represent them, so re-deriving would lose them).
        let mut metadata = schema_metadata_from_collection(&updated);
        metadata.columns = update.columns;
        Ok(metadata)
    }

    async fn handle_vector_search_v1_for_tenant(
        &self,
        request: VectorSearchRequest,
        identity: PortIdentity<'_>,
    ) -> Result<VectorOperationResponse> {
        self.vector_ops.search(request, identity).await
    }

    async fn handle_vector_search_v1(
        &self,
        request: VectorSearchRequest,
    ) -> Result<VectorOperationResponse> {
        self.vector_ops
            .search(request, PortIdentity::anonymous())
            .await
    }

    async fn handle_vector_batch_v1_for_tenant(
        &self,
        request: VectorBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse> {
        self.vector_ops.batch_upsert(request, tenant_id).await
    }

    async fn handle_vector_v1_for_tenant(
        &self,
        collection_id: &str,
        vector_id: &str,
        include_vector: bool,
        include_metadata: bool,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse> {
        self.vector_ops
            .get_vector(
                collection_id,
                vector_id,
                include_vector,
                include_metadata,
                tenant_id,
            )
            .await
    }

    async fn execute_hybrid_query(
        &self,
        request: HybridSearchRequest,
    ) -> Result<HybridSearchResponse> {
        let adapter = self
            .query_adapter
            .as_ref()
            .ok_or_else(|| anyhow!("hybrid query requires QueryAdapterPort (not wired)"))?;
        adapter.execute_hybrid(request).await
    }

    async fn execute_sql(
        &self,
        query: String,
        _parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<SqlExecutionResult> {
        let adapter = self
            .query_adapter
            .as_ref()
            .ok_or_else(|| anyhow!("SQL execution requires QueryAdapterPort (not wired)"))?;

        let start = Instant::now();
        let mut result = adapter.execute_sql(query, collection, identity).await?;
        result.execution_time_ms = start.elapsed().as_millis() as u64;
        if result.rows_scanned == 0 {
            result.rows_scanned = result.rows_affected.unwrap_or(result.rows.len() as u64);
        }
        Ok(result)
    }

    async fn execute_sql_v1(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<ExecuteQueryResponse> {
        let result = self
            .execute_sql(query, parameters, collection, identity)
            .await?;
        let rows_returned = result.rows_affected.unwrap_or(result.rows.len() as u64);
        let rows = result
            .rows
            .iter()
            .map(|row| SqlRow {
                fields: row
                    .iter()
                    .enumerate()
                    .map(|(index, value)| SqlRowField {
                        key: result
                            .columns
                            .get(index)
                            .cloned()
                            .unwrap_or_else(|| format!("column_{index}")),
                        value: Some(proxima_to_sql_value(value)),
                    })
                    .collect(),
                similarity: None,
            })
            .collect();

        Ok(ExecuteQueryResponse {
            rows,
            rows_scanned: result.rows_scanned,
            rows_returned,
            execution_time_ms: result.execution_time_ms,
            columns: result.columns,
            column_types: result.column_types,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn normalize_epoch_ms_converts_stored_microsecond_values() {
        // TD-COLLTIME-1: pre-fix writers stored microseconds; anything above
        // 1e14 is unambiguously µs and is divided to milliseconds, while
        // correct millisecond values (and 0/negative sentinels) pass through.
        let ms_now = 1_735_000_000_123_i64; // ~2024-11, epoch ms
        assert_eq!(normalize_epoch_ms(ms_now), ms_now);
        assert_eq!(normalize_epoch_ms(0), 0);
        assert_eq!(normalize_epoch_ms(-1), -1);
        let us_now = ms_now * 1000; // the same instant, stored in µs
        assert_eq!(normalize_epoch_ms(us_now), ms_now);
    }

    use proximadb_proto::v1::{Collection, CollectionConfig, VectorRecord, sql_value};
    use serde_json::json;

    #[derive(Default)]
    struct MockCollectionPort {
        calls: Mutex<Vec<String>>,
        /// When set, `set_collection_schema_columns` fails (the canonical write
        /// rejecting the update — TD-CATREPLACE-1).
        fail_canonical: bool,
    }

    #[async_trait]
    impl CollectionPort for MockCollectionPort {
        async fn get_collection(
            &self,
            identifier: &str,
            tenant_id: Option<&str>,
        ) -> Result<Option<Collection>> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("get:{identifier}:{tenant_id:?}"));
            Ok(Some(Collection {
                id: identifier.to_string(),
                ..Collection::default()
            }))
        }

        async fn create_collection(
            &self,
            config: CollectionConfig,
            tenant_id: Option<&str>,
        ) -> Result<Collection> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("create:{}:{tenant_id:?}", config.name));
            Ok(Collection {
                id: config.name.clone(),
                config: Some(config),
                ..Collection::default()
            })
        }

        async fn update_collection(
            &self,
            id: &str,
            config: CollectionConfig,
            tenant_id: Option<&str>,
        ) -> Result<Collection> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("update:{id}:{}:{tenant_id:?}", config.name));
            Ok(Collection {
                id: id.to_string(),
                config: Some(config),
                ..Collection::default()
            })
        }

        async fn set_collection_schema_columns(
            &self,
            id: &str,
            _columns: &[CollectionSchemaColumn],
            _tenant_id: Option<&str>,
        ) -> Result<()> {
            if self.fail_canonical {
                self.calls
                    .lock()
                    .unwrap()
                    .push(format!("set_columns:{id}:FAIL"));
                return Err(anyhow!("canonical schema rejected"));
            }
            self.calls.lock().unwrap().push(format!("set_columns:{id}"));
            Ok(())
        }

        async fn delete_collection(&self, id: &str, tenant_id: Option<&str>) -> Result<bool> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("delete:{id}:{tenant_id:?}"));
            Ok(true)
        }

        async fn list_collections(&self, tenant_id: Option<&str>) -> Result<Vec<Collection>> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("list:{tenant_id:?}"));
            Ok(vec![Collection {
                id: "docs".to_string(),
                ..Collection::default()
            }])
        }

        async fn resolve_collection_id(&self, identifier: &str) -> Result<Option<String>> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("resolve:{identifier}"));
            Ok(Some(format!("{identifier}-id")))
        }
    }

    #[derive(Default)]
    struct MockVectorOpsPort {
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl VectorOpsPort for MockVectorOpsPort {
        async fn search(
            &self,
            request: VectorSearchRequest,
            identity: PortIdentity<'_>,
        ) -> Result<VectorOperationResponse> {
            let tenant_id = identity.tenant_id;
            self.calls
                .lock()
                .unwrap()
                .push(format!("search:{}:{tenant_id:?}", request.collection_id));
            Ok(VectorOperationResponse::default())
        }

        async fn batch_upsert(
            &self,
            request: VectorBatchRequest,
            tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            self.calls.lock().unwrap().push(format!(
                "batch:{}:{}:{tenant_id:?}",
                request.collection_id,
                request.vectors.len()
            ));
            Ok(VectorOperationResponse::default())
        }

        async fn get_vector(
            &self,
            collection_id: &str,
            vector_id: &str,
            include_vector: bool,
            include_metadata: bool,
            tenant_id: Option<&str>,
        ) -> Result<VectorOperationResponse> {
            self.calls.lock().unwrap().push(format!(
                "get:{collection_id}:{vector_id}:{include_vector}:{include_metadata}:{tenant_id:?}"
            ));
            Ok(VectorOperationResponse::default())
        }

        async fn flush_all(&self) -> Result<()> {
            Ok(())
        }

        async fn metrics(&self) -> Result<serde_json::Value> {
            Ok(json!({"ok": true}))
        }
    }

    #[derive(Default)]
    struct MockQueryAdapterPort;

    #[async_trait]
    impl QueryAdapterPort for MockQueryAdapterPort {
        async fn vector_search(
            &self,
            _request: VectorSearchRequest,
        ) -> Result<VectorOperationResponse> {
            Ok(VectorOperationResponse::default())
        }

        async fn execute_hybrid(
            &self,
            _request: HybridSearchRequest,
        ) -> Result<HybridSearchResponse> {
            Ok(HybridSearchResponse::default())
        }

        async fn execute_sql(
            &self,
            _query: String,
            _collection: Option<String>,
            _identity: crate::service_ports::PortIdentity<'_>,
        ) -> Result<SqlExecutionResult> {
            Ok(SqlExecutionResult {
                columns: vec!["id", "score", "flag", "none", "obj"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                column_types: vec!["TEXT", "FLOAT", "BOOLEAN", "NULL", "JSON"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                rows: vec![vec![
                    ProximaValue::String("r1".to_string()),
                    ProximaValue::Float64(1.5),
                    ProximaValue::Boolean(true),
                    ProximaValue::Null,
                    proximadb_records::conversions::json_to_proxima(&json!({"nested": 1})),
                ]],
                rows_scanned: 9,
                ..Default::default()
            })
        }
    }

    fn make_handlers(
        collection: Arc<MockCollectionPort>,
        vector_ops: Arc<MockVectorOpsPort>,
        query_adapter: Option<Arc<dyn QueryAdapterPort>>,
    ) -> UnifiedHandlers {
        UnifiedHandlers::new(collection, vector_ops, query_adapter)
    }

    fn collection_request(operation: CollectionOperation) -> CollectionRequest {
        CollectionRequest {
            operation: operation as i32,
            collection_id: Some("docs".to_string()),
            collection_config: Some(CollectionConfig {
                name: "docs".to_string(),
                ..CollectionConfig::default()
            }),
            ..CollectionRequest::default()
        }
    }

    #[test]
    fn request_ids_are_hex_length_stable_and_collection_id_cache_handles_ttl_and_eviction() {
        let first = generate_request_id();
        let second = generate_request_id();
        assert_eq!(first.len(), 16);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, second);

        let cache = CollectionIdCache::new();
        cache.put("docs".to_string(), "docs-id".to_string());
        assert_eq!(cache.get("docs").as_deref(), Some("docs-id"));
        assert_eq!(CollectionIdCache::default().get("missing"), None);

        let expiring = CollectionIdCache {
            cache: std::sync::RwLock::new(HashMap::new()),
            ttl: Duration::from_nanos(0),
            max_size: 1,
        };
        expiring.put("old".to_string(), "old-id".to_string());
        assert_eq!(expiring.get("old"), None);

        let bounded = CollectionIdCache {
            cache: std::sync::RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(60),
            max_size: 1,
        };
        bounded.put("a".to_string(), "a-id".to_string());
        bounded.put("b".to_string(), "b-id".to_string());
        assert_eq!(bounded.get("a"), None);
        assert_eq!(bounded.get("b").as_deref(), Some("b-id"));
    }

    #[tokio::test]
    async fn unified_handlers_route_collection_operations_to_collection_port() {
        let collection = Arc::new(MockCollectionPort::default());
        let vector_ops = Arc::new(MockVectorOpsPort::default());
        let handlers = make_handlers(collection.clone(), vector_ops, None);

        for operation in [
            CollectionOperation::CollectionCreate,
            CollectionOperation::CollectionUpdate,
            CollectionOperation::CollectionGet,
            CollectionOperation::CollectionList,
            CollectionOperation::CollectionDelete,
            CollectionOperation::CollectionGetIdByName,
        ] {
            let response = handlers
                .handle_collection_operation_for_tenant(
                    collection_request(operation),
                    Some("tenant-a"),
                )
                .await
                .unwrap();
            assert!(response.success);
            assert!(response.processing_time_us >= 0);
        }

        let calls = collection.calls.lock().unwrap().clone();
        assert!(calls.iter().any(|call| call.starts_with("create:docs")));
        assert!(calls.iter().any(|call| call.starts_with("update:docs")));
        assert!(calls.iter().any(|call| call.starts_with("get:docs")));
        assert!(calls.iter().any(|call| call.starts_with("list:")));
        assert!(calls.iter().any(|call| call.starts_with("delete:docs")));
        // CollectionGetIdByName resolves within the request tenant (a scoped
        // get), not through the unscoped name resolver.
        assert!(!calls.iter().any(|call| call == "resolve:docs"));
        assert!(
            calls
                .iter()
                .filter(|call| call.starts_with("get:docs"))
                .count()
                >= 2
        );

        let mut missing_config = collection_request(CollectionOperation::CollectionCreate);
        missing_config.collection_config = None;
        assert!(
            handlers
                .handle_collection_operation_for_tenant(missing_config, None)
                .await
                .unwrap_err()
                .to_string()
                .contains("collection_config required")
        );
        assert!(
            handlers
                .handle_collection_operation_for_tenant(
                    collection_request(CollectionOperation::Unspecified),
                    None,
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("not implemented")
        );
    }

    #[tokio::test]
    // Deliberately exercises the deprecated v1 ExecuteHybridQuery dispatch (TD-143).
    #[allow(deprecated)]
    async fn unified_handlers_route_vector_hybrid_and_sql_operations() {
        let collection = Arc::new(MockCollectionPort::default());
        let vector_ops = Arc::new(MockVectorOpsPort::default());
        let handlers = make_handlers(
            collection,
            vector_ops.clone(),
            Some(Arc::new(MockQueryAdapterPort)),
        );

        handlers
            .handle_vector_search_v1(VectorSearchRequest {
                collection_id: "global".to_string(),
                ..VectorSearchRequest::default()
            })
            .await
            .unwrap();
        handlers
            .handle_vector_search_v1_for_tenant(
                VectorSearchRequest {
                    collection_id: "tenant".to_string(),
                    ..VectorSearchRequest::default()
                },
                PortIdentity::for_tenant("tenant-a"),
            )
            .await
            .unwrap();
        handlers
            .handle_vector_batch_v1_for_tenant(
                VectorBatchRequest {
                    collection_id: "docs".to_string(),
                    vectors: vec![VectorRecord {
                        id: "v1".to_string(),
                        vector: vec![0.1],
                        ..VectorRecord::default()
                    }],
                },
                Some("tenant-a"),
            )
            .await
            .unwrap();
        handlers
            .handle_vector_v1_for_tenant("docs", "v1", true, false, Some("tenant-a"))
            .await
            .unwrap();

        assert_eq!(vector_ops.calls.lock().unwrap().len(), 4);
        assert!(
            handlers
                .execute_hybrid_query(HybridSearchRequest::default())
                .await
                .is_ok()
        );

        let sql = handlers
            .execute_sql_v1(
                "select * from docs".to_string(),
                None,
                Some("docs".to_string()),
                crate::service_ports::PortIdentity::anonymous(),
            )
            .await
            .unwrap();
        assert_eq!(sql.columns, vec!["id", "score", "flag", "none", "obj"]);
        assert_eq!(sql.rows_scanned, 9);
        assert_eq!(sql.rows_returned, 1);
        let fields = &sql.rows[0].fields;
        assert!(matches!(
            fields.iter().find(|field| field.key == "id").and_then(|field| field.value.as_ref()).and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::StringValue(value)) if value == "r1"
        ));
        assert!(matches!(
            fields.iter().find(|field| field.key == "score").and_then(|field| field.value.as_ref()).and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::NumberValue(value)) if (*value - 1.5).abs() < f64::EPSILON
        ));
        assert!(matches!(
            fields
                .iter()
                .find(|field| field.key == "flag")
                .and_then(|field| field.value.as_ref())
                .and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::BoolValue(true))
        ));
        assert!(matches!(
            fields
                .iter()
                .find(|field| field.key == "none")
                .and_then(|field| field.value.as_ref())
                .and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::NullValue(_))
        ));
        assert!(matches!(
            fields.iter().find(|field| field.key == "obj").and_then(|field| field.value.as_ref()).and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::ObjectValue(value)) if value.fields.contains_key("nested")
        ));
    }

    #[tokio::test]
    // Deliberately exercises the deprecated v1 ExecuteHybridQuery dispatch (TD-143).
    #[allow(deprecated)]
    async fn unified_handlers_report_missing_query_adapter_explicitly_and_sql_arrays_lower() {
        let handlers = make_handlers(
            Arc::new(MockCollectionPort::default()),
            Arc::new(MockVectorOpsPort::default()),
            None,
        );
        assert!(
            handlers
                .execute_hybrid_query(HybridSearchRequest::default())
                .await
                .unwrap_err()
                .to_string()
                .contains("QueryAdapterPort")
        );
        assert!(
            handlers
                .execute_sql_v1(
                    "select 1".to_string(),
                    None,
                    None,
                    crate::service_ports::PortIdentity::anonymous(),
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("QueryAdapterPort")
        );

        struct ArrayQueryAdapter;

        #[async_trait]
        impl QueryAdapterPort for ArrayQueryAdapter {
            async fn vector_search(
                &self,
                _request: VectorSearchRequest,
            ) -> Result<VectorOperationResponse> {
                Ok(VectorOperationResponse::default())
            }

            async fn execute_hybrid(
                &self,
                _request: HybridSearchRequest,
            ) -> Result<HybridSearchResponse> {
                Ok(HybridSearchResponse::default())
            }

            async fn execute_sql(
                &self,
                _query: String,
                _collection: Option<String>,
                _identity: crate::service_ports::PortIdentity<'_>,
            ) -> Result<SqlExecutionResult> {
                Ok(SqlExecutionResult {
                    columns: vec!["value".to_string()],
                    column_types: vec!["JSON".to_string()],
                    rows: [
                        json!("text"),
                        json!(7),
                        json!(false),
                        json!(null),
                        json!({"shape": "object"}),
                    ]
                    .iter()
                    .map(|value| vec![proximadb_records::conversions::json_to_proxima(value)])
                    .collect(),
                    rows_scanned: 5,
                    ..Default::default()
                })
            }
        }

        let handlers = make_handlers(
            Arc::new(MockCollectionPort::default()),
            Arc::new(MockVectorOpsPort::default()),
            Some(Arc::new(ArrayQueryAdapter)),
        );
        let sql = handlers
            .execute_sql_v1(
                "select values".to_string(),
                None,
                None,
                crate::service_ports::PortIdentity::anonymous(),
            )
            .await
            .unwrap();
        assert_eq!(sql.rows_returned, 5);
        assert!(sql.rows.iter().all(|row| row.fields[0].key == "value"));
        assert!(matches!(
            sql.rows[4].fields[0]
                .value
                .as_ref()
                .and_then(|value| value.value.as_ref()),
            Some(sql_value::Value::ObjectValue(value)) if value.fields.contains_key("shape")
        ));
    }
    // ── TD-CATREPLACE-1: schema-update ordering ─────────────────────────

    fn schema_update(columns: Vec<CollectionSchemaColumn>) -> CollectionSchemaUpdate {
        CollectionSchemaUpdate {
            schema_id: "s1".to_string(),
            schema_version: "1".to_string(),
            enforcement: CollectionSchemaEnforcement::Hybrid,
            auto_evolve: false,
            columns,
        }
    }

    fn handlers_with(port: Arc<MockCollectionPort>) -> UnifiedHandlers {
        UnifiedHandlers::new(port, Arc::new(MockVectorOpsPort::default()), None)
    }

    /// A schema update the canonical write rejects must persist NOTHING: the
    /// narrow config write happens only after the catalog accepted the schema.
    #[tokio::test]
    async fn rejected_schema_update_persists_no_narrow_config() {
        let port = Arc::new(MockCollectionPort {
            fail_canonical: true,
            ..Default::default()
        });
        let handlers = handlers_with(port.clone());
        let err = handlers
            .update_collection_schema_metadata(
                "c1",
                schema_update(vec![CollectionSchemaColumn {
                    name: "x".to_string(),
                    data_type: ProximaType::Boolean,
                    nullable: true,
                    indexed: false,
                    filterable: true,
                    text_storage: None,
                    max_length: None,
                }]),
                None,
            )
            .await
            .expect_err("the canonical write fails");
        assert!(err.to_string().contains("rejected"));
        assert!(
            !port
                .calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c.starts_with("update:")),
            "the narrow config must not be written before the canonical schema is accepted: {:?}",
            port.calls.lock().unwrap()
        );
    }

    /// On success the canonical write happens first, then the narrow config.
    #[tokio::test]
    async fn schema_update_writes_canonical_before_narrow_config() {
        let port = Arc::new(MockCollectionPort::default());
        let handlers = handlers_with(port.clone());
        handlers
            .update_collection_schema_metadata(
                "c1",
                schema_update(vec![CollectionSchemaColumn {
                    name: "x".to_string(),
                    data_type: ProximaType::Int64,
                    nullable: true,
                    indexed: false,
                    filterable: true,
                    text_storage: None,
                    max_length: None,
                }]),
                None,
            )
            .await
            .expect("update succeeds");
        let calls = port.calls.lock().unwrap().clone();
        let set_at = calls
            .iter()
            .position(|c| c.starts_with("set_columns:"))
            .expect("canonical write");
        let update_at = calls
            .iter()
            .position(|c| c.starts_with("update:"))
            .expect("narrow write");
        assert!(set_at < update_at, "canonical before narrow: {calls:?}");
    }
}
