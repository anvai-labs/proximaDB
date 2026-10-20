//! Catalog-Aware Bulk Write Service
//!
//! Integrates Arrow Flight bulk writes with the catalog system for:
//! - Schema validation before writes
//! - Automatic table/collection creation
//! - Index metadata tracking
//! - Transactional guarantees for Spark/Arrow jobs

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
use parking_lot::RwLock;
use proximadb_catalog::{
    CatalogColumn, CatalogIndex, CatalogIndexType, CatalogTableSchema, CatalogTableStatistics,
};
use proximadb_data_model::ProximaType;
use tracing::{debug, info, warn};

use crate::catalog::{CatalogManager, TableIdentifier};
use crate::network::arrow_ipc::codec::ArrowProtoCodec;
use crate::services::operations::BulkWriteRouter;
use proximadb_records::ProximaRecord;

/// Configuration for catalog-aware bulk writes
#[derive(Debug, Clone)]
pub struct CatalogBulkWriteConfig {
    /// Automatically create tables if they don't exist
    pub auto_create_tables: bool,
    /// Create vector indexes automatically
    pub auto_create_indexes: bool,
    /// Default vector index type for auto-created indexes
    pub default_index_type: CatalogIndexType,
    /// Validate schema before writes
    pub validate_schema: bool,
    /// Update statistics after bulk writes
    pub update_statistics: bool,
    /// Transaction isolation level
    pub isolation_level: IsolationLevel,
    /// Maximum batch size for atomic commits
    pub max_batch_size: usize,
}

impl Default for CatalogBulkWriteConfig {
    fn default() -> Self {
        Self {
            auto_create_tables: true,
            auto_create_indexes: true,
            default_index_type: CatalogIndexType::Hnsw,
            validate_schema: true,
            update_statistics: true,
            isolation_level: IsolationLevel::ReadCommitted,
            max_batch_size: 100_000,
        }
    }
}

/// Transaction isolation levels for bulk writes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted (fastest, no guarantees)
    ReadUncommitted,
    /// Read committed (see only committed data)
    ReadCommitted,
    /// Repeatable read (consistent snapshot)
    RepeatableRead,
    /// Serializable (full isolation)
    Serializable,
}

/// Result of a catalog-aware bulk write operation
#[derive(Debug, Clone)]
pub struct CatalogBulkWriteResult {
    /// Number of records written
    pub records_written: u64,
    /// Table was auto-created
    pub table_created: bool,
    /// Index was auto-created
    pub index_created: bool,
    /// Schema evolution applied
    pub schema_evolved: bool,
    /// Statistics updated
    pub statistics_updated: bool,
    /// Write latency in microseconds
    pub write_latency_us: u64,
    /// Catalog operation latency in microseconds
    pub catalog_latency_us: u64,
    /// Any warnings during the operation
    pub warnings: Vec<String>,
}

/// Write mode for bulk operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkWriteMode {
    /// Append to existing data
    Append,
    /// Replace all existing data (truncate + insert)
    Overwrite,
    /// Update existing records by ID, insert new ones
    Upsert,
    /// Insert only if records don't exist
    InsertIfNotExists,
}

/// Represents an active bulk write transaction
#[derive(Debug)]
pub struct BulkWriteTransaction {
    /// Transaction ID
    pub txn_id: String,
    /// Target table
    pub table: TableIdentifier,
    /// Records buffered in this transaction
    pub buffered_records: u64,
    /// Started at timestamp
    pub started_at: std::time::Instant,
    /// Write mode
    pub write_mode: BulkWriteMode,
    /// Committed flag
    pub committed: bool,
}

/// Catalog-aware bulk write service
///
/// This service wraps Arrow bulk write operations with catalog metadata management.
/// It provides:
/// - Automatic table/collection creation
/// - Schema validation and evolution
/// - Vector index creation
/// - Statistics tracking
pub struct CatalogBulkWriteService {
    /// Catalog manager for metadata operations
    catalog_manager: Arc<CatalogManager>,
    /// Bulk write router for optimization decisions
    router: BulkWriteRouter,
    /// Configuration
    config: CatalogBulkWriteConfig,
    /// Active transactions
    active_transactions: RwLock<HashMap<String, BulkWriteTransaction>>,
}

impl CatalogBulkWriteService {
    /// Create a new catalog bulk write service
    pub fn new(catalog_manager: Arc<CatalogManager>, config: CatalogBulkWriteConfig) -> Self {
        Self {
            catalog_manager,
            router: BulkWriteRouter::new(),
            config,
            active_transactions: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(catalog_manager: Arc<CatalogManager>) -> Self {
        Self::new(catalog_manager, CatalogBulkWriteConfig::default())
    }

    /// Convert Arrow RecordBatches to canonical records with catalog validation.
    ///
    /// This is the main entry point for Spark/Arrow bulk writes.
    /// It handles:
    /// 1. Table resolution in the catalog
    /// 2. Auto-creation of tables if needed
    /// 3. Schema validation
    /// 4. Batch conversion to `ProximaRecord`
    pub async fn prepare_bulk_write(
        &self,
        table_fqn: &str,
        tenant: &str,
        batches: &[RecordBatch],
        _write_mode: BulkWriteMode,
    ) -> Result<(Vec<ProximaRecord>, CatalogBulkWriteResult)> {
        let start = std::time::Instant::now();
        let mut result = CatalogBulkWriteResult {
            records_written: 0,
            table_created: false,
            index_created: false,
            schema_evolved: false,
            statistics_updated: false,
            write_latency_us: 0,
            catalog_latency_us: 0,
            warnings: Vec::new(),
        };

        if batches.is_empty() {
            return Ok((Vec::new(), result));
        }

        // Parse table identifier, within the writing tenant's namespace.
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_fqn, Some(tenant))
            .await?;
        let catalog_start = std::time::Instant::now();

        // Get or create table schema
        let table_schema = match catalog.get_table(&table_id).await {
            Ok(schema) => schema,
            Err(_) if self.config.auto_create_tables => {
                // Infer schema from first batch
                let arrow_schema = batches[0].schema();
                let inferred = self.infer_catalog_schema(&table_id.name, &arrow_schema)?;

                info!(
                    table = %table_fqn,
                    columns = inferred.columns.len(),
                    "Auto-creating table from Arrow schema"
                );

                if !catalog.namespace_exists(&table_id.namespace).await? {
                    catalog
                        .create_namespace_for_tenant(
                            &table_id.namespace,
                            HashMap::new(),
                            Some(tenant),
                        )
                        .await?;
                }
                let created = catalog.create_table(&table_id, inferred).await?;
                result.table_created = true;
                created
            }
            Err(e) => {
                return Err(anyhow!(
                    "Table '{}' does not exist and auto_create_tables is disabled: {}",
                    table_fqn,
                    e
                ));
            }
        };

        // Validate schema compatibility if enabled
        if self.config.validate_schema {
            let arrow_schema = batches[0].schema();
            self.validate_schema_compatibility(&table_schema, &arrow_schema)?;
        }

        result.catalog_latency_us = catalog_start.elapsed().as_micros() as u64;

        // Convert batches to canonical ProximaRecord envelopes.
        let write_start = std::time::Instant::now();
        let batches_vec: Vec<RecordBatch> = batches.to_vec();
        let records = ArrowProtoCodec::batches_to_proxima_records(batches_vec)?;
        result.records_written = records.len() as u64;
        result.write_latency_us = write_start.elapsed().as_micros() as u64;

        // Auto-create vector index if needed
        if self.config.auto_create_indexes
            && result.table_created
            && let Some(vector_col) = self.find_vector_column(&table_schema)
        {
            let index = CatalogIndex::new(
                format!("{}_vector_idx", table_id.name),
                vec![vector_col.clone()],
                self.config.default_index_type,
            );

            match catalog.create_index(&table_id, index).await {
                Ok(_) => {
                    result.index_created = true;
                    info!(
                        table = %table_fqn,
                        column = %vector_col,
                        "Auto-created vector index"
                    );
                }
                Err(e) => {
                    result
                        .warnings
                        .push(format!("Failed to auto-create index: {}", e));
                }
            }
        }

        // Update statistics
        if self.config.update_statistics {
            let stats = self.compute_statistics(&table_id, result.records_written);
            if let Err(e) = catalog.update_statistics(&table_id, stats).await {
                result
                    .warnings
                    .push(format!("Failed to update statistics: {}", e));
            } else {
                result.statistics_updated = true;
                // Bump corpus_version: refreshed stats change the
                // planner's selectivity estimates, so any cached
                // PlanOutput is now computed against stale numbers.
                // Same tenant-extraction convention as the DDL paths
                // (namespace[0] when present).
                if let Some(tenant_id) = table_id.namespace.first() {
                    let version = crate::catalog::CorpusVersionRegistry::global()
                        .bump(tenant_id, &table_id.name)
                        .await;
                    tracing::debug!(
                        table = %table_id.name,
                        tenant = %tenant_id,
                        version,
                        "🔄 corpus_version bumped after stats refresh (bulk write)"
                    );
                }
            }
        }

        info!(
            table = %table_fqn,
            records = result.records_written,
            table_created = result.table_created,
            index_created = result.index_created,
            total_latency_us = start.elapsed().as_micros(),
            "Bulk write prepared"
        );

        Ok((records, result))
    }

    /// Check if the given batch size would route to the bulk lane.
    pub fn would_use_bulk_lane(&self, records: &[ProximaRecord]) -> bool {
        self.router.route_records(records).use_bulk_lane
    }

    /// Get write routing decision for a batch
    pub fn get_write_decision(&self, records: &[ProximaRecord]) -> super::BulkWriteDecision {
        self.router.route_records(records)
    }

    /// Begin a transactional bulk write session
    pub async fn begin_transaction(
        &self,
        table_fqn: &str,
        tenant: &str,
        write_mode: BulkWriteMode,
    ) -> Result<String> {
        let txn_id = uuid::Uuid::new_v4().to_string();
        let (_, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_fqn, Some(tenant))
            .await?;

        let txn = BulkWriteTransaction {
            txn_id: txn_id.clone(),
            table: table_id,
            buffered_records: 0,
            started_at: std::time::Instant::now(),
            write_mode,
            committed: false,
        };

        self.active_transactions.write().insert(txn_id.clone(), txn);

        debug!(txn_id = %txn_id, table = %table_fqn, "Bulk write transaction started");
        Ok(txn_id)
    }

    /// Add records to an active transaction
    pub fn add_to_transaction(&self, txn_id: &str, record_count: u64) -> Result<()> {
        let mut txns = self.active_transactions.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", txn_id))?;

        if txn.committed {
            return Err(anyhow!("Transaction already committed: {}", txn_id));
        }

        txn.buffered_records += record_count;
        Ok(())
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, txn_id: &str) -> Result<CatalogBulkWriteResult> {
        let mut txns = self.active_transactions.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", txn_id))?;

        if txn.committed {
            return Err(anyhow!("Transaction already committed: {}", txn_id));
        }

        txn.committed = true;
        let records = txn.buffered_records;
        let latency = txn.started_at.elapsed().as_micros() as u64;

        debug!(txn_id = %txn_id, records = records, "Transaction committed");

        Ok(CatalogBulkWriteResult {
            records_written: records,
            table_created: false,
            index_created: false,
            schema_evolved: false,
            statistics_updated: false,
            write_latency_us: latency,
            catalog_latency_us: 0,
            warnings: Vec::new(),
        })
    }

    /// Rollback a transaction
    pub fn rollback_transaction(&self, txn_id: &str) -> Result<()> {
        let txn = self
            .active_transactions
            .write()
            .remove(txn_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", txn_id))?;

        if txn.committed {
            return Err(anyhow!("Cannot rollback committed transaction: {}", txn_id));
        }

        warn!(txn_id = %txn_id, records = txn.buffered_records, "Transaction rolled back");
        Ok(())
    }

    // ============ Private Helper Methods ============

    /// Infer catalog schema from Arrow schema
    fn infer_catalog_schema(
        &self,
        table_name: &str,
        arrow_schema: &Arc<Schema>,
    ) -> Result<CatalogTableSchema> {
        let mut schema = CatalogTableSchema::new(table_name);

        for (idx, field) in arrow_schema.fields().iter().enumerate() {
            let (data_type, properties) = self.arrow_to_catalog_type(field.data_type())?;

            let column = CatalogColumn {
                id: idx as i32,
                object_id: None,
                name: field.name().clone(),
                data_type,
                nullable: field.is_nullable(),
                default_value: None,
                comment: field.metadata().get("comment").cloned(),
                properties,
                is_deleted: false,
                original_id: None,
            };

            schema = schema.with_column(column);
        }

        Ok(schema.with_primary_key(vec!["id".to_string()]))
    }

    /// Convert Arrow DataType to the canonical [`ProximaType`].
    fn arrow_to_catalog_type(
        &self,
        arrow_type: &DataType,
    ) -> Result<(ProximaType, HashMap<String, String>)> {
        use proximadb_data_model::{TimeUnit, VectorElement};
        let mut properties = HashMap::new();

        let dense_vector = || ProximaType::DenseVector {
            element: VectorElement::Float32,
            dim: 0,
        };

        let catalog_type = match arrow_type {
            DataType::Boolean => ProximaType::Boolean,
            DataType::Int8 => ProximaType::Int8,
            DataType::Int16 => ProximaType::Int16,
            DataType::Int32 => ProximaType::Int32,
            DataType::Int64 => ProximaType::Int64,
            DataType::Float32 => ProximaType::Float32,
            DataType::Float64 => ProximaType::Float64,
            DataType::Utf8 | DataType::LargeUtf8 => ProximaType::String,
            DataType::Binary | DataType::LargeBinary => ProximaType::Binary,
            DataType::Date32 | DataType::Date64 => ProximaType::Date,
            DataType::Time32(_) | DataType::Time64(_) => ProximaType::Time(TimeUnit::Nanosecond),
            DataType::Timestamp(_, _) => ProximaType::Timestamp(TimeUnit::Nanosecond),
            DataType::FixedSizeList(inner, size) => {
                if matches!(inner.data_type(), DataType::Float32 | DataType::Float64) {
                    properties.insert("dimension".to_string(), size.to_string());
                    dense_vector()
                } else {
                    ProximaType::Json // Fallback for other lists
                }
            }
            DataType::List(inner) => {
                if matches!(inner.data_type(), DataType::Float32 | DataType::Float64) {
                    dense_vector()
                } else {
                    ProximaType::Json
                }
            }
            DataType::Struct(_) => ProximaType::Json,
            DataType::Map(_, _) => ProximaType::Json,
            _ => ProximaType::Binary, // Fallback
        };

        Ok((catalog_type, properties))
    }

    /// Validate Arrow schema against catalog schema
    fn validate_schema_compatibility(
        &self,
        catalog_schema: &CatalogTableSchema,
        arrow_schema: &Arc<Schema>,
    ) -> Result<()> {
        for field in arrow_schema.fields() {
            let catalog_col = catalog_schema
                .columns
                .iter()
                .find(|c| c.name == *field.name());

            if let Some(col) = catalog_col {
                // Check type compatibility
                let (arrow_type, _) = self.arrow_to_catalog_type(field.data_type())?;
                if !self.types_compatible(&col.data_type, &arrow_type) {
                    return Err(anyhow!(
                        "Type mismatch for column '{}': catalog has {:?}, Arrow has {:?}",
                        field.name(),
                        col.data_type,
                        arrow_type
                    ));
                }
            }
            // New columns are allowed (schema evolution)
        }
        Ok(())
    }

    /// Check if two catalog types are compatible
    fn types_compatible(&self, catalog_type: &ProximaType, arrow_type: &ProximaType) -> bool {
        if catalog_type == arrow_type {
            return true;
        }

        // Allow numeric promotions
        matches!(
            (catalog_type, arrow_type),
            (
                ProximaType::Int8,
                ProximaType::Int16 | ProximaType::Int32 | ProximaType::Int64
            ) | (ProximaType::Int16, ProximaType::Int32 | ProximaType::Int64)
                | (ProximaType::Int32, ProximaType::Int64)
                | (ProximaType::Float32, ProximaType::Float64)
        )
    }

    /// Find vector column in schema
    fn find_vector_column(&self, schema: &CatalogTableSchema) -> Option<String> {
        schema
            .columns
            .iter()
            .find(|c| matches!(c.data_type, ProximaType::DenseVector { .. }))
            .map(|c| c.name.clone())
    }

    /// Compute statistics for a table
    fn compute_statistics(
        &self,
        _table: &TableIdentifier,
        new_records: u64,
    ) -> CatalogTableStatistics {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        CatalogTableStatistics {
            row_count: new_records,
            size_bytes: 0, // Would need to compute from storage
            file_count: 0,
            last_analyzed_ms: Some(now),
            column_stats: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_bulk_write_config_default() {
        let config = CatalogBulkWriteConfig::default();
        assert!(config.auto_create_tables);
        assert!(config.auto_create_indexes);
        assert!(config.validate_schema);
        assert_eq!(config.max_batch_size, 100_000);
    }

    #[test]
    fn test_isolation_level_equality() {
        assert_eq!(IsolationLevel::ReadCommitted, IsolationLevel::ReadCommitted);
        assert_ne!(IsolationLevel::ReadCommitted, IsolationLevel::Serializable);
    }

    #[test]
    fn test_write_mode_equality() {
        assert_eq!(BulkWriteMode::Append, BulkWriteMode::Append);
        assert_ne!(BulkWriteMode::Append, BulkWriteMode::Overwrite);
    }
}
