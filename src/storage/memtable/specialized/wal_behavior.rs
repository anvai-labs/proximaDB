//! WAL Behavior Wrapper with Batch Coordination
//!
//! Extends GlobalPartitionedMemtable with Write Buffer-specific behaviors using composition:
//! - Batch coordination for WAL consistency  
//! - Sequential write optimizations for compression
//! - MVCC support for recovery consistency
//! - Ordered flush operations for RLE/dictionary encodings
//! - Specialized serialization strategies

use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::RwLock;

use crate::core::bloom::strategies::composite::CompositeBloomFilter;
use crate::core::bloom::{BloomFilterConfig, BloomFilterStrategy};
use crate::storage::memtable::core::MemtableConfig;
use crate::storage::memtable::implementations::global_partitioned::GlobalPartitionedMemtable;
use crate::storage::persistence::write_ahead_log::{BatchId, WALOperation, WALStats};
use proximadb_records::ProximaRecord;

/// Canonical key format for a metadata `field=value` entry in a batch's
/// `metadata_bloom_filter`.
///
/// This is the **single source of truth** shared by the bloom *build* site
/// (`WALVectorBatch::create_bloom_filter`) and every *query* site (the WAL
/// search path's `filter_batches_with_bloom`). Keeping one helper prevents the
/// build/query separators from ever drifting apart — a past `=` vs `:` mismatch
/// silently turned the bloom pre-filter into a false-negative generator that
/// excluded *every* batch, so a metadata-filtered search returned zero results.
pub(crate) fn metadata_bloom_key(field: &str, value: &str) -> String {
    format!("{}={}", field, value)
}

fn memtable_threshold_backpressure(
    collection_id: &str,
    current_usage: u64,
    threshold: u64,
) -> Option<crate::storage::persistence::write_ahead_log::flush_policy::WalBackpressure> {
    if threshold == 0 || current_usage < threshold {
        return None;
    }
    Some(
        crate::storage::persistence::write_ahead_log::flush_policy::WalBackpressure {
            collection_id: collection_id.to_string(),
            fill_pct: (current_usage as f64 / threshold as f64) * 100.0,
        },
    )
}

/// Write Buffer-specific vector batch for tracking deserialized data
#[derive(Debug, Clone)]
pub struct WALVectorBatch {
    /// Batch coordination ID
    pub batch_id: BatchId,
    /// Deserialized vector records (ready for search/flush)
    /// Using Arc for zero-copy sharing across WAL strategies
    pub vector_records: Arc<Vec<ProximaRecord>>,
    /// Batch metadata
    pub timestamp: std::time::SystemTime,
    pub total_size_bytes: usize,
    pub is_flushed: bool,
    /// Bloom filter for fast metadata filtering (skip 95%+ irrelevant batches)
    pub metadata_bloom_filter: Option<CompositeBloomFilter>,
}

impl WALVectorBatch {
    fn flush_shape(&self) -> WALFlushBatchShape {
        let materializable_records = self
            .vector_records
            .iter()
            .filter(|record| {
                record
                    .embeddings
                    .first()
                    .is_some_and(|embedding| !embedding.values.is_empty())
            })
            .count();
        match (materializable_records, self.vector_records.len()) {
            (0, _) => WALFlushBatchShape::DeferredOnly,
            (materializable, total) if materializable == total => WALFlushBatchShape::VectorsOnly,
            _ => WALFlushBatchShape::Mixed,
        }
    }

    /// Estimate the resident size of a record slice — the CANONICAL
    /// `total_size_bytes` source for WAL batches (ADR-069/TD-WAL-1).
    ///
    /// Embedding payload bytes are exact and precision-aware
    /// (`EmbeddingCell::values_byte_size`: 4B/elem fp32, 2B fp16/bf16, …);
    /// props/identity use the same overhead model as
    /// `BulkWriteRouter::estimate_record_batch_size` (96B/prop + 256B/record +
    /// oid + 64B). An estimate is sufficient: the consumers are watermark
    /// fractions, backpressure sums, and gauges — not exact accounting.
    ///
    /// Two consumers depend on a non-zero value: the wal_behavior
    /// per-collection backpressure check and the auto-flush driver's
    /// size/capacity triggers — a batch constructed with `total_size_bytes: 0`
    /// silently disables BOTH. This is the **single owner** of the batch byte
    /// math (TD-WAL-2 V1): `BulkWriteRouter::estimate_record_batch_size`
    /// delegates here rather than keeping its own (previously fp32-only) copy.
    pub fn estimate_records_size(records: &[ProximaRecord]) -> usize {
        const PROPERTY_OVERHEAD_PER_RECORD: usize = 256;
        const ID_OVERHEAD_PER_RECORD: usize = 64;
        const PROP_BYTES: usize = 96;
        records
            .iter()
            .map(|record| {
                let embedding_bytes: usize =
                    record.embeddings.iter().map(|e| e.values_byte_size()).sum();
                let props_bytes = record.props.len() * PROP_BYTES + PROPERTY_OVERHEAD_PER_RECORD;
                let id_bytes = record.oid.len() + ID_OVERHEAD_PER_RECORD;
                embedding_bytes + props_bytes + id_bytes
            })
            .sum()
    }

    /// [`Self::estimate_records_size`] over this batch's records.
    pub fn estimated_size_bytes(&self) -> usize {
        Self::estimate_records_size(&self.vector_records)
    }

    /// Create bloom filter for batch metadata
    pub fn create_bloom_filter(&mut self) -> Result<()> {
        // Create bloom filter with optimal false positive rate
        let config = BloomFilterConfig {
            strategy: crate::core::bloom::BloomStrategy::ByteAligned,
            bits_per_key: 10, // 10 bits per key for ~1% false positive
            false_positive_rate: Some(0.01),
            expected_items: self.vector_records.len(),
            enabled: true,
            hash_algorithm: crate::core::bloom::HashAlgorithm::Murmur3,
        };

        let mut bloom_filter = CompositeBloomFilter::new(self.vector_records.len(), &config);

        // Add all props keys to bloom filter
        for record in self.vector_records.iter() {
            for (key, node) in &record.props {
                bloom_filter.insert(key.as_bytes());

                // Also add key=value pairs for exact matching
                let value_str = match node {
                    proximadb_records::ProximaTreeNode::Value(v) => {
                        use proximadb_data_model::ProximaValue;
                        match v {
                            ProximaValue::String(s) => s.clone(),
                            ProximaValue::Float64(n) => n.to_string(),
                            ProximaValue::Boolean(b) => b.to_string(),
                            ProximaValue::Int64(i) => i.to_string(),
                            _ => continue,
                        }
                    }
                    _ => continue,
                };

                let key_value = metadata_bloom_key(key, &value_str);
                bloom_filter.insert(key_value.as_bytes());
            }
        }

        self.metadata_bloom_filter = Some(bloom_filter);
        Ok(())
    }

    /// Check if batch might contain metadata key
    pub fn might_contain_metadata(&self, key: &str) -> bool {
        match &self.metadata_bloom_filter {
            Some(bloom) => bloom.might_contain(key.as_bytes()), // Use might_contain with bytes
            None => true, // No bloom filter means we can't skip
        }
    }

    /// Check if batch might contain metadata key-value pair
    pub fn might_contain_metadata_value(&self, key: &str, value: &str) -> bool {
        match &self.metadata_bloom_filter {
            Some(bloom) => {
                let key_value = metadata_bloom_key(key, value);
                bloom.might_contain(key_value.as_bytes())
            }
            None => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WALFlushBatchShape {
    VectorsOnly,
    DeferredOnly,
    Mixed,
}

#[derive(Clone, Copy)]
enum WALFlushClaimMode {
    All,
    SegmentMaterializable,
}

/// Exact, cancellation-safe ownership of immutable WAL batches for one flush.
///
/// Dropping an unfinished claim releases its batch IDs synchronously, including
/// when the owning future is cancelled. Claimed batches remain visible to WAL
/// reads until [`WALBehaviorWrapper::complete_flush_claim`] retires them after a
/// successful segment publication.
#[derive(Debug)]
pub struct WALFlushBatchClaim {
    collection_id: String,
    claim_id: u64,
    batch_ids: Vec<String>,
    batches: Vec<WALVectorBatch>,
    registry: Arc<StdMutex<FlushClaimRegistry>>,
    finalized: bool,
}

impl WALFlushBatchClaim {
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    pub fn claim_id(&self) -> u64 {
        self.claim_id
    }

    pub fn batch_ids(&self) -> &[String] {
        &self.batch_ids
    }

    pub fn batches(&self) -> &[WALVectorBatch] {
        &self.batches
    }

    fn finalize(&mut self) -> Result<()> {
        let mut registry = self.registry.lock().map_err(|_| {
            anyhow::anyhow!(
                "flush claim registry poisoned while finalizing collection '{}'",
                self.collection_id
            )
        })?;
        registry.release(&self.collection_id, self.claim_id, &self.batch_ids);
        self.finalized = true;
        Ok(())
    }
}

impl Drop for WALFlushBatchClaim {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        match self.registry.lock() {
            Ok(mut registry) => {
                registry.release(&self.collection_id, self.claim_id, &self.batch_ids);
            }
            Err(_) => {
                tracing::error!(
                    collection_id = %self.collection_id,
                    claim_id = self.claim_id,
                    "flush claim registry poisoned; cancellation rollback could not run"
                );
            }
        }
    }
}

#[derive(Debug, Default)]
struct FlushClaimRegistry {
    /// collection -> batch id -> claim id
    claimed: HashMap<String, HashMap<String, u64>>,
}

impl FlushClaimRegistry {
    fn is_claimed(&self, collection_id: &str, batch_id: &str) -> bool {
        self.claimed
            .get(collection_id)
            .is_some_and(|claims| claims.contains_key(batch_id))
    }

    fn insert(&mut self, collection_id: &str, batch_id: String, claim_id: u64) {
        self.claimed
            .entry(collection_id.to_string())
            .or_default()
            .insert(batch_id, claim_id);
    }

    fn owns_all(&self, collection_id: &str, claim_id: u64, batch_ids: &[String]) -> bool {
        self.claimed.get(collection_id).is_some_and(|claims| {
            batch_ids
                .iter()
                .all(|batch_id| claims.get(batch_id) == Some(&claim_id))
        })
    }

    fn release(&mut self, collection_id: &str, claim_id: u64, batch_ids: &[String]) {
        let mut remove_collection = false;
        if let Some(claims) = self.claimed.get_mut(collection_id) {
            for batch_id in batch_ids {
                if claims.get(batch_id) == Some(&claim_id) {
                    claims.remove(batch_id);
                }
            }
            remove_collection = claims.is_empty();
        }
        if remove_collection {
            self.claimed.remove(collection_id);
        }
    }
}

/// Write Buffer-specific batch coordinator
#[derive(Debug)]
struct BatchCoordinator {
    /// Active batches by collection (collection_id -> batch_id -> WALVectorBatch)
    batches: HashMap<String, HashMap<String, WALVectorBatch>>,
    /// Individual vector index for fast search (vector_id -> (collection_id, batch_id, index_in_batch))
    vector_index: HashMap<String, (String, String, usize)>,
    /// TD-FLUSH-3: per-batch PREDICTED segment bytes (Σ dim×9/8+8 over the
    /// batch's records — the RaBitQ+SQ8 bytes the flushed segment will occupy,
    /// as opposed to `total_size_bytes` which counts serialized-record bytes
    /// ~1.6 KB/record). Keyed (collection_id -> batch_id); maintained by
    /// `add_batch` and pruned wherever batches are removed, so
    /// `unflushed_predicted_bytes` reads the O(1) counter below; this map is
    /// the SUBTRACTION source when individual batches flush/remove.
    predicted_per_batch: HashMap<String, HashMap<String, u64>>,
    /// TD-FLUSH-3: running per-collection predicted-bytes counter maintained
    /// at WRITE time (co-design: the memtable touches every byte on entry, so
    /// the flush statistic is kept incrementally — O(1) at decision time
    /// instead of an O(#batches) sum that grows to ~900 batches per 128 MB
    /// segment while the floor accumulates). Direction (TD-FLUSH-3 S4): grow
    /// into the per-class byte spine (serialized/vector/row/f32) feeding
    /// flush policy, WAL gauges, KSU raw-GB prediction, and cache
    /// auto-sizing from ONE write-boundary accounting.
    predicted_unflushed: HashMap<String, u64>,
}

/// TD-FLUSH-3 predictor: RaBitQ (`d/8 + 8`) + SQ8 (`d`) bytes for one record's
/// first embedding — ≈ `d × 9/8 + 8`. The optional f32 tier is deliberately
/// ignored (under-predicting only DELAYS the flush → larger segments — the safe
/// direction). Verified against measured segments (254 B/vec at 128d incl. row
/// data vs 152 predicted).
fn predicted_record_bytes(record: &proximadb_records::ProximaRecord) -> u64 {
    record
        .embeddings
        .first()
        .map(|e| {
            let d = e.dim as u64;
            d + d / 8 + 8
        })
        .unwrap_or(0)
}

impl BatchCoordinator {
    fn new() -> Self {
        Self {
            batches: HashMap::new(),
            vector_index: HashMap::new(),
            predicted_per_batch: HashMap::new(),
            predicted_unflushed: HashMap::new(),
        }
    }

    /// Add a vector batch to the coordinator
    fn add_batch(&mut self, collection_id: &str, batch: WALVectorBatch) -> Result<()> {
        let batch_id = batch.batch_id.to_base62();

        // Update vector index
        for (index, vector_record) in batch.vector_records.iter().enumerate() {
            self.vector_index.insert(
                vector_record.oid.clone(),
                (collection_id.to_string(), batch_id.clone(), index),
            );
        }

        // TD-FLUSH-3: predicted segment bytes for this batch (one pass over the
        // batch at add time — batches arrive per client request, not per record).
        let predicted: u64 = batch
            .vector_records
            .iter()
            .map(predicted_record_bytes)
            .sum();
        self.predicted_per_batch
            .entry(collection_id.to_string())
            .or_default()
            .insert(batch_id.clone(), predicted);
        *self
            .predicted_unflushed
            .entry(collection_id.to_string())
            .or_default() += predicted;

        // Store batch
        self.batches
            .entry(collection_id.to_string())
            .or_default()
            .insert(batch_id, batch);

        Ok(())
    }

    /// Get all unflushed batches for a collection
    fn get_unflushed_batches(&self, collection_id: &str) -> Vec<&WALVectorBatch> {
        if let Some(collection_batches) = self.batches.get(collection_id) {
            collection_batches
                .values()
                .filter(|batch| !batch.is_flushed)
                .collect()
        } else {
            Vec::new()
        }
    }

    fn unflushed_batch_count(&self, collection_id: &str) -> usize {
        self.batches
            .get(collection_id)
            .map(|batches| batches.values().filter(|batch| !batch.is_flushed).count())
            .unwrap_or(0)
    }

    fn validate_batches_exact(&self, collection_id: &str, batch_ids: &[String]) -> Result<()> {
        let Some(collection_batches) = self.batches.get(collection_id) else {
            anyhow::bail!("collection '{}' has no WAL batches", collection_id);
        };
        if let Some(missing) = batch_ids
            .iter()
            .find(|batch_id| !collection_batches.contains_key(*batch_id))
        {
            anyhow::bail!(
                "claimed batch '{}:{}' disappeared before flush retirement",
                collection_id,
                missing
            );
        }
        Ok(())
    }

    /// Remove an exact, pre-validated set of batches as one coordinator update.
    fn remove_batches_exact(&mut self, collection_id: &str, batch_ids: &[String]) -> Result<usize> {
        self.validate_batches_exact(collection_id, batch_ids)?;

        let mut removed_records = 0usize;
        for batch_id in batch_ids {
            let removed = self
                .batches
                .get_mut(collection_id)
                .and_then(|batches| batches.remove(batch_id))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "claimed batch '{}:{}' disappeared during flush retirement",
                        collection_id,
                        batch_id
                    )
                })?;
            removed_records += removed.vector_records.len();
            for record in removed.vector_records.iter() {
                let owned_by_removed_batch = self
                    .vector_index
                    .get(&record.oid)
                    .is_some_and(|(_, indexed_batch, _)| indexed_batch == batch_id);
                if owned_by_removed_batch {
                    self.vector_index.remove(&record.oid);
                }
            }
            self.credit_predicted(collection_id, batch_id);
        }
        Ok(removed_records)
    }

    /// Sum the unflushed bytes for a collection WITHOUT cloning the batches. The
    /// inline flush trigger checks this on the hot write path (per write), where
    /// `get_unflushed_batches`'s per-batch clone was a throughput regression.
    fn unflushed_bytes(&self, collection_id: &str) -> u64 {
        self.batches
            .get(collection_id)
            .map(|collection_batches| {
                collection_batches
                    .values()
                    .filter(|batch| !batch.is_flushed)
                    .map(|batch| batch.total_size_bytes as u64)
                    .sum()
            })
            .unwrap_or(0)
    }

    /// TD-FLUSH-3: predicted segment bytes of the unflushed delta — O(1) read
    /// of the write-time counter.
    fn unflushed_predicted_bytes(&self, collection_id: &str) -> u64 {
        self.predicted_unflushed
            .get(collection_id)
            .copied()
            .unwrap_or(0)
    }

    /// Subtract a flushed/removed batch's predicted bytes from the running
    /// counter (saturating; a missing per-batch entry credits 0).
    fn credit_predicted(&mut self, collection_id: &str, batch_id: &str) {
        let credit = self
            .predicted_per_batch
            .get_mut(collection_id)
            .and_then(|m| m.remove(batch_id))
            .unwrap_or(0);
        if let Some(total) = self.predicted_unflushed.get_mut(collection_id) {
            *total = total.saturating_sub(credit);
        }
    }

    /// Mark batch as flushed
    fn mark_batch_flushed(&mut self, collection_id: &str, batch_id: &str) -> Result<()> {
        // TD-FLUSH-3: the batch leaves the unflushed set — credit its
        // predicted bytes so the floor counter tracks only unflushed data.
        self.credit_predicted(collection_id, batch_id);
        if let Some(collection_batches) = self.batches.get_mut(collection_id)
            && let Some(batch) = collection_batches.get_mut(batch_id)
        {
            batch.is_flushed = true;
            tracing::debug!("✅ Marked batch {} as flushed", batch_id);
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "Batch {}:{} not found",
            collection_id,
            batch_id
        ))
    }

    /// List all collections that have unflushed data (for shutdown flush)
    fn list_collections_with_unflushed_data(&self) -> Vec<String> {
        self.batches
            .iter()
            .filter(|(_, batches)| batches.values().any(|batch| !batch.is_flushed))
            .map(|(collection_id, _)| collection_id.clone())
            .collect()
    }

    /// Clear flushed batches from coordinator
    fn clear_flushed_batches(&mut self, collection_id: &str) -> Result<usize> {
        let mut cleared_count = 0;

        if let Some(collection_batches) = self.batches.get_mut(collection_id) {
            // OPTIMIZATION: Use retain instead of collect+remove to avoid extra allocation
            // First collect IDs from flushed batches for index cleanup
            let mut cleared_batch_records = Vec::new();
            let mut cleared_batch_ids = Vec::new();
            for (batch_id, batch) in collection_batches.iter() {
                if batch.is_flushed {
                    cleared_batch_records
                        .extend(batch.vector_records.iter().map(|v| v.oid.clone()));
                    cleared_batch_ids.push(batch_id.clone());
                }
            }

            let original_count = collection_batches.len();
            collection_batches.retain(|_, batch| !batch.is_flushed);

            cleared_count = original_count - collection_batches.len();
            // Remove vector index entries for cleared batches
            for vector_id in cleared_batch_records {
                self.vector_index.remove(&vector_id);
            }
            // TD-FLUSH-3: defensive counter credit — normally a no-op because
            // mark_batch_flushed already credited (credit removes the map
            // entry, so double-crediting is impossible).
            for batch_id in cleared_batch_ids {
                self.credit_predicted(collection_id, &batch_id);
            }
        }

        tracing::debug!(
            "🧹 Cleared {} flushed batches from collection {}",
            cleared_count,
            collection_id
        );
        Ok(cleared_count)
    }
}

/// Write Buffer-specific behavior wrapper around global partitioned memtable implementation
///
/// This uses a global partitioned memtable that supports:
/// - Global sequence ordering for flush coordination
/// - Per-collection data partitions for efficient operations
/// - Content-based search within collections on unflushed data
#[derive(Debug, Clone)]
pub struct WALBehaviorWrapper {
    /// The wrapped global partitioned memtable implementation (generic storage) - Arc for memory efficiency
    inner: Arc<GlobalPartitionedMemtable>,

    /// Write Buffer-specific batch coordinator (handles deserialized batches)
    batch_coordinator: Arc<RwLock<BatchCoordinator>>,

    /// Write Buffer-specific configuration
    config: MemtableConfig,

    /// Sequence number generator for WAL entries (Arc for Clone)
    sequence_generator: Arc<AtomicU64>,

    /// MVCC tracking: vector_id -> [sequences] - Not needed for WAL, but kept for compatibility
    mvcc_versions: Arc<RwLock<std::collections::HashMap<String, Vec<u64>>>>,

    /// Write Buffer-specific metrics
    wal_metrics: Arc<RwLock<WriteBufferMetrics>>,

    /// Flush coordination state
    #[allow(dead_code)]
    flush_state: Arc<RwLock<FlushState>>,

    /// ADR-081: exact source ownership for in-flight materializations. This is
    /// a synchronous mutex so a cancelled future's claim can roll back in Drop.
    flush_claims: Arc<StdMutex<FlushClaimRegistry>>,

    /// Monotonic process-local claim epoch.
    flush_claim_sequence: Arc<AtomicU64>,

    /// Distributed mode: idempotency tokens per collection
    idempotency_tokens:
        Arc<RwLock<std::collections::HashMap<String, std::collections::HashSet<String>>>>,
    /// Distributed mode: per-collection partition sequencers (partition_key -> seq)
    partition_sequences:
        Arc<RwLock<std::collections::HashMap<String, std::collections::HashMap<String, u64>>>>,
}

impl WALBehaviorWrapper {
    /// Create new WAL behavior wrapper with global partitioned memtable
    pub fn new(config: MemtableConfig) -> Self {
        Self {
            inner: Arc::new(GlobalPartitionedMemtable::new()),
            batch_coordinator: Arc::new(RwLock::new(BatchCoordinator::new())),
            config,
            sequence_generator: Arc::new(AtomicU64::new(1)),
            mvcc_versions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            wal_metrics: Arc::new(RwLock::new(WriteBufferMetrics::default())),
            flush_state: Arc::new(RwLock::new(FlushState::default())),
            flush_claims: Arc::new(StdMutex::new(FlushClaimRegistry::default())),
            flush_claim_sequence: Arc::new(AtomicU64::new(1)),
            idempotency_tokens: Arc::new(RwLock::new(std::collections::HashMap::new())),
            partition_sequences: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Get the wrapped implementation
    pub fn inner(&self) -> &GlobalPartitionedMemtable {
        &self.inner
    }

    /// Get next sequence number for WAL ordering
    pub fn next_sequence(&self) -> u64 {
        let old_value = self.sequence_generator.load(Ordering::SeqCst);
        let new_value = self.sequence_generator.fetch_add(1, Ordering::SeqCst);
        tracing::info!(
            "🔍 SEQUENCE_TRACE: next_sequence() - old_value={}, returned_value={}, new_current={}",
            old_value,
            new_value,
            old_value + 1
        );
        new_value
    }

    /// Get current sequence number without incrementing
    pub fn current_sequence(&self) -> u64 {
        self.sequence_generator.load(Ordering::SeqCst)
    }

    /// Add WALOperation with single deserialization (OPTIMAL: single CPU deserialize for all strategies)
    /// This deserializes the payload once and creates WALVectorBatch for storage
    pub async fn add_wal_operation(
        &self,
        collection_id: &str,
        operation: crate::storage::persistence::write_ahead_log::WALOperation,
    ) -> Result<Vec<u64>> {
        tracing::debug!(
            "🔄 WAL_BEHAVIOR: Single deserialization for {} format with {} vectors",
            operation.payload_format,
            operation.vector_count
        );

        // Single point of deserialization - leverage this for ALL strategies.
        let vector_records: Vec<ProximaRecord> = match operation.payload_format.as_str() {
            "avro" => {
                use crate::storage::persistence::write_ahead_log::serialization::{
                    AvroSerializer, VectorBatchSerializer,
                };
                let serializer = AvroSerializer::new();
                serializer.deserialize_batch(&operation.payload_data)?
            }
            "bincode" => bincode::deserialize::<Vec<ProximaRecord>>(&operation.payload_data)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize Bincode payload: {}", e))?,
            format => {
                anyhow::bail!("Unsupported payload format: {}", format);
            }
        };

        // Create WALVectorBatch from deserialized records
        let batch = WALVectorBatch {
            batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
            vector_records: Arc::new(vector_records),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: operation.payload_data.len(),
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        tracing::debug!(
            "✅ WAL_BEHAVIOR: Single deserialization complete, storing {} vectors",
            batch.vector_records.len()
        );

        // Use existing batch storage
        self.add_vector_batch(collection_id, batch).await
    }

    /// Unified batch addition method - STREAMLINED ARCHITECTURE (stores entire batch natively)
    /// This is used when WALVectorBatch is already deserialized
    pub async fn add_vector_batch(
        &self,
        collection_id: &str,
        batch: WALVectorBatch,
    ) -> Result<Vec<u64>> {
        self.add_vector_batch_internal(collection_id, batch, false)
            .await
    }

    pub async fn add_vector_batch_insert_only(
        &self,
        collection_id: &str,
        batch: WALVectorBatch,
    ) -> Result<Vec<u64>> {
        self.add_vector_batch_internal(collection_id, batch, true)
            .await
    }

    async fn add_vector_batch_internal(
        &self,
        collection_id: &str,
        mut batch: WALVectorBatch,
        insert_only: bool,
    ) -> Result<Vec<u64>> {
        // Backpressure: per-collection threshold check
        let current_usage = {
            let coord = self.batch_coordinator.read().await;
            if let Some(col) = coord.batches.get(collection_id) {
                col.values()
                    .filter(|b| !b.is_flushed)
                    .map(|b| b.total_size_bytes as u64)
                    .sum()
            } else {
                0
            }
        };
        let threshold = self.config.flush_threshold_bytes as u64;
        if let Some(backpressure) =
            memtable_threshold_backpressure(collection_id, current_usage, threshold)
        {
            return Err(anyhow::Error::new(backpressure));
        }
        let batch_id = batch.batch_id.to_base62();
        let vector_count = batch.vector_records.len();

        tracing::debug!(
            "🚀 WAL_BEHAVIOR: Starting add_vector_batch for batch {} to collection {} ({} vectors)",
            batch_id,
            collection_id,
            vector_count
        );

        // Create bloom filter if not already present
        if batch.metadata_bloom_filter.is_none() && vector_count > 0 {
            match batch.create_bloom_filter() {
                Ok(_) => {
                    tracing::debug!(
                        "✅ Created bloom filter for batch {} with {} vectors",
                        batch_id,
                        vector_count
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️ Failed to create bloom filter for batch {}: {}",
                        batch_id,
                        e
                    );
                }
            }
        }

        tracing::debug!(
            "🚀 WAL_BEHAVIOR: Batch size info - total_size_bytes: {}, vector_count: {}",
            batch.total_size_bytes,
            vector_count
        );

        // STREAMLINED: Store batch natively in GlobalPartitionedMemtable (no duplication)
        tracing::debug!("🚀 WAL_BEHAVIOR: Calling inner.add_wal_batch()...");
        let sequences = if insert_only {
            self.inner
                .add_wal_batch_insert_only(collection_id, batch.clone())
                .await?
        } else {
            self.inner
                .add_wal_batch(collection_id, batch.clone())
                .await?
        };
        tracing::debug!(
            "🚀 WAL_BEHAVIOR: inner.add_wal_batch() returned sequences: {:?}",
            sequences
        );

        // Store batch in Write Buffer-specific coordinator for backward compatibility and coordination
        tracing::debug!("🚀 WAL_BEHAVIOR: Updating batch_coordinator...");
        let mut coordinator = self.batch_coordinator.write().await;
        coordinator.add_batch(collection_id, batch.clone())?;
        drop(coordinator);

        // Update WAL metrics
        tracing::debug!("🚀 WAL_BEHAVIOR: Updating WAL metrics...");
        let mut metrics = self.wal_metrics.write().await;
        metrics.entries_written += vector_count as u64;
        metrics.bytes_written += batch.total_size_bytes as u64;
        tracing::debug!(
            "🚀 WAL_BEHAVIOR: WAL metrics updated - entries_written: {}, bytes_written: {}",
            metrics.entries_written,
            metrics.bytes_written
        );
        drop(metrics);

        // Avoid collecting global memtable stats on the hot insert path unless
        // the debug logs that consume them are enabled.
        if tracing::enabled!(tracing::Level::DEBUG) {
            let collection_stats = self.inner.get_all_collection_stats().await;
            tracing::debug!(
                "🚀 WAL_BEHAVIOR: After add, GlobalPartitionedMemtable has {} collections",
                collection_stats.len()
            );
            for (coll_id, (entry_count, size_bytes)) in &collection_stats {
                tracing::debug!(
                    "🚀 WAL_BEHAVIOR: Collection {} has {} entries, {} bytes",
                    coll_id,
                    entry_count,
                    size_bytes
                );
            }
        }

        tracing::debug!(
            "✅ WAL_BEHAVIOR: Completed add_vector_batch for batch {} with sequences: {:?}",
            batch_id,
            sequences
        );

        Ok(sequences)
    }

    /// Mark batch as flushed (Write Buffer-specific behavior)
    pub async fn mark_batch_flushed(&self, collection_id: &str, batch_id: &str) -> Result<()> {
        let mut coordinator = self.batch_coordinator.write().await;
        coordinator.mark_batch_flushed(collection_id, batch_id)?;

        tracing::info!("✅ WAL_BATCH: Marked batch {} as flushed", batch_id);
        Ok(())
    }

    /// Clear flushed batches from memory (Write Buffer-specific behavior)
    pub async fn clear_flushed_batches(&self, collection_id: &str) -> Result<usize> {
        let mut coordinator = self.batch_coordinator.write().await;
        let cleared_count = coordinator.clear_flushed_batches(collection_id)?;

        tracing::info!("🧹 WAL_BATCH: Cleared {} flushed batches", cleared_count);
        Ok(cleared_count)
    }
}

/// Write Buffer-specific implementation
impl WALBehaviorWrapper {
    /// Get all vectors from the memtable (for recovery) - MODERN
    pub async fn get_all_vectors(&self, limit: Option<usize>) -> Result<Vec<ProximaRecord>> {
        let vectors_with_sequences = self.inner.get_all_vectors(limit).await?;
        Ok(vectors_with_sequences
            .into_iter()
            .map(|(_, vector)| vector)
            .collect())
    }

    /// Get vector by ID within a specific collection (MODERN)
    pub async fn vector_by_id(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        self.inner.vector_by_id(collection_id, vector_id).await
    }

    /// The unflushed MVCC winner for an id, tombstones included (distinguishes
    /// "deleted here" from "not here").
    pub async fn latest_version_by_id(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        self.inner
            .latest_version_by_id(collection_id, vector_id)
            .await
    }

    /// Clear flushed entries for a specific collection
    pub async fn clear_flushed_by_collection_id(&self, collection_id: &str) -> Result<usize> {
        // Delegate to the string-based method
        self.clear_flushed(collection_id).await
    }

    /// Get unflushed batches for collection (MODERN - for direct storage engine flush)
    pub async fn get_unflushed_batches(&self, collection_id: &str) -> Result<Vec<WALVectorBatch>> {
        // Return deserialized vector batches directly to storage engines
        // Storage engines handle their own serialization (SST for LSM, Parquet for VIPER)
        // This avoids double serialization/deserialization overhead

        let coordinator = self.batch_coordinator.read().await;
        let unflushed_batch_refs = coordinator.get_unflushed_batches(collection_id);

        // ZERO-COPY: Share Arc references instead of cloning entire batches
        let unflushed_batches = unflushed_batch_refs
            .into_iter()
            .map(|batch_ref| {
                // ZERO-COPY records (Arc) + REUSE the bloom filter built once at
                // ingest instead of rebuilding it on every read. The old code set
                // the filter to None then re-hashed every record×prop via
                // create_bloom_filter() on each call — wasted work that the
                // no-metadata-filter search path (the common vector search)
                // discards entirely, and a per-query allocation source under
                // concurrency. Cloning copies the bitset, not the records.
                WALVectorBatch {
                    batch_id: batch_ref.batch_id,
                    vector_records: batch_ref.vector_records.clone(), // Arc clone (pointer copy)
                    timestamp: batch_ref.timestamp,
                    total_size_bytes: batch_ref.total_size_bytes,
                    is_flushed: batch_ref.is_flushed,
                    metadata_bloom_filter: batch_ref.metadata_bloom_filter.clone(),
                }
            })
            .collect();

        Ok(unflushed_batches)
    }

    /// Atomically claim every currently unclaimed WAL batch in one collection.
    ///
    /// The returned batches are ordered by `(timestamp, batch_id)` so one source
    /// set has a deterministic materialization order. New batches appended after
    /// the snapshot remain unclaimed for the next flush epoch.
    pub async fn claim_unflushed_batches(
        &self,
        collection_id: &str,
    ) -> Result<Option<WALFlushBatchClaim>> {
        self.claim_unflushed_batches_with_mode(collection_id, WALFlushClaimMode::All)
            .await
    }

    /// Atomically claim only batches that the current immutable-segment writers
    /// can publish without dropping records.
    ///
    /// A vector-only batch is eligible. A tombstone-only batch remains durable
    /// and visible in the WAL until deletion vectors (or a tombstone segment
    /// format) can publish it. A mixed batch fails closed: retirement is
    /// batch-granular, so publishing only its vector records would silently
    /// retire the colocated tombstones.
    pub async fn claim_segment_materializable_batches(
        &self,
        collection_id: &str,
    ) -> Result<Option<WALFlushBatchClaim>> {
        self.claim_unflushed_batches_with_mode(
            collection_id,
            WALFlushClaimMode::SegmentMaterializable,
        )
        .await
    }

    async fn claim_unflushed_batches_with_mode(
        &self,
        collection_id: &str,
        mode: WALFlushClaimMode,
    ) -> Result<Option<WALFlushBatchClaim>> {
        // Hold the coordinator read guard through the process-local ownership
        // transition. Otherwise a legacy retirement/drop path could remove a
        // batch between the snapshot and registry insertion, producing a claim
        // for a source that no longer exists. Cloning a WALVectorBatch is cheap
        // because the record payload is Arc-backed; the synchronous claim lock
        // is held only for map lookups/inserts and never across an await.
        let coordinator = self.batch_coordinator.read().await;
        let mut candidates: Vec<WALVectorBatch> = coordinator
            .get_unflushed_batches(collection_id)
            .into_iter()
            .cloned()
            .collect();
        candidates.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.batch_id.to_base62().cmp(&right.batch_id.to_base62()))
        });

        let mut registry = self.flush_claims.lock().map_err(|_| {
            anyhow::anyhow!(
                "flush claim registry poisoned while claiming collection '{}'",
                collection_id
            )
        })?;

        let mut unclaimed = Vec::new();
        for batch in candidates {
            let batch_id = batch.batch_id.to_base62();
            if registry.is_claimed(collection_id, &batch_id) {
                continue;
            }
            unclaimed.push((batch_id, batch));
        }

        if matches!(mode, WALFlushClaimMode::SegmentMaterializable) {
            let mixed_batch_ids: Vec<&str> = unclaimed
                .iter()
                .filter_map(|(batch_id, batch)| {
                    (batch.flush_shape() == WALFlushBatchShape::Mixed).then_some(batch_id.as_str())
                })
                .collect();
            if !mixed_batch_ids.is_empty() {
                anyhow::bail!(
                    "flush: collection '{}' contains mixed vector/tombstone WAL batches [{}]; \
                     refusing partial segment publication because retirement is batch-granular",
                    collection_id,
                    mixed_batch_ids.join(",")
                );
            }
            unclaimed.retain(|(_, batch)| batch.flush_shape() == WALFlushBatchShape::VectorsOnly);
        }

        if unclaimed.is_empty() {
            return Ok(None);
        }

        let claim_id = self.flush_claim_sequence.fetch_add(1, Ordering::Relaxed);
        let mut batches = Vec::with_capacity(unclaimed.len());
        let mut batch_ids = Vec::with_capacity(unclaimed.len());
        for (batch_id, batch) in unclaimed {
            registry.insert(collection_id, batch_id.clone(), claim_id);
            batch_ids.push(batch_id);
            batches.push(batch);
        }
        drop(registry);
        drop(coordinator);

        Ok(Some(WALFlushBatchClaim {
            collection_id: collection_id.to_string(),
            claim_id,
            batch_ids,
            batches,
            registry: self.flush_claims.clone(),
            finalized: false,
        }))
    }

    /// Retire exactly the source batches owned by a successfully published
    /// flush. Batches appended while encoding was in flight are untouched.
    pub async fn complete_flush_claim(&self, claim: &mut WALFlushBatchClaim) -> Result<usize> {
        if !Arc::ptr_eq(&self.flush_claims, &claim.registry) {
            anyhow::bail!(
                "flush claim for '{}' belongs to a different WAL behavior",
                claim.collection_id
            );
        }
        {
            let registry = self.flush_claims.lock().map_err(|_| {
                anyhow::anyhow!(
                    "flush claim registry poisoned while completing collection '{}'",
                    claim.collection_id
                )
            })?;
            if !registry.owns_all(&claim.collection_id, claim.claim_id, &claim.batch_ids) {
                anyhow::bail!(
                    "flush claim {} no longer owns every source batch for '{}'",
                    claim.claim_id,
                    claim.collection_id
                );
            }
        }

        // Lock and validate the coordinator before mutating the inner store.
        // Holding this guard across the inner retirement ensures the second
        // mirror cannot change between validation and removal. If inner
        // retirement rejects, neither mirror is changed; if it succeeds, the
        // already-validated coordinator removal cannot fail due to a race.
        let mut coordinator = self.batch_coordinator.write().await;
        coordinator.validate_batches_exact(&claim.collection_id, &claim.batch_ids)?;
        let removed = self
            .inner
            .remove_batches_exact(&claim.collection_id, &claim.batch_ids)
            .await?;
        let coordinator_removed =
            coordinator.remove_batches_exact(&claim.collection_id, &claim.batch_ids)?;
        if coordinator_removed != removed {
            tracing::warn!(
                collection_id = %claim.collection_id,
                claim_id = claim.claim_id,
                inner_records = removed,
                coordinator_records = coordinator_removed,
                "flush retirement stores reported different record counts"
            );
        }
        drop(coordinator);
        claim.finalize()?;
        Ok(removed)
    }

    /// Number of source batches currently owned by in-flight flush jobs.
    pub fn claimed_flush_batch_count(&self, collection_id: &str) -> Result<usize> {
        let registry = self.flush_claims.lock().map_err(|_| {
            anyhow::anyhow!(
                "flush claim registry poisoned while reading collection '{}'",
                collection_id
            )
        })?;
        Ok(registry
            .claimed
            .get(collection_id)
            .map(HashMap::len)
            .unwrap_or(0))
    }

    /// Count unflushed source batches without cloning their record payloads.
    ///
    /// This is the admission fast path. A count is used instead of byte sum
    /// because tombstone/legacy batches may legitimately report zero bytes and
    /// still require durable retirement.
    pub async fn unflushed_batch_count(&self, collection_id: &str) -> usize {
        self.batch_coordinator
            .read()
            .await
            .unflushed_batch_count(collection_id)
    }

    /// Sum unflushed bytes for a collection without cloning batches (hot-path
    /// accessor for the inline flush trigger — `get_unflushed_batches` clones every
    /// batch and regressed write throughput when called per write).
    pub async fn unflushed_bytes(&self, collection_id: &str) -> u64 {
        let coordinator = self.batch_coordinator.read().await;
        coordinator.unflushed_bytes(collection_id)
    }

    /// TD-FLUSH-3: predicted segment bytes (RaBitQ+SQ8) of the unflushed delta —
    /// the flush-floor input. Cheap map-sum, same contract as `unflushed_bytes`.
    pub async fn unflushed_predicted_bytes(&self, collection_id: &str) -> u64 {
        let coordinator = self.batch_coordinator.read().await;
        coordinator.unflushed_predicted_bytes(collection_id)
    }

    /// Clear flushed batches for collection (MODERN - after successful storage engine flush)
    pub async fn clear_flushed(&self, collection_id: &str) -> Result<usize> {
        // Clear data from GlobalPartitionedMemtable after successful storage engine flush
        // This prevents memory explosion and ensures data consistency
        self.inner.clear_flushed_batches(collection_id).await
    }

    /// List all collections that have unflushed data in memtable
    /// Used for graceful shutdown to flush all pending data to storage engines
    pub async fn list_collections_with_unflushed_data(&self) -> Vec<String> {
        let coordinator = self.batch_coordinator.read().await;
        coordinator.list_collections_with_unflushed_data()
    }

    /// Get statistics for WAL collection management (MODERN)
    pub async fn get_stats(&self) -> Result<HashMap<String, WALStats>> {
        let all_stats = self.inner.get_all_collection_stats().await;
        let mut stats_map = HashMap::new();

        for (collection_id, (vector_count, size_bytes)) in all_stats {
            stats_map.insert(
                collection_id,
                WALStats {
                    total_entries: vector_count as u64,
                    memory_entries: vector_count as u64,
                    disk_segments: 0,
                    total_disk_size_bytes: 0,
                    memory_size_bytes: size_bytes as u64,
                    collections_count: 1,
                    last_flush_time: None,
                    write_throughput_entries_per_sec: 0.0,
                    read_throughput_entries_per_sec: 0.0,
                    compression_ratio: 1.0,
                },
            );
        }

        Ok(stats_map)
    }

    #[allow(dead_code)]
    fn get_operation_type(&self, operation: &WALOperation) -> u8 {
        // Map operation types to numeric codes
        match operation.operation_type.as_str() {
            "upsert_batch" => 1,
            "delete_batch" => 2,
            _ => 0, // Unknown operation
        }
    }

    #[allow(dead_code)]
    async fn serialize_operation(&self, operation: &WALOperation) -> Result<Vec<u8>> {
        Ok(bincode::serialize(operation)?)
    }

    /// Flush vectors up to sequence number (MODERN)
    pub async fn flush_all_vectors(&self) -> Result<Vec<ProximaRecord>> {
        // Get all vectors from memtable
        let all_vectors_with_sequences = self.inner.get_all_vectors(None).await?;

        // Extract just the vectors
        let vectors_to_flush: Vec<ProximaRecord> = all_vectors_with_sequences
            .into_iter()
            .map(|(_, vector)| vector)
            .collect();

        // Mark all batches as flushed (simulating successful storage engine write)
        self.inner.mark_all_batches_as_flushed().await?;

        // Remove flushed vectors
        self.inner.clear_all_flushed().await?;

        // MVCC tracking no longer uses sequences - batches handle versioning

        Ok(vectors_to_flush)
    }

    /// Get Write Buffer-specific metrics
    pub async fn get_wal_metrics(&self) -> WriteBufferMetrics {
        self.wal_metrics.read().await.clone()
    }

    /// List all collections currently tracked in the write buffer (coordinator view)
    pub async fn list_collections(&self) -> Vec<String> {
        let guard = self.batch_coordinator.read().await;
        guard.batches.keys().cloned().collect()
    }

    /// Get latest version of a vector by ID (MODERN - MVCC handled by GlobalPartitionedMemtable)
    pub async fn get_latest_vector(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        // MVCC is now handled natively by GlobalPartitionedMemtable
        // No need for separate version tracking at WAL level
        self.inner.vector_by_id(collection_id, vector_id).await
    }

    /// Cleanup old versions (keep only N latest)
    pub async fn cleanup_versions(&self, vector_id: &str, keep_count: usize) -> Result<usize> {
        if !self.config.enable_mvcc {
            return Ok(0);
        }

        let mut versions = self.mvcc_versions.write().await;
        let sequences = match versions.get_mut(vector_id) {
            Some(seqs) => seqs,
            None => return Ok(0),
        };

        if sequences.len() <= keep_count {
            return Ok(0);
        }

        // Sort and keep only latest versions
        sequences.sort();
        let old_sequences = sequences
            .drain(0..sequences.len() - keep_count)
            .collect::<Vec<_>>();
        let removed_count = old_sequences.len();

        drop(versions);

        // MVCC cleanup no longer uses sequences - handled at batch level
        // Old entries are removed when batches are flushed

        Ok(removed_count)
    }
}

impl WALBehaviorWrapper {
    /// Get all vectors from all collections ordered by sequence (MODERN)
    pub async fn get_all_ordered(&self) -> Result<Vec<(u64, ProximaRecord)>> {
        self.inner.get_all_vectors(None).await
    }

    /// Get all vectors for a specific collection (MODERN)
    pub async fn get_collection_vectors(&self, collection_id: &str) -> Result<Vec<ProximaRecord>> {
        // Direct access to collection vectors from GlobalPartitionedMemtable
        let vectors = self.inner.get_collection_vectors(collection_id).await?;

        tracing::debug!(
            "🚀 MODERN_GET_ALL: Returning {} vectors for collection {} (direct VectorRecord access)",
            vectors.len(),
            collection_id
        );

        Ok(vectors)
    }

    /// Return every physical unflushed record, including tombstones, expired
    /// rows, and superseded versions. Intended for cross-source reconciliation.
    pub async fn get_collection_vectors_raw(
        &self,
        collection_id: &str,
    ) -> Result<Vec<ProximaRecord>> {
        self.inner.get_collection_vectors_raw(collection_id).await
    }

    /// Paginated, deduped, time-ordered scan of a collection's unflushed records
    /// (TD-099(3d) push-down). Returns up to `limit` records with canonical key
    /// `(updated_at_ns, oid)` strictly greater than `after`, in ascending order,
    /// passing `predicate`, excluding TTL-expired-at-`now_ns`. Byte-identical to
    /// `get_collection_vectors` + sort + cursor-filter + take, but O(log d +
    /// limit) per page once the per-collection scan index is warm.
    ///
    /// `after` is the raw `(last_updated_at_ns, last_oid)` cursor tuple — kept
    /// protocol-agnostic so the storage layer stays free of the `ScanCursor`
    /// codec (and reusable for future predicate push-downs).
    pub async fn stream_unflushed_records(
        &self,
        collection_id: &str,
        after: Option<(i64, &str)>,
        limit: usize,
        predicate: Option<&(dyn Fn(&ProximaRecord) -> bool + Send + Sync)>,
        now_ns: i64,
    ) -> Result<Vec<ProximaRecord>> {
        self.inner
            .scan_collection_paginated(collection_id, after, limit, predicate, now_ns)
            .await
    }

    /// Get current size in bytes (with actual vector data size calculation)
    pub async fn size_bytes(&self) -> usize {
        // Direct access to GlobalPartitionedMemtable
        self.inner.size_bytes().await
    }

    /// Get current entry count
    pub async fn len(&self) -> usize {
        self.inner.len().await
    }

    /// Check if the memtable has no entries
    pub async fn is_empty(&self) -> bool {
        self.inner.len().await == 0
    }

    /// Distributed: register idempotency token; returns false if duplicate
    pub async fn register_idempotency(&self, collection_id: &str, token: &str) -> bool {
        let mut map = self.idempotency_tokens.write().await;
        let set = map
            .entry(collection_id.to_string())
            .or_insert_with(Default::default);
        set.insert(token.to_string())
    }

    /// Distributed: get next partition sequence
    pub async fn next_partition_sequence(&self, collection_id: &str, partition_key: &str) -> u64 {
        let mut all = self.partition_sequences.write().await;
        let per = all
            .entry(collection_id.to_string())
            .or_insert_with(Default::default);
        let seq = per.entry(partition_key.to_string()).or_insert(0);
        *seq += 1;
        *seq
    }

    /// Distributed add: with partition & idempotency token support
    pub async fn add_vector_batch_distributed(
        &self,
        collection_id: &str,
        partition_key: Option<&str>,
        idempotency_token: Option<&str>,
        batch: WALVectorBatch,
    ) -> Result<Vec<u64>> {
        if let Some(token) = idempotency_token {
            // If token already exists, ignore batch (idempotent)
            if !self.register_idempotency(collection_id, token).await {
                tracing::info!("WAL idempotent: skipping duplicate token {}", token);
                return Ok(Vec::new());
            }
        }
        if let Some(pk) = partition_key {
            let _ = self.next_partition_sequence(collection_id, pk).await;
        }
        self.add_vector_batch(collection_id, batch).await
    }

    /// Get collections that need flushing (global WAL, collection-partitioned)
    pub async fn collections_needing_flush(&self) -> Result<Vec<crate::core::String>> {
        tracing::info!("🔍 COLLECTIONS_FLUSH_CHECK: Checking which collections need flushing...");

        // Use collection-aware flush detection from global partitioned memtable
        let threshold_bytes = self.config.flush_threshold_bytes;
        let collections_to_flush = self
            .inner
            .collections_needing_flush(threshold_bytes)
            .await?;

        tracing::info!(
            "🔍 COLLECTIONS_FLUSH_CHECK: {} collections need flushing with threshold {}MB",
            collections_to_flush.len(),
            threshold_bytes / 1024 / 1024
        );

        for collection_id in &collections_to_flush {
            let (entries, size) = self.inner.stats(collection_id).await;
            tracing::info!(
                "🔍 COLLECTIONS_FLUSH_CHECK: Collection {} has {} entries, {} bytes ({}MB)",
                collection_id,
                entries,
                size,
                size / 1024 / 1024
            );
        }

        Ok(collections_to_flush)
    }

    /// Remove a specific batch from the memtable (for atomic rollback)
    pub async fn remove_batch(&self, collection_id: &str, batch_id: &str) -> Result<()> {
        // First remove from coordinator (for backward compatibility)
        let mut coordinator = self.batch_coordinator.write().await;

        // Remove batch from coordinator
        if let Some(collection_batches) = coordinator.batches.get_mut(collection_id)
            && let Some(removed_batch) = collection_batches.remove(batch_id)
        {
            // Remove vector index entries for this batch
            for vector_record in removed_batch.vector_records.iter() {
                if !vector_record.oid.is_empty()
                    && coordinator
                        .vector_index
                        .get(&vector_record.oid)
                        .is_some_and(|(_, indexed_batch, _)| indexed_batch == batch_id)
                {
                    coordinator.vector_index.remove(&vector_record.oid);
                }
            }
            // TD-FLUSH-3: credit the removed batch's predicted bytes.
            coordinator.credit_predicted(collection_id, batch_id);
        }
        drop(coordinator);

        // IMPORTANT: Also remove from the actual GlobalPartitionedMemtable
        // This is the real storage - coordinator is just for backward compatibility
        self.inner.remove_batch(collection_id, batch_id).await
    }

    /// Get stats for all collections
    pub async fn stats(
        &self,
    ) -> Result<
        std::collections::HashMap<String, crate::storage::persistence::write_ahead_log::WALStats>,
    > {
        let mut stats = std::collections::HashMap::new();

        // Get list of all collections from the inner memtable
        let collections = self.inner.list_collections().await?;

        for collection_id in collections {
            let (entries, size) = self.inner.stats(&collection_id).await;
            stats.insert(
                collection_id.clone(),
                crate::storage::persistence::write_ahead_log::WALStats {
                    total_entries: entries as u64,
                    memory_entries: entries as u64,
                    disk_segments: 0,
                    total_disk_size_bytes: 0,
                    memory_size_bytes: size as u64,
                    collections_count: 1,
                    last_flush_time: None,
                    write_throughput_entries_per_sec: 0.0,
                    read_throughput_entries_per_sec: 0.0,
                    compression_ratio: 1.0,
                },
            );
        }

        Ok(stats)
    }

    /// Get statistics with String keys
    pub async fn get_stats_by_collection_id(
        &self,
    ) -> Result<
        std::collections::HashMap<
            crate::core::String,
            crate::storage::persistence::write_ahead_log::WALStats,
        >,
    > {
        // Call the main get_stats method and convert String keys to String
        let string_stats = self.stats().await?;
        let mut result = std::collections::HashMap::new();
        for (k, v) in string_stats {
            result.insert(k, v);
        }
        Ok(result)
    }

    /// Search for specific vector by ID (MODERN)
    pub async fn search_vector(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        self.inner.vector_by_id(collection_id, vector_id).await
    }

    /// Get vectors for specific collection with limit (MODERN)
    pub async fn get_collection_vectors_with_limit(
        &self,
        collection_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ProximaRecord>> {
        let mut vectors = self.inner.get_collection_vectors(collection_id).await?;

        // Apply limit if specified
        if let Some(limit) = limit {
            vectors.truncate(limit);
        }

        Ok(vectors)
    }

    /// Get collection-specific statistics
    pub async fn get_collection_stats(
        &self,
        collection_id: &str,
    ) -> Result<crate::storage::persistence::write_ahead_log::WALStats> {
        let all_stats = WALBehaviorWrapper::get_stats(self).await?;

        match all_stats.get(collection_id) {
            Some(stats) => Ok(stats.clone()),
            None => Ok(crate::storage::persistence::write_ahead_log::WALStats {
                total_entries: 0,
                memory_entries: 0,
                disk_segments: 0,
                total_disk_size_bytes: 0,
                memory_size_bytes: 0,
                collections_count: 0,
                last_flush_time: None,
                write_throughput_entries_per_sec: 0.0,
                read_throughput_entries_per_sec: 0.0,
                compression_ratio: 1.0,
            }),
        }
    }

    /// Drop collection from memtable (MODERN)
    pub async fn drop_collection(&self, collection_id: &str) -> Result<usize> {
        // Use the collection-specific clear method for efficient removal
        self.inner.clear_flushed_batches(collection_id).await
    }

    /// Perform maintenance operations
    pub async fn maintenance(
        &self,
    ) -> Result<crate::storage::persistence::write_ahead_log::WALStats> {
        // Cleanup old MVCC versions if enabled
        if self.config.enable_mvcc {
            let _cleaned = self.cleanup_versions("", 10).await?; // Keep 10 versions
        }

        // Return current stats after maintenance
        let all_stats = WALBehaviorWrapper::get_stats(self).await?;

        // Return aggregated stats
        let total_entries: u64 = all_stats.values().map(|s| s.total_entries).sum();
        let total_memory: u64 = all_stats.values().map(|s| s.memory_size_bytes).sum();

        Ok(crate::storage::persistence::write_ahead_log::WALStats {
            total_entries,
            memory_entries: total_entries,
            disk_segments: 0,
            total_disk_size_bytes: 0,
            memory_size_bytes: total_memory,
            collections_count: all_stats.len(),
            last_flush_time: None,
            write_throughput_entries_per_sec: 0.0,
            read_throughput_entries_per_sec: 0.0,
            compression_ratio: 1.0,
        })
    }

    /// Get unflushed batches for atomic flush (MODERN)
    pub async fn atomic_mark_for_flush(
        &self,
        collection_id: &str,
        _up_to_sequence: u64,
    ) -> Result<Vec<WALVectorBatch>> {
        // Return unflushed batches directly for storage engine processing
        self.get_unflushed_batches(collection_id).await
    }

    /// Complete flush and remove marked entries
    pub async fn complete_flush_removal(&self, collection_id: &str) -> Result<usize> {
        self.clear_flushed(collection_id).await
    }

    /// Abort flush and restore batches
    pub async fn abort_flush_restore(
        &self,
        collection_id: &str,
        _batches: Vec<WALVectorBatch>,
    ) -> Result<()> {
        // In a real implementation, this would restore the batches
        // For now, this is a no-op since we haven't actually removed them
        tracing::warn!(
            "Flush aborted for collection {}, batches preserved in memtable",
            collection_id
        );
        Ok(())
    }
}

/// Write Buffer-specific metrics
#[derive(Debug, Clone, Default)]
pub struct WriteBufferMetrics {
    pub entries_written: u64,
    pub bytes_written: u64,
    pub flushes_performed: u64,
    pub total_flushed_bytes: u64,
    pub recovery_operations: u64,
    pub mvcc_versions_active: usize,
}

/// Flush coordination state
#[derive(Debug, Clone, Default)]
struct FlushState {
    #[allow(dead_code)]
    last_flush_sequence: u64,
    #[allow(dead_code)]
    flush_in_progress: bool,
    #[allow(dead_code)]
    flush_start_time: Option<std::time::Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::{EmbeddingCell, LabelSet, ProximaRecord};
    // Tests now use unified add_vector_batch API only

    fn test_vector_record(id: &str, vector: Vec<f32>) -> ProximaRecord {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;
        let dim = vector.len() as u32;
        ProximaRecord {
            oid: id.to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 1,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: Some("test".to_string()),
            memory_type: None,
            props: std::collections::HashMap::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: if !vector.is_empty() {
                vec![EmbeddingCell::new_fp32(
                    "default",
                    "dense_vector",
                    dim,
                    vector,
                )]
            } else {
                vec![]
            },
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }

    fn test_wal_batch(records: Vec<ProximaRecord>) -> WALVectorBatch {
        WALVectorBatch {
            batch_id: BatchId::new(),
            vector_records: Arc::new(records),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: 1024,
            is_flushed: false,
            metadata_bloom_filter: None,
        }
    }

    #[tokio::test]
    async fn test_wal_behavior_wrapper() {
        let config = MemtableConfig::default();
        let wal_wrapper = WALBehaviorWrapper::new(config);

        // Create test vector records using the new unified API
        let vector_record1 = test_vector_record("test_vector_1", vec![0.1, 0.2, 0.3]);
        let vector_record2 = test_vector_record("test_vector_2", vec![0.4, 0.5, 0.6]);

        // Test first batch insertion using unified add_vector_batch API
        let batch1 = WALVectorBatch {
            batch_id: BatchId::new(),
            vector_records: Arc::new(vec![vector_record1]),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: 1024,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        let sequences1 = wal_wrapper
            .add_vector_batch("test_collection", batch1)
            .await
            .unwrap();
        let seq1 = sequences1[0];

        // Test second batch insertion
        let batch2 = WALVectorBatch {
            batch_id: BatchId::new(),
            vector_records: Arc::new(vec![vector_record2]),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: 1024,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        let sequences2 = wal_wrapper
            .add_vector_batch("test_collection", batch2)
            .await
            .unwrap();
        let seq2 = sequences2[0];

        // Verify sequence ordering (newer batches get higher sequences)
        assert!(seq2 > seq1);

        // Verify batch count
        let unflushed_batches = wal_wrapper
            .get_unflushed_batches("test_collection")
            .await
            .unwrap();
        assert_eq!(unflushed_batches.len(), 2);

        // Test retrieving all vectors
        let entries = wal_wrapper.get_all_vectors(None).await.unwrap();
        assert_eq!(entries.len(), 2);

        // Flush threshold checking is done by VectorOperationsService, not here

        // Test flush operation
        let flushed = wal_wrapper.flush_all_vectors().await.unwrap();
        assert_eq!(flushed.len(), 2);
        assert_eq!(wal_wrapper.len().await, 0);

        // Test metrics
        let metrics = wal_wrapper.get_wal_metrics().await;
        assert_eq!(metrics.entries_written, 2);
        assert_eq!(metrics.flushes_performed, 0); // flush_up_to_sequence doesn't update this metric
    }

    #[tokio::test]
    async fn test_insert_only_wrapper_rejects_duplicate_ids_in_request() {
        let wal_wrapper = WALBehaviorWrapper::new(MemtableConfig::default());
        let batch = test_wal_batch(vec![
            test_vector_record("record_1", vec![1.0, 0.0]),
            test_vector_record("record_1", vec![0.0, 1.0]),
        ]);

        let error = wal_wrapper
            .add_vector_batch_insert_only("test_collection", batch)
            .await
            .expect_err("duplicate ids in insert-only WAL batch should fail");

        assert!(error.to_string().contains("INSERT_CONFLICT"));
        assert!(
            wal_wrapper
                .vector_by_id("test_collection", "record_1")
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            wal_wrapper
                .get_unflushed_batches("test_collection")
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_insert_only_wrapper_rejects_existing_id_without_partial_batch() {
        let wal_wrapper = WALBehaviorWrapper::new(MemtableConfig::default());
        wal_wrapper
            .add_vector_batch_insert_only(
                "test_collection",
                test_wal_batch(vec![test_vector_record("record_1", vec![1.0, 0.0])]),
            )
            .await
            .unwrap();

        let error = wal_wrapper
            .add_vector_batch_insert_only(
                "test_collection",
                test_wal_batch(vec![
                    test_vector_record("record_1", vec![0.0, 1.0]),
                    test_vector_record("record_2", vec![0.5, 0.5]),
                ]),
            )
            .await
            .expect_err("existing ids in insert-only WAL batch should fail");

        assert!(error.to_string().contains("INSERT_CONFLICT"));
        assert!(
            wal_wrapper
                .vector_by_id("test_collection", "record_2")
                .await
                .unwrap()
                .is_none(),
            "failed insert-only WAL batch must not partially add new records"
        );
        assert_eq!(
            wal_wrapper
                .get_unflushed_batches("test_collection")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_wal_mvcc_functionality() {
        let mut config = MemtableConfig::default();
        config.enable_mvcc = true;

        let wal_wrapper = WALBehaviorWrapper::new(config);

        let vector_id = "test_vector_mvcc";

        // Insert multiple versions of the same vector using unified API
        for i in 0u32..3 {
            let mut props = std::collections::HashMap::new();
            props.insert(
                "version".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(i.to_string()),
                ),
            );

            let mut record = test_vector_record(vector_id, vec![i as f32, (i + 1) as f32]);
            record.props = props;
            record.record_version = (i + 1) as u64;

            let batch = WALVectorBatch {
                batch_id: BatchId::new(),
                vector_records: Arc::new(vec![record]),
                timestamp: std::time::SystemTime::now(),
                total_size_bytes: 1024,
                is_flushed: false,
                metadata_bloom_filter: None,
            };

            wal_wrapper
                .add_vector_batch("test_collection", batch)
                .await
                .unwrap();
        }

        // Test that vectors were added (using modern API)
        let all_vectors = wal_wrapper.get_all_ordered().await.unwrap();
        assert!(all_vectors.len() >= 3);

        // Test that we can retrieve vectors by collection
        let collection_vectors = wal_wrapper.get_all_vectors(None).await.unwrap();
        assert!(!collection_vectors.is_empty());

        // Verify vector data integrity
        let found_vectors: Vec<_> = all_vectors
            .iter()
            .filter(|(_, record)| record.oid == vector_id)
            .collect();
        assert!(!found_vectors.is_empty());
    }

    #[tokio::test]
    async fn adr081_flush_claim_drop_releases_exact_source_batches() {
        let wal = WALBehaviorWrapper::new(MemtableConfig::default());
        wal.add_vector_batch(
            "claim_drop",
            test_wal_batch(vec![test_vector_record("v1", vec![1.0, 0.0])]),
        )
        .await
        .unwrap();

        let claim = wal
            .claim_unflushed_batches("claim_drop")
            .await
            .unwrap()
            .expect("first claim");
        assert_eq!(claim.batch_ids().len(), 1);
        assert_eq!(wal.claimed_flush_batch_count("claim_drop").unwrap(), 1);
        assert!(
            wal.claim_unflushed_batches("claim_drop")
                .await
                .unwrap()
                .is_none(),
            "a claimed source must not be served to another flush"
        );

        drop(claim);
        assert_eq!(wal.claimed_flush_batch_count("claim_drop").unwrap(), 0);
        assert!(
            wal.claim_unflushed_batches("claim_drop")
                .await
                .unwrap()
                .is_some(),
            "drop/cancellation must return the exact source claim"
        );
    }

    #[tokio::test]
    async fn adr081_exact_flush_retirement_preserves_batch_appended_after_claim() {
        let wal = WALBehaviorWrapper::new(MemtableConfig::default());
        wal.add_vector_batch(
            "claim_exact",
            test_wal_batch(vec![test_vector_record("old", vec![1.0, 0.0])]),
        )
        .await
        .unwrap();
        let mut claim = wal
            .claim_unflushed_batches("claim_exact")
            .await
            .unwrap()
            .expect("source claim");

        wal.add_vector_batch(
            "claim_exact",
            test_wal_batch(vec![test_vector_record("new", vec![0.0, 1.0])]),
        )
        .await
        .unwrap();

        assert_eq!(
            wal.complete_flush_claim(&mut claim).await.unwrap(),
            1,
            "only the claimed record is retired"
        );
        let remaining = wal.get_unflushed_batches("claim_exact").await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].vector_records[0].oid, "new");
        assert!(
            wal.vector_by_id("claim_exact", "old")
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            wal.vector_by_id("claim_exact", "new")
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn segment_claim_retains_tombstone_only_batches() {
        let wal = WALBehaviorWrapper::new(MemtableConfig::default());
        wal.add_vector_batch(
            "segment_tombstones",
            test_wal_batch(vec![test_vector_record("dead", Vec::new())]),
        )
        .await
        .unwrap();

        assert!(
            wal.claim_segment_materializable_batches("segment_tombstones")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            wal.get_unflushed_batches("segment_tombstones")
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            wal.claimed_flush_batch_count("segment_tombstones").unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn segment_claim_retires_vectors_without_colocated_tombstone_batch() {
        let wal = WALBehaviorWrapper::new(MemtableConfig::default());
        wal.add_vector_batch(
            "segment_split",
            test_wal_batch(vec![test_vector_record("live", vec![1.0, 0.0])]),
        )
        .await
        .unwrap();
        wal.add_vector_batch(
            "segment_split",
            test_wal_batch(vec![test_vector_record("dead", Vec::new())]),
        )
        .await
        .unwrap();

        let mut claim = wal
            .claim_segment_materializable_batches("segment_split")
            .await
            .unwrap()
            .expect("vector-only batch must be claimable");
        assert_eq!(claim.batches().len(), 1);
        assert_eq!(claim.batches()[0].vector_records[0].oid, "live");
        assert_eq!(wal.complete_flush_claim(&mut claim).await.unwrap(), 1);

        let remaining = wal.get_unflushed_batches("segment_split").await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].vector_records[0].oid, "dead");
    }

    #[tokio::test]
    async fn segment_claim_rejects_mixed_batch_without_partial_ownership() {
        let wal = WALBehaviorWrapper::new(MemtableConfig::default());
        wal.add_vector_batch(
            "segment_mixed",
            test_wal_batch(vec![
                test_vector_record("live", vec![1.0, 0.0]),
                test_vector_record("dead", Vec::new()),
            ]),
        )
        .await
        .unwrap();

        let error = wal
            .claim_segment_materializable_batches("segment_mixed")
            .await
            .expect_err("mixed source cannot be published or retired partially");
        assert!(error.to_string().contains("mixed vector/tombstone"));
        assert_eq!(wal.claimed_flush_batch_count("segment_mixed").unwrap(), 0);
        assert_eq!(
            wal.get_unflushed_batches("segment_mixed")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn memtable_threshold_backpressure_is_typed_and_disableable() {
        assert!(
            memtable_threshold_backpressure("collection", 10, 0).is_none(),
            "a zero threshold disables the legacy memtable guard"
        );
        assert!(memtable_threshold_backpressure("collection", 9, 10).is_none());

        let error = memtable_threshold_backpressure("collection", 10, 10)
            .expect("usage at the threshold must be rejected");
        assert_eq!(error.collection_id, "collection");
        assert_eq!(error.fill_pct, 100.0);

        let classified = anyhow::Error::new(error);
        assert_eq!(
            crate::storage::persistence::write_ahead_log::flush_policy::write_batch_error_code(
                &classified,
                "RECORD_INSERT_FAILED"
            ),
            "WAL_BACKPRESSURE"
        );
    }
}
