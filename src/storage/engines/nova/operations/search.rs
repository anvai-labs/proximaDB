//! Search operations module for NOVA engine
//! Handles all search-related logic including hierarchical pruning and progressive refinement

use anyhow::Result;
use proximadb_records::{ProximaRecord, conversions::proxima_tree_to_value_map};
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::persistence::filesystem::FilesystemFactory;
use proximadb_distance_kernel::DistanceMetric;

/// Handles all search operations for NOVA engine
pub struct NovaSearchOperations {
    filesystem: Arc<FilesystemFactory>,
    distance_engine: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,
}

/// Operational kill-switch for TD-040 vector-bounds row-group pruning (shared
/// with the SST path). `PROXIMADB_VECTOR_BOUNDS_PRUNE_DISABLE=1`/`true` forces
/// the full-read path. Read per search (cheap); the prune is recall-preserving,
/// so this is purely an escape hatch.
fn vector_bounds_prune_disabled() -> bool {
    std::env::var("PROXIMADB_VECTOR_BOUNDS_PRUNE_DISABLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Squared L2 distance from `query` to a row-group `centroid` for ordering.
/// Dim mismatch ⇒ `INFINITY` so the group sorts to the tail (never seeded).
fn centroid_l2_sq(query: &[f32], centroid: &[f32]) -> f32 {
    if centroid.len() != query.len() {
        return f32::INFINITY;
    }
    query
        .iter()
        .zip(centroid.iter())
        .map(|(q, c)| {
            let d = q - c;
            d * d
        })
        .sum()
}

impl NovaSearchOperations {
    /// Create new search operations handler
    pub fn new(filesystem: Arc<FilesystemFactory>, distance_metric: DistanceMetric) -> Self {
        Self {
            filesystem,
            distance_engine: Arc::new(
                proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(distance_metric),
            ),
        }
    }

    /// Search vectors with unified interface
    pub async fn search_vectors_unified(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // Extract search parameters from context
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow::anyhow!("No query vector provided"))?;
        let k = ctx.top_k();
        let _distance_metric = ctx.distance_metric();
        let collection_id = &ctx.collection.id;
        let filter_expression = ctx.search_params.filter_expression.as_ref();

        info!(
            "🔍 NOVA: Searching with k={}, query_dim={}, filters={:?}",
            k,
            query_vector.len(),
            filter_expression.is_some()
        );

        let collection_size = 1000; // Default collection size estimate

        // For now, implement direct search logic here
        // Check if we should use progressive search
        if self.should_use_progressive_search(k, collection_size, filter_expression.is_some()) {
            self.search_with_progressive_refinement(ctx, collection_id)
                .await
        } else if self.should_use_streaming_search(k, collection_size) {
            self.search_with_streaming(ctx, collection_id).await
        } else {
            self.search_standard(ctx, collection_id).await
        }
    }

    /// Search with progressive refinement
    async fn search_with_progressive_refinement(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
        collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // For now, use standard search as progressive search needs more setup
        self.search_standard(ctx, collection_id).await
    }

    /// Search with streaming
    async fn search_with_streaming(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
        collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // For now, use standard search as streaming search needs more setup
        self.search_standard(ctx, collection_id).await
    }

    /// Standard search without optimization
    #[allow(clippy::single_match)]
    async fn search_standard(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
        _collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // Get search parameters from context
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow::anyhow!("No query vector provided"))?;
        let k = ctx.top_k();
        let filter_expression = ctx.search_params.filter_expression.as_ref();

        // Get files for the collection
        // NOVA stores files in {base_location}/{collection_id}/data (standard path)
        // Production behavior: metadata.storage_path is base_location
        let base_location = ctx
            .storage_url()
            .ok_or_else(|| anyhow::anyhow!("No storage path in context"))?;
        let collection_id = &ctx.collection.id;

        // ADR-031 Phase 4d: use the typed data path when the collection carries
        // a typed identity (env ON at create); else legacy (mixed-read-safe).
        let typed_identity =
            crate::storage::trait_components::path_resolver::typed_identity_from_storage_assignment(
                ctx.collection.storage_assignment.as_ref(),
            );
        let typed_identity =
            crate::storage::trait_components::path_resolver::typed_path_identity(typed_identity);
        let data_path = crate::storage::trait_components::path_resolver::collection_data_path_typed(
            base_location,
            collection_id,
            typed_identity,
        );

        debug!(
            "📂 NOVA search: base_location={}, collection_id={}",
            base_location, collection_id
        );
        debug!("📂 NOVA search: Constructed data_path={}", data_path);

        let fs = self.filesystem.get_filesystem(&data_path)?;

        // List files in the data directory
        let entries = match fs.list(&data_path).await {
            Ok(e) => e,
            Err(err) => {
                debug!(
                    "📂 NOVA search: Failed to list directory {}: {}",
                    data_path, err
                );
                return Ok(Vec::new());
            }
        };

        debug!(
            "📂 NOVA search: Listed {} entries in {}",
            entries.len(),
            data_path
        );
        for entry in &entries {
            debug!(
                "  - {} (is_dir={}, name={})",
                entry.url, entry.metadata.is_directory, entry.name
            );
        }

        let files: Vec<String> = entries
            .into_iter()
            .filter(|e| !e.metadata.is_directory && e.name.ends_with(".parquet"))
            .map(|e| format!("{}/{}", data_path, e.name))
            .collect();

        if files.is_empty() {
            debug!("📂 NOVA search: No parquet files found in {}", data_path);
            return Ok(Vec::new());
        }

        debug!(
            "📂 NOVA search: Found {} parquet files in {}",
            files.len(),
            data_path
        );

        // Use bounded priority queue to maintain only top-k results
        let mut priority_queue = BoundedPriorityQueue::new(k);
        let dimension = query_vector.len();

        // Track search statistics
        let mut files_scanned = 0usize;
        let total_files = files.len();

        // TD-040: per-row-group vector-bounds pruning may engage only for an
        // unfiltered exact-L2 search with the kill-switch off; otherwise NOVA
        // reads each file in full (today's behavior). Recall-preserving — a row
        // group is skipped only when its bounding box provably cannot hold a
        // top-k candidate (Euclidean clamp-gap lower bound > current k-th best).
        let prune_enabled = matches!(ctx.distance_metric(), DistanceMetric::Euclidean)
            && !ctx.search_params.block_prune.force_exact
            && filter_expression.is_none()
            && !vector_bounds_prune_disabled();
        let mut bounds_pruned_total: u64 = 0;

        if prune_enabled {
            // TWO passes (TD-ENGINE-MVCC-1), only when TD-040 bounds pruning is
            // on: pass 2 skips row groups, and a newer copy of an id can sit in
            // a skipped group, so the index must cover every row BEFORE scoring.
            // Pass 1 is a key-only column projection (ranged GETs of
            // id/timestamp/updated_at/version/expires_at, never the vectors).
            //
            // On the service path this branch is not reached for NOVA:
            // `StorageQueryContext` forces `force_exact` for NOVA/VIPER/RAPTOR,
            // so it is defence-in-depth for direct-engine callers.
            // Pass 1 (TD-ENGINE-MVCC-1): each id's newest copy across EVERY row of
            // EVERY file, before anything is scored, so a row is scored only if it is
            // its id's newest copy (then dead-filtered). Built from a key-only column
            // projection (ranged GETs of id/timestamp/updated_at/version/expires_at,
            // never the vectors), so memory is O(ids) and pass 2 keeps TD-040 bounds
            // pruning even for multi-file collections. Exact on every scoring path,
            // including rows a pruned pass 2 skips, and including a single flushed
            // file holding several copies of an id (flush does not dedup).
            let newest = self
                .build_newest_version_index(&files, collection_id, dimension)
                .await?;

            // Pass 2: score.
            for file_path in files {
                files_scanned += 1;

                if prune_enabled {
                    // `?` propagates a hard error (e.g. the ranged reader vanishing
                    // mid-file after the seed was already scored) rather than
                    // silently double-counting via the full-read fallback.
                    match self
                        .search_file_with_bounds_prune(
                            &file_path,
                            query_vector,
                            ctx,
                            &mut priority_queue,
                            &newest,
                        )
                        .await?
                    {
                        // File fully handled via ranged reads; nothing more to do.
                        Some(pruned) => {
                            bounds_pruned_total += pruned;
                            continue;
                        }
                        // No sidecar / ranged reader unavailable AND nothing scored
                        // yet — fall through to today's full read.
                        None => {}
                    }
                }

                // Full-read fallback (today's behavior): read the whole file.
                let records = self
                    .read_file_records(&file_path, collection_id, dimension)
                    .await?;
                self.score_records_into_queue(
                    records,
                    query_vector,
                    ctx,
                    &mut priority_queue,
                    Some(&newest),
                );
            }
        } else {
            // ONE pass, no extra I/O (co-design: round trips dominate). Every
            // file is read once in full anyway when pruning is off. Each file's
            // rows are observed into the cross-file index and the file keeps
            // its top-k live, newest-in-file candidates with their keys. Only
            // after every file has been seen do candidates pass the GLOBAL
            // newest-copy check. Exact: a dead or superseded copy is never
            // returned. Memory: one file + O(ids) + files x k candidates. A
            // file whose top-k is crowded by copies superseded elsewhere can
            // under-fill k (TD-SEARCH-DEAD-OVERFETCH-1 family).
            let mut global = proximadb_records::NewestVersionIndex::new();
            let mut candidates: Vec<(OptimizedSearchRecord, proximadb_records::VersionKey)> =
                Vec::new();
            for file_path in files {
                files_scanned += 1;
                let records = self
                    .read_file_records(&file_path, collection_id, dimension)
                    .await?;
                let mut local = proximadb_records::NewestVersionIndex::new();
                for record in &records {
                    let key = record.version_key();
                    local.observe(nova_hit_id(record), key);
                    global.observe(nova_hit_id(record), key);
                }
                let now_ns = proximadb_kernel::time::now_unix_ns();
                let mut file_hits: Vec<(OptimizedSearchRecord, proximadb_records::VersionKey)> =
                    records
                        .into_iter()
                        .filter_map(|record| {
                            let key = record.version_key();
                            (local.is_newest(nova_hit_id(&record), &key) && !record.is_dead(now_ns))
                                .then(|| (self.build_hit(record, query_vector, ctx), key))
                        })
                        .collect();
                file_hits.sort_by(|a, b| {
                    b.0.score
                        .partial_cmp(&a.0.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                file_hits.truncate(k);
                candidates.extend(file_hits);
            }
            for (hit, key) in candidates {
                if global.is_newest(&hit.id, &key) {
                    priority_queue.try_insert(hit);
                }
            }
        }

        // TD-040 observability: surface the pruned-row-group count to the
        // per-request diagnostics bus → EXPLAIN `vector_bounds_pruned` hint
        // (no-op outside a request scope, e.g. direct-engine tests).
        if bounds_pruned_total > 0 {
            crate::observability::predicate_diagnostics::record_vector_bounds_pruned(
                bounds_pruned_total,
            );
        }

        // Log search statistics
        if total_files > 1 {
            debug!(
                "📊 NOVA search: scanned {}/{} files",
                files_scanned, total_files
            );
        }

        // Get sorted results from bounded queue
        let results = priority_queue.into_sorted_vec();

        Ok(results)
    }

    /// One NOVA hit: distance + materialization, identical for every scoring
    /// path (full read, ranged read, single- or two-pass).
    fn build_hit(
        &self,
        record: ProximaRecord,
        query_vector: &[f32],
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> OptimizedSearchRecord {
        let vector = record
            .embeddings
            .first()
            .map_or(Vec::new(), |embedding| embedding.values.to_fp32_owned());
        let similarity_result =
            self.distance_engine
                .calculate_distance(query_vector, &vector, &ctx.distance_metric());
        let record_id = nova_hit_id(&record).to_string();

        OptimizedSearchRecord {
            id: record_id.clone(),
            vector_id: Some(record_id),
            score: similarity_result.normalized_score,
            similarity: Some(similarity_result.normalized_score),
            vector: Some(Arc::new(vector)),
            metadata: proxima_tree_to_value_map(&record.props),
            version: Some(record.record_version as u32),
            timestamp: Some(record.created_at_ns / 1_000_000),
            updated_at: Some(record.updated_at_ns / 1_000_000),
            expires_at: record.valid_to_ns.map(|ts| ts / 1_000_000),
            // The canonical TTL (ns): the service dead filter and the query
            // cache's earliest-expiry read this, never `expires_at`.
            valid_to_ns: record.valid_to_ns,
            ..Default::default()
        }
    }

    /// Score a batch of records into the shared top-k queue (the per-record
    /// distance + materialization extracted from `search_standard` so the
    /// full-read and ranged-read paths produce identical entries).
    fn score_records_into_queue(
        &self,
        records: Vec<ProximaRecord>,
        query_vector: &[f32],
        ctx: &crate::storage::traits::StorageQueryContext,
        priority_queue: &mut BoundedPriorityQueue,
        newest: Option<&proximadb_records::NewestVersionIndex>,
    ) {
        // Pick the winner FIRST, then dead-filter it (the shared rule,
        // `proximadb_records::VersionKey`; TD-ENGINE-MVCC-1). A row that is not
        // its id's newest copy across every file scanned is never scored, so an
        // older live copy cannot resurface when the newest copy is a tombstone
        // or TTL-expired, and neither kind can take a top-k slot (16a).
        let now_ns = proximadb_kernel::time::now_unix_ns();
        for record in records {
            if let Some(index) = newest
                && !index.is_newest(nova_hit_id(&record), &record.version_key())
            {
                continue;
            }
            if record.is_dead(now_ns) {
                continue;
            }
            let search_record = self.build_hit(record, query_vector, ctx);
            priority_queue.try_insert(search_record);
        }
    }

    /// TD-040 two-pass, recall-preserving vector-bounds prune for one NOVA file.
    ///
    /// Loads the `{file}.nova_meta` sidecar (per-row-group `ZoneMap`s), orders
    /// row groups by centroid distance, reads the nearest as a **seed** to
    /// establish a provisional top-k threshold `τ = 1/s_k − 1`, then skips any
    /// remaining row group whose Euclidean lower bound exceeds `τ` before
    /// reading the survivors — all via real ranged reads. Returns:
    /// * `Ok(Some(pruned))` — file fully handled (seed + survivors scored).
    /// * `Ok(None)` — no sidecar / ranged reader unavailable AND nothing scored
    ///   yet; the caller must full-read this file.
    /// * `Err(_)` — a hard error after the seed was already scored (caller
    ///   propagates rather than double-counting via the full-read fallback).
    ///
    /// Recall-safety: the Euclidean clamp-gap bound is a true lower bound on the
    /// L2 distance from the query to any vector in a row group's box; `τ` is the
    /// current (shared, global) k-th-best distance, so a pruned group can hold
    /// nothing that would enter the final top-k.
    async fn search_file_with_bounds_prune(
        &self,
        file_path: &str,
        query_vector: &[f32],
        ctx: &crate::storage::traits::StorageQueryContext,
        priority_queue: &mut BoundedPriorityQueue,
        newest: &proximadb_records::NewestVersionIndex,
    ) -> Result<Option<u64>> {
        use crate::storage::engines::nova::nova_ranged_reader::read_selected_row_groups;

        // Per-row-group bounds sidecar — absent ⇒ full-read fallback.
        let mut meta_reader = crate::storage::engines::nova::nova_meta_reader::NovaMetaReader::new(
            self.filesystem.clone(),
        );
        let metadata = match meta_reader.load_metadata(file_path).await {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        let stats = &metadata.row_group_stats;
        if stats.is_empty() {
            return Ok(None);
        }

        // Order row groups by ascending centroid distance (dim-mismatch → tail).
        let mut order: Vec<usize> = (0..stats.len()).collect();
        order.sort_by(|&a, &b| {
            let da = centroid_l2_sq(query_vector, &stats[a].vector_zone_map.centroid);
            let db = centroid_l2_sq(query_vector, &stats[b].vector_zone_map.centroid);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Seed: read the nearest row group. Ranged reader unavailable (and
        // nothing scored yet) ⇒ Ok(None) so the caller can full-read.
        let seed_rg = order[0];
        let seed = match read_selected_row_groups(file_path, &[seed_rg]).await? {
            Some(recs) => recs,
            None => return Ok(None),
        };
        self.score_records_into_queue(seed, query_vector, ctx, priority_queue, Some(newest));

        // Recover the provisional top-k threshold. Euclidean normalized score is
        // s = 1/(1+d) ⇒ d = 1/s − 1. Under-full queue (s_k == -inf) ⇒ no prune.
        let s_k = priority_queue.min_score_threshold();
        let tau = if s_k.is_finite() && s_k > 0.0 {
            Some(1.0 / s_k - 1.0)
        } else {
            None
        };

        // Partition the remainder: prune groups whose box lower bound > τ.
        let mut survivors: Vec<usize> = Vec::with_capacity(order.len().saturating_sub(1));
        let mut pruned: u64 = 0;
        for &rg in &order[1..] {
            let zm = &stats[rg].vector_zone_map;
            let prunable = match tau {
                Some(t) => {
                    // ZoneMap::intersects_query matches the metric by lowercase
                    // string ("euclidean"); only the Euclidean branch is a sound
                    // L2 lower bound (cosine/dot are guarded out upstream).
                    zm.min_values.len() == query_vector.len()
                        && !zm.intersects_query(query_vector, "euclidean".to_string(), t)
                }
                None => false,
            };
            if prunable {
                pruned += 1;
            } else {
                survivors.push(rg);
            }
        }

        // Read survivors via ranged reads (the pruned groups are never fetched).
        // The seed already succeeded, so the ranged reader is available for this
        // file; a `None` here means the file became unreadable mid-search — we
        // surface it as an error rather than full-read (which would double-count
        // the already-scored seed).
        if !survivors.is_empty() {
            let recs = read_selected_row_groups(file_path, &survivors)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!("NOVA ranged reader became unavailable mid-file: {file_path}")
                })?;
            self.score_records_into_queue(recs, query_vector, ctx, priority_queue, Some(newest));
        }

        Ok(Some(pruned))
    }

    /// Determine if progressive search should be used
    fn should_use_progressive_search(
        &self,
        _k: usize,
        collection_size: usize,
        has_filter: bool,
    ) -> bool {
        // Use progressive search for large collections or complex filters
        let is_large_collection = collection_size > 100000;

        has_filter || is_large_collection
    }

    /// Determine if streaming search should be used
    fn should_use_streaming_search(&self, _k: usize, collection_size: usize) -> bool {
        // Use streaming for very large collections
        collection_size > 1000000
    }

    /// Search by vector ID
    pub async fn vector_by_id(
        &self,
        _collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        // Implement ID-based search using bloom filters and hierarchical index
        debug!("🔍 NOVA: Searching for vector ID: {}", vector_id);

        // This would use the hierarchical index and bloom filters for fast ID lookup
        // For now, return a placeholder
        Ok(None)
    }

    /// Read one NOVA file in full (records with vectors).
    async fn read_file_records(
        &self,
        file_path: &str,
        collection_id: &str,
        dimension: usize,
    ) -> Result<Vec<ProximaRecord>> {
        use crate::storage::engines::core::formats::columnar::UnifiedParquetReader;
        use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
        let fs = self.filesystem.get_filesystem(file_path)?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            fs,
            collection_id.to_string(),
            "nova".to_string(),
        ));
        let reader = UnifiedParquetReader::new(
            vec![file_path.to_string()],
            dimension,
            self.filesystem.clone(),
            unified_fs,
            collection_id.to_string(),
            "nova".to_string(),
        )?;
        // `0` is the START OFFSET, not a batch size: this passed 10000, which
        // silently skipped the first 10,000 rows of any larger file.
        reader.read_all_records(0, None).await
    }

    /// Pass 1 of a NOVA search: the newest-copy index over every row of every
    /// file. Key-only projected ranged read per file; a full read only where no
    /// ranged reader is available (that file's records are dropped right after
    /// observing, so memory stays one file plus the index).
    async fn build_newest_version_index(
        &self,
        files: &[String],
        collection_id: &str,
        dimension: usize,
    ) -> Result<proximadb_records::NewestVersionIndex> {
        use crate::storage::engines::nova::nova_ranged_reader::read_version_keys;
        let mut newest = proximadb_records::NewestVersionIndex::new();
        for file_path in files {
            match read_version_keys(file_path).await? {
                Some(keys) => {
                    for (id, key) in keys {
                        newest.observe(&id, key);
                    }
                }
                None => {
                    for record in self
                        .read_file_records(file_path, collection_id, dimension)
                        .await?
                    {
                        newest.observe(nova_hit_id(&record), record.version_key());
                    }
                }
            }
        }
        Ok(newest)
    }
}

/// The id a NOVA hit is reported (and versions are resolved) under: the
/// record's `local_id` when set, else its `oid`.
fn nova_hit_id(record: &ProximaRecord) -> &str {
    record
        .local_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .unwrap_or(record.oid.as_str())
}
