//! TD-PAXDIR-1 D2: the collection-global centroid directory on a multi-segment
//! collection — recall against brute force, physical reads against the
//! per-segment path, and segments the directory does not cover.

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use tempfile::TempDir;

    use crate::core::search::SearchParams;
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, StorageAssignment, StorageEngine,
    };
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::engines::sst::centroid_directory_publisher::publish_centroid_directory;
    use crate::storage::engines::sst::core::SstEngine;
    use crate::storage::engines::sst::segment_format::write_pax_segment_compacted;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::StorageQueryContext;
    use proximadb_block_format::VectorQuant;
    use proximadb_distance_kernel::DistanceMetric;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use proximadb_storage_filesystem_types::counting::global_counters;

    const DIM: usize = 32;
    const SEGMENTS: usize = 4;
    const ROWS_PER_SEGMENT: usize = 1500;
    const CLUSTERS: usize = 16;
    const QUERIES: usize = 30;
    const TOP_K: usize = 10;

    /// Deterministic clustered corpus: `CLUSTERS` Gaussian blobs. With
    /// `segment_major`, each segment holds its own `CLUSTERS / SEGMENTS`
    /// clusters (time-ordered ingest of different topics); otherwise every
    /// segment holds every cluster (the worst case for skipping segments).
    fn corpus_with(segment_major: bool, rows_per_segment: usize) -> Vec<Vec<f32>> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut gauss = move || {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        };
        let centers: Vec<Vec<f32>> = (0..CLUSTERS)
            .map(|_| (0..DIM).map(|_| gauss() * 4.0).collect())
            .collect();
        (0..SEGMENTS * rows_per_segment)
            .map(|i| {
                let per_segment = CLUSTERS / SEGMENTS;
                let cluster = if segment_major {
                    (i / rows_per_segment) * per_segment + i % per_segment
                } else {
                    i % CLUSTERS
                };
                let center = &centers[cluster];
                center.iter().map(|c| c + gauss()).collect()
            })
            .collect()
    }

    fn corpus() -> Vec<Vec<f32>> {
        corpus_with(false, ROWS_PER_SEGMENT)
    }

    fn record(i: usize, vector: &[f32]) -> ProximaRecord {
        let ts = 1_700_000_000_000_000_000 + i as i64;
        let mut r = ProximaRecord {
            oid: format!("r{i}"),
            tenant_id: "t".into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        };
        r.embeddings.push(EmbeddingCell::new_fp32(
            "",
            "dense",
            DIM as u32,
            vector.to_vec(),
        ));
        r
    }

    fn collection(id: &str, temp_dir: &TempDir) -> Collection {
        Collection {
            id: id.to_string(),
            config: Some(CollectionConfig {
                name: id.to_string(),
                dimension: DIM as u32,
                storage_engine: Some(StorageEngine::Sst as i32),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn enable_env() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_COUNT_FS_IO", "1");
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "8");
            std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1");
        }
    }

    fn set_directory_read(on: bool) {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var(
                "PROXIMADB_PAX_READ_CENTROID_DIRECTORY",
                if on { "1" } else { "0" },
            )
        };
    }

    struct Arm {
        _temp: TempDir,
        engine: Arc<SstEngine>,
        collection: Collection,
        storage_url: String,
    }

    /// A collection of `SEGMENTS` trained segments (`rows` split evenly), with
    /// a published directory.
    async fn arm(id: &str, rows: &[Vec<f32>]) -> Arm {
        arm_with(id, rows, false).await
    }

    /// With `shared_model`, segments after the first are written with the
    /// collection model reconstructed from the first one's directory entry
    /// (TD-PAXDIR-1 D3), as a compaction with the shared-model gate does.
    async fn arm_with(id: &str, rows: &[Vec<f32>], shared_model: bool) -> Arm {
        let temp = TempDir::new().unwrap();
        let base = temp.path().to_str().unwrap().to_string();
        let collection = collection(id, &temp);
        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(format!("file://{base}"));
        let factory = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());
        let engine = Arc::new(
            SstEngine::new_with_config(
                SstConfig::default(),
                Arc::clone(&factory),
                Arc::new(UnifiedDistanceCompute::default()),
            )
            .await
            .unwrap(),
        );
        let ctx = StorageQueryContext::new(
            Arc::new(SearchParams::default()),
            Arc::new(collection.clone()),
        );
        let storage_url = ctx.collection_storage_path().unwrap();
        tokio::fs::create_dir_all(&storage_url).await.unwrap();
        let fs = factory.get_filesystem(&storage_url).unwrap();
        let mut collection_model = None;
        for s in 0..SEGMENTS {
            if shared_model && s == 1 {
                publish_centroid_directory(fs.as_ref(), &storage_url)
                    .await
                    .unwrap();
                collection_model = crate::storage::engines::sst::centroid_directory_publisher::
                    load_newest_directory(fs.as_ref(), &storage_url)
                    .await
                    .and_then(|dir| dir.collection_model())
                    .map(Arc::new);
                assert!(collection_model.is_some());
            }
            let _scope = crate::storage::engines::sst::block_cluster::CollectionModelScope::enter(
                collection_model.clone(),
            );
            let per_segment = rows.len() / SEGMENTS;
            let records: Vec<ProximaRecord> = (s * per_segment..(s + 1) * per_segment)
                .map(|i| record(i, &rows[i]))
                .collect();
            write_pax_segment_compacted(
                Path::new(&format!("{storage_url}/seg-{s}.pax")),
                &records,
                id,
                1,
                VectorQuant::RaBitQ,
                VectorQuant::Sq8,
                false,
                Some(16 * 1024),
                None,
            )
            .unwrap();
        }
        assert_eq!(
            publish_centroid_directory(fs.as_ref(), &storage_url)
                .await
                .unwrap(),
            Some(if shared_model { 2 } else { 1 })
        );
        Arm {
            _temp: temp,
            engine,
            collection,
            storage_url,
        }
    }

    async fn search(arm: &Arm, query: &[f32], k: usize) -> Vec<String> {
        let params = Arc::new(SearchParams {
            vector: Some(query.to_vec()),
            top_k: Some(k as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(params, Arc::new(arm.collection.clone()));
        Arc::clone(&arm.engine)
            .fallback_to_direct_search_arc(
                &ctx,
                &arm.collection.id,
                &arm.storage_url,
                query,
                k,
                DistanceMetric::Euclidean,
                None,
                false,
                false,
            )
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect()
    }

    fn exact_top_k(rows: &[Vec<f32>], query: &[f32], k: usize) -> Vec<String> {
        let mut scored: Vec<(f32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored
            .iter()
            .take(k)
            .map(|(_, i)| format!("r{i}"))
            .collect()
    }

    fn queries(rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Perturbed corpus points, spread over every segment and cluster.
        (0..QUERIES)
            .map(|q| {
                rows[(q * 197) % rows.len()]
                    .iter()
                    .enumerate()
                    .map(|(d, v)| v + 0.05 * ((q + d) % 7) as f32)
                    .collect()
            })
            .collect()
    }

    /// Recall and physical reads of one arm over all queries (cold arm: each
    /// arm has its own segment paths, so no cache entry is shared).
    async fn measure(arm: &Arm, rows: &[Vec<f32>], directory: bool) -> (f64, u64) {
        set_directory_read(directory);
        let counters = global_counters();
        let before = counters.total_gets();
        let mut hits = 0usize;
        for query in queries(rows) {
            let truth = exact_top_k(rows, &query, TOP_K);
            let got = search(arm, &query, TOP_K).await;
            hits += got.iter().filter(|id| truth.contains(id)).count();
        }
        let reads = counters.total_gets() - before;
        (hits as f64 / (QUERIES * TOP_K) as f64, reads)
    }

    #[tokio::test]
    async fn global_selection_keeps_recall_and_reads_less() {
        enable_env();
        let rows = corpus();
        let off = arm("paxdir_off", &rows).await;
        let on = arm("paxdir_on", &rows).await;
        let (recall_off, reads_off) = measure(&off, &rows, false).await;
        let (recall_on, reads_on) = measure(&on, &rows, true).await;
        eprintln!(
            "per-segment: recall {recall_off:.3}, reads {reads_off}; \
             directory: recall {recall_on:.3}, reads {reads_on}"
        );
        assert!(recall_off >= 0.9, "baseline recall {recall_off}");
        assert!(
            recall_on >= recall_off - 0.01,
            "directory recall {recall_on} < per-segment {recall_off} - 0.01"
        );
        assert!(
            reads_on < reads_off,
            "directory reads {reads_on} must be fewer than per-segment {reads_off}"
        );
    }

    /// Mean number of segments a query's directory plan probes (segments
    /// owning none of the selected cells are skipped with zero reads).
    fn mean_segments_probed(arm: &Arm, rows: &[Vec<f32>]) -> f64 {
        use crate::storage::engines::sst::centroid_directory_reader::plan_segments;
        use proximadb_storage_common::centroid_directory::{PaxCentroidDirectory, object_name};
        let bytes = std::fs::read(format!("{}/{}", arm.storage_url, object_name(1))).unwrap();
        let dir = Arc::new(PaxCentroidDirectory::parse(&bytes).unwrap());
        let files: Vec<(String, u64)> = std::fs::read_dir(&arm.storage_url)
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".pax"))
            .map(|e| {
                (
                    e.path().to_string_lossy().into_owned(),
                    e.metadata().unwrap().len(),
                )
            })
            .collect();
        let probed: usize = queries(rows)
            .iter()
            .map(|q| {
                let plan = plan_segments(
                    &dir,
                    files.iter().map(|(u, s)| (u.as_str(), Some(*s), None)),
                    q,
                    TOP_K,
                )
                .unwrap();
                assert_eq!(plan.len(), SEGMENTS, "every segment is usable");
                plan.values().filter(|c| !c.cells.is_empty()).count()
            })
            .sum();
        probed as f64 / QUERIES as f64
    }

    /// When segments hold different regions of the space, the directory skips
    /// the segments that own none of the globally nearest cells, at the same
    /// recall. Geometry: 32 cells of ~100 rows per segment, so the default
    /// global count (`ceil(2·sqrt(128)) = 23`) fits inside one segment's cells
    /// (with `k < 4·segments` per segment it necessarily spills). A query's
    /// own cluster spans 8 cells; the next-nearest cells belong to other
    /// clusters, which are as likely to live in another segment as in its own,
    /// so a correct global ranking still touches more than one segment.
    #[tokio::test]
    async fn segments_owning_no_selected_cell_are_skipped() {
        const ROWS: usize = 3200;
        enable_env();
        // SAFETY: nextest runs each test in its own process. Four parallel
        // files: the spawned fan-out must carry each segment's cell scope.
        unsafe {
            std::env::set_var("PROXIMADB_IVF_K", "32");
            std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "4");
        }
        let spread_rows = corpus_with(false, ROWS);
        let spread = arm("paxdir_spread", &spread_rows).await;
        let spread_probed = mean_segments_probed(&spread, &spread_rows);
        let rows = corpus_with(true, ROWS);
        let off = arm("paxdir_major_off", &rows).await;
        let on = arm("paxdir_major_on", &rows).await;
        let major_probed = mean_segments_probed(&on, &rows);
        let (recall_off, reads_off) = measure(&off, &rows, false).await;
        let (recall_on, reads_on) = measure(&on, &rows, true).await;
        eprintln!(
            "segments probed per query: spread {spread_probed:.2}, segment-major \
             {major_probed:.2} of {SEGMENTS}; segment-major per-segment: recall \
             {recall_off:.3}, reads {reads_off}; directory: recall {recall_on:.3}, \
             reads {reads_on}"
        );
        assert!(
            major_probed < spread_probed && major_probed <= 3.0,
            "segment-major layout must skip segments: {major_probed} vs {spread_probed}"
        );
        assert!(
            recall_on >= recall_off - 0.01,
            "{recall_on} vs {recall_off}"
        );
        assert!(reads_on < reads_off, "{reads_on} vs {reads_off}");
    }

    /// Skipping a segment is only sound when every planned segment runs the
    /// A0 coarse probe. Metrics outside the PAX cascade, the coarse-probe
    /// kill-switch and an explicit per-segment probe count all scan more than
    /// the directory's cells, so the directory must stay unused for them.
    #[tokio::test]
    async fn the_directory_is_unused_when_planned_segments_would_not_probe() {
        enable_env();
        let rows = corpus();
        let arm = arm("paxdir_preconditions", &rows).await;
        set_directory_read(true);
        let files = arm
            .engine
            .discover_sstable_files(&arm.storage_url)
            .await
            .unwrap();
        let plan = |metric| {
            let engine = Arc::clone(&arm.engine);
            let files = &files;
            let query = rows[3].clone();
            let url = arm.storage_url.clone();
            async move {
                engine
                    .centroid_directory_cells(&url, files, &query, TOP_K, metric, false)
                    .await
            }
        };
        assert_eq!(plan(DistanceMetric::Euclidean).await.len(), SEGMENTS);
        assert!(plan(DistanceMetric::Manhattan).await.is_empty());
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0") };
        assert!(plan(DistanceMetric::Euclidean).await.is_empty());
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "8");
        }
        assert!(plan(DistanceMetric::Euclidean).await.is_empty());
        unsafe { std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE") };
        assert_eq!(plan(DistanceMetric::Euclidean).await.len(), SEGMENTS);
        // Filtered or exact queries never plan.
        let engine = Arc::clone(&arm.engine);
        assert!(
            engine
                .centroid_directory_cells(
                    &arm.storage_url,
                    &files,
                    &rows[3],
                    TOP_K,
                    DistanceMetric::Euclidean,
                    true,
                )
                .await
                .is_empty()
        );
    }

    /// TD-PAXDIR-1 D3: segments written with the shared collection model form
    /// one directory model, so the global count is over distinct cells and a
    /// selected cell is probed in every segment holding it — at the recall of
    /// the per-segment path.
    #[tokio::test]
    async fn shared_model_segments_plan_on_distinct_cells_at_equal_recall() {
        enable_env();
        let rows = corpus();
        let off = arm("paxdir_shared_off", &rows).await;
        let on = arm_with("paxdir_shared_on", &rows, true).await;
        let bytes = std::fs::read(format!(
            "{}/{}",
            on.storage_url,
            proximadb_storage_common::centroid_directory::object_name(2)
        ))
        .unwrap();
        let dir = Arc::new(
            proximadb_storage_common::centroid_directory::PaxCentroidDirectory::parse(&bytes)
                .unwrap(),
        );
        assert_eq!(dir.models.len(), 1, "every segment speaks one cell space");
        let k = dir.collection_model().unwrap().model.k_c();
        let files: Vec<(String, u64)> = std::fs::read_dir(&on.storage_url)
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".pax"))
            .map(|e| {
                (
                    e.path().to_string_lossy().into_owned(),
                    e.metadata().unwrap().len(),
                )
            })
            .collect();
        let plan = crate::storage::engines::sst::centroid_directory_reader::plan_segments(
            &dir,
            files.iter().map(|(u, s)| (u.as_str(), Some(*s), None)),
            &rows[11],
            TOP_K,
        )
        .unwrap();
        let mut holders: std::collections::HashMap<u32, usize> = Default::default();
        for choice in plan.values() {
            for &cell in choice.cells.iter() {
                *holders.entry(cell).or_default() += 1;
            }
        }
        assert!(
            !holders.is_empty() && holders.len() <= k,
            "distinct cells: {holders:?}"
        );
        assert!(
            holders.values().any(|&segments| segments > 1),
            "a selected cell is probed in every segment holding it: {holders:?}"
        );

        let (recall_off, _) = measure(&off, &rows, false).await;
        let (recall_on, _) = measure(&on, &rows, true).await;
        eprintln!("shared model: recall {recall_on:.3} vs per-segment {recall_off:.3}");
        assert!(
            recall_on >= recall_off - 0.01,
            "{recall_on} vs {recall_off}"
        );
    }

    #[tokio::test]
    async fn segments_the_directory_does_not_cover_are_still_searched() {
        enable_env();
        let rows = corpus();
        let arm = arm("paxdir_stale", &rows).await;
        // A segment written after the directory was published (a new flush):
        // uncovered, so it must take the per-segment path.
        let fresh: Vec<f32> = (0..DIM).map(|d| 100.0 + d as f32).collect();
        write_pax_segment_compacted(
            Path::new(&format!("{}/seg-new.pax", arm.storage_url)),
            &(0..100)
                .map(|i| {
                    let mut v = fresh.clone();
                    v[0] += i as f32;
                    record(1_000_000 + i, &v)
                })
                .collect::<Vec<_>>(),
            "paxdir_stale",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .unwrap();
        set_directory_read(true);
        let got = search(&arm, &fresh, 1).await;
        assert_eq!(got, ["r1000000"]);
        // Covered segments still answer through the directory.
        let query = rows[5].clone();
        assert_eq!(search(&arm, &query, 1).await, ["r5"]);
    }

    #[tokio::test]
    async fn a_corrupt_directory_falls_back_to_the_per_segment_path() {
        enable_env();
        let rows = corpus();
        let arm = arm("paxdir_corrupt", &rows).await;
        let object = format!(
            "{}/{}",
            arm.storage_url,
            proximadb_storage_common::centroid_directory::object_name(1)
        );
        std::fs::write(&object, b"not a directory").unwrap();
        set_directory_read(true);
        let query = rows[42].clone();
        assert_eq!(search(&arm, &query, 1).await, ["r42"]);
    }

    /// The injected-latency bed of the wave tests: 20 ms inside every timed
    /// request, parallel waves, the D5 same-wave rerank, and header-only prefix
    /// reads (at synthetic scale the default 1 MiB prefix would swallow a
    /// whole segment, which a 1M-row segment's never does).
    fn enable_latency_env() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_TEST_FS_LATENCY_MS", "20");
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "16");
            std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "4");
            std::env::set_var("PROXIMADB_PAX_READ_SAME_WAVE_RERANK", "1");
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
        }
    }

    /// Load the arm's newest centroid directory into the reader's cache (its
    /// discovery LIST, then one GET), as a long-running server has: the
    /// directory is fetched once per generation (design §3.5, wave 0).
    async fn preload_directory(arm: &Arm) {
        arm.engine
            .discover_sstable_files(&arm.storage_url)
            .await
            .unwrap();
        let fs = arm
            .engine
            .filesystem()
            .get_filesystem(&arm.storage_url)
            .unwrap();
        assert!(
            crate::storage::engines::sst::centroid_directory_reader::load_directory(
                fs.as_ref(),
                &arm.storage_url,
            )
            .await
            .is_some()
        );
    }

    /// One query (`rows[q]`, top-1) in its own trace scope: asserts it finds
    /// itself and returns the query's production `IoTraceSnapshot`.
    async fn traced_search(
        arm: &Arm,
        rows: &[Vec<f32>],
        q: usize,
    ) -> crate::observability::io_trace::IoTraceSnapshot {
        use crate::observability::io_trace;
        io_trace::scope(async {
            assert_eq!(search(arm, &rows[q], 1).await, [format!("r{q}")]);
            io_trace::snapshot().expect("io_trace scope active")
        })
        .await
    }

    /// A top-1 query context for `rows[q]` under `search_mode`.
    fn search_ctx(
        arm: &Arm,
        rows: &[Vec<f32>],
        q: usize,
        search_mode: crate::core::search::SearchMode,
    ) -> StorageQueryContext {
        let params = Arc::new(SearchParams {
            vector: Some(rows[q].clone()),
            top_k: Some(1),
            distance_metric: Some(DistanceMetric::Euclidean),
            search_mode,
            ..Default::default()
        });
        StorageQueryContext::new(params, Arc::new(arm.collection.clone()))
    }

    /// One cold query's sequential waves and requests under the injected
    /// latency, on a fresh arm (empty segment caches). The directory itself
    /// is loaded first when `directory` is on: it is fetched once per
    /// generation and cached (design §3.5, wave 0).
    async fn cold_query_shape(
        id: &str,
        rows: &[Vec<f32>],
        directory: bool,
    ) -> (usize, usize, String) {
        set_directory_read(directory);
        let arm = arm(id, rows).await;
        if directory {
            preload_directory(&arm).await;
        }
        // TD-PAXDIR-1 D7: waves and requests come from the query's own
        // production trace (the injected latency sits inside each timed
        // request).
        let snap = traced_search(&arm, rows, 7).await;
        let waves = snap.waves.expect("the query issued requests") as usize;
        let requests = snap.read_requests() as usize;
        (waves, requests, format!("{snap:?}"))
    }

    /// A segment's own footer bytes, read through the engine's `FileSystem`
    /// (header prefix, then the footer range) — any backend, no path tricks.
    async fn real_footer(arm: &Arm, url: &str) -> Vec<u8> {
        use proximadb_storage_common::segment_layout::{
            SEG_HEADER_PREFIX_LEN, SegmentHeaderPrefix,
        };
        let fs = arm.engine.filesystem().get_filesystem(url).unwrap();
        let prefix = fs
            .read_range(url, 0, SEG_HEADER_PREFIX_LEN as u64)
            .await
            .unwrap();
        let header = SegmentHeaderPrefix::parse(&prefix).unwrap();
        fs.read_range(url, header.footer_off, header.footer_len)
            .await
            .unwrap()
    }

    /// TD-PAXDIR-1 D6 exit test: a covered segment's cold query reads no
    /// header, A0 or footer, so it is LIST → probed cells (A + B in one wave,
    /// D5) → top-k OID chunks: 3 sequential waves, within the §3.5 request
    /// budget. Without decode keys the header read adds a wave; with the
    /// directory off, the decode-key gate changes nothing.
    #[tokio::test]
    async fn a_covered_segment_cold_query_takes_three_waves() {
        enable_env();
        enable_latency_env();
        let rows = corpus();
        let set_keys = |on: bool| unsafe {
            std::env::set_var(
                "PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS",
                if on { "1" } else { "0" },
            )
        };

        set_keys(true);
        let (keyed_waves, keyed_requests, keyed_trace) =
            cold_query_shape("paxdir_d6_keys", &rows, true).await;
        set_keys(false);
        let (keyless_waves, keyless_requests, _) =
            cold_query_shape("paxdir_d6_keyless", &rows, true).await;
        let (off_waves, off_requests, _) = cold_query_shape("paxdir_d6_off", &rows, false).await;
        set_keys(true);
        let (off_keys_waves, off_keys_requests, _) =
            cold_query_shape("paxdir_d6_off_keys", &rows, false).await;
        eprintln!(
            "cold query waves/requests: decode keys {keyed_waves}/{keyed_requests}, \
             directory without keys {keyless_waves}/{keyless_requests}, directory off \
             {off_waves}/{off_requests} (keys gate on: {off_keys_waves}/{off_keys_requests})"
        );
        assert_eq!(
            keyed_waves, 3,
            "LIST, probed cells, top-k blocks; requests:\n{keyed_trace}"
        );
        assert!(
            keyed_requests <= 20,
            "within the §3.5 budget: {keyed_requests}\n{keyed_trace}"
        );
        // Without keys the header, A0, Region-A header and footer are
        // dependent reads, and rehydration repeats header and footer.
        assert!(
            keyed_waves < keyless_waves,
            "{keyed_waves} vs {keyless_waves}"
        );
        assert!(
            keyed_requests < keyless_requests,
            "{keyed_requests} vs {keyless_requests}"
        );
        assert_eq!(
            (off_keys_waves, off_keys_requests),
            (off_waves, off_requests),
            "the directory-off path ignores the decode-key gate"
        );
    }

    /// Decode keys change only where metadata comes from: on the same
    /// directory, results keep the directory-off recall, and a warm repeat
    /// reads less than without keys (no segment metadata is read at all, and
    /// the top-k blocks are read once). Segments the cold prefix read fetches
    /// whole keep that read — it serves every later byte — so at the default
    /// 1 MiB prefix these small test segments read exactly as without keys.
    #[tokio::test]
    async fn decode_keys_keep_recall_and_never_poison_the_metadata_cache() {
        enable_env();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_SAME_WAVE_RERANK", "1") };
        let set_keys = |on: bool| unsafe {
            std::env::set_var(
                "PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS",
                if on { "1" } else { "0" },
            )
        };
        let set_prefix = |bytes: &str| unsafe {
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", bytes)
        };
        let rows = corpus();

        // Small segments (the prefix read fetches each whole): keys unused.
        set_prefix("1048576");
        set_keys(false);
        let small = arm("paxdir_d6_small_keyless", &rows).await;
        let (_, small_keyless) = measure(&small, &rows, true).await;
        set_keys(true);
        let small = arm("paxdir_d6_small_keys", &rows).await;
        let (_, small_keys) = measure(&small, &rows, true).await;
        assert_eq!(
            small_keys, small_keyless,
            "a whole-segment prefix read wins"
        );

        // Large segments (header-only prefix reads).
        set_prefix("0");
        let off = arm("paxdir_d6_recall_off", &rows).await;
        let keyless = arm("paxdir_d6_recall_keyless", &rows).await;
        let on = arm("paxdir_d6_recall_on", &rows).await;
        let (recall_off, _) = measure(&off, &rows, false).await;
        set_keys(false);
        let (recall_keyless, reads_keyless) = measure(&keyless, &rows, true).await;
        set_keys(true);
        let (recall_on, reads_on) = measure(&on, &rows, true).await;
        eprintln!(
            "decode keys: recall {recall_on:.3} reads {reads_on}; without keys: recall \
             {recall_keyless:.3} reads {reads_keyless}; per-segment recall {recall_off:.3}; \
             small segments: {small_keys} reads either way"
        );
        assert_eq!(recall_on, recall_keyless, "keys change no result");
        assert!(
            recall_on >= recall_off - 0.01,
            "{recall_on} vs {recall_off}"
        );
        assert!(reads_on < reads_keyless, "{reads_on} vs {reads_keyless}");

        // The rebuilt footer is ANN-only; every later query must read the
        // segment's own, so directory-planned scans never admit the rebuilt
        // bytes to the invariants cache. The keyless arm proves the cache is
        // keyed by these URLs; the decoded arm then holds no entry, and a
        // per-segment query afterwards caches each segment's real footer.
        for (arm, decoded) in [(&keyless, false), (&on, true)] {
            let cache = arm.engine.segment_invariants_cache.as_ref().unwrap();
            let files = arm
                .engine
                .discover_sstable_files(&arm.storage_url)
                .await
                .unwrap();
            assert_eq!(files.len(), SEGMENTS);
            for file in &files {
                match cache.get(&file.url) {
                    Some(entry) => {
                        assert!(!decoded, "{} cached from a decoded scan", file.url);
                        assert_eq!(entry.footer_bytes, real_footer(arm, &file.url).await);
                    }
                    None => assert!(decoded, "{} never cached by its own scan", file.url),
                }
            }
        }
        set_directory_read(false);
        let query = rows[3].clone();
        assert_eq!(search(&on, &query, 1).await, ["r3"]);
        let cache = on.engine.segment_invariants_cache.as_ref().unwrap();
        for file in on
            .engine
            .discover_sstable_files(&on.storage_url)
            .await
            .unwrap()
        {
            let entry = cache.get(&file.url).expect("per-segment scan caches");
            assert_eq!(entry.footer_bytes, real_footer(&on, &file.url).await);
        }
    }

    /// TD-PAXDIR-1 D7: the default (Adaptive) search counts vectors from a
    /// listing and the scan lists again; the gate run measured that as two
    /// sequential LIST round trips per query. One search lists once.
    #[tokio::test]
    async fn an_adaptive_search_lists_the_collection_once() {
        use crate::observability::io_trace;
        enable_env();
        let rows = corpus();
        let arm = arm("paxdir_d7_one_list", &rows).await;
        set_directory_read(false);
        let ctx = search_ctx(
            &arm,
            &rows,
            9,
            crate::core::search::SearchMode::Adaptive { threshold: 1 },
        );
        let (ids, snap) = io_trace::scope(async {
            let ids: Vec<String> = Arc::clone(&arm.engine)
                .search_vectors_unified_arc(&ctx)
                .await
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();
            (ids, io_trace::snapshot().unwrap())
        })
        .await;
        assert_eq!(ids, ["r9"]);
        assert_eq!(snap.list_ops, 1, "count and scan share one listing");
    }

    /// TD-PAXDIR-1 D7: once a segment's metadata is cached (a warm process —
    /// the gate run measured a 0.999 invariants hit ratio in its "cold"
    /// phase), a directory-planned query still takes LIST → cells → top-k
    /// blocks. The block hand-off used to fire only on decoded scans, so a
    /// warm query paid the OID-chunk wave and then the rehydration wave.
    #[tokio::test]
    async fn a_warm_directory_planned_query_also_takes_three_waves() {
        use crate::observability::io_trace;
        enable_env();
        enable_latency_env();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS", "1") };
        let rows = corpus();
        let arm = arm("paxdir_d7_warm", &rows).await;
        // Warm the invariants cache with each segment's REAL metadata (a
        // per-segment query caches it; a decoded scan never does), then load
        // the directory, as a long-running server has.
        set_directory_read(false);
        for q in [3usize, 1500, 3000, 4500] {
            assert_eq!(search(&arm, &rows[q], 1).await, [format!("r{q}")]);
        }
        set_directory_read(true);
        preload_directory(&arm).await;
        let warm = traced_search(&arm, &rows, 1234).await;
        assert_eq!(
            warm.waves,
            Some(3),
            "warm: LIST, probed cells, top-k blocks ({warm:?})"
        );
    }

    /// TD-PAXDIR-1 D7: the server's search reaches the StorageEngine trait's
    /// `search_vectors_unified` and its non-Arc fan-out, where the P5b gate
    /// run found the directory dark. Through that entry a covered cold query
    /// (directory loaded, decode keys on) takes 3 waves, as on the Arc path.
    #[tokio::test]
    async fn the_server_search_path_plans_on_the_directory() {
        use crate::observability::io_trace;
        use proximadb_storage_traits::UnifiedStorageFormat;
        enable_env();
        enable_latency_env();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS", "1") };
        set_directory_read(true);
        let rows = corpus();
        let arm = arm("paxdir_d7_server_path", &rows).await;
        preload_directory(&arm).await;
        let ctx = search_ctx(
            &arm,
            &rows,
            7,
            crate::core::search::SearchMode::Approximate { nprobe: None },
        );
        let (ids, snap) = io_trace::scope(async {
            let ids: Vec<String> = UnifiedStorageFormat::search_vectors_unified(&*arm.engine, &ctx)
                .await
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();
            (ids, io_trace::snapshot().unwrap())
        })
        .await;
        assert_eq!(ids, ["r7"]);
        assert_eq!(
            snap.waves,
            Some(3),
            "server path: LIST, probed cells, top-k blocks ({snap:?})"
        );
    }
}
