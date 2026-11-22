//! ADR-042 (amended 2026-11-20) / TD-PAXMETA-1: a FILTERED vector query on a
//! directory-covered collection prunes segments and then blocks from the
//! collection segment directory (PXCD v3 per-block prune statistics) alone —
//! no header or footer read — and fetches only the surviving block extents.
//!
//! Every physical read is recorded at the backend (below the factory's
//! decorators), so "no footer read" is checked against each segment's real
//! footer extent, and the kill-switch arm is the positive control proving the
//! recorder sees footer reads when they happen.

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use std::collections::{BTreeSet, HashMap};
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use tempfile::TempDir;

    use crate::core::search::SearchParams;
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, StorageAssignment, StorageEngine,
    };
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::engines::sst::centroid_directory_publisher::publish_centroid_directory;
    use crate::storage::engines::sst::core::SstEngine;
    use crate::storage::engines::sst::segment_format::write_pax_segment_compacted_shredded;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::StorageQueryContext;
    use proximadb_block_format::{PaxBlockReader, ShredColumn, VectorQuant, record::FlatRow};
    use proximadb_data_model::{ProximaType, ProximaValue as PV};
    use proximadb_distance_kernel::DistanceMetric;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use proximadb_filter_expression::{ComparisonOperator as Op, FilterExpression};
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};
    use proximadb_storage_filesystem_types::{
        DirEntry, FileOptions, FileSystem, FilesystemFile, FsFileMetadata, FsResult,
    };
    use serde_json::json;

    const DIM: usize = 16;
    const SEGMENTS: usize = 4;
    const ROWS: usize = 600;
    const BANDS: usize = 6;
    const TOP_K: usize = 10;
    const DISABLE: &str = "PROXIMADB_PAX_READ_BLOCK_PRUNE_STATS_DISABLE";

    // ---- a backend that records every physical read ---------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Read {
        Whole(String),
        Range(String, u64, u64),
    }

    #[derive(Debug)]
    struct RecordingFs {
        inner: Arc<dyn FileSystem>,
        reads: Arc<Mutex<Vec<Read>>>,
    }

    impl RecordingFs {
        fn log(&self, read: Read) {
            self.reads
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(read);
        }
    }

    proximadb_storage_filesystem_types::forward_filesystem_impl! {
        impl FileSystem for RecordingFs => inner;
        forward [
            filesystem_type, write, supports_bounded_local_file_write, write_local_file,
            write_if_absent, append, supports_append, supports_conditional_replace, delete,
            exists, metadata, list, create_dir, create_dir_all, copy, move_file, sync,
            sync_file, sync_data, open_file,
        ];
        {
            // Not the inner's: no caller may downcast past the recorder.
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
                self.log(Read::Whole(path.to_string()));
                self.inner.read(path).await
            }
            async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
                self.log(Read::Range(path.to_string(), offset, length));
                self.inner.read_range(path, offset, length).await
            }
            fn range_coalesce_policy(
                &self,
            ) -> Option<proximadb_storage_filesystem_types::RangeCoalescePolicy> {
                self.inner.range_coalesce_policy()
            }
        }
    }

    // Silence unused-import lints for types only the macro expansion names.
    #[allow(dead_code)]
    fn _types(
        _: Option<(
            DirEntry,
            FileOptions,
            FsFileMetadata,
            Box<dyn FilesystemFile>,
        )>,
    ) {
    }

    // ---- corpus ---------------------------------------------------------

    fn gauss_stream(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            let mut uniform = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        }
    }

    fn str_prop(s: &str) -> ProximaTreeNode {
        ProximaTreeNode::Value(PV::String(s.to_string()))
    }

    /// Segment `s`: `bucket = s` everywhere; `band` 0..6 correlated with the
    /// vector cluster (so the writer's clustering keeps blocks band-pure);
    /// `color` by band; `note` absent on odd rows, null every 10th row, a
    /// long string otherwise.
    fn segment_records(s: usize, rows: usize) -> Vec<ProximaRecord> {
        let mut gauss = gauss_stream(0x9E37_79B9 ^ (s as u64 * 7919));
        let centers: Vec<Vec<f32>> = (0..BANDS)
            .map(|_| (0..DIM).map(|_| gauss() * 6.0).collect())
            .collect();
        (0..rows)
            .map(|i| {
                let band = i % BANDS;
                let vector: Vec<f32> = centers[band].iter().map(|c| c + gauss()).collect();
                let ts = 1_700_000_000_000_000_000 + (s * 10_000 + i) as i64;
                let mut r = ProximaRecord {
                    oid: format!("s{s}r{i}"),
                    tenant_id: "t".into(),
                    created_at_ns: ts,
                    updated_at_ns: ts,
                    ..Default::default()
                };
                r.embeddings
                    .push(EmbeddingCell::new_fp32("", "dense", DIM as u32, vector));
                r.props
                    .insert("bucket".into(), ProximaTreeNode::Value(PV::Int64(s as i64)));
                r.props.insert(
                    "band".into(),
                    ProximaTreeNode::Value(PV::Int64(band as i64)),
                );
                r.props
                    .insert("color".into(), str_prop(["red", "green", "blue"][band % 3]));
                if i % 10 == 0 {
                    r.props
                        .insert("note".into(), ProximaTreeNode::Value(PV::Null));
                } else if i % 2 == 0 {
                    r.props.insert(
                        "note".into(),
                        str_prop(&format!("a-long-note-value-beyond-sixteen-bytes-{band}")),
                    );
                }
                r
            })
            .collect()
    }

    fn shred_spec() -> Vec<ShredColumn> {
        vec![
            ShredColumn::typed("bucket", 100, ProximaType::Int64),
            ShredColumn::typed("band", 101, ProximaType::Int64),
            ShredColumn::typed("color", 102, ProximaType::String),
            ShredColumn::typed("note", 103, ProximaType::String),
        ]
    }

    fn write_segment(path: &Path, records: &[ProximaRecord]) {
        write_pax_segment_compacted_shredded(
            path,
            records,
            "dbp",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(4 * 1024),
            &shred_spec(),
            None,
            Some(false),
        )
        .unwrap();
    }

    fn enable_env(trained: bool) {
        // SAFETY: nextest runs each test in its own process; `#[serial]`
        // covers the shared `cargo test` process.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var(
                "PROXIMADB_PAX_WRITE_A0_TRAIN",
                if trained { "1" } else { "0" },
            );
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
            std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1");
            // Header-only cold prefix reads: a segment is then larger than the
            // prefix read, so its metadata really comes from the directory
            // (D6 rule) and header/footer reads are distinguishable.
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
            // No gap coalescing: a fetch covers exactly the planned blocks.
            std::env::set_var("PROXIMADB_PAX_COALESCE_GAP", "0");
            std::env::remove_var("PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS");
            std::env::remove_var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            std::env::remove_var(DISABLE);
        }
    }

    fn set_pruning(on: bool) {
        // SAFETY: as in `enable_env`.
        unsafe {
            if on {
                std::env::remove_var(DISABLE);
            } else {
                std::env::set_var(DISABLE, "1");
            }
        }
    }

    struct Arm {
        _temp: TempDir,
        engine: Arc<SstEngine>,
        collection: Collection,
        storage_url: String,
        reads: Arc<Mutex<Vec<Read>>>,
        counters: Arc<proximadb_storage_filesystem_types::counting::GetCounters>,
    }

    impl Arm {
        fn segment_url(&self, s: usize) -> String {
            format!("{}/seg-{s}.pax", self.storage_url)
        }
        fn take_reads(&self) -> Vec<Read> {
            std::mem::take(
                &mut *self
                    .reads
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        }
    }

    async fn arm(id: &str) -> Arm {
        arm_with(id, true).await
    }

    /// With `write_segments`, the arm holds `SEGMENTS` shredded segments and
    /// a published directory; without, an empty collection directory (the
    /// flush test ingests through `do_flush`).
    async fn arm_with(id: &str, write_segments: bool) -> Arm {
        let temp = TempDir::new().unwrap();
        let base = temp.path().to_str().unwrap().to_string();
        let collection = Collection {
            id: id.to_string(),
            config: Some(CollectionConfig {
                name: id.to_string(),
                dimension: DIM as u32,
                storage_engine: Some(StorageEngine::Sst as i32),
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                // The flush path shreds declared filterable columns only for
                // ProximaRecord collections (`resolve_flush_shred_spec`).
                enable_proxima_record: Some(true),
                filterable_columns: vec![
                    crate::proto::proximadb_v1::FilterableColumnSpec {
                        name: "bucket".into(),
                        data_type:
                            crate::proto::proximadb_v1::FilterableDataType::FilterableInteger
                                as i32,
                        ..Default::default()
                    },
                    crate::proto::proximadb_v1::FilterableColumnSpec {
                        name: "color".into(),
                        data_type:
                            crate::proto::proximadb_v1::FilterableDataType::FilterableString
                                as i32,
                        ..Default::default()
                    },
                ],
                tags: vec!["compaction:off".to_string()],
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: base.clone(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(format!("file://{base}"));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let counters =
            Arc::new(proximadb_storage_filesystem_types::counting::GetCounters::default());
        let mut factory = FilesystemFactory::create(fs_config).await.unwrap();
        let log = Arc::clone(&reads);
        factory.wrap_test_backend("file", move |inner| {
            Arc::new(RecordingFs { inner, reads: log })
        });
        factory.set_test_get_counters(Arc::clone(&counters));
        let factory = Arc::new(factory);
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
        if write_segments {
            for s in 0..SEGMENTS {
                write_segment(
                    Path::new(&format!("{storage_url}/seg-{s}.pax")),
                    &segment_records(s, ROWS),
                );
            }
            let fs = factory.get_filesystem(&storage_url).unwrap();
            assert!(
                publish_centroid_directory(fs.as_ref(), &storage_url)
                    .await
                    .unwrap()
                    .is_some()
            );
        }
        let arm = Arm {
            _temp: temp,
            engine,
            collection,
            storage_url,
            reads,
            counters,
        };
        arm.take_reads();
        arm
    }

    fn query(seed: u64) -> Vec<f32> {
        let mut gauss = gauss_stream(seed);
        (0..DIM).map(|_| gauss() * 4.0).collect()
    }

    /// The production fan-out (`fallback_to_direct_search`) with a filter;
    /// `(id, similarity)` in rank order.
    async fn search(arm: &Arm, q: &[f32], filter: &FilterExpression) -> Vec<(String, u32)> {
        let params = Arc::new(SearchParams {
            vector: Some(q.to_vec()),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(params, Arc::new(arm.collection.clone()));
        arm.engine
            .fallback_to_direct_search(
                &ctx,
                &arm.collection.id,
                &arm.storage_url,
                q,
                TOP_K,
                DistanceMetric::Euclidean,
                Some(filter),
                false,
                true,
            )
            .await
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.similarity.unwrap_or(f32::NAN).to_bits()))
            .collect()
    }

    fn cmp(field: &str, op: Op, value: serde_json::Value) -> FilterExpression {
        FilterExpression::Comparison {
            field: field.into(),
            operator: op,
            value,
        }
    }

    /// One segment's on-disk layout and the truth about which of its blocks
    /// hold a row matching `filter` (decoded independently of the stats).
    struct Layout {
        footer_extent: (u64, u64),
        blocks: Vec<(u64, u64)>,
        matching: Vec<bool>,
        stats_keep: Vec<bool>,
    }

    fn layout(path: &str, filter: &FilterExpression) -> Layout {
        let bytes = std::fs::read(path).unwrap();
        let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
        let foff = header.footer_off as usize;
        let footer =
            SegmentFooterIndex::parse(&bytes[foff..foff + header.footer_len as usize]).unwrap();
        let stats = footer
            .block_prune_stats
            .as_ref()
            .expect("the writer emits prune stats for shredded segments");
        let mut blocks = Vec::new();
        let mut matching = Vec::new();
        let mut stats_keep = Vec::new();
        for (b, block) in footer.blocks.iter().enumerate() {
            let start = block.offset;
            let end = block.offset + u64::from(block.size);
            let reader = PaxBlockReader::open(&bytes[start as usize..end as usize]).unwrap();
            let any = FlatRow::from_block_reader(&reader)
                .unwrap()
                .into_iter()
                .any(|row| {
                    crate::core::search::sql_value_filter::evaluate_filter_proxima(
                        filter,
                        &row.props_tree().unwrap(),
                    )
                });
            blocks.push((start, end));
            matching.push(any);
            stats_keep.push(stats.block_may_match(b, filter));
        }
        Layout {
            footer_extent: (header.footer_off, header.footer_off + header.footer_len),
            blocks,
            matching,
            stats_keep,
        }
    }

    fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
        a.0 < b.1 && b.0 < a.1
    }

    fn reads_of<'a>(reads: &'a [Read], url: &'a str) -> impl Iterator<Item = &'a Read> + 'a {
        reads.iter().filter(move |r| match r {
            // The engine addresses objects as `file://<path>`.
            Read::Whole(p) | Read::Range(p, ..) => {
                p.strip_prefix("file://").unwrap_or(p) == url.strip_prefix("file://").unwrap_or(url)
            }
        })
    }

    /// Per-segment classification of a query's recorded reads.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct SegmentReads {
        total: usize,
        header: usize,
        footer: usize,
        whole: usize,
        pruned_block_bytes: usize,
    }

    fn classify(reads: &[Read], url: &str, layout: &Layout) -> SegmentReads {
        let mut out = SegmentReads::default();
        for read in reads_of(reads, url) {
            out.total += 1;
            match read {
                Read::Whole(_) => out.whole += 1,
                Read::Range(_, off, len) => {
                    let extent = (*off, off + len);
                    if *off == 0 {
                        out.header += 1;
                    }
                    if overlaps(extent, layout.footer_extent) {
                        out.footer += 1;
                    }
                    for (b, block) in layout.blocks.iter().enumerate() {
                        if !layout.stats_keep[b] && overlaps(extent, *block) {
                            out.pruned_block_bytes += 1;
                        }
                    }
                }
            }
        }
        out
    }

    /// (a) + (d): a filtered query on a covered collection reads no header
    /// and no footer of any segment, never touches a segment no block of
    /// which can match, and fetches no block the stats pruned; with the kill
    /// switch the same query reads every segment's footer (the positive
    /// control) and returns the identical answer.
    async fn footerless_filtered_query(trained: bool) {
        enable_env(trained);
        // bucket == 2 (one segment) AND band IN (1, 4) (some of its blocks).
        let filter = FilterExpression::And(vec![
            cmp("bucket", Op::Equals, json!(2)),
            cmp("band", Op::In, json!([1, 4])),
        ]);
        let on = arm(if trained { "dbp-on-t" } else { "dbp-on-u" }).await;
        let off = arm(if trained { "dbp-off-t" } else { "dbp-off-u" }).await;
        let layouts: Vec<Layout> = (0..SEGMENTS)
            .map(|s| layout(&on.segment_url(s), &filter))
            .collect();
        // The stats are sound and non-vacuous on this data.
        for (s, l) in layouts.iter().enumerate() {
            for b in 0..l.blocks.len() {
                assert!(
                    !l.matching[b] || l.stats_keep[b],
                    "segment {s} block {b} holds a match but the stats prune it"
                );
            }
        }
        let kept: usize = layouts[2].stats_keep.iter().filter(|k| **k).count();
        assert!(
            kept > 0 && kept < layouts[2].blocks.len(),
            "the filter must prune some (not all) blocks of the surviving segment: kept {kept} of {}",
            layouts[2].blocks.len()
        );

        let q = query(17);
        set_pruning(true);
        let before = on.counters.total_gets();
        let got_on = search(&on, &q, &filter).await;
        let gets_on = on.counters.total_gets() - before;
        let reads_on = on.take_reads();
        set_pruning(false);
        let before = off.counters.total_gets();
        let got_off = search(&off, &q, &filter).await;
        let gets_off = off.counters.total_gets() - before;
        let reads_off = off.take_reads();
        set_pruning(true);

        assert!(!got_on.is_empty());
        assert_eq!(got_on, got_off, "pruning must not change the answer");
        for (id, _) in &got_on {
            assert!(id.starts_with("s2r"), "hit {id} outside bucket 2");
        }
        let dir_reads = reads_on
            .iter()
            .filter(|r| matches!(r, Read::Whole(p) if p.ends_with(".pxcd")))
            .count();
        assert_eq!(dir_reads, 1, "one directory GET: {reads_on:?}");
        for s in 0..SEGMENTS {
            let c = classify(&reads_on, &on.segment_url(s), &layouts[s]);
            assert_eq!(
                c.header, 0,
                "segment {s}: header read with the directory: {c:?}"
            );
            assert_eq!(
                c.footer, 0,
                "segment {s}: footer read with the directory: {c:?}"
            );
            assert_eq!(c.whole, 0, "segment {s}: whole-object read: {c:?}");
            assert_eq!(
                c.pruned_block_bytes, 0,
                "segment {s}: a pruned block was fetched: {c:?}"
            );
            if s != 2 {
                assert_eq!(c.total, 0, "pruned segment {s} was read: {c:?}");
            }
        }
        let mut footers_off = 0;
        for s in 0..SEGMENTS {
            let off_layout = layout(&off.segment_url(s), &filter);
            let c = classify(&reads_off, &off.segment_url(s), &off_layout);
            assert!(
                c.footer >= 1,
                "kill switch: segment {s} footer must be read: {c:?}"
            );
            footers_off += c.footer;
        }
        eprintln!(
            "{}",
            json!({
                "adr042_filtered_query_counts": {
                    "trained": trained,
                    "segments": SEGMENTS,
                    "on": {"gets": gets_on, "footer_reads": 0, "segments_read": 1},
                    "off": {"gets": gets_off, "footer_reads": footers_off, "segments_read": SEGMENTS},
                }
            })
        );
        assert!(gets_on < gets_off, "on {gets_on} vs off {gets_off}");
    }

    #[tokio::test]
    #[serial]
    async fn filtered_query_reads_no_footer_on_a_trained_collection() {
        footerless_filtered_query(true).await;
    }

    #[tokio::test]
    #[serial]
    async fn filtered_query_reads_no_footer_on_an_untrained_collection() {
        footerless_filtered_query(false).await;
    }

    /// (c): a segment rewritten in place after the publish (its listed size
    /// no longer matches the entry) takes the footer path; the others stay
    /// footer-less, and the answer reflects the rewritten bytes.
    #[tokio::test]
    #[serial]
    async fn a_stale_entry_falls_back_to_the_footer_path() {
        enable_env(true);
        let arm = arm("dbp-stale").await;
        // Rewrite seg-1 in place with fewer rows (another size), every row
        // now in bucket 2 so the filter must find it.
        let mut records = segment_records(1, ROWS / 2);
        for r in &mut records {
            r.props
                .insert("bucket".into(), ProximaTreeNode::Value(PV::Int64(2)));
        }
        write_segment(Path::new(&arm.segment_url(1)), &records);
        let filter = cmp("bucket", Op::Equals, json!(2));
        // Query AT a rewritten row: it must come back (distance 0).
        let q = records[3].embeddings[0].as_fp32_cow().to_vec();
        let got = search(&arm, &q, &filter).await;
        let reads = arm.take_reads();
        let stale = classify(
            &reads,
            &arm.segment_url(1),
            &layout(&arm.segment_url(1), &filter),
        );
        assert!(
            stale.footer >= 1,
            "stale entry must read its footer: {stale:?}"
        );
        for s in [0, 2, 3] {
            let c = classify(
                &reads,
                &arm.segment_url(s),
                &layout(&arm.segment_url(s), &filter),
            );
            assert_eq!(c.footer, 0, "segment {s}: {c:?}");
            if s != 2 {
                assert_eq!(c.total, 0, "segment {s} prunes from the directory: {c:?}");
            }
        }
        // The rewritten rows are found (they now match), and every hit matches.
        assert_eq!(got.first().map(|(id, _)| id.as_str()), Some("s1r3"), "{got:?}");
        assert!(
            got.iter()
                .all(|(id, _)| id.starts_with("s1r") || id.starts_with("s2r"))
        );
        set_pruning(false);
        let off = search(&arm, &q, &filter).await;
        set_pruning(true);
        assert_eq!(got, off);
    }

    /// (b) + (e): over randomized filters — equality, IN, ranges, null tests,
    /// strings (including the long `note` values whose upper bound is not
    /// stored), conjunctions, disjunctions and negations — the answer with
    /// pruning on equals the answer with the kill switch set, and every hit
    /// satisfies the filter. Two arms over identical bytes, so the pruned arm
    /// stays on the directory path for every query.
    #[tokio::test]
    #[serial]
    async fn randomized_filters_answer_identically_with_pruning_on_and_off() {
        enable_env(true);
        let on = arm("dbp-rand-on").await;
        let off = arm("dbp-rand-off").await;
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let note_values = [
            json!("a-long-note-value-beyond-sixteen-bytes-2"),
            json!("a-long-note-value-beyond-sixteen-bytes-9"),
            json!("a-long-note"),
            json!("zzz"),
            json!(null),
        ];
        let mut leaf = |next: &mut dyn FnMut(usize) -> usize| -> FilterExpression {
            match next(10) {
                0 => cmp("bucket", Op::Equals, json!(next(5))),
                1 => cmp("band", Op::In, json!([next(7), next(7)])),
                2 => cmp("band", Op::GreaterThan, json!(next(7))),
                3 => cmp("band", Op::LessThanOrEqual, json!(next(7) as f64 - 0.5)),
                4 => cmp(
                    "color",
                    Op::Equals,
                    json!(["red", "green", "blue", "pink"][next(4)]),
                ),
                5 => cmp("note", Op::IsNull, json!(null)),
                6 => cmp(
                    "note",
                    Op::Equals,
                    note_values[next(note_values.len())].clone(),
                ),
                7 => cmp("note", Op::GreaterThan, note_values[next(4)].clone()),
                8 => cmp("bucket", Op::Between, json!([next(4), next(4)])),
                _ => cmp("color", Op::NotEquals, json!("red")),
            }
        };
        let mut pruned_segments = 0u64;
        let mut seen_ids: BTreeSet<String> = BTreeSet::new();
        for round in 0..24 {
            let filter = match next(4) {
                0 => leaf(&mut next),
                1 => FilterExpression::And(vec![leaf(&mut next), leaf(&mut next)]),
                2 => FilterExpression::Or(vec![leaf(&mut next), leaf(&mut next)]),
                _ => FilterExpression::Not(Box::new(leaf(&mut next))),
            };
            let q = query(1000 + round);
            set_pruning(true);
            let trace = Arc::new(crate::observability::io_trace::IoTrace::new());
            let got_on = crate::observability::io_trace::scope_with_handle(
                Arc::clone(&trace),
                search(&on, &q, &filter),
            )
            .await;
            pruned_segments += trace.snapshot().pax_directory_segments_pruned;
            set_pruning(false);
            let got_off = search(&off, &q, &filter).await;
            set_pruning(true);
            assert_eq!(got_on, got_off, "round {round}: {filter:?}");
            seen_ids.extend(got_on.iter().map(|(id, _)| id.clone()));
        }
        on.take_reads();
        assert!(
            pruned_segments > 0,
            "the randomized filters must prune segments"
        );
        assert!(!seen_ids.is_empty());
    }

    /// Every hit of a filtered query satisfies the filter (the stats never
    /// admit a non-matching row: they only skip, the evaluator decides).
    #[tokio::test]
    #[serial]
    async fn hits_satisfy_the_filter_and_match_brute_force_membership() {
        enable_env(true);
        let arm = arm("dbp-hits").await;
        let filter = FilterExpression::Or(vec![
            cmp("color", Op::Equals, json!("blue")),
            cmp("note", Op::IsNull, json!(null)),
        ]);
        let props: HashMap<String, proximadb_records::ProximaTree> = (0..SEGMENTS)
            .flat_map(|s| segment_records(s, ROWS))
            .map(|r| (r.oid.clone(), r.props.clone()))
            .collect();
        for seed in 0..6 {
            let got = search(&arm, &query(300 + seed), &filter).await;
            assert_eq!(got.len(), TOP_K);
            for (id, _) in &got {
                assert!(
                    crate::core::search::sql_value_filter::evaluate_filter_proxima(
                        &filter, &props[id]
                    ),
                    "hit {id} does not satisfy the filter"
                );
            }
        }
    }

    /// The production write path: segments flushed through `do_flush` for a
    /// collection with declared filterable columns carry the prune stats in
    /// their footers, the flush-time directory publish copies them into v3
    /// entries, and a filtered query then skips the segments no block of
    /// which can match — with the answer the kill switch gives.
    #[tokio::test]
    #[serial]
    async fn flushed_segments_publish_prune_stats_and_filtered_queries_skip_segments() {
        use crate::storage::traits::{FlushParameters, UnifiedStorageEngine};
        enable_env(false);
        let arm = arm_with("dbp-flush", false).await;
        const FLUSHES: usize = 3;
        for s in 0..FLUSHES {
            let records: Vec<crate::proto::proximadb_v1::VectorRecord> =
                segment_records(s, 200)
                    .into_iter()
                    .map(|r| {
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "bucket".to_string(),
                            crate::proto::proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::Int64Value(
                                        s as i64,
                                    ),
                                ),
                            },
                        );
                        if let Some(ProximaTreeNode::Value(PV::String(color))) =
                            r.props.get("color")
                        {
                            metadata.insert(
                                "color".to_string(),
                                crate::proto::proximadb_v1::SqlValue {
                                    value: Some(
                                        crate::proto::proximadb_v1::sql_value::Value::StringValue(
                                            color.clone(),
                                        ),
                                    ),
                                },
                            );
                        }
                        crate::proto::proximadb_v1::VectorRecord {
                            id: r.oid.clone(),
                            vector: r.embeddings[0].as_fp32_cow().to_vec(),
                            metadata,
                            timestamp: Some(1),
                            version: Some(1),
                            ..Default::default()
                        }
                    })
                    .collect();
            let params = FlushParameters {
                collection_id: Some(arm.collection.id.clone()),
                vector_records: records.into_iter().map(Into::into).collect(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(arm.collection.clone()),
                estimated_size: 0,
                collection_properties: None,
            };
            let result = arm.engine.do_flush(&params).await.expect("flush");
            assert!(result.success);
        }
        // The newest published directory: every flushed segment has a v3 key
        // whose prune stats summarize its blocks.
        let fs = arm.engine.filesystem().get_filesystem(&arm.storage_url).unwrap();
        let dir = crate::storage::engines::sst::centroid_directory_publisher::load_newest_directory(
            fs.as_ref(),
            &arm.storage_url,
        )
        .await
        .expect("flush-time publish");
        let pax: Vec<_> = dir.segments.iter().filter(|s| s.name.ends_with(".pax")).collect();
        assert_eq!(pax.len(), FLUSHES, "{:?}", dir.segments);
        for entry in &pax {
            let key = entry.decode_key.as_ref().expect("keyed entry");
            let facts = key.filter_facts.as_ref().expect("v3 facts");
            let stats = facts.prune_stats.as_ref().expect("prune stats from the footer");
            assert_eq!(stats.block_count(), key.blocks.len());
            assert!(stats.columns.iter().any(|c| c.name == "bucket"));
        }
        arm.take_reads();

        let filter = cmp("bucket", Op::Equals, json!(1));
        let q = query(77);
        let trace = Arc::new(crate::observability::io_trace::IoTrace::new());
        let got = crate::observability::io_trace::scope_with_handle(
            Arc::clone(&trace),
            search(&arm, &q, &filter),
        )
        .await;
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.pax_directory_segments_planned, FLUSHES as u64);
        assert_eq!(
            snapshot.pax_directory_segments_pruned,
            (FLUSHES - 1) as u64,
            "bucket 0 and 2 segments are skipped"
        );
        assert!(!got.is_empty());
        assert!(got.iter().all(|(id, _)| id.starts_with("s1r")), "{got:?}");
        set_pruning(false);
        let off = search(&arm, &q, &filter).await;
        set_pruning(true);
        assert_eq!(got, off);
    }
}
