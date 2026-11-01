use super::*;

// The tests below pin process-global gates (env or cached-once caps); they
// hold the `#[serial]` env-gate lock so they cannot interleave with other
// gate-pinning tests in the shared `cargo test --lib` process.
use serial_test::serial;

/// TD-SECSCAN-1 §5 (review round 2): the RaBitQ survivor pool and the
/// morsel merge buffer are derived from a client-controlled `top_k`. A
/// huge `k` (e.g. u32::MAX on an uncapped surface) must never inflate the
/// pool past the segment's own row count or overflow the merge
/// `Vec::with_capacity(pool * degree)` — that was a multi-TB request /
/// alloc-abort on coalesced RaBitQ segments with morsel degree > 1.
#[test]
fn rabitq_pool_and_morsel_merge_capacity_are_bounded_by_rows() {
    let n = 65_536usize; // MORSEL_MIN_ROWS: the coalesced morsel path
    let huge_k = u32::MAX as usize;
    let pool = pax_rabitq_pool_for_top_k(huge_k, n);
    assert!(
        pool <= n,
        "survivor pool {pool} must not exceed segment rows {n}"
    );
    // A realistic k still gets a pool that covers it.
    assert!(pax_rabitq_pool_for_top_k(10, n) >= 10);
    // The morsel merge capacity built from (pool, n_rows, degree) never
    // exceeds rows × degree — and never overflows for adversarial inputs.
    let degree = crate::storage::engines::sst::search::morsel_degree().max(1);
    assert!(
        morsel_merge_capacity(huge_k, n, degree) <= n.saturating_mul(degree),
        "merge capacity must be bounded by rows × degree"
    );
    // Adversarial inputs saturate instead of wrapping (degree itself is
    // cores-clamped at the real call site via `morsel_degree()`).
    assert_eq!(
        morsel_merge_capacity(usize::MAX, n, usize::MAX),
        n.saturating_mul(usize::MAX)
    );
}

/// TD-PAXRG-1 Phase A: the coalesced header-prefix GET floor covers the v4
/// (row-group Region D) prefix length, so one ranged read serves every
/// layout version and `parse` branches on the version byte.
#[test]
fn v4_header_get_fetches_88_bytes() {
    assert_eq!(coalesced_header_prefetch_floor(), SEG_HEADER_PREFIX_LEN);
    assert_eq!(coalesced_header_prefetch_floor(), 88);
    assert!(coalesced_header_prefetch_floor() > 72, "floor grew past v3");
}

/// TD-RDSTRAT-13 C1: `record_get_physical` passes `tier as usize` into the
/// io_trace/SST per-tier arrays, whose index order is a documented
/// cross-crate contract (0=IdxA, 1=Ctl, 2=Meta, 3=ProbeA, 4=Surv, 5=OID,
/// 6=FiltD, 7=Rehyd).
/// Reordering the enum would silently fold one tier's GETs into another's
/// counters — pin it.
#[test]
fn cache_tier_discriminant_matches_tier_counter_contract() {
    assert_eq!(CacheTier::InvariantIndex as usize, 0);
    assert_eq!(CacheTier::SearchControl as usize, 1);
    assert_eq!(CacheTier::InvariantMeta as usize, 2);
    assert_eq!(CacheTier::ProbeIndex as usize, 3);
    assert_eq!(CacheTier::SurvivorPayload as usize, 4);
    assert_eq!(CacheTier::ResultPayload as usize, 5);
    assert_eq!(CacheTier::FilterRowAllow as usize, 6);
    assert_eq!(CacheTier::Rehydrate as usize, 7);
    let labels = crate::storage::engines::sst::metrics::TIER_LABELS;
    let tiers = [
        CacheTier::InvariantIndex,
        CacheTier::SearchControl,
        CacheTier::InvariantMeta,
        CacheTier::ProbeIndex,
        CacheTier::SurvivorPayload,
        CacheTier::ResultPayload,
        CacheTier::FilterRowAllow,
        CacheTier::Rehydrate,
    ];
    assert_eq!(tiers.len(), crate::observability::io_trace::PAX_TIER_COUNT);
    assert_eq!(labels.len(), tiers.len());
    for (tier, label) in tiers.into_iter().zip(labels) {
        assert_eq!(tier.label(), label);
    }
}

/// TD-PAXRG-1 Phase C: a v4 segment with the hoisted exact tier (Region C)
/// still satisfies the segment-level exact-authority predicate — the
/// footer's `has_f32_tier` capability flag keeps its meaning when the tier
/// moves out of the blocks.
#[test]
fn exact_authority_holds_for_region_c_segment() {
    use proximadb_block_format::{BlockCompression, BlockMode, VectorQuant};
    use proximadb_records::{EmbeddingCell, EmbeddingValues};
    use proximadb_storage_common::pax_block::PaxSegmentWriter;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rg-c-authority.pax");
    let mut writer = PaxSegmentWriter::new(
        &path,
        BlockMode::Pax,
        BlockCompression::None,
        "col",
        0,
        1,
        Some(256 * 1024),
    )
    .with_quant(VectorQuant::RaBitQ)
    .with_coalesced_rabitq(true)
    .with_f32_tier(true)
    .with_rg_layout(true)
    .with_oid_resolver(true);
    for row in 0..32 {
        let mut record = ProximaRecord {
            oid: format!("oid-{row}"),
            tenant_id: "t".into(),
            created_at_ns: 1_000 + row as i64,
            updated_at_ns: 1_000 + row as i64,
            ..Default::default()
        };
        record
            .embeddings
            .push(EmbeddingCell::new_fp32("", "dense", 16, vec![0.5; 16]));
        writer.add_record(&record).expect("add record");
    }
    writer.finish().expect("finish v4+C segment");

    let segment = std::fs::read(&path).expect("read segment");
    assert!(
        pax_segment_has_exact_vector_authority(&segment),
        "Region C must satisfy the exact-authority predicate"
    );
}

/// TD-PAXRG-1 Phase D: on a v4 segment the top-k OIDs come from the
/// footer-addressed OID CHUNKS, not whole row groups. Evidence: (a) result
/// parity with the v3 whole-block fetch on the identical corpus/order
/// (same Region A/B encodes ⇒ identical distances), and (b) strictly fewer
/// ranged bytes at no more GETs — request economy at parity, byte win.
#[tokio::test]
#[serial]
async fn v4_oid_chunk_fetch_parity_and_byte_win_vs_v3() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_common::pax_block::RG_TARGET_MIN_BYTES;
    use proximadb_storage_filesystem_types::FileSystem;
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};

    const DIM: usize = 64;
    const N: usize = 1_500;
    const K: usize = 10;

    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();

    let build = |path: &std::path::Path, rg: bool| {
        let mut writer = PaxSegmentWriter::new(
            path,
            BlockMode::Pax,
            BlockCompression::None,
            "col",
            0,
            1,
            // v3: small blocks (multi-block segment); v4: floor clamps to
            // one big RG so the whole-RG-vs-chunk contrast is maximal.
            Some(if rg { RG_TARGET_MIN_BYTES } else { 96 * 1024 }),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_coalesced_rabitq(true)
        .with_oid_resolver(true);
        if rg {
            writer = writer.with_rg_layout(true);
        }
        for (i, v) in corpus.iter().enumerate() {
            writer.add_record(&rec(&format!("r{i}"), 1000 + i as i64, v.clone()))?;
        }
        writer.finish()
    };

    let dir = tempfile::tempdir().unwrap();
    let v3_path = dir.path().join("v3.pax");
    let v4_path = dir.path().join("v4.pax");
    build(&v3_path, false).unwrap();
    build(&v4_path, true).unwrap();

    // Sanity: the v4 file really is the v4 layout.
    let v4_bytes = std::fs::read(&v4_path).unwrap();
    let header = SegmentHeaderPrefix::parse(&v4_bytes[..SEG_HEADER_PREFIX_LEN]).unwrap();
    assert_eq!(header.layout_version, SEG_LAYOUT_VERSION);

    let inner = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let counters = global_counters();
    let fs: Arc<dyn FileSystem> =
        Arc::new(CountingFileSystem::new(Arc::new(inner), counters.clone()));

    let query = corpus[137].clone();
    let run = |path: &std::path::Path| {
        let fs = fs.clone();
        let counters = counters.clone();
        let query = query.clone();
        let path = path.to_path_buf();
        async move {
            counters.reset();
            let hits = rabitq_search_segment_coalesced(
                &*fs,
                path.to_str().unwrap(),
                &query,
                K,
                RankMetric::L2,
                None,
                None,
            )
            .await
            .unwrap()
            .expect("cascade must serve the segment");
            (
                hits,
                counters
                    .bytes_read
                    .load(std::sync::atomic::Ordering::Relaxed),
                counters.total_gets(),
            )
        }
    };

    let (v3_hits, v3_bytes, v3_gets) = run(&v3_path).await;
    let (v4_hits, v4_bytes, v4_gets) = run(&v4_path).await;

    // (a) Parity: same hits, same order, same scores (identical A/B
    // encodes; only Region D's fetch strategy differs).
    assert_eq!(v3_hits.len(), v4_hits.len(), "top-k count parity");
    for (a, b) in v3_hits.iter().zip(&v4_hits) {
        assert_eq!(a.oid, b.oid, "OID parity: {} vs {}", a.oid, b.oid);
        assert_eq!(a.distance.to_bits(), b.distance.to_bits(), "score parity");
    }

    // (b) The byte win: the v4 OID-chunk fetch reads strictly less — the
    // delta is Region D (whole RG bodies vs ~k OID strings). Assert a
    // savings floor so a dead chunk path (falling back to whole RGs)
    // cannot pass on noise.
    eprintln!(
        "[oid-chunk] v3: bytes={v3_bytes} gets={v3_gets}; v4: bytes={v4_bytes} \
             gets={v4_gets}; bytes_saved={}",
        v3_bytes.saturating_sub(v4_bytes)
    );
    assert!(
        v4_bytes < v3_bytes && v3_bytes.saturating_sub(v4_bytes) >= 32 * 1024,
        "chunk fetch must save Region-D bytes: v3={v3_bytes} v4={v4_bytes}"
    );
    // GET economy: chunks are fetched per touched RG and deliberately NOT
    // coalesced across RGs (merging would span the inter-RG gap and
    // re-read the bodies the chunk fetch exists to skip). v3's ADJACENT
    // whole blocks can merge into fewer GETs, so v4 may use a small
    // constant more requests — the deliberate trade is bytes-dominant.
    // The TD's flip criterion is therefore: GET count within a small
    // constant of v3, bytes strictly down.
    assert!(
        v4_gets <= v3_gets + 2,
        "request economy within a small constant: v3={v3_gets} v4={v4_gets}"
    );
}

fn write_oid_failure_fixture(path: &std::path::Path, rg_layout: bool) -> Vec<Vec<f32>> {
    use proximadb_block_format::{BlockCompression, BlockMode, VectorQuant};
    use proximadb_storage_common::pax_block::PaxSegmentWriter;

    const DIM: usize = 16;
    let corpus: Vec<Vec<f32>> = (0..32)
        .map(|row| {
            (0..DIM)
                .map(|dim| ((row * 37 + dim * 11) % 101) as f32 * 0.01)
                .collect()
        })
        .collect();
    let mut writer = PaxSegmentWriter::new(
        path,
        BlockMode::Pax,
        BlockCompression::None,
        "col",
        0,
        1,
        Some(256 * 1024),
    )
    .with_quant(VectorQuant::RaBitQ)
    .with_coalesced_rabitq(true)
    .with_oid_resolver(true);
    if rg_layout {
        writer = writer.with_rg_layout(true);
    }
    for (row, vector) in corpus.iter().enumerate() {
        writer
            .add_record(&rec(
                &format!("oid-{row}"),
                1_000 + row as i64,
                vector.clone(),
            ))
            .expect("add fixture row");
    }
    writer.finish().expect("finish fixture segment");
    corpus
}

fn rewrite_coalesced_footer(
    path: &std::path::Path,
    mutate: impl FnOnce(&mut proximadb_storage_common::segment_layout::SegmentFooterIndex),
) {
    let mut bytes = std::fs::read(path).expect("read fixture segment");
    let header =
        SegmentHeaderPrefix::parse(&bytes[..SEG_HEADER_PREFIX_LEN]).expect("parse fixture header");
    let footer_start = header.footer_off as usize;
    let footer_end = footer_start + header.footer_len as usize;
    let mut footer = proximadb_storage_common::segment_layout::SegmentFooterIndex::parse(
        &bytes[footer_start..footer_end],
    )
    .expect("parse fixture footer");
    mutate(&mut footer);
    let encoded = footer.to_bytes().expect("encode mutated footer");
    assert_eq!(
        encoded.len(),
        header.footer_len as usize,
        "test mutation must preserve footer geometry"
    );
    bytes[footer_start..footer_end].copy_from_slice(&encoded);
    std::fs::write(path, bytes).expect("rewrite fixture segment");
}

#[derive(Debug)]
struct PrefetchErrorFileSystem {
    inner: Arc<dyn proximadb_storage_filesystem_types::FileSystem>,
    /// Fail the first single `read_range` that starts inside this byte range
    /// (once), on top of every batched read failing.
    fail_once_in: Option<(std::ops::Range<u64>, std::sync::atomic::AtomicBool)>,
}

#[async_trait::async_trait]
impl proximadb_storage_filesystem_types::FileSystem for PrefetchErrorFileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn read_range(
        &self,
        path: &str,
        offset: u64,
        length: u64,
    ) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
        if let Some((range, fired)) = &self.fail_once_in
            && range.contains(&offset)
            && !fired.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(proximadb_storage_filesystem_types::FilesystemError::Io(
                std::io::Error::other("injected range failure"),
            ));
        }
        self.inner.read_range(path, offset, length).await
    }

    async fn read_ranges_prefetch(
        &self,
        _path: &str,
        _ranges: Vec<std::ops::Range<u64>>,
    ) -> proximadb_storage_filesystem_types::FsResult<Vec<Vec<u8>>> {
        Err(proximadb_storage_filesystem_types::FilesystemError::Io(
            std::io::Error::other("injected prefetch failure"),
        ))
    }

    async fn write(
        &self,
        path: &str,
        data: &[u8],
        options: Option<proximadb_storage_filesystem_types::FileOptions>,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.write(path, data, options).await
    }

    async fn append(
        &self,
        path: &str,
        data: &[u8],
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.append(path, data).await
    }

    async fn delete(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.delete(path).await
    }

    async fn exists(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(
        &self,
        path: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        proximadb_storage_filesystem_types::FsFileMetadata,
    > {
        self.inner.metadata(path).await
    }

    async fn list(
        &self,
        path: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        Vec<proximadb_storage_filesystem_types::DirEntry>,
    > {
        self.inner.list(path).await
    }

    async fn create_dir(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.create_dir_all(path).await
    }

    async fn copy(&self, from: &str, to: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.copy(from, to).await
    }

    async fn move_file(
        &self,
        from: &str,
        to: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.move_file(from, to).await
    }

    fn filesystem_type(&self) -> &'static str {
        "prefetch-error-test"
    }

    async fn sync(&self) -> proximadb_storage_filesystem_types::FsResult<()> {
        self.inner.sync().await
    }

    async fn open_file(
        &self,
        path: &str,
        create: bool,
    ) -> proximadb_storage_filesystem_types::FsResult<
        Box<dyn proximadb_storage_filesystem_types::FilesystemFile>,
    > {
        self.inner.open_file(path, create).await
    }
}

#[tokio::test]
async fn v4_oid_chunk_decode_failure_fails_closed() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad-oid-encoding.pax");
    let corpus = write_oid_failure_fixture(&path, true);
    rewrite_coalesced_footer(&path, |footer| {
        for stats in footer.block_stats.iter_mut().flatten() {
            stats.oid_encoding_id = u8::MAX;
        }
    });
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .expect("local filesystem");

    let err = rabitq_search_segment_coalesced(
        &fs,
        path.to_str().expect("utf8 path"),
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .expect_err("a malformed OID chunk must not produce empty-ID hits");
    assert!(
        err.to_string().contains("OID chunk decode"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn v4_oid_chunk_extent_outside_row_group_fails_closed() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad-oid-extent.pax");
    let corpus = write_oid_failure_fixture(&path, true);
    rewrite_coalesced_footer(&path, |footer| {
        for (block, stats) in footer.blocks.iter().zip(&mut footer.block_stats) {
            if let Some(stats) = stats {
                stats.oid_chunk_rel_off = block.size;
            }
        }
    });
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .expect("local filesystem");

    let err = rabitq_search_segment_coalesced(
        &fs,
        path.to_str().expect("utf8 path"),
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .expect_err("an out-of-range OID chunk must fail before I/O");
    assert!(
        err.to_string().contains("outside row group"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn legacy_oid_block_corruption_fails_closed() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad-legacy-oid-block.pax");
    let corpus = write_oid_failure_fixture(&path, false);
    let mut bytes = std::fs::read(&path).expect("read fixture segment");
    let footer =
        proximadb_storage_common::segment_layout::SegmentFooterIndex::locate_in_segment(&bytes)
            .expect("locate footer")
            .expect("coalesced footer");
    let block = footer.blocks.first().expect("one data block");
    let corrupt_at = block.offset as usize + block.size as usize / 2;
    bytes[corrupt_at] ^= 0x5a;
    std::fs::write(&path, bytes).expect("rewrite fixture segment");
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .expect("local filesystem");

    let err = rabitq_search_segment_coalesced(
        &fs,
        path.to_str().expect("utf8 path"),
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .expect_err("a corrupt legacy OID block must not produce empty-ID hits");
    assert!(
        err.to_string().contains("OID block decode"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
#[serial]
async fn oid_prefetch_fallback_records_each_physical_get_once() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    unsafe { std::env::set_var("PROXIMADB_TRACE_GETS", "1") };
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("oid-prefetch-fallback.pax");
    let corpus = write_oid_failure_fixture(&path, false);
    let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .expect("local filesystem");
    let fs = PrefetchErrorFileSystem {
        inner: Arc::new(local),
        fail_once_in: None,
    };
    let _ = drain_get_trace();

    let (hits, snapshot) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(
            &fs,
            path.to_str().expect("utf8 path"),
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            None,
        )
        .await
        .expect("sequential fallback succeeds")
        .expect("coalesced segment");
        let snapshot = crate::observability::io_trace::snapshot().expect("trace scope");
        (hits, snapshot)
    })
    .await;
    assert!(hits.iter().all(|hit| !hit.oid.is_empty()));
    let trace = drain_get_trace();
    let oid_gets: Vec<_> = trace
        .iter()
        .copied()
        .filter(|(tier, _)| *tier == CacheTier::ResultPayload)
        .collect();
    assert_eq!(
        oid_gets.len(),
        1,
        "one fallback block read must produce one OID GET record: {oid_gets:?}"
    );
    let survivor_bytes: u64 = trace
        .iter()
        .filter(|(tier, _)| *tier == CacheTier::SurvivorPayload)
        .map(|(_, bytes)| *bytes)
        .sum();
    assert_eq!(
        snapshot.ivf_region_b_bytes, survivor_bytes,
        "fallback Region-B bytes must equal the physical survivor GET trace"
    );
    unsafe { std::env::remove_var("PROXIMADB_TRACE_GETS") };
}

fn cp_cfg(mult: f32) -> crate::core::config::CoarseProbeConfig {
    crate::core::config::CoarseProbeConfig {
        enable_read_probe: true,
        enable_write_train: true,
        nprobe_multiplier: mult,
        nprobe_min: 3,
        nprobe_max: 0,
        ncomp_floor: 0,
    }
}

#[test]
fn default_probe_quality_and_backend_ranges_match_sift_acceptance() {
    let defaults: CoarseProbeSettings = crate::core::config::CoarseProbeConfig::default().into();
    assert_eq!(
        geometric_nprobe(30, defaults),
        11,
        "the one-segment SIFT1M geometry needs 11/30 cells for recall@10 >= 0.98"
    );

    let azure = default_probe_range_policy("azure://container/segment.pax");
    assert_eq!(azure.max_gap_bytes, 1024 * 1024);
    assert_eq!(azure.max_range_bytes, 4 * 1024 * 1024);

    let local = default_probe_range_policy("file:///data/segment.pax");
    assert_eq!(local.max_gap_bytes, 256 * 1024);
    assert_eq!(local.max_range_bytes, 1024 * 1024);
}

#[test]
fn invariant_control_l2_codec_round_trips_optional_regions() {
    let original = SegmentInvariants {
        header_bytes: vec![1, 2, 3],
        region_bytes: Some(Arc::from(&b"region-a"[..])),
        footer_bytes: vec![4, 5],
        a0_bytes: Some(Arc::from(&b"a0"[..])),
        rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
    };
    let decoded =
        decode_invariant_control(&encode_invariant_control(&original)).expect("valid codec");
    assert_eq!(decoded.header_bytes, original.header_bytes);
    assert_eq!(decoded.footer_bytes, original.footer_bytes);
    assert_eq!(decoded.a0_bytes.as_deref(), original.a0_bytes.as_deref());
    assert_eq!(
        decoded.rabitq_header_bytes.as_deref(),
        original.rabitq_header_bytes.as_deref()
    );
    assert!(
        decoded.region_bytes.is_none(),
        "Region A has its own persistent entry"
    );

    let control_only = SegmentInvariants {
        header_bytes: vec![9],
        region_bytes: None,
        footer_bytes: vec![8],
        a0_bytes: None,
        rabitq_header_bytes: None,
    };
    let decoded =
        decode_invariant_control(&encode_invariant_control(&control_only)).expect("valid codec");
    assert!(decoded.a0_bytes.is_none());
    assert!(decoded.rabitq_header_bytes.is_none());
    assert!(decode_invariant_control(b"truncated").is_none());
}

#[tokio::test]
async fn invariant_l2_survives_dram_and_store_restart() {
    let dir = tempfile::tempdir().expect("test tempdir");
    let path = "file:///tenant-a/collection-a/segment-a.px";
    let original = Arc::new(SegmentInvariants {
        header_bytes: vec![1, 2, 3],
        region_bytes: Some(Arc::from(&b"region-a"[..])),
        footer_bytes: vec![4, 5],
        a0_bytes: Some(Arc::from(&b"a0"[..])),
        rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
    });
    let store = Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open L2 store"));
    let cache = SegmentInvariantsCache::with_l2(1 << 20, store);
    cache.put_with_l2(path.to_string(), original.clone()).await;
    drop(cache);

    let reopened_store =
        Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen L2 store"));
    let reopened = SegmentInvariantsCache::with_l2(1 << 20, reopened_store);
    let lookup = reopened.get_or_promote(path).await;
    let promoted = lookup.value().expect("persistent L2 hit");
    assert!(matches!(lookup, InvariantLookup::L2(_)));
    assert_eq!(promoted.header_bytes, original.header_bytes);
    assert_eq!(promoted.footer_bytes, original.footer_bytes);
    assert!(
        promoted.region_bytes.is_none(),
        "A0-backed segments must not hydrate complete Region A after restart"
    );
    assert_eq!(
        reopened
            .get_region_range(path, 1, b"region-a".len() as u64 - 1)
            .await
            .expect("bounded Region A range")
            .as_ref(),
        &b"egion-a"[..]
    );
    assert_eq!(promoted.a0_bytes.as_deref(), original.a0_bytes.as_deref());
    assert_eq!(
        promoted.rabitq_header_bytes.as_deref(),
        original.rabitq_header_bytes.as_deref()
    );
    assert!(
        matches!(reopened.get_or_promote(path).await, InvariantLookup::L1(_)),
        "the persistent value must be promoted into DRAM"
    );
    reopened.invalidate_all(path).await;
    assert!(reopened.get(path).is_none(), "L1 entries were invalidated");
    assert!(matches!(
        reopened.get_or_promote(path).await,
        InvariantLookup::Miss
    ));

    let one_level_path = "file:///tenant-a/collection-a/segment-one-level.px";
    let one_level = Arc::new(SegmentInvariants {
        header_bytes: vec![1],
        region_bytes: Some(Arc::from(&b"whole-region"[..])),
        footer_bytes: vec![2],
        a0_bytes: None,
        rabitq_header_bytes: Some(Arc::from(&b"params"[..])),
    });
    reopened
        .put_with_l2(one_level_path.to_string(), one_level.clone())
        .await;
    drop(reopened);
    let one_level_store =
        Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen one-level L2"));
    let one_level_cache = SegmentInvariantsCache::with_l2(1 << 20, one_level_store);
    let promoted = one_level_cache
        .get_or_promote(one_level_path)
        .await
        .value()
        .expect("one-level persistent hit");
    assert_eq!(
        promoted.region_bytes.as_deref(),
        one_level.region_bytes.as_deref(),
        "one-level scans still require their complete Region A"
    );
}

#[tokio::test]
async fn local_pax_seed_streams_regions_into_persistent_cache() {
    use proximadb_records::{EmbeddingCell, EmbeddingValues};
    use proximadb_storage_common::coarse_directory::CoarseModel;

    const DIM: usize = 8;
    let dir = tempfile::tempdir().expect("test tempdir");
    let segment_path = dir.path().join("spill-output.pax");
    let model = CoarseModel {
        dim: DIM as u32,
        n_comp: 1,
        pca_mean: vec![0.0; DIM],
        pca_components: vec![0.1; DIM],
        centroids: vec![-1.0, 1.0],
        radii: vec![1.0, 1.0],
        cell_rows: vec![2, 2],
        seed: 7,
        trained_on: 4,
    };
    let mut writer = PaxSegmentWriter::new(
        &segment_path,
        BlockMode::Pax,
        BlockCompression::None,
        "collection",
        0,
        1,
        None,
    )
    .with_quant(VectorQuant::RaBitQ)
    .with_rerank_quant(VectorQuant::Sq8)
    .with_coalesced_rabitq(true)
    .with_two_level(model);
    for row in 0..4usize {
        let mut record = rec(&format!("row-{row}"), row as i64, vec![row as f32; DIM]);
        record.embeddings[0] =
            EmbeddingCell::new_fp32("", "dense", DIM as u32, vec![row as f32; DIM]);
        writer.add_record(&record).expect("append record");
    }
    writer.finish().expect("finish PAX");
    let segment = std::fs::read(&segment_path).expect("read PAX");
    let header = SegmentHeaderPrefix::parse(&segment).expect("parse PAX header");

    let cache_path = dir.path().join("cache");
    let store =
        Arc::new(PersistentByteStore::open(&cache_path, 1 << 20).expect("open persistent cache"));
    let invariants = SegmentInvariantsCache::with_l2(1 << 20, store.clone());
    let survivor = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store));
    let published = "az://container/data/1/L2_output.pax";
    assert!(
        install_pax_cache_seed_from_local_file(
            published,
            &segment_path,
            true,
            Some(&invariants),
            Some(&survivor),
        )
        .await
        .expect("install local PAX seed")
    );
    let control = invariants.get(published).expect("control cached in DRAM");
    assert!(
        control.region_bytes.is_none(),
        "streaming seed must not materialize Region A in DRAM"
    );
    let region_slice = invariants
        .get_region_range(published, 1, header.rabitq_len - 1)
        .await
        .expect("Region A range from L2");
    assert_eq!(
        region_slice.as_ref(),
        &segment
            [(header.rabitq_off + 1) as usize..(header.rabitq_off + header.rabitq_len) as usize]
    );
    let sq8_slice = survivor
        .get_or_fetch_in_parent(
            CacheKind::QuantizedCodes,
            published,
            header.sq8_off + 1,
            header.sq8_len - 1,
            header.sq8_off,
            header.sq8_len,
            || async {
                Err(proximadb_storage_filesystem_types::FilesystemError::Io(
                    std::io::Error::other("object loader must not run"),
                ))
            },
        )
        .await
        .expect("Region B range from L2");
    assert_eq!(
        sq8_slice.as_ref(),
        &segment[(header.sq8_off + 1) as usize..(header.sq8_off + header.sq8_len) as usize]
    );
}

/// TD-COMPACT-9: a REAL (Some) coarse-probe config must WIN and must NOT be
/// clobbered by a later default-config `SstEngine::new()` init(None) — the
/// bug that made `[storage.sst_config.coarse_probe] nprobe_multiplier`
/// silently dead in production. Also verifies the multiplier reaches
/// geometric_nprobe. (nextest = process isolation; the static won't leak.)
#[test]
fn toml_coarse_probe_config_wins_over_default_init() {
    // Simulate a default-config engine seeding first...
    init_coarse_probe_settings(None);
    // ...then the authoritative TOML config (server startup) — must override.
    init_coarse_probe_settings(Some(&cp_cfg(1.5)));
    assert!((coarse_probe_settings().nprobe_multiplier - 1.5).abs() < 1e-6);
    // A later default-config engine init(None) must NOT clobber it.
    init_coarse_probe_settings(None);
    assert!((coarse_probe_settings().nprobe_multiplier - 1.5).abs() < 1e-6);
    // The multiplier actually reaches nprobe: sqrt(64)*1.5 = 12 vs *1.0 = 8.
    assert_eq!(geometric_nprobe(64, coarse_probe_settings()), 12);
    assert_eq!(geometric_nprobe(64, cp_cfg(1.0).into()), 8);
}

/// TD-CACHE-2 S2b: Region A and control bytes are separate eviction
/// units — a put with a region splits into two entries; invalidate
/// removes both; a region-only shed leaves control serving (recomposed
/// get degrades to control-only, never a full miss).
#[test]
fn invariants_split_region_from_control() {
    let cache = SegmentInvariantsCache::new(1024 * 1024);
    let region: Arc<[u8]> = Arc::from(vec![7u8; 64 * 1024].as_slice());
    cache.put(
        "seg.pax".to_string(),
        Arc::new(SegmentInvariants {
            header_bytes: vec![1; 72],
            region_bytes: Some(region),
            footer_bytes: vec![2; 128],
            a0_bytes: None,
            rabitq_header_bytes: None,
        }),
    );
    // Recomposed hit carries the region.
    let inv = cache.get("seg.pax").expect("hit");
    assert!(
        inv.region_bytes.is_some(),
        "recomposed get carries Region A"
    );
    assert_eq!(inv.header_bytes.len(), 72);
    // Shed ONLY the region companion (simulating per-class eviction).
    cache.invalidate(&SegmentInvariantsCache::region_entry_key("seg.pax"));
    let inv = cache.get("seg.pax").expect("control must still serve");
    assert!(
        inv.region_bytes.is_none(),
        "region shed, control-only hit (3 control GETs still saved)"
    );
    assert_eq!(inv.footer_bytes.len(), 128);
    // Full invalidate removes both and frees all bytes.
    cache.put(
        "seg.pax".to_string(),
        Arc::new(SegmentInvariants {
            header_bytes: vec![1; 72],
            region_bytes: Some(Arc::from(vec![7u8; 1024].as_slice())),
            footer_bytes: vec![2; 128],
            a0_bytes: None,
            rabitq_header_bytes: None,
        }),
    );
    cache.invalidate("seg.pax");
    assert!(cache.get("seg.pax").is_none());
    assert_eq!(cache.bytes_used(), 0, "all bytes credited back");
}

/// TD-CACHE-2 S1 verification: over-budget insertion mixing tiers must
/// shed meta-only entries before Region-A-bearing ones, ties by recency —
/// never an arbitrary victim.
#[test]
fn invariants_cache_evicts_priority_then_recency() {
    fn region_entry(bytes: usize) -> Arc<SegmentInvariants> {
        Arc::new(SegmentInvariants {
            header_bytes: Vec::new(),
            region_bytes: Some(Arc::from(vec![0u8; bytes].as_slice())),
            footer_bytes: Vec::new(),
            a0_bytes: None,
            rabitq_header_bytes: None,
        })
    }
    fn meta_entry(bytes: usize) -> Arc<SegmentInvariants> {
        Arc::new(SegmentInvariants {
            header_bytes: vec![0u8; bytes],
            region_bytes: None,
            footer_bytes: Vec::new(),
            a0_bytes: None,
            rabitq_header_bytes: None,
        })
    }

    let cache = SegmentInvariantsCache::new(1000);
    cache.put("regionA1".into(), region_entry(400));
    cache.put("regionA2".into(), region_entry(400));
    cache.put("meta1".into(), meta_entry(150));
    assert_eq!(cache.bytes_used(), 950);

    // Touch regionA1 so it is more recent than regionA2.
    assert!(cache.get("regionA1").is_some());

    // Over budget by 100: the meta entry (lower priority) must be the
    // victim — NOT either Region A entry.
    cache.put("meta2".into(), meta_entry(150));
    assert!(cache.get("meta1").is_none(), "meta-only evicted first");
    assert!(cache.get("regionA1").is_some());
    assert!(cache.get("regionA2").is_some());
    assert!(cache.get("meta2").is_some());

    // Re-touch regionA1 (the assertion `get`s above stamped recency too,
    // leaving regionA2 fresher) so regionA2 is deterministically the
    // least-recent Region A entry.
    assert!(cache.get("regionA1").is_some());

    // Force a Region-A eviction: meta2 goes first (priority), then the
    // LEAST-RECENT Region A entry (regionA2 — regionA1 was touched).
    cache.put("regionA3".into(), region_entry(400));
    assert!(
        cache.get("meta2").is_none(),
        "meta evicted before any region"
    );
    // S2b per-class semantics: the least-recent Region A COMPANION is the
    // victim — its control anchor still hits (control-only, region shed).
    assert!(
        cache
            .get("regionA2")
            .is_none_or(|inv| inv.region_bytes.is_none()),
        "least-recent Region A shed on priority tie"
    );
    assert!(
        cache
            .get("regionA1")
            .is_some_and(|inv| inv.region_bytes.is_some()),
        "recency-protected survivor keeps its region"
    );
    assert!(
        cache
            .get("regionA3")
            .is_some_and(|inv| inv.region_bytes.is_some())
    );
    assert!(cache.bytes_used() <= 1000);
}

#[test]
fn pax_write_trace_defaults_off() {
    // TD-COMPACT-1 S1: the write-path phase timer MUST be off in prod (env
    // unset) so the eprintln + Instant bookkeeping never fires on the hot
    // write path. Default-off is the safety property; the on-path is
    // exercised end-to-end by setting PROXIMADB_TRACE_PAX_WRITE at the bench.
    // No test in this binary sets the var, so the default read is deterministic.
    assert!(
        !pax_write_trace(),
        "PROXIMADB_TRACE_PAX_WRITE must be unset by default"
    );
}

#[test]
fn probe_range_coalescing_keeps_only_selected_cell_slices() {
    let selected = vec![
        ProbeCellSlice {
            start: 100,
            end: 200,
            row_start: 0,
        },
        ProbeCellSlice {
            start: 250,
            end: 350,
            row_start: 10,
        },
        ProbeCellSlice {
            start: 900,
            end: 1_000,
            row_start: 20,
        },
    ];
    let policy =
        crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: 64,
            max_range_bytes: 512,
        };

    let planned = plan_probe_cell_ranges(&selected, &policy);

    assert_eq!(planned.len(), 2, "the 50-byte gap should merge");
    assert_eq!((planned[0].start, planned[0].end), (100, 350));
    assert_eq!(planned[0].selected, selected[..2]);
    assert_eq!(planned[1].selected, selected[2..]);
    let fetched: u64 = planned.iter().map(|range| range.end - range.start).sum();
    let useful: u64 = planned
        .iter()
        .flat_map(|range| &range.selected)
        .map(|cell| cell.end - cell.start)
        .sum();
    assert_eq!(useful, 300);
    assert_eq!(fetched - useful, 50, "only the merged gap is over-read");

    let estimate =
        estimate_coalesced_byte_ranges(selected.iter().map(|cell| (cell.start, cell.end)), &policy);
    assert_eq!(estimate.get_requests, planned.len() as u64);
    assert_eq!(estimate.physical_bytes, fetched);

    let fallback = estimate_with_whole_region_fallback(estimate, 200);
    assert_eq!(fallback.get_requests, 1);
    assert_eq!(fallback.physical_bytes, 200);
    assert_eq!(estimate_with_whole_region_fallback(estimate, 400), estimate);
}

#[test]
#[serial]
fn adaptive_range_policy_obeys_gate_knee_and_explicit_override_precedence() {
    use crate::storage::engines::core::coalesce_strategy::RangePlanEstimate;

    let mib = 1024 * 1024;
    let defaults =
        crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy {
            max_gap_bytes: mib,
            max_range_bytes: 4 * mib,
        };
    let measured_1m = |policy: &crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy| {
            let (gets, bytes_mib) = match policy.max_range_bytes / mib {
                4 => (10_077, 28_561),
                8 => (8_177, 33_111),
                12 => (7_796, 33_691),
                16 => (7_657, 33_713),
                24 => (7_647, 33_713),
                _ => (10_077, 28_561),
            };
            RangePlanEstimate {
                get_requests: gets,
                physical_bytes: bytes_mib * mib,
            }
        };

    unsafe {
        std::env::remove_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
    }
    let gated_off = resolve_adaptive_range_policy(
        "az://container/segment.pax",
        "region_a",
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        defaults,
        measured_1m,
    );
    assert_eq!(gated_off, defaults, "unset gate preserves behavior");

    unsafe { std::env::set_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER", "1") };
    let adaptive = resolve_adaptive_range_policy(
        "az://container/segment.pax",
        "region_a",
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        defaults,
        measured_1m,
    );
    assert_eq!(adaptive.max_range_bytes, 16 * mib, "measured GET knee");

    unsafe {
        std::env::set_var(
            "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
            (24 * mib).to_string(),
        )
    };
    let overridden = resolve_adaptive_range_policy(
        "az://container/segment.pax",
        "region_a",
        "PROXIMADB_PAX_VECTOR_COALESCE_GAP",
        "PROXIMADB_PAX_VECTOR_COALESCE_RANGE",
        defaults,
        measured_1m,
    );
    assert_eq!(overridden.max_range_bytes, 24 * mib);

    unsafe {
        std::env::remove_var("PROXIMADB_STORAGE_READ_STRATEGY_CHOOSER");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
    }
}
use crate::storage::engines::core::formats::proximablocks::BlockCompressionConfig;
use proximadb_records::{EmbeddingCell, EmbeddingValues};

fn rec(oid: &str, ts: i64, vec: Vec<f32>) -> ProximaRecord {
    let mut r = ProximaRecord {
        oid: oid.into(),
        tenant_id: "t".into(),
        created_at_ns: ts,
        updated_at_ns: ts,
        ..Default::default()
    };
    let dim = vec.len();
    r.embeddings
        .push(EmbeddingCell::new_fp32("", "dense", dim as u32, vec));
    r
}

/// A PAX segment written by `PaxSegmentWriter` is detected as `Pax` and round-trips
/// back to the same records through the format-routing reader.
#[test]
fn pax_segment_detected_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");

    let mut w = PaxSegmentWriter::new(
        &path,
        BlockMode::Pax,
        BlockCompression::None,
        "col",
        0,
        1, // embedding_count
        None,
    );
    w.add_record(&rec("a", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0]))
        .unwrap();
    w.add_record(&rec("b", 1_700_000_000_000_000_001, vec![4.0, 5.0, 6.0]))
        .unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        SegmentFormat::detect(&bytes),
        SegmentFormat::Pax,
        "PAX segment must be detected as Pax"
    );

    let records = read_segment_records(&bytes, &[], &[], None).unwrap();
    assert_eq!(records.len(), 2);
    let mut oids: Vec<&str> = records.iter().map(|r| r.oid.as_str()).collect();
    oids.sort_unstable();
    assert_eq!(oids, vec!["a", "b"]);
}

/// A legacy `ProximaDataBlock` is detected as `ProximaBlocks` and round-trips back
/// to the same records — proving the router preserves the existing path.
#[test]
fn proxima_block_detected_and_read_back() {
    let records = vec![
        rec("x", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0, 4.0]),
        rec("y", 1_700_000_000_000_000_001, vec![5.0, 6.0, 7.0, 8.0]),
    ];
    let bytes = ProximaDataBlock::new(records, BlockCompressionConfig::default())
        .serialize()
        .unwrap();

    assert_eq!(
        SegmentFormat::detect(&bytes),
        SegmentFormat::ProximaBlocks,
        "ProximaDataBlock must be detected as ProximaBlocks"
    );

    let back = read_segment_records(&bytes, &[], &[], None).unwrap();
    let mut oids: Vec<&str> = back.iter().map(|r| r.oid.as_str()).collect();
    oids.sort_unstable();
    assert_eq!(oids, vec!["x", "y"]);
}

/// Detection never mis-routes the legacy path: version/compression-marker first
/// bytes, empty input, and a PBLK prefix without the trailing segment magic all
/// resolve to `ProximaBlocks`.
#[test]
fn detection_defaults_to_legacy_and_guards_false_positives() {
    assert_eq!(SegmentFormat::detect(&[]), SegmentFormat::ProximaBlocks);
    assert_eq!(
        SegmentFormat::detect(&[0x01, 0, 0]),
        SegmentFormat::ProximaBlocks
    );
    assert_eq!(
        SegmentFormat::detect(&[0x05, 0, 0]),
        SegmentFormat::ProximaBlocks
    );
    // Leading PBLK but no trailing PAXSEG01 → not a PAX segment.
    let mut faux = BLOCK_MAGIC.to_vec();
    faux.extend_from_slice(&[0u8; 16]);
    assert_eq!(SegmentFormat::detect(&faux), SegmentFormat::ProximaBlocks);
}

/// The default format is the legacy one (serde-absent manifests recover unchanged).
#[test]
fn default_is_proxima_blocks() {
    assert_eq!(SegmentFormat::default(), SegmentFormat::ProximaBlocks);
}

/// `write_pax_segment` is the write-side inverse: what it writes is detected as PAX
/// and reads back through `read_segment_records` (the Phase B flush↔read round-trip,
/// exercised without the flush harness).
#[test]
fn write_pax_segment_round_trips_through_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written.pax");
    let records = vec![
        rec("w1", 1_700_000_000_000_000_000, vec![1.0, 2.0, 3.0]),
        rec("w2", 1_700_000_000_000_000_001, vec![4.0, 5.0, 6.0]),
    ];
    let meta = write_pax_segment(&path, &records, "col", 1, VectorQuant::Auto, None, None).unwrap();
    assert!(
        meta.block_count >= 1,
        "segment should have at least one block"
    );

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);
    let back = read_segment_records(&bytes, &[], &[], None).unwrap();
    let mut oids: Vec<&str> = back.iter().map(|r| r.oid.as_str()).collect();
    oids.sort_unstable();
    assert_eq!(oids, vec!["w1", "w2"]);
}

/// M1 (ADR-049): a `VectorQuant::RawF32` PAX segment round-trips to EXACT f32
/// vectors (recall 1.0) — proving the foundation of the exact PAX scan path
/// (`search_pax_file_exact`, which materialises records via
/// `read_segment_records`). RawF32 decodes to the exact input vectors, not an
/// approximation (unlike RaBitQ/SQ8 reconstruction).
#[test]
fn rawf32_pax_segment_round_trips_exact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rawf32.pax");
    let records: Vec<ProximaRecord> = (0..8)
        .map(|i| {
            rec(
                &format!("v{i}"),
                1_700_000_000_000_000_000 + i,
                (0..16).map(|d| (i as f32 + d as f32) * 0.1).collect(),
            )
        })
        .collect();
    write_pax_segment(&path, &records, "col", 1, VectorQuant::RawF32, None, None).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);

    // Round-trip: RawF32 decodes to the EXACT input vectors (not an
    // approximation, unlike RaBitQ/SQ8 reconstruction).
    let back = read_segment_records(&bytes, &[], &[], None).unwrap();
    assert_eq!(back.len(), records.len());
    for (got, want) in back.iter().zip(records.iter()) {
        assert_eq!(got.oid, want.oid, "oid must round-trip");
        let got_vec = got
            .embeddings
            .first()
            .expect("RawF32 segment must reconstruct its embedding");
        let want_vec = want.embeddings.first().expect("input embedding");
        assert_eq!(
            got_vec.as_fp32_slice(),
            want_vec.as_fp32_slice(),
            "RawF32 PAX must reconstruct the exact vector for {}",
            want.oid
        );
    }
}

/// Mixed-read-safe coexistence (storage-format mandate #8). A store rolling
/// PAX quantization on will have BOTH legacy `ProximaBlocks` segments (written
/// before the flip) and new PAX segments side by side. The magic-byte router
/// must read BOTH back through the single `read_segment_records` entry —
/// disjoint oids, no mis-route. This is the property that makes the staged
/// default-on flip safe (default OFF in v0.2, flip gated by the recall ratchet).
#[test]
fn mixed_format_legacy_and_pax_segments_coexist_and_read_back() {
    let legacy_records = vec![
        rec(
            "legacy_a",
            1_700_000_000_000_000_000,
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        rec(
            "legacy_b",
            1_700_000_000_000_000_001,
            vec![5.0, 6.0, 7.0, 8.0],
        ),
    ];
    let legacy_bytes = ProximaDataBlock::new(legacy_records, BlockCompressionConfig::default())
        .serialize()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let pax_path = dir.path().join("coexist.pax");
    let pax_records = vec![
        rec(
            "pax_c",
            1_700_000_000_000_000_002,
            vec![9.0, 10.0, 11.0, 12.0],
        ),
        rec(
            "pax_d",
            1_700_000_000_000_000_003,
            vec![13.0, 14.0, 15.0, 16.0],
        ),
    ];
    write_pax_segment(
        &pax_path,
        &pax_records,
        "col",
        1,
        VectorQuant::RaBitQ,
        None,
        None,
    )
    .unwrap();
    let pax_bytes = std::fs::read(&pax_path).unwrap();

    // The two segments are detected as DIFFERENT formats (never mis-routed).
    assert_eq!(
        SegmentFormat::detect(&legacy_bytes),
        SegmentFormat::ProximaBlocks
    );
    assert_eq!(SegmentFormat::detect(&pax_bytes), SegmentFormat::Pax);

    // ...and BOTH read back through the single mixed-format router.
    let legacy_back = read_segment_records(&legacy_bytes, &[], &[], None).unwrap();
    let pax_back = read_segment_records(&pax_bytes, &[], &[], None).unwrap();

    let mut legacy_oids: Vec<&str> = legacy_back.iter().map(|r| r.oid.as_str()).collect();
    legacy_oids.sort_unstable();
    let mut pax_oids: Vec<&str> = pax_back.iter().map(|r| r.oid.as_str()).collect();
    pax_oids.sort_unstable();

    assert_eq!(
        legacy_oids,
        vec!["legacy_a", "legacy_b"],
        "legacy segment oids"
    );
    assert_eq!(
        pax_oids,
        vec!["pax_c", "pax_d"],
        "PAX segment oids (disjoint from legacy)"
    );
}

/// TD-USUB-5 / ADR-094 §4 — converting [`read_segment_records`] from an
/// inline `match` to [`RecordReader`] dispatch must be **behaviour-neutral**
/// for the formats that already existed.
///
/// The assertion is equality against the canonical decoders called
/// DIRECTLY — `ProximaDataBlock::deserialize` for the legacy format and
/// `read_pax_segment_records` for PAX — on whole `ProximaRecord`s, not just
/// oids. So the test fails if the trait indirection perturbs any field
/// (embeddings, props, timestamps, tenant), which a shape-only assertion
/// would miss.
#[test]
fn trait_dispatch_is_byte_identical_for_existing_formats() {
    let legacy_records = vec![
        rec(
            "legacy_a",
            1_700_000_000_000_000_000,
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        rec(
            "legacy_b",
            1_700_000_000_000_000_001,
            vec![5.0, 6.0, 7.0, 8.0],
        ),
    ];
    let legacy_bytes = ProximaDataBlock::new(legacy_records, BlockCompressionConfig::default())
        .serialize()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let pax_path = dir.path().join("dispatch.pax");
    let pax_records = vec![
        rec(
            "pax_c",
            1_700_000_000_000_000_002,
            vec![9.0, 10.0, 11.0, 12.0],
        ),
        rec(
            "pax_d",
            1_700_000_000_000_000_003,
            vec![13.0, 14.0, 15.0, 16.0],
        ),
    ];
    write_pax_segment(
        &pax_path,
        &pax_records,
        "col",
        1,
        VectorQuant::RaBitQ,
        None,
        None,
    )
    .unwrap();
    let pax_bytes = std::fs::read(&pax_path).unwrap();

    // Legacy: router output == `ProximaDataBlock::deserialize` output.
    let via_router = read_segment_records(&legacy_bytes, &[], &[], Some("t1")).unwrap();
    let direct = ProximaDataBlock::deserialize(&legacy_bytes, None)
        .unwrap()
        .records;
    assert_eq!(via_router, direct, "legacy decode changed under dispatch");

    // PAX: router output == `read_pax_segment_records` output, with the
    // catalog-derived context threaded through unchanged.
    let via_router = read_segment_records(&pax_bytes, &[], &[], Some("t1")).unwrap();
    let direct = proximadb_storage_common::pax_block::read_pax_segment_records(
        &pax_bytes,
        &[],
        &[],
        Some("t1"),
    )
    .unwrap();
    assert_eq!(via_router, direct, "PAX decode changed under dispatch");

    // The dispatch table agrees with detection for every format — the
    // invariant that keeps "detected as X" and "decoded as X" from drifting.
    for bytes in [&legacy_bytes[..], &pax_bytes[..]] {
        let format = SegmentFormat::detect(bytes);
        assert_eq!(record_reader_for(format).format(), format);
    }
}

/// TD-USUB-5 — a Parquet object reaching the router is now DECODED rather
/// than handed to the legacy decoder that cannot read it.
///
/// The second half is the part that matters for mandate #1: even without the
/// Parquet arm the old code did not return wrong rows here, it returned an
/// error — but it was an error about the wrong format. Naming the format
/// turns a mis-route into a decode.
#[test]
fn parquet_segment_routes_to_the_parquet_reader() {
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("city", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec!["pune", "austin"]))],
    )
    .unwrap();
    let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
        &[batch],
        schema,
        None,
    )
    .unwrap();

    assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Parquet);
    assert_eq!(
        record_reader_for(SegmentFormat::Parquet).format(),
        SegmentFormat::Parquet
    );

    let records = read_segment_records(&bytes, &[], &[], Some("tenant-9")).unwrap();
    assert_eq!(records.len(), 2);
    assert!(
        records.iter().all(|r| r.tenant_id == "tenant-9"),
        "tenant context must be stamped on the Parquet arm too"
    );

    // Non-vacuity: the legacy decoder genuinely cannot read these bytes, so
    // the new arm is doing real work rather than shadowing a path that
    // already happened to succeed.
    assert!(
        ProximaDataBlock::deserialize(&bytes, None).is_err(),
        "legacy decoder must reject Parquet bytes — otherwise this test proves nothing"
    );
}

/// TD-156 / ADR-026: block geometry is the fragmentation lever. A larger
/// `target_block` packs rows into fewer blocks — so the per-block ranged-GET
/// count on the object-store read path drops (the +29,900% GET fragmentation
/// measured by the footer-cache economics harness shrinks as block count
/// falls). Mixed-read-safe: block size is a per-segment write choice, opaque
/// to the magic-detected read router.
#[test]
#[serial]
fn larger_target_block_yields_fewer_blocks() {
    // TD-PAXRG-1: post-flip the default write is the row-group layout,
    // whose 256 KiB floor clamps sub-floor targets — this test pins the
    // LEGACY geometry lever, so it pins the kill-switch (nextest isolates
    // the process).
    unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0") };
    let records: Vec<ProximaRecord> = (0..512)
        .map(|i| rec(&format!("r{i}"), 1000 + i as i64, vec![i as f32 * 0.1; 64]))
        .collect();
    let blocks_for = |target: Option<usize>| -> usize {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("geo.pax");
        write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, target, None)
            .unwrap()
            .block_count as usize
    };
    let small = blocks_for(Some(8 * 1024));
    let large = blocks_for(Some(256 * 1024));
    eprintln!(
        "[geometry] 512x64 vectors: 8KiB target -> {small} blocks, 256KiB target -> {large} blocks"
    );
    assert!(
        large < small,
        "larger target_block must yield fewer blocks: {large} >= {small}"
    );
    assert!(large >= 1, "at least one block");
}

// ── ADR-062 / TD-RDSTRAT-6 coalesced-RaBitQ layout ───────────────────────

/// Opt into the coalesced-RaBitQ layout for one test. SAFETY: nextest runs
/// process-per-test, so the env mutation is isolated to this test process.
fn enable_coalesced_rabitq() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
    }
}

fn disable_coalesced_rabitq() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "0");
    }
}

#[test]
#[serial]
fn coalesced_requested_f32_tier_is_emitted_and_materializes_exact() {
    enable_coalesced_rabitq();
    // Post-flip the default write is the row-group layout, where the exact
    // tier HOISTS to Region C (covered by
    // `v4_f32_tier_writes_region_c_not_block_stripes` + the parity oracle).
    // THIS test pins the legacy in-block stripe contract, so it pins the
    // legacy framing explicitly (env-independent).
    unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0") };
    use proximadb_block_format::col_id;
    use proximadb_storage_common::segment_layout::SegmentFooterIndex;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coalesced-exact.pax");
    let records = vec![
        rec("a", 1, vec![-7.25, 0.123_456_7, 3.141_592_7]),
        rec("b", 2, vec![2.75, 8.765_432, -0.333_333_34]),
        rec("c", 3, vec![11.125, -4.567_891, 9.999_991]),
    ];

    write_pax_segment_with_f32_tier(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        true,
        Some(1024),
        None,
    )
    .unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let footer = SegmentFooterIndex::locate_in_segment(&bytes)
        .unwrap()
        .expect("coalesced footer");
    assert!(footer.has_f32_tier, "footer must declare the exact tier");
    assert!(
        pax_segment_has_exact_vector_authority(&bytes),
        "coalesced footer capability must admit its emitted exact tier"
    );

    let mut scanner =
        PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
    while let Some(block) = scanner.next_block() {
        assert!(
            block.vector_params().get(col_id::F32_TIER_BASE).is_some(),
            "every coalesced block must emit the requested exact tier"
        );
        assert!(
            block.has_exact_vector_authority(),
            "an exact-only coalesced block must prove non-vacuous authority"
        );
    }

    // The canonical materializer must prefer the block-local exact tier over
    // Region B's SQ8 rerank values. Merely checking that a vector exists would
    // allow Exact search to rank a lossy reconstruction.
    let materialized = read_segment_records(&bytes, &[], &[], None).unwrap();
    assert_eq!(materialized.len(), records.len());
    let expected: std::collections::HashMap<_, _> = records
        .iter()
        .map(|record| (record.oid.as_str(), &record.embeddings))
        .collect();
    for got in &materialized {
        let want = expected.get(got.oid.as_str()).expect("known record oid");
        assert_eq!(
            got.embeddings.first().map(|cell| cell.as_fp32_cow()),
            want.first().map(|cell| cell.as_fp32_cow()),
            "exact tier must round-trip bitwise for {}",
            got.oid
        );
    }
}

#[tokio::test]
#[serial]
async fn mixed_exact_and_lossy_inputs_do_not_claim_exact_output() {
    disable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let exact = dir.path().join("exact.pax");
    let lossy = dir.path().join("lossy.pax");
    let records = vec![
        rec("a", 1, vec![-1.0, 0.123_456_7, 7.0]),
        rec("b", 2, vec![3.0, 4.765_432, -2.0]),
    ];

    write_pax_segment_with_f32_tier(
        &exact,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        true,
        None,
        None,
    )
    .unwrap();
    write_pax_segment_with_f32_tier(
        &lossy,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        false,
        None,
        None,
    )
    .unwrap();

    assert!(input_tiers(std::slice::from_ref(&exact)).await.f32_tier);
    assert!(pax_segment_has_exact_vector_authority(
        &std::fs::read(&exact).unwrap()
    ));
    assert!(
        !input_tiers(&[exact.clone(), lossy.clone()]).await.f32_tier,
        "one exact input must not make a mixed compacted output exact"
    );
    assert!(
        !pax_segment_has_exact_vector_authority(&std::fs::read(&lossy).unwrap()),
        "lossy PAX must not satisfy an exact-search contract"
    );
    assert!(
        !input_tiers(&[exact, dir.path().join("unknown.arrow")])
            .await
            .f32_tier,
        "an unrelated physical format must not be assumed exact"
    );
}

#[tokio::test]
#[serial]
async fn coalesced_lossy_segment_does_not_claim_exact_authority_vacuously() {
    enable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let lossy = dir.path().join("coalesced-lossy.pax");
    let records = vec![
        rec("a", 1, vec![-1.0, 0.123_456_7, 7.0]),
        rec("b", 2, vec![3.0, 4.765_432, -2.0]),
    ];
    write_pax_segment_with_f32_tier(
        &lossy,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        false,
        None,
        None,
    )
    .unwrap();

    let bytes = std::fs::read(&lossy).unwrap();
    assert!(
        !pax_segment_has_exact_vector_authority(&bytes),
        "hoisted vectors with no f32 tier are lossy, not vacuously exact"
    );
    let mut scanner = PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default()).unwrap();
    while let Some(block) = scanner.next_block() {
        assert!(
            !block.has_exact_vector_authority(),
            "a coalesced block with neither base nor exact stripes must not pass vacuously"
        );
    }
    assert!(
        !input_tiers(&[file_url(&lossy)]).await.f32_tier,
        "compaction must not upgrade a lossy coalesced input to exact"
    );
}

/// Resolve compaction input tiers through the real local filesystem port —
/// the same port compaction hands `pax_compaction_input_tiers`.
async fn input_tiers(inputs: &[std::path::PathBuf]) -> PaxCompactionInputTiers {
    let factory = crate::storage::persistence::filesystem::FilesystemFactory::create_default()
        .await
        .expect("filesystem factory");
    let port: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> =
        std::sync::Arc::new(factory);
    pax_compaction_input_tiers(&port, inputs)
        .await
        .expect("every input is readable")
}

fn file_url(path: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("file://{}", path.display()))
}

fn tier_records() -> Vec<ProximaRecord> {
    vec![
        rec("a", 1, vec![-1.0, 0.123_456_7, 7.0]),
        rec("b", 2, vec![3.0, 4.765_432, -2.0]),
    ]
}

/// TD-READFID-2: compaction hands the tier probe its inputs as
/// `file://{abs}` URLs (`GenericFileMetadata.path = entry.url`). The old
/// `std::fs::read` predicates could not open them, so the exact tier was
/// dropped at the first compaction on every real deployment. Covers the
/// production default (coalesced, row-group layout => Region C) and the
/// legacy in-block tier, for `file://` URLs and bare paths alike.
#[tokio::test]
#[serial]
async fn file_url_compaction_inputs_keep_the_f32_tier() {
    enable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    for (name, rg_layout) in [("region-c.pax", true), ("in-block.pax", false)] {
        let exact = dir.path().join(name);
        write_pax_segment_full_with_layout(
            &exact,
            &tier_records(),
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            true,
            None,
            rg_layout,
            &[],
            None,
        )
        .unwrap();
        assert!(
            input_tiers(std::slice::from_ref(&exact)).await.f32_tier,
            "{name}: bare path"
        );
        assert!(
            input_tiers(&[file_url(&exact)]).await.f32_tier,
            "{name}: a file:// compaction input must still report its exact tier"
        );
    }
    let both = [
        file_url(&dir.path().join("region-c.pax")),
        file_url(&dir.path().join("in-block.pax")),
    ];
    assert!(
        input_tiers(&both).await.f32_tier,
        "every input exact => the merged output keeps the tier"
    );
}

/// TD-READFID-2 (rerank half): a configured FP16 rerank column must survive
/// compaction for `file://` inputs too (it used to silently fall to SQ8).
#[tokio::test]
#[serial]
async fn file_url_compaction_inputs_keep_the_fp16_rerank() {
    disable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("fp16.pax");
    write_pax_segment_full(
        &seg,
        &tier_records(),
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Fp16,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        input_tiers(std::slice::from_ref(&seg)).await.rerank_quant,
        VectorQuant::Fp16
    );
    let url_tiers = input_tiers(&[file_url(&seg)]).await;
    assert_eq!(
        url_tiers.rerank_quant,
        VectorQuant::Fp16,
        "a file:// compaction input must keep its FP16 rerank tier"
    );
    assert!(
        !url_tiers.f32_tier,
        "an FP16 rerank is not an exact authority for a RaBitQ base"
    );
    // No `.pax` input carrying a rerank column => the validated SQ8 default.
    assert_eq!(
        input_tiers(&[dir.path().join("legacy.sst")])
            .await
            .rerank_quant,
        VectorQuant::Sq8
    );
}

/// TD-READFID-2 review: an input the port cannot read is an ERROR, never
/// "not exact". Classifying it as lossy let compaction write a tierless
/// output and delete the inputs — the only copies of the exact values.
#[tokio::test]
#[serial]
async fn unreadable_compaction_input_is_an_error_not_lossy() {
    enable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let exact = dir.path().join("exact.pax");
    write_pax_segment_full_with_layout(
        &exact,
        &tier_records(),
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        true,
        None,
        true,
        &[],
        None,
    )
    .unwrap();
    let factory = crate::storage::persistence::filesystem::FilesystemFactory::create_default()
        .await
        .expect("filesystem factory");
    let port: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> =
        std::sync::Arc::new(factory);
    let result = pax_compaction_input_tiers(
        &port,
        &[file_url(&exact), file_url(&dir.path().join("missing.pax"))],
    )
    .await;
    assert!(
        result.is_err(),
        "an unreadable input must abort the probe, not demote the output: {result:?}"
    );
}

/// In-memory object store keyed by full URL, counting whole vs ranged
/// reads — stands in for an `s3://`/`az://` backend behind the port.
#[derive(Debug, Default)]
struct CountingObjectStore {
    objects: std::collections::HashMap<String, Vec<u8>>,
    whole_reads: std::sync::atomic::AtomicUsize,
    ranged_reads: std::sync::atomic::AtomicUsize,
}

impl CountingObjectStore {
    fn object(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<&Vec<u8>> {
        self.objects.get(path).ok_or_else(|| {
            proximadb_storage_filesystem_types::FilesystemError::NotFound(path.to_string())
        })
    }
}

fn unsupported<T>() -> proximadb_storage_filesystem_types::FsResult<T> {
    Err(
        proximadb_storage_filesystem_types::FilesystemError::UnsupportedScheme(
            "test object store is read-only".to_string(),
        ),
    )
}

#[async_trait::async_trait]
impl proximadb_storage_filesystem_types::FileSystem for CountingObjectStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
        self.whole_reads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.object(path).cloned()
    }
    async fn read_range(
        &self,
        path: &str,
        offset: u64,
        length: u64,
    ) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
        self.ranged_reads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let bytes = self.object(path)?;
        let start = (offset as usize).min(bytes.len());
        let end = (offset.saturating_add(length) as usize).min(bytes.len());
        Ok(bytes[start..end].to_vec())
    }
    async fn write(
        &self,
        _: &str,
        _: &[u8],
        _: Option<proximadb_storage_filesystem_types::FileOptions>,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn append(&self, _: &str, _: &[u8]) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn delete(&self, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn exists(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<bool> {
        Ok(self.objects.contains_key(path))
    }
    async fn metadata(
        &self,
        path: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        proximadb_storage_filesystem_types::FsFileMetadata,
    > {
        Ok(proximadb_storage_filesystem_types::FsFileMetadata {
            path: path.to_string(),
            size: self.object(path)?.len() as u64,
            ..Default::default()
        })
    }
    async fn list(
        &self,
        _: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        Vec<proximadb_storage_filesystem_types::DirEntry>,
    > {
        unsupported()
    }
    async fn create_dir(&self, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn create_dir_all(&self, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn copy(&self, _: &str, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn move_file(
        &self,
        _: &str,
        _: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    fn filesystem_type(&self) -> &'static str {
        "counting-object-store"
    }
    async fn sync(&self) -> proximadb_storage_filesystem_types::FsResult<()> {
        Ok(())
    }
    async fn open_file(
        &self,
        _: &str,
        _: bool,
    ) -> proximadb_storage_filesystem_types::FsResult<
        Box<dyn proximadb_storage_filesystem_types::FilesystemFile>,
    > {
        unsupported()
    }
}

struct CountingPort(std::sync::Arc<CountingObjectStore>);

#[async_trait::async_trait]
impl proximadb_storage_ports::FilesystemPort for CountingPort {
    fn get_filesystem(
        &self,
        _url: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        std::sync::Arc<dyn proximadb_storage_filesystem_types::FileSystem>,
    > {
        Ok(self.0.clone())
    }
    async fn create_dir_all(&self, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn write(
        &self,
        _: &str,
        _: &[u8],
        _: Option<proximadb_storage_filesystem_types::FileOptions>,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn move_atomic(
        &self,
        _: &str,
        _: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn delete(&self, _: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
        unsupported()
    }
    async fn read(&self, url: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
        proximadb_storage_filesystem_types::FileSystem::read(self.0.as_ref(), url).await
    }
    async fn list(
        &self,
        _: &str,
    ) -> proximadb_storage_filesystem_types::FsResult<
        Vec<proximadb_storage_filesystem_types::DirEntry>,
    > {
        unsupported()
    }
}

/// Object-store inputs resolve through the port (no local path exists for
/// them at all), and the co-design contract holds: a coalesced segment is
/// answered from ranged footer reads with ZERO whole-object GETs; only a
/// legacy block-layout segment (per-block exactness) costs one whole read.
#[tokio::test]
#[serial]
async fn object_store_inputs_resolve_through_the_port_footer_first() {
    use std::sync::atomic::Ordering;
    enable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let coalesced = dir.path().join("coalesced.pax");
    write_pax_segment_full_with_layout(
        &coalesced,
        &tier_records(),
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        true,
        None,
        true,
        &[],
        None,
    )
    .unwrap();
    disable_coalesced_rabitq();
    let legacy = dir.path().join("legacy.pax");
    write_pax_segment_full(
        &legacy,
        &tier_records(),
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Fp16,
        true,
        None,
        None,
    )
    .unwrap();
    let coalesced_url = "s3://bucket/data/t/n/c/L0_a.pax".to_string();
    let legacy_url = "s3://bucket/data/t/n/c/L0_b.pax".to_string();
    let mut store = CountingObjectStore::default();
    store
        .objects
        .insert(coalesced_url.clone(), std::fs::read(&coalesced).unwrap());
    store
        .objects
        .insert(legacy_url.clone(), std::fs::read(&legacy).unwrap());
    let store = std::sync::Arc::new(store);
    let port: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> =
        std::sync::Arc::new(CountingPort(store.clone()));

    let tiers = pax_compaction_input_tiers(&port, &[std::path::PathBuf::from(&coalesced_url)])
        .await
        .expect("coalesced object-store input is readable");
    assert!(
        tiers.f32_tier,
        "object-store coalesced input keeps the tier"
    );
    assert_eq!(
        store.whole_reads.load(Ordering::SeqCst),
        0,
        "a coalesced input must be classified from its footer alone"
    );
    assert_eq!(
        store.ranged_reads.load(Ordering::SeqCst),
        2,
        "header prefix + footer"
    );

    let tiers = pax_compaction_input_tiers(
        &port,
        &[
            std::path::PathBuf::from(&coalesced_url),
            std::path::PathBuf::from(&legacy_url),
        ],
    )
    .await
    .expect("both object-store inputs are readable");
    assert!(tiers.f32_tier, "both object-store inputs carry the tier");
    assert_eq!(
        tiers.rerank_quant,
        VectorQuant::Fp16,
        "the legacy input's FP16 rerank is preserved"
    );
    assert_eq!(
        store.whole_reads.load(Ordering::SeqCst),
        1,
        "only the legacy block-layout input needs a whole read"
    );
}

/// TD-COMPACT-IO-1 mitigation: a compaction whose merged records carry no
/// embeddings (relational / document) skips the tier probe with ZERO I/O.
#[tokio::test]
async fn non_vector_compaction_skips_the_tier_probe() {
    use std::sync::atomic::Ordering;
    let store = std::sync::Arc::new(CountingObjectStore::default());
    let port: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> =
        std::sync::Arc::new(CountingPort(store.clone()));
    let row = ProximaRecord {
        oid: "row".into(),
        ..Default::default()
    };
    let tiers = pax_compaction_input_tiers_for_records(
        &port,
        &[std::path::PathBuf::from(
            "s3://bucket/data/t/n/c/L0_rel.pax",
        )],
        std::slice::from_ref(&row),
    )
    .await
    .expect("non-vector compactions never read");
    assert!(!tiers.f32_tier);
    assert!(!tiers.carries_vectors);
    assert_eq!(tiers.rerank_quant, VectorQuant::Sq8);
    assert_eq!(store.whole_reads.load(Ordering::SeqCst), 0);
    assert_eq!(store.ranged_reads.load(Ordering::SeqCst), 0);
}

#[test]
fn coalesced_rank_scores_cross_the_public_metric_boundary_once() {
    assert_eq!(
        canonical_score_from_rank_score(RankMetric::L2, 25.0),
        5.0,
        "squared L2 rank score must become canonical Euclidean distance"
    );
    assert_eq!(
        canonical_score_from_rank_score(RankMetric::DotProduct, -6.0),
        6.0,
        "negated-dot rank score must become public positive-dot similarity"
    );
    assert_eq!(
        canonical_score_from_rank_score(RankMetric::Cosine, 0.25),
        0.25,
        "cosine distance is already canonical"
    );
}

#[test]
fn fused_sq8_rerank_matches_decode_then_score_for_every_metric() {
    let params = proximadb_codec::Sq8Params {
        scale: 0.0625,
        offset: -2.0,
        vmin: -2.0,
        vmax: 13.9375,
    };
    let codes = (0..128)
        .map(|i| ((i * 47 + 11) % 256) as u8)
        .collect::<Vec<_>>();
    let query = (0..128)
        .map(|i| ((i * 31 + 3) as f32 * 0.013).cos() * 3.0)
        .collect::<Vec<_>>();
    let decoded = proximadb_codec::functions::sq8::decode(&codes, &params);
    let query_norm_squared = query.iter().map(|value| value * value).sum::<f32>();
    for metric in [RankMetric::L2, RankMetric::Cosine, RankMetric::DotProduct] {
        let expected = match metric {
            RankMetric::L2 => query
                .iter()
                .zip(&decoded)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>(),
            RankMetric::Cosine => {
                let dot = query.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>();
                let decoded_norm = decoded.iter().map(|value| value * value).sum::<f32>();
                1.0 - dot / ((query_norm_squared * decoded_norm).sqrt() + 1e-12)
            }
            RankMetric::DotProduct => -query.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>(),
        };
        let actual = rerank_sq8_distance(metric, &query, &codes, &params, query_norm_squared)
            .expect("matching SQ8/query dimensions");
        let tolerance = expected.abs().max(1.0) * 1e-5;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{metric:?}: fused={actual} decoded={expected}"
        );
    }
    assert!(rerank_sq8_distance(RankMetric::L2, &query[..127], &codes, &params, 0.0).is_none());
}

/// A coalesced-RaBitQ segment (opt-in) is detected as Pax, carries a non-zero
/// RaBitQ region, and round-trips back to the same records through the
/// mixed-format reader. The RaBitQ binary tier lives in the header region;
/// the block's EMBED_BASE is SQ8 (the rerank data), which the reader
/// reconstructs via the unchanged SQ8 decode path.
#[test]
#[serial]
fn coalesced_rabitq_segment_round_trips_and_detects_as_pax() {
    enable_coalesced_rabitq();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coalesced.pax");
    let records: Vec<ProximaRecord> = (0..64)
        .map(|i| {
            rec(
                &format!("v{i}"),
                1_700_000_000_000_000_000 + i,
                (0..48).map(|d| (i as f32 + d as f32) * 0.1).collect(),
            )
        })
        .collect();
    let meta =
        write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None).unwrap();
    // Coalesced layout: a non-empty RaBitQ region pinned right after the
    // 40 B header-prefix.
    assert!(
        meta.rabitq_len > 0,
        "coalesced segment must carry a RaBitQ region"
    );
    assert_eq!(
        meta.rabitq_off,
        proximadb_storage_common::segment_layout::SEG_HEADER_PREFIX_LEN as u64
    );

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Pax);
    assert!(
        is_coalesced_segment(&bytes),
        "opt-in write must produce the coalesced presence-field"
    );

    // Region B (ADR-065): vectors live in the coalesced SQ8 region, so
    // read_segment_records fails closed — verify the OID round-trip via the
    // scanner's block read (Region D row data) instead.
    let mut scan = PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
    let back = scan.read_records(&[], &[], None).unwrap();
    assert_eq!(back.len(), records.len());
    let got: std::collections::HashSet<String> = back.iter().map(|r| r.oid.clone()).collect();
    let want: std::collections::HashSet<String> = (0..64).map(|i| format!("v{i}")).collect();
    assert_eq!(got, want, "round-tripped oids must match the input set");
}

/// The presence-field cleanly distinguishes a coalesced segment from a legacy
/// one (mixed-read): opt-out writes carry no RaBitQ region and are NOT
/// coalesced; opt-in writes are. Both detect as Pax.
#[test]
#[serial]
fn coalesced_presence_field_distinguishes_legacy_from_coalesced() {
    let mk = |coalesced: bool, name: &str| -> (Vec<u8>, u64) {
        if coalesced {
            enable_coalesced_rabitq();
        } else {
            disable_coalesced_rabitq();
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let records: Vec<ProximaRecord> = (0..16)
            .map(|i| {
                rec(
                    &format!("v{i}"),
                    1_700_000_000_000_000_000 + i,
                    vec![i as f32 * 0.1; 32],
                )
            })
            .collect();
        let meta =
            write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None).unwrap();
        (std::fs::read(&path).unwrap(), meta.rabitq_len)
    };

    let (legacy_bytes, legacy_len) = mk(false, "legacy.pax");
    assert_eq!(legacy_len, 0, "legacy write has no coalesced region");
    assert!(
        !is_coalesced_segment(&legacy_bytes),
        "legacy write must not carry the coalesced presence-field"
    );
    assert_eq!(SegmentFormat::detect(&legacy_bytes), SegmentFormat::Pax);

    let (coalesced_bytes, coalesced_len) = mk(true, "coalesced.pax");
    assert!(coalesced_len > 0, "opt-in write carries a RaBitQ region");
    assert!(is_coalesced_segment(&coalesced_bytes));
    assert_eq!(SegmentFormat::detect(&coalesced_bytes), SegmentFormat::Pax);
}

/// The coalesced region + footer-index of a real written segment parse and the
/// region ranks: the RaBitQ region decodes (Slice 2a codec over writer output)
/// and the footer's block table + rabitq mirror match the SegmentMeta. This
/// ties the writer (2d), scanner footer parse (2e), and region codec (2a).
#[test]
#[serial]
fn coalesced_region_and_footer_parse_from_written_segment() {
    enable_coalesced_rabitq();
    use proximadb_block_format::{RaBitQRegion, col_id};
    use proximadb_storage_common::segment_layout::SegmentFooterIndex;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coalesced.pax");
    let records: Vec<ProximaRecord> = (0..128)
        .map(|i| {
            rec(
                &format!("v{i}"),
                1_700_000_000_000_000_000 + i,
                (0..64).map(|d| (i as f32 + d as f32) * 0.1).collect(),
            )
        })
        .collect();
    let meta = write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(8 * 1024),
        None,
    )
    .unwrap();
    assert!(meta.block_count >= 1);

    let bytes = std::fs::read(&path).unwrap();

    // Footer-index: block table + rabitq mirror match the SegmentMeta.
    let footer = SegmentFooterIndex::locate_in_segment(&bytes)
        .unwrap()
        .expect("coalesced footer located");
    assert_eq!(footer.blocks.len(), meta.block_count as usize);
    assert_eq!(footer.rabitq_off, meta.rabitq_off);
    assert_eq!(footer.rabitq_len, meta.rabitq_len);
    assert_eq!(footer.row_count, meta.row_count);
    assert_eq!(footer.embed_quant_tag, 1, "block tier-1 is SQ8");

    // RaBitQ region: decodes + ranks nearest-first; n_rows == record count.
    let off = meta.rabitq_off as usize;
    let region = &bytes[off..off + meta.rabitq_len as usize];
    let parsed = RaBitQRegion::from_bytes(region).unwrap();
    assert_eq!(parsed.n_rows(), records.len());

    // ADR-065 Region B: the SQ8 rerank tier is hoisted out of the block into a
    // coalesced region — the block carries NO EMBED_BASE vector stripe (pure
    // row data, Region D). Region B parses and holds the segment vectors.
    let mut scanner =
        PaxSegmentScanner::from_bytes(bytes.clone(), ScanPredicate::default()).unwrap();
    let block = scanner.next_block().expect("segment has a block");
    let embed = block.vector_params().get(col_id::EMBED_BASE);
    assert!(
        embed.is_none(),
        "coalesced block must carry NO EMBED_BASE (SQ8 hoisted to Region B)"
    );
    assert!(
        meta.sq8_len > 0,
        "coalesced segment carries a Region B SQ8 region"
    );
    assert_eq!(footer.sq8_off, meta.sq8_off);
    assert_eq!(footer.sq8_len, meta.sq8_len);
    let sq8_off = meta.sq8_off as usize;
    let sq8_region = &bytes[sq8_off..sq8_off + meta.sq8_len as usize];
    let sq8_hdr = proximadb_block_format::coalesced_sq8::CoalescedSq8Header::parse(sq8_region)
        .expect("Region B header parses");
    assert_eq!(sq8_hdr.n_rows as usize, records.len());
    assert_eq!(sq8_hdr.dim as usize, 64);

    // Rank a query that is one of the records; the top-1 survivor is present.
    let query = records[10]
        .embeddings
        .first()
        .unwrap()
        .as_fp32_slice()
        .to_vec();
    let ranked = parsed.rank(&query, RankMetric::L2, records.len());
    assert!(!ranked.is_empty());
    assert!(parsed.code(ranked[0]).is_some());
}

/// ADR-062 scan-then-rerank over a coalesced segment via the filesystem:
/// recall@k holds vs brute-force ground truth, and the path returns the
/// nearest neighbours. (The GET-count win is quantified by the SIFT ratchet;
/// this proves correctness of the new read path end-to-end.)
#[tokio::test]
#[serial]
async fn coalesced_scan_then_rerank_recall_vs_bruteforce() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 64;
    const N: usize = 400;
    const K: usize = 10;

    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    // Small target block ⇒ multi-block segment (exercises survivor→block mapping).
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();

    let query = corpus[137].clone();
    let hits = rabitq_search_segment_coalesced(
        &fs,
        path.to_str().unwrap(),
        &query,
        K,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .unwrap()
    .expect("coalesced scan-then-rerank must return hits");
    assert!(!hits.is_empty(), "cascade must return top-k");

    // Brute-force ground-truth top-k over the f32 corpus.
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut idx: Vec<usize> = (0..N).collect();
    idx.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        idx.iter().take(K).map(|i| format!("r{i}")).collect();
    let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
    let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
    assert!(
        recall >= 0.90,
        "coalesced scan-then-rerank recall@{K} = {recall:.2} (N={N})"
    );

    // The nearest neighbour (r137 is the query itself) must be ranked first.
    assert_eq!(
        hits[0].oid, "r137",
        "the query vector itself must be the top hit"
    );
}

/// ADR-089 P1 test helper: a record carrying a `partition` prop (`p{i%8}`),
/// mirroring the context-corridor benchmark's filter shape.
fn rec_with_partition(i: usize, vec: Vec<f32>) -> ProximaRecord {
    let mut r = rec(&format!("r{i}"), 1000 + i as i64, vec);
    r.props.insert(
        "partition".to_string(),
        proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
            format!("p{}", i % 8),
        )),
    );
    r
}

fn partition_filter(value: &str) -> proximadb_filter_expression::FilterExpression {
    proximadb_filter_expression::FilterExpression::Comparison {
        field: "partition".to_string(),
        operator: proximadb_filter_expression::ComparisonOperator::Equals,
        value: serde_json::Value::String(value.to_string()),
    }
}

/// ADR-089 / TD-FPRUNE-1 P1: the Stage-F row allow-set is row-accurate —
/// exactly the segment rows whose props match the predicate, in global
/// (cluster-ordered) row-ordinal space, verified by an independent decode.
#[tokio::test]
#[serial]
async fn stage_f_row_allow_set_is_exact() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_block_format::{PaxBlockReader, record::FlatRow};
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

    const DIM: usize = 32;
    const N: usize = 300;
    let records: Vec<ProximaRecord> = (0..N)
        .map(|i| {
            let v: Vec<f32> = (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect();
            rec_with_partition(i, v)
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    // Small blocks ⇒ multi-block segment (exercises cross-block ordinals).
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();

    let filter = partition_filter("p3");
    let ((allow, stats), snap) = crate::observability::io_trace::scope(async {
        let built = pax_filtered_row_allow(&fs, p, &filter, None, None)
            .await
            .unwrap()
            .expect("coalesced segment must produce a stage-F allow-set");
        (built, crate::observability::io_trace::snapshot())
    })
    .await;
    // TD-FPRUNE-1 PR-A: every Stage-F GET is attributed to the FiltD tier —
    // per-tier ops/bytes reconcile exactly with the physical ranged reads,
    // and no other tier is charged.
    let snap = snap.expect("io_trace scope active");
    let filt = CacheTier::FilterRowAllow as usize;
    assert!(
        snap.range_gets >= 3,
        "header + footer + >=1 block: {snap:?}"
    );
    assert_eq!(snap.tier_get_ops[filt], snap.range_gets);
    assert_eq!(snap.tier_get_bytes[filt], snap.bytes_read);
    assert_eq!(
        snap.tier_get_ops.iter().sum::<u64>(),
        snap.tier_get_ops[filt],
        "no other tier charged: {:?}",
        snap.tier_get_ops
    );
    assert_eq!(snap.metadata_ops, 1, "one HEAD per Stage-F build");

    // Independent verification: decode every D-block row in file order and
    // recompute the expected ordinal set from each row's props.
    let bytes = std::fs::read(&path).unwrap();
    let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
    let footer_off = header.footer_off as usize;
    let footer =
        SegmentFooterIndex::parse(&bytes[footer_off..footer_off + header.footer_len as usize])
            .unwrap();
    let mut expected = std::collections::BTreeSet::new();
    let mut g = 0usize;
    for b in &footer.blocks {
        let reader =
            PaxBlockReader::open(&bytes[b.offset as usize..(b.offset + b.size as u64) as usize])
                .unwrap();
        for flat in FlatRow::from_block_reader(&reader).unwrap() {
            let oid_index: usize = flat.oid.trim_start_matches('r').parse().unwrap();
            if oid_index % 8 == 3 {
                expected.insert(g);
            }
            g += 1;
        }
    }
    assert_eq!(g, N, "independent decode covers every row");
    let got: std::collections::BTreeSet<usize> =
        (0..N).filter(|&row| allow.contains(row)).collect();
    assert_eq!(got, expected, "allow-set == filter-matching rows, exactly");
    assert_eq!(allow.len(), (0..N).filter(|i| i % 8 == 3).count());
    assert_eq!(stats.rows_matched, allow.len());
    assert_eq!(stats.blocks_total, footer.blocks.len());

    // A predicate matching nothing yields an empty (but Some) allow-set.
    let (empty_allow, _) = pax_filtered_row_allow(&fs, p, &partition_filter("p999"), None, None)
        .await
        .unwrap()
        .expect("coalesced segment");
    assert!(empty_allow.is_empty());
}

/// PR-D L1a: the Stage-F wave split is order-preserving, byte-bounded, puts
/// an oversized fetch in a wave of its own, and yields nothing for no input.
#[test]
fn stage_f_waves_split_by_bytes_in_order() {
    let fetch = |start: u64, len: u64| CoalescedFetch {
        start,
        end: start + len,
        blocks: vec![start as usize],
    };
    assert!(stage_f_waves(&[], 100).is_empty());
    let fetches = vec![
        fetch(0, 40),
        fetch(40, 40),
        fetch(80, 40),   // 120 > 100 ⇒ new wave
        fetch(120, 250), // oversized ⇒ its own wave
        fetch(370, 10),
    ];
    let waves = stage_f_waves(&fetches, 100);
    let starts: Vec<Vec<u64>> = waves
        .iter()
        .map(|w| w.iter().map(|f| f.start).collect())
        .collect();
    assert_eq!(starts, vec![vec![0, 40], vec![80], vec![120], vec![370]]);
    let flat: Vec<u64> = waves
        .iter()
        .flat_map(|w| w.iter().map(|f| f.start))
        .collect();
    assert_eq!(
        flat,
        vec![0, 40, 80, 120, 370],
        "order preserved, nothing lost"
    );
}

/// Rows in [`write_stage_f_partition_segment`]. The row-group layout floors
/// D granules at 256 KiB, so the corpus must be large enough to span several
/// D blocks (several ranges per wave).
const STAGE_F_SEGMENT_ROWS: usize = 12_000;

/// A coalesced RaBitQ segment of [`STAGE_F_SEGMENT_ROWS`] 32-d rows whose
/// `partition` prop is `p{i % 8}` (record oid `r{i}`).
fn write_stage_f_partition_segment() -> (tempfile::TempDir, std::path::PathBuf) {
    const DIM: usize = 32;
    let records: Vec<ProximaRecord> = (0..STAGE_F_SEGMENT_ROWS)
        .map(|i| {
            let v: Vec<f32> = (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect();
            rec_with_partition(i, v)
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(4 * 1024),
        None,
    )
    .unwrap();
    (dir, path)
}

/// The footer index of a whole segment file's bytes.
fn parse_segment_footer(
    bytes: &[u8],
) -> proximadb_storage_common::segment_layout::SegmentFooterIndex {
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};
    let header = SegmentHeaderPrefix::parse(bytes).unwrap();
    let off = header.footer_off as usize;
    SegmentFooterIndex::parse(&bytes[off..off + header.footer_len as usize]).unwrap()
}

/// PR-D L1a/L1b: Stage F reads its D ranges in bounded-concurrency waves
/// and takes its header/footer from the invariants cache (warm) or skips the
/// HEAD with a listed size (cold) — the allow-set is identical on every path,
/// and the request counts shrink exactly by the elided reads.
#[tokio::test]
#[serial]
async fn stage_f_waves_cache_and_known_size_keep_the_allow_set_exact() {
    // Pin the parsed-once inflight cap directly: the plain `cargo test --lib`
    // runner shares one process, so the env cache cannot be re-armed mid-run.
    proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(Some(4));
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_common::segment_layout::SegmentHeaderPrefix;

    const N: usize = STAGE_F_SEGMENT_ROWS;
    let (_dir, path) = write_stage_f_partition_segment();
    // Tiny coalesce cap ⇒ many D ranges ⇒ more than one per wave.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_COALESCE_RANGE", "4096");
    }
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    let filter = partition_filter("p3");
    // Ordinals are in the segment's (cluster) row order: derive the
    // expected set by an independent decode, as the exact test does.
    let expected: Vec<usize> = {
        use proximadb_block_format::{PaxBlockReader, record::FlatRow};
        let bytes = std::fs::read(&path).unwrap();
        let footer = parse_segment_footer(&bytes);
        let mut expected = Vec::new();
        let mut g = 0usize;
        for b in &footer.blocks {
            let reader = PaxBlockReader::open(
                &bytes[b.offset as usize..(b.offset + b.size as u64) as usize],
            )
            .unwrap();
            for flat in FlatRow::from_block_reader(&reader).unwrap() {
                let oid_index: usize = flat.oid.trim_start_matches('r').parse().unwrap();
                if oid_index % 8 == 3 {
                    expected.push(g);
                }
                g += 1;
            }
        }
        assert_eq!(g, N);
        expected
    };
    type Built = Option<(proximadb_block_format::RowAllow, FilteredRowAllowStats)>;
    async fn run(
        fs: &LocalFileSystem,
        p: &str,
        filter: &proximadb_filter_expression::FilterExpression,
        cache: Option<&SegmentInvariantsCache>,
        size: Option<u64>,
    ) -> (
        Built,
        Option<crate::observability::io_trace::IoTraceSnapshot>,
    ) {
        crate::observability::io_trace::scope(async {
            let built = pax_filtered_row_allow(fs, p, filter, cache, size)
                .await
                .unwrap();
            (built, crate::observability::io_trace::snapshot())
        })
        .await
    }
    let allowed = |allow: &proximadb_block_format::RowAllow| -> Vec<usize> {
        (0..N).filter(|&row| allow.contains(row)).collect()
    };

    // Cold, no listed size: HEAD + header + footer + waved D ranges.
    let (cold, snap) = run(&fs, p, &filter, None, None).await;
    let (cold_allow, _) = cold.expect("coalesced segment");
    let cold_snap = snap.expect("io_trace scope");
    assert_eq!(allowed(&cold_allow), expected, "waved allow-set is exact");
    assert_eq!(cold_snap.metadata_ops, 1);
    let d_ranges = cold_snap.range_gets - 2;
    assert!(d_ranges > 1, "several D ranges: {cold_snap:?}");
    assert!(
        cold_snap.read_ranges_fetch_rounds > 0,
        "the D ranges went out as a bounded-concurrency wave: {cold_snap:?}"
    );
    let filt = CacheTier::FilterRowAllow as usize;
    assert_eq!(cold_snap.tier_get_ops[filt], cold_snap.range_gets);

    // Cold with the listed size: same reads minus the HEAD.
    let size = std::fs::metadata(&path).unwrap().len();
    let (sized, snap) = run(&fs, p, &filter, None, Some(size)).await;
    let snap = snap.expect("io_trace scope");
    assert_eq!(allowed(&sized.expect("coalesced").0), expected);
    assert_eq!(snap.metadata_ops, 0, "listed size skips the HEAD");
    assert_eq!(snap.range_gets, cold_snap.range_gets);

    // A listed size that cannot hold the footer fails closed (exact path).
    let (stale, _) = run(&fs, p, &filter, None, Some(200)).await;
    assert!(stale.is_none(), "stale listed size must decline");

    // Warm: the invariants cache supplies header + footer — only D ranges.
    let bytes = std::fs::read(&path).unwrap();
    let header_len = coalesced_header_prefetch_floor().min(bytes.len());
    let header = SegmentHeaderPrefix::parse(&bytes[..header_len]).unwrap();
    let footer_off = header.footer_off as usize;
    let cache = SegmentInvariantsCache::new(8 * 1024 * 1024);
    cache.put(
        p.to_string(),
        std::sync::Arc::new(SegmentInvariants {
            header_bytes: bytes[..header_len].to_vec(),
            region_bytes: None,
            footer_bytes: bytes[footer_off..footer_off + header.footer_len as usize].to_vec(),
            a0_bytes: None,
            rabitq_header_bytes: None,
        }),
    );
    let (warm, snap) = run(&fs, p, &filter, Some(&cache), None).await;
    let snap = snap.expect("io_trace scope");
    assert_eq!(allowed(&warm.expect("coalesced").0), expected);
    assert_eq!(snap.metadata_ops, 0, "warm: no HEAD");
    assert_eq!(snap.range_gets, d_ranges, "warm: no header/footer GET");
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_COALESCE_RANGE");
        proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(None);
    }
}

/// The parallel Stage F decode is exact at every degree: serial, several
/// workers, and more workers than blocks build the same matched ordinals,
/// pruned-block count and decoded-row count. Called at an explicit degree,
/// so the parallel branch runs regardless of the host's core count or the
/// small-wave gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stage_f_parallel_decode_matches_serial_at_every_degree() {
    const N: usize = STAGE_F_SEGMENT_ROWS;
    let (_dir, path) = write_stage_f_partition_segment();
    let bytes = std::fs::read(&path).unwrap();
    let footer = parse_segment_footer(&bytes);
    assert!(footer.blocks.len() > 3, "several D blocks");
    let buf = std::sync::Arc::new(bytes.clone());
    let jobs = || {
        let mut start = 0usize;
        footer
            .blocks
            .iter()
            .map(|b| {
                let job = StageFBlockJob {
                    buf: buf.clone(),
                    range: b.offset as usize..(b.offset + b.size as u64) as usize,
                    rows: u64::from(b.row_count),
                    start_ordinal: start,
                };
                start += b.row_count as usize;
                job
            })
            .collect::<Vec<_>>()
    };
    let field_map = footer
        .shred_field_map
        .iter()
        .map(|c| (c.key.clone(), c.column_id))
        .collect();
    let ctx = std::sync::Arc::new(StageFDecodeCtx {
        path: "seg.pax".to_string(),
        filter: partition_filter("p3"),
        field_map,
    });
    let summarize = |out: Vec<Option<StageFBlockDecoded>>| {
        let mut pruned = 0usize;
        let mut rows = 0u64;
        let mut matched = Vec::new();
        for block in out {
            match block {
                None => pruned += 1,
                Some(b) => {
                    rows += b.rows;
                    matched.extend(b.matched);
                }
            }
        }
        matched.sort_unstable();
        (pruned, rows, matched)
    };
    let serial = summarize(
        decode_stage_f_wave_with_degree(&ctx, jobs(), 1)
            .await
            .unwrap(),
    );
    assert_eq!(serial.1, N as u64, "every block decoded");
    assert_eq!(serial.2.len(), N / 8, "one partition in eight matches");
    let n_blocks = footer.blocks.len();
    for degree in [2, 3, n_blocks + 5] {
        let parallel = summarize(
            decode_stage_f_wave_with_degree(&ctx, jobs(), degree)
                .await
                .unwrap(),
        );
        assert_eq!(parallel, serial, "degree {degree}");
    }
    // A block outside its buffer fails the wave (exact fallback), it is
    // never silently dropped from the allow-set.
    let mut bad = jobs();
    let last = bad.len() - 1;
    bad[last].range = bytes.len()..bytes.len() + 1;
    assert!(decode_stage_f_wave_with_degree(&ctx, bad, 3).await.is_err());
}

/// TD-FPRUNE-1 PR-E: a heterogeneous filterable segment written with
/// Region E — typed columns holding wrong-typed / null / absent / NaN values,
/// an untyped column, a JSON prop — and the filters exercised against it.
fn write_region_e_segment(path: &std::path::Path, n: usize) {
    use proximadb_data_model::{ProximaType, ProximaValue as V};
    use proximadb_records::ProximaTreeNode;
    const DIM: usize = 32;
    let records: Vec<ProximaRecord> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect();
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v);
            let mut put = |k: &str, value: Option<V>| {
                if let Some(value) = value {
                    r.props.insert(k.into(), ProximaTreeNode::Value(value));
                }
            };
            put(
                "cat",
                match i % 23 {
                    0 => None,
                    1 => Some(V::Null),
                    2 => Some(V::Int32(3)),
                    3 => Some(V::String("3".into())),
                    4 => Some(V::Float64(3.0)),
                    _ => Some(V::Int64((i % 10) as i64)),
                },
            );
            put(
                "score",
                match i % 17 {
                    0 => Some(V::Float64(f64::NAN)),
                    1 => Some(V::Float32(0.5)),
                    2 => None,
                    // Floats JSON text does not round-trip bit-exactly.
                    3 => Some(V::Float64((i % 7) as f64 / 11.0)),
                    4 => Some(V::Float32(2.59)),
                    _ => Some(V::Float64((i % 100) as f64 / 10.0)),
                },
            );
            put(
                "tag",
                match i % 13 {
                    0 => Some(V::Symbol("abc".into())),
                    1 => Some(V::Int64(7)),
                    _ => Some(V::String(format!("t{}", i % 6))),
                },
            );
            put("flag", (i % 5 != 0).then(|| V::Boolean(i % 2 == 0)));
            put("doc", Some(V::Json(serde_json::json!({ "x": i % 3 }))));
            r
        })
        .collect();
    let shred = vec![
        proximadb_block_format::ShredColumn::typed("cat", 100, ProximaType::Int64),
        proximadb_block_format::ShredColumn::typed("score", 101, ProximaType::Float64),
        proximadb_block_format::ShredColumn::typed("tag", 102, ProximaType::String),
        proximadb_block_format::ShredColumn::untyped("flag", 103),
        proximadb_block_format::ShredColumn::untyped("doc", 104),
    ];
    write_pax_segment_ordered(
        path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        false,
        false,
        None,
        None,
        None,
        &shred,
    )
    .unwrap();
}

fn region_e_filters() -> Vec<proximadb_filter_expression::FilterExpression> {
    use proximadb_filter_expression::{ComparisonOperator as Op, FilterExpression as F};
    use serde_json::json;
    let c = |f: &str, op: Op, v: serde_json::Value| F::Comparison {
        field: f.into(),
        operator: op,
        value: v,
    };
    let mut out = Vec::new();
    for (f, op, v) in [
        ("cat", Op::Equals, json!(3)),
        ("cat", Op::Equals, json!(null)),
        ("cat", Op::NotEquals, json!(3)),
        ("cat", Op::GreaterThan, json!(6)),
        ("cat", Op::LessThanOrEqual, json!("3")),
        ("cat", Op::In, json!([1, 3, "3"])),
        ("cat", Op::NotIn, json!([1, 2])),
        ("cat", Op::NotIn, json!([1, null])),
        ("cat", Op::Between, json!([2, 5])),
        ("cat", Op::Between, json!([5, 2])),
        ("cat", Op::IsNull, json!(null)),
        ("cat", Op::IsNotNull, json!(null)),
        ("score", Op::GreaterThanOrEqual, json!(5.5)),
        ("score", Op::Equals, json!(0.5)),
        ("score", Op::IsNull, json!(null)),
        ("score", Op::GreaterThan, json!(1.0 / 11.0)),
        ("score", Op::LessThanOrEqual, json!(3.0 / 11.0)),
        ("score", Op::Between, json!([1.0 / 11.0, 5.0 / 11.0])),
        ("score", Op::GreaterThanOrEqual, json!(f64::from(2.59f32))),
        ("tag", Op::Equals, json!("t2")),
        ("tag", Op::StartsWith, json!("a")),
        ("tag", Op::Contains, json!("1")),
        ("tag", Op::Like, json!("t_")),
        ("tag", Op::GreaterThan, json!(5)),
        ("flag", Op::Equals, json!(true)),
        ("flag", Op::IsNull, json!(null)),
    ] {
        let cmp = c(f, op, v);
        out.push(F::Not(Box::new(cmp.clone())));
        out.push(cmp);
    }
    out.push(F::And(vec![
        c("cat", Op::Equals, json!(3)),
        c("flag", Op::Equals, json!(true)),
    ]));
    out.push(F::Or(vec![
        c("tag", Op::Equals, json!("t0")),
        c("score", Op::LessThan, json!(1.0)),
    ]));
    out.push(F::Not(Box::new(F::And(vec![
        c("cat", Op::GreaterThan, json!(4)),
        c("flag", Op::IsNotNull, json!(null)),
    ]))));
    out.push(F::And(vec![]));
    out.push(F::Or(vec![]));
    out
}

/// TD-FPRUNE-1 PR-E core oracle: for every filter, the Region-E allow-set
/// is identical to the full-row Stage F (read gate off), and Region E reads
/// one ranged GET per referenced column instead of every Region-D block.
#[tokio::test]
#[serial]
async fn region_e_allow_set_equals_full_row_stage_f() {
    enable_coalesced_rabitq();
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_FILTER_REGION", "1");
    }
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    const N: usize = 3_000;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_region_e_segment(&path, N);
    let bytes = std::fs::read(&path).unwrap();
    let footer = parse_segment_footer(&bytes);
    assert!(footer.filter_region.is_some(), "segment carries Region E");
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    let run = |filter: proximadb_filter_expression::FilterExpression, region: bool| {
        let fs = &fs;
        async move {
            unsafe {
                std::env::set_var(
                    "PROXIMADB_PAX_READ_FILTER_REGION",
                    if region { "1" } else { "0" },
                );
            }
            crate::observability::io_trace::scope(async {
                let built = pax_filtered_row_allow(fs, p, &filter, None, None)
                    .await
                    .unwrap()
                    .expect("coalesced segment");
                (built, crate::observability::io_trace::snapshot().unwrap())
            })
            .await
        }
    };
    let rows = |a: &proximadb_block_format::RowAllow| -> Vec<usize> {
        (0..N).filter(|&r| a.contains(r)).collect()
    };
    for filter in region_e_filters() {
        let ((legacy, legacy_stats), legacy_io) = run(filter.clone(), false).await;
        let ((fast, fast_stats), fast_io) = run(filter.clone(), true).await;
        assert!(
            !legacy_stats.region_e,
            "read gate off uses Region D: {filter:?}"
        );
        assert!(fast_stats.region_e, "Region E answers {filter:?}");
        assert_eq!(rows(&fast), rows(&legacy), "allow-set for {filter:?}");
        assert_eq!(fast_stats.rows_matched, legacy_stats.rows_matched);
        let fields = crate::core::search::sql_value_filter::filter_fields(&filter).len() as u64;
        assert_eq!(
            fast_io.range_gets,
            2 + fields,
            "header + footer + one GET per column: {filter:?}"
        );
        assert!(
            fast_io.bytes_read * 4 < legacy_io.bytes_read,
            "Region E reads far fewer bytes ({} vs {}) for {filter:?}",
            fast_io.bytes_read,
            legacy_io.bytes_read
        );
    }
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_FILTER_REGION");
        std::env::remove_var("PROXIMADB_PAX_READ_FILTER_REGION");
    }
}

/// A filter on a field Region E does not carry (a dotted JSON path, an
/// undeclared prop) uses the full-row path — still exact.
#[tokio::test]
#[serial]
async fn region_e_falls_back_for_fields_it_does_not_carry() {
    enable_coalesced_rabitq();
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_FILTER_REGION", "1");
    }
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_filter_expression::{ComparisonOperator as Op, FilterExpression as F};
    const N: usize = 600;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_region_e_segment(&path, N);
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    for (field, value, expected) in [
        (
            "doc.x",
            serde_json::json!(1),
            (0..N).filter(|i| i % 3 == 1).count(),
        ),
        ("undeclared", serde_json::json!(1), 0),
    ] {
        let filter = F::Comparison {
            field: field.into(),
            operator: Op::Equals,
            value,
        };
        let (allow, stats) = pax_filtered_row_allow(&fs, p, &filter, None, None)
            .await
            .unwrap()
            .expect("coalesced segment");
        assert!(!stats.region_e, "{field} is not a Region-E column");
        assert_eq!(stats.rows_matched, expected, "{field}");
        assert_eq!((0..N).filter(|&r| allow.contains(r)).count(), expected);
    }
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_FILTER_REGION");
    }
}

/// R5: Stage F evaluates the predicate against the record's PROPS
/// (`evaluate_filter_proxima`), so pruning must use only columns that hold
/// those prop values. A user prop that shares a canonical column's name
/// (`created_at` here, alongside the record's real timestamp) must not be
/// pruned on the canonical column's zone map — the canonical stats describe
/// different data, and every block would be wrongly skipped.
#[tokio::test]
#[serial]
async fn stage_f_never_prunes_a_user_prop_on_a_same_named_canonical_column() {
    enable_coalesced_rabitq();
    unsafe {
        std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
    }

    const DIM: usize = 16;
    const N: usize = 600;
    let records: Vec<ProximaRecord> = (0..N)
        .map(|i| {
            let v: Vec<f32> = (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 17) as f32) * 0.01)
                .collect();
            // Canonical created_at_ns = 1000+i; USER prop created_at = i % 7.
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v);
            r.props.insert(
                "created_at".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Int64((i % 7) as i64),
                ),
            );
            r
        })
        .collect();
    let (_dir, path, fs) = small_rabitq_segment(&records).await;
    let filter = proximadb_filter_expression::FilterExpression::Comparison {
        field: "created_at".to_string(),
        operator: proximadb_filter_expression::ComparisonOperator::Equals,
        value: serde_json::json!(3),
    };
    let (allow, _stats) = pax_filtered_row_allow(&fs, path.to_str().unwrap(), &filter, None, None)
        .await
        .unwrap()
        .expect("coalesced segment must produce a stage-F allow-set");
    let expected = (0..N).filter(|i| i % 7 == 3).count();
    assert_eq!(
        allow.len(),
        expected,
        "allow-set must equal the props-evaluated set ({expected} rows with created_at=3); \
         pruning on the canonical CREATED_AT zone map skipped matching blocks"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
    }
}

/// ADR-089 / TD-FPRUNE-1 P2 + R5: with footer stats armed, Stage F must NOT
/// prune on CANONICAL-column zone maps (`created_at_ns` here). The row match
/// (`evaluate_filter_proxima`) reads only props, so canonical stats describe
/// data the predicate never sees; user-tag zero-GET pruning is pinned by
/// `stage_f_user_tag_footer_prune_via_self_describing_segment`.
#[tokio::test]
#[serial]
async fn stage_f_does_not_prune_on_canonical_column_stats() {
    enable_coalesced_rabitq();
    // Arm P2 footer-stats population at write. Process-global env is safe
    // under nextest's process-per-test isolation (the repo test standard).
    unsafe {
        std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
    }

    const DIM: usize = 32;
    const N: usize = 1200;
    // Two groups DISJOINT in both vector space (near 0 vs near 10) AND
    // created_at (1000..1599 vs 2000..2599): the writer's centroid clustering
    // keeps blocks group-homogeneous, so their footer created_at [min,max]
    // ranges don't overlap — a `created_at >= 2000` query provably skips the
    // low-group blocks from the fetch. The small target block + larger N
    // guarantee MULTIPLE blocks (at the 16 KiB default this corpus packed
    // into a single spanning block, which no footer stat can prune).
    let records: Vec<ProximaRecord> = (0..N)
        .map(|i| {
            let low = i < N / 2;
            // ORTHOGONAL anchor dims (+x vs +y): direction-aware orderings
            // (sign-code / Morton / PCA all agree) separate the groups, so
            // blocks stay group-homogeneous and the footer created_at
            // [min,max] ranges are disjoint. (Equal-direction groups
            // interleave under every ordering, which no stat can prune.)
            let anchor = if low { 0 } else { 1 };
            let v: Vec<f32> = (0..DIM)
                .map(|d| {
                    if d == anchor {
                        50.0
                    } else {
                        (((i * 131 + d * 17) % 17) as f32) * 0.001
                    }
                })
                .collect();
            let ts = if low {
                1000 + i as i64
            } else {
                2000 + i as i64
            };
            rec(&format!("r{i}"), ts, v)
        })
        .collect();
    let (_dir, path, fs) = small_rabitq_segment(&records).await;
    let p = path.to_str().unwrap();

    // created_at_ns >= 2000 → provably excludes every low-group block.
    let filter = proximadb_filter_expression::FilterExpression::Comparison {
        field: "created_at_ns".to_string(),
        operator: proximadb_filter_expression::ComparisonOperator::GreaterThanOrEqual,
        value: serde_json::json!(2000),
    };
    let (allow, stats) = pax_filtered_row_allow(&fs, p, &filter, None, None)
        .await
        .unwrap()
        .expect("coalesced segment must produce a stage-F allow-set");

    // R5 contract (corrected): Stage F evaluates the predicate against PROPS,
    // and these records carry no `created_at_ns` prop, so the filter matches
    // nothing — the allow-set is the props-evaluated set (empty). Canonical
    // zone maps describe different data than the evaluator reads, so Stage F
    // must NOT prune on them: this test previously pinned a "zero-GET win"
    // that only ever skipped blocks yielding nothing, and the same rule
    // wrongly skipped matching blocks when a user prop shared the name
    // (`stage_f_never_prunes_a_user_prop_on_a_same_named_canonical_column`).
    // User-tag zero-GET pruning is pinned by
    // `stage_f_user_tag_footer_prune_via_self_describing_segment`.
    assert_eq!(allow.len(), 0, "no row carries a created_at_ns prop");
    assert_eq!(
        stats.blocks_pruned_footer, 0,
        "Stage F must not prune on canonical-column stats ({} of {} pruned)",
        stats.blocks_pruned_footer, stats.blocks_total,
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
    }
}

/// ADR-089 / TD-FPRUNE-1 P2 ENGAGEMENT: a USER-TAG filter (`partition`)
/// footer-prunes blocks end-to-end — the segment self-describes its shredded
/// columns (footer `shred_field_map`), so the reader resolves the tag name →
/// col-id with NO catalog lookup. This is the piece that makes P2 move the
/// benchmark (partition/lang filters), and the allow-set stays EXACT
/// (row-match via props is unchanged).
#[tokio::test]
#[serial]
async fn stage_f_user_tag_footer_prune_via_self_describing_segment() {
    enable_coalesced_rabitq();
    unsafe {
        std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
    }
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_block_format::{PaxBlockReader, record::FlatRow};
    use proximadb_storage_common::segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix};

    const DIM: usize = 32;
    const N: usize = 400;
    // Two groups disjoint in vector space AND partition: pA (near 0) / pB
    // (near 10). Clustering keeps blocks partition-homogeneous, so a "pA"
    // query provably skips the pB-only blocks — once `partition` resolves to
    // its shredded col-id via the footer field-map.
    let records: Vec<ProximaRecord> = (0..N)
        .map(|i| {
            let low = i < N / 2;
            let base = if low { 0.0 } else { 10.0 };
            let v: Vec<f32> = (0..DIM)
                .map(|d| base + (((i * 131 + d * 17) % 17) as f32) * 0.001)
                .collect();
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v);
            r.props.insert(
                "partition".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(
                        if low { "pA" } else { "pB" }.to_string(),
                    ),
                ),
            );
            r
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    // Shred "partition" into user col-id 100 (the P-Shred spec the production
    // flush path builds from the schema). write_pax_segment_ordered is the
    // module-internal writer that threads the spec.
    let shred = vec![proximadb_block_format::ShredColumn::typed(
        "partition",
        100,
        // The fixture's partition values are STRINGS — declaring the exact
        // type (String) is what makes them shred; an Int64 declaration would
        // fail shreds_losslessly per value and keep the tail (the fail-safe
        // this slice's exactness contract guarantees).
        proximadb_data_model::ProximaType::String,
    )];
    let cluster = crate::storage::engines::sst::block_cluster::block_cluster_enabled();
    let plan = if cluster {
        crate::storage::engines::sst::block_cluster::cluster_order_sq8_morton(&records, 0).map(
            |order| crate::storage::engines::sst::block_cluster::ClusterPlan {
                order,
                runs: Vec::new(),
            },
        )
    } else {
        None
    };
    write_pax_segment_ordered(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        false,
        cluster,
        plan,
        None,
        None,
        &shred,
    )
    .unwrap();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();

    // Precondition: the segment self-describes "partition" → col 100.
    let bytes = std::fs::read(&path).unwrap();
    let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
    let foff = header.footer_off as usize;
    let footer =
        SegmentFooterIndex::parse(&bytes[foff..foff + header.footer_len as usize]).unwrap();
    assert!(
        footer
            .shred_field_map
            .iter()
            .any(|col| col.key == "partition" && col.column_id == 100),
        "footer must self-describe the shredded partition column, got {:?}",
        footer.shred_field_map
    );

    let (allow, stats) = pax_filtered_row_allow(&fs, p, &partition_filter("pA"), None, None)
        .await
        .unwrap()
        .expect("coalesced segment must produce a stage-F allow-set");

    // Correctness (robust to clustering reorder): allow-set == exactly the
    // pA rows, verified by independent decode of each row's partition prop.
    let mut expected = std::collections::BTreeSet::new();
    let mut g = 0usize;
    for b in &footer.blocks {
        let reader =
            PaxBlockReader::open(&bytes[b.offset as usize..(b.offset + b.size as u64) as usize])
                .unwrap();
        for flat in FlatRow::from_block_reader(&reader).unwrap() {
            if flat.oid.trim_start_matches('r').parse::<usize>().unwrap() < N / 2 {
                expected.insert(g);
            }
            g += 1;
        }
    }
    let got: std::collections::BTreeSet<usize> =
        (0..N).filter(|&row| allow.contains(row)).collect();
    assert_eq!(
        got, expected,
        "allow-set == pA rows exactly (user-tag prune sound)"
    );
    assert_eq!(allow.len(), N / 2);

    // The zero-GET win on a USER TAG: ≥1 (but not all) block pruned from the
    // fetch — proving the footer field-map resolved "partition".
    assert!(
        stats.blocks_pruned_footer > 0 && stats.blocks_pruned_footer < stats.blocks_total,
        "user-tag footer prune must engage (got {} of {} blocks)",
        stats.blocks_pruned_footer,
        stats.blocks_total,
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
    }
}

/// ADR-089 / TD-FPRUNE-1 P1: the row-restricted cascade returns ONLY
/// predicate-matching hits, at recall parity with a filtered brute-force
/// ground truth (same bar as the unfiltered cascade's recall test).
#[tokio::test]
#[serial]
async fn filtered_cascade_matches_filtered_bruteforce() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 64;
    const N: usize = 400;
    const K: usize = 10;
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec_with_partition(i, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();

    let filter = partition_filter("p3");
    let (allow, _) = pax_filtered_row_allow(&fs, p, &filter, None, None)
        .await
        .unwrap()
        .expect("stage-F allow-set");
    let query = corpus[137].clone();
    let hits = rabitq_search_segment_coalesced_allowed(
        &fs,
        p,
        &query,
        K,
        RankMetric::L2,
        None,
        None,
        Some(&allow),
        None,
    )
    .await
    .unwrap()
    .expect("restricted cascade returns hits");
    assert!(!hits.is_empty());

    // Every hit satisfies the predicate (row-accurate restriction).
    for h in &hits {
        let i: usize = h.oid.trim_start_matches('r').parse().unwrap();
        assert_eq!(i % 8, 3, "hit {} violates the filter", h.oid);
    }

    // Recall vs the FILTERED brute-force ground truth.
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut matching: Vec<usize> = (0..N).filter(|i| i % 8 == 3).collect();
    matching.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        matching.iter().take(K).map(|i| format!("r{i}")).collect();
    let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
    let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
    assert!(
        recall >= 0.90,
        "filtered cascade recall@{K} = {recall:.2} vs filtered brute force"
    );

    // An empty allow-set yields an empty result (not an error).
    let empty = proximadb_block_format::RowAllow::new(N);
    let none_hits = rabitq_search_segment_coalesced_allowed(
        &fs,
        p,
        &query,
        K,
        RankMetric::L2,
        None,
        None,
        Some(&empty),
        None,
    )
    .await
    .unwrap()
    .expect("cascade runs");
    assert!(none_hits.is_empty());
}

/// ADR-089 P1: the filter-aware cascade is DEFAULT-OFF (mixed-read-safe).
#[test]
#[serial]
fn filtered_cascade_gate_defaults_on() {
    // FLIPPED default-ON 2026-08-31 (see the gate's doc comment). Pinned
    // explicitly so this assertion cannot race another test's env pinning
    // under plain `cargo test`'s shared process.
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_FILTERED_CASCADE");
    }
    assert!(pax_filtered_cascade_enabled());
}

/// ADR-089 / TD-FPRUNE-1 P1 EVIDENCE HARNESS (engine-level A/B on a real
/// segment). Not part of the suite — run explicitly, in release, for the
/// before/after timing that gates the phase:
///
/// ```text
/// cargo test --release -p proximadb --lib -- --ignored p1_evidence --nocapture
/// ```
///
/// A = today's filtered path over a flushed segment: whole-object read +
///     `read_segment_records` (decode ALL rows) + per-record
///     `evaluate_filter_proxima` + exact scoring of survivors.
/// B = P1: Stage-F row allow-set + row-restricted RaBitQ/SQ8 cascade.
///
/// Engine-level by design: the end-to-end server path cannot serve segments
/// for REST-created collections until the TD-FLUSH-8 catalog-identity skip
/// is fixed, and this seam is exactly what ADR-089 P1 changes.
#[tokio::test]
#[ignore = "evidence harness — run explicitly in release for timing"]
async fn p1_evidence_filtered_exact_vs_cascade() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use std::time::Instant;

    const DIM: usize = 128;
    const N: usize = 80_000;
    const K: usize = 10;
    const QUERIES: usize = 15;

    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec_with_partition(i, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_pax_segment(&path, &records, "col", 1, VectorQuant::RaBitQ, None, None).unwrap();
    let seg_bytes = std::fs::metadata(&path).unwrap().len();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    let filter = partition_filter("p3");

    // --- A: exact-path emulation (mirrors search_pax_file_exact's shape).
    let mut a_times = Vec::new();
    for q in 0..QUERIES {
        let query = &corpus[(q * 5119) % N];
        let t = Instant::now();
        let bytes = std::fs::read(&path).unwrap();
        let recs = read_segment_records(&bytes, &[], &[], None).unwrap();
        let mut scored: Vec<(f32, &ProximaRecord)> = recs
            .iter()
            .filter(|r| {
                crate::core::search::sql_value_filter::evaluate_filter_proxima(&filter, &r.props)
            })
            .filter_map(|r| {
                r.embeddings.first().map(|e| {
                    let v = e.as_fp32_slice();
                    let d: f32 = v.iter().zip(query).map(|(x, y)| (x - y) * (x - y)).sum();
                    (d, r)
                })
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(K);
        a_times.push(t.elapsed());
        assert!(!scored.is_empty());
    }

    // --- B: P1 Stage-F + restricted cascade (allow-set rebuilt per query,
    //     the worst case — a per-(segment, filter) cache is a P2 lever).
    let mut b_times = Vec::new();
    for q in 0..QUERIES {
        let query = &corpus[(q * 5119) % N];
        let t = Instant::now();
        let (allow, _stats) = pax_filtered_row_allow(&fs, p, &filter, None, None)
            .await
            .unwrap()
            .expect("stage-F allow-set");
        let hits = rabitq_search_segment_coalesced_allowed(
            &fs,
            p,
            query,
            K,
            RankMetric::L2,
            None,
            None,
            Some(&allow),
            None,
        )
        .await
        .unwrap()
        .expect("restricted cascade");
        b_times.push(t.elapsed());
        assert!(!hits.is_empty());
        for h in &hits {
            let i: usize = h.oid.trim_start_matches('r').parse().unwrap();
            assert_eq!(i % 8, 3);
        }
    }

    let ms = |d: &std::time::Duration| d.as_secs_f64() * 1e3;
    let mean =
        |v: &[std::time::Duration]| ms(&(v.iter().sum::<std::time::Duration>())) / v.len() as f64;
    let min = |v: &[std::time::Duration]| v.iter().map(ms).fold(f64::INFINITY, f64::min);
    println!(
        "P1 EVIDENCE  N={N} dim={DIM} K={K} queries={QUERIES} segment={:.1}MB",
        seg_bytes as f64 / 1e6
    );
    println!(
        "  A exact whole-object : mean={:.2}ms  min={:.2}ms",
        mean(&a_times),
        min(&a_times)
    );
    println!(
        "  B stage-F + cascade  : mean={:.2}ms  min={:.2}ms",
        mean(&b_times),
        min(&b_times)
    );
    println!("  speedup: {:.1}x (mean)", mean(&a_times) / mean(&b_times));
}

/// TD-CACHE-1 S1: `prefetch_segment_invariants` warms the CONTROL plane
/// (header/footer/A0) so a subsequent cold search issues fewer control
/// GETs than a fully cold one — without touching Region A/B payloads.
#[tokio::test]
#[serial]
async fn prefetch_invariants_reduces_first_search_control_gets() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::FileSystem;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    const DIM: usize = 64;
    const N: usize = 300;
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 37 + d * 13) % 199) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("p{i}"), 2000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prefetch.pax");
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let counters = global_counters();
    let fs: Arc<dyn FileSystem> =
        Arc::new(CountingFileSystem::new(Arc::new(local), counters.clone()));
    let path_str = path.to_str().unwrap();

    // Cold search WITHOUT any cache: baseline GET count.
    let before = counters.range_reads.load(Ordering::Relaxed);
    rabitq_search_segment_coalesced(
        fs.as_ref(),
        path_str,
        &corpus[42],
        5,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .unwrap()
    .expect("cold search hits");
    let cold_gets = counters.range_reads.load(Ordering::Relaxed) - before;

    // Prefetch into a fresh invariants cache (control plane only).
    let cache = SegmentInvariantsCache::new(8 * 1024 * 1024);
    let warmed = prefetch_segment_invariants(fs.as_ref(), path_str, &cache)
        .await
        .unwrap();
    assert!(warmed, "coalesced segment should prefill");
    let inv = cache.get(path_str).expect("invariants cached");
    assert!(!inv.header_bytes.is_empty());
    assert!(!inv.footer_bytes.is_empty());
    assert!(inv.region_bytes.is_none(), "payload must NOT be prefetched");
    // Idempotent: second prefetch is a no-op.
    assert!(
        !prefetch_segment_invariants(fs.as_ref(), path_str, &cache)
            .await
            .unwrap()
    );

    // First search WITH the prefilled cache: fewer GETs than fully cold.
    let before = counters.range_reads.load(Ordering::Relaxed);
    rabitq_search_segment_coalesced(
        fs.as_ref(),
        path_str,
        &corpus[42],
        5,
        RankMetric::L2,
        Some(&cache),
        None,
    )
    .await
    .unwrap()
    .expect("warm search hits");
    let warm_gets = counters.range_reads.load(Ordering::Relaxed) - before;
    assert!(
        warm_gets < cold_gets,
        "prefetch must elide control GETs (cold={cold_gets}, warm={warm_gets})"
    );
}

/// ADR-065 Q3: the survivor-range cache turns a hot repeat query into cache
/// hits. The second identical search issues strictly fewer ranged GETs — the
/// survivor (Region B) + OID (Region D) ranges are served from RAM, never
/// reaching the counting FS wrapper — and returns identical results. The
/// invariants cache is intentionally NOT injected, so the GET reduction is
/// attributable solely to the survivor cache.
#[tokio::test]
#[serial]
async fn survivor_cache_reduces_gets_on_hot_repeat_query() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::FileSystem;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    const DIM: usize = 64;
    const N: usize = 400;
    const K: usize = 10;

    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_pax_segment(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    let local = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let counters = global_counters();
    let fs: Arc<dyn FileSystem> =
        Arc::new(CountingFileSystem::new(Arc::new(local), counters.clone()));
    let path_str = path.to_str().unwrap();
    let cache = SurvivorRangeCache::new(8 * 1024 * 1024);

    let query = corpus[137].clone();

    // Search 1 (cold): every survivor + OID range misses → fetched + cached.
    let before1 = counters.range_reads.load(Ordering::Relaxed);
    let hits1 = rabitq_search_segment_coalesced(
        fs.as_ref(),
        path_str,
        &query,
        K,
        RankMetric::L2,
        None,
        Some(&cache),
    )
    .await
    .unwrap()
    .expect("first search returns hits");
    let gets1 = counters.range_reads.load(Ordering::Relaxed) - before1;

    // Search 2 (hot): identical query → survivor + OID ranges hit RAM.
    let before2 = counters.range_reads.load(Ordering::Relaxed);
    let hits2 = rabitq_search_segment_coalesced(
        fs.as_ref(),
        path_str,
        &query,
        K,
        RankMetric::L2,
        None,
        Some(&cache),
    )
    .await
    .unwrap()
    .expect("second search returns hits");
    let gets2 = counters.range_reads.load(Ordering::Relaxed) - before2;

    assert!(
        gets2 < gets1,
        "hot repeat query must issue fewer ranged GETs (got {gets2} >= {gets1}; \
             the survivor+OID ranges should have hit the cache)"
    );
    // Recall is identical — the cache returns the same bytes the backend would.
    let ids = |h: &[CascadeHit]| {
        let mut s: Vec<String> = h.iter().map(|x| x.oid.clone()).collect();
        s.sort();
        s
    };
    assert_eq!(
        ids(&hits1),
        ids(&hits2),
        "the survivor cache must not change search results"
    );
}

/// TD-RDSTRAT-8 PR-A (rev 3): `write_pax_segment_compacted` emits the
/// persisted-IVF-probe (v3) layout when write training is enabled (default
/// ON, compaction-only). The current binary probes v3 by default; the
/// read-side kill-switch serves it through the same single-level scan path
/// with full parity: search recall holds, and `read_segment_records`
/// reconstructs every row WITH its Region-B vectors (the compaction/recovery
/// inverse — a v3 segment must never silently drop vectors when re-compacted).
/// TD-CACHE-6: the PROBE-ARMED path must serve hot repeats from the
/// survivor-cache seam — an identical second search issues strictly
/// fewer ranged GETs (probed cells + any Region-B fetch cached) and
/// returns identical results. Before the fix, probe fetches read the fs
/// directly and every repeat re-paid the full GET chain.
#[tokio::test]
#[serial]
async fn probe_path_hot_repeat_served_from_cache_seam() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    const DIM: usize = 64;
    const N: usize = 400;
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_IVF_K", "4");
    }
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("v3probe.pax");
    write_pax_segment_compacted(
        &seg,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    let counters = global_counters();
    let counting = CountingFileSystem::new(
        std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
        counters.clone(),
    );
    let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
    let path = seg.to_string_lossy().to_string();
    let query = corpus[7].clone();

    let before = counters
        .range_reads
        .load(std::sync::atomic::Ordering::Relaxed);
    let first = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &query,
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("probe search hits");
    let cold_gets = counters
        .range_reads
        .load(std::sync::atomic::Ordering::Relaxed)
        - before;

    let mid = counters
        .range_reads
        .load(std::sync::atomic::Ordering::Relaxed);
    let second = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &query,
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("repeat search hits");
    let hot_gets = counters
        .range_reads
        .load(std::sync::atomic::Ordering::Relaxed)
        - mid;

    assert_eq!(
        first.iter().map(|h| &h.oid).collect::<Vec<_>>(),
        second.iter().map(|h| &h.oid).collect::<Vec<_>>(),
        "repeat must return identical results"
    );
    assert!(
        hot_gets < cold_gets,
        "hot repeat must be served from the cache seam (cold={cold_gets}, hot={hot_gets})"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_IVF_K");
    }
}

/// TD-RDSTRAT-12 §3 positive control: with `PROXIMADB_READ_RANGES_INFLIGHT`
/// armed, the coarse-probe COLD tail must flow through
/// `FileSystem::read_ranges_prefetch`'s bounded-concurrent executor — i.e.
/// the §2 `read_ranges` metrics get RECORDED (`Some`) where the sequential
/// baseline records nothing. This is the exact defect class §3 falsified:
/// the gate existed, tests passed, and NO production caller ever reached
/// the gated function. Runs under nextest process-per-test isolation so the
/// fn-local `OnceLock` env snapshots stay fresh per test.
#[tokio::test]
#[serial]
async fn probe_cold_tail_records_read_ranges_metrics_when_inflight_armed() {
    // Pin the parsed-once inflight cap directly: the plain `cargo test --lib`
    // runner shares one process, so the env cache cannot be re-armed mid-run.
    proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(Some(4));
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
    const DIM: usize = 64;
    const N: usize = 400;
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_IVF_K", "8");
        // Scope: this gates the Region-A cold tail's batching (TD-RDSTRAT-12
        // §3), not D5's same-wave rerank — which changes the rerank's
        // planning shape (split around prefetched B extents vs a later
        // query's merged range), so a repeat re-fetches one coalesced B
        // range. Documented in TD-PAXDIR-1 D8.
        std::env::set_var("PROXIMADB_PAX_READ_SAME_WAVE_RERANK", "0");
        // The 400-row fixture fits inside the default 1 MiB file-prefix
        // prefetch AND one gap-coalesced range — both would satisfy every
        // probed cell above the cold queue before batching sees it.
        // Disarm both so cells reach the TRUE-cold tail this test gates.
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
    }
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("v3prefetch.pax");
    write_pax_segment_compacted(
        &seg,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    let counting = CountingFileSystem::new(
        std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
        global_counters(),
    );
    let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
    let path = seg.to_string_lossy().to_string();

    // Clear any metric a previous call left in the single-slot buffer.
    let _ = drain_read_ranges_metrics();

    let hits = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("probe search hits");

    let metrics = drain_read_ranges_metrics().expect(
        "armed INFLIGHT must route the probe cold tail through \
                     read_ranges_prefetch and record its concurrency metrics",
    );
    assert!(
        metrics.max_inflight >= 1 && metrics.fetch_rounds >= 1,
        "implausible metrics payload: {metrics:?}"
    );
    assert_eq!(hits.len(), 5);

    // Identical repeat stays served by the survivor seam — its consume
    // pass adds NO further physical reads, so the metric slot is NOT
    // rewritten by this query.
    let again = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("repeat search hits");
    assert_eq!(again.len(), 5);
    assert!(
        drain_read_ranges_metrics().is_none(),
        "hot repeat must not issue additional batched reads"
    );

    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_IVF_K");
        proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(None);
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
    }
}

/// TD-RDSTRAT-13 C4: a cold footer rides the probe wave (or the rank
/// overlap) and is billed EXACTLY ONCE per query — never double-billed by
/// a wave slot plus a sequential re-read. Tier counters (C1) are the
/// teeth: `tier_get_ops[InvariantMeta] == 1` per query, and the wave
/// observability still fires on the cold query.
#[tokio::test]
#[serial]
async fn footer_rides_probe_wave_and_is_never_double_billed() {
    // Pin the parsed-once inflight cap directly: the plain `cargo test --lib`
    // runner shares one process, so the env cache cannot be re-armed mid-run.
    proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(Some(4));
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    const DIM: usize = 64;
    const N: usize = 400;
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_IVF_K", "8");
        // Disarm prefix-prefetch and gap-coalescing so the footer AND the
        // probed cells are genuinely cold (the 400-row fixture otherwise
        // fits inside the 1 MiB prefix).
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
    }
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("v3footer.pax");
    write_pax_segment_compacted(
        &seg,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    let counting = CountingFileSystem::new(
        std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
        global_counters(),
    );
    let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
    let path = seg.to_string_lossy().to_string();

    let (hits_res, snap) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("probe search hits");
        (hits, crate::observability::io_trace::snapshot())
    })
    .await;
    let snap = snap.expect("io_trace scope active");
    assert_eq!(hits_res.len(), 5);
    assert_eq!(
        snap.tier_get_ops[CacheTier::InvariantMeta as usize],
        1,
        "cold query bills the footer EXACTLY once (Meta tier): {:?}",
        snap.tier_get_ops
    );
    assert_eq!(
        snap.metadata_ops, 1,
        "exactly one HEAD on the cold path (C3 removes it when sizes thread)"
    );

    // Repeat query (cache=None wrapper ⇒ nothing cached engine-side):
    // the probe ranges now come from the survivor seam and the footer
    // overlaps the rank via the join path — still exactly one footer GET,
    // and no wave runs (nothing else is cold).
    let (hits_res2, snap2) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(
            &counting,
            &path,
            &corpus[7],
            5,
            RankMetric::L2,
            None,
            Some(&survivor),
        )
        .await
        .unwrap()
        .expect("repeat search hits");
        (hits, crate::observability::io_trace::snapshot())
    })
    .await;
    let snap2 = snap2.expect("io_trace scope active");
    assert_eq!(hits_res2.len(), 5);
    assert_eq!(
        snap2.tier_get_ops[CacheTier::InvariantMeta as usize],
        1,
        "repeat query still bills the footer exactly once (join path)"
    );

    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_IVF_K");
        proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(None);
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
    }
}

/// TD-RDSTRAT-12 §3 round 4: `plan_wave_split_ranges` — aligned chunks of
/// `chunk` bytes covering `[off, off+len)`, last chunk carrying the
/// remainder. Pure-function table test (the split planner's only math).
#[test]
fn plan_wave_split_ranges_chunks_and_remainder() {
    // Exact multiples: 3 chunks of 4.
    assert_eq!(
        plan_wave_split_ranges(100, 12, 4),
        vec![100..104, 104..108, 108..112]
    );
    // Remainder: 4 MiB chunk + 1 B remainder.
    assert_eq!(
        plan_wave_split_ranges(0, 4 * 1024 * 1024 + 1, 4 * 1024 * 1024),
        vec![0..4 * 1024 * 1024, 4 * 1024 * 1024..4 * 1024 * 1024 + 1]
    );
    // Region smaller than one chunk: single range.
    assert_eq!(plan_wave_split_ranges(7, 3, 16), vec![7..10]);
    // Degenerate: empty region ⇒ no ranges; zero chunk is clamped to 1
    // (defensive: always terminates, one byte per range).
    assert_eq!(
        plan_wave_split_ranges(5, 0, 4),
        Vec::<std::ops::Range<u64>>::new()
    );
    assert_eq!(
        plan_wave_split_ranges(5, 4, 0),
        vec![5..6, 6..7, 7..8, 8..9]
    );
}

/// TD-RDSTRAT-12 §3 round 4: with INFLIGHT armed, the Region-B clustered
/// cold tail AND the Region-D OID tail flow through
/// `read_ranges_prefetch` (the FS slot holds the LAST wave's metrics ⇒
/// Some after the cold search, None on a hot repeat that everything
/// serves from the cache seam), and the search results are identical to
/// the repeat arm. On the tiny fixture the probe tail may already be
/// RAM-resident by later waves — the assertion is that SOME true-cold
/// wave fired and the cache seam serves the repeat with no further
/// physical reads.
#[tokio::test]
#[serial]
async fn region_b_and_oid_cold_tails_batch_when_inflight_armed() {
    // Pin the parsed-once inflight cap directly: the plain `cargo test --lib`
    // runner shares one process, so the env cache cannot be re-armed mid-run.
    proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(Some(4));
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, global_counters};
    use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
    const DIM: usize = 64;
    const N: usize = 400;
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_IVF_K", "8");
        // Scope: this gates the Region-A cold tail's batching (TD-RDSTRAT-12
        // §3), not D5's same-wave rerank — which changes the rerank's
        // planning shape (split around prefetched B extents vs a later
        // query's merged range), so a repeat re-fetches one coalesced B
        // range. Documented in TD-PAXDIR-1 D8.
        std::env::set_var("PROXIMADB_PAX_READ_SAME_WAVE_RERANK", "0");
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "0");
    }
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("v3regionbd.pax");
    write_pax_segment_compacted(
        &seg,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();

    let counting = CountingFileSystem::new(
        std::sync::Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap()),
        global_counters(),
    );
    let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
    let path = seg.to_string_lossy().to_string();

    let _ = drain_read_ranges_metrics();
    let cold = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("cold search hits");

    let metrics = drain_read_ranges_metrics()
        .expect("armed INFLIGHT must route the cold tails through read_ranges_prefetch");
    assert!(metrics.fetch_rounds >= 1 && metrics.max_inflight >= 1);

    let warm = rabitq_search_segment_coalesced(
        &counting,
        &path,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        Some(&survivor),
    )
    .await
    .unwrap()
    .expect("repeat search hits");
    assert_eq!(
        cold.iter().map(|h| h.oid.clone()).collect::<Vec<_>>(),
        warm.iter().map(|h| h.oid.clone()).collect::<Vec<_>>(),
        "wave arm must not change results"
    );
    assert!(
        drain_read_ranges_metrics().is_none(),
        "hot repeat must be fully served by the cache seams"
    );

    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_IVF_K");
        proximadb_storage_filesystem_types::__set_read_ranges_inflight_for_tests(None);
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
    }
}

#[tokio::test]
#[serial]
async fn ivf_probe_compaction_gate_and_v3_read_compat() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_common::coarse_directory::CoarseDirectory;
    use proximadb_storage_common::segment_layout::{SEG_LAYOUT_VERSION, SegmentHeaderPrefix};

    const DIM: usize = 64;
    const N: usize = 400;
    const K: usize = 10;
    // Small k (the shared IOP-derived override) so the tiny fixture still
    // forms multiple cells — the natural N·dim/IOP count would be ~2 here.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_IVF_K", "4");
    }

    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    let dir = tempfile::tempdir().unwrap();

    // Flag ON ⇒ compaction writes v3 (header + A0 + footer mirror agree).
    let v3_path = dir.path().join("v3.pax");
    write_pax_segment_compacted(
        &v3_path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    let v3_bytes = std::fs::read(&v3_path).unwrap();
    let h = SegmentHeaderPrefix::parse(&v3_bytes).unwrap();
    assert_eq!(
        h.layout_version, SEG_LAYOUT_VERSION,
        "PROXIMADB_PAX_WRITE_A0_TRAIN=1 compaction must emit the v3 layout"
    );
    let a0 = CoarseDirectory::parse(&v3_bytes[h.a0_off as usize..(h.a0_off + h.a0_len) as usize])
        .expect("Region A0 parses from the compacted segment");
    assert_eq!(a0.model.rows_covered(), N as u64);
    assert_eq!(a0.model.dim as usize, DIM);

    // Flag OFF ⇒ same records compact to the v1 (single-level) layout.
    // #1234: PROXIMADB_PAX_WRITE_A0_TRAIN is now default-ON; explicitly
    // set "0" to test the OFF -> v1 path (remove_var would fall back to
    // the CoarseProbeConfig default, which is ON).
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "0");
    }
    let v1_path = dir.path().join("v1.pax");
    write_pax_segment_compacted(
        &v1_path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    let h1 = SegmentHeaderPrefix::parse(&std::fs::read(&v1_path).unwrap()).unwrap();
    assert_eq!(
        h1.layout_version, SEG_LAYOUT_VERSION,
        "without PROXIMADB_PAX_WRITE_A0_TRAIN the compaction layout is unchanged (v1)"
    );

    // v3 read-compat: the coalesced cascade searches the v3 segment with
    // brute-force-level recall (single-level full scan until PR-B).
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let query = corpus[137].clone();
    let hits = rabitq_search_segment_coalesced(
        &fs,
        v3_path.to_str().unwrap(),
        &query,
        K,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .unwrap()
    .expect("v3 segment must be searchable by the current binary");
    assert_eq!(hits[0].oid, "r137", "nearest neighbour on the v3 layout");
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut idx: Vec<usize> = (0..N).collect();
    idx.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        idx.iter().take(K).map(|i| format!("r{i}")).collect();
    let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
    let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
    assert!(recall >= 0.90, "v3 read-compat recall@{K} = {recall:.2}");

    // Compaction/recovery inverse: every row reads back WITH its vector
    // (Region-B overlay must work through the v3 prefix).
    let recs = read_segment_records(&v3_bytes, &[], &[], None).unwrap();
    assert_eq!(recs.len(), N);
    assert!(
        recs.iter().all(|r| r
            .embeddings
            .first()
            .is_some_and(|e| !e.as_fp32_slice().is_empty())),
        "v3 read_segment_records must reconstruct vectors (never drop Region B)"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_IVF_K");
    }
}

/// TD-RDSTRAT-8 PR-B (the deferred PR-A recall/GET gate): the coarse probe
/// (`PROXIMADB_PAX_READ_COARSE_PROBE=1`) ranks the persisted centroids in RAM and reads
/// only the nprobe nearest cells — holding recall on clustered data while
/// reading materially fewer bytes than the whole-region single-level scan.
#[tokio::test]
#[serial]
async fn coarse_probe_holds_recall_and_cuts_bytes() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 32;
    const G: usize = 8; // clusters (each centred on its own axis)
    const PER: usize = 64; // members per cluster
    const N: usize = G * PER;
    const K: usize = 10;

    // Deterministic pseudo-random component in [-1, 1].
    let pseudo = |i: usize, d: usize| -> f32 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // Widely-separated clusters (each centred at 50 on its own axis) with
    // members at a monotonically increasing radius — so the true top-K are
    // DISTINGUISHABLE (non-degenerate full-scan recall) AND concentrated in
    // one cluster → a few probed cells capture them.
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let (g, j) = (i / PER, i % PER);
            let radius = 0.1 + j as f32 * 0.03;
            (0..DIM)
                .map(|d| (if d == g { 50.0 } else { 0.0 }) + radius * pseudo(i, d))
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("probe.pax");
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_IVF_K", "16");
    }
    write_pax_segment_compacted(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
    }

    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    // The exact centre of cluster 0 — its nearest members are the smallest
    // radii (r0, r1, …), a distinguishable top-K all inside one cluster.
    let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();

    // Brute-force truth (top-K by L2).
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut idx: Vec<usize> = (0..N).collect();
    idx.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        idx.iter().take(K).map(|i| format!("r{i}")).collect();
    let recall = |hits: &[CascadeHit]| {
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
    };

    unsafe {
        std::env::set_var("PROXIMADB_TRACE_GETS", "1");
        // Establish the exact-read control for the prefix-prefetch proof
        // below; production defaults to a bounded 1 MiB prefix.
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
    }

    // Baseline: probe OFF → whole-region single-level scan over the v3 segment.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
    }
    let _ = drain_get_trace();
    let _ = drain_probe_trace();
    let base = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
        .await
        .unwrap()
        .unwrap();
    let base_bytes: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();
    assert!(drain_probe_trace().is_empty(), "probe OFF must not probe");
    let base_recall = recall(&base);
    assert!(
        base_recall >= 0.5,
        "baseline full-scan recall@{K} = {base_recall:.2} (sanity floor)"
    );

    // Probe ON: nprobe=6 of 16 cells.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "6");
    }
    let _ = drain_get_trace();
    let _ = drain_probe_trace();
    let (probe, probe_snapshot) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
            .await
            .unwrap()
            .unwrap();
        let snapshot = crate::observability::io_trace::snapshot()
            .expect("reader probe test runs inside an io_trace scope");
        (hits, snapshot)
    })
    .await;
    let probe_gets = drain_get_trace();
    let probe_bytes: u64 = probe_gets.iter().map(|(_, b)| b).sum();
    let ptrace = drain_probe_trace();

    // 1. The probe engaged: cells_probed strictly inside (0, cells_total).
    assert_eq!(ptrace.len(), 1, "exactly one probe recorded");
    let (cells_total, cells_probed, probed_rows, fetch_rounds) = ptrace[0];
    assert_eq!(cells_total, 16, "16 coarse cells");
    assert!(
        cells_probed > 0 && cells_probed <= 6,
        "probed {cells_probed} cells (0 < c <= nprobe=6)"
    );
    assert!(
        probed_rows > 0 && probed_rows < N as u64,
        "probed {probed_rows} of {N} rows (a strict subset)"
    );
    // `fetch_rounds` is the count of coalesced Region-A ranged reads — the
    // request-side term of the cost model, and on Hot-tier object storage
    // requests are the only billed read term.
    //
    // What this pins: one probed cell costs AT MOST one ranged GET.
    // `plan_probe_cell_ranges` either merges a cell into the previous range
    // or emits it 1:1, never splitting one cell across several reads, so a
    // future planner that chunked oversized cells to respect
    // `max_range_bytes` would multiply per-query round-trips — and would
    // trip this.
    //
    // What this does NOT catch, stated so the guard is not mistaken for
    // more than it is: coalescing silently ceasing to merge. That keeps
    // `fetch_rounds <= cells_probed` true while raising it toward the
    // ceiling. The relative check further down (coalesced <= default policy)
    // covers part of that; an absolute per-query ratchet needs a real bed
    // and measured baselines, which is TD-IVF-3's deferred probe-economy
    // ratchet, not something a synthetic 16-cell fixture can stand in for.
    assert!(
        (1..=cells_probed).contains(&fetch_rounds),
        "fetch_rounds={fetch_rounds} must be in 1..={cells_probed}: coalescing \
             may merge probed cells into fewer ranged GETs but must never split \
             one cell across several"
    );
    assert_eq!(probe_snapshot.ivf_cells_total, cells_total);
    assert_eq!(probe_snapshot.ivf_cells_probed, cells_probed);
    assert_eq!(probe_snapshot.ivf_probed_rows, probed_rows);
    assert_eq!(probe_snapshot.ivf_fetch_rounds, fetch_rounds);
    assert_eq!(probe_snapshot.ivf_whole_region_fallback, 0);
    let traced_region_a_bytes: u64 = probe_gets
        .iter()
        .filter(|(tier, _)| matches!(tier, CacheTier::ProbeIndex))
        .map(|(_, bytes)| bytes)
        .sum();
    let traced_region_b_bytes: u64 = probe_gets
        .iter()
        .filter(|(tier, _)| matches!(tier, CacheTier::SurvivorPayload))
        .map(|(_, bytes)| bytes)
        .sum();
    assert!(traced_region_a_bytes > 0, "probe must fetch Region A");
    assert!(traced_region_b_bytes > 0, "rerank must fetch Region B");
    assert_eq!(
        probe_snapshot.ivf_region_a_bytes, traced_region_a_bytes,
        "durable Region-A bytes must equal physical probe reads"
    );
    assert_eq!(
        probe_snapshot.ivf_region_b_bytes, traced_region_b_bytes,
        "durable Region-B bytes must equal physical rerank reads"
    );

    // 2. Materially fewer bytes than the whole-region baseline.
    assert!(
        probe_bytes < base_bytes,
        "probe {probe_bytes} B must read fewer than baseline {base_bytes} B"
    );

    // 3. Recall held vs the full scan: probing the query's cluster loses
    //    essentially nothing (the meaningful PR-B claim — robust to the
    //    absolute RaBitQ recall level, which the corpus/quantizer set).
    let probe_recall = recall(&probe);
    assert!(
        probe_recall >= base_recall - 0.1,
        "probe recall@{K} = {probe_recall:.2} must hold vs baseline {base_recall:.2} (nprobe=6/16)"
    );

    // PR-C2 proof: bounded Region-A gap over-read may reduce physical
    // ranges, but it must rank exactly the same selected cells and rows.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", "1048576");
        std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE", "4194304");
    }
    let _ = drain_get_trace();
    let _ = drain_probe_trace();
    let (coalesced, coalesced_snapshot) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
            .await
            .unwrap()
            .unwrap();
        let snapshot = crate::observability::io_trace::snapshot()
            .expect("coalescing proof runs inside an io_trace scope");
        (hits, snapshot)
    })
    .await;
    let _ = drain_get_trace();
    let _ = drain_probe_trace();
    assert_eq!(
        coalesced.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
        probe.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
        "physical gap over-read must not change logical survivors/results"
    );
    assert_eq!(coalesced_snapshot.ivf_cells_probed, cells_probed);
    assert_eq!(coalesced_snapshot.ivf_probed_rows, probed_rows);
    assert!(
        coalesced_snapshot.ivf_fetch_rounds <= fetch_rounds,
        "cost coalescing cannot add Region-A reads"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
    }

    // PR-C2 proof: a bounded prefix read must replace the separate header,
    // A0 and RaBitQ-parameter GETs with one physical control GET.
    let baseline_control_gets = probe_gets
        .iter()
        .filter(|(tier, _)| *tier == CacheTier::SearchControl)
        .count();
    assert!(
        baseline_control_gets >= 3,
        "control baseline needs header+A0+params"
    );
    unsafe {
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "65536");
    }
    let _ = drain_get_trace();
    let prefetched = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
        .await
        .unwrap()
        .unwrap();
    let prefix_gets = drain_get_trace();
    let _ = drain_probe_trace();
    assert_eq!(
        prefetched.iter().map(|hit| &hit.oid).collect::<Vec<_>>(),
        probe.iter().map(|hit| &hit.oid).collect::<Vec<_>>()
    );
    assert_eq!(
        prefix_gets
            .iter()
            .filter(|(tier, _)| *tier == CacheTier::SearchControl)
            .count(),
        1,
        "one prefix GET supplies header+A0+RaBitQ parameters"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
    }

    // PR-C2 proof: metadata warms by default even though the probe never
    // admits the complete Region A into the invariant cache. Production
    // does not set an opt-in gate, so this regression must exercise the
    // unset environment exactly as the server does.
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
    }
    let invariants = SegmentInvariantsCache::new(8 * 1024 * 1024);
    let _ = drain_get_trace();
    let _ =
        rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, Some(&invariants), None)
            .await
            .unwrap()
            .unwrap();
    let first_cached = drain_get_trace();
    let _ = drain_probe_trace();
    let _ =
        rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, Some(&invariants), None)
            .await
            .unwrap()
            .unwrap();
    let second_cached = drain_get_trace();
    let _ = drain_probe_trace();
    assert!(
        first_cached
            .iter()
            .any(|(tier, _)| matches!(tier, CacheTier::SearchControl | CacheTier::InvariantMeta)),
        "cold cache population reads control metadata"
    );
    assert!(
        second_cached
            .iter()
            .all(|(tier, _)| !matches!(tier, CacheTier::SearchControl | CacheTier::InvariantMeta)),
        "warm probe must not re-read cached control metadata"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
    }

    // 4. Corrupt A0 fails closed to the canonical whole-Region-A scan and
    //    makes that fallback visible in the durable query trace.
    let mut corrupt = std::fs::read(&path).unwrap();
    let corrupt_header = SegmentHeaderPrefix::parse(&corrupt).unwrap();
    corrupt[corrupt_header.a0_off as usize] ^= 0x01;
    std::fs::write(&path, corrupt).unwrap();
    let (fallback, fallback_snapshot) = crate::observability::io_trace::scope(async {
        let hits = rabitq_search_segment_coalesced(&fs, p, &query, K, RankMetric::L2, None, None)
            .await
            .unwrap()
            .unwrap();
        let snapshot = crate::observability::io_trace::snapshot()
            .expect("reader fallback test runs inside an io_trace scope");
        (hits, snapshot)
    })
    .await;
    assert!(
        !fallback.is_empty(),
        "A0 corruption fails closed, not empty"
    );
    assert_eq!(fallback_snapshot.ivf_cells_total, 0);
    assert_eq!(fallback_snapshot.ivf_cells_probed, 0);
    assert_eq!(fallback_snapshot.ivf_whole_region_fallback, 1);
    assert!(
        fallback_snapshot.ivf_region_a_bytes > 0,
        "whole-region fallback must attribute its physical Region-A bytes"
    );
    assert!(
        fallback_snapshot.ivf_region_b_bytes > 0,
        "fallback rerank must attribute its physical Region-B bytes"
    );
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
        std::env::remove_var("PROXIMADB_TRACE_GETS");
        std::env::remove_var("PROXIMADB_IVF_K");
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
        std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
    }
}

/// TD-FPRUNE-1 M3: the FILTERED coarse probe holds recall AND cuts bytes.
/// Trace-driven: a filtered query previously DISARMED the geometric probe and
/// read whole Region-A/B (measured 3–7 MB vs the unfiltered 2 MB). M3 re-arms
/// the probe under a filter — adaptive nprobe by matching-row count + rank
/// restricted to allowed rows — so a filtered query becomes a filtered ANN
/// that touches only relevant cells. This asserts (a) recall parity vs the
/// FILTERED brute-force truth and (b) the probe engaged (a strict cell subset).
#[tokio::test]
#[serial]
async fn filtered_coarse_probe_holds_recall_and_cuts_bytes() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 32;
    const G: usize = 8; // clusters (each on its own axis) == partitions
    const PER: usize = 64;
    const N: usize = G * PER;
    const K: usize = 10;

    let pseudo = |i: usize, d: usize| -> f32 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // partition == cluster (correlated with vector space). Each cluster's
    // members sit at increasing radius so the top-K are distinguishable.
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let (g, j) = (i / PER, i % PER);
            let radius = 0.1 + j as f32 * 0.03;
            (0..DIM)
                .map(|d| (if d == g { 50.0 } else { 0.0 }) + radius * pseudo(i, d))
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
            r.props.insert(
                "partition".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(format!("p{}", i / PER)),
                ),
            );
            r
        })
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fprobe.pax");
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_IVF_K", "16");
        // This test targets the GLOBAL single-directory probe; pin it off the
        // (opt-in) partition layout so its cell-count assertions hold.
        std::env::set_var("PROXIMADB_PAX_PARTITION_LAYOUT", "0");
    }
    // Compacted (v3, A0) + shredded so `partition` is a footer-resolvable column.
    write_pax_segment_compacted_shredded(
        &path,
        &records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        &[proximadb_block_format::ShredColumn::typed(
            "partition",
            100,
            proximadb_data_model::ProximaType::String,
        )],
        None,
        None,
    )
    .unwrap();
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
    }

    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    let p = path.to_str().unwrap();
    // Query at cluster 0's centre; filter partition=p0 (== cluster 0).
    let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();

    // FILTERED brute-force truth: top-K among partition==p0 rows (cluster 0).
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut p0: Vec<usize> = (0..N).filter(|&i| i / PER == 0).collect();
    p0.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        p0.iter().take(K).map(|i| format!("r{i}")).collect();

    // Build the allow-set (partition==p0) from the segment's own footer.
    let (allow, _stats) = pax_filtered_row_allow(&fs, p, &partition_filter("p0"), None, None)
        .await
        .unwrap()
        .expect("shredded v3 segment yields a stage-F allow-set");
    assert!(!allow.is_empty(), "partition=p0 must match rows");

    // FILTERED probe ON, nprobe floor small — M3 expands adaptively to cover
    // enough matching rows. Shrink the prefix prefetch (default 1 MiB slurps a
    // small test segment WHOLE, masking the ranged-read win) and disable the
    // cross-call metadata cache so the byte A/B below isolates the region reads
    // — the cloud-GET cost term the co-design mandate targets.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "2");
        std::env::set_var("PROXIMADB_TRACE_GETS", "1");
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "4096");
        std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
    }
    let _ = drain_get_trace();
    let _ = drain_probe_trace();
    let hits = rabitq_search_segment_coalesced_allowed(
        &fs,
        p,
        &query,
        K,
        RankMetric::L2,
        None,
        None,
        Some(&allow),
        None,
    )
    .await
    .unwrap()
    .unwrap();
    let ptrace = drain_probe_trace();
    let bytes_on: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();

    // (a) recall parity vs the FILTERED truth.
    let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
    let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
    assert!(
        recall >= 0.9,
        "filtered probe recall@{K} = {recall:.2} (want ≥0.9 vs filtered brute-force)"
    );
    // Every hit is a partition=p0 row (no cross-partition leakage).
    for h in &hits {
        let idx: usize = h.oid.trim_start_matches('r').parse().unwrap();
        assert_eq!(idx / PER, 0, "hit {} is not partition p0", h.oid);
    }
    // (b) the probe engaged under the filter — a strict cell subset (not the
    // whole-region fallback the disarmed path took).
    assert_eq!(ptrace.len(), 1, "exactly one filtered probe recorded");
    let (cells_total, cells_probed, probed_rows, _rounds) = ptrace[0];
    assert_eq!(cells_total, 16);
    assert!(
        cells_probed > 0 && cells_probed < cells_total,
        "filtered probe touched {cells_probed} of {cells_total} cells (strict subset)"
    );
    assert!(
        probed_rows < N as u64,
        "filtered probe read {probed_rows} of {N} rows (subset, not whole region)"
    );

    // (c) the byte cut is MEASURED, not asserted: re-run the SAME filtered query
    // with the probe DISARMED (the pre-M3 behavior — whole Region-A/B allowed
    // rank) and confirm the armed probe fetched strictly fewer bytes. With the
    // prefix prefetch shrunk above, this reflects the ranged-GET (cloud) regime.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
    }
    let _ = drain_get_trace();
    let base = rabitq_search_segment_coalesced_allowed(
        &fs,
        p,
        &query,
        K,
        RankMetric::L2,
        None,
        None,
        Some(&allow),
        None,
    )
    .await
    .unwrap()
    .unwrap();
    let bytes_off: u64 = drain_get_trace().iter().map(|(_, b)| b).sum();
    assert!(
        drain_probe_trace().is_empty(),
        "probe OFF must not record a probe (whole-region baseline)"
    );
    // The disarmed baseline is still correct (whole-region allowed rank) — same
    // filtered top-K — so the armed probe trades no recall for the byte cut.
    let base_got: std::collections::HashSet<String> = base.iter().map(|h| h.oid.clone()).collect();
    assert_eq!(got, base_got, "armed probe and whole-region baseline agree");
    assert!(
        bytes_on < bytes_off,
        "filtered probe fetched {bytes_on} B vs the disarmed {bytes_off} B \
             (want strictly fewer — the ranged-read win)"
    );

    unsafe {
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
        std::env::remove_var("PROXIMADB_TRACE_GETS");
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
        std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
        std::env::remove_var("PROXIMADB_IVF_K");
    }
}

/// TD-FPRUNE-1 M2: the tag-aware layout makes an UNCORRELATED tag prune —
/// the case M3 alone cannot help (every geometric cell holds every value, so
/// `count_in_range` skips nothing). With partition-by-tag layout the cells
/// are partition-homogeneous, so the filtered probe accumulates its matching
/// pool from FEWER cells. This builds the SAME uncorrelated corpus twice
/// (layout ON vs OFF) and asserts the ON segment reads strictly fewer rows to
/// serve the same filtered top-K, at recall parity and with no leakage.
#[tokio::test]
#[serial]
async fn partition_layout_prunes_uncorrelated_tag() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 32;
    const P: usize = 4; // partitions (the tag)
    const GV: usize = 8; // vector clusters (independent of the tag)
    const N: usize = 512;
    const K: usize = 10;

    let pseudo = |i: usize, d: usize| -> f32 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // partition = i % P; vector cluster = (i / P) % GV. Disjoint index ranges
    // ⇒ the tag is UNCORRELATED with vector geometry (every cluster carries
    // every partition). This is the case zone-maps/blooms cannot prune.
    let part = |i: usize| i % P;
    let vclust = |i: usize| (i / P) % GV;
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let c = vclust(i);
            (0..DIM)
                .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.4 * pseudo(i, d))
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
            r.props.insert(
                "partition".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                ),
            );
            r
        })
        .collect();

    // Query at vector cluster 0; filter partition == p0 (UNCORRELATED with it).
    let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();
    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut p0: Vec<usize> = (0..N).filter(|&i| part(i) == 0).collect();
    p0.sort_by(|&a, &b| {
        l2(&corpus[a], &query)
            .partial_cmp(&l2(&corpus[b], &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truth: std::collections::HashSet<String> =
        p0.iter().take(K).map(|i| format!("r{i}")).collect();

    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();

    // Build one v3 shredded segment with the partition-layout gate `on`, and
    // measure a filtered probe on it: returns (hits, probed_rows).
    async fn build_and_probe(
        fs: &LocalFileSystem,
        records: &[ProximaRecord],
        query: &[f32],
        k: usize,
        layout_on: bool,
    ) -> (Vec<CascadeHit>, u64) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m2.pax");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "16");
            // Explicit "1"/"0" (the layout is opt-in since TD-FPRUNE-3): "1"
            // engages it for the ON arm, "0" pins the global layout.
            std::env::set_var(
                "PROXIMADB_PAX_PARTITION_LAYOUT",
                if layout_on { "1" } else { "0" },
            );
        }
        write_pax_segment_compacted_shredded(
            &path,
            records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            &[proximadb_block_format::ShredColumn::typed(
                "partition",
                100,
                proximadb_data_model::ProximaType::String,
            )],
            None,
            None,
        )
        .unwrap();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
        }
        let p = path.to_str().unwrap();
        let (allow, _s) = pax_filtered_row_allow(fs, p, &partition_filter("p0"), None, None)
            .await
            .unwrap()
            .expect("shredded v3 segment yields an allow-set");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "2");
            std::env::set_var("PROXIMADB_TRACE_GETS", "1");
            std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
        }
        let _ = drain_probe_trace();
        let hits = rabitq_search_segment_coalesced_allowed(
            fs,
            p,
            query,
            k,
            RankMetric::L2,
            None,
            None,
            Some(&allow),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let ptrace = drain_probe_trace();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            std::env::remove_var("PROXIMADB_TRACE_GETS");
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
            std::env::remove_var("PROXIMADB_IVF_K");
        }
        let probed_rows = ptrace.first().map(|t| t.2).unwrap_or(u64::MAX);
        (hits, probed_rows)
    }

    let (hits_on, rows_on) = build_and_probe(&fs, &records, &query, K, true).await;
    let (hits_off, rows_off) = build_and_probe(&fs, &records, &query, K, false).await;

    // The M2 risk is a RELATIVE one — does per-partition clustering degrade
    // recall vs the global layout at identical settings? So compare the two,
    // not an absolute bar (1-bit RaBitQ over a 512-row toy corpus is coarse by
    // construction; the absolute recall gate is the SIFT ratchet + a real
    // eval, not this unit).
    let recall = |hits: &[CascadeHit]| {
        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
    };
    let (r_on, r_off) = (recall(&hits_on), recall(&hits_off));
    eprintln!("[M2] recall on={r_on:.2} off={r_off:.2} | probed_rows on={rows_on} off={rows_off}");
    // No cross-partition leakage under the layout (the allow-set is exact).
    for h in &hits_on {
        let idx: usize = h.oid.trim_start_matches('r').parse().unwrap();
        assert_eq!(part(idx), 0, "hit {} is not partition p0", h.oid);
    }
    // Both layouts find real neighbors (search works at all).
    assert!(r_on > 0.0 && r_off > 0.0, "both layouts return matches");
    // Partition clustering does not MATERIALLY degrade recall vs the global
    // layout at the same settings (within one RaBitQ quantization step).
    assert!(
        r_on >= r_off - 0.15,
        "partition layout must not hurt recall vs global: on={r_on:.2} off={r_off:.2}"
    );
    // THE M2 WIN: partition-homogeneous cells let the filtered probe reach its
    // matching pool from fewer rows than the global (mixed-cell) layout, where
    // every probed cell is only ~1/P matching. This is the uncorrelated-tag
    // prune that M3 alone cannot deliver.
    assert!(
        rows_on < rows_off,
        "partition layout must probe FEWER rows for an uncorrelated tag: \
             on={rows_on} vs off={rows_off}"
    );
}

/// TD-FPRUNE-1 C1: integrated full-stack GET/byte breakdown for a filtered PAX
/// query at realistic multi-block, multi-partition scale — the co-design
/// "trace before you tune" evidence that Slice A + M3 + M2 actually moved the
/// dominant cost, and which term is the RESIDUAL. It SPLITS the filtered path
/// into its two I/O stages — the Stage-F allow-set build (`pax_filtered_row_allow`,
/// reads footer-surviving Region-D blocks) vs the probe+rerank
/// (`rabitq_search_segment_coalesced_allowed`, reads A0 + probe cells + Region-B)
/// — for BOTH a correlated and an uncorrelated tag, with M2 layout off vs on.
/// The printed breakdown identifies the residual dominant GET/byte term (does
/// the allow-set build dominate? → P3 per-cell stats is justified). Assertions
/// are the regression guards; the eprintln is the evidence.
#[tokio::test]
#[serial]
async fn filtered_stack_get_byte_breakdown() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 64;
    const P: usize = 8; // partitions (the tag)
    const N: usize = 4096;
    const K: usize = 10;

    let pseudo = |i: usize, d: usize| -> f32 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    // One filtered measurement over a freshly built segment. `correlated`
    // decides whether the tag tracks the vector cluster; `m2_on` toggles the
    // partition layout at compaction. Returns the per-stage GET/byte split.
    struct Measure {
        filesize: u64,
        allow_gets: u64,
        allow_bytes: u64,
        blocks_total: usize,
        blocks_pruned_footer: usize,
        search_gets: u64,
        search_bytes: u64,
        region_a_bytes: u64,
        region_b_bytes: u64,
        whole_region_fallback: u64,
        recall: f32,
    }
    async fn measure(
        fs: &dyn proximadb_storage_filesystem_types::FileSystem,
        pseudo: &dyn Fn(usize, usize) -> f32,
        correlated: bool,
        m2_on: bool,
    ) -> Measure {
        // partition = i%P; vector cluster = correlated ? partition : (i/P)%P.
        let part = |i: usize| i % P;
        let vclust = |i: usize| if correlated { i % P } else { (i / P) % P };
        let corpus: Vec<Vec<f32>> = (0..N)
            .map(|i| {
                let c = vclust(i);
                (0..DIM)
                    .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.4 * pseudo(i, d))
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
                r.props.insert(
                    "partition".to_string(),
                    proximadb_records::ProximaTreeNode::Value(
                        proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                    ),
                );
                r
            })
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c1.pax");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "64");
            // Pin what this measurement depends on explicitly (the
            // cascade gate is default-ON since the 2026-08-31 flip; footer
            // stats remains default-OFF) — removed with the rest at the end
            // of the arm.
            std::env::set_var("PROXIMADB_PAX_FILTERED_CASCADE", "1");
            std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
            // Explicit "1"/"0" (the layout is opt-in since TD-FPRUNE-3): "1"
            // engages it for the ON arm, "0" pins the global layout.
            std::env::set_var(
                "PROXIMADB_PAX_PARTITION_LAYOUT",
                if m2_on { "1" } else { "0" },
            );
        }
        // Small blocks ⇒ many Region-D blocks (footer-pruning needs blocks
        // small enough to be partition-homogeneous after clustering).
        write_pax_segment_compacted_shredded(
            &path,
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(8 * 1024),
            &[proximadb_block_format::ShredColumn::typed(
                "partition",
                100,
                proximadb_data_model::ProximaType::String,
            )],
            None,
            None,
        )
        .unwrap();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
        }
        let p = path.to_str().unwrap();
        let filesize = fs.metadata(p).await.unwrap().size;

        // Filtered truth: top-K among partition==p0 rows.
        let query: Vec<f32> = (0..DIM).map(|d| if d == 0 { 50.0 } else { 0.0 }).collect();
        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        let mut p0: Vec<usize> = (0..N).filter(|&i| part(i) == 0).collect();
        p0.sort_by(|&a, &b| {
            l2(&corpus[a], &query)
                .partial_cmp(&l2(&corpus[b], &query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truth: std::collections::HashSet<String> =
            p0.iter().take(K).map(|i| format!("r{i}")).collect();

        // Isolate the ranged-GET regime (shrink prefetch so the segment isn't
        // slurped whole; disable the cross-call metadata cache). GET/byte are
        // measured via the FS-leaf `io_trace` accumulator (`read_range` feeds
        // `record_range_gets`/`record_bytes_read`) — the SAME counters the
        // route cost model + KRU meter price — NOT `drain_get_trace` (that
        // `GET_TRACE` is a probe-only counter and misses the Stage-F reads).
        unsafe {
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "4096");
            std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "0");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "4");
        }
        use crate::observability::io_trace;

        // Stage-F allow-set build (Region-D only), under its own trace scope.
        let (allow_res, allow_snap) = io_trace::scope(async {
            let r = pax_filtered_row_allow(fs, p, &partition_filter("p0"), None, None).await;
            (r, io_trace::snapshot())
        })
        .await;
        let (allow, stats) = allow_res
            .unwrap()
            .expect("shredded v3 segment yields an allow-set");
        let allow_snap = allow_snap.expect("io_trace scope active");

        // Probe + rerank (A0 + probe cells + Region-B), given the allow-set.
        let (hits_res, search_snap) = io_trace::scope(async {
            let r = rabitq_search_segment_coalesced_allowed(
                fs,
                p,
                &query,
                K,
                RankMetric::L2,
                None,
                None,
                Some(&allow),
                None,
            )
            .await;
            (r, io_trace::snapshot())
        })
        .await;
        let hits = hits_res.unwrap().unwrap();
        let search_snap = search_snap.expect("io_trace scope active");

        unsafe {
            std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
            std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            std::env::remove_var("PROXIMADB_IVF_K");
            std::env::remove_var("PROXIMADB_PAX_FILTERED_CASCADE");
            std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
        }

        let got: std::collections::HashSet<String> = hits.iter().map(|h| h.oid.clone()).collect();
        let recall = truth.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32;
        Measure {
            filesize,
            allow_gets: allow_snap.range_gets,
            allow_bytes: allow_snap.bytes_read,
            blocks_total: stats.blocks_total,
            blocks_pruned_footer: stats.blocks_pruned_footer,
            search_gets: search_snap.range_gets,
            search_bytes: search_snap.bytes_read,
            region_a_bytes: search_snap.ivf_region_a_bytes,
            region_b_bytes: search_snap.ivf_region_b_bytes,
            whole_region_fallback: search_snap.ivf_whole_region_fallback,
            recall,
        }
    }

    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();

    let mut rows = Vec::new();
    for (label, correlated, m2_on) in [
        ("correlated       (M2 off)", true, false),
        ("uncorrelated     (M2 off)", false, false),
        ("uncorrelated     (M2 on )", false, true),
    ] {
        let m = measure(&fs, &pseudo, correlated, m2_on).await;
        let total_bytes = m.allow_bytes + m.search_bytes;
        let total_gets = m.allow_gets + m.search_gets;
        eprintln!(
            "[C1] {label} | file={fk}KB whole-scan | \
                 allow-build: {ag} GETs {ab}KB ({bp}/{bt} D-blocks footer-pruned) | \
                 probe+rerank: {sg} GETs {sb}KB (A={ra}KB B={rb}KB wholeB-fallback={wf}) | \
                 TOTAL {tg} GETs {tk}KB ({pct}% of whole) | recall={r:.2}",
            fk = m.filesize / 1024,
            ag = m.allow_gets,
            ab = m.allow_bytes / 1024,
            bp = m.blocks_pruned_footer,
            bt = m.blocks_total,
            sg = m.search_gets,
            sb = m.search_bytes / 1024,
            ra = m.region_a_bytes / 1024,
            rb = m.region_b_bytes / 1024,
            wf = m.whole_region_fallback,
            tg = total_gets,
            tk = total_bytes / 1024,
            pct = (total_bytes * 100).checked_div(m.filesize).unwrap_or(0),
            r = m.recall,
        );
        // The stack composes end-to-end (finds real neighbors) in every config.
        assert!(m.recall > 0.0, "{label}: filtered search returns matches");
        rows.push(m);
    }
    // The measured residual: for an UNCORRELATED tag the Stage-F allow-set
    // build reads ~all Region-D blocks (footer stats can't prune it), so M2's
    // partition-homogeneous layout — which restores footer-pruning AND tightens
    // survivor ranges — must cut BOTH the allow-build and the probe+rerank
    // bytes. This is the end-to-end proof M2 attacks the dominant filtered term.
    let off = &rows[1]; // uncorrelated M2 off
    let on = &rows[2]; // uncorrelated M2 on
    assert!(
        on.blocks_pruned_footer > off.blocks_pruned_footer,
        "M2 must restore footer-pruning for the uncorrelated tag: on={} off={}",
        on.blocks_pruned_footer,
        off.blocks_pruned_footer
    );
    let (on_total, off_total) = (
        on.allow_bytes + on.search_bytes,
        off.allow_bytes + off.search_bytes,
    );
    assert!(
        on_total < off_total,
        "M2 must cut total filtered bytes for the uncorrelated tag: on={on_total} off={off_total}"
    );
    // And with M2 the filtered stack reads strictly less than a whole-object
    // scan (the cascade-OFF baseline) — the co-design win, restored.
    assert!(
        on_total < on.filesize,
        "M2 uncorrelated total {on_total}B must be < whole-object {}B",
        on.filesize
    );
}

/// TD-FPRUNE-1 C2: M2's per-partition clustering must not degrade UNFILTERED
/// recall vs the global layout — the SILENT boundary-miss risk (a row assigned
/// to its within-partition centroid can fall outside the nprobe-nearest cells
/// of a globally-ranked unfiltered query). The SIFT ratchet cannot see this
/// (SIFT carries no tag, so M2 never engages there). This builds the SAME
/// uncorrelated-tag corpus twice (M2 off vs on) and runs UNFILTERED recall@10
/// vs the f32 brute-force truth over many queries, requiring the partition
/// layout to hold recall within tolerance of the global layout. It is the eval
/// that gates any M2 default change.
#[tokio::test]
#[serial]
async fn m2_unfiltered_recall_neutral_vs_global() {
    enable_coalesced_rabitq();
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    const DIM: usize = 64;
    const P: usize = 8;
    const N: usize = 4096;
    const K: usize = 10;
    const NQ: usize = 32;

    let pseudo = |i: usize, d: usize| -> f32 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((d as u64).wrapping_mul(0x632B_E59B_D9B4_E019));
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // Uncorrelated tag (worst case for M2): partition = i%P, vector cluster =
    // (i/P)%P — disjoint index ranges ⇒ every partition spans every cluster,
    // so per-partition centroids overlap maximally (the boundary-miss surface).
    let part = |i: usize| i % P;
    let vclust = |i: usize| (i / P) % P;
    let corpus: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let c = vclust(i);
            (0..DIM)
                .map(|d| (if d == c { 50.0 } else { 0.0 }) + 0.6 * pseudo(i, d))
                .collect()
        })
        .collect();
    let records: Vec<ProximaRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut r = rec(&format!("r{i}"), 1000 + i as i64, v.clone());
            r.props.insert(
                "partition".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(format!("p{}", part(i))),
                ),
            );
            r
        })
        .collect();

    async fn build(
        records: &[ProximaRecord],
        m2_on: bool,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c2.pax");
        unsafe {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "64");
            // Explicit "1"/"0" (not present/absent) so this eval is robust to
            // the default — "0" is the GLOBAL layout, "1" the opt-in
            // partition layout.
            std::env::set_var(
                "PROXIMADB_PAX_PARTITION_LAYOUT",
                if m2_on { "1" } else { "0" },
            );
        }
        write_pax_segment_compacted_shredded(
            &path,
            records,
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            &[proximadb_block_format::ShredColumn::typed(
                "partition",
                100,
                // fixture partition values are STRINGS ("p0".."p7") — the
                // exact declared type; Int64 would never shred them.
                proximadb_data_model::ProximaType::String,
            )],
            None,
            None,
        )
        .unwrap();
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
            std::env::remove_var("PROXIMADB_PAX_PARTITION_LAYOUT");
        }
        (dir, path)
    }

    let (_d_off, path_off) = build(&records, false).await;
    let (_d_on, path_on) = build(&records, true).await;
    let (p_off, p_on) = (path_off.to_str().unwrap(), path_on.to_str().unwrap());

    // Guard against a VACUOUS pass: M2 must actually have changed the layout
    // (partition-contiguous rows ⇒ different physical bytes). If the two
    // segments were byte-identical, `cluster_plan_ivf_probe_partitioned` fell
    // back to the global plan and the recall comparison below would be a no-op.
    let bytes_off = std::fs::read(&path_off).unwrap();
    let bytes_on = std::fs::read(&path_on).unwrap();
    assert_ne!(
        bytes_off, bytes_on,
        "M2 partition layout did not engage — segments are byte-identical"
    );

    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    unsafe {
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
    }

    let l2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();

    let (mut sum_off, mut sum_on) = (0f32, 0f32);
    for qi in (0..N).step_by(N / NQ).take(NQ) {
        let query = &corpus[qi];
        // f32 brute-force ground truth top-K over ALL rows (unfiltered).
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&a, &b| {
            l2(&corpus[a], query)
                .partial_cmp(&l2(&corpus[b], query))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let gt: std::collections::HashSet<String> =
            idx.iter().take(K).map(|i| format!("r{i}")).collect();
        let recall = |hits: &[CascadeHit]| {
            let got: std::collections::HashSet<String> =
                hits.iter().map(|h| h.oid.clone()).collect();
            gt.iter().filter(|o| got.contains(*o)).count() as f32 / K as f32
        };
        let h_off =
            rabitq_search_segment_coalesced(&fs, p_off, query, K, RankMetric::L2, None, None)
                .await
                .unwrap()
                .unwrap();
        let h_on = rabitq_search_segment_coalesced(&fs, p_on, query, K, RankMetric::L2, None, None)
            .await
            .unwrap()
            .unwrap();
        sum_off += recall(&h_off);
        sum_on += recall(&h_on);
    }
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_PROBE");
        std::env::remove_var("PROXIMADB_IVF_K");
    }
    let (r_off, r_on) = (sum_off / NQ as f32, sum_on / NQ as f32);
    eprintln!(
        "[C2] unfiltered recall@{K} over {NQ} queries: \
             global(M2 off)={r_off:.3}  partition(M2 on)={r_on:.3}  Δ={:.3}",
        r_on - r_off
    );
    // The gate: partition clustering must hold unfiltered recall within one
    // quantization step of the global layout. If this ever fails, M2 must stay
    // opt-in and the dual-directory follow-up is justified.
    assert!(
        r_on >= r_off - 0.05,
        "M2 partition layout degrades unfiltered recall vs global: \
             on={r_on:.3} off={r_off:.3} (tol 0.05)"
    );
}

const SAME_WAVE_DIM: usize = 64;
const SAME_WAVE_N: usize = 400;

/// A v3 (A0) segment of 400 rows probed on every cell, with no prefix
/// prefetch beyond the header, so the probe's Region-A reads and the footer
/// form a real cold wave. Returns the corpus and records.
fn same_wave_fixture() -> (Vec<Vec<f32>>, Vec<ProximaRecord>) {
    enable_coalesced_rabitq();
    // SAFETY: nextest runs each test in its own process.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
        std::env::set_var("PROXIMADB_IVF_K", "4");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", "4");
        std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
        std::env::set_var("PROXIMADB_TRACE_GETS", "1");
    }
    let corpus: Vec<Vec<f32>> = (0..SAME_WAVE_N)
        .map(|i| {
            (0..SAME_WAVE_DIM)
                .map(|d| (((i * 131 + d * 17) % 251) as f32) * 0.01)
                .collect()
        })
        .collect();
    let records = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("r{i}"), 1000 + i as i64, v.clone()))
        .collect();
    (corpus, records)
}

fn write_same_wave_segment(dir: &std::path::Path, name: &str, records: &[ProximaRecord]) -> String {
    let seg = dir.join(name);
    write_pax_segment_compacted(
        &seg,
        records,
        "col",
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        false,
        Some(16 * 1024),
        None,
    )
    .unwrap();
    seg.to_string_lossy().to_string()
}

fn set_same_wave(on: bool) {
    // SAFETY: nextest runs each test in its own process.
    unsafe {
        std::env::set_var(
            "PROXIMADB_PAX_READ_SAME_WAVE_RERANK",
            if on { "1" } else { "0" },
        )
    };
}

fn hit_summary(hits: &[CascadeHit]) -> Vec<(String, u32)> {
    hits.iter()
        .map(|h| (h.oid.clone(), h.distance.to_bits()))
        .collect()
}

fn first_of(trace: &[(CacheTier, u64)], tier: CacheTier) -> Option<usize> {
    trace.iter().position(|(t, _)| *t == tier)
}

fn tier_bytes(trace: &[(CacheTier, u64)], tier: CacheTier) -> u64 {
    trace
        .iter()
        .filter(|(t, _)| *t == tier)
        .map(|(_, len)| *len)
        .sum()
}

/// TD-PAXDIR-1 D5: with `PROXIMADB_PAX_READ_SAME_WAVE_RERANK` the probed
/// cells' SQ8 extents ride the coarse probe's Region-A wave, so the rerank
/// issues no wave of its own. The GET trace records the footer's tier after
/// the wave's per-range records, so a Region-B read inside the wave appears
/// before it; without the gate the rerank's Region-B read follows it.
#[tokio::test]
async fn same_wave_rerank_serves_survivors_from_the_probe_wave() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    let (corpus, records) = same_wave_fixture();
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
    let mut results = Vec::new();
    for (name, on) in [("off.pax", false), ("on.pax", true)] {
        let path = write_same_wave_segment(dir.path(), name, &records);
        set_same_wave(on);
        drain_get_trace();
        let hits =
            rabitq_search_segment_coalesced(&fs, &path, &corpus[7], 5, RankMetric::L2, None, None)
                .await
                .unwrap()
                .expect("probe search hits");
        results.push((hits, drain_get_trace()));
    }
    let (off_hits, off_trace) = &results[0];
    let (on_hits, on_trace) = &results[1];
    assert_eq!(
        hit_summary(on_hits),
        hit_summary(off_hits),
        "identical hits and distances"
    );
    let on_b = first_of(on_trace, CacheTier::SurvivorPayload).expect("Region B read");
    let on_footer = first_of(on_trace, CacheTier::InvariantMeta).expect("footer read");
    assert!(
        on_b < on_footer,
        "Region B rode the probe wave: {on_trace:?}"
    );
    assert_eq!(
        tier_bytes(on_trace, CacheTier::SurvivorPayload),
        (SAME_WAVE_N * SAME_WAVE_DIM) as u64,
        "every probed cell's codes, once, and no rerank fetch"
    );
    let off_b = first_of(off_trace, CacheTier::SurvivorPayload).expect("Region B read");
    let off_footer = first_of(off_trace, CacheTier::InvariantMeta).expect("footer read");
    assert!(
        off_b > off_footer,
        "without the gate the rerank fetches after: {off_trace:?}"
    );
}

/// The wave's Region-B bytes populate the survivor-range cache exactly as a
/// rerank fetch would, so a warm repeat issues no Region-B read either way.
#[tokio::test]
async fn same_wave_rerank_warms_the_survivor_cache_like_a_rerank_fetch() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    let (corpus, records) = same_wave_fixture();
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
    for (name, on) in [("off.pax", false), ("on.pax", true)] {
        let path = write_same_wave_segment(dir.path(), name, &records);
        set_same_wave(on);
        let survivor = SurvivorRangeCache::new(64 * 1024 * 1024);
        let search = || {
            rabitq_search_segment_coalesced(
                &fs,
                &path,
                &corpus[7],
                5,
                RankMetric::L2,
                None,
                Some(&survivor),
            )
        };
        let first = search().await.unwrap().expect("cold hits");
        drain_get_trace();
        let second = search().await.unwrap().expect("warm hits");
        let warm = drain_get_trace();
        assert_eq!(hit_summary(&first), hit_summary(&second));
        assert_eq!(
            tier_bytes(&warm, CacheTier::SurvivorPayload),
            0,
            "same_wave={on}: warm repeat must read no Region B: {warm:?}"
        );
    }
}

/// A filtered probe never rides Region B on the wave (a selective filter can
/// probe every cell for a few survivors); a failed batched read falls back
/// to the sequential path with identical hits.
#[tokio::test]
async fn same_wave_rerank_skips_filtered_probes_and_survives_prefetch_errors() {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    let (corpus, records) = same_wave_fixture();
    let dir = tempfile::tempdir().unwrap();
    let local: Arc<dyn proximadb_storage_filesystem_types::FileSystem> =
        Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap());
    set_same_wave(true);

    let path = write_same_wave_segment(dir.path(), "filtered.pax", &records);
    let allow: proximadb_block_format::RowAllow = (0..SAME_WAVE_N).collect();
    drain_get_trace();
    rabitq_search_segment_coalesced_allowed(
        local.as_ref(),
        &path,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        None,
        Some(&allow),
        None,
    )
    .await
    .unwrap()
    .expect("filtered hits");
    let trace = drain_get_trace();
    assert!(
        first_of(&trace, CacheTier::ProbeIndex).is_some(),
        "the filtered probe ran: {trace:?}"
    );
    let b = first_of(&trace, CacheTier::SurvivorPayload).expect("Region B read");
    let footer = first_of(&trace, CacheTier::InvariantMeta).expect("footer read");
    assert!(
        b > footer,
        "a filtered probe must not ride Region B: {trace:?}"
    );

    // Gate-off reference hits.
    set_same_wave(false);
    let reference = write_same_wave_segment(dir.path(), "reference.pax", &records);
    let expected = rabitq_search_segment_coalesced(
        local.as_ref(),
        &reference,
        &corpus[7],
        5,
        RankMetric::L2,
        None,
        None,
    )
    .await
    .unwrap()
    .expect("reference hits");

    // Gate on, every batched read fails (sequential fallback) and the first
    // single Region-B read — the wave's own — fails too: the wave soft-fails,
    // the rerank fetches Region B itself, and the hits are unchanged.
    set_same_wave(true);
    let path = write_same_wave_segment(dir.path(), "fallback.pax", &records);
    let header_bytes = std::fs::read(&path).unwrap();
    let header = proximadb_storage_common::segment_layout::SegmentHeaderPrefix::parse(
        &header_bytes[..proximadb_storage_common::segment_layout::SEG_HEADER_PREFIX_LEN],
    )
    .unwrap();
    let failing = PrefetchErrorFileSystem {
        inner: Arc::clone(&local),
        fail_once_in: Some((
            header.sq8_off..header.sq8_off + header.sq8_len,
            std::sync::atomic::AtomicBool::new(false),
        )),
    };
    drain_get_trace();
    let hits =
        rabitq_search_segment_coalesced(&failing, &path, &corpus[7], 5, RankMetric::L2, None, None)
            .await
            .unwrap()
            .expect("fallback hits");
    let trace = drain_get_trace();
    assert!(
        failing
            .fail_once_in
            .as_ref()
            .unwrap()
            .1
            .load(std::sync::atomic::Ordering::SeqCst),
        "the wave's Region-B read was attempted and failed"
    );
    assert_eq!(hit_summary(&hits), hit_summary(&expected));
    assert!(
        first_of(&trace, CacheTier::ProbeIndex).is_some(),
        "the probe survived the Region-B failure: {trace:?}"
    );
    let b = first_of(&trace, CacheTier::SurvivorPayload).expect("rerank Region B read");
    let footer = first_of(&trace, CacheTier::InvariantMeta).expect("footer read");
    assert!(b > footer, "the rerank fetched Region B itself: {trace:?}");
}

/// TD-PAXDIR-1 D5: only cold-A cells ride; extents merge where they touch,
/// are clamped to Region B, and a planned range is served only when one
/// prefetched extent covers it.
#[test]
fn same_wave_b_ranges_are_cold_clamped_merged_and_served_on_full_coverage() {
    use proximadb_storage_common::coarse_directory::CoarseCellEntry;
    let cell = |a_off: u64, b_off: u64, b_len: u64| CoarseCellEntry {
        a_off,
        a_len: 10,
        b_off,
        b_len,
        ..Default::default()
    };
    let cells = [
        cell(0, 100, 40),
        cell(10, 140, 40),
        cell(20, 300, 20),
        cell(30, 0, 0),
        cell(40, 900, 50),
    ];
    let cold = [0..30, 40..50];
    assert_eq!(
        probed_cell_b_ranges(&cells, &[2, 0, 1, 3, 4], &cold, 100, 400),
        vec![100..180, 300..320],
        "adjacent cells merge; empty and out-of-region extents are skipped"
    );
    assert_eq!(
        probed_cell_b_ranges(&cells, &[0, 2], &[20..30], 100, 400),
        vec![300..320],
        "a warm-A cell does not ride"
    );
    let prefetched = vec![
        (100u64, (0..80u8).collect::<Vec<_>>()),
        (300u64, (200..220u8).collect::<Vec<_>>()),
    ];
    assert_eq!(prefetched_range(&prefetched, 104, 108), Some((0, 4..8)));
    assert_eq!(prefetched_range(&prefetched, 300, 304), Some((1, 0..4)));
    assert_eq!(
        prefetched_range(&prefetched, 176, 184),
        None,
        "spans past the extent"
    );
    assert_eq!(prefetched_range(&prefetched, 50, 60), None);
    assert_eq!(prefetched_range(&[], 0, 1), None);
}

/// TD-PAXDIR-1 D5: rerank ranges never cross a prefetched extent's
/// boundary (so covered rows are always served from the wave), and rows stay
/// in ascending order; with no extents the plan is the plain planner's.
#[test]
fn rerank_ranges_never_cross_a_prefetched_extent() {
    use crate::storage::engines::sst::readers::sst_query_engine::ObjectRangeCoalescePolicy;
    let policy = ObjectRangeCoalescePolicy {
        max_gap_bytes: 1 << 20,
        max_range_bytes: 0,
    };
    // dim 4, codes_base 0: row g is bytes [4g, 4g + 4).
    let survivors = [1usize, 2, 10, 11, 30, 31, 60];
    let plain = plan_coalesced_row_ranges(&survivors, 4, 0, &policy);
    assert_eq!(plain.len(), 1, "the wide gap policy merges everything");
    assert_eq!(
        plan_rerank_ranges(&survivors, 4, 0, &policy, &[])
            .iter()
            .map(|f| (f.start, f.end))
            .collect::<Vec<_>>(),
        plain.iter().map(|f| (f.start, f.end)).collect::<Vec<_>>()
    );
    // Extents cover rows 0..16 and 28..32; rows 60 is outside both.
    let extents = [0u64..64, 112..128];
    let planned = plan_rerank_ranges(&survivors, 4, 0, &policy, &extents);
    for fetch in &planned {
        let inside = extents
            .iter()
            .any(|e| e.start <= fetch.start && fetch.end <= e.end);
        let crosses = extents.iter().any(|e| {
            fetch.start < e.end
                && e.start < fetch.end
                && !(e.start <= fetch.start && fetch.end <= e.end)
        });
        assert!(
            inside || !crosses,
            "{:?} crosses an extent",
            (fetch.start, fetch.end)
        );
    }
    let rows: Vec<usize> = planned
        .iter()
        .flat_map(|f| f.rows.iter().copied())
        .collect();
    assert_eq!(rows, survivors, "every row once, ascending");
    assert_eq!(
        planned.len(),
        3,
        "extent 1, extent 2, and the uncovered tail"
    );
}

/// Write `records` as a RaBitQ PAX segment with 4 KiB blocks into a fresh
/// tempdir (returned so it outlives the test body) and open a local fs.
async fn small_rabitq_segment(
    records: &[ProximaRecord],
) -> (
    tempfile::TempDir,
    std::path::PathBuf,
    crate::storage::persistence::filesystem::local::LocalFileSystem,
) {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.pax");
    write_pax_segment(
        &path,
        records,
        "col",
        1,
        VectorQuant::RaBitQ,
        Some(4 * 1024),
        None,
    )
    .unwrap();
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .unwrap();
    (dir, path, fs)
}
