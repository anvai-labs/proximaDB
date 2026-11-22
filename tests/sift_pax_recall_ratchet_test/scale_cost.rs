//! Isolated worker for scripts/bench/pax_scale_cost.py. Each reader process
//! starts with fresh DRAM and opens the fixture and L2 named by the driver.
use super::*;
use proximadb_cache::PersistentByteStore;
use proximadb_storage_common::{
    coarse_directory::CoarseDirectory,
    iops_budget::IopsBudget,
    segment_layout::{SegmentFooterIndex, SegmentHeaderPrefix},
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Run {
    action: String,
    storage: String,
    scratch: PathBuf,
    rows: usize,
    queries: usize,
    query: usize,
    route: String,
    budget_mib: u64,
    cap_mib: u64,
    wire_markers: bool,
}

fn emit(value: Value) {
    println!("PAX_SCALE_COST {value}");
}

async fn prepare(run: &Run, col: &Collection, engine: &SstEngine) {
    let base =
        read_vec_records_f32(&dataset_path("sift_base.fvecs").unwrap(), Some(run.rows)).unwrap();
    assert_eq!(base.len(), run.rows);
    let queries = read_vec_records_f32(
        &dataset_path("sift_query.fvecs").unwrap(),
        Some(run.queries),
    )
    .unwrap();
    assert_eq!(queries.len(), run.queries);
    let (truth, _) = exact_ground_truth(
        &base,
        &queries,
        dataset_path("sift_groundtruth.ivecs").as_deref(),
    );
    std::fs::write(
        run.scratch.join("truth.json"),
        serde_json::to_vec(&truth).unwrap(),
    )
    .unwrap();
    let records: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(i, v)| ivf2_record(i, v))
        .collect();
    drop(base);
    let path = run.scratch.join("fixture.pax");
    let root = get_counts_query_ctx(col, vec![0.; DIMENSION], SearchMode::Exact)
        .collection_storage_path()
        .unwrap();
    write_pax_segment_compacted(
        &path,
        &records,
        &col.id,
        1,
        VectorQuant::RaBitQ,
        VectorQuant::Sq8,
        true,
        Some(IopsBudget::for_path(&root).target_block_bytes() as usize),
        Some(&root),
    )
    .unwrap();
    drop(records);
    let bytes = std::fs::read(&path).unwrap();
    let header = SegmentHeaderPrefix::parse(&bytes).unwrap();
    let extent = |start: u64, len: u64| &bytes[start as usize..(start + len) as usize];
    let footer = SegmentFooterIndex::parse(extent(header.footer_off, header.footer_len)).unwrap();
    assert!(footer.has_f32_tier && header.c_len > 0);
    assert_eq!(footer.row_count, run.rows as u64);
    let cells = if header.a0_len == 0 {
        0
    } else {
        CoarseDirectory::parse(extent(header.a0_off, header.a0_len))
            .unwrap()
            .cells
            .len()
    };
    let fs = engine.filesystem().get_filesystem(&root).unwrap();
    fs.create_dir_all(&root).await.unwrap();
    fs.write(&format!("{root}/L1_scale.pax"), &bytes, None)
        .await
        .unwrap();
    emit(
        json!({"rows":run.rows, "segment_bytes":bytes.len(), "blocks":footer.blocks.len(),
        "cells":cells, "f32_region_bytes":header.c_len}),
    );
}

#[tokio::test]
#[ignore = "isolated PAX cost worker; invoke scripts/bench/pax_scale_cost.py"]
async fn worker() {
    let Ok(raw) = std::env::var("PROXIMADB_PAX_SCALE_COST_RUN") else {
        return;
    };
    let run: Run = serde_json::from_str(&raw).unwrap();
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let mut col = collection("pax_scale", run.storage.clone());
    let config = col.config.as_mut().unwrap();
    config.tags.push("pax_f32_tier:on".into());
    config.index_policy = Some(proximadb::proto::proximadb_v1::IndexPolicy {
        byte_budget: run.budget_mib << 20,
        ..Default::default()
    });
    let engine = SstEngine::new().await.unwrap();
    if run.action == "prepare" {
        prepare(&run, &col, &engine).await;
        return;
    }
    let cache_dir = run.scratch.join(format!("{}-{}", run.route, run.query));
    let store = Arc::new(PersistentByteStore::open(cache_dir, 1 << 30).unwrap());
    let engine = engine.with_warm_tier_caches(
        Some(Arc::new(SegmentInvariantsCache::with_l2(
            128 << 20,
            store.clone(),
        ))),
        Some(Arc::new(SurvivorRangeCache::with_resolver_and_l2(
            256 << 20,
            None,
            Some(store),
        ))),
    );
    let queries = read_vec_records_f32(
        &dataset_path("sift_query.fvecs").unwrap(),
        Some(run.query + 1),
    )
    .unwrap();
    let truth: Vec<std::collections::HashSet<String>> =
        serde_json::from_slice(&std::fs::read(run.scratch.join("truth.json")).unwrap()).unwrap();
    let mode = match run.route.as_str() {
        "exact" => SearchMode::Exact,
        "ann" => SearchMode::approximate(),
        "adaptive" => SearchMode::default(),
        other => panic!("unknown route {other}"),
    };
    let phases: &[&str] = match run.action.as_str() {
        "cold_repeat" => &["cold", "l1_repeat"],
        "reopen" => &["l2_reopen"],
        "policy" => &["policy"],
        other => panic!("unknown action {other}"),
    };
    let fs = engine.filesystem().get_filesystem(&run.storage).unwrap();
    for phase in phases {
        // The emulator access log brackets wire attempts by marker name,
        // independently of clock synchronization. Markers are outside the
        // query's collection prefix, io_trace scope and elapsed time.
        let marker = format!(
            "{}/_meter/{}-{}-{}-{}-{phase}",
            run.storage, run.query, run.route, run.cap_mib, run.budget_mib
        );
        if run.wire_markers {
            fs.write(&format!("{marker}.start"), b"1", None)
                .await
                .unwrap();
        }
        let start = Instant::now();
        let (ids, s) = traced_query(&engine, &col, queries[run.query].clone(), mode.clone()).await;
        let elapsed_us = start.elapsed().as_micros() as u64;
        if run.wire_markers {
            fs.write(&format!("{marker}.end"), b"1", None)
                .await
                .unwrap();
        }
        let hits: std::collections::HashSet<_> = ids.iter().cloned().collect();
        assert_eq!(hits.len(), TOP_K);
        let overlap = hits.intersection(&truth[run.query]).count();
        assert_eq!(s.vector_accesses.len(), 1);
        let actual = s.vector_accesses[0].actual_path.as_str();
        if run.route != "adaptive" {
            assert_eq!(actual, run.route);
        }
        if actual == "exact" {
            assert_eq!(overlap, TOP_K, "exact oracle mismatch");
        }
        emit(
            json!({"rows": run.rows, "query":run.query, "phase":phase, "route":run.route,
            "actual":actual, "budget_mib":run.budget_mib, "overlap":overlap, "us":elapsed_us,
            "marker":marker, "ids":ids, "get":s.get_ops,
            "ranged":s.range_gets, "head":s.head_ops, "list":s.list_ops, "bytes":s.bytes_read,
            "cells_probed":s.ivf_cells_probed, "rows_probed":s.ivf_probed_rows,
            "whole_region_fallback":s.ivf_whole_region_fallback,
            "footer_hit":s.footer_hits, "footer_miss":s.footer_misses,
            "l1_hit":s.survivor_l1_hits, "l1_miss":s.survivor_l1_misses,
            "l2_hit":s.l2_hits, "l2_miss":s.l2_misses}),
        );
    }
}
