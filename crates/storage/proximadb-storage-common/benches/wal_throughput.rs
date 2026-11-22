//! Criterion benchmarks for canonical WAL entry throughput.
//!
//! These benchmarks measure the overhead of the canonical durability layer
//! introduced in Phase 5 of RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE.
//!
//! Run with: `cargo bench -p proximadb-storage-common`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::{
    CanonicalOperation, CanonicalWalEntry, EdgeRef, ProjectionDirective, SnapshotManifest,
    latest_checkpoint, recover_from_canonical_wal,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_record(oid: &str) -> ProximaRecord {
    ProximaRecord {
        oid: oid.into(),
        ..Default::default()
    }
}

fn make_directives(n: usize) -> Vec<ProjectionDirective> {
    (0..n)
        .map(|i| match i % 4 {
            0 => ProjectionDirective::DocumentJsonPathIndex {
                collection_id: "docs".into(),
                indexed_paths: vec!["$.title".into(), "$.tags[*]".into()],
            },
            1 => ProjectionDirective::FullTextIndex {
                collection_id: "docs".into(),
                indexed_fields: vec!["body".into()],
            },
            2 => ProjectionDirective::AdjacencyTableRow {
                graph_id: "kg".into(),
                node_oid: format!("node-{i}"),
                edge_refs: vec![EdgeRef {
                    src_oid: format!("node-{i}"),
                    dst_oid: format!("node-{}", i + 1),
                    edge_type: "RELATES_TO".into(),
                    weight: Some(1.0),
                }],
            },
            _ => ProjectionDirective::CsrRebuild {
                graph_id: "kg".into(),
            },
        })
        .collect()
}

fn make_upsert_entry(seq: u64, directives: Vec<ProjectionDirective>) -> CanonicalWalEntry {
    CanonicalWalEntry::new(
        seq,
        CanonicalOperation::RecordUpsert {
            collection_id: "bench_col".into(),
            record: Box::new(make_record(&format!("rec-{seq}"))),
            projections: directives,
        },
        None,
    )
}

fn make_checkpoint(seq: u64) -> CanonicalWalEntry {
    CanonicalWalEntry::new(
        seq,
        CanonicalOperation::Checkpoint(SnapshotManifest {
            sequence_number: seq,
            timestamp_ms: 0,
            collection_ids: vec!["bench_col".into()],
            projection_freshness: vec![],
        }),
        None,
    )
}

// ── Noop rebuilder for throughput benchmarks ──────────────────────────────

struct NoopRebuilder;
impl proximadb_storage_common::wal_entry::ProjectionRebuilder for NoopRebuilder {
    fn apply_directive(
        &mut self,
        _record: Option<&ProximaRecord>,
        _directive: &ProjectionDirective,
    ) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_wal_entry_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_entry_construction");

    for n_directives in [0usize, 1, 3, 6] {
        let dirs = make_directives(n_directives);
        group.bench_with_input(
            BenchmarkId::new("directives", n_directives),
            &dirs,
            |b, dirs| {
                b.iter(|| make_upsert_entry(1, dirs.clone()));
            },
        );
    }
    group.finish();
}

fn bench_wal_entry_serde(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_entry_serde");

    for n_directives in [1usize, 3, 6] {
        let entry = make_upsert_entry(1, make_directives(n_directives));

        group.bench_with_input(
            BenchmarkId::new("serialize_json", n_directives),
            &entry,
            |b, e| {
                b.iter(|| serde_json::to_vec(e).unwrap());
            },
        );

        let serialized = serde_json::to_vec(&entry).unwrap();
        group.bench_with_input(
            BenchmarkId::new("deserialize_json", n_directives),
            &serialized,
            |b, bytes| {
                b.iter(|| serde_json::from_slice::<CanonicalWalEntry>(bytes).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_recover_from_canonical_wal(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery_throughput");

    for n_entries in [100u64, 1_000, 10_000] {
        // Build a log with a checkpoint at 10% and entries after it.
        let checkpoint_at = n_entries / 10;
        let mut entries: Vec<CanonicalWalEntry> = (1..=checkpoint_at)
            .map(|seq| make_upsert_entry(seq, make_directives(2)))
            .collect();
        entries.push(make_checkpoint(checkpoint_at));
        entries.extend(
            (checkpoint_at + 1..=n_entries).map(|seq| make_upsert_entry(seq, make_directives(2))),
        );

        group.bench_with_input(
            BenchmarkId::new("entries_after_checkpoint", n_entries),
            &entries,
            |b, log| {
                let cp_lsn = latest_checkpoint(log)
                    .map(|m| m.sequence_number)
                    .unwrap_or(0);
                b.iter(|| {
                    let mut rb = NoopRebuilder;
                    recover_from_canonical_wal(log, &mut rb, cp_lsn)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_replay", n_entries),
            &entries,
            |b, log| {
                b.iter(|| {
                    let mut rb = NoopRebuilder;
                    recover_from_canonical_wal(log, &mut rb, 0)
                });
            },
        );
    }
    group.finish();
}

fn bench_latest_checkpoint_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("latest_checkpoint_scan");

    for n_entries in [1_000u64, 10_000, 100_000] {
        let checkpoint_interval = n_entries / 5;
        let entries: Vec<CanonicalWalEntry> = (1..=n_entries)
            .map(|seq| {
                if seq % checkpoint_interval == 0 {
                    make_checkpoint(seq)
                } else {
                    make_upsert_entry(seq, vec![])
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("entries", n_entries),
            &entries,
            |b, log| {
                b.iter(|| latest_checkpoint(log));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_wal_entry_construction,
    bench_wal_entry_serde,
    bench_recover_from_canonical_wal,
    bench_latest_checkpoint_scan,
);
criterion_main!(benches);
