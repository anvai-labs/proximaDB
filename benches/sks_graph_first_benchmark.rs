// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! SKS Graph-First Architecture Benchmark
//!
//! This benchmark compares the legacy SKS architecture (split storage) with the
//! new graph-first architecture (Orion as primary storage).
//!
//! Metrics measured:
//! - Entity insertion throughput
//! - Hybrid query performance (vector similarity + graph traversal)
//! - Memory overhead
//! - Cache locality

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

// Test fixtures
#[path = "../tests/common/mod.rs"]
mod common;
use common::sks_fixtures::TestKnowledgeGraph;

/// Benchmark entity insertion for small graphs (100 entities)
fn bench_entity_insertion_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_entity_insertion_small");
    group.throughput(Throughput::Elements(100));
    group.measurement_time(Duration::from_secs(10));

    let graph = TestKnowledgeGraph::small();

    // TODO: Implement when OrionBackedEntityStore is ready
    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            // Simulate legacy approach:
            // 1. Insert vectors into SST
            // 2. Insert relations into HashMap
            // 3. Insert metadata into KV store
            // This is a placeholder - actual implementation will come later
            black_box(&graph.entities);
            black_box(&graph.relations);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            // Simulate graph-first approach:
            // 1. Insert entities as nodes into Orion (includes embeddings + metadata)
            // 2. Insert relations as edges into Orion CSR
            // This is a placeholder - actual implementation will come later
            black_box(&graph.entities);
            black_box(&graph.relations);
        });
    });

    group.finish();
}

/// Benchmark entity insertion for medium graphs (1K entities)
fn bench_entity_insertion_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_entity_insertion_medium");
    group.throughput(Throughput::Elements(1000));
    group.measurement_time(Duration::from_secs(15));

    let graph = TestKnowledgeGraph::medium();

    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            black_box(&graph.entities);
            black_box(&graph.relations);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            black_box(&graph.entities);
            black_box(&graph.relations);
        });
    });

    group.finish();
}

/// Benchmark hybrid queries: vector similarity + graph traversal
fn bench_hybrid_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_hybrid_query");
    group.measurement_time(Duration::from_secs(10));

    let graph = TestKnowledgeGraph::research_papers();

    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            // Simulate legacy approach:
            // 1. Vector search in SST (returns entity IDs)
            // 2. Lookup relations in HashMap (separate data structure)
            // 3. Lookup entity metadata in KV store
            // 4. Merge results (expensive!)
            black_box(&graph.embeddings[0]);
            black_box(&graph.relations);
            black_box(&graph.entities[0]);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            // Simulate graph-first approach:
            // 1. AXIS vector search for initial candidates (returns node IDs)
            // 2. Orion graph traversal (CSR for O(1) neighbor access)
            // 3. Node properties already cached with adjacency list
            black_box(&graph.embeddings[0]);
            black_box(&graph.relations);
            black_box(&graph.entities[0]);
        });
    });

    group.finish();
}

/// Benchmark memory overhead
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_memory_overhead");
    group.measurement_time(Duration::from_secs(10));

    let graph = TestKnowledgeGraph::medium();

    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            // Simulate legacy memory layout:
            // - Vectors in SST/VIPER format
            // - Relations in HashMap (pointer overhead)
            // - Metadata in separate KV store
            // Total: ~3x memory overhead due to fragmentation
            let vectors_size = graph.embeddings.len() * graph.embeddings[0].len() * 4;
            let relations_size = graph.relations.len() * 64; // HashMap overhead
            let metadata_size = graph.entities.len() * 256; // KV overhead
            black_box(vectors_size + relations_size + metadata_size);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            // Simulate graph-first memory layout:
            // - Nodes with inline embeddings (cache-friendly)
            // - CSR for relations (compact, O(1) access)
            // - Metadata stored with nodes (no separate lookup)
            // Total: ~1.5x memory overhead (compressed CSR)
            let nodes_size = graph.entities.len() * (graph.embeddings[0].len() * 4 + 256);
            let csr_size = graph.relations.len() * 12; // CSR: src, dst, weight
            black_box(nodes_size + csr_size);
        });
    });

    group.finish();
}

/// Benchmark citation graph traversal (research papers fixture)
fn bench_citation_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_citation_traversal");
    group.measurement_time(Duration::from_secs(10));

    let graph = TestKnowledgeGraph::research_papers();

    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            // Simulate traversing citation graph in legacy approach:
            // 1. Start from seed paper (entity ID)
            // 2. Lookup relations in HashMap (cache miss likely)
            // 3. For each cited paper, lookup entity metadata (another cache miss)
            // 4. Repeat for 2-hop traversal
            black_box(&graph.entities[0]);
            black_box(&graph.relations);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            // Simulate traversing citation graph in graph-first approach:
            // 1. Start from seed node (Orion node ID)
            // 2. CSR lookup for outgoing edges (O(1), cache-friendly)
            // 3. Node properties already cached with adjacency list
            // 4. 2-hop traversal uses same CSR structure
            black_box(&graph.entities[0]);
            black_box(&graph.relations);
        });
    });

    group.finish();
}

/// Benchmark e-commerce product graph queries
fn bench_ecommerce_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("sks_ecommerce_query");
    group.measurement_time(Duration::from_secs(10));

    let graph = TestKnowledgeGraph::ecommerce();

    group.bench_function("legacy_split_storage", |b| {
        b.iter(|| {
            // Simulate product recommendation query:
            // 1. Vector search for similar products
            // 2. Lookup "related_to" edges in HashMap
            // 3. Lookup product metadata (price, category, rating) in KV
            // 4. Filter by metadata (separate data structure)
            black_box(&graph.embeddings[0]);
            black_box(&graph.relations);
            black_box(&graph.entities[0]);
        });
    });

    group.bench_function("graph_first_orion", |b| {
        b.iter(|| {
            // Simulate product recommendation query:
            // 1. AXIS vector search for similar products
            // 2. CSR traversal for "related_to" edges
            // 3. Metadata filter applied during traversal (same cache line)
            black_box(&graph.embeddings[0]);
            black_box(&graph.relations);
            black_box(&graph.entities[0]);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_entity_insertion_small,
    bench_entity_insertion_medium,
    bench_hybrid_query,
    bench_memory_overhead,
    bench_citation_traversal,
    bench_ecommerce_query,
);

criterion_main!(benches);
