// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Value codec benchmark: old Value-first path vs new one-pass visitor.
//!
//! Measures the kernel-level cost of deserializing JSON bytes into `SqlValue`
//! via two paths:
//! 1. Old path: `serde_json::from_str` → `serde_json::Value` → `serde_json::from_value::<SqlValue>`
//!    (two passes, every string allocated twice)
//! 2. New path: `sql_value_from_slice` (one pass, zero intermediate `serde_json::Value`)
//!
//! Both paths now use the same visitor internally, so the comparison is
//! indicative — the real win is the elimination of the intermediate DOM
//! allocation, which is visible in the throughput numbers.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use proximadb::proto::proximadb_v1::SqlValue;
use proximadb::proto::serde_impls::sql_value_from_slice;

fn generate_small_object() -> String {
    r#"{"name": "Alice", "age": 30, "active": true}"#.to_string()
}

fn generate_string_heavy() -> String {
    let mut s = String::from("{");
    for i in 0..50 {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!(r#""key_{}": "value_{}""#, i, i));
    }
    s.push('}');
    s
}

fn generate_numeric_array() -> String {
    let arr: Vec<i32> = (0..1000).collect();
    serde_json::to_string(&arr).unwrap()
}

fn generate_deep_nested() -> String {
    r#"{"a": {"b": {"c": {"d": {"e": "deep"}}}}}"#.to_string()
}

fn generate_tagged_string() -> String {
    r#"{"string_value": "test"}"#.to_string()
}

fn generate_tagged_array() -> String {
    r#"{"array_value": {"values": [1, 2, 3, 4, 5]}}"#.to_string()
}

fn bench_deserialize_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_codec_deserialize");

    let fixtures = [
        ("small_object", generate_small_object()),
        ("string_heavy", generate_string_heavy()),
        ("numeric_array", generate_numeric_array()),
        ("deep_nested", generate_deep_nested()),
        ("tagged_string", generate_tagged_string()),
        ("tagged_array", generate_tagged_array()),
    ];

    for (name, doc) in fixtures {
        let bytes = doc.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_function(BenchmarkId::new("visitor_from_slice", name), |b| {
            b.iter(|| {
                let _: SqlValue =
                    sql_value_from_slice(black_box(bytes)).expect("deserialize must succeed");
            });
        });

        group.bench_function(BenchmarkId::new("dom_via_value", name), |b| {
            b.iter(|| {
                let json_value: serde_json::Value =
                    serde_json::from_str(black_box(&doc)).expect("parse must succeed");
                let _: SqlValue =
                    serde_json::from_value(json_value).expect("deserialize must succeed");
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_deserialize_paths);
criterion_main!(benches);
