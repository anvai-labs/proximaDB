// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Write-amplification of the PAX residual `PROPS` tail (TD-USUB-6 2b).
//!
//! # What this measures, and why it is not a timing benchmark
//!
//! TD-USUB-6 is justified on removing a **double write**: today every shredded
//! prop is stored twice — once in its typed stripe, once in the msgpack `PROPS`
//! tail, which the writer's own comment calls "a DUPLICATED pruning/projection
//! index". The residual tail drops the tail copy wherever the stripe reproduces
//! the value exactly.
//!
//! The quantity that claim rests on is **bytes written per row**, which is
//! deterministic: the same records produce byte-identical blocks. So this is a
//! plain `harness = false` reporter rather than a criterion benchmark — there is
//! no distribution to sample, and criterion's timing noise would only obscure an
//! exact number. Run it and the output is reproducible to the byte.
//!
//! # The measurement is deliberately PER PATH
//!
//! ADR-094's §2.1 "declared columns authoritative" cannot be global (see
//! TD-USUB-6): only a caller with an EXACT `ProximaType` can reconstruct the
//! original variant, and therefore only such a caller can drop anything from the
//! tail.
//!
//! * **Relational** (`record_store.rs`) resolves each promoted key against
//!   `CatalogTableSchema`, so it declares exactly — measured below.
//! * **Vector/SST flush** (`flush/mod.rs`) has only the coarse v1
//!   `FilterableDataType` (no integer width, no `TimeUnit`), so it declares
//!   nothing exactly and **gets no reduction by construction** — also shown
//!   below, as a control rather than as an assumption.
//!
//! Reporting one blended number across both paths would overstate the win.
//!
//! Run: `cargo bench --bench pax_residual_tail_write_amplification`

use proximadb_block_format::{BlockCompression, BlockMode, PaxBlockWriter, ShredColumn, col_id};
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_records::{ProximaRecord, ProximaTreeNode};

/// Rows per measured block. Large enough that per-block fixed overhead (header,
/// footer, column metas) does not dominate the per-row figure.
const ROWS: usize = 2_000;

/// A relational-shaped row: a few exactly-typed scalar props of the kind a
/// catalog declares, plus one undeclared prop that must survive in the tail
/// either way.
fn row(i: usize) -> ProximaRecord {
    let mut r = ProximaRecord {
        oid: format!("order-{i:06}"),
        tenant_id: "tenant-a".to_string(),
        ..Default::default()
    };
    let mut put = |k: &str, v: ProximaValue| {
        r.props.insert(k.to_string(), ProximaTreeNode::Value(v));
    };
    put("status", ProximaValue::String("shipped".to_string()));
    put("quantity", ProximaValue::Int32(i as i32 % 97));
    put(
        "unit_price",
        ProximaValue::Float64((i % 1000) as f64 * 1.25),
    );
    put(
        "customer",
        ProximaValue::String(format!("customer-{:04}", i % 512)),
    );
    // Undeclared: stays in the tail in every configuration.
    put(
        "note",
        ProximaValue::String("free-form annotation".to_string()),
    );
    r
}

fn block_bytes(columns: Vec<ShredColumn>, residual: bool) -> usize {
    // Set the mode EXPLICITLY rather than through the env gate: the gate is a
    // process-wide `OnceLock`, so toggling the variable inside one process has no
    // effect after the first read — an earlier draft of this reporter did exactly
    // that and silently measured the same mode twice, reporting a ~0.1% "win"
    // that was really a stuck flag.
    let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "orders", 0, 0)
        .with_residual_tail(residual)
        .with_shred_columns(columns);
    for i in 0..ROWS {
        w.add_record(&row(i)).expect("add_record");
    }
    w.flush().expect("flush").len()
}

fn relational_columns() -> Vec<ShredColumn> {
    vec![
        ShredColumn::typed("status", col_id::USER_BASE, ProximaType::String),
        ShredColumn::typed("quantity", col_id::USER_BASE + 1, ProximaType::Int32),
        ShredColumn::typed("unit_price", col_id::USER_BASE + 2, ProximaType::Float64),
        ShredColumn::typed("customer", col_id::USER_BASE + 3, ProximaType::String),
    ]
}

/// The SST/vector shape: the same four columns, but declared only coarsely —
/// which `with_shred_spec` models as `declared_type: None`.
fn vector_columns() -> Vec<ShredColumn> {
    vec![
        ShredColumn::untyped("status", col_id::USER_BASE),
        ShredColumn::untyped("quantity", col_id::USER_BASE + 1),
        ShredColumn::untyped("unit_price", col_id::USER_BASE + 2),
        ShredColumn::untyped("customer", col_id::USER_BASE + 3),
    ]
}

fn pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        return 0.0;
    }
    (before as f64 - after as f64) / before as f64 * 100.0
}

fn main() {
    println!("PAX residual-tail write amplification (TD-USUB-6 2b)");
    println!("rows per block: {ROWS}");
    println!("shredded props: status(String) quantity(Int32) unit_price(Float64) customer(String)");
    println!("undeclared prop retained in the tail in every configuration: note(String)");
    println!();

    let rel_off = block_bytes(relational_columns(), false);
    let rel_on = block_bytes(relational_columns(), true);
    let vec_off = block_bytes(vector_columns(), false);
    let vec_on = block_bytes(vector_columns(), true);

    println!("path        gate   block bytes   bytes/row   reduction");
    println!(
        "relational  off    {rel_off:>11}   {:>9.2}   {:>8}",
        rel_off as f64 / ROWS as f64,
        "-"
    );
    println!(
        "relational  on     {rel_on:>11}   {:>9.2}   {:>7.1}%",
        rel_on as f64 / ROWS as f64,
        pct(rel_off, rel_on)
    );
    println!(
        "vector/SST  off    {vec_off:>11}   {:>9.2}   {:>8}",
        vec_off as f64 / ROWS as f64,
        "-"
    );
    println!(
        "vector/SST  on     {vec_on:>11}   {:>9.2}   {:>7.1}%",
        vec_on as f64 / ROWS as f64,
        pct(vec_off, vec_on)
    );
    println!();

    // The control is an assertion, not a hope: a path that cannot declare an
    // exact type must not shrink, because nothing may be dropped from its tail.
    assert_eq!(
        vec_off, vec_on,
        "vector/SST path declares no exact types, so the residual gate MUST be a no-op there"
    );
    assert!(
        rel_on < rel_off,
        "relational path must shrink: {rel_on} !< {rel_off}"
    );

    println!(
        "relational reduction: {} bytes ({:.1}% of block) over {ROWS} rows",
        rel_off - rel_on,
        pct(rel_off, rel_on)
    );
    println!("vector/SST reduction: 0 bytes (no exact declared types — by construction)");
}
