//! Integration test for the financial Monte Carlo option-pricing workload.
//!
//! Demonstrates the end-to-end DataFusion path the notebook/Spark-migration blueprint
//! targets (`docs/12-design/PROXIMA_NOTEBOOK_PSEUDO_DISTRIBUTED_BLUEPRINT_2026_06_04.adoc`):
//!
//!   1. option contracts are stored as **Parquet** and read through ProximaDB's canonical
//!      `FileSystem` trait (local `file://` here; `s3://` in production via the same code);
//!   2. each Parquet row group becomes a scan split → real intra-node parallelism;
//!   3. the `mc_price` DataFusion scalar UDF prices each contract via Monte Carlo;
//!   4. results are validated against the closed-form Black-Scholes oracle.
//!
//! Run: `cargo test --features datafusion-integration --test financial_option_pricing_test`.

#![cfg(feature = "datafusion-integration")]

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use proximadb::compute::montecarlo::black_scholes;
use proximadb::datafusion::{create_session_context, register_parquet_path};
use proximadb::storage::persistence::filesystem::FilesystemFactory;

/// One synthetic option contract.
#[derive(Clone)]
struct Contract {
    id: String,
    spot: f64,
    strike: f64,
    vol: f64,
    rate: f64,
    t: f64,
    is_call: bool,
}

/// Generate a deterministic set of ATM/ITM-ish contracts (substantial prices keep the
/// relative-error check meaningful) and cycle combinations to exceed one row group.
fn make_contracts(n: usize) -> Vec<Contract> {
    let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
    let vols = [0.15, 0.25];
    let ts = [0.5, 1.0];
    (0..n)
        .map(|i| {
            let strike = strikes[i % strikes.len()];
            let vol = vols[(i / strikes.len()) % vols.len()];
            let t = ts[(i / (strikes.len() * vols.len())) % ts.len()];
            Contract {
                id: format!("opt_{i:05}"),
                spot: 100.0,
                strike,
                vol,
                rate: 0.03,
                t,
                is_call: i % 2 == 0,
            }
        })
        .collect()
}

/// Write the contracts to a Parquet file with a small row-group size so the file contains
/// multiple row groups (→ multiple scan splits).
fn write_parquet(path: &std::path::Path, contracts: &[Contract], max_row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("spot", DataType::Float64, false),
        Field::new("strike", DataType::Float64, false),
        Field::new("vol", DataType::Float64, false),
        Field::new("rate", DataType::Float64, false),
        Field::new("t", DataType::Float64, false),
        Field::new("is_call", DataType::Boolean, false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            contracts.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            contracts.iter().map(|c| c.spot).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            contracts.iter().map(|c| c.strike).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            contracts.iter().map(|c| c.vol).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            contracts.iter().map(|c| c.rate).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            contracts.iter().map(|c| c.t).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            contracts.iter().map(|c| c.is_call).collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(max_row_group_size))
        .build();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn mc_price_over_parquet_matches_black_scholes() {
    let tmp = tempfile::tempdir().unwrap();
    let parquet_path = tmp.path().join("options.parquet");

    let contracts = make_contracts(512);
    write_parquet(&parquet_path, &contracts, 256); // 512 rows / 256 -> 2 row groups

    let url = format!("file://{}", parquet_path.display());

    // Read through the canonical FileSystem trait (local backend here).
    let factory = FilesystemFactory::create_default().await.unwrap();
    let fs = factory.get_filesystem(&url).unwrap();

    let ctx = create_session_context().unwrap();
    let table = register_parquet_path(&ctx, fs, "options", &url)
        .await
        .unwrap();

    // The Parquet file has multiple row groups → multiple splits → real parallelism.
    assert!(
        table.split_count() > 1,
        "expected multiple row groups (splits), got {}",
        table.split_count()
    );

    let df = ctx
        .sql("SELECT id, mc_price(spot, strike, vol, rate, t, is_call, 200000) AS price FROM options")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let mut prices: HashMap<String, f64> = HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let price_col = batch
            .column_by_name("price")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            prices.insert(ids.value(i).to_string(), price_col.value(i));
        }
    }

    assert_eq!(
        prices.len(),
        contracts.len(),
        "every contract must be priced"
    );

    // Each Monte Carlo price (200k paths, deterministic seed) must track the closed-form
    // Black-Scholes value. Tolerance covers sampling error; a small absolute floor handles
    // lower-priced contracts.
    for c in &contracts {
        let mc = *prices.get(&c.id).expect("price for contract");
        let bs = black_scholes(c.spot, c.strike, c.vol, c.rate, c.t, c.is_call);
        let tol = 0.05 * bs.abs() + 0.2;
        assert!(
            (mc - bs).abs() <= tol,
            "{}: mc={mc:.4} bs={bs:.4} (strike={}, vol={}, t={}, call={})",
            c.id,
            c.strike,
            c.vol,
            c.t,
            c.is_call
        );
    }
}
