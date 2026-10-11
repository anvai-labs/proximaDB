// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! In-process service that surfaces the TST time-series engine (TD-TS-1).
//!
//! The `TimeSeriesEngine` (`crate::storage::engines::tst`) is a full time-partitioned
//! columnar engine, but its native methods sit on no trait and were unreachable from
//! the network layer. This service wraps it so the v2 REST timeseries surface
//! (`src/network/rest/v2/timeseries.rs`) can create collections, ingest points, and
//! query / aggregate — the contract the Python SDK already expects. It calls the
//! TST-native methods (`insert_record` / `query_time_range`) directly, never the
//! stubbed vector-shaped trait methods.

use crate::proto::proximadb_v1::{SqlValue, VectorRecord, sql_value};
use crate::storage::engines::tst::{TimeSeriesConfig, TimeSeriesEngine};
use anyhow::{Result, anyhow};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;

static TIMESERIES_SERVICE: OnceLock<Arc<TimeSeriesService>> = OnceLock::new();

/// Initialise the process-global time-series service (idempotent — first wins).
/// Called once at server bootstrap with `<data_dir>/timeseries` as the base path.
pub fn init_timeseries_service(base_path: PathBuf) -> Result<()> {
    if TIMESERIES_SERVICE.get().is_some() {
        return Ok(());
    }
    let service = Arc::new(TimeSeriesService::new(base_path)?);
    let _ = TIMESERIES_SERVICE.set(service);
    Ok(())
}

/// The process-global time-series service, if initialised.
pub fn timeseries_service() -> Option<Arc<TimeSeriesService>> {
    TIMESERIES_SERVICE.get().cloned()
}

/// A value column in a time-series collection (mirrors the SDK `ValueColumn`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsValueColumn {
    pub name: String,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub aggregation: Option<String>,
}

/// Time-series collection config (mirrors the SDK `TimeSeriesCollectionConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsCollectionConfig {
    pub name: String,
    #[serde(default = "default_timestamp_column")]
    pub timestamp_column: String,
    #[serde(default)]
    pub value_columns: Vec<TsValueColumn>,
    #[serde(default)]
    pub tag_columns: Vec<String>,
    #[serde(default)]
    pub retention_ms: Option<i64>,
}

fn default_timestamp_column() -> String {
    "timestamp".to_string()
}

/// A single time-series point: epoch-millis timestamp + named numeric values + string tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsPoint {
    pub timestamp: i64,
    #[serde(default)]
    pub values: HashMap<String, f64>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

/// Multiplexes a `TimeSeriesEngine` PER TENANT — each rooted at `<base_path>/<tenant>` — so
/// tenants are isolated STRUCTURALLY by physical storage path, not by folding the tenant into the
/// collection name. Collection configs are registered per `(tenant, name)`. Logical collection
/// names stay tenant-clean; the tenant is the structural key dimension, applied once here.
pub struct TimeSeriesService {
    /// Root under which each tenant's engine hangs (`<base_path>/<tenant>`).
    base_path: PathBuf,
    /// One engine per tenant, created lazily. Each `insert_record` needs `&mut engine`, so every
    /// tenant's engine is independently locked (writes exclusive, queries shared) — and no
    /// tenant's partitions/WAL/segments ever share a path with another's.
    engines: RwLock<HashMap<String, Arc<RwLock<TimeSeriesEngine>>>>,
    /// Declared collection configs, keyed by `(tenant, collection_name)`.
    collections: RwLock<HashMap<(String, String), TsCollectionConfig>>,
}

impl TimeSeriesService {
    /// Build the service rooted at `base_path` (e.g. `<data_dir>/timeseries`); per-tenant engines
    /// hang off `<base_path>/<tenant>`.
    pub fn new(base_path: PathBuf) -> Result<Self> {
        Ok(Self {
            base_path,
            engines: RwLock::new(HashMap::new()),
            collections: RwLock::new(HashMap::new()),
        })
    }

    /// Get — or lazily create — the engine for `tenant`, rooted at `<base_path>/<tenant>`. The
    /// tenant is validated as a path segment (foundation `validate_request_tenant`) so it can
    /// never traverse or shadow a control-plane system subtree.
    async fn engine_for(&self, tenant: &str) -> Result<Arc<RwLock<TimeSeriesEngine>>> {
        proximadb_tenant::validate_request_tenant(tenant)
            .map_err(|e| anyhow!("invalid tenant '{tenant}': {e}"))?;
        if let Some(engine) = self.engines.read().await.get(tenant) {
            return Ok(engine.clone());
        }
        let mut engines = self.engines.write().await;
        // Re-check under the write lock — another task may have created it meanwhile.
        if let Some(engine) = engines.get(tenant) {
            return Ok(engine.clone());
        }
        let config = TimeSeriesConfig {
            base_path: self.base_path.join(tenant),
            ..Default::default()
        };
        let mut engine = TimeSeriesEngine::with_config(config)
            .map_err(|e| anyhow!("failed to init TimeSeriesEngine for tenant '{tenant}': {e}"))?;
        // F3 / TD-WAL-2 (durability parity): make timeseries writes crash-durable.
        // The WAL writer (`TstWalWriter`) and the replay handler (`TstWalRecovery`)
        // already exist and are engine-tested; only this service wiring was missing
        // — `engine_for` never enabled the WAL, so inserts landed only in in-memory
        // partitions and were LOST on a crash before partition rotation. Enable the
        // append writer (creates the file on first run), then replay any pre-crash
        // entries into this fresh engine's partitions before it serves. The WAL is
        // rooted per-tenant next to the partitions.
        let wal_path = self.base_path.join(tenant).join("wal");
        engine
            .enable_wal(&wal_path)
            .await
            .map_err(|e| anyhow!("failed to enable TimeSeries WAL for tenant '{tenant}': {e}"))?;
        engine
            .recover_from_wal(&wal_path)
            .await
            .map_err(|e| anyhow!("failed to recover TimeSeries WAL for tenant '{tenant}': {e}"))?;
        let handle = Arc::new(RwLock::new(engine));
        engines.insert(tenant.to_string(), handle.clone());
        Ok(handle)
    }

    pub async fn create_collection(&self, tenant: &str, config: TsCollectionConfig) -> Result<()> {
        proximadb_tenant::validate_request_tenant(tenant)
            .map_err(|e| anyhow!("invalid tenant '{tenant}': {e}"))?;
        self.collections
            .write()
            .await
            .insert((tenant.to_string(), config.name.clone()), config);
        Ok(())
    }

    /// List a SINGLE tenant's collections (structural scope — never all tenants').
    pub async fn list_collections(&self, tenant: &str) -> Vec<TsCollectionConfig> {
        self.collections
            .read()
            .await
            .iter()
            .filter(|((t, _), _)| t == tenant)
            .map(|(_, cfg)| cfg.clone())
            .collect()
    }

    pub async fn delete_collection(&self, tenant: &str, name: &str) -> bool {
        self.collections
            .write()
            .await
            .remove(&(tenant.to_string(), name.to_string()))
            .is_some()
    }

    /// Ingest points → TST-native `insert_record` (values in metadata, timestamp partitioned),
    /// into the TENANT's engine.
    pub async fn ingest(
        &self,
        tenant: &str,
        collection: &str,
        points: Vec<TsPoint>,
    ) -> Result<usize> {
        let engine_handle = self.engine_for(tenant).await?;
        let mut engine = engine_handle.write().await;
        let mut inserted = 0usize;
        for point in points {
            let ts = Utc
                .timestamp_millis_opt(point.timestamp)
                .single()
                .ok_or_else(|| anyhow!("invalid timestamp {}", point.timestamp))?;
            let mut metadata: HashMap<String, SqlValue> = HashMap::new();
            let mut vector: Vec<f32> = Vec::with_capacity(point.values.len());
            for (k, v) in &point.values {
                metadata.insert(
                    k.clone(),
                    SqlValue {
                        value: Some(sql_value::Value::NumberValue(*v)),
                    },
                );
                vector.push(*v as f32);
            }
            for (k, v) in &point.tags {
                metadata.insert(
                    k.clone(),
                    SqlValue {
                        value: Some(sql_value::Value::StringValue(v.clone())),
                    },
                );
            }
            let record = VectorRecord {
                id: format!("{collection}:{}", point.timestamp),
                vector,
                metadata,
                timestamp: Some(point.timestamp),
                ..Default::default()
            };
            engine
                .insert_record(collection, ts, record)
                .await
                .map_err(|e| anyhow!("insert_record failed: {e}"))?;
            inserted += 1;
        }
        // F3 / TD-WAL-2: fsync the WAL once per ingest batch so the points are
        // crash-durable before we return success (the append only buffers).
        engine
            .flush_wal()
            .await
            .map_err(|e| anyhow!("failed to flush TimeSeries WAL: {e}"))?;
        Ok(inserted)
    }

    /// Query a time range → raw points (TST-native `query_time_range`) from the TENANT's engine.
    pub async fn query(
        &self,
        tenant: &str,
        collection: &str,
        start_ms: i64,
        end_ms: i64,
        limit: Option<usize>,
    ) -> Result<Vec<TsPoint>> {
        let start = Utc
            .timestamp_millis_opt(start_ms)
            .single()
            .ok_or_else(|| anyhow!("invalid start_time"))?;
        let end = Utc
            .timestamp_millis_opt(end_ms)
            .single()
            .ok_or_else(|| anyhow!("invalid end_time"))?;
        let engine_handle = self.engine_for(tenant).await?;
        let records = engine_handle
            .read()
            .await
            .query_time_range(collection, start, end, limit)
            .await
            .map_err(|e| anyhow!("query_time_range failed: {e}"))?;
        Ok(records.into_iter().map(record_to_point).collect())
    }

    /// Aggregate a time range into fixed `bucket_ms` buckets, applying `aggregation`
    /// (avg | sum | min | max | count | stddev | first | last) to each value column.
    pub async fn aggregate(
        &self,
        tenant: &str,
        collection: &str,
        start_ms: i64,
        end_ms: i64,
        aggregation: &str,
        bucket_ms: i64,
    ) -> Result<Vec<serde_json::Value>> {
        let points = self
            .query(tenant, collection, start_ms, end_ms, None)
            .await?;
        let bucket = bucket_ms.max(1);
        let mut buckets: BTreeMap<i64, HashMap<String, Vec<f64>>> = BTreeMap::new();
        for point in points {
            let key = (point.timestamp / bucket) * bucket;
            let cols = buckets.entry(key).or_default();
            for (name, value) in point.values {
                cols.entry(name).or_default().push(value);
            }
        }
        Ok(buckets
            .into_iter()
            .map(|(bucket_start, cols)| {
                let values: serde_json::Map<String, serde_json::Value> = cols
                    .into_iter()
                    .map(|(name, vals)| {
                        (
                            name,
                            serde_json::json!(apply_aggregation(aggregation, &vals)),
                        )
                    })
                    .collect();
                serde_json::json!({ "bucket_start": bucket_start, "values": values })
            })
            .collect())
    }
}

fn record_to_point(record: VectorRecord) -> TsPoint {
    let mut values = HashMap::new();
    let mut tags = HashMap::new();
    for (k, sv) in record.metadata {
        match sv.value {
            Some(sql_value::Value::NumberValue(n)) => {
                values.insert(k, n);
            }
            Some(sql_value::Value::Int64Value(i)) => {
                values.insert(k, i as f64);
            }
            Some(sql_value::Value::StringValue(s)) => {
                tags.insert(k, s);
            }
            // TD-PROTO-2: JSONB becomes a compact-JSON tag instead of being
            // silently dropped by the `_`.
            Some(sql_value::Value::JsonbValue(b)) => {
                tags.insert(
                    k,
                    proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(&b),
                );
            }
            _ => {}
        }
    }
    TsPoint {
        timestamp: record.timestamp.unwrap_or(0),
        values,
        tags,
    }
}

fn apply_aggregation(aggregation: &str, vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let n = vals.len() as f64;
    match aggregation {
        "sum" => vals.iter().sum(),
        "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "count" => n,
        "first" => vals[0],
        "last" => *vals.last().unwrap_or(&0.0),
        "stddev" => {
            let mean = vals.iter().sum::<f64>() / n;
            (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
        }
        // "avg" / "mean" and anything else → mean
        _ => vals.iter().sum::<f64>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn point(ts: i64, metric: &str, v: f64) -> TsPoint {
        TsPoint {
            timestamp: ts,
            values: HashMap::from([(metric.to_string(), v)]),
            tags: HashMap::new(),
        }
    }

    fn config(name: &str) -> TsCollectionConfig {
        TsCollectionConfig {
            name: name.to_string(),
            timestamp_column: "timestamp".to_string(),
            value_columns: vec![],
            tag_columns: vec![],
            retention_ms: None,
        }
    }

    /// The canonical isolation rubric: two tenants write the SAME tenant-clean logical
    /// collection name, and neither can read the other's data — isolation is structural
    /// (a per-tenant engine rooted at a separate path), NOT a `{tenant}::name` fold.
    #[tokio::test]
    async fn tenants_are_isolated_under_the_same_logical_collection_name() {
        let tmp = TempDir::new().expect("tempdir");
        let svc = TimeSeriesService::new(tmp.path().to_path_buf()).expect("service");

        // Both tenants use the identical, tenant-CLEAN collection name "sensor".
        svc.ingest("acme", "sensor", vec![point(1_000, "v", 1.0)])
            .await
            .expect("acme ingest");
        svc.ingest("globex", "sensor", vec![point(1_000, "v", 99.0)])
            .await
            .expect("globex ingest");

        let acme = svc
            .query("acme", "sensor", 0, 10_000, None)
            .await
            .expect("acme query");
        let globex = svc
            .query("globex", "sensor", 0, 10_000, None)
            .await
            .expect("globex query");

        // Each tenant sees ONLY its own value — no cross-tenant bleed despite the shared name.
        assert_eq!(acme.len(), 1, "acme sees only its own point");
        assert_eq!(acme[0].values.get("v"), Some(&1.0));
        assert_eq!(globex.len(), 1, "globex sees only its own point");
        assert_eq!(globex[0].values.get("v"), Some(&99.0));

        // A tenant that never wrote "sensor" reads nothing (no shared global engine).
        let stranger = svc
            .query("stranger", "sensor", 0, 10_000, None)
            .await
            .expect("stranger query");
        assert!(stranger.is_empty(), "unrelated tenant sees no data");
    }

    /// F3 / TD-WAL-2 (durability parity): a timeseries write survives a simulated
    /// restart via the WAL. Before the service-level WAL wiring, `engine_for` never
    /// enabled the WAL, so an unflushed point was lost the moment the in-memory
    /// engine dropped — this test failed (recovered 0). With the wiring, `ingest`
    /// appends to the per-tenant WAL and a fresh service on the same base path
    /// replays it.
    #[tokio::test]
    async fn timeseries_write_survives_restart_via_wal() {
        let tmp = TempDir::new().expect("tempdir");
        let base = tmp.path().to_path_buf();

        // First "process": ingest a point, then drop the service (simulated crash —
        // in-memory partitions are gone; the WAL persists under base/<tenant>/wal).
        {
            let svc = TimeSeriesService::new(base.clone()).expect("service");
            svc.ingest("acme", "sensor", vec![point(1_000, "v", 42.0)])
                .await
                .expect("ingest");
        } // svc dropped here → engine + WAL writer dropped (WAL flushed to disk)

        // Second "process": a fresh service on the SAME base path recovers from WAL.
        let svc2 = TimeSeriesService::new(base).expect("service restart");
        let recovered = svc2
            .query("acme", "sensor", 0, 10_000, None)
            .await
            .expect("query after restart");
        assert_eq!(
            recovered.len(),
            1,
            "the pre-restart point must be recovered from the WAL"
        );
        assert_eq!(recovered[0].values.get("v"), Some(&42.0));
    }

    /// `list_collections` is tenant-scoped — the former cross-tenant leak is gone.
    #[tokio::test]
    async fn list_collections_returns_only_the_requesting_tenants_collections() {
        let tmp = TempDir::new().expect("tempdir");
        let svc = TimeSeriesService::new(tmp.path().to_path_buf()).expect("service");

        svc.create_collection("acme", config("sensor"))
            .await
            .expect("acme create");
        svc.create_collection("globex", config("sensor"))
            .await
            .expect("globex create");
        svc.create_collection("globex", config("weather"))
            .await
            .expect("globex create 2");

        let acme = svc.list_collections("acme").await;
        assert_eq!(acme.len(), 1, "acme has exactly one collection");
        assert_eq!(acme[0].name, "sensor");

        let globex = svc.list_collections("globex").await;
        assert_eq!(globex.len(), 2, "globex has two, none of acme's");
    }

    /// The tenant is validated as a structural path segment — traversal / reserved
    /// prefixes are fail-closed before an engine is ever created.
    #[tokio::test]
    async fn invalid_tenants_are_rejected_fail_closed() {
        let tmp = TempDir::new().expect("tempdir");
        let svc = TimeSeriesService::new(tmp.path().to_path_buf()).expect("service");

        assert!(
            svc.ingest("../evil", "sensor", vec![]).await.is_err(),
            "path traversal rejected"
        );
        assert!(
            svc.ingest("_system", "sensor", vec![]).await.is_err(),
            "reserved-prefix tenant rejected"
        );
        assert!(
            svc.create_collection("bad/tenant", config("sensor"))
                .await
                .is_err(),
            "separator in tenant rejected"
        );
    }
}
