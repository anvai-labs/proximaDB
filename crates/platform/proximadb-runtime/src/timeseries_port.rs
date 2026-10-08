//! Time-series operations port (ADR-094) — the runtime contract the converged
//! v2 timeseries handlers consume, with the process-global installation slot
//! the root `TimeSeriesService` fills at boot (the service itself is
//! root-resident over the native TST engine).

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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

use std::collections::HashMap;

/// Time-series operations port (ADR-094). Tenant isolation is structural: the
/// request tenant selects a per-tenant engine in the implementation, and the
/// collection name stays tenant-clean.
#[async_trait]
pub trait TimeseriesOpsPort: Send + Sync {
    async fn create_collection(&self, tenant: &str, config: TsCollectionConfig) -> anyhow::Result<()>;
    async fn list_collections(&self, tenant: &str) -> Vec<TsCollectionConfig>;
    async fn delete_collection(&self, tenant: &str, name: &str) -> bool;
    async fn ingest(
        &self,
        tenant: &str,
        collection_id: &str,
        points: Vec<TsPoint>,
    ) -> anyhow::Result<usize>;
    async fn query(
        &self,
        tenant: &str,
        collection_id: &str,
        start_time: i64,
        end_time: i64,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<TsPoint>>;
    async fn aggregate(
        &self,
        tenant: &str,
        collection_id: &str,
        start_time: i64,
        end_time: i64,
        aggregation: &str,
        bucket_ms: i64,
    ) -> anyhow::Result<Vec<serde_json::Value>>;
}

static TIMESERIES_PORT: std::sync::OnceLock<Arc<dyn TimeseriesOpsPort>> = std::sync::OnceLock::new();

/// Install the process-global timeseries port. Called by the root boot path
/// right after the concrete service singleton is initialised. Idempotent
/// (first installation wins, matching the singleton's OnceLock semantics).
pub fn install_timeseries_port(port: Arc<dyn TimeseriesOpsPort>) {
    let _ = TIMESERIES_PORT.set(port);
}

/// The installed timeseries port, if the service was initialised.
pub fn timeseries_port() -> Option<Arc<dyn TimeseriesOpsPort>> {
    TIMESERIES_PORT.get().cloned()
}
