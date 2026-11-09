//! # External Table Strategy
//!
//! Compile-safe facade strategy for SQL that references external tables.
//!
//! The strategy currently performs lightweight detection only. Execution is
//! intentionally explicit: external-table scans are recognized, then rejected
//! with a clear "not wired" error instead of silently falling through to the
//! wrong strategy or returning fabricated results.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use tracing::{info, instrument};

use crate::catalog::CatalogManager;
use crate::query::facade::{
    ExecutionMetrics, QueryContent, QueryContext, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy, QueryType,
};

/// External table strategy for SQL that references catalog-backed tables.
pub struct ExternalTableStrategy {
    catalog_manager: Arc<CatalogManager>,
    priority: i32,
}

impl ExternalTableStrategy {
    /// Create a new strategy.
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self {
            catalog_manager,
            priority: 70,
        }
    }

    /// Override strategy priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    fn extract_sql<'a>(&self, request: &'a QueryRequest) -> Result<&'a str> {
        match &request.content {
            QueryContent::Sql(query) => Ok(query.as_str()),
            _ => Err(anyhow!("ExternalTableStrategy requires SQL content")),
        }
    }

    fn has_external_tables(&self, sql: &str) -> bool {
        let normalized = sql.to_ascii_lowercase();

        normalized.contains("iceberg.")
            || normalized.contains("delta.")
            || normalized.contains("glue.")
            || normalized.contains("parquet.")
            || normalized.contains("external.")
    }
}

#[async_trait]
impl QueryStrategy for ExternalTableStrategy {
    fn name(&self) -> &str {
        "external_table"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        if request.query_type != QueryType::Sql && request.query_type != QueryType::Federated {
            return false;
        }

        let Ok(sql) = self.extract_sql(request) else {
            return false;
        };

        self.has_external_tables(sql)
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    #[instrument(skip(self, request, _ctx))]
    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let start = Instant::now();
        let sql = self.extract_sql(&request)?;

        info!("External table query detected by facade");

        let _ = &self.catalog_manager;
        let _metrics = ExecutionMetrics {
            execution_path: "external_table".to_string(),
            strategy_name: self.name().to_string(),
            execution_time_ms: start.elapsed().as_millis() as u64,
            planning_time_ms: 0,
            results_scanned: 0,
            results_returned: 0,
            cache_hit: false,
            extra: serde_json::json!({ "sql": sql }),
        };

        let _unused_result = QueryResult {
            data: QueryResultData::Rows(Vec::new()),
            metrics: None,
        };

        Err(anyhow!(
            "External table queries are detected but not wired into live execution yet"
        ))
    }
}

/// Placeholder scanner surface for future external table execution.
pub struct ExternalTableScanner {
    catalog_manager: Arc<CatalogManager>,
}

impl ExternalTableScanner {
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self { catalog_manager }
    }

    pub async fn scan_table(
        &self,
        _catalog: &str,
        _namespace: &[String],
        _table: &str,
        _projections: Vec<String>,
        _predicates: Vec<String>,
    ) -> Result<Vec<RecordBatch>> {
        let _ = &self.catalog_manager;
        Err(anyhow!(
            "External table scanning is not wired into live execution yet"
        ))
    }

    pub async fn get_table_stats(
        &self,
        _catalog: &str,
        _namespace: &[String],
        _table: &str,
    ) -> Result<ExternalTableStatistics> {
        let _ = &self.catalog_manager;
        Err(anyhow!(
            "External table statistics are not wired into live execution yet"
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExternalTableStatistics {
    pub row_count: u64,
    pub num_files: u64,
    pub size_bytes: u64,
}

/// Placeholder predicate pushdown surface for future execution.
pub struct ExternalPredicatePushdown {
    catalog_manager: Arc<CatalogManager>,
}

impl ExternalPredicatePushdown {
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self { catalog_manager }
    }

    pub fn pushdown_predicates(
        &self,
        _catalog: &str,
        _namespace: &[String],
        _table: &str,
        _predicates: &[String],
    ) -> Result<Vec<PushedPredicate>> {
        let _ = &self.catalog_manager;
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct PushedPredicate {
    pub column: String,
    pub expression: String,
    pub partition_prunable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::facade::QueryRequest;

    #[test]
    fn test_has_external_tables() {
        let strategy = ExternalTableStrategy::new(Arc::new(CatalogManager::new()));

        assert!(strategy.has_external_tables("SELECT * FROM iceberg.db.table"));
        assert!(strategy.has_external_tables("SELECT * FROM delta.db.table"));
        assert!(strategy.has_external_tables("SELECT * FROM glue.db.table"));
        assert!(!strategy.has_external_tables("SELECT * FROM local_table"));
    }

    #[test]
    fn test_priority() {
        let strategy = ExternalTableStrategy::new(Arc::new(CatalogManager::new()));
        assert_eq!(strategy.priority(), 70);
    }

    #[test]
    fn test_can_handle_external_sql_and_federated_queries() {
        let strategy = ExternalTableStrategy::new(Arc::new(CatalogManager::new()));

        assert!(strategy.can_handle(&QueryRequest::sql("SELECT * FROM iceberg.analytics.events")));
        assert!(strategy.can_handle(&QueryRequest::federated(
            "SELECT * FROM parquet.lakehouse.docs"
        )));
        assert!(!strategy.can_handle(&QueryRequest::sql("SELECT * FROM local_table")));
        assert!(!strategy.can_handle(&QueryRequest::vector_search(vec![0.1, 0.2], 5)));
    }

    #[tokio::test]
    async fn test_execute_returns_explicit_not_wired_error() {
        let strategy = ExternalTableStrategy::new(Arc::new(CatalogManager::new()));
        let request = QueryRequest::sql("SELECT * FROM iceberg.analytics.events");
        let ctx = QueryContext::new(1000);

        let err = strategy.execute(request, &ctx).await.unwrap_err();

        assert!(
            err.to_string()
                .contains("not wired into live execution yet"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_scanner_returns_explicit_not_wired_errors() {
        let scanner = ExternalTableScanner::new(Arc::new(CatalogManager::new()));

        let scan_err = scanner
            .scan_table(
                "iceberg",
                &["analytics".to_string()],
                "events",
                vec!["id".to_string()],
                vec!["id > 10".to_string()],
            )
            .await
            .unwrap_err();
        assert!(
            scan_err
                .to_string()
                .contains("not wired into live execution yet"),
            "unexpected scan error: {scan_err}"
        );

        let stats_err = scanner
            .get_table_stats("iceberg", &["analytics".to_string()], "events")
            .await
            .unwrap_err();
        assert!(
            stats_err
                .to_string()
                .contains("not wired into live execution yet"),
            "unexpected stats error: {stats_err}"
        );
    }

    #[test]
    fn test_predicate_pushdown_is_explicit_noop() {
        let pushdown = ExternalPredicatePushdown::new(Arc::new(CatalogManager::new()));

        let pushed = pushdown
            .pushdown_predicates(
                "iceberg",
                &["analytics".to_string()],
                "events",
                &["event_type = 'click'".to_string()],
            )
            .unwrap();

        assert!(pushed.is_empty());
    }
}
