// Tantivy-based full-text search index for logs
//
// Provides:
// - Tokenized log message indexing with BM25 ranking
// - Multi-field search (message, service, source, custom fields)
// - Phrase and fuzzy queries
// - Time-partitioned index segments for efficient pruning
// - Score-based ranking for search results

use std::sync::RwLock;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, STRING, Schema, TEXT, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::proto::proximadb_v1::{
    LogEntry, Severity, SqlValue, sql_value::Value as SqlValueVariant,
};

/// Full-text search index for log entries
pub struct TantivyLogIndex {
    /// Namespace name
    namespace: String,
    /// Tantivy index
    index: Index,
    /// Index writer (mutex-protected for thread safety)
    writer: RwLock<IndexWriter>,
    /// Index reader for searching (refreshable)
    reader: RwLock<IndexReader>,
    /// Schema
    #[allow(dead_code)]
    schema: Schema,
    /// Log ID field (stored for retrieval)
    id_field: Field,
    /// Timestamp field (for time-based filtering)
    timestamp_field: Field,
    /// Message field (main text search)
    message_field: Field,
    /// Service field
    service_field: Field,
    /// Source field
    source_field: Field,
    /// Severity field (indexed for filtering)
    severity_field: Field,
    /// Combined searchable text field (message + all fields)
    all_text_field: Field,
    /// Document count
    doc_count: RwLock<u64>,
}

/// Log search result with score
#[derive(Debug, Clone)]
pub struct LogSearchResult {
    /// Log entry ID
    pub id: String,
    /// Timestamp in nanoseconds
    pub timestamp_ns: i64,
    /// BM25 relevance score
    pub score: f32,
}

/// Search options for log queries
#[derive(Debug, Clone, Default)]
pub struct LogSearchOptions {
    /// Maximum results to return
    pub limit: usize,
    /// Time range start (optional)
    pub start_time_ns: Option<i64>,
    /// Time range end (optional)
    pub end_time_ns: Option<i64>,
    /// Filter by service
    pub service_filter: Option<String>,
    /// Filter by source
    pub source_filter: Option<String>,
    /// Filter by severity
    pub severity_filter: Option<Severity>,
    /// Fields to search (empty = all fields)
    pub search_fields: Vec<String>,
    /// Enable fuzzy matching
    pub fuzzy: bool,
}

impl LogSearchOptions {
    /// Create default options with limit
    pub fn with_limit(limit: usize) -> Self {
        Self {
            limit,
            ..Default::default()
        }
    }

    /// Set time range filter
    pub fn time_range(mut self, start_ns: i64, end_ns: i64) -> Self {
        self.start_time_ns = Some(start_ns);
        self.end_time_ns = Some(end_ns);
        self
    }

    /// Set service filter
    pub fn service(mut self, service: &str) -> Self {
        self.service_filter = Some(service.to_string());
        self
    }

    /// Set source filter
    pub fn source(mut self, source: &str) -> Self {
        self.source_filter = Some(source.to_string());
        self
    }

    /// Set severity filter
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity_filter = Some(severity);
        self
    }
}

impl TantivyLogIndex {
    /// Create a new log index for a namespace
    pub fn new(namespace: &str) -> Result<Self> {
        // Build schema with all log fields
        let mut schema_builder = Schema::builder();

        // ID field - stored for retrieval
        let id_field = schema_builder.add_text_field("_id", STORED | STRING);

        // Timestamp field - indexed as i64 for range queries
        let timestamp_field = schema_builder.add_i64_field("timestamp_ns", INDEXED | STORED | FAST);

        // Text indexing options with position tracking for phrase queries
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        // Main text fields
        let message_field = schema_builder.add_text_field("message", text_options.clone());
        let service_field = schema_builder.add_text_field("service", TEXT | STORED);
        let source_field = schema_builder.add_text_field("source", TEXT | STORED);
        let severity_field = schema_builder.add_text_field("severity", STRING | STORED);

        // Combined all-text field for broad searches
        let all_text_field = schema_builder.add_text_field("_all", TEXT);

        let schema = schema_builder.build();

        // Create in-memory index (can be extended to disk-based)
        let index = Index::create_in_ram(schema.clone());

        // Create writer with 50MB heap
        let writer = index
            .writer(50_000_000)
            .context("Failed to create log index writer")?;

        // Create reader with reload policy
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create log index reader")?;

        Ok(Self {
            namespace: namespace.to_string(),
            index,
            writer: RwLock::new(writer),
            reader: RwLock::new(reader),
            schema,
            id_field,
            timestamp_field,
            message_field,
            service_field,
            source_field,
            severity_field,
            all_text_field,
            doc_count: RwLock::new(0),
        })
    }

    /// Create a disk-based log index
    pub fn new_on_disk(namespace: &str, path: &str) -> Result<Self> {
        // Build schema with all log fields
        let mut schema_builder = Schema::builder();

        let id_field = schema_builder.add_text_field("_id", STORED | STRING);
        let timestamp_field = schema_builder.add_i64_field("timestamp_ns", INDEXED | STORED | FAST);

        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let message_field = schema_builder.add_text_field("message", text_options.clone());
        let service_field = schema_builder.add_text_field("service", TEXT | STORED);
        let source_field = schema_builder.add_text_field("source", TEXT | STORED);
        let severity_field = schema_builder.add_text_field("severity", STRING | STORED);
        let all_text_field = schema_builder.add_text_field("_all", TEXT);

        let schema = schema_builder.build();

        // The tantivy index is deliberately LOCAL (fast mmap'd search over
        // logs whose durable copy lives in the log storage) — reject
        // object-store URLs fail-fast instead of creating a literal
        // `s3:`/`adls:` directory (TD-OBJSTORE-1, #960).
        if path.contains("://") {
            anyhow::bail!(
                "tantivy log index path must be a local directory, got URL-shaped '{path}' — \
                 the index stays local; only the log STORAGE is object-store-durable"
            );
        }

        // Ensure directory exists
        std::fs::create_dir_all(path).context("Failed to create index directory")?;

        // Create or open disk-based index
        let index = Index::create_in_dir(path, schema.clone())
            .or_else(|_| Index::open_in_dir(path))
            .context("Failed to create/open disk index")?;

        let writer = index
            .writer(50_000_000)
            .context("Failed to create log index writer")?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create log index reader")?;

        Ok(Self {
            namespace: namespace.to_string(),
            index,
            writer: RwLock::new(writer),
            reader: RwLock::new(reader),
            schema,
            id_field,
            timestamp_field,
            message_field,
            service_field,
            source_field,
            severity_field,
            all_text_field,
            doc_count: RwLock::new(0),
        })
    }

    /// Index a single log entry
    pub fn index_log(&self, log_id: &str, log: &LogEntry) -> Result<()> {
        let mut doc = TantivyDocument::default();

        // Add core fields
        doc.add_text(self.id_field, log_id);
        doc.add_i64(self.timestamp_field, log.timestamp_ns);
        doc.add_text(self.message_field, &log.message);

        // Add optional fields
        if let Some(ref service) = log.service {
            doc.add_text(self.service_field, service);
        }
        if let Some(ref source) = log.source {
            doc.add_text(self.source_field, source);
        }

        // Add severity
        let severity_str = Severity::try_from(log.severity)
            .map_or_else(|_| "UNKNOWN".to_string(), |s| format!("{:?}", s));
        doc.add_text(self.severity_field, &severity_str);

        // Build combined all-text field
        let mut all_text = log.message.clone();
        if let Some(ref service) = log.service {
            all_text.push(' ');
            all_text.push_str(service);
        }
        if let Some(ref source) = log.source {
            all_text.push(' ');
            all_text.push_str(source);
        }
        // Add custom fields to all-text
        for (key, value) in &log.fields {
            all_text.push(' ');
            all_text.push_str(key);
            all_text.push(':');
            all_text.push_str(&self.sql_value_to_string(value));
        }
        doc.add_text(self.all_text_field, &all_text);

        // Add to index
        let writer = self
            .writer
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;
        writer.add_document(doc)?;

        // Update count
        let mut count = self
            .doc_count
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire doc_count lock: {}", e))?;
        *count += 1;

        Ok(())
    }

    /// Index multiple log entries in batch
    pub fn index_logs(&self, logs: &[(String, LogEntry)]) -> Result<usize> {
        let writer = self
            .writer
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;
        let mut indexed = 0;

        for (log_id, log) in logs {
            let mut doc = TantivyDocument::default();

            doc.add_text(self.id_field, log_id);
            doc.add_i64(self.timestamp_field, log.timestamp_ns);
            doc.add_text(self.message_field, &log.message);

            if let Some(ref service) = log.service {
                doc.add_text(self.service_field, service);
            }
            if let Some(ref source) = log.source {
                doc.add_text(self.source_field, source);
            }

            let severity_str = Severity::try_from(log.severity)
                .map_or_else(|_| "UNKNOWN".to_string(), |s| format!("{:?}", s));
            doc.add_text(self.severity_field, &severity_str);

            // Build all-text
            let mut all_text = log.message.clone();
            if let Some(ref service) = log.service {
                all_text.push(' ');
                all_text.push_str(service);
            }
            if let Some(ref source) = log.source {
                all_text.push(' ');
                all_text.push_str(source);
            }
            for (key, value) in &log.fields {
                all_text.push(' ');
                all_text.push_str(key);
                all_text.push(':');
                all_text.push_str(&self.sql_value_to_string(value));
            }
            doc.add_text(self.all_text_field, &all_text);

            writer.add_document(doc)?;
            indexed += 1;
        }

        // Update count
        let mut count = self
            .doc_count
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire doc_count lock: {}", e))?;
        *count += indexed as u64;

        Ok(indexed)
    }

    /// Commit pending changes to the index
    pub fn commit(&self) -> Result<()> {
        let mut writer = self
            .writer
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;
        writer.commit()?;

        // Reload reader to see newly committed documents
        let reader = self
            .reader
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire reader lock: {}", e))?;
        reader.reload()?;

        Ok(())
    }

    /// Delete a log entry from the index
    pub fn delete_log(&self, log_id: &str) -> Result<()> {
        let term = Term::from_field_text(self.id_field, log_id);
        let writer = self
            .writer
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;
        writer.delete_term(term);

        let mut count = self
            .doc_count
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire doc_count lock: {}", e))?;
        if *count > 0 {
            *count -= 1;
        }

        Ok(())
    }

    /// Delete all logs before a timestamp
    pub fn delete_before(&self, timestamp_ns: i64) -> Result<u64> {
        // Note: Tantivy doesn't support range delete directly
        // We need to search and delete individually
        let reader = self
            .reader
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire reader lock: {}", e))?;
        let searcher = reader.searcher();

        // Find all docs before timestamp
        let query = tantivy::query::RangeQuery::new(
            std::ops::Bound::Unbounded,
            std::ops::Bound::Excluded(tantivy::schema::Term::from_field_i64(
                self.timestamp_field,
                timestamp_ns,
            )),
        );

        let top_docs = searcher.search(&query, &TopDocs::with_limit(100_000).order_by_score())?;

        let writer = self
            .writer
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;
        let mut deleted = 0u64;

        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_value) = doc.get_first(self.id_field)
                && let Some(id) = id_value.as_str()
            {
                let term = Term::from_field_text(self.id_field, id);
                writer.delete_term(term);
                deleted += 1;
            }
        }

        // Update count
        let mut count = self
            .doc_count
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire doc_count lock: {}", e))?;
        *count = count.saturating_sub(deleted);

        Ok(deleted)
    }

    /// Search logs with a query string
    pub fn search(
        &self,
        query_str: &str,
        options: &LogSearchOptions,
    ) -> Result<Vec<LogSearchResult>> {
        // Determine which fields to search
        let search_fields = if options.search_fields.is_empty() {
            vec![self.message_field, self.all_text_field]
        } else {
            options
                .search_fields
                .iter()
                .filter_map(|f| match f.as_str() {
                    "message" => Some(self.message_field),
                    "service" => Some(self.service_field),
                    "source" => Some(self.source_field),
                    "_all" | "all" => Some(self.all_text_field),
                    _ => None,
                })
                .collect()
        };

        if search_fields.is_empty() {
            return Ok(vec![]);
        }

        // Parse the text query
        let query_parser = QueryParser::for_index(&self.index, search_fields);
        let text_query = query_parser
            .parse_query(query_str)
            .context("Failed to parse log search query")?;

        // Build combined query with filters
        let final_query = self.build_filtered_query(text_query, options)?;

        // Execute search
        let reader = self
            .reader
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire reader lock: {}", e))?;
        let searcher = reader.searcher();
        let limit = if options.limit > 0 {
            options.limit
        } else {
            100
        };
        let top_docs = searcher
            .search(&final_query, &TopDocs::with_limit(limit).order_by_score())
            .context("Log search failed")?;

        // Extract results
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            let id = doc
                .get_first(self.id_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default();

            let timestamp_ns = doc
                .get_first(self.timestamp_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            // Apply time filter (post-filter for accuracy)
            if let Some(start) = options.start_time_ns
                && timestamp_ns < start
            {
                continue;
            }
            if let Some(end) = options.end_time_ns
                && timestamp_ns > end
            {
                continue;
            }

            results.push(LogSearchResult {
                id,
                timestamp_ns,
                score,
            });
        }

        Ok(results)
    }

    /// Search with phrase query (exact phrase match)
    pub fn search_phrase(
        &self,
        phrase: &str,
        options: &LogSearchOptions,
    ) -> Result<Vec<LogSearchResult>> {
        // Wrap phrase in quotes for exact match
        let quoted_phrase = format!("\"{}\"", phrase.replace('"', "\\\""));
        self.search(&quoted_phrase, options)
    }

    /// Build filtered query combining text query with filters
    fn build_filtered_query(
        &self,
        text_query: Box<dyn Query>,
        options: &LogSearchOptions,
    ) -> Result<Box<dyn Query>> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, text_query)];

        // Add service filter
        if let Some(ref service) = options.service_filter {
            let term = Term::from_field_text(self.service_field, service);
            let term_query = TermQuery::new(term, IndexRecordOption::Basic);
            clauses.push((Occur::Must, Box::new(term_query)));
        }

        // Add source filter
        if let Some(ref source) = options.source_filter {
            let term = Term::from_field_text(self.source_field, source);
            let term_query = TermQuery::new(term, IndexRecordOption::Basic);
            clauses.push((Occur::Must, Box::new(term_query)));
        }

        // Add severity filter
        if let Some(severity) = options.severity_filter {
            let severity_str = format!("{:?}", severity);
            let term = Term::from_field_text(self.severity_field, &severity_str);
            let term_query = TermQuery::new(term, IndexRecordOption::Basic);
            clauses.push((Occur::Must, Box::new(term_query)));
        }

        // Add time range filter using range query
        if options.start_time_ns.is_some() || options.end_time_ns.is_some() {
            let to_term = |bound: std::ops::Bound<i64>| match bound {
                std::ops::Bound::Included(v) => std::ops::Bound::Included(
                    tantivy::schema::Term::from_field_i64(self.timestamp_field, v),
                ),
                std::ops::Bound::Excluded(v) => std::ops::Bound::Excluded(
                    tantivy::schema::Term::from_field_i64(self.timestamp_field, v),
                ),
                std::ops::Bound::Unbounded => std::ops::Bound::Unbounded,
            };

            let start = options
                .start_time_ns
                .map_or(std::ops::Bound::Unbounded, std::ops::Bound::Included);
            let end = options
                .end_time_ns
                .map_or(std::ops::Bound::Unbounded, std::ops::Bound::Included);

            let range_query = tantivy::query::RangeQuery::new(to_term(start), to_term(end));
            clauses.push((Occur::Must, Box::new(range_query)));
        }

        if clauses.len() == 1 {
            Ok(clauses
                .pop()
                .ok_or_else(|| anyhow::anyhow!("Failed to pop query from clauses"))?
                .1)
        } else {
            Ok(Box::new(BooleanQuery::new(clauses)))
        }
    }

    /// Get document count
    ///
    /// unwrap_or_else is acceptable here: This is a simple getter that returns u64 (not Result).
    /// Recovering from poisoned lock via into_inner() is valid for non-critical read operations.
    /// If the lock is poisoned (thread panicked while holding), we recover the inner value
    /// rather than failing the entire operation.
    pub fn doc_count(&self) -> u64 {
        *self.doc_count.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Get namespace name
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Convert SqlValue to string for indexing
    fn sql_value_to_string(&self, value: &SqlValue) -> String {
        // Scalars render through the ONE shared crate fn (bytes hex, Jsonb
        // JSON text, Null ""); container arms stay per-site (token text).
        if let Some(s) = super::sql_scalar_to_string(value) {
            return s;
        }
        match &value.value {
            // TD-PROTO-2 round 4: the _all fulltext field indexes the decoded
            // JSON text — a byte-length placeholder made fulltext search over
            // JSONB attributes return zero hits while the structured filter
            // path matched. (No legacy-index concern: the variant became
            // producible in this change.)
            Some(SqlValueVariant::JsonbValue(b)) => {
                proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(b)
            }
            Some(SqlValueVariant::ArrayValue(arr)) => arr
                .values
                .iter()
                .map(|v| self.sql_value_to_string(v))
                .collect::<Vec<_>>()
                .join(" "),
            Some(SqlValueVariant::ObjectValue(obj)) => obj
                .fields
                .iter()
                .map(|(k, v)| format!("{}:{}", k, self.sql_value_to_string(v)))
                .collect::<Vec<_>>()
                .join(" "),
            None => String::new(),
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_log(
        id: &str,
        message: &str,
        service: &str,
        severity: Severity,
        ts_ns: i64,
    ) -> (String, LogEntry) {
        (
            id.to_string(),
            LogEntry {
                timestamp_ns: ts_ns,
                severity: severity as i32,
                message: message.to_string(),
                fields: HashMap::new(),
                source: Some("host1".to_string()),
                service: Some(service.to_string()),
            },
        )
    }

    fn make_log_with_fields(
        id: &str,
        message: &str,
        service: &str,
        severity: Severity,
        ts_ns: i64,
        fields: HashMap<String, SqlValue>,
    ) -> (String, LogEntry) {
        (
            id.to_string(),
            LogEntry {
                timestamp_ns: ts_ns,
                severity: severity as i32,
                message: message.to_string(),
                fields,
                source: Some("host1".to_string()),
                service: Some(service.to_string()),
            },
        )
    }

    #[test]
    fn test_index_creation() {
        let index = TantivyLogIndex::new("test_ns");
        assert!(index.is_ok());
        let index = index.expect("Failed to create index");
        assert_eq!(index.namespace(), "test_ns");
        assert_eq!(index.doc_count(), 0);
    }

    #[test]
    fn test_index_single_log() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let (id, log) = make_log(
            "log1",
            "Connection timeout error",
            "api",
            Severity::Error,
            1000,
        );

        let result = index.index_log(&id, &log);
        assert!(result.is_ok());
        assert_eq!(index.doc_count(), 1);

        index.commit().expect("Failed to commit");
    }

    #[test]
    fn test_index_batch_logs() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log("log1", "Error occurred", "api", Severity::Error, 1000),
            make_log("log2", "Request completed", "web", Severity::Info, 2000),
            make_log(
                "log3",
                "Database connection failed",
                "db",
                Severity::Error,
                3000,
            ),
        ];

        let result = index.index_logs(&logs);
        assert!(result.is_ok());
        assert_eq!(result.expect("Failed to get result"), 3);
        assert_eq!(index.doc_count(), 3);

        index.commit().expect("Failed to commit");
    }

    #[test]
    fn test_search_simple() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log(
                "log1",
                "Connection timeout error",
                "api",
                Severity::Error,
                1000,
            ),
            make_log(
                "log2",
                "Request completed successfully",
                "web",
                Severity::Info,
                2000,
            ),
            make_log(
                "log3",
                "Database connection failed",
                "db",
                Severity::Error,
                3000,
            ),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for "connection"
        let options = LogSearchOptions::with_limit(10);
        let results = index
            .search("connection", &options)
            .expect("Failed to search");

        assert_eq!(results.len(), 2);
        // Both log1 and log3 contain "connection"
    }

    #[test]
    fn test_search_with_service_filter() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log("log1", "Error in api service", "api", Severity::Error, 1000),
            make_log("log2", "Error in web service", "web", Severity::Error, 2000),
            make_log("log3", "Error in api gateway", "api", Severity::Error, 3000),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for "error" with service filter
        let options = LogSearchOptions::with_limit(10).service("api");
        let results = index.search("error", &options).expect("Failed to search");

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_with_time_range() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log("log1", "Old error message", "api", Severity::Error, 1000),
            make_log("log2", "Recent error message", "api", Severity::Error, 5000),
            make_log(
                "log3",
                "Latest error message",
                "api",
                Severity::Error,
                10000,
            ),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search with time range
        let options = LogSearchOptions::with_limit(10).time_range(4000, 11000);
        let results = index.search("error", &options).expect("Failed to search");

        assert_eq!(results.len(), 2);
        // Should find log2 and log3
    }

    #[test]
    fn test_search_phrase() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log(
                "log1",
                "Connection timeout error occurred",
                "api",
                Severity::Error,
                1000,
            ),
            make_log(
                "log2",
                "Error connection timeout",
                "api",
                Severity::Error,
                2000,
            ),
            make_log(
                "log3",
                "Timeout in connection pool",
                "api",
                Severity::Error,
                3000,
            ),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for exact phrase
        let options = LogSearchOptions::with_limit(10);
        let results = index
            .search_phrase("connection timeout", &options)
            .expect("Failed to search phrase");

        // Only log1 has exact phrase "connection timeout"
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_with_severity_filter() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log("log1", "Application started", "api", Severity::Info, 1000),
            make_log("log2", "Application error", "api", Severity::Error, 2000),
            make_log("log3", "Application warning", "api", Severity::Warn, 3000),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for "application" with severity filter
        let options = LogSearchOptions::with_limit(10).severity(Severity::Error);
        let results = index
            .search("application", &options)
            .expect("Failed to search");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "log2");
    }

    #[test]
    fn test_delete_log() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log("log1", "First message", "api", Severity::Info, 1000),
            make_log("log2", "Second message", "api", Severity::Info, 2000),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        assert_eq!(index.doc_count(), 2);

        // Delete one log
        index.delete_log("log1").expect("Failed to delete log");
        index.commit().expect("Failed to commit");

        assert_eq!(index.doc_count(), 1);
    }

    #[test]
    fn test_search_with_custom_fields() {
        let mut fields = HashMap::new();
        fields.insert(
            "request_id".to_string(),
            SqlValue {
                value: Some(SqlValueVariant::StringValue("req-12345".to_string())),
            },
        );

        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![
            make_log_with_fields(
                "log1",
                "Request processed",
                "api",
                Severity::Info,
                1000,
                fields,
            ),
            make_log("log2", "Another request", "api", Severity::Info, 2000),
        ];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for request_id value (indexed in _all field)
        let options = LogSearchOptions::with_limit(10);
        let results = index
            .search("req-12345", &options)
            .expect("Failed to search");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "log1");
    }

    #[test]
    fn fulltext_search_indexes_decoded_jsonb_fields() {
        let document = serde_json::json!({"request_id": "req-jsonb-42"});
        let fields = [(
            "context".to_string(),
            SqlValue {
                value: Some(SqlValueVariant::JsonbValue(
                    proximadb_data_model::ProximaValue::to_jsonb_vec(&document)
                        .expect("encode JSONB test value"),
                )),
            },
        )]
        .into_iter()
        .collect();
        let index = TantivyLogIndex::new("test_ns").expect("create index");
        let logs = vec![make_log_with_fields(
            "jsonb-log",
            "Request processed",
            "api",
            Severity::Info,
            1000,
            fields,
        )];

        index.index_logs(&logs).expect("index logs");
        index.commit().expect("commit index");
        let results = index
            .search("req-jsonb-42", &LogSearchOptions::with_limit(10))
            .expect("search JSONB contents");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "jsonb-log");
    }

    #[test]
    fn test_search_options_builder() {
        let options = LogSearchOptions::with_limit(50)
            .time_range(1000, 5000)
            .service("api")
            .source("host1")
            .severity(Severity::Error);

        assert_eq!(options.limit, 50);
        assert_eq!(options.start_time_ns, Some(1000));
        assert_eq!(options.end_time_ns, Some(5000));
        assert_eq!(options.service_filter, Some("api".to_string()));
        assert_eq!(options.source_filter, Some("host1".to_string()));
        assert_eq!(options.severity_filter, Some(Severity::Error));
    }

    #[test]
    fn test_empty_search_results() {
        let index = TantivyLogIndex::new("test_ns").expect("Failed to create index");
        let logs = vec![make_log(
            "log1",
            "Connection error",
            "api",
            Severity::Error,
            1000,
        )];

        index.index_logs(&logs).expect("Failed to index logs");
        index.commit().expect("Failed to commit");

        // Search for non-existent term
        let options = LogSearchOptions::with_limit(10);
        let results = index
            .search("nonexistent_term_xyz", &options)
            .expect("Failed to search");

        assert!(results.is_empty());
    }
}
