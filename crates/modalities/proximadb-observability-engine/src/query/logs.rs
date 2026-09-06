// Log query engine
//
// Provides:
// - Full-text search in log messages (via Tantivy - see ObservabilityStorage::search_logs_tantivy)
// - Field-based filtering (via LogFilter for in-memory filtering)
// - Aggregations (count, histogram)
// - Pattern detection
//
// Note: This module provides simple in-memory log filtering. For full-text search
// with BM25 ranking and phrase matching, use the Tantivy integration available
// through ObservabilityStorage::search_logs_tantivy in mod.rs.

use std::collections::HashMap;

use crate::proto::proximadb_v1::{LogEntry, Severity, SqlValue};

/// Log query builder
pub struct LogQueryBuilder {
    /// Time range start
    start_time_ns: Option<i64>,
    /// Time range end
    end_time_ns: Option<i64>,
    /// Service filter
    services: Vec<String>,
    /// Source filter
    sources: Vec<String>,
    /// Severity filter
    severities: Vec<Severity>,
    /// Attribute filters
    attributes: HashMap<String, String>,
    /// Text search query
    text_query: Option<String>,
    /// Limit
    limit: usize,
    /// Offset
    offset: usize,
}

impl LogQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            start_time_ns: None,
            end_time_ns: None,
            services: Vec::new(),
            sources: Vec::new(),
            severities: Vec::new(),
            attributes: HashMap::new(),
            text_query: None,
            limit: 100,
            offset: 0,
        }
    }

    /// Set time range
    pub fn time_range(mut self, start_ns: i64, end_ns: i64) -> Self {
        self.start_time_ns = Some(start_ns);
        self.end_time_ns = Some(end_ns);
        self
    }

    /// Filter by service
    pub fn service(mut self, service: &str) -> Self {
        self.services.push(service.to_string());
        self
    }

    /// Filter by source
    pub fn source(mut self, source: &str) -> Self {
        self.sources.push(source.to_string());
        self
    }

    /// Filter by severity
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severities.push(severity);
        self
    }

    /// Filter by attribute
    pub fn attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Set text search query
    pub fn text(mut self, query: &str) -> Self {
        self.text_query = Some(query.to_string());
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set result offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Build the query
    pub fn build(self) -> LogQuery {
        LogQuery {
            start_time_ns: self.start_time_ns.unwrap_or(0),
            end_time_ns: self.end_time_ns.unwrap_or(i64::MAX),
            services: self.services,
            sources: self.sources,
            severities: self.severities,
            attributes: self.attributes,
            text_query: self.text_query,
            limit: self.limit,
            offset: self.offset,
        }
    }
}

impl Default for LogQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled log query
#[derive(Debug, Clone)]
pub struct LogQuery {
    /// Time range start
    pub start_time_ns: i64,
    /// Time range end
    pub end_time_ns: i64,
    /// Service filter
    pub services: Vec<String>,
    /// Source filter
    pub sources: Vec<String>,
    /// Severity filter
    pub severities: Vec<Severity>,
    /// Attribute filters
    pub attributes: HashMap<String, String>,
    /// Text search query
    pub text_query: Option<String>,
    /// Limit
    pub limit: usize,
    /// Offset
    pub offset: usize,
}

impl LogQuery {
    /// Check if a log entry matches this query
    pub fn matches(&self, log: &LogEntry) -> bool {
        // Time filter
        if log.timestamp_ns < self.start_time_ns || log.timestamp_ns > self.end_time_ns {
            return false;
        }

        // Service filter
        if !self.services.is_empty() {
            match &log.service {
                Some(service) if self.services.contains(service) => {}
                _ => return false,
            }
        }

        // Source filter
        if !self.sources.is_empty() {
            match &log.source {
                Some(source) if self.sources.contains(source) => {}
                _ => return false,
            }
        }

        // Severity filter
        if !self.severities.is_empty() {
            let log_sev = Severity::try_from(log.severity).unwrap_or(Severity::Unspecified);
            if !self.severities.contains(&log_sev) {
                return false;
            }
        }

        // Attribute filters (now stored in fields as SqlValue)
        for (key, value) in &self.attributes {
            match log.fields.get(key) {
                Some(sql_val) => {
                    // Extract string value from SqlValue for comparison
                    let matches = match &sql_val.value {
                        Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(s)) => {
                            s == value
                        }
                        _ => false,
                    };
                    if !matches {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Text search (simple substring matching)
        // Note: For full-text search with BM25 ranking, use ObservabilityStorage::search_logs_tantivy
        // which provides Tantivy-based full-text search with proper scoring and phrase matching
        if let Some(query) = &self.text_query {
            let message_lower = log.message.to_lowercase();
            let query_lower = query.to_lowercase();

            // Simple substring match for in-memory filtering
            if !message_lower.contains(&query_lower) {
                return false;
            }
        }

        true
    }
}

/// Log aggregation types
#[derive(Debug, Clone)]
pub enum LogAggregation {
    /// Count logs
    Count,
    /// Group by field
    GroupBy(String),
    /// Time histogram with fixed-width buckets.
    Histogram {
        /// Bucket width in nanoseconds.
        interval_ns: i64,
    },
    /// Top values for a field
    TopValues {
        /// Field name to aggregate.
        field: String,
        /// Maximum number of top values to return.
        limit: usize,
    },
    /// Group by multiple fields
    GroupByMultiple(Vec<String>),
    /// Time histogram with additional GROUP BY.
    HistogramGroupBy {
        /// Bucket width in nanoseconds.
        interval_ns: i64,
        /// Fields to group by within each time bucket.
        group_by: Vec<String>,
    },
}

/// Log aggregation result
#[derive(Debug, Clone)]
pub struct LogAggregationResult {
    /// Aggregation type
    pub aggregation: LogAggregation,
    /// Results
    pub buckets: Vec<AggregationBucket>,
    /// Total count of matched logs
    pub total_count: u64,
    /// Query execution time in milliseconds
    pub query_time_ms: u64,
}

/// Aggregation bucket
#[derive(Debug, Clone)]
pub struct AggregationBucket {
    /// Bucket key (field value or timestamp)
    pub key: String,
    /// Count in this bucket
    pub count: u64,
    /// Group labels (for multi-field GROUP BY)
    pub labels: HashMap<String, String>,
    /// Timestamp bucket (for histograms)
    pub timestamp_bucket_ns: Option<i64>,
}

impl AggregationBucket {
    /// Create a simple bucket with just key and count
    pub fn simple(key: String, count: u64) -> Self {
        Self {
            key,
            count,
            labels: HashMap::new(),
            timestamp_bucket_ns: None,
        }
    }

    /// Create a bucket with group labels
    pub fn with_labels(labels: HashMap<String, String>, count: u64) -> Self {
        let key = labels
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        Self {
            key,
            count,
            labels,
            timestamp_bucket_ns: None,
        }
    }

    /// Create a histogram bucket
    pub fn histogram(timestamp_ns: i64, count: u64) -> Self {
        Self {
            key: timestamp_ns.to_string(),
            count,
            labels: HashMap::new(),
            timestamp_bucket_ns: Some(timestamp_ns),
        }
    }
}

/// Log aggregation executor
pub struct LogAggregator;

impl LogAggregator {
    /// Execute an aggregation on filtered logs
    pub fn aggregate(logs: &[LogEntry], aggregation: &LogAggregation) -> LogAggregationResult {
        let start = std::time::Instant::now();
        let total_count = logs.len() as u64;

        let buckets = match aggregation {
            LogAggregation::Count => {
                vec![AggregationBucket::simple("total".to_string(), total_count)]
            }

            LogAggregation::GroupBy(field) => Self::group_by_field(logs, field),

            LogAggregation::GroupByMultiple(fields) => Self::group_by_multiple_fields(logs, fields),

            LogAggregation::Histogram { interval_ns } => Self::time_histogram(logs, *interval_ns),

            LogAggregation::HistogramGroupBy {
                interval_ns,
                group_by,
            } => Self::time_histogram_grouped(logs, *interval_ns, group_by),

            LogAggregation::TopValues { field, limit } => Self::top_values(logs, field, *limit),
        };

        LogAggregationResult {
            aggregation: aggregation.clone(),
            buckets,
            total_count,
            query_time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Group logs by a single field
    fn group_by_field(logs: &[LogEntry], field: &str) -> Vec<AggregationBucket> {
        let mut groups: HashMap<String, u64> = HashMap::new();

        for log in logs {
            let value = Self::extract_field_value(log, field);
            *groups.entry(value).or_insert(0) += 1;
        }

        let mut buckets: Vec<_> = groups
            .into_iter()
            .map(|(key, count)| AggregationBucket::simple(key, count))
            .collect();

        // Sort by count descending
        buckets.sort_by_key(|b| std::cmp::Reverse(b.count));
        buckets
    }

    /// Group logs by multiple fields
    fn group_by_multiple_fields(logs: &[LogEntry], fields: &[String]) -> Vec<AggregationBucket> {
        let mut groups: HashMap<Vec<(String, String)>, u64> = HashMap::new();

        for log in logs {
            let mut key: Vec<(String, String)> = fields
                .iter()
                .map(|f| (f.clone(), Self::extract_field_value(log, f)))
                .collect();
            key.sort_by(|a, b| a.0.cmp(&b.0));
            *groups.entry(key).or_insert(0) += 1;
        }

        let mut buckets: Vec<_> = groups
            .into_iter()
            .map(|(key_vec, count)| {
                let labels: HashMap<String, String> = key_vec.into_iter().collect();
                AggregationBucket::with_labels(labels, count)
            })
            .collect();

        buckets.sort_by_key(|b| std::cmp::Reverse(b.count));
        buckets
    }

    /// Create time histogram buckets
    fn time_histogram(logs: &[LogEntry], interval_ns: i64) -> Vec<AggregationBucket> {
        let mut buckets_map: HashMap<i64, u64> = HashMap::new();

        for log in logs {
            let bucket_time = (log.timestamp_ns / interval_ns) * interval_ns;
            *buckets_map.entry(bucket_time).or_insert(0) += 1;
        }

        let mut buckets: Vec<_> = buckets_map
            .into_iter()
            .map(|(ts, count)| AggregationBucket::histogram(ts, count))
            .collect();

        // Sort by timestamp ascending
        buckets.sort_by(|a, b| {
            a.timestamp_bucket_ns
                .unwrap_or(0)
                .cmp(&b.timestamp_bucket_ns.unwrap_or(0))
        });
        buckets
    }

    /// Time histogram with additional grouping
    fn time_histogram_grouped(
        logs: &[LogEntry],
        interval_ns: i64,
        group_by: &[String],
    ) -> Vec<AggregationBucket> {
        // Key: (timestamp_bucket, sorted group labels)
        let mut buckets_map: HashMap<(i64, Vec<(String, String)>), u64> = HashMap::new();

        for log in logs {
            let bucket_time = (log.timestamp_ns / interval_ns) * interval_ns;
            let mut group_key: Vec<(String, String)> = group_by
                .iter()
                .map(|f| (f.clone(), Self::extract_field_value(log, f)))
                .collect();
            group_key.sort_by(|a, b| a.0.cmp(&b.0));

            *buckets_map.entry((bucket_time, group_key)).or_insert(0) += 1;
        }

        let mut buckets: Vec<_> = buckets_map
            .into_iter()
            .map(|((ts, key_vec), count)| {
                let labels: HashMap<String, String> = key_vec.into_iter().collect();
                let mut bucket = AggregationBucket::with_labels(labels, count);
                bucket.timestamp_bucket_ns = Some(ts);
                bucket
            })
            .collect();

        // Sort by timestamp, then by key
        buckets.sort_by(|a, b| {
            let ts_cmp = a
                .timestamp_bucket_ns
                .unwrap_or(0)
                .cmp(&b.timestamp_bucket_ns.unwrap_or(0));
            if ts_cmp == std::cmp::Ordering::Equal {
                a.key.cmp(&b.key)
            } else {
                ts_cmp
            }
        });
        buckets
    }

    /// Get top N values for a field
    fn top_values(logs: &[LogEntry], field: &str, limit: usize) -> Vec<AggregationBucket> {
        let mut groups = Self::group_by_field(logs, field);
        groups.truncate(limit);
        groups
    }

    /// Extract field value from log entry
    fn extract_field_value(log: &LogEntry, field: &str) -> String {
        match field.to_lowercase().as_str() {
            "service" => log
                .service
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string()),
            "source" => log
                .source
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string()),
            "severity" => Severity::try_from(log.severity)
                .map_or_else(|_| "UNKNOWN".to_string(), |s| format!("{:?}", s)),
            "level" => Severity::try_from(log.severity)
                .map_or_else(|_| "UNKNOWN".to_string(), |s| format!("{:?}", s)),
            _ => {
                // Check in fields map
                if let Some(sql_value) = log.fields.get(field) {
                    Self::sql_value_to_string(sql_value)
                } else {
                    "<missing>".to_string()
                }
            }
        }
    }

    /// Convert SqlValue to string for grouping
    fn sql_value_to_string(value: &SqlValue) -> String {
        // Group-by keys keep THIS site's distinct null/unset/bytes bucket
        // spellings (a null dimension counts separately from an explicitly
        // empty one — collapsing them changed bucket counts); scalars +
        // Jsonb render through the ONE shared crate fn.
        use crate::proto::proximadb_v1::sql_value::Value;
        match &value.value {
            Some(Value::NullValue(_)) => "<null>".to_string(),
            Some(Value::BytesValue(b)) => format!("<bytes:{}>", b.len()),
            Some(Value::ArrayValue(arr)) => format!("<array:{}>", arr.values.len()),
            Some(Value::ObjectValue(obj)) => format!("<object:{}>", obj.fields.len()),
            None => "<empty>".to_string(),
            _ => super::sql_scalar_to_string(value).unwrap_or_else(|| "<empty>".to_string()),
        }
    }
}

impl LogQuery {
    /// Execute aggregation on logs matching this query
    pub fn aggregate(
        &self,
        logs: &[LogEntry],
        aggregation: &LogAggregation,
    ) -> LogAggregationResult {
        // First filter logs
        let filtered: Vec<_> = logs
            .iter()
            .filter(|log| self.matches(log))
            .cloned()
            .collect();
        // Then aggregate
        LogAggregator::aggregate(&filtered, aggregation)
    }
}

/// Log pattern detector
pub struct PatternDetector {
    /// Minimum pattern occurrence
    min_occurrences: usize,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(min_occurrences: usize) -> Self {
        Self { min_occurrences }
    }

    /// Detect patterns in logs
    pub fn detect(&self, logs: &[LogEntry]) -> Vec<LogPattern> {
        let mut patterns: HashMap<String, LogPattern> = HashMap::new();

        for log in logs {
            let pattern = self.extract_pattern(&log.message);

            patterns
                .entry(pattern.clone())
                .and_modify(|p| {
                    p.count += 1;
                    p.examples.push(log.message.clone());
                    if p.examples.len() > 3 {
                        p.examples.remove(0);
                    }
                })
                .or_insert_with(|| LogPattern {
                    pattern: pattern.clone(),
                    count: 1,
                    examples: vec![log.message.clone()],
                });
        }

        patterns
            .into_values()
            .filter(|p| p.count >= self.min_occurrences)
            .collect()
    }

    /// Extract pattern from log message
    fn extract_pattern(&self, message: &str) -> String {
        // Replace numbers with <NUM>
        let mut pattern = String::new();
        let mut in_number = false;

        for c in message.chars() {
            if c.is_ascii_digit() {
                if !in_number {
                    pattern.push_str("<NUM>");
                    in_number = true;
                }
            } else {
                in_number = false;
                pattern.push(c);
            }
        }

        // Replace UUIDs with <UUID>
        // Replace IPs with <IP>
        // This is a simplified version

        pattern
    }
}

/// Detected log pattern
#[derive(Debug, Clone)]
pub struct LogPattern {
    /// Pattern template
    pub pattern: String,
    /// Occurrence count
    pub count: usize,
    /// Example messages
    pub examples: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log(message: &str, severity: Severity, service: &str) -> LogEntry {
        LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message: message.to_string(),
            fields: HashMap::new(),
            source: Some(String::new()),
            service: Some(service.to_string()),
        }
    }

    #[test]
    fn test_query_builder() {
        let query = LogQueryBuilder::new()
            .service("api")
            .severity(Severity::Error)
            .text("connection")
            .limit(50)
            .build();

        assert_eq!(query.services, vec!["api"]);
        assert_eq!(query.severities, vec![Severity::Error]);
        assert_eq!(query.text_query, Some("connection".to_string()));
        assert_eq!(query.limit, 50);
    }

    #[test]
    fn test_query_matches() {
        let query = LogQueryBuilder::new()
            .service("api")
            .severity(Severity::Error)
            .build();

        let log1 = make_log("Error occurred", Severity::Error, "api");
        let log2 = make_log("Info message", Severity::Info, "api");
        let log3 = make_log("Error occurred", Severity::Error, "web");

        assert!(query.matches(&log1));
        assert!(!query.matches(&log2));
        assert!(!query.matches(&log3));
    }

    #[test]
    fn test_pattern_detector() {
        let detector = PatternDetector::new(2);

        let logs = vec![
            make_log("Connection timeout after 30s", Severity::Error, "api"),
            make_log("Connection timeout after 45s", Severity::Error, "api"),
            make_log("Connection timeout after 60s", Severity::Error, "api"),
            make_log("User 123 logged in", Severity::Info, "auth"),
        ];

        let patterns = detector.detect(&logs);
        assert!(!patterns.is_empty());
    }

    // ==================== Log Aggregation Tests ====================

    fn make_log_with_ts(message: &str, severity: Severity, service: &str, ts_ns: i64) -> LogEntry {
        LogEntry {
            timestamp_ns: ts_ns,
            severity: severity as i32,
            message: message.to_string(),
            fields: HashMap::new(),
            source: Some("host1".to_string()),
            service: Some(service.to_string()),
        }
    }

    fn make_log_with_fields(
        message: &str,
        severity: Severity,
        service: &str,
        ts_ns: i64,
        fields: HashMap<String, SqlValue>,
    ) -> LogEntry {
        LogEntry {
            timestamp_ns: ts_ns,
            severity: severity as i32,
            message: message.to_string(),
            fields,
            source: Some("host1".to_string()),
            service: Some(service.to_string()),
        }
    }

    fn make_string_sql_value(s: &str) -> SqlValue {
        use crate::proto::proximadb_v1::sql_value::Value;
        SqlValue {
            value: Some(Value::StringValue(s.to_string())),
        }
    }

    #[test]
    fn test_aggregation_count() {
        let logs = vec![
            make_log("Error 1", Severity::Error, "api"),
            make_log("Error 2", Severity::Error, "api"),
            make_log("Info 1", Severity::Info, "web"),
        ];

        let result = LogAggregator::aggregate(&logs, &LogAggregation::Count);

        assert_eq!(result.total_count, 3);
        assert_eq!(result.buckets.len(), 1);
        assert_eq!(result.buckets[0].key, "total");
        assert_eq!(result.buckets[0].count, 3);
    }

    #[test]
    fn test_aggregation_group_by_service() {
        let logs = vec![
            make_log("Error 1", Severity::Error, "api"),
            make_log("Error 2", Severity::Error, "api"),
            make_log("Info 1", Severity::Info, "web"),
            make_log("Info 2", Severity::Info, "auth"),
        ];

        let result =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("service".to_string()));

        assert_eq!(result.total_count, 4);
        assert_eq!(result.buckets.len(), 3);
        // Should be sorted by count descending
        assert_eq!(result.buckets[0].key, "api");
        assert_eq!(result.buckets[0].count, 2);
    }

    #[test]
    fn test_aggregation_group_by_severity() {
        let logs = vec![
            make_log("Error 1", Severity::Error, "api"),
            make_log("Error 2", Severity::Error, "api"),
            make_log("Error 3", Severity::Error, "web"),
            make_log("Info 1", Severity::Info, "web"),
        ];

        let result =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("severity".to_string()));

        assert_eq!(result.total_count, 4);
        assert_eq!(result.buckets.len(), 2);
        // Error should be first (3 occurrences)
        assert_eq!(result.buckets[0].key, "Error");
        assert_eq!(result.buckets[0].count, 3);
        assert_eq!(result.buckets[1].key, "Info");
        assert_eq!(result.buckets[1].count, 1);
    }

    #[test]
    fn test_aggregation_group_by_multiple() {
        let logs = vec![
            make_log("E1", Severity::Error, "api"),
            make_log("E2", Severity::Error, "api"),
            make_log("E3", Severity::Error, "web"),
            make_log("I1", Severity::Info, "api"),
        ];

        let result = LogAggregator::aggregate(
            &logs,
            &LogAggregation::GroupByMultiple(vec!["service".to_string(), "severity".to_string()]),
        );

        assert_eq!(result.total_count, 4);
        // (api, Error)=2, (web, Error)=1, (api, Info)=1
        assert_eq!(result.buckets.len(), 3);
        // First should be api+Error with count 2
        assert_eq!(result.buckets[0].count, 2);
        assert_eq!(
            result.buckets[0].labels.get("service"),
            Some(&"api".to_string())
        );
        assert_eq!(
            result.buckets[0].labels.get("severity"),
            Some(&"Error".to_string())
        );
    }

    #[test]
    fn test_aggregation_time_histogram() {
        let interval_ns = 60_000_000_000i64; // 1 minute in ns
        // Align base to bucket boundary for predictable results
        let base_ns = (1_000_000_000_000i64 / interval_ns) * interval_ns;

        let logs = vec![
            make_log_with_ts("E1", Severity::Error, "api", base_ns),
            make_log_with_ts("E2", Severity::Error, "api", base_ns + 30_000_000_000), // +30s (same bucket)
            make_log_with_ts(
                "E3",
                Severity::Error,
                "api",
                base_ns + interval_ns + 10_000_000_000,
            ), // next bucket
            make_log_with_ts(
                "E4",
                Severity::Error,
                "api",
                base_ns + interval_ns * 2 + 5_000_000_000,
            ), // third bucket
        ];

        let result = LogAggregator::aggregate(&logs, &LogAggregation::Histogram { interval_ns });

        assert_eq!(result.total_count, 4);
        assert_eq!(result.buckets.len(), 3);
        // First bucket should have 2 logs (E1 and E2)
        assert_eq!(result.buckets[0].count, 2);
        assert!(result.buckets[0].timestamp_bucket_ns.is_some());
        // Second bucket should have 1 log (E3)
        assert_eq!(result.buckets[1].count, 1);
        // Third bucket should have 1 log (E4)
        assert_eq!(result.buckets[2].count, 1);
    }

    #[test]
    fn test_aggregation_histogram_grouped() {
        let base_ns = 1_000_000_000_000i64;
        let interval_ns = 60_000_000_000i64;

        let logs = vec![
            make_log_with_ts("E1", Severity::Error, "api", base_ns),
            make_log_with_ts("E2", Severity::Error, "web", base_ns + 10_000_000_000),
            make_log_with_ts("E3", Severity::Error, "api", base_ns + 70_000_000_000),
        ];

        let result = LogAggregator::aggregate(
            &logs,
            &LogAggregation::HistogramGroupBy {
                interval_ns,
                group_by: vec!["service".to_string()],
            },
        );

        assert_eq!(result.total_count, 3);
        // Buckets: (bucket0, api), (bucket0, web), (bucket1, api)
        assert_eq!(result.buckets.len(), 3);
    }

    #[test]
    fn test_aggregation_top_values() {
        let logs = vec![
            make_log("E1", Severity::Error, "api"),
            make_log("E2", Severity::Error, "api"),
            make_log("E3", Severity::Error, "api"),
            make_log("E4", Severity::Error, "web"),
            make_log("E5", Severity::Error, "web"),
            make_log("E6", Severity::Error, "auth"),
        ];

        let result = LogAggregator::aggregate(
            &logs,
            &LogAggregation::TopValues {
                field: "service".to_string(),
                limit: 2,
            },
        );

        assert_eq!(result.total_count, 6);
        assert_eq!(result.buckets.len(), 2);
        assert_eq!(result.buckets[0].key, "api");
        assert_eq!(result.buckets[0].count, 3);
        assert_eq!(result.buckets[1].key, "web");
        assert_eq!(result.buckets[1].count, 2);
    }

    #[test]
    fn test_aggregation_group_by_custom_field() {
        let mut fields1 = HashMap::new();
        fields1.insert("status_code".to_string(), make_string_sql_value("200"));
        let mut fields2 = HashMap::new();
        fields2.insert("status_code".to_string(), make_string_sql_value("500"));
        let mut fields3 = HashMap::new();
        fields3.insert("status_code".to_string(), make_string_sql_value("200"));

        let logs = vec![
            make_log_with_fields("req1", Severity::Info, "api", 0, fields1),
            make_log_with_fields("req2", Severity::Error, "api", 0, fields2),
            make_log_with_fields("req3", Severity::Info, "api", 0, fields3),
        ];

        let result =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("status_code".to_string()));

        assert_eq!(result.buckets.len(), 2);
        assert_eq!(result.buckets[0].key, "200");
        assert_eq!(result.buckets[0].count, 2);
        assert_eq!(result.buckets[1].key, "500");
        assert_eq!(result.buckets[1].count, 1);
    }

    #[test]
    fn aggregation_keeps_null_distinct_from_empty_string() {
        use crate::proto::proximadb_v1::sql_value::Value;

        let null_fields = [(
            "status".to_string(),
            SqlValue {
                value: Some(Value::NullValue(0)),
            },
        )]
        .into_iter()
        .collect();
        let empty_fields = [(
            "status".to_string(),
            SqlValue {
                value: Some(Value::StringValue(String::new())),
            },
        )]
        .into_iter()
        .collect();
        let logs = vec![
            make_log_with_fields("null", Severity::Info, "api", 0, null_fields),
            make_log_with_fields("empty", Severity::Info, "api", 0, empty_fields),
        ];

        let result =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("status".to_string()));
        let keys: std::collections::BTreeSet<_> = result
            .buckets
            .iter()
            .map(|bucket| bucket.key.as_str())
            .collect();

        assert_eq!(keys, ["", "<null>"].into_iter().collect());
    }

    #[test]
    fn aggregation_groups_jsonb_by_document_not_byte_length() {
        use crate::proto::proximadb_v1::sql_value::Value;

        let make_fields = |region: &str| {
            let document = serde_json::json!({"region": region});
            [(
                "context".to_string(),
                SqlValue {
                    value: Some(Value::JsonbValue(
                        proximadb_data_model::ProximaValue::to_jsonb_vec(&document)
                            .expect("encode JSONB test value"),
                    )),
                },
            )]
            .into_iter()
            .collect()
        };
        let logs = vec![
            make_log_with_fields("east", Severity::Info, "api", 0, make_fields("east")),
            make_log_with_fields("west", Severity::Info, "api", 0, make_fields("west")),
        ];

        let result =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("context".to_string()));
        let keys: std::collections::BTreeSet<_> =
            result.buckets.iter().map(|bucket| &bucket.key).collect();

        assert_eq!(result.buckets.len(), 2);
        assert!(keys.contains(&serde_json::json!({"region": "east"}).to_string()));
        assert!(keys.contains(&serde_json::json!({"region": "west"}).to_string()));
    }

    #[test]
    fn test_query_aggregate() {
        let logs = vec![
            make_log("Error 1", Severity::Error, "api"),
            make_log("Error 2", Severity::Error, "api"),
            make_log("Info 1", Severity::Info, "api"),
            make_log("Error 3", Severity::Error, "web"),
        ];

        // Query only api service
        let query = LogQueryBuilder::new().service("api").build();

        // Aggregate by severity on filtered logs
        let result = query.aggregate(&logs, &LogAggregation::GroupBy("severity".to_string()));

        // Should only include api logs (3 total)
        assert_eq!(result.total_count, 3);
        assert_eq!(result.buckets.len(), 2);
        assert_eq!(result.buckets[0].key, "Error");
        assert_eq!(result.buckets[0].count, 2);
        assert_eq!(result.buckets[1].key, "Info");
        assert_eq!(result.buckets[1].count, 1);
    }

    #[test]
    fn test_aggregation_empty_logs() {
        let logs: Vec<LogEntry> = vec![];

        let result = LogAggregator::aggregate(&logs, &LogAggregation::Count);
        assert_eq!(result.total_count, 0);
        assert_eq!(result.buckets[0].count, 0);

        let result2 =
            LogAggregator::aggregate(&logs, &LogAggregation::GroupBy("service".to_string()));
        assert_eq!(result2.total_count, 0);
        assert!(result2.buckets.is_empty());
    }
}
