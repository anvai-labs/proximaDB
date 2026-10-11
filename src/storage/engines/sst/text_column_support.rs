/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! TEXT Column Support for SST Engine
//!
//! This module integrates the columnar TEXT storage infrastructure with the SST engine:
//! - TextColumnWriter: For writing TEXT columns during flush operations
//! - TextColumnReader: For reading TEXT values with lazy sidecar loading
//! - TextColumnFilterEvaluator: For filtering on TEXT columns during queries
//!
//! ## Storage Strategies
//!
//! TEXT columns use adaptive storage based on content size:
//! - **Inline** (<4KB): Stored directly in VectorRecord metadata
//! - **Chunked** (4KB-1MB): Split into chunks with per-chunk storage
//! - **Sidecar** (>1MB): Stored in separate files with references
//!
//! ## Integration Points
//!
//! 1. **Flush Path**: `process_text_columns()` extracts TEXT metadata and routes to storage
//! 2. **Query Path**: `evaluate_text_filter()` applies TEXT predicates efficiently
//! 3. **Read Path**: `load_text_values()` retrieves TEXT content with lazy loading

use std::collections::HashMap;
use tracing::debug;

use crate::core::types::{ColumnDataType, TextStorageStrategy};
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::formats::columnar::{
    TextColumnFilterEvaluator, TextColumnReader, TextColumnWriter, TextComparisonOp,
    TextFilterBuilder, TextStorageConfig,
};

/// TEXT column processor for SST flush operations
///
/// Handles extraction and storage of TEXT-typed metadata columns during
/// memtable flush to SST files.
pub struct SstTextColumnProcessor {
    /// Per-column writers for TEXT fields
    writers: HashMap<String, TextColumnWriter>,

    /// Configuration for TEXT storage
    config: TextStorageConfig,

    /// Base path for sidecar files
    sidecar_base_path: Option<String>,

    /// Column definitions indicating which metadata fields are TEXT type
    text_column_definitions: HashMap<String, TextColumnDefinition>,
}

/// Definition of a TEXT column for schema-aware processing
#[derive(Debug, Clone)]
pub struct TextColumnDefinition {
    /// Column name
    pub name: String,

    /// Storage strategy
    pub storage_strategy: TextStorageStrategy,

    /// Enable n-gram bloom filter for CONTAINS queries
    pub enable_ngram_bloom: bool,

    /// N-gram size (default: 3)
    pub ngram_size: usize,
}

impl Default for TextColumnDefinition {
    fn default() -> Self {
        Self {
            name: String::new(),
            storage_strategy: TextStorageStrategy::Adaptive,
            enable_ngram_bloom: false,
            ngram_size: 3,
        }
    }
}

impl SstTextColumnProcessor {
    /// Create a new TEXT column processor with default configuration
    pub fn new() -> Self {
        Self {
            writers: HashMap::new(),
            config: TextStorageConfig::default(),
            sidecar_base_path: None,
            text_column_definitions: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TextStorageConfig) -> Self {
        Self {
            writers: HashMap::new(),
            config,
            sidecar_base_path: None,
            text_column_definitions: HashMap::new(),
        }
    }

    /// Set the base path for sidecar files
    pub fn with_sidecar_path(mut self, path: String) -> Self {
        self.sidecar_base_path = Some(path.clone());
        self.config.sidecar_base_path = Some(path);
        self
    }

    /// Register a TEXT column definition
    pub fn register_text_column(&mut self, definition: TextColumnDefinition) {
        let name = definition.name.clone();
        self.text_column_definitions
            .insert(name.clone(), definition);

        // Create a writer for this column
        let mut col_config = self.config.clone();
        if let Some(def) = self.text_column_definitions.get(&name) {
            col_config.strategy = def.storage_strategy;
            col_config.enable_ngram_bloom = def.enable_ngram_bloom;
            col_config.ngram_size = def.ngram_size;
        }

        self.writers.insert(name, TextColumnWriter::new(col_config));
    }

    /// Register TEXT columns from collection schema
    pub fn register_from_schema(&mut self, schema_columns: &[(String, ColumnDataType)]) {
        for (name, data_type) in schema_columns {
            if data_type.is_text() {
                let definition = TextColumnDefinition {
                    name: name.clone(),
                    storage_strategy: match data_type {
                        ColumnDataType::TextLarge => TextStorageStrategy::Sidecar,
                        _ => TextStorageStrategy::Adaptive,
                    },
                    enable_ngram_bloom: false,
                    ngram_size: 3,
                };
                self.register_text_column(definition);
            }
        }
    }

    /// Process TEXT columns from a batch of VectorRecords
    ///
    /// Extracts TEXT-typed metadata values and routes them to appropriate storage.
    /// Returns a mapping of record_id -> (column_name -> storage_reference) for
    /// any values that were moved to chunked or sidecar storage.
    pub fn process_batch(
        &mut self,
        records: &[VectorRecord],
    ) -> Result<TextColumnBatchResult, TextProcessingError> {
        let mut result = TextColumnBatchResult::default();

        for record in records {
            self.process_record(record, &mut result)?;
        }

        // Collect statistics
        for (col_name, writer) in &self.writers {
            let stats = writer.stats();
            result.stats.insert(
                col_name.clone(),
                TextColumnStats {
                    inline_count: stats.inline_count,
                    chunked_count: stats.chunked_count,
                    sidecar_count: stats.sidecar_count,
                    total_bytes: stats.inline_bytes + stats.sidecar_bytes,
                },
            );
        }

        debug!(
            "Processed {} TEXT columns for {} records",
            self.writers.len(),
            records.len()
        );

        Ok(result)
    }

    /// Process TEXT columns from a single VectorRecord
    fn process_record(
        &mut self,
        record: &VectorRecord,
        result: &mut TextColumnBatchResult,
    ) -> Result<(), TextProcessingError> {
        for (col_name, writer) in &mut self.writers {
            // Check if this column exists in the record's metadata
            if let Some(sql_value) = record.metadata.get(col_name) {
                // Extract string value from SqlValue
                if let Some(value) = &sql_value.value {
                    match value {
                        crate::proto::proximadb_v1::sql_value::Value::StringValue(text) => {
                            // Write the text value
                            writer.write(&record.id, text).map_err(|e| {
                                TextProcessingError::WriteError(format!(
                                    "Failed to write TEXT column '{}': {}",
                                    col_name, e
                                ))
                            })?;

                            // Check if it was stored as chunked or sidecar
                            if let Some(storage_type) = writer.get_storage_type(&record.id) {
                                match storage_type {
                                    crate::storage::engines::core::formats::columnar::StorageType::Chunked => {
                                        result.chunked_references
                                            .entry(record.id.clone())
                                            .or_default()
                                            .insert(col_name.clone(), format!("__chunked__:{}", col_name));
                                    }
                                    crate::storage::engines::core::formats::columnar::StorageType::Sidecar => {
                                        result.sidecar_references
                                            .entry(record.id.clone())
                                            .or_default()
                                            .insert(col_name.clone(), format!("__sidecar__:{}", col_name));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::NullValue(_) => {
                            writer.write_null(&record.id);
                        }
                        _ => {
                            // Non-string value - skip TEXT processing
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get TEXT chunks for separate storage
    pub fn get_all_chunks(
        &self,
    ) -> Vec<(
        &str,
        &[crate::storage::engines::core::formats::columnar::TextChunk],
    )> {
        self.writers
            .iter()
            .map(|(name, writer)| (name.as_str(), writer.get_chunks()))
            .collect()
    }

    /// Get sidecar references for external storage
    pub fn get_all_sidecar_refs(
        &self,
    ) -> Vec<(
        &str,
        &[crate::storage::engines::core::formats::columnar::SidecarRef],
    )> {
        self.writers
            .iter()
            .map(|(name, writer)| (name.as_str(), writer.get_sidecar_refs()))
            .collect()
    }

    /// Clear all writers for next batch
    pub fn clear(&mut self) {
        for writer in self.writers.values_mut() {
            writer.clear();
        }
    }

    /// Check if any TEXT columns are registered
    pub fn has_text_columns(&self) -> bool {
        !self.text_column_definitions.is_empty()
    }

    /// Get registered TEXT column names
    pub fn text_column_names(&self) -> Vec<&str> {
        self.text_column_definitions
            .keys()
            .map(|s| s.as_str())
            .collect()
    }
}

impl Default for SstTextColumnProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing TEXT columns for a batch
#[derive(Debug, Default)]
pub struct TextColumnBatchResult {
    /// References to chunked storage (record_id -> column_name -> reference)
    pub chunked_references: HashMap<String, HashMap<String, String>>,

    /// References to sidecar storage (record_id -> column_name -> reference)
    pub sidecar_references: HashMap<String, HashMap<String, String>>,

    /// Statistics per column
    pub stats: HashMap<String, TextColumnStats>,
}

/// Statistics for a TEXT column
#[derive(Debug, Default, Clone)]
pub struct TextColumnStats {
    pub inline_count: u64,
    pub chunked_count: u64,
    pub sidecar_count: u64,
    pub total_bytes: u64,
}

/// Errors that can occur during TEXT column processing
#[derive(Debug, thiserror::Error)]
pub enum TextProcessingError {
    #[error("Write error: {0}")]
    WriteError(String),

    #[error("Read error: {0}")]
    ReadError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Filter error: {0}")]
    FilterError(String),
}

/// TEXT column filter evaluator for SST queries
///
/// Integrates TextColumnFilterEvaluator with SST's three-stage filtering pipeline.
pub struct SstTextFilterEvaluator {
    /// Per-column filter evaluators
    evaluators: HashMap<String, TextColumnFilterEvaluator>,

    /// Filter builder for statistics tracking
    filter_builder: TextFilterBuilder,
}

impl SstTextFilterEvaluator {
    /// Create a new TEXT filter evaluator
    pub fn new() -> Self {
        Self {
            evaluators: HashMap::new(),
            filter_builder: TextFilterBuilder::new(),
        }
    }

    /// Register a TEXT column for filtering
    pub fn register_column(&mut self, column_name: String) {
        let evaluator = TextColumnFilterEvaluator::new(column_name.clone());
        self.evaluators.insert(column_name.clone(), evaluator);
        self.filter_builder.add_column(column_name);
    }

    /// Register a TEXT column with bloom filter for CONTAINS optimization
    pub fn register_column_with_bloom(
        &mut self,
        column_name: String,
        _bloom_data: &[u8],
        ngram_size: usize,
    ) -> Result<(), TextProcessingError> {
        // Deserialize bloom filter from stored data
        // Note: _bloom_data would be used to deserialize a pre-built bloom filter
        // For now, we create a fresh one with estimated capacity
        let bloom_config = crate::core::bloom::BloomFilterConfig::for_sstable(10000);
        let bloom = crate::core::bloom::factory::BloomFilterFactory::create(&bloom_config);

        let evaluator = TextColumnFilterEvaluator::new(column_name.clone())
            .with_bloom_filter(bloom, ngram_size);

        self.evaluators.insert(column_name.clone(), evaluator);
        self.filter_builder.add_column(column_name);

        Ok(())
    }

    /// Evaluate a TEXT filter expression
    ///
    /// Returns indices of records that match the filter.
    pub fn evaluate(
        &self,
        column_name: &str,
        op: &TextComparisonOp,
        value: &str,
        text_values: &[Option<String>],
    ) -> Result<Vec<usize>, TextProcessingError> {
        let evaluator = self.evaluators.get(column_name).ok_or_else(|| {
            TextProcessingError::FilterError(format!(
                "No evaluator registered for TEXT column '{}'",
                column_name
            ))
        })?;

        Ok(evaluator.evaluate(op, value, text_values))
    }

    /// Convert a FilterExpression to TextComparisonOp if applicable
    pub fn convert_filter_expression(
        expr: &proximadb_filter_expression::FilterExpression,
    ) -> Option<(String, TextComparisonOp, String)> {
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

        match expr {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                let text_op = match operator {
                    ComparisonOperator::Equals => Some(TextComparisonOp::Equals),
                    ComparisonOperator::NotEquals => Some(TextComparisonOp::NotEquals),
                    ComparisonOperator::Contains => Some(TextComparisonOp::Contains),
                    ComparisonOperator::StartsWith => Some(TextComparisonOp::StartsWith),
                    ComparisonOperator::EndsWith => Some(TextComparisonOp::EndsWith),
                    ComparisonOperator::Like => Some(TextComparisonOp::RegexMatch),
                    ComparisonOperator::IsNull => Some(TextComparisonOp::IsNull),
                    ComparisonOperator::IsNotNull => Some(TextComparisonOp::IsNotNull),
                    _ => None,
                }?;

                // Extract string value
                let str_value = value.as_str().map(|s| s.to_string()).unwrap_or_default();

                Some((field.clone(), text_op, str_value))
            }
            _ => None,
        }
    }

    /// Get filter statistics
    pub fn stats(&self) -> &crate::storage::engines::core::formats::columnar::TextFilterStats {
        self.filter_builder.stats()
    }
}

impl Default for SstTextFilterEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// TEXT column reader for SST queries
///
/// Provides lazy loading of TEXT content with caching for sidecar files.
pub struct SstTextColumnReader {
    /// Per-column readers
    readers: HashMap<String, TextColumnReader>,

    /// Configuration
    config: TextStorageConfig,
}

impl SstTextColumnReader {
    /// Create a new TEXT column reader
    pub fn new(config: TextStorageConfig) -> Self {
        Self {
            readers: HashMap::new(),
            config,
        }
    }

    /// Register a TEXT column for reading
    pub fn register_column(&mut self, column_name: String) {
        let reader = TextColumnReader::new(self.config.clone());
        self.readers.insert(column_name, reader);
    }

    /// Extract TEXT values from VectorRecords for a specific column
    pub fn extract_text_values(
        &self,
        records: &[VectorRecord],
        column_name: &str,
    ) -> Vec<Option<String>> {
        records
            .iter()
            .map(|record| {
                record.metadata.get(column_name).and_then(|sql_value| {
                    sql_value.value.as_ref().and_then(|v| match v {
                        crate::proto::proximadb_v1::sql_value::Value::StringValue(s) => {
                            Some(s.clone())
                        }
                        // TD-PROTO-2: JSONB decodes to compact JSON text
                        // instead of being dropped by the `_` wildcard.
                        crate::proto::proximadb_v1::sql_value::Value::JsonbValue(b) => {
                            Some(proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(b))
                        }
                        _ => None,
                    })
                })
            })
            .collect()
    }

    /// Clear all reader caches
    pub fn clear_caches(&mut self) {
        for reader in self.readers.values_mut() {
            reader.clear_cache();
        }
    }
}

impl Default for SstTextColumnReader {
    fn default() -> Self {
        Self::new(TextStorageConfig::default())
    }
}

/// Builder for integrating TEXT column support into SST operations
pub struct SstTextSupportBuilder {
    processor: Option<SstTextColumnProcessor>,
    filter_evaluator: Option<SstTextFilterEvaluator>,
    reader: Option<SstTextColumnReader>,
}

impl SstTextSupportBuilder {
    pub fn new() -> Self {
        Self {
            processor: None,
            filter_evaluator: None,
            reader: None,
        }
    }

    /// Enable TEXT column processing for flush operations
    pub fn with_processor(mut self, config: TextStorageConfig) -> Self {
        self.processor = Some(SstTextColumnProcessor::with_config(config));
        self
    }

    /// Enable TEXT column filtering for queries
    pub fn with_filter_evaluator(mut self) -> Self {
        self.filter_evaluator = Some(SstTextFilterEvaluator::new());
        self
    }

    /// Enable TEXT column reading for queries
    pub fn with_reader(mut self, config: TextStorageConfig) -> Self {
        self.reader = Some(SstTextColumnReader::new(config));
        self
    }

    /// Build the TEXT support components
    pub fn build(self) -> SstTextSupport {
        SstTextSupport {
            processor: self.processor,
            filter_evaluator: self.filter_evaluator,
            reader: self.reader,
        }
    }
}

impl Default for SstTextSupportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete TEXT column support for SST operations
pub struct SstTextSupport {
    /// TEXT column processor for flush operations
    pub processor: Option<SstTextColumnProcessor>,

    /// TEXT column filter evaluator for queries
    pub filter_evaluator: Option<SstTextFilterEvaluator>,

    /// TEXT column reader for queries
    pub reader: Option<SstTextColumnReader>,
}

impl SstTextSupport {
    /// Create with all components enabled using default configuration
    pub fn default_all() -> Self {
        SstTextSupportBuilder::new()
            .with_processor(TextStorageConfig::default())
            .with_filter_evaluator()
            .with_reader(TextStorageConfig::default())
            .build()
    }

    /// Check if TEXT processing is enabled
    pub fn has_processor(&self) -> bool {
        self.processor.is_some()
    }

    /// Check if TEXT filtering is enabled
    pub fn has_filter_evaluator(&self) -> bool {
        self.filter_evaluator.is_some()
    }

    /// Check if TEXT reading is enabled
    pub fn has_reader(&self) -> bool {
        self.reader.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};

    fn create_test_record(id: &str, text_field: &str, text_value: &str) -> VectorRecord {
        let mut metadata = HashMap::new();
        metadata.insert(
            text_field.to_string(),
            SqlValue {
                value: Some(Value::StringValue(text_value.to_string())),
            },
        );

        VectorRecord {
            id: id.to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata,
            timestamp: Some(chrono::Utc::now().timestamp()),
            updated_at: None,
            expires_at: None,
            version: Some(1),
            source: None,
        }
    }

    #[test]
    fn test_text_column_processor_basic() {
        let mut processor = SstTextColumnProcessor::new();

        // Register a TEXT column
        processor.register_text_column(TextColumnDefinition {
            name: "description".to_string(),
            storage_strategy: TextStorageStrategy::Adaptive,
            enable_ngram_bloom: false,
            ngram_size: 3,
        });

        assert!(processor.has_text_columns());
        assert!(processor.text_column_names().contains(&"description"));
    }

    #[test]
    fn test_text_column_processor_batch() {
        let mut processor = SstTextColumnProcessor::new();

        processor.register_text_column(TextColumnDefinition {
            name: "title".to_string(),
            storage_strategy: TextStorageStrategy::Inline,
            enable_ngram_bloom: false,
            ngram_size: 3,
        });

        let records = vec![
            create_test_record("rec1", "title", "Hello World"),
            create_test_record("rec2", "title", "Goodbye World"),
        ];

        let result = processor.process_batch(&records).unwrap();

        // Check stats
        let stats = result.stats.get("title").unwrap();
        assert_eq!(stats.inline_count, 2);
    }

    #[test]
    fn test_text_filter_evaluator_basic() {
        let mut evaluator = SstTextFilterEvaluator::new();
        evaluator.register_column("description".to_string());

        let text_values = vec![
            Some("Hello World".to_string()),
            Some("Goodbye World".to_string()),
            None,
            Some("Hello There".to_string()),
        ];

        // Test CONTAINS
        let matches = evaluator
            .evaluate(
                "description",
                &TextComparisonOp::Contains,
                "World",
                &text_values,
            )
            .unwrap();
        assert_eq!(matches, vec![0, 1]);

        // Test STARTS_WITH
        let matches = evaluator
            .evaluate(
                "description",
                &TextComparisonOp::StartsWith,
                "Hello",
                &text_values,
            )
            .unwrap();
        assert_eq!(matches, vec![0, 3]);

        // Test IS_NULL
        let matches = evaluator
            .evaluate("description", &TextComparisonOp::IsNull, "", &text_values)
            .unwrap();
        assert_eq!(matches, vec![2]);
    }

    #[test]
    fn test_text_reader_extract_values() {
        let reader = SstTextColumnReader::default();

        let records = vec![
            create_test_record("rec1", "content", "First content"),
            create_test_record("rec2", "content", "Second content"),
        ];

        let values = reader.extract_text_values(&records, "content");
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], Some("First content".to_string()));
        assert_eq!(values[1], Some("Second content".to_string()));
    }

    #[test]
    fn test_sst_text_support_builder() {
        let support = SstTextSupportBuilder::new()
            .with_processor(TextStorageConfig::default())
            .with_filter_evaluator()
            .with_reader(TextStorageConfig::default())
            .build();

        assert!(support.has_processor());
        assert!(support.has_filter_evaluator());
        assert!(support.has_reader());
    }

    #[test]
    fn test_convert_filter_expression() {
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

        let expr = FilterExpression::Comparison {
            field: "title".to_string(),
            operator: ComparisonOperator::Contains,
            value: serde_json::json!("search term"),
        };

        let result = SstTextFilterEvaluator::convert_filter_expression(&expr);
        assert!(result.is_some());

        let (field, op, value) = result.unwrap();
        assert_eq!(field, "title");
        assert_eq!(op, TextComparisonOp::Contains);
        assert_eq!(value, "search term");
    }
}
