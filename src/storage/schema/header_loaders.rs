//! # Header Loaders - Bridge Between Engines and ProximaHeaderCache
//!
//! This module provides HeaderLoader implementations for each storage engine,
//! extracting metadata from existing readers into the unified CachedHeader format.
//!
//! ## Reuse Strategy
//!
//! - **Parquet Engines (NOVA/RAPTOR)**: Uses SharedParquetFormatReader for footer caching
//! - **ProximaBlocks Engines (SST/HELIX/SWIFT)**: Uses RowBasedHeader parsing
//!
//! ## Integration Points
//!
//! - `SharedParquetFormatReader::get_metadata()` → `CachedHeader.rowgroups`
//! - `RowBasedHeader::layout_metadata` → `CachedHeader.rowgroups`
//! - `CollectionMetadata::filterable_columns` → `CachedHeader.column_stats`

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parquet::file::reader::FileReader;
use tracing::{debug, info};

use super::header_cache::{
    CachedHeader, ColumnBounds, ColumnValue, EncodingInfo, HeaderLoader, RowGroupMeta, SpatialRange,
};
use crate::storage::persistence::filesystem::FileSystem;

// ============================================================================
// Parquet-Based Header Loader (NOVA, RAPTOR)
// ============================================================================

/// Header loader for Parquet-based engines (NOVA, RAPTOR).
/// Reuses SharedParquetFormatReader infrastructure for footer caching.
pub struct ParquetHeaderLoader {
    /// Filesystem for I/O operations
    filesystem: Arc<dyn FileSystem>,
    /// Engine type identifier
    engine_type: String,
}

impl ParquetHeaderLoader {
    /// Create a new Parquet header loader.
    pub fn new(filesystem: Arc<dyn FileSystem>, engine_type: &str) -> Self {
        Self {
            filesystem,
            engine_type: engine_type.to_string(),
        }
    }

    /// Create loader for NOVA engine.
    pub fn nova(filesystem: Arc<dyn FileSystem>) -> Self {
        Self::new(filesystem, "nova")
    }

    /// Create loader for RAPTOR engine.
    pub fn raptor(filesystem: Arc<dyn FileSystem>) -> Self {
        Self::new(filesystem, "raptor")
    }

    /// Extract rowgroup metadata from Parquet footer.
    fn extract_rowgroups_from_parquet(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
    ) -> Vec<RowGroupMeta> {
        let mut rowgroups = Vec::with_capacity(metadata.num_row_groups());

        for i in 0..metadata.num_row_groups() {
            let rg_metadata = metadata.row_group(i);
            let row_count = rg_metadata.num_rows();
            let total_byte_size = rg_metadata.total_byte_size() as u64;

            // Calculate offset from first column
            let offset = rg_metadata.column(0).file_offset() as u64;

            let mut rg = RowGroupMeta::new(i, offset, total_byte_size, row_count);

            // Extract column statistics using the ColumnChunkMetaData API
            // Note: We skip detailed stats extraction here for simplicity
            // The existing SharedParquetFormatReader handles this more comprehensively

            // Add encoding info based on engine type
            rg.encoding = Some(self.get_encoding_info(rg_metadata));

            // Add spatial range for RAPTOR (Z-order)
            if self.engine_type == "raptor" {
                rg.spatial_range = self.extract_zorder_range(rg_metadata);
            }

            // Add zone map for NOVA
            if self.engine_type == "nova" {
                rg.spatial_range = self.extract_zonemap(rg_metadata);
            }

            rowgroups.push(rg);
        }

        rowgroups
    }

    /// Get encoding info from rowgroup metadata.
    fn get_encoding_info(
        &self,
        rg_metadata: &parquet::file::metadata::RowGroupMetaData,
    ) -> EncodingInfo {
        // Check for quantized columns (typically named with _int8, _binary, _pq suffix)
        let has_int8 = (0..rg_metadata.num_columns()).any(|i| {
            rg_metadata
                .column(i)
                .column_path()
                .string()
                .contains("int8")
        });
        let has_binary = (0..rg_metadata.num_columns()).any(|i| {
            rg_metadata
                .column(i)
                .column_path()
                .string()
                .contains("binary")
        });
        let has_pq = (0..rg_metadata.num_columns())
            .any(|i| rg_metadata.column(i).column_path().string().contains("pq"));

        let quantization_type = if has_pq {
            "PQ".to_string()
        } else if has_int8 {
            "INT8".to_string()
        } else if has_binary {
            "Binary".to_string()
        } else {
            "None".to_string()
        };

        EncodingInfo {
            quantization_type,
            bits: if has_int8 {
                Some(8)
            } else if has_binary {
                Some(1)
            } else {
                None
            },
            pq_subquantizers: if has_pq { Some(64) } else { None }, // Default PQ config
            pq_codebook_size: if has_pq { Some(256) } else { None },
        }
    }

    /// Extract Z-order range from RAPTOR rowgroup metadata.
    fn extract_zorder_range(
        &self,
        _rg_metadata: &parquet::file::metadata::RowGroupMetaData,
    ) -> Option<SpatialRange> {
        // RAPTOR stores Z-order codes in dedicated columns
        // For now, return None - full implementation would parse the z_order_code column stats
        None
    }

    /// Extract zone map from NOVA rowgroup metadata.
    fn extract_zonemap(
        &self,
        _rg_metadata: &parquet::file::metadata::RowGroupMetaData,
    ) -> Option<SpatialRange> {
        // NOVA stores per-dimension min/max in column statistics
        // For now, return None - full implementation would aggregate per-dimension bounds
        None
    }
}

#[async_trait]
impl HeaderLoader for ParquetHeaderLoader {
    async fn load_header(&self, path: &str) -> anyhow::Result<CachedHeader> {
        debug!("Loading Parquet header from: {}", path);

        // Read Parquet footer
        let data = self.filesystem.read(path).await?;
        let file_bytes = Bytes::from(data);
        let reader = parquet::file::reader::SerializedFileReader::new(file_bytes.clone())?;
        let metadata = reader.metadata().clone();

        // Build cached header
        let mut header = CachedHeader::new(path.to_string(), 0);
        header.format_type = self.engine_type.clone();
        header.file_size = file_bytes.len() as u64;
        header.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Extract rowgroup metadata
        header.rowgroups = self.extract_rowgroups_from_parquet(&metadata);

        // Estimate header size (metadata structure)
        header.header_size_bytes =
            std::mem::size_of::<CachedHeader>() + header.rowgroups.len() * 1024; // Approximate per-rowgroup overhead

        // Extract schema version from Parquet key-value metadata
        if let Some(kv_metadata) = metadata.file_metadata().key_value_metadata() {
            for kv in kv_metadata {
                if kv.key == "proximadb.schema_version"
                    && let Some(ref value) = kv.value
                    && let Ok(v) = value.parse::<u32>()
                {
                    header.schema_version = v;
                }
                if kv.key == "proximadb.schema_fingerprint"
                    && let Some(ref value) = kv.value
                    && let Ok(v) = value.parse::<u64>()
                {
                    header.schema_fingerprint = v;
                }
                // Store all ProximaDB metadata
                if kv.key.starts_with("proximadb.")
                    && let Some(ref value) = kv.value
                    && let Some(key_suffix) = kv.key.strip_prefix("proximadb.")
                {
                    header
                        .engine_metadata
                        .insert(key_suffix.to_string(), value.clone());
                }
            }
        }

        info!(
            "Loaded Parquet header for {} ({} rowgroups, {} bytes)",
            path,
            header.rowgroups.len(),
            header.file_size
        );

        Ok(header)
    }

    fn supports_format(&self, format_type: &str) -> bool {
        matches!(
            format_type.to_lowercase().as_str(),
            "viper" | "nova" | "raptor" | "parquet"
        )
    }
}

// ============================================================================
// ProximaBlocks Header Loader (SST, HELIX, SWIFT)
// ============================================================================

/// Header loader for ProximaBlocks-based engines (SST, HELIX, SWIFT).
/// Reuses RowBasedHeader parsing infrastructure.
pub struct ProximaBlocksHeaderLoader {
    /// Filesystem for I/O operations
    filesystem: Arc<dyn FileSystem>,
    /// Engine type identifier
    engine_type: String,
}

impl ProximaBlocksHeaderLoader {
    /// Create a new ProximaBlocks header loader.
    pub fn new(filesystem: Arc<dyn FileSystem>, engine_type: &str) -> Self {
        Self {
            filesystem,
            engine_type: engine_type.to_string(),
        }
    }

    /// Create loader for SST engine.
    pub fn sst(filesystem: Arc<dyn FileSystem>) -> Self {
        Self::new(filesystem, "sst")
    }

    /// Create loader for HELIX engine.
    pub fn helix(filesystem: Arc<dyn FileSystem>) -> Self {
        Self::new(filesystem, "helix")
    }

    /// Create loader for SWIFT engine.
    pub fn swift(filesystem: Arc<dyn FileSystem>) -> Self {
        Self::new(filesystem, "swift")
    }

    /// Parse ProximaBlocks header from raw bytes.
    fn parse_header(&self, data: &[u8]) -> anyhow::Result<ProximaBlocksHeaderInfo> {
        // ProximaBlocks header format:
        // [0..8]: Magic bytes "PROXIMA\0"
        // [8..12]: Version (u32)
        // [12..16]: Header size (u32)
        // [16..header_size]: JSON-encoded metadata
        // Rest: Block data

        if data.len() < 16 {
            return Err(anyhow::anyhow!("File too small for ProximaBlocks header"));
        }

        let magic = &data[0..8];
        if magic != b"PROXIMA\0" && magic != b"PROXBLK\0" {
            return Err(anyhow::anyhow!("Invalid ProximaBlocks magic: {:?}", magic));
        }

        let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let header_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        if data.len() < header_size {
            return Err(anyhow::anyhow!(
                "Incomplete header: {} < {}",
                data.len(),
                header_size
            ));
        }

        // Parse JSON metadata from header
        let header_json = &data[16..header_size];
        let metadata: serde_json::Value =
            serde_json::from_slice(header_json).unwrap_or(serde_json::json!({}));

        Ok(ProximaBlocksHeaderInfo {
            version,
            header_size,
            metadata,
            total_size: data.len() as u64,
        })
    }

    /// Extract blocks from ProximaBlocks metadata.
    fn extract_blocks(&self, info: &ProximaBlocksHeaderInfo) -> Vec<RowGroupMeta> {
        let mut blocks = Vec::new();

        // Parse blocks from metadata JSON
        if let Some(block_array) = info.metadata.get("blocks").and_then(|b| b.as_array()) {
            for (i, block) in block_array.iter().enumerate() {
                let offset = block.get("offset").and_then(|o| o.as_u64()).unwrap_or(0);
                let length = block.get("length").and_then(|l| l.as_u64()).unwrap_or(0);
                let row_count = block.get("row_count").and_then(|r| r.as_i64()).unwrap_or(0);

                let mut rg = RowGroupMeta::new(i, offset, length, row_count);

                // Extract spatial range based on engine type
                rg.spatial_range = match self.engine_type.as_str() {
                    "helix" => {
                        // HELIX uses Hilbert curve codes
                        let min = block
                            .get("hilbert_min")
                            .and_then(|h| h.as_u64())
                            .unwrap_or(0);
                        let max = block
                            .get("hilbert_max")
                            .and_then(|h| h.as_u64())
                            .unwrap_or(u64::MAX);
                        let order = block
                            .get("hilbert_order")
                            .and_then(|o| o.as_u64())
                            .unwrap_or(16) as u8;
                        Some(SpatialRange::Hilbert { min, max, order })
                    }
                    "swift" => {
                        // SWIFT uses AdaCurve learned codes
                        let min = block
                            .get("adacurve_min")
                            .and_then(|a| a.as_u64())
                            .unwrap_or(0);
                        let max = block
                            .get("adacurve_max")
                            .and_then(|a| a.as_u64())
                            .unwrap_or(u64::MAX);
                        let model_version = block
                            .get("adacurve_version")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                        Some(SpatialRange::AdaCurve {
                            min,
                            max,
                            model_version,
                        })
                    }
                    "sst" => {
                        // SST uses block ranges
                        let start = block
                            .get("block_start")
                            .and_then(|b| b.as_u64())
                            .unwrap_or(i as u64) as u32;
                        let end = block
                            .get("block_end")
                            .and_then(|b| b.as_u64())
                            .unwrap_or(i as u64 + 1) as u32;
                        Some(SpatialRange::BlockRange {
                            start_block: start,
                            end_block: end,
                        })
                    }
                    _ => None,
                };

                // Extract centroid if present
                if let Some(centroid_array) = block.get("centroid").and_then(|c| c.as_array()) {
                    rg.centroid = Some(
                        centroid_array
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect(),
                    );
                }

                // Extract column bounds if present
                if let Some(stats_obj) = block.get("column_stats").and_then(|s| s.as_object()) {
                    for (col_name, col_stats) in stats_obj {
                        if let Some(bounds) = Self::json_to_column_bounds(col_stats) {
                            rg.column_stats.insert(col_name.clone(), bounds);
                        }
                    }
                }

                // Extract compression info
                rg.compression = block
                    .get("compression")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string());

                // Extract encoding info
                if let Some(encoding) = block.get("encoding").and_then(|e| e.as_object()) {
                    rg.encoding = Some(EncodingInfo {
                        quantization_type: encoding
                            .get("quantization")
                            .and_then(|q| q.as_str())
                            .unwrap_or("None")
                            .to_string(),
                        bits: encoding
                            .get("bits")
                            .and_then(|b| b.as_u64())
                            .map(|b| b as u8),
                        pq_subquantizers: encoding
                            .get("pq_subquantizers")
                            .and_then(|p| p.as_u64())
                            .map(|p| p as u8),
                        pq_codebook_size: encoding
                            .get("pq_codebook_size")
                            .and_then(|c| c.as_u64())
                            .map(|c| c as u16),
                    });
                }

                blocks.push(rg);
            }
        }

        // If no blocks in metadata, create a single block for the whole file
        if blocks.is_empty() {
            let data_offset = info.header_size as u64;
            let data_length = info.total_size.saturating_sub(data_offset);
            blocks.push(RowGroupMeta::new(0, data_offset, data_length, 0));
        }

        blocks
    }

    /// Convert JSON stats to ColumnBounds.
    fn json_to_column_bounds(stats: &serde_json::Value) -> Option<ColumnBounds> {
        let min_val = stats.get("min")?;
        let max_val = stats.get("max")?;

        let min = Self::json_to_column_value(min_val)?;
        let max = Self::json_to_column_value(max_val)?;
        let null_count = stats
            .get("null_count")
            .and_then(|n| n.as_i64())
            .unwrap_or(0);
        let distinct_count = stats.get("distinct_count").and_then(|d| d.as_i64());

        Some(ColumnBounds {
            min,
            max,
            null_count,
            distinct_count,
        })
    }

    /// Convert JSON value to ColumnValue.
    fn json_to_column_value(val: &serde_json::Value) -> Option<ColumnValue> {
        match val {
            serde_json::Value::Null => Some(ColumnValue::Null),
            serde_json::Value::Bool(b) => Some(ColumnValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(ColumnValue::Int64(i))
                } else {
                    n.as_f64().map(ColumnValue::Float64)
                }
            }
            serde_json::Value::String(s) => Some(ColumnValue::String(s.clone())),
            _ => None,
        }
    }
}

/// Parsed ProximaBlocks header information.
struct ProximaBlocksHeaderInfo {
    #[allow(dead_code)]
    version: u32,
    header_size: usize,
    metadata: serde_json::Value,
    total_size: u64,
}

#[async_trait]
impl HeaderLoader for ProximaBlocksHeaderLoader {
    async fn load_header(&self, path: &str) -> anyhow::Result<CachedHeader> {
        debug!("Loading ProximaBlocks header from: {}", path);

        // Read first 64KB to get header (headers are typically < 16KB)
        let data = self.filesystem.read(path).await?;
        let header_info = self.parse_header(&data)?;

        // Build cached header
        let mut header = CachedHeader::new(path.to_string(), 0);
        header.format_type = self.engine_type.clone();
        header.file_size = data.len() as u64;
        header.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Extract schema info from metadata
        if let Some(schema_fingerprint) = header_info
            .metadata
            .get("schema_fingerprint")
            .and_then(|f| f.as_u64())
        {
            header.schema_fingerprint = schema_fingerprint;
        }
        if let Some(schema_version) = header_info
            .metadata
            .get("schema_version")
            .and_then(|v| v.as_u64())
        {
            header.schema_version = schema_version as u32;
        }

        // Extract blocks
        header.rowgroups = self.extract_blocks(&header_info);

        // Estimate header size
        header.header_size_bytes = header_info.header_size;

        // Store engine-specific metadata
        if let Some(engine_meta) = header_info
            .metadata
            .get("engine")
            .and_then(|e| e.as_object())
        {
            for (key, value) in engine_meta {
                if let Some(s) = value.as_str() {
                    header.engine_metadata.insert(key.clone(), s.to_string());
                }
            }
        }

        info!(
            "Loaded ProximaBlocks header for {} ({} blocks, {} bytes)",
            path,
            header.rowgroups.len(),
            header.file_size
        );

        Ok(header)
    }

    fn supports_format(&self, format_type: &str) -> bool {
        matches!(
            format_type.to_lowercase().as_str(),
            "sst" | "helix" | "swift" | "proximablocks"
        )
    }
}

// ============================================================================
// Unified Header Loader Registry
// ============================================================================

/// Registry of all header loaders for different engine types.
pub struct HeaderLoaderRegistry {
    /// Registered header loaders
    loaders: Vec<Arc<dyn HeaderLoader>>,
}

impl HeaderLoaderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            loaders: Vec::new(),
        }
    }

    /// Create a registry with all default loaders.
    pub fn with_defaults(filesystem: Arc<dyn FileSystem>) -> Self {
        let mut registry = Self::new();

        // Parquet-based loaders
        registry.register(Arc::new(ParquetHeaderLoader::nova(filesystem.clone())));
        registry.register(Arc::new(ParquetHeaderLoader::raptor(filesystem.clone())));

        // ProximaBlocks-based loaders
        registry.register(Arc::new(ProximaBlocksHeaderLoader::sst(filesystem.clone())));
        registry.register(Arc::new(ProximaBlocksHeaderLoader::helix(
            filesystem.clone(),
        )));
        registry.register(Arc::new(ProximaBlocksHeaderLoader::swift(filesystem)));

        registry
    }

    /// Register a new loader.
    pub fn register(&mut self, loader: Arc<dyn HeaderLoader>) {
        self.loaders.push(loader);
    }

    /// Find a loader for the given format type.
    pub fn find_loader(&self, format_type: &str) -> Option<Arc<dyn HeaderLoader>> {
        self.loaders
            .iter()
            .find(|l| l.supports_format(format_type))
            .cloned()
    }

    /// Load header using appropriate loader.
    pub async fn load_header(
        &self,
        path: &str,
        format_hint: Option<&str>,
    ) -> anyhow::Result<CachedHeader> {
        // Try format hint first
        if let Some(fmt) = format_hint
            && let Some(loader) = self.find_loader(fmt)
        {
            return loader.load_header(path).await;
        }

        // Auto-detect from file extension
        let format = Self::detect_format_from_path(path);
        if let Some(loader) = self.find_loader(&format) {
            return loader.load_header(path).await;
        }

        // Try all loaders
        for loader in &self.loaders {
            match loader.load_header(path).await {
                Ok(header) => return Ok(header),
                Err(e) => {
                    debug!("Loader failed for {}: {:?}", path, e);
                }
            }
        }

        Err(anyhow::anyhow!("No suitable loader found for: {}", path))
    }

    /// Detect format from file path.
    fn detect_format_from_path(path: &str) -> String {
        if path.ends_with(".parquet") {
            "viper".to_string()
        } else if path.ends_with(".sst") {
            "sst".to_string()
        } else if path.ends_with(".helix") {
            "helix".to_string()
        } else if path.ends_with(".swift") {
            "swift".to_string()
        } else if path.ends_with(".nova") {
            "nova".to_string()
        } else if path.ends_with(".raptor") {
            "raptor".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

impl Default for HeaderLoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            HeaderLoaderRegistry::detect_format_from_path("/data/test.parquet"),
            "viper"
        );
        assert_eq!(
            HeaderLoaderRegistry::detect_format_from_path("/data/test.sst"),
            "sst"
        );
        assert_eq!(
            HeaderLoaderRegistry::detect_format_from_path("/data/test.helix"),
            "helix"
        );
        assert_eq!(
            HeaderLoaderRegistry::detect_format_from_path("/data/test.swift"),
            "swift"
        );
    }

    #[test]
    fn test_json_to_column_value() {
        let int_val = serde_json::json!(42);
        let result = ProximaBlocksHeaderLoader::json_to_column_value(&int_val);
        assert!(matches!(result, Some(ColumnValue::Int64(42))));

        let str_val = serde_json::json!("hello");
        let result = ProximaBlocksHeaderLoader::json_to_column_value(&str_val);
        assert!(matches!(result, Some(ColumnValue::String(s)) if s == "hello"));

        let null_val = serde_json::json!(null);
        let result = ProximaBlocksHeaderLoader::json_to_column_value(&null_val);
        assert!(matches!(result, Some(ColumnValue::Null)));
    }
}
