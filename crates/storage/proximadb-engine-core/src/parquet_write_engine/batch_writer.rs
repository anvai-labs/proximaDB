//! Batch Parquet Writer
//!
//! This module provides batch writing capabilities for Parquet files,
//! optimized for bulk operations where all data is available upfront.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

use proximadb_proto::proximadb_v1::FilterableColumnSpec;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::metadata_collector::MetadataCollector;

use super::{
    streaming_writer::StreamingParquetWriter, writer_config::ParquetWriterConfig,
    writer_statistics::StreamingParquetWriterStats,
};

/// Batch Parquet writer for bulk operations
pub struct BatchParquetWriter {
    config: ParquetWriterConfig,
    file_path: String,
    dimension: usize,
    filterable_columns: Option<Vec<FilterableColumnSpec>>,
    metadata_collector: Option<Box<dyn MetadataCollector>>,
    /// Filesystem port (TD-DECOMP-78 seam) — injected root-side.
    filesystem_factory: Option<std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>>,
    /// Quantization encoders (TD-DECOMP-78 seam) — injected root-side.
    quantization_engine:
        Option<std::sync::Arc<dyn proximadb_storage_ports::QuantizationEnginePort>>,
}

impl BatchParquetWriter {
    /// Inject the filesystem + quantization ports (root composition root).
    pub fn with_ports(
        mut self,
        filesystem_factory: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
        quantization_engine: Option<
            std::sync::Arc<dyn proximadb_storage_ports::QuantizationEnginePort>,
        >,
    ) -> Self {
        self.filesystem_factory = Some(filesystem_factory);
        self.quantization_engine = quantization_engine;
        self
    }

    /// Create new batch writer
    pub fn new<P: AsRef<Path>>(
        file_path: P,
        dimension: usize,
        config: ParquetWriterConfig,
    ) -> Self {
        Self {
            config,
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            dimension,
            filesystem_factory: None,
            quantization_engine: None,
            filterable_columns: None,
            metadata_collector: None,
        }
    }

    /// Set filterable columns for the writer
    pub fn with_filterable_columns(mut self, columns: Vec<FilterableColumnSpec>) -> Self {
        self.filterable_columns = Some(columns);
        self
    }

    /// Set metadata collector for hierarchical metadata (NOVA engine)
    pub fn with_metadata_collector(mut self, collector: Box<dyn MetadataCollector>) -> Self {
        self.metadata_collector = Some(collector);
        self
    }

    /// Write all records at once with optional metadata collection
    pub async fn write_all(
        &mut self,
        records: &[ProximaRecord],
    ) -> Result<(
        StreamingParquetWriterStats,
        Option<Box<dyn MetadataCollector>>,
    )> {
        info!(
            "Batch writing {} records to {}",
            records.len(),
            self.file_path
        );

        // Create streaming writer with batch configuration
        let filesystem_factory = self.filesystem_factory.clone().ok_or_else(|| {
            anyhow::anyhow!("filesystem port required (inject via with_ports root-side)")
        })?;
        let mut writer = StreamingParquetWriter::new(
            &self.file_path,
            self.dimension,
            self.config.clone(),
            self.filterable_columns.as_deref(),
            filesystem_factory,
            self.quantization_engine.clone(),
        )
        .await?;

        // Add metadata collector if provided
        if let Some(collector) = self.metadata_collector.take() {
            writer = writer.with_metadata_collector(collector);
        }

        // Calculate optimal batch size based on row group size
        let batch_size = self.config.write_batch_size.min(self.config.row_group_size);

        // Write records in batches
        for chunk in records.chunks(batch_size) {
            writer
                .write_batch(chunk)
                .await
                .context("Failed to write batch")?;
        }

        // Finalize and get statistics
        let (stats, _data, collector) = writer.finalize().await?;
        Ok((stats, collector))
    }

    /// Write all records and return only statistics (convenience method)
    pub async fn write_all_simple(
        &mut self,
        records: &[ProximaRecord],
    ) -> Result<StreamingParquetWriterStats> {
        let (stats, _) = self.write_all(records).await?;
        Ok(stats)
    }
}

/// Builder for BatchParquetWriter
pub struct BatchWriterBuilder {
    file_path: Option<String>,
    dimension: Option<usize>,
    config: ParquetWriterConfig,
    filterable_columns: Option<Vec<FilterableColumnSpec>>,
    metadata_collector: Option<Box<dyn MetadataCollector>>,
}

impl BatchWriterBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            file_path: None,
            dimension: None,
            config: ParquetWriterConfig::default(),
            filterable_columns: None,
            metadata_collector: None,
        }
    }

    /// Set file path
    pub fn with_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.file_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set vector dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: ParquetWriterConfig) -> Self {
        self.config = config;
        self
    }

    /// Set filterable columns
    pub fn with_filterable_columns(mut self, columns: Vec<FilterableColumnSpec>) -> Self {
        self.filterable_columns = Some(columns);
        self
    }

    /// Set metadata collector
    pub fn with_metadata_collector(mut self, collector: Box<dyn MetadataCollector>) -> Self {
        self.metadata_collector = Some(collector);
        self
    }

    /// Build the writer
    pub fn build(self) -> Result<BatchParquetWriter> {
        let file_path = self
            .file_path
            .ok_or_else(|| anyhow::anyhow!("File path is required"))?;
        let dimension = self
            .dimension
            .ok_or_else(|| anyhow::anyhow!("Dimension is required"))?;

        let mut writer = BatchParquetWriter::new(file_path, dimension, self.config);

        if let Some(columns) = self.filterable_columns {
            writer = writer.with_filterable_columns(columns);
        }

        if let Some(collector) = self.metadata_collector {
            writer = writer.with_metadata_collector(collector);
        }

        Ok(writer)
    }
}

impl Default for BatchWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test_local_port {
    //! tokio::fs-backed `FilesystemPort` for batch-writer tests (they write real
    //! local files; the port keeps the production seam intact).
    use async_trait::async_trait;
    use proximadb_storage_filesystem_types::{
        DirEntry, FileOptions, FileSystem, FilesystemError, FsFileMetadata, FsResult,
    };
    use std::sync::Arc;

    pub(super) fn local_port() -> Arc<dyn proximadb_storage_ports::FilesystemPort> {
        Arc::new(TokioLocalPort)
    }

    struct TokioLocalPort;

    fn io_err(e: std::io::Error) -> FilesystemError {
        FilesystemError::Io(e)
    }

    #[async_trait]
    impl proximadb_storage_ports::FilesystemPort for TokioLocalPort {
        fn get_filesystem(&self, _url: &str) -> FsResult<Arc<dyn FileSystem>> {
            Ok(Arc::new(LocalFs))
        }
        async fn create_dir_all(&self, url: &str) -> FsResult<()> {
            let path = url.trim_start_matches("file://");
            tokio::fs::create_dir_all(path)
                .await
                .map_err(|e| FilesystemError::Io(e))
        }
        async fn write(
            &self,
            url: &str,
            data: &[u8],
            _options: Option<FileOptions>,
        ) -> FsResult<()> {
            let path = url.trim_start_matches("file://");
            if let Some(parent) = std::path::Path::new(path).parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| FilesystemError::Io(e))?;
            }
            tokio::fs::write(path, data)
                .await
                .map_err(|e| FilesystemError::Io(e))
        }
        async fn move_atomic(&self, from_url: &str, to_url: &str) -> FsResult<()> {
            tokio::fs::rename(
                from_url.trim_start_matches("file://"),
                to_url.trim_start_matches("file://"),
            )
            .await
            .map_err(|e| FilesystemError::Io(e))
        }
        async fn delete(&self, url: &str) -> FsResult<()> {
            tokio::fs::remove_file(url.trim_start_matches("file://"))
                .await
                .map_err(|e| FilesystemError::Io(e))
        }
        async fn read(&self, url: &str) -> FsResult<Vec<u8>> {
            tokio::fs::read(url.trim_start_matches("file://"))
                .await
                .map_err(|e| FilesystemError::Io(e))
        }
        async fn list(&self, _url: &str) -> FsResult<Vec<DirEntry>> {
            Ok(Vec::new())
        }
    }

    /// Minimal local `FileSystem` backed by tokio::fs — enough for the
    /// streaming-writer finalize path in tests.
    #[derive(Debug)]
    struct LocalFs;

    fn strip(p: &str) -> &str {
        p.trim_start_matches("file://")
    }

    #[async_trait]
    impl FileSystem for LocalFs {
        fn filesystem_type(&self) -> &'static str {
            "local-test"
        }
        async fn create_dir_all(&self, path: &str) -> FsResult<()> {
            tokio::fs::create_dir_all(strip(path)).await.map_err(io_err)
        }
        async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
            tokio::fs::copy(strip(from), strip(to))
                .await
                .map_err(io_err)
                .map(|_| ())
        }
        async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
            tokio::fs::rename(strip(from), strip(to))
                .await
                .map_err(io_err)
        }
        async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
            tokio::fs::read(strip(path)).await.map_err(io_err)
        }
        async fn write(&self, path: &str, data: &[u8], _o: Option<FileOptions>) -> FsResult<()> {
            if let Some(parent) = std::path::Path::new(strip(path)).parent() {
                tokio::fs::create_dir_all(parent).await.map_err(io_err)?;
            }
            tokio::fs::write(strip(path), data).await.map_err(io_err)
        }
        async fn append(&self, path: &str, data: &[u8]) -> FsResult<()> {
            use tokio::io::AsyncWriteExt;
            let mut f = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(strip(path))
                .await
                .map_err(io_err)?;
            f.write_all(data).await.map_err(io_err)
        }
        async fn delete(&self, path: &str) -> FsResult<()> {
            tokio::fs::remove_file(strip(path)).await.map_err(io_err)
        }
        async fn create_dir(&self, path: &str) -> FsResult<()> {
            tokio::fs::create_dir(strip(path)).await.map_err(io_err)
        }
        async fn exists(&self, path: &str) -> FsResult<bool> {
            Ok(std::path::Path::new(strip(path)).exists())
        }
        async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
            let m = tokio::fs::metadata(strip(path)).await.map_err(io_err)?;
            Ok(FsFileMetadata {
                path: strip(path).to_string(),
                size: m.len(),
                created: None,
                modified: m.modified().ok().map(chrono::DateTime::from),
                is_directory: m.is_dir(),
                permissions: None,
                etag: None,
                storage_class: None,
            })
        }
        async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
            let mut rd = tokio::fs::read_dir(strip(path)).await.map_err(io_err)?;
            let mut out = Vec::new();
            while let Some(e) = rd.next_entry().await.map_err(io_err)? {
                out.push(DirEntry {
                    name: e.file_name().to_string_lossy().to_string(),
                    url: format!("file://{}", e.path().display()),
                    metadata: FsFileMetadata {
                        path: e.path().display().to_string(),
                        size: 0,
                        created: None,
                        modified: None,
                        is_directory: e.file_type().await.map(|t| t.is_dir()).unwrap_or(false),
                        permissions: None,
                        etag: None,
                        storage_class: None,
                    },
                });
            }
            Ok(out)
        }
        async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let mut f = tokio::fs::File::open(strip(path)).await.map_err(io_err)?;
            f.seek(std::io::SeekFrom::Start(offset))
                .await
                .map_err(io_err)?;
            let mut buf = vec![0u8; length as usize];
            f.read_exact(&mut buf).await.map_err(io_err)?;
            Ok(buf)
        }
        async fn sync(&self) -> FsResult<()> {
            Err(FilesystemError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "sync not needed in tests",
            )))
        }
        async fn open_file(
            &self,
            path: &str,
            _create: bool,
        ) -> FsResult<Box<dyn proximadb_storage_filesystem_types::FilesystemFile>> {
            let _ = path;
            Err(FilesystemError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "open_file not needed in tests",
            )))
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_batch_writer_basic() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_batch.parquet");

        let config = ParquetWriterConfig::default();
        let mut writer = BatchParquetWriter::new(&file_path, 128, config)
            .with_ports(test_local_port::local_port(), None);

        let records = vec![
            {
                let mut r = ProximaRecord {
                    oid: "test_1".to_string(),
                    ..Default::default()
                };
                r.embeddings.push(EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![1.0; 128],
                ));
                r
            },
            {
                let mut r = ProximaRecord {
                    oid: "test_2".to_string(),
                    ..Default::default()
                };
                r.embeddings.push(EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![2.0; 128],
                ));
                r
            },
        ];

        let stats = writer.write_all_simple(&records).await.unwrap();
        assert_eq!(stats.total_records, 2);
        assert!(stats.compressed_size > 0);
    }

    #[tokio::test]
    async fn test_batch_writer_with_filterable_columns() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_batch_filterable.parquet");

        let config = ParquetWriterConfig::default();
        let columns = vec![FilterableColumnSpec {
            name: "category".to_string(),
            data_type: 0, // STRING type
            indexed: false,
            supports_range: false,
            estimated_cardinality: Some(100),
        }];

        let mut writer = BatchParquetWriter::new(&file_path, 64, config)
            .with_filterable_columns(columns)
            .with_ports(test_local_port::local_port(), None);

        let mut r = ProximaRecord {
            oid: "test_1".to_string(),
            ..Default::default()
        };
        r.embeddings.push(EmbeddingCell::new_fp32(
            "default",
            "vector",
            64,
            vec![1.0; 64],
        ));
        r.props.insert(
            "category".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "test_category".to_string(),
            )),
        );
        let records = vec![r];

        let stats = writer.write_all_simple(&records).await.unwrap();
        assert_eq!(stats.total_records, 1);
        assert_eq!(stats.filterable_columns_count, 1);
    }

    #[test]
    fn test_batch_writer_builder() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_builder.parquet");

        let writer = BatchWriterBuilder::new()
            .with_path(&file_path)
            .with_dimension(256)
            .with_config(ParquetWriterConfig::for_analytics())
            .build()
            .unwrap();

        assert_eq!(writer.dimension, 256);
        assert_eq!(writer.file_path, file_path.to_string_lossy());
    }
}
