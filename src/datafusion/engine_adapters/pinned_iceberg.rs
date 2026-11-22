//! Iceberg snapshot binding and Arrow batch source implementation.

use std::ops::Range;

use arrow_schema::SchemaRef;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use proximadb_iceberg_engine::{IcebergObjectStoreBridge, PinnedIcebergReader};

use super::pinned_external::validate_external_schema;
use crate::query::execution::pinned_source::{PinnedBatchSource, PinnedBatchStream};
use crate::services::ddl::pinned_iceberg::PinnedIcebergLocation;

pub use super::pinned_table::PinnedTable;

/// Read-only provider constructed from a pinned Iceberg snapshot reader.
pub type PinnedIcebergTable = PinnedTable;

impl PinnedBatchSource for PinnedIcebergReader {
    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn split_count(&self) -> usize {
        self.snapshot().plan.files.data_files.len()
    }

    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream> {
        Ok(self
            .scan_files(splits, projection, batch_size)?
            .map_err(anyhow::Error::from)
            .boxed())
    }
}

impl PinnedTable {
    /// Wrap a reader that has already resolved the authorized metadata URI.
    pub fn new(reader: PinnedIcebergReader) -> Self {
        Self::from_source(reader)
    }
}

pub(super) async fn open_reader(
    location: PinnedIcebergLocation,
    columns: &[proximadb_catalog::CatalogColumn],
) -> anyhow::Result<PinnedIcebergReader> {
    let bridge = if location.table_root.starts_with("file://") {
        #[cfg(not(unix))]
        anyhow::bail!("confined local Iceberg reads require Unix");
        #[cfg(unix)]
        {
            use std::sync::Arc;

            use super::confined_file_store::ConfinedFileStore;
            use crate::services::ddl::external_location_root;
            use proximadb_object_store::ProximaObjectStore;

            let root = external_location_root(&location.table_root).ok_or_else(|| {
                anyhow::anyhow!("external Iceberg root is outside the operator allowlist")
            })?;
            let relative = location
                .table_root
                .strip_prefix(&root)
                .ok_or_else(|| anyhow::anyhow!("invalid external Iceberg root"))?;
            let relative = Path::parse(relative)?;
            let directory = url::Url::parse(&root)?.to_file_path().map_err(|_| {
                anyhow::anyhow!("external Iceberg allowlist root is not a local directory URI")
            })?;
            let store = ConfinedFileStore::open(directory, relative).await?;
            IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(store)))
                .with_root_url(&location.table_root)
        }
    } else {
        IcebergObjectStoreBridge::from_url(&location.table_root)?
    };
    let reader = bridge
        .open_pinned_iceberg_reader(
            &Path::ROOT,
            &location.metadata_uri,
            location.expected_delta_version,
        )
        .await?;
    validate_external_schema(&reader.schema(), columns)
        .map_err(|e| anyhow::anyhow!("external Iceberg {e}"))?;
    Ok(reader)
}
