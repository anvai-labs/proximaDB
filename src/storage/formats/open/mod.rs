/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Open Table Format Implementations
//!
//! This module provides connectors for open table formats that enable
//! storage-compute separation and interoperability with the broader
//! data lakehouse ecosystem.
//!
//! ## Supported Formats
//!
//! | Format | Status | Features |
//! |--------|--------|----------|
//! | **Delta Lake** | ✓ Implemented | ACID, Time Travel, Z-ordering |
//! | **Iceberg** | Not in this module | Published by the warehouse path via `ObjectStoreBridge::publish_iceberg_table` |
//! | **Hudi** | Planned | Upserts, Incremental Queries |
//! | **LanceDB** | Planned | Vector-native, IVF+PQ |
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Open Table Format Layer                       │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
//! │  │   Delta Lake    │  │    Iceberg      │  │      Hudi       │ │
//! │  │  _delta_log/    │  │   metadata/     │  │    .hoodie/     │ │
//! │  │  Parquet files  │  │  Parquet files  │  │  Parquet files  │ │
//! │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
//! │           └────────────────────┼────────────────────┘           │
//! │                                ▼                                 │
//! │                     ┌─────────────────────┐                     │
//! │                     │  OpenTableFormat    │                     │
//! │                     │       Trait         │                     │
//! │                     └─────────────────────┘                     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::formats::open::DeltaLakeFormat;
//! use proximadb::storage::formats::{OpenTableFormat, ReadContext};
//!
//! // Delta Lake
//! let delta = DeltaLakeFormat::new("/path/to/delta/table").await?;
//! let snapshot = delta.get_current_snapshot("/path/to/delta/table").await?;
//! let batches = delta.read_snapshot(&snapshot, &ReadContext::default()).await?;
//! ```
//!
//! Iceberg is NOT served from here. The warehouse path publishes Iceberg
//! metadata through `ObjectStoreBridge::publish_iceberg_table`; the unused
//! parallel connector that used to live in this module was deleted by TD-USUB-9.

// Delta Lake connector
pub mod delta;

// TD-USUB-9: the Apache Iceberg connector module was DELETED. It was 1507 lines
// with zero non-doc references anywhere in the workspace — a parallel, unused
// implementation of a seam the warehouse path already serves through
// `ObjectStoreBridge::publish_iceberg_table`. Two implementations of one
// external contract is how they drift; the unused one is the one to remove.

// Re-exports
pub use delta::{DeltaLakeConfig, DeltaLakeFormat};

// ============================================================================
// Common Types
// ============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for object storage access
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageOptions {
    /// Storage URL (s3://, gs://, azure://, file://)
    pub url: String,

    /// AWS region (for S3)
    pub region: Option<String>,

    /// AWS access key ID
    pub access_key_id: Option<String>,

    /// AWS secret access key
    pub secret_access_key: Option<String>,

    /// AWS session token
    pub session_token: Option<String>,

    /// Azure storage account
    pub account_name: Option<String>,

    /// Azure storage account key
    pub account_key: Option<String>,

    /// GCS service account credentials JSON
    pub gcs_credentials: Option<String>,

    /// Additional storage options
    pub options: HashMap<String, String>,
}

impl StorageOptions {
    /// Create options for local filesystem
    pub fn local(path: &str) -> Self {
        Self {
            url: format!("file://{}", path),
            ..Default::default()
        }
    }

    /// Create options for S3
    pub fn s3(bucket: &str, region: &str) -> Self {
        Self {
            url: format!("s3://{}", bucket),
            region: Some(region.to_string()),
            ..Default::default()
        }
    }

    /// Create options for Azure Blob Storage
    pub fn azure(container: &str, account: &str) -> Self {
        Self {
            url: format!("azure://{}", container),
            account_name: Some(account.to_string()),
            ..Default::default()
        }
    }

    /// Create options for GCS
    pub fn gcs(bucket: &str) -> Self {
        Self {
            url: format!("gs://{}", bucket),
            ..Default::default()
        }
    }

    /// With AWS credentials
    pub fn with_aws_credentials(mut self, access_key: &str, secret_key: &str) -> Self {
        self.access_key_id = Some(access_key.to_string());
        self.secret_access_key = Some(secret_key.to_string());
        self
    }

    /// With Azure credentials
    pub fn with_azure_credentials(mut self, account: &str, key: &str) -> Self {
        self.account_name = Some(account.to_string());
        self.account_key = Some(key.to_string());
        self
    }

    /// With additional option
    pub fn with_option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }
}

/// Table format metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    /// Table name
    pub name: String,

    /// Table location (path)
    pub location: String,

    /// Format type (delta, iceberg, etc.)
    pub format: String,

    /// Current version/snapshot ID
    pub current_version: i64,

    /// Total size in bytes
    pub size_bytes: u64,

    /// Total number of files
    pub file_count: usize,

    /// Total row count
    pub row_count: u64,

    /// Partition columns
    pub partition_columns: Vec<String>,

    /// Table properties
    pub properties: HashMap<String, String>,

    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last modified timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_options_local() {
        let opts = StorageOptions::local("/tmp/test");
        assert_eq!(opts.url, "file:///tmp/test");
    }

    #[test]
    fn test_storage_options_s3() {
        let opts = StorageOptions::s3("my-bucket", "us-west-2");
        assert_eq!(opts.url, "s3://my-bucket");
        assert_eq!(opts.region, Some("us-west-2".to_string()));
    }

    #[test]
    fn test_storage_options_with_credentials() {
        let opts = StorageOptions::s3("bucket", "us-east-1").with_aws_credentials("AKID", "SECRET");
        assert_eq!(opts.access_key_id, Some("AKID".to_string()));
        assert_eq!(opts.secret_access_key, Some("SECRET".to_string()));
    }
}
