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

//! # Columnar Query Module - M2 Dual Columnar Execution
//!
//! This module provides the `ColumnarReadProvider` abstraction for unified
//! columnar data access in ProximaDB, enabling efficient query execution
//! across in-memory and on-disk data sources.
//!
//! ## Architecture
//!
//! ```text
//! UnifiedQueryFacade
//!        |
//!        v
//! ColumnarStrategy (QueryStrategy impl)
//!        |
//!        +---> ColumnarReadProvider (trait)
//!              |
//!              +---> ArrowInMemoryProvider (cached data)
//!              +---> ParquetRangePrunedProvider (NOVA)
//!              +---> ProximaBlockProvider (SST columnar)
//! ```
//!
//! ## Key Features
//!
//! - **Predicate Pushdown**: Filter evaluation at storage level
//! - **Projection Pushdown**: Read only required columns
//! - **Statistics Pruning**: Skip irrelevant row groups/blocks
//! - **Streaming**: Memory-efficient batch-at-a-time processing
//! - **Zero-Copy**: Arrow IPC for cached data access
//!
//! ## Usage
//!
//! ```ignore
//! use crate::query::columnar::{ColumnarReadProvider, PredicatePushdownConfig};
//!
//! // Create provider for Parquet files
//! let provider = ParquetRangePrunedProvider::new(files, filesystem, collection_id, dim).await?;
//!
//! // Read with predicate pushdown
//! let config = PredicatePushdownConfig {
//!     filter: Some(filter_expression),
//!     projection: Some(vec!["id".into(), "vector".into()]),
//!     ..Default::default()
//! };
//! let batches = provider.read_batches(config).await?;
//! ```

pub mod provider;
pub mod providers;

// Re-export main types
pub use provider::{
    ColumnarAccessStats, ColumnarBatchStream, ColumnarCapabilities, ColumnarRange,
    ColumnarReadProvider, PredicatePushdownConfig,
};

// Re-export provider implementations
pub use providers::{ArrowInMemoryProvider, ParquetRangePrunedProvider};
