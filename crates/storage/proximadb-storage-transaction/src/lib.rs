/*
 * Copyright 2025 ProximaDB
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

//! Multi-Model Transaction Manager
//!
//! Provides ACID transaction support across Vector, Document, Graph, and Observability stores.
//! Implements distributed two-phase commit (2PC) with proper isolation and rollback.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 MultiModelTransactionManager                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
//! │  │ Transaction  │  │   2PC        │  │   Isolation Level   │  │
//! │  │   Context    │  │ Coordinator  │  │   Management        │  │
//! │  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
//! ├─────────┼─────────────────┼─────────────────────┼─────────────┤
//! │   ┌─────▼─────┐     ┌─────▼─────┐         ┌─────▼─────┐       │
//! │   │  Vector   │     │ Document  │         │   Graph   │       │
//! │   │  Store    │     │  Store    │         │   Store   │       │
//! │   └───────────┘     └───────────┘         └───────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! let tx_manager = MultiModelTransactionManager::new(config);
//!
//! // Begin transaction
//! let tx = tx_manager.begin(IsolationLevel::Serializable).await?;
//!
//! // Perform operations across different stores
//! tx.vector_insert("embeddings", records).await?;
//! tx.document_insert("metadata", docs).await?;
//! tx.graph_add_edge("relationships", edge).await?;
//!
//! // Commit atomically across all stores
//! tx.commit().await?;
//! ```
//!
//! Extracted from the root crate's `src/storage/transaction` as a slice of the root-crate
//! decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`).
//! The root crate re-exports it as `crate::storage::transaction` for source compatibility.

pub mod context;
pub mod isolation;
pub mod log;
pub mod manager;
pub mod operations;

pub use context::{OperationType, TransactionContext, TransactionOperation};
pub use isolation::{ConflictResolution, IsolationLevel};
pub use manager::{MultiModelTransactionManager, TransactionConfig};
pub use operations::{DocumentOperation, GraphOperation, ObservabilityOperation, VectorOperation};
