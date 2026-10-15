//! Foundation search-result and filter-value types for ProximaDB.
//!
//! Extracted from `proximadb::core::search` as part of the root-crate
//! decomposition (Slice D / D1). These are the self-contained, foundation-tier
//! leaves of the search module — search-result records, the bounded result
//! queue, SQL/JSON value filtering, and JSON value (de)serialization — with no
//! upward dependencies on the storage/compute/index/query orchestration layers.
//!
//! The orchestration types that remain in `proximadb::core::search`
//! (the search engines) depend *down* on this crate; the root re-exports every
//! module here so existing `crate::core::search::{results, bounded_queue,
//! sql_value_filter, json_value_serde, json_comparison, search_params}::*`
//! paths resolve unchanged. The `UnifiedSearchParams` cluster has been hoisted
//! here too (gap 1) — it depended only on foundation types.

pub mod block_prune;
pub mod bounded_queue;
pub mod compiled_filter;
pub mod json_comparison;
pub mod json_value_serde;
pub mod predicate_shortfall;
pub mod results;
pub mod search_params;
pub mod sql_value_filter;
pub mod top_k_limit;

pub use predicate_shortfall::PredicateShortfall;
