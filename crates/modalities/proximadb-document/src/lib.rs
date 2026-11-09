//! # ProximaDB Document Modality
//!
//! This crate contains document facade contracts, canonical record mapping, and
//! document projection primitives for ProximaDB.
//!
//! ## Architecture
//!
//! The document modality is organized into several key modules:
//!
//! - **`record`** - Canonical `ProximaRecord` mapping and storage contracts
//! - **`projection`** - Rebuildable projection contracts over canonical records
//! - **`structural_projection`** - JSON shape/path projection profiling
//! - **`query`** - Document query expressions (JSON path filters, full-text search)
//! - **`index`** - Full-text indexing/projection contracts (Tantivy-based)
//!
//! ## Foundation
//!
//! This crate serves as the modality boundary for document operations across
//! ProximaDB. Durable document truth flows through `ProximaRecord`; JSON path,
//! full-text, and columnar variation structures are projections over that
//! canonical record spine.
//!
//! It provides reusable contracts and implementations for:
//!
//! - Storage engines that need canonical document record mapping
//! - Query executors that need document operations
//! - Index builders that need full-text or JSON path projection inputs
//!
//! ## Dependencies
//!
//! - `proximadb-kernel` - Core error types and foundational contracts
//! - `proximadb-records` - Canonical `ProximaRecord` envelope
//! - `proximadb-data-model` - Canonical `ProximaValue` rich type system
//! - `proximadb-query-filter` - Filter expression contracts
//! - `arrow` - Columnar data structures for document operations

pub mod projection;
pub mod query;
pub mod record;
pub mod structural_projection;

// Re-export common types for convenience
pub use projection::{
    DocumentPath, DocumentProjection, DocumentProjectionDescriptor, DocumentProjectionKind,
    ProjectionApplyResult, document_value_at_path, projection_source_values,
    record_belongs_to_document_collection,
};
pub use query::{DocumentQueryExpr, DocumentSort, PathFilter};
pub use record::{
    CanonicalDocument, CanonicalDocumentStore, DOC_BLOB_PROP, DOCUMENT_COLLECTION_PROP,
    DOCUMENT_RECORD_LABEL, DOCUMENT_TYPE_PROP, DocumentRecordKey, DocumentRecordMetadata,
    canonical_document_from_record,
};
pub use structural_projection::{
    JsonLeafKind, JsonPathStats, JsonPresenceSummary, JsonShapeId, JsonShapeStats,
    JsonStructuralProfile, JsonStructuralProjection, JsonStructuralProjectionMetadata,
    JsonStructuralProjectionOptions, JsonTypedPathStripe, build_structural_projection,
    build_structural_projection_with_options, profile_props,
};

// TODO: Move these from src/storage/document and src/schema/document
// pub mod index;
// pub mod storage;
// pub mod aggregation;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_module_imports() {
        // Basic test to verify the module structure is working
        let _expr = DocumentQueryExpr {
            collection: "test".to_string(),
            path_filters: vec![],
            text_search: None,
            projection: vec![],
            sort: None,
            limit: None,
        };
        // More comprehensive tests will be added as modules are extracted
    }
}
