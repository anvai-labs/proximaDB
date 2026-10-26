//! Collection management
//!
//! Handles collection lifecycle, metadata, and configuration

pub mod engine_selector;
pub mod manager;
pub mod owner_backfill;
pub mod recall_target;
pub mod security;
pub mod storage_class;

pub use manager::{
    CollectionService as Collections, CollectionServiceBuilder as Builder,
    CollectionServiceResponse as Response,
};
pub use security::SecureCollectionService;

/// Type alias for [`Collections`] (primary collection management service).
pub type Manager = Collections;
