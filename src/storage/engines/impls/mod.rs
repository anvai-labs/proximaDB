//! Storage Engine Implementations - DEPRECATED
//!
//! **⚠️ DEPRECATED**: This module is deprecated and will be removed in a future release.
//! All storage engines have been moved to the top-level `src/storage/engines/` directory.
//!
//! ## Migration Guide
//!
//! Update your imports from:
//! ```rust,ignore
//! use proximadb::storage::engines::impls::SstEngine;
//! ```
//!
//! To:
//! ```rust,ignore
//! use proximadb::storage::engines::SstEngine;
//! ```
//!
//! ## Engine Locations (Post-Consolidation)
//!
//! All engines are now directly available at `crate::storage::engines::<engine_name>`:
//! - **SST**: `crate::storage::engines::sst` (Phase 1: Moved from impls/sst/)
//! - **HELIX**: `crate::storage::engines::helix` (Phase 1: Moved from impls/helix/)
//! - **NOVA**: `crate::storage::engines::nova` (Phase 1: Moved from impls/nova/)
//! - **SWIFT**: `crate::storage::engines::swift` (Phase 1: Moved from impls/swift/)
//! - **RAPTOR**: `crate::storage::engines::raptor` (Phase 1: Moved from impls/raptor/)
//! - **CEDAR**: `crate::storage::engines::cedar` (Phase 2: Moved from impls/cedar/)
//! - **EventLog**: `crate::storage::engines::eventlog` (Phase 2: Moved from impls/eventlog/)
//! - **SEQUOIA**: `crate::storage::engines::sequoia` (Phase 2: Moved from impls/sequoia/)
//! - **TST**: `crate::storage::engines::tst` (Phase 2: Moved from impls/tst/)
//!
//! CHRONO and TITAN were removed as unused stubs — see TD-ENGINE-1.

// All engines moved to engines/ level in Phase 1 & Phase 2 consolidation
// Test infrastructure has been inlined into individual engine modules
