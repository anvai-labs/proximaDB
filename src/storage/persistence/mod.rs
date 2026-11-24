//! Data Persistence Layer
//!
//! This module provides all data persistence components ensuring durability and
//! crash recovery for vector data. It implements a write-ahead log (WAL) for
//! ACID guarantees, unified filesystem abstraction for cloud storage, and
//! intelligent disk management for optimal I/O performance.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    API Layer                             │
//! │              (Vector Operations)                         │
//! └────────────────────┬────────────────────────────────────┘
//!                      ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              Write-Ahead Log (WAL)                       │
//! │         - MemTable buffering                             │
//! │         - Sequential I/O for durability                   │
//! │         - Crash recovery support                         │
//! └────────────────────┬────────────────────────────────────┘
//!                      ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │            Filesystem Abstraction                        │
//! │    - Local FS, S3, Azure, GCS                           │
//! │    - Unified caching layer                              │
//! │    - Atomic operations                                   │
//! └────────────────────┬────────────────────────────────────┘
//!                      ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              Storage Engines                             │
//! │   SST, NOVA, HELIX, SWIFT, RAPTOR                      │
//! └─────────────────────────────────────────────────────────┘
//! ```

pub mod disk_manager;
pub mod filesystem;
pub mod write_ahead_log;

// Re-export main persistence types
pub use disk_manager::DiskManager;
pub use filesystem::{FilesystemConfig, FilesystemFactory};
pub use write_ahead_log::{
    WALConfig, WriteAheadLogManager, WriteBufferStrategyType as WALStrategyType,
};
// WalFactory removed - use WALBatchFactory for modern implementations
