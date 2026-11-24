//! Unified Read Access Strategy for all storage engines
//!
//! This module defines a consistent strategy pattern for choosing between
//! cached reads (via UnifiedCachingFilesystem) and direct reads (via FilesystemFactory).
//!
//! ## Architecture Decision
//!
//! All engines must implement this strategy pattern to optimize for:
//! - Selective reads: Use caching to reduce cloud API calls
//! - Full scans: Use direct reads to avoid cache pollution
//! - Writes: Always use direct filesystem (no caching benefit)

use proximadb_filter_expression::FilterExpression;

/// Unified read access strategy used across all storage engines
///
/// This enum provides consistent naming for read strategies to improve
/// code readability and maintenance across SST, SWIFT, NOVA, HELIX, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadAccessStrategy {
    /// Direct read without caching - for full sequential scans
    /// Used for: Compaction, AXIS indexing, batch operations, migrations
    /// Benefits: Avoids cache pollution, reduces memory usage
    DirectStream,

    /// Cached read with selective access - for point/range queries
    /// Used for: Point lookups, filtered scans, search operations
    /// Benefits: Reduces cloud API calls, improves latency
    CachedSelective { filter: Option<FilterExpression> },

    /// Cached read optimized for search workloads
    /// Used for: Vector similarity search with metadata filtering
    /// Benefits: Caches bloom filters and indexes for repeated access
    CachedSearch { prefetch_metadata: bool },

    /// Cached read for metadata-only operations
    /// Used for: Schema discovery, statistics gathering
    /// Benefits: Only caches metadata, not data blocks
    CachedMetadataOnly,

    /// Hybrid strategy that starts cached but falls back to direct
    /// Used for: Adaptive workloads that may change access patterns
    /// Benefits: Best of both worlds with runtime adaptation
    Adaptive {
        initial_strategy: Box<ReadAccessStrategy>,
        fallback_threshold: usize, // Switch after N cache misses
    },
}

impl ReadAccessStrategy {
    /// Check if this strategy should use caching
    pub fn should_use_cache(&self) -> bool {
        match self {
            Self::DirectStream => false,
            Self::CachedSelective { .. } => true,
            Self::CachedSearch { .. } => true,
            Self::CachedMetadataOnly => true,
            Self::Adaptive {
                initial_strategy, ..
            } => initial_strategy.should_use_cache(),
        }
    }

    /// Check if this strategy should prefetch metadata
    pub fn should_prefetch_metadata(&self) -> bool {
        match self {
            Self::DirectStream => false,
            Self::CachedSelective { .. } => false,
            Self::CachedSearch { prefetch_metadata } => *prefetch_metadata,
            Self::CachedMetadataOnly => true,
            Self::Adaptive {
                initial_strategy, ..
            } => initial_strategy.should_prefetch_metadata(),
        }
    }

    /// Check if this strategy should cache data blocks
    pub fn should_cache_data_blocks(&self) -> bool {
        match self {
            Self::DirectStream => false,
            Self::CachedSelective { .. } => true,
            Self::CachedSearch { .. } => true,
            Self::CachedMetadataOnly => false,
            Self::Adaptive {
                initial_strategy, ..
            } => initial_strategy.should_cache_data_blocks(),
        }
    }

    /// Convert legacy SST ReadStrategy to unified strategy
    pub fn from_sst_strategy(
        strategy: &super::super::sst::readers::sst_query_engine::ReadStrategy,
    ) -> Self {
        use super::super::sst::readers::sst_query_engine::ReadStrategy;

        match strategy {
            ReadStrategy::CompactionDirect => Self::DirectStream,
            ReadStrategy::FullScan => Self::DirectStream,
            ReadStrategy::FilteredScan(filter) => Self::CachedSelective {
                filter: Some(filter.clone()),
            },
            ReadStrategy::SearchOptimized => Self::CachedSearch {
                prefetch_metadata: true,
            },
        }
    }
}

/// Trait that all storage engine readers must implement
pub trait StrategyAwareReader {
    /// Get the current read strategy
    fn strategy(&self) -> &ReadAccessStrategy;

    /// Set the read strategy
    fn set_strategy(&mut self, strategy: ReadAccessStrategy);

    /// Check if using cached filesystem
    fn is_using_cache(&self) -> bool {
        self.strategy().should_use_cache()
    }
}

/// Standard reader naming convention for all engines
pub mod naming {
    /// Prefix for direct readers (no caching)
    pub const DIRECT_READER_PREFIX: &str = "Direct";

    /// Prefix for cached readers
    pub const CACHED_READER_PREFIX: &str = "Cached";

    /// Prefix for unified readers (strategy-aware)
    pub const UNIFIED_READER_PREFIX: &str = "Unified";

    /// Generate standard reader name for an engine
    pub fn reader_name(engine: &str, reader_type: &str) -> String {
        format!("{}{}{}", reader_type, engine.to_uppercase(), "Reader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_cache_decisions() {
        let direct = ReadAccessStrategy::DirectStream;
        assert!(!direct.should_use_cache());
        assert!(!direct.should_prefetch_metadata());
        assert!(!direct.should_cache_data_blocks());

        let cached = ReadAccessStrategy::CachedSelective { filter: None };
        assert!(cached.should_use_cache());
        assert!(!cached.should_prefetch_metadata());
        assert!(cached.should_cache_data_blocks());

        let search = ReadAccessStrategy::CachedSearch {
            prefetch_metadata: true,
        };
        assert!(search.should_use_cache());
        assert!(search.should_prefetch_metadata());
        assert!(search.should_cache_data_blocks());

        let metadata_only = ReadAccessStrategy::CachedMetadataOnly;
        assert!(metadata_only.should_use_cache());
        assert!(metadata_only.should_prefetch_metadata());
        assert!(!metadata_only.should_cache_data_blocks());
    }

    #[test]
    fn test_naming_conventions() {
        use naming::*;

        assert_eq!(reader_name("SST", DIRECT_READER_PREFIX), "DirectSSTReader");
        assert_eq!(reader_name("SST", CACHED_READER_PREFIX), "CachedSSTReader");
        assert_eq!(
            reader_name("SST", UNIFIED_READER_PREFIX),
            "UnifiedSSTReader"
        );

        assert_eq!(
            reader_name("Swift", DIRECT_READER_PREFIX),
            "DirectSWIFTReader"
        );
        assert_eq!(
            reader_name("Nova", CACHED_READER_PREFIX),
            "CachedNOVAReader"
        );
    }
}
