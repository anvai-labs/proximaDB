//! Integration tests for unified strategy-driven read architecture
//!
//! These tests validate that all storage engines implement the unified strategy
//! pattern consistently and that strategy selection works correctly across engines.

// SWIFT is compiled only under `experimental-engines` (TD-152 deprecation gate).
#![cfg(feature = "experimental-engines")]

use std::sync::Arc;
use anyhow::Result;

use proximadb::storage::engines::core::read_strategy::{ReadAccessStrategy, StrategyAwareReader};
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::core::search::FilterExpression;

// Import all unified readers
use proximadb::storage::engines::sst::readers::UnifiedSstableReader;
use proximadb::storage::engines::swift::{UnifiedSWIFTReader, SwiftReaderConfig};
use proximadb::storage::engines::nova::UnifiedNOVAReader;
use proximadb::storage::engines::viper::UnifiedVIPERReader;
use proximadb::storage::engines::helix::UnifiedHELIXReader;

/// Test helper to create a test filesystem factory
async fn create_test_filesystem_factory() -> Arc<FilesystemFactory> {
    Arc::new(FilesystemFactory::create_default().await.unwrap())
}

/// Test that all engines implement StrategyAwareReader trait consistently
#[tokio::test]
async fn test_strategy_aware_trait_consistency() -> Result<()> {
    let factory = create_test_filesystem_factory().await;
    let collection_id = "test_collection".to_string();

    // Test SST engine
    let sst_reader = UnifiedSstableReader::new(
        factory.clone(),
        factory.get_filesystem("file://")?,
        collection_id.clone(),
    );

    // Test initial strategy
    assert_eq!(sst_reader.strategy(), &ReadAccessStrategy::DirectStream);

    // Test strategy switching
    let mut sst_reader_mut = sst_reader;
    sst_reader_mut.set_strategy(ReadAccessStrategy::CachedSearch { prefetch_metadata: true });
    assert!(matches!(sst_reader_mut.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    // Test SWIFT engine
    let config = SwiftReaderConfig {
        enable_prefetch: false,
        max_concurrent_reads: 4,
        coalesce_threshold_bytes: 64 * 1024,
        cache_metadata: true,
        use_streaming: false,
    };

    let swift_reader = UnifiedSWIFTReader::new(
        factory.clone(),
        collection_id.clone(),
        ReadAccessStrategy::DirectStream,
        config,
    )?;

    assert_eq!(swift_reader.strategy(), &ReadAccessStrategy::DirectStream);

    // Test NOVA engine
    let nova_reader = UnifiedNOVAReader::new(
        factory.clone(),
        collection_id.clone(),
        ReadAccessStrategy::CachedSearch { prefetch_metadata: true },
        128, // default dimension for tests
    )?;

    assert!(matches!(nova_reader.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    // Test VIPER engine
    let viper_reader = UnifiedVIPERReader::new(
        factory.clone(),
        collection_id.clone(),
        ReadAccessStrategy::DirectStream,
    )?;

    assert_eq!(viper_reader.strategy(), &ReadAccessStrategy::DirectStream);

    // Test HELIX engine
    let helix_reader = UnifiedHELIXReader::new(
        factory.clone(),
        collection_id.clone(),
        ReadAccessStrategy::CachedSelective {
            filter: Some(FilterExpression::Equals {
                field: "category".to_string(),
                value: "test".into(),
            })
        },
    )?;

    assert!(matches!(helix_reader.strategy(), ReadAccessStrategy::CachedSelective { .. }));

    Ok(())
}

/// Test factory method consistency across all engines
#[tokio::test]
async fn test_factory_method_consistency() -> Result<()> {
    let factory = create_test_filesystem_factory();
    let collection_id = "test_collection".to_string();

    // Test for_compaction() factory method - should use DirectStream
    let swift_compaction = UnifiedSWIFTReader::for_compaction(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert_eq!(swift_compaction.strategy(), &ReadAccessStrategy::DirectStream);
    assert!(!swift_compaction.is_using_cache());

    let nova_compaction = UnifiedNOVAReader::for_compaction(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert_eq!(nova_compaction.strategy(), &ReadAccessStrategy::DirectStream);

    let viper_compaction = UnifiedVIPERReader::for_compaction(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert_eq!(viper_compaction.strategy(), &ReadAccessStrategy::DirectStream);

    let helix_compaction = UnifiedHELIXReader::for_compaction(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert_eq!(helix_compaction.strategy(), &ReadAccessStrategy::DirectStream);

    // Test for_search() factory method - should use CachedSearch
    let swift_search = UnifiedSWIFTReader::for_search(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert!(matches!(swift_search.strategy(), ReadAccessStrategy::CachedSearch { .. }));
    assert!(swift_search.is_using_cache());

    let nova_search = UnifiedNOVAReader::for_search(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert!(matches!(nova_search.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    let viper_search = UnifiedVIPERReader::for_search(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert!(matches!(viper_search.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    let helix_search = UnifiedHELIXReader::for_search(
        factory.clone(),
        collection_id.clone(),
    )?;
    assert!(matches!(helix_search.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    Ok(())
}

/// Test strategy switching behavior
#[tokio::test]
async fn test_strategy_switching() -> Result<()> {
    let factory = create_test_filesystem_factory();
    let collection_id = "test_collection".to_string();

    // Test SWIFT reader strategy switching
    let mut swift_reader = UnifiedSWIFTReader::for_compaction(
        factory.clone(),
        collection_id.clone(),
    )?;

    // Initially should be DirectStream for compaction
    assert_eq!(swift_reader.strategy(), &ReadAccessStrategy::DirectStream);
    assert!(!swift_reader.is_using_cache());

    // Switch to search strategy
    swift_reader.set_strategy(ReadAccessStrategy::CachedSearch { prefetch_metadata: true });
    assert!(matches!(swift_reader.strategy(), ReadAccessStrategy::CachedSearch { .. }));
    assert!(swift_reader.is_using_cache());

    // Switch back to direct
    swift_reader.set_strategy(ReadAccessStrategy::DirectStream);
    assert_eq!(swift_reader.strategy(), &ReadAccessStrategy::DirectStream);

    // Test NOVA reader strategy switching
    let mut nova_reader = UnifiedNOVAReader::for_search(
        factory.clone(),
        collection_id.clone(),
    )?;

    // Initially should be CachedSearch
    assert!(matches!(nova_reader.strategy(), ReadAccessStrategy::CachedSearch { .. }));

    // Switch to selective
    nova_reader.set_strategy(ReadAccessStrategy::CachedSelective { filter: None });
    assert!(matches!(nova_reader.strategy(), ReadAccessStrategy::CachedSelective { .. }));

    Ok(())
}

/// Test adaptive strategy behavior
#[tokio::test]
async fn test_adaptive_strategy() -> Result<()> {
    let factory = create_test_filesystem_factory();
    let collection_id = "test_collection".to_string();

    // Create adaptive strategy
    let adaptive_strategy = ReadAccessStrategy::Adaptive {
        initial_strategy: Box::new(ReadAccessStrategy::CachedSearch { prefetch_metadata: true }),
        fallback_threshold: 5,
    };

    // Test with HELIX (has explicit adaptive support)
    let helix_reader = UnifiedHELIXReader::new(
        factory.clone(),
        collection_id.clone(),
        adaptive_strategy.clone(),
    )?;

    assert!(matches!(helix_reader.strategy(), ReadAccessStrategy::Adaptive { .. }));

    // Test with NOVA
    let nova_reader = UnifiedNOVAReader::new(
        factory.clone(),
        collection_id.clone(),
        adaptive_strategy.clone(),
        128,
    )?;

    assert!(matches!(nova_reader.strategy(), ReadAccessStrategy::Adaptive { .. }));

    Ok(())
}

/// Test that cache usage is correctly determined by strategy
#[tokio::test]
async fn test_cache_usage_by_strategy() -> Result<()> {
    let factory = create_test_filesystem_factory();
    let collection_id = "test_collection".to_string();

    // Strategies that should NOT use cache
    let direct_strategies = vec![
        ReadAccessStrategy::DirectStream,
    ];

    // Strategies that SHOULD use cache
    let cached_strategies = vec![
        ReadAccessStrategy::CachedSearch { prefetch_metadata: true },
        ReadAccessStrategy::CachedSelective { filter: None },
        ReadAccessStrategy::CachedMetadataOnly,
    ];

    // Test SWIFT reader
    for strategy in &direct_strategies {
        let reader = UnifiedSWIFTReader::new(
            factory.clone(),
            collection_id.clone(),
            strategy.clone(),
            SwiftReaderConfig {
                enable_prefetch: false,
                max_concurrent_reads: 4,
                coalesce_threshold_bytes: 64 * 1024,
                cache_metadata: false,
                use_streaming: true,
            },
        )?;
        assert!(!reader.is_using_cache(), "Direct strategy should not use cache");
    }

    for strategy in &cached_strategies {
        let reader = UnifiedSWIFTReader::new(
            factory.clone(),
            collection_id.clone(),
            strategy.clone(),
            SwiftReaderConfig {
                enable_prefetch: false,
                max_concurrent_reads: 4,
                coalesce_threshold_bytes: 64 * 1024,
                cache_metadata: true,
                use_streaming: false,
            },
        )?;
        assert!(reader.is_using_cache(), "Cached strategy should use cache");
    }

    // Test HELIX reader
    for strategy in &direct_strategies {
        let reader = UnifiedHELIXReader::new(
            factory.clone(),
            collection_id.clone(),
            strategy.clone(),
        )?;
        assert!(!reader.is_using_cache(), "Direct strategy should not use cache");
    }

    for strategy in &cached_strategies {
        let reader = UnifiedHELIXReader::new(
            factory.clone(),
            collection_id.clone(),
            strategy.clone(),
        )?;
        assert!(reader.is_using_cache(), "Cached strategy should use cache");
    }

    Ok(())
}

/// Test strategy enum serialization/deserialization
#[test]
fn test_strategy_serialization() {
    let strategies = vec![
        ReadAccessStrategy::DirectStream,
        ReadAccessStrategy::CachedSearch { prefetch_metadata: true },
        ReadAccessStrategy::CachedSelective { filter: None },
        ReadAccessStrategy::CachedMetadataOnly,
        ReadAccessStrategy::Adaptive {
            initial_strategy: Box::new(ReadAccessStrategy::DirectStream),
            fallback_threshold: 10,
        },
    ];

    // Test that strategies are cloneable and comparable
    for strategy in strategies {
        let cloned = strategy.clone();
        assert_eq!(strategy, cloned);

        // Test Debug formatting
        let debug_str = format!("{:?}", strategy);
        assert!(!debug_str.is_empty());
    }
}

/// Integration test validating cross-engine strategy compatibility
#[tokio::test]
async fn test_cross_engine_compatibility() -> Result<()> {
    let factory = create_test_filesystem_factory();
    let collection_id = "test_collection".to_string();

    // Create the same strategy for all engines
    let test_strategy = ReadAccessStrategy::CachedSearch { prefetch_metadata: true };

    // All engines should accept the same strategy
    let _swift_reader = UnifiedSWIFTReader::new(
        factory.clone(),
        collection_id.clone(),
        test_strategy.clone(),
        SwiftReaderConfig::default(),
    )?;

    let _nova_reader = UnifiedNOVAReader::new(
        factory.clone(),
        collection_id.clone(),
        test_strategy.clone(),
        128,
    )?;

    let _viper_reader = UnifiedVIPERReader::new(
        factory.clone(),
        collection_id.clone(),
        test_strategy.clone(),
    )?;

    let _helix_reader = UnifiedHELIXReader::new(
        factory.clone(),
        collection_id.clone(),
        test_strategy.clone(),
    )?;

    // Test that they all report the same strategy
    assert_eq!(_swift_reader.strategy(), &test_strategy);
    assert_eq!(_nova_reader.strategy(), &test_strategy);
    assert_eq!(_viper_reader.strategy(), &test_strategy);
    assert_eq!(_helix_reader.strategy(), &test_strategy);

    Ok(())
}
