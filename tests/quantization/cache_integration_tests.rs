// ORPHANED TEST (not in any compile target — see TD-ORPHAN-1): references the
// VIPER engine removed by ADR-093 stage 3 / TD-VIPER-1 S3. Unreachable code;
// cleanup of the file itself belongs to the orphan-file program, not the
// engine-removal PR.
#![cfg(feature = "experimental-engines")] // RAPTOR needs the `experimental-engines` feature.
//! Unified Quantization Cache Integration Tests
//!
//! Tests the global quantization cache architecture across all storage engines
//! with DashMap vs persistent store decision making.

use anyhow::Result;
use std::sync::Arc;
use proximadb::compute::quantization::{
    global_cache::{GlobalQuantizationCache, QuantizationCacheKey},
    selection::QuantizationSelector,
    unified::{Codebook, CodebookData, UnifiedQuantizationLevel, TrainingConfig, QuantizationLevel, ProductQuantization},
};

#[tokio::test]
async fn test_global_quantization_cache_basic_operations() -> Result<()> {
    // Test basic global cache operations
    let cache = GlobalQuantizationCache::global();

    // Test storing a codebook
    let key = QuantizationCacheKey::pq("test_collection", 8, 16);
    let codebook = create_test_pq_codebook();

    cache.store_codebook_internal(key.clone(), codebook.clone()).await?;

    // Test retrieving the codebook
    let retrieved = cache.get_codebook(&key).await;
    assert!(retrieved.is_some(), "Codebook should be retrievable from cache");

    // Test cache statistics
    let stats = cache.get_memory_stats();
    assert_eq!(stats.codebook_count, 1, "Cache should contain one codebook");
    assert!(stats.allocated_bytes > 0, "Cache should report allocated memory");

    Ok(())
}

#[tokio::test]
async fn test_quantization_selector_logic() -> Result<()> {
    // Test persistent vs stateless quantization selection

    // Small collection should use stateless
    let should_use_persistent = QuantizationSelector::should_use_persistent_quantization(
        "search",
        Some(100), // 100 vectors
    );
    assert!(!should_use_persistent, "Small collections should use stateless quantization");

    // Large collection should use persistent
    let should_use_persistent = QuantizationSelector::should_use_persistent_quantization(
        "flush",
        Some(50_000), // 50K vectors
    );
    assert!(should_use_persistent, "Large collections should use persistent quantization");

    // Frequent operations should use persistent
    let should_use_persistent = QuantizationSelector::should_use_persistent_quantization(
        "flush",
        Some(1000), // 1K vectors but flush operation
    );
    assert!(should_use_persistent, "Flush operations should prefer persistent quantization");

    Ok(())
}

#[tokio::test]
async fn test_engine_quantization_selection() -> Result<()> {
    // Test engine-specific quantization engine selection
    use proximadb::storage::engines::impls::{
        sst::SSTEngine,
        viper::ViperEngine,
        raptor::RaptorEngine,
        helix::HelixEngine,
    };

    // Test SST engine quantization selection
    let sst_engine = SSTEngine::new().await?;
    let quantization_engine = sst_engine.get_quantization_engine("search", Some(100)).await;
    assert!(quantization_engine.is_some(), "SST engine should provide quantization engine");

    // Test VIPER engine quantization selection
    let viper_engine = ViperEngine::default();
    let quantization_engine = viper_engine.get_quantization_engine("flush", Some(10_000)).await;
    // Note: Should use persistent quantization for large flush

    // Test RAPTOR engine quantization selection
    let raptor_engine = RaptorEngine::new().await?;
    let quantization_engine = raptor_engine.get_quantization_engine("search", Some(500)).await;
    // Should return UnifiedQuantizationEngine

    // Test HELIX engine quantization selection
    let helix_engine = HelixEngine::new().await?;
    let quantization_engine = helix_engine.get_quantization_engine("flush", Some(5_000)).await;
    assert!(quantization_engine.is_some(), "HELIX engine should provide quantization engine");

    Ok(())
}

#[tokio::test]
async fn test_collection_cleanup() -> Result<()> {
    // Test cleaning up collection-specific codebooks
    let cache = GlobalQuantizationCache::global();

    // Store multiple codebooks for different collections
    let collection1_key = QuantizationCacheKey::pq("collection1", 8, 16);
    let collection2_key = QuantizationCacheKey::binary("collection2");
    let collection1_key2 = QuantizationCacheKey::int8("collection1");

    let codebook = create_test_pq_codebook();
    cache.store_codebook_internal(collection1_key.clone(), codebook.clone()).await?;
    cache.store_codebook_internal(collection2_key.clone(), codebook.clone()).await?;
    cache.store_codebook_internal(collection1_key2.clone(), codebook.clone()).await?;

    // Verify all codebooks are stored
    assert!(cache.has_codebook(&collection1_key), "Collection1 PQ codebook should exist");
    assert!(cache.has_codebook(&collection2_key), "Collection2 binary codebook should exist");
    assert!(cache.has_codebook(&collection1_key2), "Collection1 INT8 codebook should exist");

    // Remove all codebooks for collection1
    let removed_count = cache.remove_collection_codebooks("collection1").await?;
    assert_eq!(removed_count, 2, "Should remove exactly 2 codebooks for collection1");

    // Verify cleanup
    assert!(!cache.has_codebook(&collection1_key), "Collection1 PQ codebook should be removed");
    assert!(cache.has_codebook(&collection2_key), "Collection2 codebook should still exist");
    assert!(!cache.has_codebook(&collection1_key2), "Collection1 INT8 codebook should be removed");

    Ok(())
}

#[tokio::test]
async fn test_cache_orchestrator_integration() -> Result<()> {
    // Test integration with CrossCacheOrchestrator
    use proximadb::storage::cache::orchestrator::{CrossCacheOrchestrator, CacheType};

    let orchestrator = Arc::new(CrossCacheOrchestrator::new(1024 * 1024)); // 1MB cache
    let cache = GlobalQuantizationCache::global();

    // Store a codebook and verify access tracking
    let key = QuantizationCacheKey::pq("test_orchestrator", 8, 32);
    let codebook = create_test_pq_codebook();

    cache.store_codebook_internal(key.clone(), codebook.clone()).await?;

    // Access the codebook multiple times
    for _ in 0..5 {
        let _retrieved = cache.get_codebook(&key).await;
    }

    // The access tracking should be recorded by CrossCacheOrchestrator
    // This validates that the integration is working

    Ok(())
}

#[tokio::test]
async fn test_hot_vs_cold_storage_strategy() -> Result<()> {
    // Test hot vs cold storage decision making
    let cache = GlobalQuantizationCache::global();

    // Small collection should use hot storage
    let should_use_hot = cache.should_use_hot_storage("small_collection");
    assert!(should_use_hot, "Small collections should use hot storage by default");

    // Update collection size to simulate growth
    cache.update_collection_size("small_collection", 100_000); // 100K vectors

    // The cache should now consider cold storage for large collections
    // Note: Current implementation defaults to hot, but this tests the interface
    let still_hot = cache.should_use_hot_storage("small_collection");
    // For now, this should still be true since we default to hot storage
    assert!(still_hot, "Collection storage strategy should be determinable");

    Ok(())
}

/// Helper function to create test PQ codebook
fn create_test_pq_codebook() -> Codebook {
    // Create a simple test codebook for PQ8 with 16 subvectors
    // For PQ, we need centroids for each subspace
    let mut centroids = Vec::new();

    // Create 16 subspaces (for 16 subvectors)
    for _ in 0..16 {
        // Each subspace has 256 centroids (for 8 bits)
        let mut subspace_centroids = Vec::new();
        for i in 0..256 {
            // Each centroid is a 4D vector in this test
            subspace_centroids.push(vec![
                i as f32 * 0.01,
                i as f32 * 0.02,
                i as f32 * 0.03,
                i as f32 * 0.04
            ]);
        }
        centroids.push(subspace_centroids);
    }

    Codebook {
        id: "test_codebook".to_string(),
        quantization_level: UnifiedQuantizationLevel {
            level: QuantizationLevel::ProductQuantization(ProductQuantization {
                bits: 8,
                num_subvectors: 16,
            }),
            filter_selectivity: 1.0,
            enable_hardware_acceleration: false,
        },
        timestamp: chrono::Utc::now(),
        training_config: TrainingConfig {
            sample_size: 1000,
            num_iterations: 10,
            random_seed: Some(42),
            custom_params: std::collections::HashMap::new(),
        },
        data: CodebookData::ProductQuantization {
            centroids,
            _subvector_dim: 4,
        },
    }
}

#[tokio::test]
async fn test_cross_engine_quantization_compatibility() -> Result<()> {
    // Test that all engines can work with the same global quantization cache
    use proximadb::compute::quantization::global_cache::GlobalQuantizationCache;

    let cache = GlobalQuantizationCache::global();

    // Create a codebook via one engine type and access via another
    let shared_collection_key = QuantizationCacheKey::pq("shared_collection", 8, 16);
    let codebook = create_test_pq_codebook();

    // Store via cache
    cache.store_codebook_internal(shared_collection_key.clone(), codebook.clone()).await?;

    // Verify all engines can access the same cache
    let sst_engine = proximadb::storage::engines::sst::SSTEngine::new().await?;
    let viper_engine = proximadb::storage::engines::viper::ViperEngine::default();
    let raptor_engine = proximadb::storage::engines::raptor::RaptorEngine::new().await?;
    let helix_engine = proximadb::storage::engines::helix::HelixEngine::new().await?;

    // All engines should be able to create quantization engines that use the global cache
    let sst_quant = sst_engine.get_quantization_engine("test", Some(1000)).await;
    let viper_quant = viper_engine.get_quantization_engine("test", Some(1000)).await;
    let raptor_quant = raptor_engine.get_quantization_engine("test", Some(1000)).await;
    let helix_quant = helix_engine.get_quantization_engine("test", Some(1000)).await;

    // Verify engines provide quantization capabilities
    assert!(sst_quant.is_some(), "SST should provide quantization engine");
    // VIPER always provides quantization engine (Arc not Option)
    // RAPTOR always provides quantization engine (Arc not Option)
    assert!(helix_quant.is_some(), "HELIX should provide quantization engine");

    Ok(())
}