//! Async-aware benchmarks for VIPER engine Parquet operations
//!
//! This version properly handles async initialization without runtime conflicts

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proximadb::{
    compute::distance_computation::{DistanceMetric, engine::UnifiedDistanceCompute},
    core::{config::ViperConfig, hardware_capabilities},
    proto::proximadb_v1::{
        Collection, CollectionConfig, QuantizationConfig, StorageAssignment, StorageEngine,
        VectorRecord,
    },
    storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory},
    storage::{
        engines::viper::engine::ViperEngine,
        traits::{FlushParameters, UnifiedStorageEngine},
    },
};
use std::collections::HashMap;
use std::sync::{Arc, Once};

/// Global initialization for hardware capabilities
static INIT: Once = Once::new();

/// Initialize hardware capabilities once for all benchmarks
fn init_hardware() {
    INIT.call_once(|| {
        let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    });
}

/// Create quantization config with specified modes
fn create_quantization_config(
    enable_binary: bool,
    enable_int8: bool,
    enable_pq: bool,
    pq_segments: u32,
    pq_bits: u32,
) -> QuantizationConfig {
    QuantizationConfig {
        enabled: Some(enable_binary || enable_int8 || enable_pq),
        strategy: if enable_binary && enable_int8 && enable_pq {
            Some(3) // MEMORY_OPTIMIZED
        } else if enable_binary {
            Some(2) // SPEED_OPTIMIZED
        } else if enable_int8 && !enable_pq {
            Some(1) // ACCURACY_OPTIMIZED
        } else {
            Some(0) // SMART_DEFAULTS
        },
        custom_levels: vec![],
        enable_progressive_search: Some(enable_binary || enable_int8 || enable_pq),
        binary_filter_selectivity: Some(0.3),
        int8_ranking_selectivity: Some(0.1),
        pq_ranking_selectivity: Some(0.05),
        training_sample_size: Some(10000),
        quality_threshold: Some(0.95),
        enable_adaptive_training: Some(true),
        optimize_for_storage: Some(enable_pq),
        optimize_for_memory: Some(enable_binary && enable_pq),
        enable_simd_acceleration: Some(true),
        enable_binary: Some(enable_binary),
        enable_int8: Some(enable_int8),
        enable_pq: Some(enable_pq),
        pq_segments: Some(pq_segments),
        pq_bits: Some(pq_bits),
        pq_codebooks: Some(256),
        binary_threshold: Some(0.5),
        int8_threshold: Some(0.3),
        pq_threshold: Some(0.1),
        enable_turboquant: None,
    }
}

/// Generate test vectors with proto format
fn generate_vectors(count: usize, dimension: usize) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| {
            let vector = (0..dimension)
                .map(|j| ((i + j) as f32 * 0.001) % 1.0)
                .collect();

            VectorRecord {
                id: format!("vec_{:08}", i),
                vector,
                metadata: HashMap::new(),
                timestamp: Some(i as i64),
                updated_at: Some(chrono::Utc::now().timestamp()),
                expires_at: None,
                version: Some(1),
                source: None,
            }
        })
        .collect()
}

/// Create VIPER engine using async constructor
async fn create_viper_engine() -> Arc<dyn UnifiedStorageEngine> {
    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(
        FilesystemFactory::create(filesystem_config)
            .await
            .expect("Failed to create filesystem"),
    );
    let viper_config = ViperConfig::default();
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());

    let engine = ViperEngine::new()
        .await
        .expect("Failed to create VIPER engine");

    Arc::new(engine) as Arc<dyn UnifiedStorageEngine>
}

/// Benchmark VIPER engine flush operations with different quantization modes
fn bench_viper_flush(c: &mut Criterion) {
    init_hardware();

    // Create runtime once for all benchmarks
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    // Test configurations
    let configs = vec![
        (
            "no_quantization",
            create_quantization_config(false, false, false, 8, 8),
        ),
        (
            "binary_only",
            create_quantization_config(true, false, false, 8, 8),
        ),
        (
            "int8_only",
            create_quantization_config(false, true, false, 8, 8),
        ),
        (
            "pq_only",
            create_quantization_config(false, false, true, 32, 8),
        ),
        (
            "all_modes",
            create_quantization_config(true, true, true, 32, 8),
        ),
    ];

    for (config_name, quant_config) in configs {
        let mut group = c.benchmark_group(format!("viper_flush_{}", config_name));

        for size in [1024, 5120, 10240].iter() {
            let vectors = generate_vectors(*size, 768);

            group.bench_with_input(BenchmarkId::new("flush", size), size, |b, _| {
                // Pre-create the engine outside the benchmark loop
                let engine = runtime.block_on(create_viper_engine());

                b.iter(|| {
                    runtime.block_on(async {
                        let collection = Collection {
                            // SC1 (#2259) added the field; the literal
                            // initializer here predates it and broke every
                            // --all-targets check on develop (LIGHT CI does
                            // not compile benches). None = NOT_ACTIVE truth.
                            storage_class_status: None,
                            id: "bench-viper".to_string(),
                            config: Some(CollectionConfig {
                                name: "bench-viper".to_string(),
                                dimension: 768,
                                distance_metric: Some(DistanceMetric::Euclidean as i32),
                                storage_engine: Some(StorageEngine::Viper as i32),
                                quantization: Some(quant_config.clone()),
                                ..Default::default()
                            }),
                            created_at: 0,
                            updated_at: 0,
                            stats: None,
                            storage_assignment: Some(StorageAssignment {
                                primary_path: "/tmp/proximadb-bench/viper".to_string(),
                                backup_paths: vec![],
                                engine: StorageEngine::Viper as i32,
                                engine_config: HashMap::new(),
                                base_location: "/tmp/proximadb-bench/viper".to_string(),
                                assigned_at: 0,
                                ..Default::default()
                            }),
                        };

                        let params = FlushParameters {
                            collection_id: Some("bench-viper".to_string()),
                            vector_records: vectors
                                .iter()
                                .cloned()
                                .map(proximadb::proto::defaults::vector_record_to_proxima_record)
                                .collect(),
                            force: true,
                            synchronous: true,
                            collection_config: Some(collection),
                            ..Default::default()
                        };

                        let _ = engine.flush(params).await;
                    })
                });
            });
        }

        group.finish();
    }
}

// Configure with consistent settings across all benchmarks
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(40)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = bench_viper_flush
}
criterion_main!(benches);
