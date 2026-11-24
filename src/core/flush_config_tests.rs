// Unit tests for flush and compaction configuration loading

#[cfg(test)]
mod tests {
    use super::super::config::*;

    #[test]
    fn test_wal_config_defaults() {
        let config = WriteBufferUserConfig::default();

        // Test size thresholds
        assert_eq!(config.write_buffer_size_mb, 8192); // 8GB global
        assert_eq!(config.memory_flush_size_bytes, 16 * 1024 * 1024); // 16MB per collection

        // Test other settings
        assert_eq!(config.memtable_type, "BTree");
        assert_eq!(config.sync_mode, "PerBatch");
        assert!(config.enable_wal);
    }

    #[test]
    fn test_sst_config_defaults() {
        let config = SstConfig::default();

        // Test compaction settings
        assert_eq!(config.compaction_threshold, 5); // Min 5 files before compaction
        assert_eq!(config.compaction_strategy, "leveled");
        assert_eq!(config.level_count, 7);
        assert_eq!(config.max_files_per_level, 10);

        // Test block size
        assert_eq!(config.block_size_kb, 256); // 256KB blocks (default for low-latency NVMe)
        assert_eq!(config.block_size_bytes(), 256 * 1024);

        // Test compression - server default is lz4 (faster than no compression)
        assert_eq!(config.compression, "lz4");
        assert_eq!(config.compression_level, 3);
    }

    #[test]
    fn test_wal_config_custom_values() {
        let config = WriteBufferUserConfig {
            write_buffer_size_mb: 16384,               // 16GB
            memory_flush_size_bytes: 32 * 1024 * 1024, // 32MB
            memtable_type: "SkipList".to_string(),
            sync_mode: "Periodic".to_string(),
            write_buffer_directory: "/custom/path".to_string(),
            enable_wal: false,
            global_manifest_url: None,
            wal_local_dir: None,
            // TD-SPINE-5 (ADR-098 D3): unset ⇒ inferred at boot.
            durability_mode: None,
            resolved_durability: None,
            // ADR-069 tiered-flush knobs.
            flush_interval_secs: 300,
            flush_floor_predicted_mb: 128,
            wal_max_bytes: 25 * 1024 * 1024 * 1024,
            high_watermark_pct: 0.75,
            critical_watermark_pct: 0.9,
        };

        assert_eq!(config.write_buffer_size_mb, 16384);
        assert_eq!(config.memory_flush_size_bytes, 32 * 1024 * 1024);
        assert_eq!(config.memtable_type, "SkipList");
        assert_eq!(config.sync_mode, "Periodic");
        assert!(!config.enable_wal);
        assert_eq!(config.flush_interval_secs, 300);
        assert_eq!(config.wal_max_bytes, 25 * 1024 * 1024 * 1024);
        assert_eq!(config.high_watermark_pct, 0.75);
        assert_eq!(config.critical_watermark_pct, 0.9);
    }

    #[test]
    fn test_wal_config_defaults_arm_time_floor() {
        // ADR-069 D2: the time floor defaults to 300s (RPO safety net) so the
        // auto-flush driver spawns by default and segments materialize while the
        // server runs. The inline size trigger handles throughput; wal_max_bytes
        // stays 0 so S4 write-shedding stays OFF.
        let config = WriteBufferUserConfig::default();
        assert_eq!(config.flush_interval_secs, 300);
        assert_eq!(config.wal_max_bytes, 0);
        assert_eq!(config.high_watermark_pct, 0.80);
        assert_eq!(config.critical_watermark_pct, 0.95);
    }

    #[test]
    fn test_flush_knobs_propagate_through_to_engine_config() {
        // The LIVE wiring: user TOML knobs must reach WalPerformanceConfig via
        // to_engine_config() (the database.rs path), and reconcile into a FlushPolicy
        // with the periodic scheduler armed.
        use crate::storage::persistence::write_ahead_log::flush_policy::FlushPolicy;

        let user = WriteBufferUserConfig {
            memory_flush_size_bytes: 8 * 1024 * 1024,
            flush_interval_secs: 120,
            wal_max_bytes: 1000,
            high_watermark_pct: 0.80,
            critical_watermark_pct: 0.95,
            ..Default::default()
        };
        let wal = user.to_engine_config();
        assert_eq!(wal.performance.memory_flush_size_bytes, 8 * 1024 * 1024);
        assert_eq!(wal.performance.flush_interval_secs, 120);
        assert_eq!(wal.performance.wal_max_bytes, 1000);
        assert_eq!(wal.performance.high_watermark_pct, 0.80);
        assert_eq!(wal.performance.critical_watermark_pct, 0.95);

        let policy = FlushPolicy::from_performance(&wal.performance);
        assert!(policy.needs_scheduler(), "time+capacity armed ⇒ scheduler");
        assert_eq!(policy.high_watermark_bytes, 800);
        assert_eq!(policy.critical_watermark_bytes, 950);
    }

    #[test]
    fn test_wal_config_values_propagate() {
        // Test that TOML config values would propagate correctly
        let toml_config = WriteBufferUserConfig {
            write_buffer_size_mb: 8192,
            memory_flush_size_bytes: 16777216, // 16MB
            memtable_type: "BTree".to_string(),
            sync_mode: "perbatch".to_string(),
            write_buffer_directory: "./test_data/write_buffer".to_string(),
            enable_wal: true,
            global_manifest_url: None,
            ..Default::default()
        };

        // Verify values are as expected
        assert_eq!(toml_config.memory_flush_size_bytes, 16777216);
        assert_eq!(
            toml_config.write_buffer_size_mb * 1024 * 1024,
            8192 * 1024 * 1024
        );
        assert!(toml_config.enable_wal);
        assert_eq!(
            toml_config.write_buffer_directory,
            "./test_data/write_buffer"
        );
    }

    #[test]
    fn test_sst_config_validation() {
        // Test valid config
        let valid_config = SstConfig {
            level_count: 7,
            compaction_threshold: 5,
            block_size_kb: 1024, // 1MB - new default
            decompression_cache_config: None,
            ..Default::default()
        };
        assert!(valid_config.validate().is_ok());

        // Test invalid level count
        let invalid_levels = SstConfig {
            level_count: 0,
            decompression_cache_config: None,
            ..Default::default()
        };
        assert!(invalid_levels.validate().is_err());

        // Test invalid compaction threshold
        let invalid_threshold = SstConfig {
            compaction_threshold: 0,
            decompression_cache_config: None,
            ..Default::default()
        };
        assert!(invalid_threshold.validate().is_err());

        // Test block size too small
        let small_blocks = SstConfig {
            block_size_kb: 128, // Less than 256KB - invalid
            decompression_cache_config: None,
            ..Default::default()
        };
        assert!(small_blocks.validate().is_err());

        // Test block size too large
        let large_blocks = SstConfig {
            block_size_kb: 5000, // More than 4MB - too large (new max)
            decompression_cache_config: None,
            ..Default::default()
        };
        assert!(large_blocks.validate().is_err());
    }

    #[test]
    fn test_flush_threshold_hierarchy() {
        // Test that we have proper hierarchy:
        // 1. Per-collection size: 16MB
        // 2. Per-collection count: 20k vectors
        // 3. Global size: 8GB
        // 4. Global count: 1M vectors (hardcoded in VectorOperationsService)

        let config = WriteBufferUserConfig::default();

        // Per-collection thresholds
        let per_collection_size_mb = config.memory_flush_size_bytes / (1024 * 1024);
        assert_eq!(per_collection_size_mb, 16);

        // Global thresholds
        let global_size_mb = config.write_buffer_size_mb;
        assert_eq!(global_size_mb, 8192); // 8GB

        // Verify hierarchy: global >> per-collection
        assert!(global_size_mb > (per_collection_size_mb * 100) as u64); // Can hold many collections
    }
}
