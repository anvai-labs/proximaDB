// Tests for StorageConfig with new storage locations

#[cfg(test)]
mod tests {
    use crate::core::config::{
        AssignmentConfig, BloomFilterConfig, OptimizationConfig, StorageConfig, StorageLocation,
        WriteBufferUserConfig,
    };

    #[test]
    fn test_storage_locations_config() {
        let config = StorageConfig {
            storage_locations: vec![
                StorageLocation {
                    url: "file:///nvme1/proximadb".to_string(),
                    weight: 1,
                    tags: vec!["fast".to_string(), "local".to_string()],
                    io_budget: None,
                },
                StorageLocation {
                    url: "s3://my-bucket/proximadb".to_string(),
                    weight: 2,
                    tags: vec!["cloud".to_string(), "archive".to_string()],
                    io_budget: None,
                },
            ],
            metadata_url: "file:///nvme1/proximadb/metadata_info".to_string(),
            assignment_config: AssignmentConfig::default(),
            mmap_enabled: true,
            sst_config: Default::default(),
            viper_config: Default::default(),
            wal_config: Default::default(),
            cache_size_mb: 2048,
            bloom_filter_config: Some(BloomFilterConfig {
                bits_per_key: 12,
                enabled: true,
                ..Default::default()
            }),
            filesystem_config: Default::default(),
            compaction_config: Default::default(),
            prune_mode: None,
            optimization: OptimizationConfig::default(),
            reservation_pools: Vec::new(),
        };

        let storage_urls = config.storage_urls();
        assert_eq!(storage_urls.len(), 2);
        assert_eq!(storage_urls[0], "file:///nvme1/proximadb");
        assert_eq!(storage_urls[1], "s3://my-bucket/proximadb");

        assert_eq!(config.metadata_url, "file:///nvme1/proximadb/metadata_info");
    }

    #[test]
    fn test_url_derivation() {
        let config = StorageConfig {
            storage_locations: vec![
                StorageLocation {
                    url: "file:///nvme1/proximadb".to_string(),
                    weight: 1,
                    tags: vec![],
                    io_budget: None,
                },
                StorageLocation {
                    url: "s3://bucket/proximadb/".to_string(), // With trailing slash
                    weight: 1,
                    tags: vec![],
                    io_budget: None,
                },
            ],
            metadata_url: "file:///fast-ssd/metadata_info".to_string(),
            viper_config: Default::default(),
            wal_config: Default::default(),
            ..Default::default()
        };

        let write_buffer_urls = config.write_buffer_urls();
        assert_eq!(write_buffer_urls.len(), 2);
        assert_eq!(write_buffer_urls[0], "file:///nvme1/proximadb/wal");
        assert_eq!(write_buffer_urls[1], "s3://bucket/proximadb/wal"); // Trailing slash handled

        let data_urls = config.data_urls();
        assert_eq!(data_urls.len(), 2);
        assert_eq!(data_urls[0], "file:///nvme1/proximadb/data");
        assert_eq!(data_urls[1], "s3://bucket/proximadb/data");

        let index_urls = config.index_urls();
        assert_eq!(index_urls.len(), 2);
        assert_eq!(index_urls[0], "file:///nvme1/proximadb/index");
        assert_eq!(index_urls[1], "s3://bucket/proximadb/index");
    }

    #[test]
    fn test_heterogeneous_storage() {
        let config = StorageConfig {
            storage_locations: vec![
                StorageLocation {
                    url: "file:///local/proximadb".to_string(),
                    weight: 1,
                    tags: vec!["local".to_string()],
                    io_budget: None,
                },
                StorageLocation {
                    url: "s3://aws-bucket/proximadb".to_string(),
                    weight: 2,
                    tags: vec!["cloud".to_string(), "aws".to_string()],
                    io_budget: None,
                },
                StorageLocation {
                    url: "gs://gcp-bucket/proximadb".to_string(),
                    weight: 2,
                    tags: vec!["cloud".to_string(), "gcp".to_string()],
                    io_budget: None,
                },
                StorageLocation {
                    url: "adls://azure-account.dfs.core.windows.net/container/proximadb"
                        .to_string(),
                    weight: 2,
                    tags: vec!["cloud".to_string(), "azure".to_string()],
                    io_budget: None,
                },
            ],
            metadata_url: "file:///fast-ssd/metadata_info".to_string(),
            viper_config: Default::default(),
            wal_config: Default::default(),
            ..Default::default()
        };

        let urls = config.storage_urls();
        assert_eq!(urls.len(), 4);
        assert!(urls[0].starts_with("file://"));
        assert!(urls[1].starts_with("s3://"));
        assert!(urls[2].starts_with("gs://"));
        assert!(urls[3].starts_with("adls://"));

        // WAL URLs should be derived correctly for each
        let write_buffer_urls = config.write_buffer_urls();
        assert_eq!(write_buffer_urls.len(), 4);
        assert_eq!(write_buffer_urls[0], "file:///local/proximadb/wal");
        assert_eq!(write_buffer_urls[1], "s3://aws-bucket/proximadb/wal");
        assert_eq!(write_buffer_urls[2], "gs://gcp-bucket/proximadb/wal");
        assert_eq!(
            write_buffer_urls[3],
            "adls://azure-account.dfs.core.windows.net/container/proximadb/wal"
        );
    }

    #[test]
    fn test_assignment_config() {
        let config = StorageConfig {
            storage_locations: vec![StorageLocation {
                url: "file:///disk1".to_string(),
                weight: 1,
                tags: vec![],
                io_budget: None,
            }],
            metadata_url: "file:///disk1/metadata_info".to_string(),
            assignment_config: AssignmentConfig {
                strategy: "hash".to_string(),
                affinity: true,
            },
            viper_config: Default::default(),
            wal_config: Default::default(),
            ..Default::default()
        };

        assert!(config.assignment_config.affinity);
    }

    #[test]
    fn test_default_storage_config() {
        let config = StorageConfig::default();

        // Should have default storage locations
        assert!(!config.storage_locations.is_empty());

        // Should have proper metadata URL
        assert!(!config.metadata_url.is_empty());
        assert!(config.metadata_url.starts_with("file://"));

        // Assignment config should default to hash with affinity
        assert!(config.assignment_config.affinity);
    }

    #[test]
    fn test_wal_config_values() {
        // Test with custom values that should be used instead of defaults
        let wal_config = WriteBufferUserConfig {
            write_buffer_size_mb: 8192,        // 8GB
            memory_flush_size_bytes: 16777216, // 16MB
            memtable_type: "BTree".to_string(),
            sync_mode: "PerBatch".to_string(),
            write_buffer_directory: "./test_wal".to_string(),
            enable_wal: true,
            global_manifest_url: None,
            ..Default::default()
        };

        // Verify the values are set correctly
        assert_eq!(wal_config.write_buffer_size_mb, 8192);
        assert_eq!(wal_config.memory_flush_size_bytes, 16777216); // 16MB not 2MB!
        assert_eq!(wal_config.memtable_type, "BTree");
        assert_eq!(wal_config.sync_mode, "PerBatch");
        assert_eq!(wal_config.write_buffer_directory, "./test_wal");
        assert!(wal_config.enable_wal);
    }

    #[test]
    fn test_wal_config_from_toml() {
        // Test loading from TOML string
        let toml_str = r#"
            write_buffer_size_mb = 4096
            memory_flush_size_bytes = 33554432  # 32MB
            memtable_type = "SkipList"
            sync_mode = "Periodic"
            write_buffer_directory = "/tmp/wal"
            enable_wal = false
        "#;

        let wal_config: WriteBufferUserConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(wal_config.write_buffer_size_mb, 4096);
        assert_eq!(wal_config.memory_flush_size_bytes, 33554432); // 32MB
        assert_eq!(wal_config.memtable_type, "SkipList");
        assert_eq!(wal_config.sync_mode, "Periodic");
        assert_eq!(wal_config.write_buffer_directory, "/tmp/wal");
        assert!(!wal_config.enable_wal);
    }

    // --- Tests inlined from tests/unit/core/config_tests.rs ---

    #[test]
    fn test_default_config() {
        use crate::core::config::Config;

        let config = Config::default();

        // Test default server config
        assert_eq!(config.server.node_id, "node-1");
        assert_eq!(config.server.bind_address, "127.0.0.1");
        assert_eq!(config.server.port, 5678);

        // Test default storage config
        assert!(!config.storage.storage_locations.is_empty());
        assert!(config.storage.metadata_url.contains("metadata"));
        assert_eq!(config.storage.cache_size_mb, 512);
        assert!(config.storage.mmap_enabled);

        // Test default SST config - sst_config is Option<SstConfig>
        if let Some(ref sst_config) = config.storage.sst_config {
            assert_eq!(sst_config.level_count, 7);
            assert_eq!(sst_config.compaction_threshold, 5); // Default is 5, not 3
        }

        // Test default API config
        assert_eq!(config.api.rest_port, 5678);
        assert_eq!(config.api.grpc_port, 5679);
        assert_eq!(config.api.max_request_size_mb, 100);
        assert_eq!(config.api.timeout_seconds, 60); // Default is 60 seconds
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        use crate::core::config::Config;

        // Test that default config can be serialized and deserialized
        let original = Config::default();

        // Serialize to TOML
        let toml_str = toml::to_string(&original).expect("Failed to serialize config");

        // Deserialize back
        let recovered: Config = toml::from_str(&toml_str).expect("Failed to deserialize config");

        // Verify key values match
        assert_eq!(original.server.node_id, recovered.server.node_id);
        assert_eq!(original.server.bind_address, recovered.server.bind_address);
        assert_eq!(original.server.port, recovered.server.port);
        assert_eq!(original.api.rest_port, recovered.api.rest_port);
        assert_eq!(original.api.grpc_port, recovered.api.grpc_port);
        assert_eq!(
            original.storage.cache_size_mb,
            recovered.storage.cache_size_mb
        );
    }
}

// ---- TD-IOBUDGET-1: per-location I/O budget resolution, validation, registration ----

#[cfg(test)]
mod io_budget_tests {
    use crate::core::config::{StorageConfig, StorageLocation};
    use proximadb_config::{DiskClass, IoBudgetConfig};
    use proximadb_storage_common::iops_budget::IopsBudget;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn disk_class_selects_the_documented_profile() {
        let cfg = |dc: Option<DiskClass>| IoBudgetConfig {
            disk_class: dc,
            ..Default::default()
        };
        // ssd → LOCAL profile; hdd → CLOUD profile (ADR-073).
        assert_eq!(
            StorageConfig::resolve_location_io_budget(
                "file:///mnt/nvme",
                &cfg(Some(DiskClass::Ssd))
            )
            .unwrap(),
            IopsBudget::LOCAL
        );
        assert_eq!(
            StorageConfig::resolve_location_io_budget(
                "file:///mnt/sas",
                &cfg(Some(DiskClass::Hdd))
            )
            .unwrap(),
            IopsBudget::CLOUD
        );
        // cloud is scheme-aware: file:// gets CLOUD, s3:// keeps its own S3 profile.
        assert_eq!(
            StorageConfig::resolve_location_io_budget(
                "file:///mnt/sas",
                &cfg(Some(DiskClass::Cloud))
            )
            .unwrap(),
            IopsBudget::CLOUD
        );
        assert_eq!(
            StorageConfig::resolve_location_io_budget("s3://bucket", &cfg(Some(DiskClass::Cloud)))
                .unwrap(),
            IopsBudget::S3
        );
        assert_eq!(
            StorageConfig::resolve_location_io_budget(
                "az://container",
                &cfg(Some(DiskClass::Cloud))
            )
            .unwrap(),
            IopsBudget::AZURE
        );
    }

    #[test]
    fn explicit_bytes_override_the_profile_per_field() {
        // S3 profile {512K, 8M, 16M} with target lowered to 4 MiB.
        let cfg = IoBudgetConfig {
            disk_class: None,
            min_bytes: None,
            target_bytes: Some(4 * MIB),
            max_bytes: None,
        };
        assert_eq!(
            StorageConfig::resolve_location_io_budget("s3://bucket", &cfg).unwrap(),
            IopsBudget {
                min: 512 * 1024,
                target: 4 * MIB,
                max: 16 * MIB
            }
        );
        // The measured TD-SEARCH-3 opt-in: 16 MiB target on S3.
        let cfg16 = IoBudgetConfig {
            target_bytes: Some(16 * MIB),
            max_bytes: Some(16 * MIB),
            disk_class: None,
            min_bytes: None,
        };
        assert_eq!(
            StorageConfig::resolve_location_io_budget("s3://bucket", &cfg16).unwrap(),
            IopsBudget {
                min: 512 * 1024,
                target: 16 * MIB,
                max: 16 * MIB
            }
        );
    }

    #[test]
    fn resolve_is_fail_closed_on_bad_bounds() {
        // min below the 64 KiB floor.
        let tiny_min = IoBudgetConfig {
            min_bytes: Some(1024),
            target_bytes: Some(MIB),
            max_bytes: Some(2 * MIB),
            disk_class: None,
        };
        let err = StorageConfig::resolve_location_io_budget("s3://bucket", &tiny_min).unwrap_err();
        assert!(err.contains("floor"), "{err}");
        // min > target.
        let inverted = IoBudgetConfig {
            min_bytes: Some(4 * MIB),
            target_bytes: Some(2 * MIB),
            max_bytes: Some(8 * MIB),
            disk_class: None,
        };
        let err = StorageConfig::resolve_location_io_budget("s3://bucket", &inverted).unwrap_err();
        assert!(err.contains("min ≤ target ≤ max"), "{err}");
        // A per-field override that breaks the profile's own invariant
        // (CLOUD target raised past CLOUD max without raising max) is rejected.
        let target_past_max = IoBudgetConfig {
            disk_class: Some(DiskClass::Hdd),
            target_bytes: Some(16 * MIB),
            min_bytes: None,
            max_bytes: None,
        };
        assert!(StorageConfig::resolve_location_io_budget("file:///x", &target_past_max).is_err());
    }

    #[test]
    fn toml_parse_fails_closed_on_typos() {
        // Valid parse.
        let cfg: IoBudgetConfig =
            toml::from_str("disk_class = 'hdd'\ntarget_bytes = 4194304").unwrap();
        assert_eq!(cfg.disk_class, Some(DiskClass::Hdd));
        assert_eq!(cfg.target_bytes, Some(4 * MIB));
        // Unknown disk_class value → LOAD error, not a silent default.
        assert!(toml::from_str::<IoBudgetConfig>("disk_class = 'nvme'").is_err());
        // Unknown KEY (the deny_unknown_fields typo guard) → LOAD error.
        assert!(toml::from_str::<IoBudgetConfig>("target_byets = 4194304").is_err());
        // StorageLocation deserializes with (and without) the io_budget table.
        #[derive(serde::Deserialize)]
        struct Locations {
            locs: Vec<StorageLocation>,
        }
        let parsed: Locations = toml::from_str(
            r#"
            [[locs]]
            url = "file:///mnt/sas"
            weight = 1
            tags = ["durable"]
            [locs.io_budget]
            disk_class = "hdd"
            max_bytes = 16777216

            [[locs]]
            url = "s3://bucket"
            weight = 2
            tags = []
        "#,
        )
        .unwrap();
        assert_eq!(parsed.locs.len(), 2);
        assert_eq!(
            parsed.locs[0].io_budget.as_ref().unwrap().disk_class,
            Some(DiskClass::Hdd)
        );
        assert!(parsed.locs[1].io_budget.is_none());
    }

    #[test]
    fn registration_is_fail_closed_and_seeds_the_leaf() {
        // A location whose budget violates the bounds → Err naming the URL.
        let bad = StorageConfig {
            storage_locations: vec![StorageLocation {
                url: "file:///iobudget-root-test-bad".to_string(),
                weight: 1,
                tags: vec![],
                io_budget: Some(IoBudgetConfig {
                    min_bytes: Some(8 * MIB),
                    target_bytes: Some(4 * MIB),
                    max_bytes: Some(16 * MIB),
                    disk_class: None,
                }),
            }],
            ..StorageConfig::default()
        };
        let err = bad.register_io_budgets().unwrap_err().to_string();
        assert!(err.contains("iobudget-root-test-bad"), "{err}");
        // Nothing was registered for the rejected location.
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-root-test-bad/seg.pax"),
            IopsBudget::LOCAL
        );

        // A valid config registers per-location budgets the leaf honors, and
        // an unconfigured location keeps its scheme default.
        let good = StorageConfig {
            storage_locations: vec![
                StorageLocation {
                    url: "file:///iobudget-root-test-hdd".to_string(),
                    weight: 1,
                    tags: vec![],
                    io_budget: Some(IoBudgetConfig {
                        disk_class: Some(DiskClass::Hdd),
                        max_bytes: Some(4 * MIB),
                        min_bytes: None,
                        target_bytes: None,
                    }),
                },
                StorageLocation {
                    url: "s3://iobudget-root-test-bucket".to_string(),
                    weight: 1,
                    tags: vec![],
                    io_budget: Some(IoBudgetConfig {
                        target_bytes: Some(16 * MIB),
                        max_bytes: Some(16 * MIB),
                        disk_class: None,
                        min_bytes: None,
                    }),
                },
            ],
            ..StorageConfig::default()
        };
        good.register_io_budgets().unwrap();
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-root-test-hdd/coll/seg.pax"),
            IopsBudget {
                min: 512 * 1024,
                target: 4 * MIB,
                max: 4 * MIB
            }
        );
        assert_eq!(
            IopsBudget::for_path("s3://iobudget-root-test-bucket/coll/seg.pax"),
            IopsBudget {
                min: 512 * 1024,
                target: 16 * MIB,
                max: 16 * MIB
            }
        );
    }
}

// ---- TD-IOBUDGET-1 review findings: precedence pinning, round-trip, duplicate-URL conflict, reseed ----

#[cfg(test)]
mod io_budget_review_tests {
    use crate::core::config::{StorageConfig, StorageLocation};
    use proximadb_config::{DiskClass, IoBudgetConfig};
    use proximadb_storage_common::iops_budget::IopsBudget;

    const MIB: u64 = 1024 * 1024;

    /// Finding F1 (pinned semantics): the env var supplies the device CLASS
    /// wherever a budget leaves `disk_class` unset — a partial TOML budget
    /// (bytes only) refines the env-selected profile rather than suppressing
    /// it; an explicit `disk_class` beats the env.
    #[test]
    fn env_disk_class_supplies_class_for_partial_budget() {
        struct EnvGuard;
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("PROXIMADB_DISK_CLASS") };
            }
        }
        unsafe { std::env::set_var("PROXIMADB_DISK_CLASS", "hdd") };
        let _guard = EnvGuard;

        let partial = IoBudgetConfig {
            disk_class: None,
            max_bytes: Some(4 * MIB),
            min_bytes: None,
            target_bytes: None,
        };
        assert_eq!(
            StorageConfig::resolve_location_io_budget("file:///mnt/sas", &partial).unwrap(),
            IopsBudget {
                min: 512 * 1024,
                target: 4 * MIB,
                max: 4 * MIB
            }
        );

        let explicit = IoBudgetConfig {
            disk_class: Some(DiskClass::Ssd),
            max_bytes: Some(4 * MIB),
            min_bytes: None,
            target_bytes: None,
        };
        assert_eq!(
            StorageConfig::resolve_location_io_budget("file:///mnt/sas", &explicit).unwrap(),
            IopsBudget {
                min: 256 * 1024,
                target: MIB,
                max: 4 * MIB
            }
        );
    }

    /// Finding F3: serialization round-trips both shapes, and a location
    /// WITHOUT a budget must not emit an `io_budget` key at all
    /// (skip_serializing_if guards the serialized byte-shape).
    #[test]
    fn storage_location_serialization_round_trips() {
        let with = StorageLocation {
            url: "s3://iobudget-roundtrip".to_string(),
            weight: 2,
            tags: vec!["cloud".to_string()],
            io_budget: Some(IoBudgetConfig {
                disk_class: Some(DiskClass::Cloud),
                target_bytes: Some(16 * MIB),
                ..Default::default()
            }),
        };
        let serialized = toml::to_string(&with).unwrap();
        let parsed: StorageLocation = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, with);

        let without = StorageLocation {
            url: "file:///iobudget-roundtrip".to_string(),
            weight: 1,
            tags: vec![],
            io_budget: None,
        };
        let serialized = toml::to_string(&without).unwrap();
        assert!(!serialized.contains("io_budget"), "{serialized}");
        let parsed: StorageLocation = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, without);
    }

    /// Finding F4: the registry is URL-keyed, so two entries for the same URL
    /// with DIFFERENT resolved budgets would silently let the last one win —
    /// that must die at startup, naming the URL.
    #[test]
    fn duplicate_url_with_conflicting_budgets_fails_closed() {
        let config = StorageConfig {
            storage_locations: vec![
                StorageLocation {
                    url: "file:///iobudget-root-test-dup".to_string(),
                    weight: 3,
                    tags: vec![],
                    io_budget: Some(IoBudgetConfig {
                        disk_class: Some(DiskClass::Ssd),
                        ..Default::default()
                    }),
                },
                StorageLocation {
                    url: "file:///iobudget-root-test-dup".to_string(),
                    weight: 1,
                    tags: vec![],
                    io_budget: Some(IoBudgetConfig {
                        disk_class: Some(DiskClass::Hdd),
                        ..Default::default()
                    }),
                },
            ],
            ..StorageConfig::default()
        };
        let err = config.register_io_budgets().unwrap_err().to_string();
        assert!(err.contains("iobudget-root-test-dup"), "{err}");
        // Nothing won: the conflicting URL never registers.
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-root-test-dup/seg.pax"),
            IopsBudget::LOCAL
        );
    }

    /// Finding N8: seeding is authoritative-REPLACE — a location dropped from
    /// a later config must not keep serving its stale budget to an in-process
    /// re-boot (embedded re-init, test suites).
    #[test]
    fn dropped_location_budget_is_cleared_on_reseed() {
        let with_budget = StorageConfig {
            storage_locations: vec![StorageLocation {
                url: "file:///iobudget-root-test-drop".to_string(),
                weight: 1,
                tags: vec![],
                io_budget: Some(IoBudgetConfig {
                    disk_class: Some(DiskClass::Hdd),
                    ..Default::default()
                }),
            }],
            ..StorageConfig::default()
        };
        with_budget.register_io_budgets().unwrap();
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-root-test-drop/seg.pax"),
            IopsBudget::CLOUD
        );

        let without_budget = StorageConfig {
            storage_locations: vec![StorageLocation {
                url: "file:///iobudget-root-test-other".to_string(),
                weight: 1,
                tags: vec![],
                io_budget: None,
            }],
            ..StorageConfig::default()
        };
        without_budget.register_io_budgets().unwrap();
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-root-test-drop/seg.pax"),
            IopsBudget::LOCAL
        );
    }
}
