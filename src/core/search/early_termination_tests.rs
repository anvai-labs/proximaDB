#[cfg(test)]
mod tests {
    use super::super::multi_tier_deduplication::{
        DataFreshnessTier, DeduplicationStorageEngine, MultiTierDeduplicator, TieredSearchCandidate,
    };

    use chrono::Utc;
    use proximadb_records::{EmbeddingCell, LabelSet, ProximaRecord, ProximaTree};

    fn make_record(oid: &str) -> ProximaRecord {
        let now_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        ProximaRecord {
            oid: oid.to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 1,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
            memory_type: None,
            props: ProximaTree::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }

    fn candidate(
        oid: &str,
        similarity: f32,
        tier: DataFreshnessTier,
        seq: u64,
        file_path: Option<String>,
    ) -> TieredSearchCandidate {
        TieredSearchCandidate {
            record: make_record(oid),
            tier,
            similarity,
            engine: DeduplicationStorageEngine::SST,
            timestamp: Utc::now(),
            sequence: seq,
            file_path,
        }
    }

    #[test]
    fn test_early_termination_disabled_with_ordering() {
        let k = 3;
        let mut deduplicator = MultiTierDeduplicator::with_k(k);
        deduplicator.set_requires_ordering(true);

        let candidates = vec![
            candidate("vec1", 0.9, DataFreshnessTier::Unflushed, 10, None),
            candidate("vec2", 0.8, DataFreshnessTier::Flushed, 8, None),
            candidate(
                "vec3",
                0.7,
                DataFreshnessTier::Compacted,
                5,
                Some("file1.sstable".to_string()),
            ),
            candidate(
                "vec4",
                0.6,
                DataFreshnessTier::Compacted,
                4,
                Some("file2.sstable".to_string()),
            ),
        ];

        deduplicator.add_tier_results(candidates);

        assert!(!deduplicator.is_early_terminated());

        let results = deduplicator.get_final_results(k);
        assert_eq!(results.len(), 3);

        assert_eq!(results[0].record.oid, "vec1");
        assert_eq!(results[1].record.oid, "vec2");
        assert_eq!(results[2].record.oid, "vec3");
    }

    #[test]
    fn test_early_termination_enabled_without_ordering() {
        let k = 3;
        let mut deduplicator = MultiTierDeduplicator::with_k(k);
        deduplicator.set_requires_ordering(false);

        let candidates = vec![
            candidate("vec1", 0.9, DataFreshnessTier::Unflushed, 10, None),
            candidate("vec2", 0.8, DataFreshnessTier::Flushed, 8, None),
            candidate(
                "vec3",
                0.7,
                DataFreshnessTier::Compacted,
                5,
                Some("file1.sstable".to_string()),
            ),
        ];

        deduplicator.add_tier_results(candidates);

        assert!(deduplicator.is_early_terminated());

        let results = deduplicator.get_final_results(k);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_early_termination_with_duplicates() {
        let k = 2;
        let mut deduplicator = MultiTierDeduplicator::with_k(k);
        deduplicator.set_requires_ordering(false);

        let candidates = vec![
            candidate("vec1", 0.9, DataFreshnessTier::Unflushed, 10, None),
            candidate("vec1", 0.85, DataFreshnessTier::Flushed, 8, None), // duplicate
            candidate(
                "vec2",
                0.7,
                DataFreshnessTier::Compacted,
                5,
                Some("file1.sstable".to_string()),
            ),
        ];

        deduplicator.add_tier_results(candidates);

        assert!(deduplicator.is_early_terminated());

        let results = deduplicator.get_final_results(k);
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].record.oid, "vec1");
        assert_eq!(results[0].similarity, 0.9);
        assert_eq!(results[1].record.oid, "vec2");
    }

    #[test]
    fn test_no_early_termination_when_insufficient_results() {
        let k = 5;
        let mut deduplicator = MultiTierDeduplicator::with_k(k);
        deduplicator.set_requires_ordering(false);

        let candidates = vec![
            candidate("vec1", 0.9, DataFreshnessTier::Unflushed, 10, None),
            candidate("vec2", 0.8, DataFreshnessTier::Flushed, 8, None),
        ];

        deduplicator.add_tier_results(candidates);

        assert!(!deduplicator.is_early_terminated());

        let results = deduplicator.get_final_results(k);
        assert_eq!(results.len(), 2);
    }
}
