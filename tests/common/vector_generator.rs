#![allow(dead_code, unused_imports, unused_variables)]
//! Test Vector Generation Utilities
//!
//! Provides various patterns for generating test vectors:
//! - Random vectors
//! - Sequential vectors
//! - Clustered vectors (for spatial algorithm testing)
//! - Vectors with metadata
//!
//! This eliminates duplicated vector generation code across 10+ test files.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use proximadb_data_model::ProximaValue;
use proximadb_records::{
    EmbeddingCell, EmbeddingValues, LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode,
};

/// Helper to get the current time in nanoseconds since Unix epoch.
fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}

/// Build a ProximaRecord from its essential fields.
fn make_record(id: &str, vector: Vec<f32>, props: ProximaTree) -> ProximaRecord {
    let ts = now_ns();
    let dim = vector.len() as u32;
    ProximaRecord {
        schema_version: proximadb_records::schema_version::default_schema_version(),
        oid: id.to_string(),
        local_id: None,
        tid: None,
        variation_id: None,
        record_version: 1,
        spec_version: 1,
        tenant_id: String::new(),
        permitted_principals: Vec::new(),
        rls_policy_id: None,
        created_at_ns: ts,
        updated_at_ns: ts,
        valid_from_ns: None,
        valid_to_ns: None,
        origin: None,
        actor: None,
        method: Some("test".to_string()),
        memory_type: None,
        props,
        refs: Vec::new(),
        edge: None,
        embeddings: if !vector.is_empty() {
            vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                dim,
                vector,
            )]
        } else {
            vec![]
        },
        sequence: None,
        labels: LabelSet::new(),
        branch_id: None,
    }
}

/// Generate random vectors with uniform distribution
///
/// # Examples
///
/// ```
/// use tests::common::vector_generator;
///
/// let vectors = vector_generator::random("my_collection", 1000, 128);
/// assert_eq!(vectors.len(), 1000);
/// assert_eq!(vectors[0].embeddings[0].values.len(), 128);
/// ```
pub fn random(collection_id: &str, count: usize, dimension: usize) -> Vec<ProximaRecord> {
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            make_record(&format!("vec_{}", i), vector, HashMap::new())
        })
        .collect()
}

/// Generate random vectors with a fixed seed (deterministic)
///
/// Useful for reproducible tests
pub fn random_seeded(
    collection_id: &str,
    count: usize,
    dimension: usize,
    seed: u64,
) -> Vec<ProximaRecord> {
    random_seeded_with_prefix("vec", count, dimension, seed)
}

/// Generate random vectors with a fixed seed and custom ID prefix
///
/// Useful when tests expect specific ID formats (e.g., "test_vec_0")
pub fn random_seeded_with_prefix(
    id_prefix: &str,
    count: usize,
    dimension: usize,
    seed: u64,
) -> Vec<ProximaRecord> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            make_record(&format!("{}_{}", id_prefix, i), vector, HashMap::new())
        })
        .collect()
}

/// Generate sequential vectors (useful for debugging)
///
/// Each dimension increases linearly: [0.0, 1.0, 2.0, ...]
pub fn sequential(collection_id: &str, count: usize, dimension: usize) -> Vec<ProximaRecord> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension).map(|d| (i * dimension + d) as f32).collect();
            make_record(&format!("vec_{}", i), vector, HashMap::new())
        })
        .collect()
}

/// Generate normalized random vectors (unit length)
///
/// Useful for cosine similarity testing
pub fn normalized(collection_id: &str, count: usize, dimension: usize) -> Vec<ProximaRecord> {
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

            // Normalize to unit length
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                vec.iter_mut().for_each(|x| *x /= magnitude);
            }

            make_record(&format!("vec_{}", i), vec, HashMap::new())
        })
        .collect()
}

/// Generate clustered vectors (for testing spatial algorithms like HELIX)
///
/// Creates `num_clusters` clusters with `count/num_clusters` vectors each.
/// Each cluster has a random center with vectors distributed around it.
pub fn clustered(
    collection_id: &str,
    count: usize,
    dimension: usize,
    num_clusters: usize,
) -> Vec<ProximaRecord> {
    let mut rng = rand::thread_rng();
    let per_cluster = count / num_clusters;

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();

    let mut vectors = Vec::new();

    for (cluster_idx, center) in centers.iter().enumerate() {
        for vec_idx in 0..per_cluster {
            let id = format!("vec_{}_{}", cluster_idx, vec_idx);

            // Generate vector near cluster center
            let vector: Vec<f32> = center
                .iter()
                .map(|&c| c + rng.gen_range(-1.0..1.0)) // Small deviation from center
                .collect();

            let mut props = HashMap::new();
            props.insert(
                "cluster".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(cluster_idx as i64)),
            );
            // (Int64 is correct for cluster index)

            vectors.push(make_record(&id, vector, props));
        }
    }

    vectors
}

/// Generate vectors with metadata
///
/// Provides a flexible way to create vectors with custom metadata
pub fn with_metadata<F>(
    collection_id: &str,
    count: usize,
    dimension: usize,
    metadata_fn: F,
) -> Vec<ProximaRecord>
where
    F: Fn(usize) -> ProximaTree,
{
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            make_record(&format!("vec_{}", i), vector, metadata_fn(i))
        })
        .collect()
}

/// Preset vector generators for common test scenarios
pub mod presets {
    use super::*;

    /// Generate vectors for filter testing (with various metadata types)
    pub fn for_filter_tests(
        collection_id: &str,
        count: usize,
        dimension: usize,
    ) -> Vec<ProximaRecord> {
        with_metadata(collection_id, count, dimension, |i| {
            let mut props = HashMap::new();

            // String metadata
            props.insert(
                "category".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("category_{}", i % 5))),
            );

            // Float metadata
            props.insert(
                "price".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float64((i as f64) * 10.0)),
            );

            // Boolean metadata
            props.insert(
                "in_stock".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(i % 2 == 0)),
            );

            // Integer metadata
            props.insert(
                "created_at".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(1600000000 + (i as i64))),
            );

            props
        })
    }

    /// Generate e-commerce product vectors
    pub fn ecommerce_products(
        collection_id: &str,
        count: usize,
        dimension: usize,
    ) -> Vec<ProximaRecord> {
        with_metadata(collection_id, count, dimension, |i| {
            let mut props = HashMap::new();
            let categories = ["electronics", "clothing", "home", "toys", "books"];
            let brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"];

            props.insert(
                "category".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(
                    categories[i % categories.len()].to_string(),
                )),
            );

            props.insert(
                "brand".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(brands[i % brands.len()].to_string())),
            );

            props.insert(
                "price".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float64(10.0 + (i % 1000) as f64 * 0.99)),
            );

            props.insert(
                "rating".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float64(1.0 + (i % 5) as f64 * 0.8)),
            );

            props.insert(
                "in_stock".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(i % 3 != 0)),
            );

            props
        })
    }

    /// Generate document vectors for RAG testing
    pub fn rag_documents(
        collection_id: &str,
        count: usize,
        dimension: usize,
    ) -> Vec<ProximaRecord> {
        with_metadata(collection_id, count, dimension, |i| {
            let mut props = HashMap::new();
            let doc_types = ["article", "paper", "blog", "documentation"];

            props.insert(
                "doc_type".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(
                    doc_types[i % doc_types.len()].to_string(),
                )),
            );

            props.insert(
                "word_count".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(100 + (i % 2000) as i64)),
            );

            props.insert(
                "published_date".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(1640000000 + (i as i64 * 86400))),
            );

            props.insert(
                "verified".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(i % 4 != 0)),
            );

            props
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaTreeNode;

    fn get_values(record: &ProximaRecord) -> &[f32] {
        record
            .embeddings
            .first()
            .map(|e| e.as_fp32_slice())
            .unwrap_or(&[])
    }

    #[test]
    fn test_random_generation() {
        let vectors = random("test_coll", 100, 128);
        assert_eq!(vectors.len(), 100);
        assert_eq!(get_values(&vectors[0]).len(), 128);
        assert!(!vectors[0].oid.is_empty());
    }

    #[test]
    fn test_random_seeded_deterministic() {
        let vectors1 = random_seeded("test_coll", 10, 64, 42);
        let vectors2 = random_seeded("test_coll", 10, 64, 42);

        // Same seed should produce identical vectors
        for (v1, v2) in vectors1.iter().zip(vectors2.iter()) {
            assert_eq!(get_values(v1), get_values(v2));
        }
    }

    #[test]
    fn test_sequential_generation() {
        let vectors = sequential("test_coll", 10, 8);
        assert_eq!(get_values(&vectors[0])[0], 0.0);
        assert_eq!(get_values(&vectors[0])[1], 1.0);
        assert_eq!(get_values(&vectors[1])[0], 8.0);
    }

    #[test]
    fn test_normalized_generation() {
        let vectors = normalized("test_coll", 10, 128);

        for rec in vectors {
            let vals = get_values(&rec);
            let magnitude: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (magnitude - 1.0).abs() < 0.001,
                "Vector not normalized: {}",
                magnitude
            );
        }
    }

    #[test]
    fn test_clustered_generation() {
        let vectors = clustered("test_coll", 100, 64, 5);
        assert_eq!(vectors.len(), 100);

        // Verify cluster metadata
        for (i, rec) in vectors.iter().enumerate() {
            assert!(
                rec.props.contains_key("cluster"),
                "record {} missing cluster prop",
                i
            );
            let cluster_id = match &rec.props["cluster"] {
                ProximaTreeNode::Value(ProximaValue::Int64(id)) => *id,
                _ => panic!("Invalid cluster metadata"),
            };
            assert_eq!(cluster_id, (i / 20) as i64);
        }
    }

    #[test]
    fn test_with_metadata() {
        use proximadb_records::ProximaTreeNode;
        let vectors = with_metadata("test_coll", 10, 64, |i| {
            let mut props = HashMap::new();
            props.insert(
                "index".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64(i as i64)),
            );
            props
        });

        for (i, rec) in vectors.iter().enumerate() {
            assert!(rec.props.contains_key("index"));
            let idx = match &rec.props["index"] {
                ProximaTreeNode::Value(ProximaValue::Int64(id)) => *id,
                _ => panic!("Invalid index metadata"),
            };
            assert_eq!(idx, i as i64);
        }
    }

    #[test]
    fn test_preset_filter_tests() {
        let vectors = presets::for_filter_tests("test_coll", 20, 128);

        for rec in vectors {
            assert!(rec.props.contains_key("category"));
            assert!(rec.props.contains_key("price"));
            assert!(rec.props.contains_key("in_stock"));
            assert!(rec.props.contains_key("created_at"));
        }
    }

    #[test]
    fn test_preset_ecommerce() {
        let vectors = presets::ecommerce_products("test_coll", 20, 128);

        for rec in vectors {
            assert!(rec.props.contains_key("category"));
            assert!(rec.props.contains_key("brand"));
            assert!(rec.props.contains_key("price"));
            assert!(rec.props.contains_key("rating"));
            assert!(rec.props.contains_key("in_stock"));
        }
    }
}
