//! Pure input-validation helpers extracted from `VectorOperationsService`
//! (Phase 2.1 god-object decomposition, slice 4).
//!
//! These are stateless validators over records / query vectors against a
//! collection's config (finite embeddings, dimension match, required/unique
//! ids). `VectorOperationsService` keeps a thin `&self` wrapper that resolves
//! the collection, then delegates the pure checks here.

use anyhow::Result;
use proximadb_records::{EmbeddingValues, ProximaRecord};

use crate::proto::proximadb_v1::{Collection, CollectionConfig};

/// Find the first non-finite value in an embedding (across precisions), if any,
/// as `(dimension_index, value_as_f32)`.
pub(crate) fn first_non_finite_embedding_value(values: &EmbeddingValues) -> Option<(usize, f32)> {
    match values {
        EmbeddingValues::Fp32(v) => v
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite()),
        EmbeddingValues::Fp16(v) => v
            .iter()
            .map(|value| value.to_f32())
            .enumerate()
            .find(|(_, value)| !value.is_finite()),
        EmbeddingValues::Bf16(v) => v
            .iter()
            .map(|value| value.to_f32())
            .enumerate()
            .find(|(_, value)| !value.is_finite()),
        EmbeddingValues::Int8Scalar { scale, .. } | EmbeddingValues::UInt8Scalar { scale, .. } => {
            if scale.is_finite() {
                None
            } else {
                Some((0, *scale))
            }
        }
    }
}

/// Validate a batch of records against the (already-resolved) collection config:
/// finite embeddings, dimension match, and — when the collection has indexes —
/// non-empty, bounded, batch-unique ids.
pub(crate) fn validate_records_for_insert(
    collection_id: &str,
    config: &CollectionConfig,
    records: &[ProximaRecord],
) -> Result<()> {
    let has_indexes = !config.index_configs.is_empty();
    let requires_id = has_indexes;
    let expected_dimension = config.dimension;

    if !requires_id && expected_dimension == 0 {
        return Ok(());
    }

    let mut seen_ids = if requires_id {
        Some(std::collections::HashSet::<&str>::with_capacity(
            records.len(),
        ))
    } else {
        None
    };

    for (i, record) in records.iter().enumerate() {
        let dim = record
            .embeddings
            .first()
            .map(|e| e.values.len())
            .unwrap_or(0);

        for (embedding_idx, embedding) in record.embeddings.iter().enumerate() {
            if let Some((dimension_idx, value)) =
                first_non_finite_embedding_value(&embedding.values)
            {
                return Err(anyhow::anyhow!(
                    "Record at index {} embedding {} contains non-finite value at dimension {}: {}",
                    i,
                    embedding_idx,
                    dimension_idx,
                    value
                ));
            }
        }

        // `dim == 0` ⇒ a vectorless record: a tombstone (delete) OR a metadata-only record
        // (e.g. an ADR-009 document facade record with no embedding). The engine stores these,
        // excludes them from the vector index and ANN search (they carry no vector to score),
        // and still serves them by id/scan — so they are valid in any collection. Only a
        // *non-zero* dimension that mismatches the collection is a real error (a likely
        // forgot-to-match-dims bug), which is still rejected.
        if dim != 0 && expected_dimension > 0 && dim != expected_dimension as usize {
            return Err(anyhow::anyhow!(
                "Record at index {} has dimension {} but collection '{}' expects dimension {}",
                i,
                dim,
                collection_id,
                expected_dimension
            ));
        }

        if let Some(ref mut seen) = seen_ids {
            if record.oid.is_empty() {
                return Err(anyhow::anyhow!(
                    "Record at index {} has empty ID. Collection '{}' requires valid IDs",
                    i,
                    collection_id
                ));
            }

            if record.oid.len() > 256 {
                return Err(anyhow::anyhow!(
                    "Record ID '{}' exceeds maximum length of 256 characters",
                    record.oid
                ));
            }

            if !seen.insert(record.oid.as_str()) {
                return Err(anyhow::anyhow!(
                    "Duplicate ID '{}' found in batch. All IDs must be unique",
                    record.oid
                ));
            }
        }
    }

    Ok(())
}

// The typed client-argument errors live in the foundation search-types crate
// so the API crate's handlers (which cannot see this crate) classify them the
// same way (TD-SECSCAN-1 §5).
pub use proximadb_search_types::client_argument::{
    InvalidQueryVector, QueryVectorProblem, client_argument_message, is_client_argument_error,
    label_client_collection,
};

/// The collection label a client-facing message may carry: ONLY the
/// identifier the caller supplied (`requested`), never anything resolved from
/// the catalog (TD-SECSCAN-1 §5).
///
/// The search authorities validate after the caller's identifier may have
/// been rewritten to the internal numeric object id; when `requested` is that
/// id the label is `None` ("the collection"), and the rewriting call site
/// attaches the caller's own identifier with [`label_client_collection`].
/// The catalog name is deliberately not used: an error message echoes only
/// what the request contained.
pub(crate) fn client_collection_label(requested: &str, collection: &Collection) -> Option<String> {
    (!requested.is_empty() && requested != collection.id).then(|| requested.to_string())
}

/// [`validate_query_vector_for_search`] plus the non-empty requirement of the
/// v1 search authority (`unified_search_v1_inner`).
pub(crate) fn validate_query_vector_presence_and_shape(
    collection_id: &str,
    collection: &Collection,
    query_vector: &[f32],
) -> Result<()> {
    if query_vector.is_empty() {
        return Err(InvalidQueryVector::new(
            client_collection_label(collection_id, collection),
            QueryVectorProblem::Empty,
        )
        .into());
    }
    validate_query_vector_for_search(collection_id, collection, query_vector)
}

/// Validate a query vector against the collection: finite values and a
/// dimension that matches the collection config (when configured).
pub(crate) fn validate_query_vector_for_search(
    collection_id: &str,
    collection: &Collection,
    query_vector: &[f32],
) -> Result<()> {
    if let Some((index, value)) = query_vector
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InvalidQueryVector::new(
            client_collection_label(collection_id, collection),
            QueryVectorProblem::NonFinite { index, value },
        )
        .into());
    }

    let expected_dimension = collection
        .config
        .as_ref()
        .map(|config| config.dimension)
        .unwrap_or_default();
    if expected_dimension > 0 && query_vector.len() != expected_dimension as usize {
        return Err(InvalidQueryVector::new(
            client_collection_label(collection_id, collection),
            QueryVectorProblem::Dimension {
                actual: query_vector.len(),
                expected: expected_dimension as usize,
            },
        )
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;

    fn dim8_config() -> CollectionConfig {
        CollectionConfig {
            dimension: 8,
            ..Default::default()
        }
    }

    fn record_no_vector(oid: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        }
    }

    fn record_with_dim(oid: &str, dim: usize) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            embeddings: vec![EmbeddingCell {
                dim: dim as u32,
                values: EmbeddingValues::Fp32(vec![0.1; dim]),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn metadata_only_record_passes_in_a_dimensioned_collection() {
        // ADR-009: a vectorless (dim-0) record — e.g. a metadata-only document facade record —
        // is accepted in a dimensioned collection; it simply doesn't participate in the ANN
        // index/search. Only a *non-zero* dimension mismatch is an error.
        let out = validate_records_for_insert("c", &dim8_config(), &[record_no_vector("m1")]);
        assert!(out.is_ok(), "metadata-only record should pass: {out:?}");
    }

    #[test]
    fn matching_dimension_record_passes() {
        let out = validate_records_for_insert("c", &dim8_config(), &[record_with_dim("v1", 8)]);
        assert!(out.is_ok(), "dim-8 record in a dim-8 collection: {out:?}");
    }

    #[test]
    fn wrong_nonzero_dimension_is_still_rejected() {
        // Regression guard: relaxing dim-0 must NOT relax a genuine (non-zero) dimension bug.
        let out = validate_records_for_insert("c", &dim8_config(), &[record_with_dim("v1", 2)]);
        assert!(
            out.is_err(),
            "a dim-2 record in a dim-8 collection must be rejected"
        );
    }

    /// A catalog-resolved collection: the search authorities validate AFTER
    /// resolving the client's name to the internal numeric object id.
    fn resolved_collection(object_id: &str, name: &str) -> Collection {
        Collection {
            id: object_id.to_string(),
            config: Some(CollectionConfig {
                name: name.to_string(),
                dimension: 4,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// TD-SECSCAN-1 §5 residual: a client-facing message echoes the
    /// identifier the caller supplied — here its own collection name, which an
    /// authority validating BEFORE the name→id rewrite still holds.
    #[test]
    fn dimension_error_echoes_the_callers_own_identifier() {
        let collection = resolved_collection("16", "orders");
        for vector in [vec![1.0f32; 3], vec![f32::NAN; 4]] {
            let err = validate_query_vector_for_search("orders", &collection, &vector)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(err.contains("'orders'"), "must name 'orders': {err}");
            assert!(!err.contains("'16'"), "must not echo the object id: {err}");
        }
    }

    /// A request that addressed the collection by object id gets a message
    /// built only from what it sent: never the catalog name (which the request
    /// did not contain), and not the object id either.
    #[test]
    fn object_id_reference_never_echoes_the_catalog_name() {
        let collection = resolved_collection("17", "catalog_only_name");
        for vector in [vec![1.0f32; 3], vec![f32::INFINITY; 4], Vec::new()] {
            let err = validate_query_vector_presence_and_shape("17", &collection, &vector)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(!err.is_empty(), "{vector:?} must be rejected");
            assert!(
                !err.contains("catalog_only_name"),
                "echoed a name the request did not contain: {err}"
            );
            assert!(!err.contains("'17'"), "must not echo the object id: {err}");
            assert!(err.contains("the collection"), "{err}");
        }
    }

    #[test]
    fn empty_vector_is_rejected_by_the_presence_check_only() {
        let collection = resolved_collection("16", "orders");
        let err = validate_query_vector_presence_and_shape("orders", &collection, &[])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert_eq!(err, "Query vector for collection 'orders' is empty");
    }
}
