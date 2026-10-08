//! # Analytics REST Handlers
//!
//! Endpoints for analytical computations (Entanglement Index, etc.).
//! All handlers are stateless pure-math operations or stub `NotImplemented`
//! responses — no root-crate concrete type dependencies.

use std::collections::HashMap;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use std::sync::Arc;

use proximadb_records::{EmbeddingValues, tree_get};
use proximadb_runtime::RecordOpsPort;
use serde::{Deserialize, Serialize};

use crate::rest::TenantContext;
use axum::extract::Extension;
use tracing::debug;

use crate::rest::errors::{RestError, RestResult};

// ── State (stateless for now) ─────────────────────────────────────────────────

/// Axum state for analytics endpoints.
///
/// Stateless for the `compute_entanglement` endpoint; collection-level EI
/// requires a `VectorOpsPort` which is not yet wired in — that endpoint
/// returns `NotImplemented`.
#[derive(Clone)]
pub struct AnalyticsRestState {
    /// Record scan authority — the collection-EI handler reads the tenant's
    /// records through the port (ADR-094).
    pub record_ops: Arc<dyn RecordOpsPort>,
}

// ── Legacy stub types kept for re-export compatibility ────────────────────────

/// Analytics handler stub.
pub struct AnalyticsHandler;

impl AnalyticsHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AnalyticsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// AQL handler stub.
pub struct AqlHandler;

impl AqlHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AqlHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

/// One chunk in an Entanglement Index request.
#[derive(Debug, Deserialize)]
pub struct ChunkInput {
    pub chunk_id: String,
    pub topic: String,
    pub embedding: Vec<f32>,
}

/// Request body for `POST /api/v1/analytics/entanglement`.
#[derive(Debug, Deserialize)]
pub struct EntanglementRequest {
    pub chunks: Vec<ChunkInput>,
}

/// Query params for collection-level EI.
#[derive(Debug, Deserialize)]
pub struct CollectionEiParams {
    /// Field in record metadata to use as the topic label.
    pub topic_field: String,
    /// Maximum number of records to analyze (default: 1000).
    pub limit: Option<usize>,
}

/// Entanglement Index response.
#[derive(Debug, Serialize)]
pub struct EntanglementResponse {
    pub overall_ei: f64,
    pub per_topic_ei: HashMap<String, f64>,
    pub chunks_analyzed: usize,
    pub topics_analyzed: usize,
    pub skipped_singletons: usize,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn create_analytics_router() -> Router<AnalyticsRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            .route("/entanglement", post(compute_entanglement))
            .route(
                "/collections/{collection_id}/entanglement",
                get(get_collection_entanglement),
            ),
    )
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn compute_entanglement(
    State(_): State<AnalyticsRestState>,
    Json(request): Json<EntanglementRequest>,
) -> RestResult<Json<EntanglementResponse>> {
    debug!("EI request with {} chunks", request.chunks.len());

    if request.chunks.is_empty() {
        return Ok(Json(EntanglementResponse {
            overall_ei: 0.0,
            per_topic_ei: HashMap::new(),
            chunks_analyzed: 0,
            topics_analyzed: 0,
            skipped_singletons: 0,
        }));
    }

    let report = entanglement_index(&request.chunks)
        .map_err(|e| RestError::InvalidArgument(e.to_string()))?;
    Ok(Json(report))
}

/// Collection-level EI: scan the tenant's records through the record-ops port.
async fn get_collection_entanglement(
    State(state): State<AnalyticsRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(collection_id): Path<String>,
    Query(params): Query<CollectionEiParams>,
) -> RestResult<Json<EntanglementResponse>> {
    let limit = params.limit.unwrap_or(1000);
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let (records, _cursor) = state
        .record_ops
        .handle_record_scan_paginated_for_tenant(
            &collection_id,
            None,
            limit,
            true,  // include_vector - EI needs the embeddings
            true,  // include_props  - EI needs the topic field
            Some(&tenant.tenant_id),
            None,
            now_ns,
        )
        .await
        .map_err(|e| RestError::Internal(format!("record scan failed: {e}")))?;

    let chunks: Vec<ChunkInput> = records
        .into_iter()
        .filter_map(|r| {
            // Topic: a string/symbol property under the caller-chosen field.
            let topic = match tree_get(&r.props, &params.topic_field) {
                Some(proximadb_data_model::ProximaValue::String(s))
                | Some(proximadb_data_model::ProximaValue::Symbol(s)) => s.clone(),
                _ => return None,
            };
            // Embedding: the first fp32-representable cell set. Quantized cells
            // are skipped (EI needs full precision to be meaningful).
            let embedding = r.embeddings.iter().find_map(|cell| match &cell.values {
                EmbeddingValues::Fp32(v) => Some(v.clone()),
                _ => None,
            })?;
            let chunk_id = if r.oid.is_empty() {
                r.local_id.unwrap_or_default()
            } else {
                r.oid
            };
            Some(ChunkInput { chunk_id, topic, embedding })
        })
        .collect();

    if chunks.is_empty() {
        return Err(RestError::InvalidArgument(format!(
            "No records in collection '{}' have a string field '{}'",
            collection_id, params.topic_field
        )));
    }

    let report = entanglement_index(&chunks)
        .map_err(|e| RestError::InvalidArgument(e.to_string()))?;
    Ok(Json(report))
}

// ── Entanglement Index — inline pure-math implementation ─────────────────────
//
// Mirrors the algorithm in `src/analytics/entanglement.rs` without the
// `UnifiedDistanceCompute` dependency so this crate stays root-free.

fn entanglement_index(chunks: &[ChunkInput]) -> Result<EntanglementResponse, String> {
    if chunks.is_empty() {
        return Ok(EntanglementResponse {
            overall_ei: 0.0,
            per_topic_ei: HashMap::new(),
            chunks_analyzed: 0,
            topics_analyzed: 0,
            skipped_singletons: 0,
        });
    }

    // Validate consistent dimension.
    let dim = chunks[0].embedding.len();
    for c in chunks {
        if c.embedding.len() != dim {
            return Err(format!(
                "Chunk '{}' has dimension {} but expected {}",
                c.chunk_id,
                c.embedding.len(),
                dim
            ));
        }
        if l2_norm(&c.embedding) == 0.0 {
            return Err(format!("Chunk '{}' has a zero-norm embedding", c.chunk_id));
        }
    }

    // L2-normalize all embeddings.
    let norms: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| {
            let n = l2_norm(&c.embedding);
            c.embedding.iter().map(|x| x / n).collect()
        })
        .collect();

    // Build topic → indices map.
    let mut topic_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        topic_indices.entry(&c.topic).or_default().push(i);
    }

    // Compute entangled(x) for each non-singleton chunk.
    let eps = 1e-8_f64;
    let mut entangled_vals: Vec<f64> = Vec::new();
    let mut per_topic_sums: HashMap<&str, (f64, usize)> = HashMap::new();
    let mut skipped: usize = 0;

    for (i, c) in chunks.iter().enumerate() {
        let same_topic = &topic_indices[c.topic.as_str()];
        if same_topic.len() < 2 {
            skipped += 1;
            continue;
        }

        // Mean intra-topic cosine (excluding self).
        let intra: f64 = same_topic
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| dot(&norms[i], &norms[j]) as f64)
            .sum::<f64>()
            / (same_topic.len() - 1) as f64;

        // Mean inter-topic cosine.
        let inter_count = chunks.len() - same_topic.len();
        let inter: f64 = if inter_count == 0 {
            0.0
        } else {
            chunks
                .iter()
                .enumerate()
                .filter(|(j, ch)| ch.topic != c.topic && *j != i)
                .map(|(j, _)| dot(&norms[i], &norms[j]) as f64)
                .sum::<f64>()
                / inter_count as f64
        };

        let ev = (inter / intra.max(eps)).clamp(0.0, 1.0);
        entangled_vals.push(ev);

        let entry = per_topic_sums.entry(&c.topic).or_insert((0.0, 0));
        entry.0 += ev;
        entry.1 += 1;
    }

    let chunks_analyzed = entangled_vals.len();
    let overall_ei = if chunks_analyzed == 0 {
        0.0
    } else {
        entangled_vals.iter().sum::<f64>() / chunks_analyzed as f64
    };

    let per_topic_ei: HashMap<String, f64> = per_topic_sums
        .into_iter()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(topic, (sum, n))| (topic.to_string(), sum / n as f64))
        .collect();

    let topics_analyzed = per_topic_ei.len();

    Ok(EntanglementResponse {
        overall_ei,
        per_topic_ei,
        chunks_analyzed,
        topics_analyzed,
        skipped_singletons: skipped,
    })
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_ei() {
        let r = entanglement_index(&[]).unwrap();
        assert_eq!(r.overall_ei, 0.0);
        assert_eq!(r.chunks_analyzed, 0);
    }

    #[test]
    fn test_singleton_skipped() {
        let chunks = vec![
            ChunkInput {
                chunk_id: "a".into(),
                topic: "A".into(),
                embedding: vec![1.0, 0.0],
            },
            ChunkInput {
                chunk_id: "b".into(),
                topic: "B".into(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let r = entanglement_index(&chunks).unwrap();
        // Both are singletons → 0 analyzed, EI = 0
        assert_eq!(r.chunks_analyzed, 0);
        assert_eq!(r.skipped_singletons, 2);
    }

    #[test]
    fn test_perfect_separation() {
        // Two topics, orthogonal embeddings within each topic
        let chunks = vec![
            ChunkInput {
                chunk_id: "a1".into(),
                topic: "A".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            ChunkInput {
                chunk_id: "a2".into(),
                topic: "A".into(),
                embedding: vec![0.9, 0.1, 0.0],
            },
            ChunkInput {
                chunk_id: "b1".into(),
                topic: "B".into(),
                embedding: vec![0.0, 0.0, 1.0],
            },
            ChunkInput {
                chunk_id: "b2".into(),
                topic: "B".into(),
                embedding: vec![0.0, 0.1, 0.9],
            },
        ];
        let r = entanglement_index(&chunks).unwrap();
        // A and B are near-orthogonal → low EI
        assert!(r.overall_ei < 0.3, "EI={} but expected < 0.3", r.overall_ei);
        assert_eq!(r.chunks_analyzed, 4);
    }
}
