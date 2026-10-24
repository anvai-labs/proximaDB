//! `OrtScorerSession` — real ort-backed [`ScorerSession`] impl (R-5b).
//!
//! Feature-gated behind `real-onnx`. The default build keeps the
//! [`MockScorerSession`](crate::scorer_session::MockScorerSession)
//! primitive only; production deployments rebuild with
//! `--features real-onnx` to swap in live ONNX inference.
//!
//! Contract (v1):
//! - The wrapped ONNX model must accept a **float** input tensor shaped
//!   `[batch_size, input_width]` and produce a **float** output tensor
//!   whose first column is the per-row score. Bi-encoder rerankers
//!   that take pre-encoded features match this contract.
//! - Tokenized cross-encoders that take `int64` token-id tensors need a
//!   separate trait surface (R-5b.1: `TokenizedScorerSession`). They
//!   don't fit here cleanly because the trait signature is float-only.
//!
//! Threading: `ort::Session::run` takes `&mut self`, so the session
//! lives behind `std::sync::Mutex`. Concurrent queries serialize
//! through the mutex; deployments needing higher throughput can pool
//! N sessions (mirroring the proximadb-embedding pattern) — R-5b.2.

use crate::descriptor::ModelDescriptor;
use crate::scorer_session::ScorerSession;
use proximadb_rank_core::{RankError, RankResult};
use std::sync::Mutex;

/// Real ort-backed scorer session. Wraps an `ort::Session` in a mutex
/// (because Session::run requires &mut self) and exposes the
/// `ScorerSession` trait.
pub struct OrtScorerSession {
    descriptor: ModelDescriptor,
    session: Mutex<ort::session::Session>,
    /// Estimated resident memory — set from descriptor.size_bytes at
    /// construction; LRU eviction reads it via `memory_bytes()`.
    memory_bytes: usize,
}

impl OrtScorerSession {
    /// Load an ONNX model from disk.
    ///
    /// `model_path` must point to an `.onnx` file (or a directory
    /// ort understands). `descriptor.size_bytes` is consulted for the
    /// memory-budget estimate exposed via `ScorerSession::memory_bytes`.
    ///
    /// Errors: maps ort load failures to
    /// [`RankError::ModelLoad`](proximadb_rank_core::RankError::ModelLoad)
    /// with the path + descriptor key in the message.
    pub fn load_from_file(
        descriptor: ModelDescriptor,
        model_path: &std::path::Path,
    ) -> RankResult<Self> {
        // TD-ORT-1: one environment per process, owned by
        // proximadb-onnx-runtime and shared with the embedding pool.
        let session = ort::session::Session::builder(
            proximadb_onnx_runtime::shared_environment().map_err(|e| RankError::ModelLoad {
                model_id: descriptor.key.to_string(),
                reason: format!("ort environment: {e}"),
            })?,
        )
        .map_err(|e| RankError::ModelLoad {
            model_id: descriptor.key.to_string(),
            reason: format!("ort builder: {e}"),
        })?
        .commit_from_file(model_path)
        .map_err(|e| RankError::ModelLoad {
            model_id: descriptor.key.to_string(),
            reason: format!("ort commit_from_file({}): {e}", model_path.display()),
        })?;
        let memory_bytes = descriptor.estimated_memory_bytes();
        Ok(Self {
            descriptor,
            session: Mutex::new(session),
            memory_bytes,
        })
    }
}

impl ScorerSession for OrtScorerSession {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn score(&self, rows: &[Vec<f32>]) -> RankResult<Vec<f32>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        // All rows must share the same width — the ndarray batched
        // tensor below is rectangular.
        let width = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!(
                        "row {i} has width {} but row 0 has width {}; OrtScorerSession requires rectangular input",
                        row.len(),
                        width
                    ),
                });
            }
        }

        // Build a flat batch tensor [batch, width].
        let batch = rows.len();
        let mut flat: Vec<f32> = Vec::with_capacity(batch * width);
        for row in rows {
            flat.extend_from_slice(row);
        }
        let input = ndarray::Array2::from_shape_vec((batch, width), flat).map_err(|e| {
            RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("ndarray shape error: {e}"),
            }
        })?;

        // The input tensor name comes from the descriptor's input_spec.
        // v1 contract: the first declared input slot receives the batch
        // tensor. Models with multiple input slots need
        // R-5b.1's TokenizedScorerSession variant.
        let input_name = self
            .descriptor
            .input_spec
            .first()
            .map(|s| s.name.clone())
            .ok_or_else(|| RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: "descriptor.input_spec is empty — OrtScorerSession needs at least one named input".into(),
            })?;

        // Wrap the ndarray in an ort Value (matches the
        // proximadb-embedding pattern). Value::from_array takes
        // ownership of the array so we don't have a borrow-vs-Value
        // lifetime mismatch.
        let input_value =
            ort::value::Value::from_array(input).map_err(|e| RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("ort Value::from_array: {e}"),
            })?;
        let inputs = ort::inputs![input_name => input_value];
        let mut session = self.session.lock().map_err(|e| RankError::ModelInference {
            model_id: self.descriptor.key.to_string(),
            reason: format!("session mutex poisoned: {e}"),
        })?;
        let outputs = session.run(inputs).map_err(|e| RankError::ModelInference {
            model_id: self.descriptor.key.to_string(),
            reason: format!("ort session.run: {e}"),
        })?;

        // First output slot is the score tensor. Extract column 0 as
        // the per-row score (matches the bi-encoder reranker contract).
        let output_name = self
            .descriptor
            .output_spec
            .first()
            .map(|s| s.name.clone())
            .ok_or_else(|| RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: "descriptor.output_spec is empty — OrtScorerSession needs at least one named output".into(),
            })?;
        let tensor =
            outputs
                .get(output_name.as_str())
                .ok_or_else(|| RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!("output slot {output_name:?} missing from session.run result"),
                })?;
        let array = tensor
            .try_extract_array::<f32>()
            .map_err(|e| RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("output tensor extract f32: {e}"),
            })?;

        // Output shape: [batch] (1D) or [batch, N] (take col 0).
        let scores = if array.ndim() == 1 {
            array.iter().copied().collect::<Vec<f32>>()
        } else {
            let view2 = array
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!("output ndim != 2 cannot reshape: {e}"),
                })?;
            (0..batch).map(|i| view2[(i, 0)]).collect()
        };
        if scores.len() != batch {
            return Err(RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("output produced {} scores for {} rows", scores.len(), batch),
            });
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{DType, ModelDescriptor, ModelFramework, ModelKey, TensorIoSpec};

    fn descriptor_with_named_io(model: &str) -> ModelDescriptor {
        ModelDescriptor {
            key: ModelKey::new(model, "test"),
            tenant: None,
            uri: format!("file:///tmp/{model}.onnx"),
            sha256: [0; 32],
            size_bytes: 1024,
            framework: ModelFramework::Onnx,
            dtype: DType::Fp32,
            input_spec: vec![TensorIoSpec {
                name: "input".into(),
                shape: vec![None, Some(4)],
                dtype: DType::Fp32,
            }],
            output_spec: vec![TensorIoSpec {
                name: "score".into(),
                shape: vec![None, Some(1)],
                dtype: DType::Fp32,
            }],
            max_batch_size: 32,
            seq: 0,
            created_at_ms: 0,
        }
    }

    #[test]
    fn descriptor_default_max_batch_size_when_omitted() {
        // Pure-data test that compiles without ort doing any work.
        let desc = descriptor_with_named_io("x");
        assert_eq!(desc.max_batch_size, 32);
        assert_eq!(desc.input_spec.len(), 1);
        assert_eq!(desc.output_spec.len(), 1);
    }

    /// Integration test: when `PROXIMADB_TEST_ONNX_PATH` is set to a
    /// real ONNX file path, exercise the full load → score pipeline.
    /// This is a manual / CI fixture — the test silently no-ops
    /// otherwise so the default test run stays green without an ONNX
    /// fixture in tree.
    ///
    /// To run locally:
    /// `PROXIMADB_TEST_ONNX_PATH=/path/to/model.onnx \
    ///   cargo test -p proximadb-rank-onnx --features real-onnx \
    ///   ort_scorer_session_loads_and_scores`
    #[test]
    fn ort_scorer_session_loads_and_scores_when_fixture_available() {
        let Ok(path) = std::env::var("PROXIMADB_TEST_ONNX_PATH") else {
            eprintln!("skipping ort integration test: PROXIMADB_TEST_ONNX_PATH not set");
            return;
        };
        let p = std::path::PathBuf::from(path);
        if !p.exists() {
            eprintln!("skipping ort integration test: file {p:?} not found");
            return;
        }
        let desc = descriptor_with_named_io("fixture");
        let session = OrtScorerSession::load_from_file(desc, &p)
            .expect("loading the configured ONNX fixture must succeed");
        // The fixture must declare input named "input" + output named
        // "score" matching descriptor_with_named_io. Two-row batch
        // exercises the rectangular-input + per-row-score path.
        let scores = session
            .score(&[vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]])
            .expect("scoring two rows must succeed against the fixture");
        assert_eq!(scores.len(), 2);
    }
}
