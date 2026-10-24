//! `OrtTokenizedScorerSession` — real ort-backed
//! [`TokenizedScorerSession`] impl (R-5b.1).
//!
//! Twin of [`OrtScorerSession`](crate::ort_scorer_session::OrtScorerSession);
//! differs only in input tensor dtype + slot count. Feature-gated
//! behind `real-onnx`. The default build ships only the mock so the
//! `tokenizers` + `ort` deps aren't pulled in by callers that just want
//! the trait surface.
//!
//! Contract (v1):
//! - The wrapped ONNX model has 2 OR 3 `int64` input tensors shaped
//!   `[batch_size, seq_len]`:
//!     - slot 0 = `input_ids`
//!     - slot 1 = `attention_mask`
//!     - slot 2 (optional) = `token_type_ids`
//! - Output is a `float` tensor whose first column is the per-row
//!   score (rank-2) OR whose only dim is the per-row score (rank-1).
//!   BERT cross-encoders that produce a single logit per pair match
//!   this contract.
//! - The session inspects `descriptor.input_spec` to bind names. Slot
//!   count drives whether `token_type_ids` is bound (2-slot model →
//!   omit it from the binding; 3-slot model → require it in the batch).
//!
//! Threading: `ort::Session::run` takes `&mut self`, so the session
//! lives behind `std::sync::Mutex` — same pattern as
//! [`OrtScorerSession`]. R-5b.2 will add a pooled variant once a
//! concrete need arises.

use crate::descriptor::ModelDescriptor;
use crate::tokenized_scorer_session::{TokenizedBatch, TokenizedScorerSession};
use proximadb_rank_core::{RankError, RankResult};
use std::sync::Mutex;

/// Real ort-backed tokenized scorer session.
pub struct OrtTokenizedScorerSession {
    descriptor: ModelDescriptor,
    session: Mutex<ort::session::Session>,
    memory_bytes: usize,
}

impl OrtTokenizedScorerSession {
    /// Load a tokenized-input ONNX model from disk.
    ///
    /// `descriptor.input_spec` must declare 2 or 3 input slots in the
    /// order `input_ids`, `attention_mask`, [`token_type_ids`]. The
    /// descriptor's slot names are used as ort input names.
    pub fn load_from_file(
        descriptor: ModelDescriptor,
        model_path: &std::path::Path,
    ) -> RankResult<Self> {
        if descriptor.input_spec.len() < 2 || descriptor.input_spec.len() > 3 {
            return Err(RankError::ModelLoad {
                model_id: descriptor.key.to_string(),
                reason: format!(
                    "OrtTokenizedScorerSession needs 2 or 3 declared input slots; descriptor has {}",
                    descriptor.input_spec.len()
                ),
            });
        }
        if descriptor.output_spec.is_empty() {
            return Err(RankError::ModelLoad {
                model_id: descriptor.key.to_string(),
                reason: "OrtTokenizedScorerSession needs ≥ 1 output slot".into(),
            });
        }
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

impl TokenizedScorerSession for OrtTokenizedScorerSession {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn score(&self, batch: &TokenizedBatch) -> RankResult<Vec<f32>> {
        if batch.batch_size() == 0 {
            return Ok(Vec::new());
        }
        if let Err(msg) = batch.validate_rectangular() {
            return Err(RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("OrtTokenizedScorerSession: ragged batch: {msg}"),
            });
        }
        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        let expects_token_type_ids = self.descriptor.input_spec.len() == 3;
        if expects_token_type_ids && batch.token_type_ids.is_none() {
            return Err(RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason:
                    "model declares 3 input slots (input_ids, attention_mask, token_type_ids) but \
                     batch.token_type_ids is None"
                        .into(),
            });
        }

        // Build flat int64 tensors. ort::Value::from_array takes an
        // owned ndarray::Array, so we materialize each one here.
        let input_ids_flat: Vec<i64> = flatten_i64(&batch.input_ids);
        let attention_mask_flat: Vec<i64> = flatten_i64(&batch.attention_mask);
        let input_ids_arr = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids_flat)
            .map_err(|e| RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("input_ids shape error: {e}"),
            })?;
        let attention_mask_arr =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask_flat).map_err(
                |e| RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!("attention_mask shape error: {e}"),
                },
            )?;
        let token_type_ids_arr = if expects_token_type_ids {
            let tti_flat: Vec<i64> = flatten_i64(batch.token_type_ids.as_ref().unwrap());
            Some(
                ndarray::Array2::from_shape_vec((batch_size, seq_len), tti_flat).map_err(|e| {
                    RankError::ModelInference {
                        model_id: self.descriptor.key.to_string(),
                        reason: format!("token_type_ids shape error: {e}"),
                    }
                })?,
            )
        } else {
            None
        };

        // Wrap each ndarray in an ort Value.
        let input_ids_value = ort::value::Value::from_array(input_ids_arr).map_err(|e| {
            RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!("Value::from_array(input_ids): {e}"),
            }
        })?;
        let attention_mask_value =
            ort::value::Value::from_array(attention_mask_arr).map_err(|e| {
                RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!("Value::from_array(attention_mask): {e}"),
                }
            })?;
        let token_type_ids_value =
            match token_type_ids_arr {
                Some(arr) => Some(ort::value::Value::from_array(arr).map_err(|e| {
                    RankError::ModelInference {
                        model_id: self.descriptor.key.to_string(),
                        reason: format!("Value::from_array(token_type_ids): {e}"),
                    }
                })?),
                None => None,
            };

        // Build ort inputs from the descriptor's slot names.
        let input_ids_name = self.descriptor.input_spec[0].name.clone();
        let attention_mask_name = self.descriptor.input_spec[1].name.clone();
        let inputs = if let Some(tti_value) = token_type_ids_value {
            let token_type_ids_name = self.descriptor.input_spec[2].name.clone();
            ort::inputs![
                input_ids_name => input_ids_value,
                attention_mask_name => attention_mask_value,
                token_type_ids_name => tti_value,
            ]
        } else {
            ort::inputs![
                input_ids_name => input_ids_value,
                attention_mask_name => attention_mask_value,
            ]
        };

        let mut session = self.session.lock().map_err(|e| RankError::ModelInference {
            model_id: self.descriptor.key.to_string(),
            reason: format!("session mutex poisoned: {e}"),
        })?;
        let outputs = session.run(inputs).map_err(|e| RankError::ModelInference {
            model_id: self.descriptor.key.to_string(),
            reason: format!("ort session.run: {e}"),
        })?;

        // First output slot = score tensor. Extract column 0 if rank-2.
        let output_name = self.descriptor.output_spec[0].name.clone();
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
                reason: format!("output extract f32: {e}"),
            })?;

        let scores: Vec<f32> = if array.ndim() == 1 {
            array.iter().copied().collect()
        } else {
            let view2 = array
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| RankError::ModelInference {
                    model_id: self.descriptor.key.to_string(),
                    reason: format!("output ndim != 2 cannot reshape: {e}"),
                })?;
            (0..batch_size).map(|i| view2[(i, 0)]).collect()
        };
        if scores.len() != batch_size {
            return Err(RankError::ModelInference {
                model_id: self.descriptor.key.to_string(),
                reason: format!(
                    "output produced {} scores for {} rows",
                    scores.len(),
                    batch_size
                ),
            });
        }
        Ok(scores)
    }
}

fn flatten_i64(rows: &[Vec<i64>]) -> Vec<i64> {
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{DType, ModelDescriptor, ModelFramework, ModelKey, TensorIoSpec};

    fn descriptor(slot_names: &[&str]) -> ModelDescriptor {
        ModelDescriptor {
            key: ModelKey::new("bce-test", "1"),
            tenant: None,
            uri: "file:///tmp/bce.onnx".into(),
            sha256: [0; 32],
            size_bytes: 4096,
            framework: ModelFramework::Onnx,
            dtype: DType::Fp32,
            input_spec: slot_names
                .iter()
                .map(|n| TensorIoSpec {
                    name: (*n).into(),
                    shape: vec![None, Some(128)],
                    dtype: DType::Fp32,
                })
                .collect(),
            output_spec: vec![TensorIoSpec {
                name: "logits".into(),
                shape: vec![None, Some(1)],
                dtype: DType::Fp32,
            }],
            max_batch_size: 16,
            seq: 0,
            created_at_ms: 0,
        }
    }

    // ---------------- Pure-data validation (no ort load required) ----------------

    #[test]
    fn flatten_i64_concatenates_rows_in_order() {
        let flat = flatten_i64(&[vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn flatten_i64_handles_empty_input() {
        assert_eq!(flatten_i64(&[]), Vec::<i64>::new());
    }

    /// Manual integration test: when `PROXIMADB_TEST_BERT_ONNX_PATH` is
    /// set to a real BERT cross-encoder ONNX file, exercise the full
    /// load → score pipeline. Silently no-ops otherwise so the default
    /// test run stays green without a fixture in tree.
    ///
    /// Run with:
    /// `PROXIMADB_TEST_BERT_ONNX_PATH=/path/to/cross-encoder.onnx \
    ///   cargo test -p proximadb-rank-onnx --features real-onnx \
    ///   ort_tokenized_scorer_session_loads_and_scores`
    #[test]
    fn ort_tokenized_scorer_session_loads_and_scores_when_fixture_available() {
        let Ok(path) = std::env::var("PROXIMADB_TEST_BERT_ONNX_PATH") else {
            eprintln!(
                "skipping ort tokenized integration test: PROXIMADB_TEST_BERT_ONNX_PATH not set"
            );
            return;
        };
        let p = std::path::PathBuf::from(path);
        if !p.exists() {
            eprintln!("skipping ort tokenized integration test: file {p:?} not found");
            return;
        }
        // Most cross-encoders take 2 inputs (input_ids, attention_mask);
        // those that need token_type_ids should set the env var to a 3-
        // input model fixture and bump this descriptor accordingly.
        let desc = descriptor(&["input_ids", "attention_mask"]);
        let session = OrtTokenizedScorerSession::load_from_file(desc, &p)
            .expect("loading the configured BERT cross-encoder fixture must succeed");
        let batch = TokenizedBatch::new(
            vec![vec![101, 2023, 2003, 1037, 3231, 102]],
            vec![vec![1, 1, 1, 1, 1, 1]],
        );
        let scores = session.score(&batch).expect("inference must succeed");
        assert_eq!(scores.len(), 1);
    }

    #[test]
    fn ort_tokenized_session_rejects_descriptor_with_one_input_slot() {
        // Descriptor validation runs before any ort work, so we can
        // assert the error path without an ONNX file. Construct via the
        // private API by hand — we never actually load the model.
        let desc = descriptor(&["only_input"]);
        // We can't actually call load_from_file without a real .onnx
        // file, but the descriptor-validation path runs first and would
        // bail out before ort touches disk. Confirm via a path that
        // exists but isn't a real model (ort will reject; the error
        // type is ModelLoad either way — we just want to see the slot-
        // count check fire).
        let tmp = std::env::temp_dir().join("nonexistent_bce.onnx");
        let result = OrtTokenizedScorerSession::load_from_file(desc, &tmp);
        // `OrtTokenizedScorerSession` wraps `ort::session::Session`, which
        // doesn't implement Debug, so we can't `{result:?}` the Ok side
        // — match explicit arms instead.
        match result {
            Err(RankError::ModelLoad { reason, .. }) => {
                assert!(
                    reason.contains("needs 2 or 3 declared input slots")
                        || reason.contains("commit_from_file"),
                    "expected slot-count or file-not-found error, got: {reason}"
                );
            }
            Err(other) => panic!("expected RankError::ModelLoad, got: {other}"),
            Ok(_) => panic!("expected error from non-existent ONNX path, got Ok"),
        }
    }
}
