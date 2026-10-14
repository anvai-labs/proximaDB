//! `BertPairTokenizingDocFeatureExtractor` — production
//! `TokenizedDocFeatureExtractor` for BERT-family cross-encoders.
//! R-5b.1.2.
//!
//! Pulls `(query_text, doc_text)` pairs and produces a
//! [`TokenizedBatch`] via `tokenizers::Tokenizer::encode_batch`.
//! Closes the last gap between R-5b.1.1's pipeline wiring and a real
//! BERT cross-encoder running end-to-end through REST / gRPC / Arrow
//! Flight.
//!
//! Feature-gated behind `bert-tokenizer` so the default
//! `proximadb-rank-onnx` build stays light. Deployments that opt into
//! cross-encoder reranking rebuild with `--features bert-tokenizer`
//! (typically also `real-onnx`).
//!
//! Per-request query plumbing (R-5b.1.3):
//! Query text flows in via the `QueryContext` parameter on
//! [`TokenizedDocFeatureExtractor::extract_batch`]. The extractor is
//! stateless — one shared `Arc<…>` instance handles concurrent
//! requests, each carrying its own `qctx.query_text`. The earlier
//! `Arc<RwLock<Arc<str>>>` interior-mutability hack from R-5b.1.2 is
//! gone; the one-rerank-at-a-time deployment caveat it carried is
//! resolved.
//!
//! Spec: roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md §4.4.

use std::collections::HashMap;
use std::sync::Arc;

use proximadb_rank_core::{DocHandle, QueryContext, RankError, RankResult};

use crate::tokenized_doc_feature_extractor::TokenizedDocFeatureExtractor;
use crate::tokenized_scorer_session::TokenizedBatch;

// ---------------------------------------------------------------------------
// DocTextSource — pluggable interface for fetching doc text by handle
// ---------------------------------------------------------------------------

/// Resolve a `DocHandle` to its source text. The text source is the
/// boundary between the rank framework and the attribute / storage
/// layer that owns the actual doc payload. Production wiring will
/// plug an attribute-store reader; tests use the in-memory HashMap
/// impl below.
///
/// Returns `None` when the doc has no text (e.g. a numeric-id-only
/// handle from a hybrid retrieval result the framework can't resolve).
/// The extractor rejects missing text: scoring an empty surrogate would
/// silently turn an incomplete hydration path into a ranking decision.
pub trait DocTextSource: Send + Sync {
    fn doc_text(&self, doc: DocHandle) -> RankResult<Option<String>>;
}

/// In-memory text source. The most common test fixture; also useful
/// as a small-cache wrapper for production paths that batch-load doc
/// text into a process-local map.
pub struct HashMapDocTextSource {
    texts: HashMap<DocHandle, String>,
}

impl HashMapDocTextSource {
    pub fn new() -> Self {
        Self {
            texts: HashMap::new(),
        }
    }

    pub fn with_entries<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = (DocHandle, S)>,
        S: Into<String>,
    {
        let mut s = Self::new();
        for (d, t) in entries {
            s.insert(d, t);
        }
        s
    }

    pub fn insert(&mut self, doc: DocHandle, text: impl Into<String>) -> &mut Self {
        self.texts.insert(doc, text.into());
        self
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

impl Default for HashMapDocTextSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DocTextSource for HashMapDocTextSource {
    fn doc_text(&self, doc: DocHandle) -> RankResult<Option<String>> {
        Ok(self.texts.get(&doc).cloned())
    }
}

// ---------------------------------------------------------------------------
// BertPairTokenizingDocFeatureExtractor
// ---------------------------------------------------------------------------

/// Tokenized doc-feature extractor that runs `tokenizer.encode_batch`
/// over `(query_text, doc_text)` pairs to produce a `TokenizedBatch`
/// suitable for a BERT cross-encoder.
///
/// Construction:
/// - `tokenizer`: shared `Arc<tokenizers::Tokenizer>` (same instance
///   embedding crate uses; one tokenizer file per deployment).
/// - `doc_text_source`: resolves doc text by handle.
/// - `max_seq_len`: hard cap on tokenized sequence length. Padding is
///   batch-longest so the resulting batch is rectangular (a hard
///   requirement for the ONNX session) without paying `max_seq_len`
///   compute per row. Tokenizer overflow is detected and rejected
///   rather than silently truncated.
/// - `emit_token_type_ids`: when true, produce `token_type_ids`
///   (segment ids) — required by BERT-base/MiniLM-L-12-v2 style
///   models; some MiniLM-derived models don't take them.
///
/// The extractor is stateless across requests — one shared instance
/// serves concurrent reranks. Per-request query text flows in via
/// `extract_batch`'s `QueryContext` argument (R-5b.1.3).
pub struct BertPairTokenizingDocFeatureExtractor {
    tokenizer: Arc<tokenizers::Tokenizer>,
    doc_text_source: Arc<dyn DocTextSource>,
    max_seq_len: usize,
    emit_token_type_ids: bool,
    /// The tokenizer's real pad id (BERT 0, XLM-R 1, …). Used by the conversion
    /// layer when padding rows to the batch width.
    pad_id: i64,
}

impl BertPairTokenizingDocFeatureExtractor {
    /// Construct from a `tokenizer.json` path. Production deployments (and
    /// out-of-crate callers that do not depend on `tokenizers` directly) load
    /// from disk; this keeps the tokenizer type inside the crate.
    #[cfg(feature = "bert-tokenizer")]
    pub fn from_tokenizer_file(
        tokenizer_path: &std::path::Path,
        doc_text_source: Arc<dyn DocTextSource>,
        max_seq_len: usize,
        emit_token_type_ids: bool,
    ) -> RankResult<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| RankError::ModelLoad {
                model_id: "bert_pair_extractor".into(),
                reason: format!("tokenizer load {}: {e}", tokenizer_path.display()),
            })?;
        Ok(Self::new(
            Arc::new(tokenizer),
            doc_text_source,
            max_seq_len,
            emit_token_type_ids,
        ))
    }

    pub fn new(
        tokenizer: Arc<tokenizers::Tokenizer>,
        doc_text_source: Arc<dyn DocTextSource>,
        max_seq_len: usize,
        emit_token_type_ids: bool,
    ) -> Self {
        let max_seq_len = max_seq_len.max(1);
        // R-5b.1.4: clone the shared tokenizer and configure truncation +
        // overflow capture at max_seq_len. Padding is deliberately NOT
        // configured here: rows keep their true length, overflow is judged on
        // real content, and the conversion pass downstream is the single owner
        // of width (it pads to the longest row in the batch — see
        // TD-SELECTOR-1 gate 5 for why `max_seq_len`-wide padding was ~11×
        // wasted compute). The conversion still provides shape defense and the
        // u32 → i64 conversion the tokenizer doesn't do.
        //
        // We clone the inner Tokenizer rather than mutate the shared
        // Arc — the same `Arc<Tokenizer>` is used by the embedding
        // crate's `SharedTokenizer` for token-count chunking, which
        // wants per-call defaults. Per-extractor cloning isolates the
        // truncation config to the rerank path.
        let (configured, pad_id) =
            configure_tokenizer_for_pair_encoding(tokenizer.as_ref().clone(), max_seq_len);
        Self {
            tokenizer: Arc::new(configured),
            doc_text_source,
            max_seq_len,
            emit_token_type_ids,
            pad_id,
        }
    }
}

/// Configure truncation + overflow capture on a cloned tokenizer, disable any
/// tokenizer-level padding, and resolve the model family's real pad id.
///
/// Width policy (TD-SELECTOR-1 gate 5): the conversion layer in `extract_batch`
/// pads every row to the LONGEST row in the batch. Batch-rectangular is all the
/// ONNX session contract requires; padding to `max_seq_len` instead made every
/// pair pay the full budget — measured ~11× wasted compute on a realistic bed
/// (median pair 37 tokens against a 512 budget). The tokenizer therefore emits
/// unpadded rows and never double-pads.
///
/// The pad id comes from the tokenizer's own config when present, falling back
/// through the common special-token spellings. Hardcoding 0 is wrong for
/// non-BERT families (XLM-R's `<pad>` is id 1), which would silently corrupt
/// padded positions for those tokenizers.
fn configure_tokenizer_for_pair_encoding(
    mut tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
) -> (tokenizers::Tokenizer, i64) {
    use tokenizers::utils::truncation::{
        TruncationDirection, TruncationParams, TruncationStrategy,
    };

    let pad_id = match tokenizer.get_padding() {
        Some(existing) => existing.pad_id as i64,
        None => ["[PAD]", "<pad>", "<|pad|>", "<pad_token>"]
            .iter()
            .find_map(|spelling| tokenizer.token_to_id(spelling))
            .unwrap_or(0) as i64,
    };

    // Rows leave the tokenizer at their true length; the conversion pass pads.
    tokenizer.with_padding(None);
    // with_truncation returns Result because params can be inconsistent.
    // If configuration ever fails, the unconfigured tokenizer emits a row
    // wider than max_seq_len and the explicit pre-conversion guard rejects it.
    let _ = tokenizer.with_truncation(Some(TruncationParams {
        max_length: max_seq_len,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: TruncationDirection::Right,
    }));
    (tokenizer, pad_id)
}

impl TokenizedDocFeatureExtractor for BertPairTokenizingDocFeatureExtractor {
    fn extract_batch(&self, docs: &[DocHandle], qctx: &QueryContext) -> RankResult<TokenizedBatch> {
        if docs.is_empty() {
            return Ok(TokenizedBatch::default());
        }
        // R-5b.1.3: query text comes from the per-request QueryContext.
        // Missing or blank query text is a request-contract failure,
        // not a meaningful input to a query-document cross-encoder.
        let query: Arc<str> = qctx
            .query_text
            .clone()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| RankError::ModelInference {
                model_id: "bert_pair_extractor".into(),
                reason: "query_text is required for pair reranking".into(),
            })?;
        // Resolve every document before inference. Missing or blank text
        // means candidate hydration is incomplete and must fail closed.
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(docs.len());
        for &doc in docs {
            let text = self
                .doc_text_source
                .doc_text(doc)?
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| RankError::ModelInference {
                    model_id: "bert_pair_extractor".into(),
                    reason: format!("document text is required for handle {}", doc.0),
                })?;
            pairs.push((query.to_string(), text));
        }

        // Build the EncodeInput::Dual list. `tokenizers::encode_batch`
        // accepts owned strings via the `Into<EncodeInput>` impl on
        // `(String, String)`.
        let encodings =
            self.tokenizer
                .encode_batch(pairs, true)
                .map_err(|e| RankError::ModelInference {
                    model_id: "bert_pair_extractor".into(),
                    reason: format!("tokenizer.encode_batch: {e}"),
                })?;

        // Overflow check on TRUE content length (rows are unpadded), then pad
        // every row to the longest row in this batch. Batch-rectangular is all
        // the session contract requires; padding to `max_seq_len` instead cost
        // ~11× compute on realistic beds (TD-SELECTOR-1 gate 5). Production
        // input is never truncated here.
        let batch_width = encodings
            .iter()
            .map(|enc| enc.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(encodings.len());
        let mut attention_mask: Vec<Vec<i64>> = Vec::with_capacity(encodings.len());
        let mut token_type_ids: Option<Vec<Vec<i64>>> = if self.emit_token_type_ids {
            Some(Vec::with_capacity(encodings.len()))
        } else {
            None
        };

        for enc in &encodings {
            if !enc.get_overflowing().is_empty() || enc.get_ids().len() > self.max_seq_len {
                return Err(RankError::ModelInference {
                    model_id: "bert_pair_extractor".into(),
                    reason: format!(
                        "query-document pair exceeds max_seq_len {}; split the document before reranking",
                        self.max_seq_len
                    ),
                });
            }
            input_ids.push(pad_or_truncate_to_i64(
                enc.get_ids(),
                batch_width,
                /*pad=*/ self.pad_id,
            ));
            // Padded positions are masked out, so attention and type-id rows
            // pad with 0 regardless of the family's pad token id.
            attention_mask.push(pad_or_truncate_to_i64(
                enc.get_attention_mask(),
                batch_width,
                /*pad=*/ 0,
            ));
            if let Some(tti) = token_type_ids.as_mut() {
                tti.push(pad_or_truncate_to_i64(
                    enc.get_type_ids(),
                    batch_width,
                    /*pad=*/ 0,
                ));
            }
        }

        let batch = TokenizedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
        };
        // Belt-and-braces: the conversion loop guarantees
        // rectangular output, but verify it before returning so a
        // future refactor that breaks the invariant fails loud.
        debug_assert!(batch.validate_rectangular().is_ok());
        Ok(batch)
    }
}

/// Pad a `&[u32]` to exactly `target_len` elements as `i64`. Truncates
/// when input is too long. The extractor needs i64 because the BERT
/// ONNX export expects int64 inputs; `tokenizers` returns u32.
fn pad_or_truncate_to_i64(src: &[u32], target_len: usize, pad: i64) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(target_len);
    let take = src.len().min(target_len);
    for &v in &src[..take] {
        out.push(v as i64);
    }
    while out.len() < target_len {
        out.push(pad);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic whitespace tokenizer for shape
    /// testing. WordLevel requires the unk_token to be present in the
    /// vocabulary (it's looked up on encode failure), so we seed a
    /// small vocab containing only `[UNK]` and the special pair
    /// tokens. Every other word in the test queries / docs is
    /// unknown and tokenizes to the [UNK] id — fine because these
    /// tests assert shapes, not vocab fidelity.
    fn synthetic_tokenizer() -> Arc<tokenizers::Tokenizer> {
        use std::collections::HashMap;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::tokenizer::Tokenizer;
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        // Seed real vocab entries so the per-request-qctx test can
        // verify that different queries tokenize differently. Without
        // these, every test-input word maps to [UNK] and queries are
        // indistinguishable on the wire.
        for (i, w) in ["alpha", "beta", "gamma", "delta", "doc"]
            .iter()
            .enumerate()
        {
            vocab.insert((*w).to_string(), 10 + i as u32);
        }
        let model = WordLevel::builder()
            .unk_token("[UNK]".to_string())
            // tokenizers 0.23 takes an ahash map here; collect() lets the
            // parameter type drive the conversion without naming it.
            .vocab(vocab.into_iter().collect())
            .build()
            .expect("synthetic WordLevel build");
        let mut tk = Tokenizer::new(model);
        tk.with_pre_tokenizer(Some(Whitespace {}));
        Arc::new(tk)
    }

    fn doc_text_source(entries: &[(u32, &str)]) -> Arc<dyn DocTextSource> {
        Arc::new(HashMapDocTextSource::with_entries(
            entries.iter().map(|(d, t)| (DocHandle(*d), *t)),
        ))
    }

    /// Convenience: build (extractor, qctx) so tests stay readable
    /// despite query text now living on the per-request context.
    fn extractor_with(
        query: &str,
        max_seq_len: usize,
        emit_token_type_ids: bool,
        doc_texts: &[(u32, &str)],
    ) -> (BertPairTokenizingDocFeatureExtractor, QueryContext) {
        let e = BertPairTokenizingDocFeatureExtractor::new(
            synthetic_tokenizer(),
            doc_text_source(doc_texts),
            max_seq_len,
            emit_token_type_ids,
        );
        let qctx = QueryContext {
            query_text: Some(Arc::<str>::from(query)),
            ..QueryContext::default()
        };
        (e, qctx)
    }

    // ---------------- HashMapDocTextSource ----------------

    #[test]
    fn hashmap_source_returns_some_for_known_doc() {
        let s = HashMapDocTextSource::with_entries([(DocHandle(1), "hello world")]);
        assert_eq!(
            s.doc_text(DocHandle(1)).unwrap().as_deref(),
            Some("hello world")
        );
    }

    #[test]
    fn hashmap_source_returns_none_for_unknown_doc() {
        let s = HashMapDocTextSource::new();
        assert!(s.doc_text(DocHandle(99)).unwrap().is_none());
    }

    #[test]
    fn hashmap_source_insert_replaces_existing_value() {
        let mut s = HashMapDocTextSource::new();
        s.insert(DocHandle(1), "first");
        s.insert(DocHandle(1), "second");
        assert_eq!(s.doc_text(DocHandle(1)).unwrap().as_deref(), Some("second"));
        assert_eq!(s.len(), 1);
    }

    // ---------------- pad_or_truncate_to_i64 ----------------

    #[test]
    fn pad_or_truncate_pads_short_input_with_zeros() {
        let out = pad_or_truncate_to_i64(&[1, 2, 3], 6, 0);
        assert_eq!(out, vec![1, 2, 3, 0, 0, 0]);
    }

    #[test]
    fn pad_or_truncate_truncates_long_input() {
        let out = pad_or_truncate_to_i64(&[1, 2, 3, 4, 5, 6], 3, 0);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn pad_or_truncate_exact_length_is_pass_through() {
        let out = pad_or_truncate_to_i64(&[1, 2, 3], 3, 0);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn pad_or_truncate_empty_input_emits_all_padding() {
        let out = pad_or_truncate_to_i64(&[], 4, 7);
        assert_eq!(out, vec![7, 7, 7, 7]);
    }

    // ---------------- BertPairTokenizingDocFeatureExtractor ----------------

    #[test]
    fn extractor_empty_docs_returns_empty_batch_without_tokenizer_call() {
        let (e, qctx) = extractor_with("query", 16, false, &[(1, "doc one")]);
        let b = e.extract_batch(&[], &qctx).unwrap();
        assert_eq!(b.batch_size(), 0);
        assert_eq!(b.seq_len(), 0);
    }

    #[test]
    fn extractor_pads_to_batch_longest_not_max_seq_len() {
        // Width policy (TD-SELECTOR-1 gate 5): rows pad to the longest row in
        // the batch, NOT to max_seq_len — padding every pair to the budget cost
        // ~11× compute on realistic beds. With a generous budget and short docs,
        // the emitted width must be the longest actual row.
        let (e, qctx) = extractor_with(
            "alpha beta",
            64,
            false,
            &[
                (1, "doc one"),
                (2, "doc two"),
                (3, "doc three with more words"),
            ],
        );
        let b = e
            .extract_batch(&[DocHandle(1), DocHandle(2), DocHandle(3)], &qctx)
            .unwrap();
        assert_eq!(b.batch_size(), 3);
        let width = b.seq_len();
        assert!(
            width < 64,
            "width must be batch-longest ({width}), not the 64 budget"
        );
        let longest = b
            .attention_mask
            .iter()
            .map(|row| row.iter().sum::<i64>())
            .max()
            .unwrap();
        assert_eq!(
            width as i64, longest,
            "the longest row has no padding, so its length IS the batch width"
        );
        assert!(b.validate_rectangular().is_ok());
    }

    #[test]
    fn extractor_rejects_overlength_pair_instead_of_truncating() {
        let (e, qctx) = extractor_with(
            "alpha beta",
            6,
            false,
            &[(1, "doc three with extra tokens beyond the limit")],
        );
        let error = e.extract_batch(&[DocHandle(1)], &qctx).unwrap_err();
        assert!(error.to_string().contains("exceeds max_seq_len 6"));
        assert!(error.to_string().contains("split the document"));
    }

    #[test]
    fn extractor_emits_token_type_ids_when_configured() {
        let (e, qctx) = extractor_with("query", 8, true, &[(1, "doc")]);
        let b = e.extract_batch(&[DocHandle(1)], &qctx).unwrap();
        assert!(b.token_type_ids.is_some());
        let tti = b.token_type_ids.clone().unwrap();
        assert_eq!(tti.len(), 1);
        // Single-row batch: width is that row's own true length, under the cap.
        assert_eq!(tti[0].len(), b.seq_len());
        assert!(tti[0].len() <= 8);
    }

    #[test]
    fn extractor_omits_token_type_ids_by_default() {
        let (e, qctx) = extractor_with("query", 8, false, &[(1, "doc")]);
        let b = e.extract_batch(&[DocHandle(1)], &qctx).unwrap();
        assert!(b.token_type_ids.is_none());
    }

    #[test]
    fn extractor_rejects_missing_doc_text() {
        let (e, qctx) = extractor_with("query", 8, false, &[(1, "the real doc")]);
        let error = e
            .extract_batch(&[DocHandle(1), DocHandle(99)], &qctx)
            .unwrap_err();
        assert!(error.to_string().contains("handle 99"));
    }

    #[test]
    fn extractor_uses_per_request_query_text_from_qctx() {
        // R-5b.1.3: query text now flows from QueryContext, not from
        // extractor state. Two different qctx values yield two
        // different tokenizations even though the extractor instance
        // is the same — confirms the per-request plumbing.
        let (e, _) = extractor_with("ignored-construction-arg", 8, false, &[(1, "doc")]);
        let qctx_a = QueryContext {
            query_text: Some(Arc::<str>::from("alpha")),
            ..QueryContext::default()
        };
        let qctx_b = QueryContext {
            query_text: Some(Arc::<str>::from("beta gamma delta")),
            ..QueryContext::default()
        };
        let a = e.extract_batch(&[DocHandle(1)], &qctx_a).unwrap();
        let b = e.extract_batch(&[DocHandle(1)], &qctx_b).unwrap();
        // Same doc, same vocab → identical token ids when the query
        // also matches; here the queries differ so at least one
        // input_id position must differ between batches. Single-row
        // batches are padded to their own length, so widths differ
        // with the queries.
        assert!(a.input_ids[0].len() <= 8 && b.input_ids[0].len() <= 8);
        assert_ne!(
            a.input_ids[0], b.input_ids[0],
            "different query_text must produce different tokens"
        );
    }

    #[test]
    fn extractor_rejects_missing_query_text() {
        let (e, _) = extractor_with("ignored", 8, false, &[(1, "doc")]);
        let qctx = QueryContext::default(); // query_text is None
        let error = e.extract_batch(&[DocHandle(1)], &qctx).unwrap_err();
        assert!(error.to_string().contains("query_text is required"));
    }

    #[test]
    fn extractor_clamps_zero_seq_len_then_rejects_an_unrepresentable_pair() {
        // Defensive: passing max_seq_len = 0 at construction clamps to 1,
        // then the pair contract fails closed because query + doc cannot fit.
        let e = BertPairTokenizingDocFeatureExtractor::new(
            synthetic_tokenizer(),
            doc_text_source(&[(1, "doc")]),
            0,
            false,
        );
        let qctx = QueryContext {
            query_text: Some(Arc::<str>::from("query")),
            ..QueryContext::default()
        };
        let error = e.extract_batch(&[DocHandle(1)], &qctx).unwrap_err();
        assert!(error.to_string().contains("max_seq_len 1"));
    }

    // ---------------- R-5b.1.4: tokenizer pre-padding/truncation ----------------

    #[test]
    fn extractor_constructor_does_not_mutate_shared_tokenizer_state() {
        // The shared Arc<Tokenizer> the caller hands in must NOT have
        // its padding/truncation config touched — other consumers (e.g.
        // the embedding crate's SharedTokenizer) rely on per-call
        // defaults. Verify by encoding the same input twice via the
        // shared Arc before and after constructing the extractor.
        let shared = synthetic_tokenizer();
        let pre = shared.encode("alpha beta", false).unwrap();
        let _e = BertPairTokenizingDocFeatureExtractor::new(
            shared.clone(),
            doc_text_source(&[(1, "doc")]),
            32,
            false,
        );
        let post = shared.encode("alpha beta", false).unwrap();
        assert_eq!(
            pre.get_ids(),
            post.get_ids(),
            "constructor must not mutate the caller's shared tokenizer"
        );
        assert_eq!(pre.get_ids().len(), post.get_ids().len());
    }

    #[test]
    fn extractor_tokenizer_outputs_are_rectangular_at_batch_longest() {
        // TD-SELECTOR-1 gate 5 (#1741): rows pad to the batch's longest
        // row, NOT to max_seq_len — padding to max_seq_len cost ~11x
        // compute on realistic beds. The synthetic tokenizer has no
        // post-processor, so a pair's true length is query words + doc
        // words: the longest bed row here is query 3 + doc 4 = 7.
        let (e, qctx) = extractor_with(
            "alpha beta gamma",
            10,
            false,
            &[
                (1, "doc"),                    // shortest
                (2, "alpha beta gamma delta"), // batch longest: 7
                (3, "doc alpha"),              // middle
            ],
        );
        let b = e
            .extract_batch(&[DocHandle(1), DocHandle(2), DocHandle(3)], &qctx)
            .unwrap();
        assert_eq!(b.batch_size(), 3);
        for row in &b.input_ids {
            assert_eq!(
                row.len(),
                7,
                "rows must be rectangular at the batch-longest length"
            );
        }
        for row in &b.attention_mask {
            assert_eq!(row.len(), 7);
        }
        // Batch-longest, not max_seq_len: every doc fits, so the width
        // must stay strictly below the configured max.
        assert!(b.input_ids[0].len() < 10);
        assert!(b.validate_rectangular().is_ok());
    }

    #[test]
    fn extractor_attention_mask_marks_padding_positions_zero() {
        // R-5b.1.4 (as amended by TD-SELECTOR-1 gate 5, #1741): rows pad
        // to the batch-longest length and the attention_mask marks the
        // padded positions 0. A single short pair has no padding, so the
        // bed mixes a short and a longer doc: width = 1 query word + 4
        // doc words = 5, the short row masks its last 3 positions.
        let (e, qctx) = extractor_with(
            "alpha",
            10,
            false,
            &[(1, "doc"), (2, "alpha beta gamma delta")],
        );
        let b = e
            .extract_batch(&[DocHandle(1), DocHandle(2)], &qctx)
            .unwrap();
        let mask = &b.attention_mask[0];
        let real = mask.iter().filter(|&&x| x == 1).count();
        let pad = mask.iter().filter(|&&x| x == 0).count();
        assert_eq!(real, 2, "1 query word + 1 doc word are real tokens");
        assert_eq!(pad, 3, "padding to the batch-longest row is masked 0");
        assert_eq!(real + pad, 5);
    }

    #[test]
    fn extractor_doc_text_lookup_errors_propagate() {
        struct BrokenSource;
        impl DocTextSource for BrokenSource {
            fn doc_text(&self, _: DocHandle) -> RankResult<Option<String>> {
                Err(RankError::ModelInference {
                    model_id: "broken_source".into(),
                    reason: "attribute store unavailable".into(),
                })
            }
        }
        let e = BertPairTokenizingDocFeatureExtractor::new(
            synthetic_tokenizer(),
            Arc::new(BrokenSource),
            4,
            false,
        );
        let qctx = QueryContext {
            query_text: Some(Arc::<str>::from("query")),
            ..QueryContext::default()
        };
        match e.extract_batch(&[DocHandle(1)], &qctx) {
            Err(RankError::ModelInference { reason, .. }) => {
                assert!(reason.contains("attribute store unavailable"));
            }
            other => panic!("expected ModelInference, got {other:?}"),
        }
    }

    #[test]
    fn extractor_is_dyn_compatible_as_doc_feature_extractor() {
        let (e, qctx) = extractor_with("q", 4, false, &[(1, "doc")]);
        let dyn_e: Arc<dyn TokenizedDocFeatureExtractor> = Arc::new(e);
        let b = dyn_e.extract_batch(&[DocHandle(1)], &qctx).unwrap();
        assert_eq!(b.batch_size(), 1);
    }
}
