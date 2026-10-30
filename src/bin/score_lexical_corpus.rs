//! Seal a source-level lexical run through a ProximaDB production lexical engine.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use proximadb::storage::document::indexes::fulltext::FullTextIndex as DocumentFullTextIndex;
use proximadb::storage::engines::core::formats::columnar::fulltext_index::{
    FullTextIndex as CanonicalFullTextIndex, TokenizerConfig,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaTree, ProximaTreeNode};
use proximadb_search_types::search_params::LexicalQueryMode;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

#[derive(Debug, Parser)]
#[command(about = "Score JSONL sources with a ProximaDB production lexical engine")]
struct Args {
    #[arg(long)]
    documents: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value_t = 100)]
    top_k: usize,
    #[arg(long, value_enum, default_value_t = QueryErrorPolicy::Fail)]
    query_error_policy: QueryErrorPolicy,
    #[arg(long, value_enum, default_value_t = Engine::DocumentTantivy)]
    engine: Engine,
    #[arg(long, value_enum, default_value_t = QueryModeArg::NaturalLanguage)]
    query_mode: QueryModeArg,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Engine {
    DocumentTantivy,
    CanonicalHybrid,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum QueryModeArg {
    NaturalLanguage,
    AdvancedSyntax,
}

impl From<QueryModeArg> for LexicalQueryMode {
    fn from(value: QueryModeArg) -> Self {
        match value {
            QueryModeArg::NaturalLanguage => Self::NaturalLanguage,
            QueryModeArg::AdvancedSyntax => Self::AdvancedSyntax,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum QueryErrorPolicy {
    Fail,
    ZeroPad,
}

#[derive(Debug, Deserialize)]
struct DocumentInput {
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    body: String,
}

#[derive(Debug, Deserialize)]
struct QueryInput {
    id: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct RunRow<'a> {
    query_id: &'a str,
    rank: usize,
    score: f32,
    source_id: &'a str,
}

enum EvalIndex {
    DocumentTantivy {
        _root: tempfile::TempDir,
        // Both variants boxed: tantivy 0.27's index handle grew past clippy's
        // large_enum_variant threshold, and boxing only this variant left
        // CanonicalHybrid as the new oversized one. These are long-lived
        // eval harness handles, so the extra indirection is noise.
        index: Box<DocumentFullTextIndex>,
    },
    CanonicalHybrid(Box<CanonicalFullTextIndex>),
}

impl EvalIndex {
    fn build(engine: Engine, documents: &[DocumentInput]) -> Result<Self> {
        match engine {
            Engine::DocumentTantivy => {
                // Reopen after commit so the evaluation observes the same persisted
                // segment boundary used after a process restart, without relying on
                // the asynchronous OnCommitWithDelay reader refresh.
                let root = tempfile::tempdir()?;
                {
                    let mut index =
                        DocumentFullTextIndex::new_persistent("source-eval", root.path())?;
                    index.add_field("title")?;
                    index.add_field("body")?;
                    for document in documents {
                        index.index_document(&document.id, &document_tree(document))?;
                    }
                    index.commit()?;
                }
                let mut index = DocumentFullTextIndex::new_persistent("source-eval", root.path())?;
                index.add_field("title")?;
                index.add_field("body")?;
                Ok(Self::DocumentTantivy {
                    _root: root,
                    index: Box::new(index),
                })
            }
            Engine::CanonicalHybrid => {
                let mut index = CanonicalFullTextIndex::new(TokenizerConfig::for_keyword_search());
                for document in documents {
                    let text = format!("{}\n\n{}", document.title, document.body);
                    index.add_document(&document.id, &text)?;
                }
                Ok(Self::CanonicalHybrid(Box::new(index)))
            }
        }
    }

    fn search(
        &self,
        query: &str,
        limit: usize,
        mode: LexicalQueryMode,
    ) -> Result<Vec<(String, f32)>> {
        match self {
            Self::DocumentTantivy { index, .. } => index.search_with_mode(query, limit, mode),
            Self::CanonicalHybrid(index) => Ok(index
                .search_with_mode(query, limit, mode)?
                .into_iter()
                .map(|result| (result.doc_id, result.score as f32))
                .collect()),
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut source = File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    // sha2 0.11 no longer implements io::Write for its digest types; stream
    // through update() instead (same bytes hashed, same output).
    loop {
        let n = source
            .read(&mut buf)
            .with_context(|| format!("failed to hash {}", path.display()))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn load_jsonl<T: DeserializeOwned>(path: &Path, kind: &str) -> Result<Vec<T>> {
    let source = File::open(path).with_context(|| format!("failed to open {kind}"))?;
    let mut values = Vec::new();
    for (index, line) in BufReader::new(source).lines().enumerate() {
        let line = line.with_context(|| format!("failed to read {kind} line {}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(
            serde_json::from_str(&line)
                .with_context(|| format!("invalid {kind} JSON at line {}", index + 1))?,
        );
    }
    if values.is_empty() {
        bail!("{kind} contains no records");
    }
    Ok(values)
}

fn validate_inputs(
    documents: &[DocumentInput],
    queries: &[QueryInput],
    top_k: usize,
) -> Result<()> {
    if top_k == 0 || top_k > documents.len() {
        bail!("top_k must be in 1..={}", documents.len());
    }
    let mut document_ids = std::collections::HashSet::new();
    for document in documents {
        if document.id.trim().is_empty() || !document_ids.insert(document.id.as_str()) {
            bail!("document IDs must be non-empty and unique");
        }
        if document.title.trim().is_empty() && document.body.trim().is_empty() {
            bail!("document '{}' has no title or body", document.id);
        }
    }
    let mut query_ids = std::collections::HashSet::new();
    for query in queries {
        if query.id.trim().is_empty() || !query_ids.insert(query.id.as_str()) {
            bail!("query IDs must be non-empty and unique");
        }
        if query.text.trim().is_empty() {
            bail!("query '{}' has no text", query.id);
        }
    }
    Ok(())
}

fn document_tree(document: &DocumentInput) -> ProximaTree {
    ProximaTree::from([
        (
            "title".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(document.title.clone())),
        ),
        (
            "body".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(document.body.clone())),
        ),
    ])
}

fn atomic_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let parent = path.parent().context("output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, value)?;
    temporary.write_all(b"\n")?;
    temporary.flush()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to persist {}", path.display()))?;
    Ok(())
}

fn engine_manifest(engine: Engine, query_mode: QueryModeArg) -> serde_json::Value {
    match engine {
        Engine::DocumentTantivy => json!({
            "id": "document_tantivy",
            "implementation": "proximadb::storage::document::indexes::fulltext::FullTextIndex",
            "implementation_sha256": sha256_bytes(include_bytes!("../storage/document/indexes/fulltext.rs")),
            "library": "tantivy",
            "library_requirement": "0.22",
            "query_mode": query_mode,
            "query_interpretation": match query_mode {
                QueryModeArg::NaturalLanguage => "configured field analyzers; OR terms; no query parser",
                QueryModeArg::AdvancedSyntax => "Tantivy QueryParser strict",
            },
            "scoring": "Tantivy BM25 defaults",
            "text_fields": ["title", "body"],
        }),
        Engine::CanonicalHybrid => json!({
            "id": "canonical_hybrid",
            "implementation": "proximadb_storage_common::text_search::fulltext_index::FullTextIndex",
            "implementation_sha256": sha256_bytes(include_bytes!("../../crates/storage/proximadb-storage-common/src/text_search/fulltext_index.rs")),
            "library": null,
            "query_mode": query_mode,
            "query_interpretation": match query_mode {
                QueryModeArg::NaturalLanguage => "TokenizerConfig::for_keyword_search; OR terms",
                QueryModeArg::AdvancedSyntax => "unsupported; fails explicitly",
            },
            "scoring": "ProximaDB in-memory BM25 defaults",
            "text_fields": ["combined title and body"],
        }),
    }
}

fn run(args: &Args) -> Result<serde_json::Value> {
    let documents: Vec<DocumentInput> = load_jsonl(&args.documents, "documents")?;
    let queries: Vec<QueryInput> = load_jsonl(&args.queries, "queries")?;
    validate_inputs(&documents, &queries, args.top_k)?;

    let index = EvalIndex::build(args.engine, &documents)?;
    let query_mode = LexicalQueryMode::from(args.query_mode);

    let output_parent = args.output.parent().context("output path has no parent")?;
    std::fs::create_dir_all(output_parent)?;
    let mut temporary = NamedTempFile::new_in(output_parent)?;
    let mut query_errors = Vec::new();
    {
        let mut output = BufWriter::new(&mut temporary);
        for query in &queries {
            let matched: HashMap<String, f32> =
                match index.search(&query.text, documents.len(), query_mode) {
                    Ok(results) => results.into_iter().collect(),
                    Err(error) => match args.query_error_policy {
                        QueryErrorPolicy::Fail => {
                            return Err(error).with_context(|| {
                                format!(
                                    "query '{}' failed for {:?} under {:?}",
                                    query.id, args.engine, args.query_mode
                                )
                            });
                        }
                        QueryErrorPolicy::ZeroPad => {
                            query_errors.push(json!({
                                "query_id": query.id,
                                "error": format!("{error:#}"),
                            }));
                            HashMap::new()
                        }
                    },
                };
            let mut ranked: Vec<(&str, f32)> = documents
                .iter()
                .map(|document| {
                    (
                        document.id.as_str(),
                        matched.get(&document.id).copied().unwrap_or(0.0),
                    )
                })
                .collect();
            if ranked.iter().any(|(_, score)| !score.is_finite()) {
                bail!("Tantivy returned a non-finite score");
            }
            ranked.sort_by(|left, right| {
                right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0))
            });
            for (rank, (source_id, score)) in ranked.into_iter().take(args.top_k).enumerate() {
                serde_json::to_writer(
                    &mut output,
                    &RunRow {
                        query_id: &query.id,
                        rank: rank + 1,
                        score,
                        source_id,
                    },
                )?;
                output.write_all(b"\n")?;
            }
        }
        output.flush()?;
    }
    temporary
        .persist(&args.output)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to persist {}", args.output.display()))?;

    let manifest = json!({
        "schema_version": 3,
        "producer_sha256": sha256_bytes(include_bytes!("score_lexical_corpus.rs")),
        "query_contract_sha256": sha256_bytes(include_bytes!("../../crates/foundation/proximadb-search-types/src/search_params.rs")),
        "engine": engine_manifest(args.engine, args.query_mode),
        "candidate_granularity": "source",
        "tie_break": "score descending, source_id ascending",
        "zero_score_policy": "pad the full source universe deterministically before top-k",
        "query_error_policy": args.query_error_policy,
        "query_error_count": query_errors.len(),
        "query_errors": query_errors,
        "top_k": args.top_k,
        "document_count": documents.len(),
        "query_count": queries.len(),
        "run_row_count": queries.len() * args.top_k,
        "documents_path": args.documents.canonicalize()?.display().to_string(),
        "documents_sha256": sha256_file(&args.documents)?,
        "queries_path": args.queries.canonicalize()?.display().to_string(),
        "queries_sha256": sha256_file(&args.queries)?,
        "run_path": args.output.canonicalize()?.display().to_string(),
        "run_sha256": sha256_file(&args.output)?,
        "limitations": [
            "quality evidence only; no serving latency claim",
            "engines differ in field modeling and analyzers; score magnitudes are not cross-engine calibrated",
        ],
    });
    atomic_json(&args.manifest, &manifest)?;
    Ok(manifest)
}

fn main() -> Result<()> {
    let manifest = run(&Args::parse())?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_title_matches_and_emits_complete_deterministic_depth() -> Result<()> {
        let root = tempfile::tempdir()?;
        let documents = root.path().join("documents.jsonl");
        let queries = root.path().join("queries.jsonl");
        let output = root.path().join("run.jsonl");
        let manifest_path = root.path().join("manifest.json");
        std::fs::write(
            &documents,
            concat!(
                "{\"id\":\"doc-z\",\"title\":\"needle title\",\"body\":\"plain\"}\n",
                "{\"id\":\"doc-b\",\"title\":\"other\",\"body\":\"plain\"}\n",
                "{\"id\":\"doc-a\",\"title\":\"other\",\"body\":\"plain\"}\n"
            ),
        )?;
        std::fs::write(&queries, "{\"id\":\"q1\",\"text\":\"needle\"}\n")?;

        let manifest = run(&Args {
            documents,
            queries,
            output: output.clone(),
            manifest: manifest_path,
            top_k: 3,
            query_error_policy: QueryErrorPolicy::Fail,
            engine: Engine::DocumentTantivy,
            query_mode: QueryModeArg::NaturalLanguage,
        })?;

        let rows: Vec<serde_json::Value> = load_jsonl(&output, "run")?;
        let ids: Vec<&str> = rows
            .iter()
            .filter_map(|row| row.get("source_id").and_then(serde_json::Value::as_str))
            .collect();
        assert_eq!(ids, vec!["doc-z", "doc-a", "doc-b"]);
        assert_eq!(manifest["run_row_count"], 3);
        assert_eq!(manifest["engine"]["library"], "tantivy");
        assert_eq!(
            manifest["engine"]["implementation_sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
        assert_eq!(
            manifest["query_contract_sha256"].as_str().map(str::len),
            Some(64)
        );
        Ok(())
    }

    #[test]
    fn advanced_mode_zero_pad_records_strict_query_parser_failures() -> Result<()> {
        let root = tempfile::tempdir()?;
        let documents = root.path().join("documents.jsonl");
        let queries = root.path().join("queries.jsonl");
        let output = root.path().join("run.jsonl");
        std::fs::write(
            &documents,
            concat!(
                "{\"id\":\"doc-b\",\"title\":\"plain\",\"body\":\"text\"}\n",
                "{\"id\":\"doc-a\",\"title\":\"plain\",\"body\":\"text\"}\n"
            ),
        )?;
        std::fs::write(&queries, "{\"id\":\"q-bad\",\"text\":\"unknown:value\"}\n")?;

        let manifest = run(&Args {
            documents,
            queries,
            output: output.clone(),
            manifest: root.path().join("manifest.json"),
            top_k: 2,
            query_error_policy: QueryErrorPolicy::ZeroPad,
            engine: Engine::DocumentTantivy,
            query_mode: QueryModeArg::AdvancedSyntax,
        })?;

        let rows: Vec<serde_json::Value> = load_jsonl(&output, "run")?;
        let ids: Vec<&str> = rows
            .iter()
            .filter_map(|row| row.get("source_id").and_then(serde_json::Value::as_str))
            .collect();
        assert_eq!(ids, vec!["doc-a", "doc-b"]);
        assert_eq!(manifest["query_error_count"], 1);
        assert_eq!(manifest["query_errors"][0]["query_id"], "q-bad");
        Ok(())
    }

    #[test]
    fn natural_mode_has_zero_errors_for_prose_punctuation() -> Result<()> {
        let root = tempfile::tempdir()?;
        let documents = root.path().join("documents.jsonl");
        let queries = root.path().join("queries.jsonl");
        let output = root.path().join("run.jsonl");
        std::fs::write(
            &documents,
            concat!(
                "{\"id\":\"doc-b\",\"title\":\"plain\",\"body\":\"text\"}\n",
                "{\"id\":\"doc-a\",\"title\":\"Bugs and Drugs\",\"body\":\"Pork Yersinia\"}\n"
            ),
        )?;
        std::fs::write(
            &queries,
            "{\"id\":\"q-prose\",\"text\":\"Bugs & Drugs in Pork: Yersinia?\"}\n",
        )?;

        let manifest = run(&Args {
            documents,
            queries,
            output: output.clone(),
            manifest: root.path().join("manifest.json"),
            top_k: 2,
            query_error_policy: QueryErrorPolicy::Fail,
            engine: Engine::DocumentTantivy,
            query_mode: QueryModeArg::NaturalLanguage,
        })?;

        let rows: Vec<serde_json::Value> = load_jsonl(&output, "run")?;
        assert_eq!(rows[0]["source_id"], "doc-a");
        assert_eq!(manifest["query_error_count"], 0);
        assert_eq!(manifest["engine"]["query_mode"], "natural_language");
        Ok(())
    }
}
