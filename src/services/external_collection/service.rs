//! `ExternalCollectionService` — register / build / search external collections
//! (Phase 8 F5 / TD-090).
//!
//! Register catalogs the external table un-copied (`FederatedRead` authority +
//! a `VectorAnn` projection in `ExternalSnapshotRegistered`). Build reads the
//! external source, constructs a ProximaDB-owned IVF index **in place** (via
//! `AxisManager::rebuild_and_swap_ivf_index`, which writes nothing to WAL or
//! storage), registers the IVF routing strategy, and flips the projection to
//! `Fresh` with the snapshot fingerprint as lineage. Search delegates to the
//! same `AxisManager::query` path native collections use.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use proximadb_catalog::{
    CatalogColumn, CatalogProjection, CatalogProjectionKind, CatalogStorageLayout,
    CatalogTableSchema, ProjectionFreshnessState,
};
use proximadb_data_model::ProximaType;

use proximadb_records::ProximaRecord;

use super::registry::ExternalCollectionRegistry;
#[cfg(feature = "axis")]
use super::source_reader::read_external_records;
use super::source_reader::{read_external_text, read_records_by_ids, snapshot_fingerprint};
use super::types::{ExternalCollection, ExternalCollectionSpec, ExternalCollectionStatus};
use crate::catalog::CatalogManager;
use crate::core::search::hybrid::{BM25Result, FusionStrategy, HybridFusionEngine, VectorResult};
#[cfg(feature = "axis")]
use crate::index::AxisManager;
#[cfg(feature = "axis")]
use crate::index::axis::management::manager::{AxisHybridQuery, VectorQuery};
#[cfg(feature = "axis")]
use crate::index::axis::types::{Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification};
use crate::storage::engines::core::formats::columnar::fulltext_index::{
    FullTextIndex, TokenizerConfig,
};
use proximadb_search_types::search_params::LexicalQueryMode;

/// Reciprocal-rank-fusion constant for external hybrid search (the native
/// default). Rank-based, so robust to the BM25-vs-cosine score-scale mismatch.
const HYBRID_RRF_K: usize = 60;
/// Candidate-pool multiplier per side before fusion (oversample so fusion has
/// enough overlap to work with).
const HYBRID_POOL_FACTOR: usize = 5;

/// Canonical name of the projection that tracks the ProximaDB-owned vector
/// index built over the external source.
pub const EXTERNAL_INDEX_PROJECTION: &str = "external_index";

/// A scored search hit with the full record federated from the external source.
#[derive(Debug, Clone)]
pub struct ExternalHit {
    /// Record id (the source `id_column` value).
    pub id: String,
    /// Similarity score from the vector index.
    pub score: f32,
    /// Full record fetched from the external source (props = non-vector columns;
    /// empty if the source row could not be fetched).
    pub record: ProximaRecord,
}

/// Outcome of an on-demand `refresh`.
#[derive(Debug, Clone, PartialEq)]
pub struct RefreshOutcome {
    /// Whether the source changed since the last build (fingerprint mismatch).
    pub stale_detected: bool,
    /// Whether the index was rebuilt (true iff `stale_detected`).
    pub rebuilt: bool,
    /// The current snapshot fingerprint (new one if rebuilt, else unchanged).
    pub snapshot_id: String,
    /// Records indexed after a rebuild (unchanged if not rebuilt).
    pub indexed_record_count: u64,
}

/// Control-plane facade for external collections.
pub struct ExternalCollectionService {
    registry: Arc<ExternalCollectionRegistry>,
    catalog_manager: Arc<CatalogManager>,
    #[cfg(feature = "axis")]
    axis_manager: Arc<AxisManager>,
    #[cfg(feature = "axis")]
    axis_runtime_enabled: bool,
    /// F5 Slice 3: per-collection BM25 inverted index over the source text
    /// column (keyed by collection name). In-memory; built on `build`/`refresh`.
    fulltext_indexes: Arc<RwLock<HashMap<String, FullTextIndex>>>,
}

impl ExternalCollectionService {
    pub fn new(
        registry: Arc<ExternalCollectionRegistry>,
        catalog_manager: Arc<CatalogManager>,
        #[cfg(feature = "axis")] axis_manager: Arc<AxisManager>,
    ) -> Self {
        Self {
            registry,
            catalog_manager,
            #[cfg(feature = "axis")]
            axis_manager,
            #[cfg(feature = "axis")]
            axis_runtime_enabled: false,
            fulltext_indexes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Apply the process-level AXIS policy resolved during server boot.
    #[cfg(feature = "axis")]
    pub fn with_axis_runtime_enabled(mut self, enabled: bool) -> Self {
        self.axis_runtime_enabled = enabled;
        self
    }

    #[cfg(feature = "axis")]
    fn require_axis_runtime_enabled(&self) -> Result<()> {
        if self.axis_runtime_enabled {
            Ok(())
        } else {
            anyhow::bail!(
                "external vector indexes require storage.optimization.enable_axis_indexes=true"
            )
        }
    }

    /// Shared registry handle.
    pub fn registry(&self) -> Arc<ExternalCollectionRegistry> {
        self.registry.clone()
    }

    /// Register an external collection un-copied: catalog it with `FederatedRead`
    /// authority + a `VectorAnn` projection in `ExternalSnapshotRegistered`, and
    /// record it in the durable registry. Does **not** read or copy the source.
    pub async fn register(&self, spec: ExternalCollectionSpec) -> Result<ExternalCollection> {
        // Registered for the tenant in scope (internal execution = the default
        // tenant); names are unique per tenant.
        let owner = crate::query::request_scope::resolution_tenant(&spec.name)?;
        let owner_key = proximadb_tenant::resolve_request_tenant(owner.as_deref());
        if self
            .registry
            .list_all()
            .iter()
            .any(|ec| ec.spec.name == spec.name && ec.visible_to(&owner_key))
        {
            anyhow::bail!("external collection '{}' is already registered", spec.name);
        }
        let snapshot_id = snapshot_fingerprint(&spec.location)
            .with_context(|| format!("fingerprint external source '{}'", spec.location))?;

        let (catalog, identifier) = self
            .catalog_manager
            .resolve_table_scoped(
                &spec.name,
                crate::query::request_scope::resolution_tenant(&spec.name)?.as_deref(),
            )
            .await?;
        if catalog.table_exists(&identifier).await? {
            anyhow::bail!(
                "a catalog table named '{}' already exists; cannot register as external",
                spec.name
            );
        }
        // Ensure the namespace exists (idempotent — ignore "already exists").
        let _ = catalog
            .create_namespace(&identifier.namespace, std::collections::HashMap::new())
            .await;

        let layout = CatalogStorageLayout::federated_read(
            "external_source",
            spec.format.catalog_format(),
            spec.location.clone(),
        );
        let projection = CatalogProjection::rebuildable(
            EXTERNAL_INDEX_PROJECTION,
            CatalogProjectionKind::VectorAnn,
            &spec.name,
        )
        .with_freshness_state(ProjectionFreshnessState::ExternalSnapshotRegistered)
        .with_lineage(snapshot_id.clone(), snapshot_id.clone());

        // A `Vector` catalog column must declare its `dimension` property
        // (enforced by `validate_storage_contract`); `CatalogColumn` exposes the
        // property map directly (no builder method).
        let mut vector_col = CatalogColumn::new(
            1,
            &spec.vector_column,
            ProximaType::DenseVector {
                element: proximadb_data_model::VectorElement::Float32,
                dim: 0,
            },
        );
        vector_col
            .properties
            .insert("dimension".to_string(), spec.dimension.to_string());
        let mut schema = CatalogTableSchema::new(spec.name.clone())
            .with_column(CatalogColumn::new(0, &spec.id_column, ProximaType::String))
            .with_column(vector_col)
            .with_projection(projection);
        // `CatalogTableSchema::default()` seeds one `InternalCanonical` layout;
        // an external collection has no internal-canonical storage, so replace
        // it with the federated-read layout (rather than appending behind it).
        schema.storage_layouts = vec![layout];
        catalog
            .create_table(&identifier, schema)
            .await
            .with_context(|| format!("catalog external collection '{}'", spec.name))?;

        let mut ec = ExternalCollection::new(spec, snapshot_id);
        ec.tenant = owner;
        self.registry.upsert(ec.clone());
        Ok(ec)
    }

    /// Build the ProximaDB-owned IVF index in place over the external source and
    /// publish it (`Fresh`). Reads the source but copies nothing into ProximaDB
    /// storage. Returns the number of records indexed.
    pub async fn build(&self, id: &str) -> Result<usize> {
        #[cfg(feature = "axis")]
        self.require_axis_runtime_enabled()?;
        let mut ec = self
            .visible(id)
            .ok_or_else(|| anyhow::anyhow!("external collection '{id}' not found"))?;
        let collection_id = ec.spec.name.clone();

        ec.status = ExternalCollectionStatus::Building;
        self.registry.upsert(ec.clone());

        match self.build_inner(&ec).await {
            Ok(count) => {
                ec.status = ExternalCollectionStatus::Ready;
                ec.indexed_record_count = count as u64;
                ec.error = None;
                self.registry.upsert(ec);
                self.set_projection_state(&collection_id, |p| {
                    p.freshness_state = ProjectionFreshnessState::Fresh;
                })
                .await?;
                Ok(count)
            }
            Err(err) => {
                let msg = format!("{err:#}");
                ec.status = ExternalCollectionStatus::Failed;
                ec.error = Some(msg.clone());
                self.registry.upsert(ec);
                // Best-effort: mark the projection unusable so planners skip it.
                let _ = self
                    .set_projection_state(&collection_id, |p| {
                        p.freshness_state = ProjectionFreshnessState::RebuildRequired;
                    })
                    .await;
                Err(err)
            }
        }
    }

    /// Whether the external source changed since the last build — recomputes the
    /// source fingerprint and compares it to the stored snapshot id.
    pub fn is_stale(&self, id: &str) -> Result<bool> {
        let ec = self
            .visible(id)
            .ok_or_else(|| anyhow::anyhow!("external collection '{id}' not found"))?;
        let current = snapshot_fingerprint(&ec.spec.location)?;
        Ok(current != ec.snapshot_id)
    }

    /// On-demand staleness refresh: if the source changed, flip the projection to
    /// `RebuildRequired` (observable), rebuild the index in place, then publish
    /// `Fresh` with the new snapshot id. A no-op (`rebuilt = false`) when the
    /// source is unchanged. This is the remediation for the advisory
    /// `RebuildRequired` state; a future background sweep can call it.
    pub async fn refresh(&self, id: &str) -> Result<RefreshOutcome> {
        #[cfg(feature = "axis")]
        self.require_axis_runtime_enabled()?;
        let mut ec = self
            .visible(id)
            .ok_or_else(|| anyhow::anyhow!("external collection '{id}' not found"))?;
        let collection_id = ec.spec.name.clone();
        let current = snapshot_fingerprint(&ec.spec.location)?;
        if current == ec.snapshot_id {
            return Ok(RefreshOutcome {
                stale_detected: false,
                rebuilt: false,
                snapshot_id: ec.snapshot_id,
                indexed_record_count: ec.indexed_record_count,
            });
        }

        ec.status = ExternalCollectionStatus::Building;
        self.registry.upsert(ec.clone());
        // Observable: the index is known-stale until the rebuild completes.
        self.set_projection_state(&collection_id, |p| {
            p.freshness_state = ProjectionFreshnessState::RebuildRequired;
        })
        .await?;

        match self.build_inner(&ec).await {
            Ok(count) => {
                ec.snapshot_id = current.clone();
                ec.indexed_record_count = count as u64;
                ec.status = ExternalCollectionStatus::Ready;
                ec.error = None;
                self.registry.upsert(ec);
                let snap = current.clone();
                self.set_projection_state(&collection_id, move |p| {
                    p.freshness_state = ProjectionFreshnessState::Fresh;
                    p.source_range = Some(snap.clone());
                    p.last_included_position = Some(snap);
                })
                .await?;
                Ok(RefreshOutcome {
                    stale_detected: true,
                    rebuilt: true,
                    snapshot_id: current,
                    indexed_record_count: count as u64,
                })
            }
            Err(err) => {
                ec.status = ExternalCollectionStatus::Failed;
                ec.error = Some(format!("{err:#}"));
                self.registry.upsert(ec);
                Err(err)
            }
        }
    }

    /// Read + build + register-strategy. Separated so `build` can centralize
    /// status/projection transitions around it.
    #[cfg(feature = "axis")]
    async fn build_inner(&self, ec: &ExternalCollection) -> Result<usize> {
        self.require_axis_runtime_enabled()?;
        let records = read_external_records(&ec.spec)?;
        let count = records.len();
        let built = self
            .axis_manager
            .rebuild_and_swap_ivf_index(&ec.spec.name, &records)
            .await?;
        if !built {
            anyhow::bail!(
                "external collection '{}': too few vectors to build an IVF index ({} read)",
                ec.spec.name,
                count
            );
        }
        self.register_ivf_strategy(&ec.spec.name, ec.spec.dimension, count)
            .await?;

        // F5 Slice 3: build the BM25 inverted index over the text column (if
        // configured) so search can fuse lexical + vector results.
        self.build_fulltext_index(&ec.spec)?;
        Ok(count)
    }

    /// External collections index into AXIS, so building is unsupported in a
    /// PAX-exact-scan (`axis` off) build.
    #[cfg(not(feature = "axis"))]
    async fn build_inner(&self, _ec: &ExternalCollection) -> Result<usize> {
        anyhow::bail!("external collections require the `axis` feature (PAX-exact-scan build)")
    }

    /// Build (or rebuild) the per-collection BM25 index from the source text
    /// column. No-op when `text_column` is unset. Errors propagate to the build.
    fn build_fulltext_index(&self, spec: &ExternalCollectionSpec) -> Result<()> {
        if spec.text_column.is_none() {
            return Ok(());
        }
        let docs = read_external_text(spec)?;
        let mut index = FullTextIndex::new(TokenizerConfig::for_keyword_search());
        for (oid, text) in docs {
            if index.contains_document(&oid) {
                continue;
            }
            // Tokenizer/store errors on a single doc shouldn't fail the build.
            let _ = index.add_document(&oid, &text);
        }
        self.fulltext_indexes
            .write()
            .map_err(|_| anyhow::anyhow!("fulltext index lock poisoned"))?
            .insert(spec.name.clone(), index);
        Ok(())
    }

    /// Whether `collection_id` (by name) has a built BM25 index.
    pub fn has_fulltext_index(&self, collection_name: &str) -> bool {
        self.fulltext_indexes
            .read()
            .map(|m| m.contains_key(collection_name))
            .unwrap_or(false)
    }

    /// Idempotently build the BM25 fulltext index for `spec`. No-op when
    /// it already exists — survives restarts where the in-memory map
    /// dropped but the source data is still on disk.
    fn ensure_fulltext_index(&self, spec: &ExternalCollectionSpec) -> Result<()> {
        if self.has_fulltext_index(&spec.name) {
            return Ok(());
        }
        self.build_fulltext_index(spec)
    }

    /// Register the IVF routing strategy so `AxisManager::query` routes this
    #[cfg(feature = "axis")]
    /// collection to its (already-built) IVF index. The nlist/nprobe here are
    /// routing hints; the served index keeps the parameters it was built with.
    async fn register_ivf_strategy(
        &self,
        collection_id: &str,
        dimension: usize,
        n: usize,
    ) -> Result<()> {
        let nlist = ((n as f32).sqrt() as usize * 2).clamp(16, 256) as u32;
        let nprobe = (nlist / 2).max(1);
        let strategy = IndexSelectionStrategy {
            indexes: vec![IndexSpecification::new(
                Data::DenseVector { dimension },
                IndexAlgorithm::IVF {
                    nlist,
                    nprobe,
                    quantizer: None,
                },
            )],
            routing_rules: vec![],
        };
        self.axis_manager
            .update_collection_strategy(collection_id, strategy)
            .await
    }

    /// Vector-only search: returns scored hits with the **full records federated
    /// from the external source** (ProximaDB owns the index, not the records, so
    /// the text/metadata are fetched un-copied at retrieval time). Delegates the
    /// vector search to the same `AxisManager::query` path native collections use.
    pub async fn search(&self, id: &str, query: Vec<f32>, k: usize) -> Result<Vec<ExternalHit>> {
        self.hybrid_search(id, query, None, k).await
    }

    /// Hybrid search (F5 Slice 3): vector (IVF) results fused with BM25 lexical
    /// results via reciprocal-rank fusion (the default). When `text_query` is
    /// `None` or the collection has no BM25 index, this is vector-only.
    pub async fn hybrid_search(
        &self,
        id: &str,
        query: Vec<f32>,
        text_query: Option<String>,
        k: usize,
    ) -> Result<Vec<ExternalHit>> {
        self.hybrid_search_with_fusion(
            id,
            query,
            text_query,
            k,
            FusionStrategy::ReciprocalRank { k: HYBRID_RRF_K },
        )
        .await
    }

    /// Hybrid search with an explicit fusion strategy (e.g. weighted-linear).
    /// Same as [`Self::hybrid_search`] but the caller picks how BM25 and vector
    /// results are combined. The join key is the external row id; records for
    /// fused-in BM25-only hits are fetched lazily.
    pub async fn hybrid_search_with_fusion(
        &self,
        id: &str,
        query: Vec<f32>,
        text_query: Option<String>,
        k: usize,
        fusion: FusionStrategy,
    ) -> Result<Vec<ExternalHit>> {
        let ec = self
            .visible(id)
            .ok_or_else(|| anyhow::anyhow!("external collection '{id}' not found"))?;
        if !ec.is_ready() {
            anyhow::bail!(
                "external collection '{}' is not ready (status {:?}); build it first",
                ec.spec.name,
                ec.status
            );
        }

        // Oversample each side so fusion has overlap to work with.
        // TD-SECSCAN-1 (sec review): saturate the oversample product and keep
        // the pool within cap * factor (k itself is capped at the boundary).
        let pool = hybrid_pool_size(k);
        let vector_hits = self.vector_hits(&ec, query, pool).await?;

        // Lazily (re)build the BM25 index when a text query is supplied and the
        // in-memory index is absent — e.g. after a restart, where the registry +
        // the persisted IVF survive but the in-memory BM25 does not. Rebuilt from
        // the (un-copied) source on the first hybrid query, mirroring the IVF
        // load-on-demand path.
        if text_query
            .as_deref()
            .map(|t| !t.trim().is_empty())
            .unwrap_or(false)
        {
            self.ensure_fulltext_index(&ec.spec)?;
        }

        // Fuse only with a non-empty text query AND a built BM25 index; otherwise
        // return vector-only (current behaviour).
        let text_query = match text_query {
            Some(t) if !t.trim().is_empty() && self.has_fulltext_index(&ec.spec.name) => t,
            _ => {
                let mut hits = vector_hits;
                hits.truncate(k);
                return Ok(hits);
            }
        };

        // BM25 side.
        let bm25_results: Vec<BM25Result> = {
            let guard = self
                .fulltext_indexes
                .read()
                .map_err(|_| anyhow::anyhow!("fulltext index lock poisoned"))?;
            let index = guard
                .get(&ec.spec.name)
                .ok_or_else(|| anyhow::anyhow!("BM25 index missing for '{}'", ec.spec.name))?;
            index
                .search_with_mode(&text_query, pool, LexicalQueryMode::NaturalLanguage)?
                .into_iter()
                .map(|r| BM25Result {
                    doc_id: r.doc_id,
                    score: r.score,
                    highlights: None,
                    metadata: HashMap::new(),
                })
                .collect()
        };

        // Vector side (RRF is rank-based, so the f32→f64 score cast is fine).
        let vector_results: Vec<VectorResult> = vector_hits
            .iter()
            .map(|h| VectorResult {
                doc_id: h.id.clone(),
                score: h.score as f64,
                distance: 0.0,
                metadata: HashMap::new(),
            })
            .collect();

        let fused = HybridFusionEngine::new(fusion)
            .with_top_k(k)
            .fuse(bm25_results, vector_results)
            .map_err(|e| anyhow::anyhow!("hybrid fusion failed: {e}"))?;

        // Assemble records: reuse the vector hits' records; fetch any BM25-only ids.
        let mut by_id: std::collections::HashMap<String, ProximaRecord> =
            vector_hits.into_iter().map(|h| (h.id, h.record)).collect();
        let missing: Vec<String> = fused
            .iter()
            .filter(|f| !by_id.contains_key(&f.doc_id))
            .map(|f| f.doc_id.clone())
            .collect();
        if !missing.is_empty() {
            for r in read_records_by_ids(&ec.spec, &missing)? {
                by_id.insert(r.oid.clone(), r);
            }
        }

        Ok(fused
            .into_iter()
            .map(|f| {
                let record = by_id.remove(&f.doc_id).unwrap_or_else(|| ProximaRecord {
                    oid: f.doc_id.clone(),
                    ..Default::default()
                });
                ExternalHit {
                    id: f.doc_id,
                    score: f.fused_score as f32,
                    record,
                }
            })
            .collect())
    }

    /// Run the IVF vector search and federate full records — the shared vector
    /// half of `search`/`hybrid_search`.
    #[cfg(feature = "axis")]
    async fn vector_hits(
        &self,
        ec: &ExternalCollection,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<ExternalHit>> {
        self.require_axis_runtime_enabled()?;
        let q = AxisHybridQuery {
            collection_id: ec.spec.name.clone(),
            vector_query: Some(VectorQuery::Dense {
                vector: query,
                similarity_threshold: 0.0,
            }),
            top_k: k,
            ..AxisHybridQuery::default()
        };
        let scored: Vec<(String, f32)> = self
            .axis_manager
            .query(q)
            .await?
            .results
            .into_iter()
            .map(|r| (r.vector_id, r.similarity))
            .collect();

        let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
        let mut by_id: std::collections::HashMap<String, ProximaRecord> =
            read_records_by_ids(&ec.spec, &ids)?
                .into_iter()
                .map(|r| (r.oid.clone(), r))
                .collect();

        Ok(scored
            .into_iter()
            .map(|(id, score)| {
                let record = by_id.remove(&id).unwrap_or_else(|| ProximaRecord {
                    oid: id.clone(),
                    ..Default::default()
                });
                ExternalHit { id, score, record }
            })
            .collect())
    }

    /// External-collection vector search routes through `AxisManager::query`;
    /// unsupported in a PAX-exact-scan (`axis` off) build.
    #[cfg(not(feature = "axis"))]
    async fn vector_hits(
        &self,
        _ec: &ExternalCollection,
        _query: Vec<f32>,
        _k: usize,
    ) -> Result<Vec<ExternalHit>> {
        anyhow::bail!("external collections require the `axis` feature (PAX-exact-scan build)")
    }

    /// Look up a registry record by id.
    pub fn get(&self, id: &str) -> Option<ExternalCollection> {
        self.visible(id)
    }

    /// The registered external collections the scope in force may see
    /// (newest first): a request sees only its tenant's; internal execution
    /// sees all; no scope sees none.
    pub fn list(&self) -> Vec<ExternalCollection> {
        let all = self.registry.list_all();
        match crate::query::request_scope::current_scope() {
            Some(crate::query::request_scope::RequestScope::Internal(_)) => all,
            Some(crate::query::request_scope::RequestScope::Tenant(tenant)) => all
                .into_iter()
                .filter(|ec| ec.visible_to(&tenant))
                .collect(),
            None => Vec::new(),
        }
    }

    /// The external collection `id` when the scope in force may see it.
    fn visible(&self, id: &str) -> Option<ExternalCollection> {
        let ec = self.registry.get(id)?;
        match crate::query::request_scope::current_scope() {
            Some(crate::query::request_scope::RequestScope::Internal(_)) => Some(ec),
            Some(crate::query::request_scope::RequestScope::Tenant(tenant)) => {
                ec.visible_to(&tenant).then_some(ec)
            }
            None => None,
        }
    }

    /// Read the external-index projection (for EXPLAIN / route-health / tests).
    pub async fn projection(&self, collection_id: &str) -> Result<Option<CatalogProjection>> {
        let (catalog, identifier) = self
            .catalog_manager
            .resolve_table_scoped(
                collection_id,
                crate::query::request_scope::resolution_tenant(collection_id)?.as_deref(),
            )
            .await?;
        if !catalog.table_exists(&identifier).await? {
            return Ok(None);
        }
        let schema = catalog.get_table(&identifier).await?;
        Ok(schema
            .projections
            .into_iter()
            .find(|p| p.name == EXTERNAL_INDEX_PROJECTION))
    }

    /// Mutate the external-index projection in place (TD-CATREPLACE-1: no
    /// drop window).
    async fn set_projection_state<F>(&self, collection_id: &str, mutate: F) -> Result<()>
    where
        F: FnOnce(&mut CatalogProjection),
    {
        let (catalog, identifier) = self
            .catalog_manager
            .resolve_table_scoped(
                collection_id,
                crate::query::request_scope::resolution_tenant(collection_id)?.as_deref(),
            )
            .await?;
        if !catalog.table_exists(&identifier).await? {
            anyhow::bail!("external collection '{collection_id}' is not cataloged");
        }
        let mut schema = catalog.get_table(&identifier).await?;
        match schema
            .projections
            .iter_mut()
            .find(|p| p.name == EXTERNAL_INDEX_PROJECTION)
        {
            Some(proj) => mutate(proj),
            None => {
                let mut proj = CatalogProjection::rebuildable(
                    EXTERNAL_INDEX_PROJECTION,
                    CatalogProjectionKind::VectorAnn,
                    collection_id,
                );
                mutate(&mut proj);
                schema.projections.push(proj);
            }
        }
        // TD-CATREPLACE-1: replace in place — no drop window.
        catalog.replace_table(&identifier, schema).await?;
        Ok(())
    }
}

/// Per-side oversample pool for hybrid fusion: `k * HYBRID_POOL_FACTOR`,
/// saturating, bounded by `cap * HYBRID_POOL_FACTOR` (TD-SECSCAN-1 §5; `k`
/// itself is capped at the request boundary).
fn hybrid_pool_size(k: usize) -> usize {
    let cap = proximadb_search_types::top_k_limit::max_top_k();
    k.saturating_mul(HYBRID_POOL_FACTOR)
        .max(k)
        .min(cap.saturating_mul(HYBRID_POOL_FACTOR))
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    /// TD-SECSCAN-1 §5 follow-up: the pool bound follows the CONFIGURED cap.
    /// With `PROXIMADB_SEARCH_MAX_TOP_K` raised above the 10,000 default, a
    /// request at the override keeps its full oversample instead of being
    /// clamped to the default-derived bound.
    #[test]
    fn hybrid_pool_honours_configured_cap_override() {
        crate::services::top_k_env_test_support::with_max_top_k_override("50000", || {
            assert_eq!(hybrid_pool_size(50_000), 50_000 * HYBRID_POOL_FACTOR);
            assert_eq!(hybrid_pool_size(usize::MAX), 50_000 * HYBRID_POOL_FACTOR);
            assert_eq!(hybrid_pool_size(3), 3 * HYBRID_POOL_FACTOR);
        });
    }
}

#[cfg(all(test, feature = "axis"))]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, StringBuilder};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    async fn catalog_manager_with_default() -> Arc<CatalogManager> {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_extsvc_cat_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("default", &format!("file://{}", tmp.display()))
            .await
            .unwrap();
        manager.set_default_catalog("default").await.unwrap();
        manager
    }

    async fn axis_manager() -> Arc<AxisManager> {
        Arc::new(
            AxisManager::new(crate::index::AxisConfig::default())
                .await
                .unwrap(),
        )
    }

    /// Write `n` vectors with **unique directions** (one-hot at position `i`,
    /// so under cosine each row's nearest neighbor is itself). Requires
    /// `dim >= n`. Rebuild needs >= 16 vectors to cluster.

    /// Allowlist a fixture's own directory for this test process.
    ///
    /// `list_parquet_files` is now fail-closed behind
    /// `PROXIMADB_EXTERNAL_TABLE_ROOTS` (TD-SECSCAN-1: the location used to go
    /// straight to `fs::metadata` / `fs::read_dir`). Unset rejects everything,
    /// which is the shipped default, so a test that reads a real fixture must
    /// opt in. Called from the fixture WRITERS, so every test that creates one
    /// is covered without 14 separate call sites.
    ///
    /// `set_var` is `unsafe` in edition 2024 and racy across threads; this is
    /// safe under nextest, which gives each test its own process (the repo's
    /// mandated runner). A bare path, not `file://`, because these tests pass
    /// bare locations and the allowlist match is textual.
    fn allowlist_fixture_dir(path: &std::path::Path) {
        let dir = path
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/".to_string());
        unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &dir) };
    }

    fn write_parquet_fixture(
        path: &std::path::Path,
        n: usize,
        dim: usize,
    ) -> Vec<(String, Vec<f32>)> {
        allowlist_fixture_dir(path);
        assert!(dim >= n, "one-hot fixture requires dim >= n");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        let mut id_b = StringBuilder::new();
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        let mut expect = Vec::new();
        for i in 0..n {
            let id = format!("row-{i}");
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0 + (i as f32) * 0.001;
            id_b.append_value(&id);
            for x in &v {
                vec_b.values().append_value(*x);
            }
            vec_b.append(true);
            expect.push((id, v));
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_b.finish()), Arc::new(vec_b.finish())],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        expect
    }

    fn fixture_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "proximadb_extsvc_{}.parquet",
            uuid::Uuid::new_v4().simple()
        ))
    }

    /// Like `write_parquet_fixture` but adds `text: Utf8` + `year: Int64` columns
    /// so federated-fetch props can be asserted. One-hot directions, `dim >= n`.
    fn write_meta_fixture(path: &std::path::Path, n: usize, dim: usize) {
        allowlist_fixture_dir(path);
        use arrow_array::builder::{Int64Builder, StringBuilder};
        assert!(dim >= n, "one-hot fixture requires dim >= n");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("year", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        let mut id_b = StringBuilder::new();
        let mut text_b = StringBuilder::new();
        let mut year_b = Int64Builder::new();
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        for i in 0..n {
            id_b.append_value(format!("row-{i}"));
            text_b.append_value(format!("text-{i}"));
            year_b.append_value(2000 + i as i64);
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0 + (i as f32) * 0.001;
            for x in &v {
                vec_b.values().append_value(*x);
            }
            vec_b.append(true);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_b.finish()),
                Arc::new(text_b.finish()),
                Arc::new(year_b.finish()),
                Arc::new(vec_b.finish()),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn register_catalogs_federated_read_and_external_snapshot() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_parquet_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat.clone(),
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);

            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            let ec = svc.register(spec).await.unwrap();
            assert_eq!(ec.status, ExternalCollectionStatus::Registered);

            // Catalog shows FederatedRead authority + external location.
            let (catalog, identifier) = cat.resolve_table("ext_docs").await.unwrap();
            let schema = catalog.get_table(&identifier).await.unwrap();
            let layout = &schema.storage_layouts[0];
            assert_eq!(
                layout.authority,
                proximadb_catalog::CatalogAuthorityMode::FederatedRead
            );
            assert_eq!(layout.location.as_deref(), Some(path.to_str().unwrap()));

            // Projection registered as ExternalSnapshotRegistered with snapshot lineage.
            let proj = svc.projection("ext_docs").await.unwrap().unwrap();
            assert_eq!(
                proj.freshness_state,
                ProjectionFreshnessState::ExternalSnapshotRegistered
            );
            assert_eq!(proj.source_range.as_deref(), Some(ec.snapshot_id.as_str()));

            // Re-register is rejected.
            let dup = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            assert!(svc.register(dup).await.is_err());

            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn runtime_disabled_rejects_build_before_reading_the_source() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_parquet_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let axis = axis_manager().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis.clone(),
            );
            let spec = ExternalCollectionSpec::parquet(
                "disabled_ext",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            let ec = svc.register(spec).await.unwrap();

            std::fs::remove_file(&path).unwrap();
            let error = svc.build(&ec.id).await.unwrap_err();
            assert!(
                error.to_string().contains("enable_axis_indexes=true"),
                "runtime policy must fail before the missing source is read: {error:#}"
            );
            assert!(!axis.has_served_index("disabled_ext").await);
        })
        .await
    }

    #[tokio::test]
    async fn build_in_place_makes_index_queryable_and_fresh() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            let expect = write_parquet_fixture(&path, 32, 32);
            let cat = catalog_manager_with_default().await;
            let axis = axis_manager().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat.clone(),
                axis.clone(),
            )
            .with_axis_runtime_enabled(true);

            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                32,
            );
            let ec = svc.register(spec).await.unwrap();

            let count = svc.build(&ec.id).await.unwrap();
            assert_eq!(count, 32);
            assert!(axis.has_ivf_index("ext_docs").await);

            // Status Ready + projection Fresh.
            let got = svc.get(&ec.id).unwrap();
            assert_eq!(got.status, ExternalCollectionStatus::Ready);
            assert_eq!(got.indexed_record_count, 32);
            let proj = svc.projection("ext_docs").await.unwrap().unwrap();
            assert_eq!(proj.freshness_state, ProjectionFreshnessState::Fresh);

            // Search returns the exact nearest external row for a row's own vector.
            let (qid, qvec) = &expect[5];
            let hits = svc.search(&ec.id, qvec.clone(), 3).await.unwrap();
            assert!(!hits.is_empty(), "external index must be queryable");
            assert_eq!(&hits[0].id, qid, "top-1 must be the exact external row");

            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn search_before_build_is_rejected() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_parquet_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            let ec = svc.register(spec).await.unwrap();
            assert!(svc.search(&ec.id, vec![0.0; 20], 3).await.is_err());
            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn search_returns_federated_props_from_source() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_meta_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            let ec = svc.register(spec).await.unwrap();
            svc.build(&ec.id).await.unwrap();

            // Query row-3's own one-hot vector → top-1 is row-3 with its source props.
            let mut q = vec![0.0f32; 20];
            q[3] = 1.0;
            let hits = svc.search(&ec.id, q, 3).await.unwrap();
            assert_eq!(hits[0].id, "row-3");
            let props = &hits[0].record.props;
            match &props["text"] {
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(s),
                ) => assert_eq!(s, "text-3"),
                other => panic!("text prop wrong: {other:?}"),
            }
            match &props["year"] {
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Int64(y),
                ) => assert_eq!(*y, 2003),
                other => panic!("year prop wrong: {other:?}"),
            }
            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn refresh_rebuilds_only_when_source_changed() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_meta_fixture(&path, 20, 24);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                24,
            );
            let ec = svc.register(spec).await.unwrap();
            svc.build(&ec.id).await.unwrap();
            let snap0 = svc.get(&ec.id).unwrap().snapshot_id;

            // Unchanged source → not stale, refresh is a no-op.
            assert!(!svc.is_stale(&ec.id).unwrap());
            let r0 = svc.refresh(&ec.id).await.unwrap();
            assert!(!r0.stale_detected && !r0.rebuilt);
            assert_eq!(r0.snapshot_id, snap0);

            // Mutate the source (more rows → different fingerprint).
            write_meta_fixture(&path, 24, 24);
            assert!(svc.is_stale(&ec.id).unwrap());
            let r1 = svc.refresh(&ec.id).await.unwrap();
            assert!(
                r1.stale_detected && r1.rebuilt,
                "changed source must rebuild"
            );
            assert_eq!(r1.indexed_record_count, 24);
            assert_ne!(r1.snapshot_id, snap0, "snapshot id must advance");

            let got = svc.get(&ec.id).unwrap();
            assert_eq!(got.status, ExternalCollectionStatus::Ready);
            assert_eq!(got.indexed_record_count, 24);
            let proj = svc.projection("ext_docs").await.unwrap().unwrap();
            assert_eq!(proj.freshness_state, ProjectionFreshnessState::Fresh);
            assert_eq!(proj.source_range.as_deref(), Some(r1.snapshot_id.as_str()));
            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    // ─── F5 Slice 3: BM25 + hybrid ──────────────────────────────────────────

    #[tokio::test]
    async fn build_populates_bm25_index_only_when_text_column_set() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_meta_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;

            // With text_column → BM25 index built.
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat.clone(),
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            )
            .with_text_column("text");
            let ec = svc.register(spec).await.unwrap();
            svc.build(&ec.id).await.unwrap();
            assert!(svc.has_fulltext_index("ext_docs"));

            // Without text_column → no BM25 index.
            let svc2 = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                catalog_manager_with_default().await,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec2 = ExternalCollectionSpec::parquet(
                "ext_plain",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            );
            let ec2 = svc2.register(spec2).await.unwrap();
            svc2.build(&ec2.id).await.unwrap();
            assert!(!svc2.has_fulltext_index("ext_plain"));

            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn hybrid_search_fuses_lexical_match_with_vector() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            // row-i: one-hot vector at i, text "text-i".
            write_meta_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            )
            .with_text_column("text");
            let ec = svc.register(spec).await.unwrap();
            svc.build(&ec.id).await.unwrap();

            // Vector query points at row-3; lexical query is row-7's own text.
            let mut qv = vec![0.0f32; 20];
            qv[3] = 1.0;

            // Vector-only: nearest is row-3.
            let vonly = svc.search(&ec.id, qv.clone(), 5).await.unwrap();
            assert_eq!(
                vonly[0].id, "row-3",
                "vector-only top-1 is the nearest vector"
            );

            // Hybrid: fusion pulls in the lexical-only match (row-7) alongside row-3.
            let hybrid = svc
                .hybrid_search(&ec.id, qv, Some("text-7".to_string()), 5)
                .await
                .unwrap();
            let ids: Vec<&str> = hybrid.iter().map(|h| h.id.as_str()).collect();
            assert!(
                ids.contains(&"row-7"),
                "BM25 brought the lexical match into fused results: {ids:?}"
            );
            assert!(
                ids.contains(&"row-3"),
                "vector match retained in fused results: {ids:?}"
            );
            // Hits still carry federated props.
            let row7 = hybrid.iter().find(|h| h.id == "row-7").unwrap();
            assert!(row7.record.props.contains_key("text"));

            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn hybrid_search_lazily_rebuilds_bm25_after_restart() {
        crate::query::request_scope::with_internal_scope("test", async move {
            let path = fixture_path();
            write_meta_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let registry = Arc::new(ExternalCollectionRegistry::new());
            let axis = axis_manager().await;

            // Build with the first service (BM25 populated in its in-memory map).
            let svc1 = ExternalCollectionService::new(registry.clone(), cat.clone(), axis.clone())
                .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            )
            .with_text_column("text");
            let ec = svc1.register(spec).await.unwrap();
            svc1.build(&ec.id).await.unwrap();

            // Simulate a restart: a fresh service shares the durable registry + the
            // (still-resident) IVF index, but starts with an EMPTY BM25 map.
            let svc2 =
                ExternalCollectionService::new(registry, cat, axis).with_axis_runtime_enabled(true);
            assert!(
                !svc2.has_fulltext_index("ext_docs"),
                "fresh service starts without BM25"
            );

            let mut qv = vec![0.0f32; 20];
            qv[3] = 1.0;
            let hybrid = svc2
                .hybrid_search(&ec.id, qv, Some("text-7".to_string()), 5)
                .await
                .unwrap();

            // The first hybrid query lazily rebuilt the BM25 index from the source,
            // so fusion still surfaces the lexical-only match.
            assert!(
                svc2.has_fulltext_index("ext_docs"),
                "first hybrid query rebuilt the BM25 index"
            );
            let ids: Vec<&str> = hybrid.iter().map(|h| h.id.as_str()).collect();
            assert!(
                ids.contains(&"row-7"),
                "lazily-rebuilt BM25 fused the lexical match: {ids:?}"
            );
            assert!(ids.contains(&"row-3"), "vector match retained: {ids:?}");

            let _ = std::fs::remove_file(&path);
        })
        .await
    }

    #[tokio::test]
    async fn hybrid_search_with_weighted_fusion_returns_fused_hits() {
        crate::query::request_scope::with_internal_scope("test", async move {
            use crate::core::search::hybrid::FusionStrategy;
            let path = fixture_path();
            write_meta_fixture(&path, 20, 20);
            let cat = catalog_manager_with_default().await;
            let svc = ExternalCollectionService::new(
                Arc::new(ExternalCollectionRegistry::new()),
                cat,
                axis_manager().await,
            )
            .with_axis_runtime_enabled(true);
            let spec = ExternalCollectionSpec::parquet(
                "ext_docs",
                path.to_str().unwrap(),
                "id",
                "vector",
                20,
            )
            .with_text_column("text");
            let ec = svc.register(spec).await.unwrap();
            svc.build(&ec.id).await.unwrap();

            let mut qv = vec![0.0f32; 20];
            qv[3] = 1.0;
            // Weighted-linear (50/50) fusion also surfaces the lexical match.
            let hits = svc
                .hybrid_search_with_fusion(
                    &ec.id,
                    qv,
                    Some("text-7".to_string()),
                    5,
                    FusionStrategy::WeightedLinear {
                        alpha: 0.5,
                        bm25_normalize: true,
                        vector_normalize: true,
                    },
                )
                .await
                .unwrap();
            let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            assert!(
                ids.contains(&"row-7"),
                "weighted fusion surfaces the lexical match: {ids:?}"
            );
            assert!(ids.contains(&"row-3"), "vector match retained: {ids:?}");

            let _ = std::fs::remove_file(&path);
        })
        .await
    }
}
