//! End-to-end test for the R-7c production `RankServices` wiring.
//!
//! Exercises the full path that production servers take:
//!
//! 1. Construct `CanonicalWalRankProfileStore` + `RankServices` (with
//!    Prometheus metrics registered against a private `Registry` so the test
//!    is hermetic).
//! 2. Install a rank profile via the REST `install_rank_profile_inner`
//!    dispatcher — same code path the HTTP route lowers into.
//! 3. Issue a rank search through `RankServices` via the same entry point
//!    REST `/api/v1/rank/search`, gRPC `RankServiceImpl::rank_search`, and
//!    Arrow Flight `rank_features_export` all share
//!    (`handle_rank_search_with_metrics`).
//! 4. Assert the response shape (5 hits, phase tag, profile name round-trips).
//! 5. Scrape the rank-metric registry and assert the spec §4.10 metrics
//!    actually emitted observations.
//!
//! This validates the wire-up of: REST install → catalog WAL → registry
//! recovery path → live `RankServices.handle_rank_search` → spec metrics.
//! The pgwire SQL `RERANK(...)` form goes through the same
//! `handle_rank_search_with_metrics` entry point per the existing
//! production wiring, so this test covers the metric / install plumbing
//! end-to-end without the cost of binding a TCP socket.

use std::sync::Arc;

use prometheus::{Encoder, Registry, TextEncoder};
use proximadb::core::search::hybrid::FusionStrategy;
use proximadb::network::rest::canonical::rank::{
    CandidateProvider, HybridCoordinatorAdapter, HybridSearchBackend, MockRangeCandidateProvider,
    RankSearchRequest, RankServices,
};
use proximadb::network::rest::canonical::rank_profile::{
    InstallRankProfileRequest, install_rank_profile_inner,
};
use proximadb::observability::rank_metrics::RankPipelineMetrics;
use proximadb::services::record_store::TableWalAppender;
use proximadb::services::{CanonicalWalRankProfileStore, MemoryTableWalAppender, RankProfileStore};
use proximadb_rank_core::RankResult;

const PROFILE_NAME: &str = "test_ce";
const PROFILE_SPEC: &str = r#"
match_features = ["bm25(\"body\")"]

[first_phase]
expression = "bm25(\"body\")"
heap_size = 50
"#;

/// No-op backend for the hybrid coordinator. The test exercises the
/// candidate provider path (`MockRangeCandidateProvider`) rather than the
/// hybrid retrieval path, so this backend just satisfies the type contract.
struct NoopBackend;

#[async_trait::async_trait]
impl HybridSearchBackend for NoopBackend {
    async fn scope_collection(&self, _collection: &str, _tenant: &str) -> RankResult<()> {
        Ok(())
    }
    async fn bm25_search(
        &self,
        _collection: &str,
        _query: &str,
    ) -> RankResult<Vec<proximadb::core::search::hybrid::BM25Result>> {
        Ok(Vec::new())
    }
    async fn vector_search(
        &self,
        _collection: &str,
        _vector: &[f32],
    ) -> RankResult<Vec<proximadb::core::search::hybrid::VectorResult>> {
        Ok(Vec::new())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rest_install_then_search_emits_spec_metrics() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build the production rank pipeline against a hermetic registry. ──
    let registry = Registry::new();
    let metrics = Arc::new(RankPipelineMetrics::register(&registry)?);

    let appender: Arc<dyn TableWalAppender> = Arc::new(MemoryTableWalAppender::new());
    let store: Arc<dyn RankProfileStore> = Arc::new(CanonicalWalRankProfileStore::new(appender));

    // The candidate provider returns a fixed range of `DocHandle`s so we get
    // a deterministic 5-row result for the assertion. The HybridCoordinator
    // adapter is constructed but never invoked by the candidate-provider
    // path; we wire it to mirror the production shape.
    let candidates: Arc<dyn CandidateProvider> = Arc::new(MockRangeCandidateProvider { count: 5 });
    let _coordinator = HybridCoordinatorAdapter::new(
        FusionStrategy::ReciprocalRank { k: 60 },
        Arc::new(NoopBackend),
    );
    let services = Arc::new(RankServices::new(candidates).with_metrics(metrics.clone()));

    // ── 2. Install a profile via the REST dispatcher. ────────────────────────
    let dto = install_rank_profile_inner(
        store.as_ref(),
        services.as_ref(),
        InstallRankProfileRequest {
            name: PROFILE_NAME.to_string(),
            tenant: None,
            spec: PROFILE_SPEC.to_string(),
        },
    )
    .await
    .expect("REST install must succeed");
    assert_eq!(dto.name, PROFILE_NAME);
    assert_eq!(dto.version, 1);
    assert_eq!(dto.spec, PROFILE_SPEC);

    // Catalog + live registry both observe the install.
    assert!(
        store.get(PROFILE_NAME).await?.is_some(),
        "profile must persist in the durable catalog"
    );
    assert!(
        services.profile_registry.get(PROFILE_NAME).is_some(),
        "profile must be installed in the live registry"
    );

    // ── 3. Issue a rank search through the same code path REST + gRPC + Arrow
    //    Flight share at the production seam. ─────────────────────────────────
    let request = RankSearchRequest {
        scope: proximadb::network::rest::canonical::rank::RankScope::Internal,
        collection: "docs".to_string(),
        query_vector: vec![0.1, 0.2],
        query_text: Some("body content".to_string()),
        k: 5,
        rank_profile: Some(PROFILE_NAME.to_string()),
        rank_overrides: None,
    };
    // Same plumbing the REST `/api/v1/rank/search` dispatcher uses: unpack
    // the registry + candidate provider + blueprint factory + per-profile
    // scorer + metrics out of `RankServices` and call into
    // `handle_rank_search_with_metrics` (the shared production entry point
    // for REST, gRPC, and Arrow Flight).
    let second_phase_scorer = request
        .rank_profile
        .as_deref()
        .and_then(|name| services.second_phase_scorer(name));
    let response = proximadb::network::rest::canonical::rank::handle_rank_search_with_metrics(
        request,
        services.profile_registry.as_ref(),
        services.candidate_provider.as_ref(),
        services.blueprint_factory.clone(),
        second_phase_scorer,
        services.rerank_config.clone(),
        services.metrics.clone(),
    )
    .await
    .expect("rank search must succeed against the installed profile");

    // ── 4. Validate the response shape. ──────────────────────────────────────
    assert_eq!(
        response.hits.len(),
        5,
        "expected 5 hits from MockRangeCandidateProvider; got {}",
        response.hits.len()
    );
    assert_eq!(response.rank_profile.as_deref(), Some(PROFILE_NAME));
    assert_eq!(response.rank_profile_version, Some(1));

    // ── 5. Scrape the registry; assert spec §4.10 metrics actually emit. ────
    let mut buf = Vec::new();
    TextEncoder::new().encode(&registry.gather(), &mut buf)?;
    let scrape = String::from_utf8(buf)?;

    // Profile install bumped `proximadb_rank_profile_reload_total{outcome="ok"}`.
    assert!(
        scrape.contains("proximadb_rank_profile_reload_total"),
        "profile reload counter must appear in scrape:\n{scrape}"
    );
    assert!(
        scrape.contains(&format!("profile=\"{PROFILE_NAME}\"")),
        "reload counter must carry the profile label:\n{scrape}"
    );

    // The first-phase scoring path emits per-phase latency for the installed
    // profile (the bm25 feature path captures `proximadb_rank_feature_*` only
    // when the per-doc feature is actually evaluated).
    assert!(
        scrape.contains("proximadb_rank_phase_latency_us"),
        "phase latency histogram must appear in scrape:\n{scrape}"
    );

    Ok(())
}
