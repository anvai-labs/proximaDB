//! Dispatchers for the durable rank-profile catalog (commit 5/5 of the R-7c
//! production wiring). These intended v2 routes are not mounted yet and are
//! therefore not part of the live REST surface.
//!
//! `POST /api/v2/rank/profiles` — install or replace a profile.
//! `GET  /api/v2/rank/profiles/{name}` — fetch a profile by name.
//! `DELETE /api/v2/rank/profiles/{name}` — remove a profile.
//!
//! The dispatchers are plain async functions (no axum extractors) so the
//! REST router glue is decoupled from the axum version in the dep graph —
//! matches the pattern used by `rank::rank_search_dispatch`.

use crate::errors::{ApiError, ApiResult};
use crate::network::rest::canonical::handlers::AppState;
use serde::{Deserialize, Serialize};

/// Body of the intended `POST /api/v2/rank/profiles` route. The `spec` field carries the raw
/// TOML body that `RankProfileStore::install` persists and the live
/// `RankServices` registry compiles via `parse_single` + `CompiledRankProfile`.
#[derive(Debug, Clone, Deserialize)]
pub struct InstallRankProfileRequest {
    /// Profile name (catalog key).
    pub name: String,
    /// Optional tenant scope. `None` = single-tenant / unscoped.
    #[serde(default)]
    pub tenant: Option<String>,
    /// TOML body of the profile spec.
    pub spec: String,
}

/// Wire-form view of a stored rank profile. Mirrors `StoredRankProfile`
/// but lives in the REST surface so the catalog crate's internal shape
/// stays free to evolve.
#[derive(Debug, Clone, Serialize)]
pub struct RankProfileDto {
    pub name: String,
    pub version: u32,
    pub tenant: Option<String>,
    pub spec: String,
    pub created_at_ms: i64,
}

impl From<crate::services::StoredRankProfile> for RankProfileDto {
    fn from(profile: crate::services::StoredRankProfile) -> Self {
        Self {
            name: profile.name,
            version: profile.version,
            tenant: profile.tenant,
            spec: profile.spec_toml,
            created_at_ms: profile.created_at_ms,
        }
    }
}

/// Install or replace a rank profile end-to-end: parse + compile against the
/// live `BlueprintFactory`, then persist to the canonical-WAL-backed store and
/// install into the live `ProfileRegistry`. On compile failure nothing is
/// persisted and `proximadb_rank_profile_reload_total{outcome="error"}` is
/// bumped so dashboards alert on bad ops actions.
pub async fn install_rank_profile_dispatch(
    app_state: AppState,
    req: InstallRankProfileRequest,
) -> ApiResult<RankProfileDto> {
    let store = app_state.rank_profile_store.as_ref().ok_or_else(|| {
        ApiError::NotImplemented(
            "rank-profile catalog not configured — server started without RankProfileStore"
                .to_string(),
        )
    })?;
    let services = app_state.rank_services.as_ref().ok_or_else(|| {
        ApiError::NotImplemented(
            "rank-services registry not configured — server started without RankServices"
                .to_string(),
        )
    })?;
    install_rank_profile_inner(store.as_ref(), services.as_ref(), req).await
}

/// Fetch a profile by name. Returns 404 when the profile is not installed.
pub async fn get_rank_profile_dispatch(
    app_state: AppState,
    name: String,
) -> ApiResult<RankProfileDto> {
    let store = app_state.rank_profile_store.as_ref().ok_or_else(|| {
        ApiError::NotImplemented(
            "rank-profile catalog not configured — server started without RankProfileStore"
                .to_string(),
        )
    })?;
    get_rank_profile_inner(store.as_ref(), name).await
}

/// Remove a profile from the durable catalog and the live registry.
/// `if_exists=true` returns 204 when the profile is already absent; otherwise
/// the request fails with 404.
pub async fn remove_rank_profile_dispatch(
    app_state: AppState,
    name: String,
    if_exists: bool,
) -> ApiResult<()> {
    let store = app_state.rank_profile_store.as_ref().ok_or_else(|| {
        ApiError::NotImplemented(
            "rank-profile catalog not configured — server started without RankProfileStore"
                .to_string(),
        )
    })?;
    let services = app_state.rank_services.as_deref();
    remove_rank_profile_inner(store.as_ref(), services, name, if_exists).await
}

// -------------------------------------------------------------------------
// Inner dispatchers — take dependencies directly so tests don't need a full
// `AppState`. The HTTP/AppState wrappers above are thin and just unpack.
// -------------------------------------------------------------------------

pub async fn install_rank_profile_inner(
    store: &dyn crate::services::RankProfileStore,
    services: &crate::network::rest::canonical::rank::RankServices,
    req: InstallRankProfileRequest,
) -> ApiResult<RankProfileDto> {
    use proximadb_rank_profile::{CompiledRankProfile, dsl::parse_single};

    if req.name.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "rank profile name must be non-empty".to_string(),
        ));
    }
    if req.spec.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "rank profile spec body must be non-empty".to_string(),
        ));
    }

    let mut spec = parse_single(&req.name, &req.spec)
        .map_err(|e| ApiError::InvalidArgument(format!("invalid rank profile spec: {e}")))?;

    // Validate up-front (no persist on validation failure). `validate` is the
    // same precondition `CompiledRankProfile::compile` runs internally, just
    // exposed separately so we can guard the catalog write without having to
    // build the compiled artifact twice.
    proximadb_rank_profile::validator::validate(&spec, &services.blueprint_factory).map_err(
        |e| {
            services.record_profile_reload_error(&req.name);
            ApiError::InvalidArgument(format!("rank profile compile failed: {e}"))
        },
    )?;

    // Persist now that validation passed. The store assigns the monotonic
    // version that the compiled spec inherits, so REST / gRPC / Arrow Flight
    // rank responses can attribute hits back to a stable profile version.
    let stored = store
        .install(&req.name, req.spec.clone(), req.tenant.clone(), None)
        .await
        .map_err(|e| ApiError::Internal(format!("rank profile catalog write failed: {e}")))?;
    spec.version = stored.version;

    let compiled =
        CompiledRankProfile::compile(spec, services.blueprint_factory.clone()).map_err(|e| {
            ApiError::Internal(format!(
                "rank profile compile after validate succeeded should not fail: {e}"
            ))
        })?;

    services.install_profile(compiled);
    Ok(RankProfileDto::from(stored))
}

pub async fn get_rank_profile_inner(
    store: &dyn crate::services::RankProfileStore,
    name: String,
) -> ApiResult<RankProfileDto> {
    let stored = store
        .get(&name)
        .await
        .map_err(|e| ApiError::Internal(format!("rank profile catalog read failed: {e}")))?;
    stored
        .map(RankProfileDto::from)
        .ok_or_else(|| ApiError::NotFound(format!("rank profile '{name}' not found")))
}

pub async fn remove_rank_profile_inner(
    store: &dyn crate::services::RankProfileStore,
    services: Option<&crate::network::rest::canonical::rank::RankServices>,
    name: String,
    if_exists: bool,
) -> ApiResult<()> {
    let removed = store
        .remove(&name)
        .await
        .map_err(|e| ApiError::Internal(format!("rank profile catalog delete failed: {e}")))?;
    if !removed && !if_exists {
        return Err(ApiError::NotFound(format!(
            "rank profile '{name}' not found"
        )));
    }
    if let Some(services) = services {
        services.profile_registry.remove(&name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::record_store::TableWalAppender;
    use crate::services::{CanonicalWalRankProfileStore, MemoryTableWalAppender};
    use std::sync::Arc;

    const VALID_SPEC: &str = "[first_phase]\nexpression = \"1.0\"\nheap_size = 50\n";
    const BROKEN_SPEC: &str = "[first_phase]\nexpression = \"definitely_not_a_feature(\\\"missing\\\")\"\nheap_size = 50\n";

    fn rank_pipeline() -> (
        Arc<dyn crate::services::RankProfileStore>,
        Arc<crate::network::rest::canonical::rank::RankServices>,
    ) {
        use crate::network::rest::canonical::rank::{MockRangeCandidateProvider, RankServices};

        let appender: Arc<dyn TableWalAppender> = Arc::new(MemoryTableWalAppender::new());
        let store: Arc<dyn crate::services::RankProfileStore> =
            Arc::new(CanonicalWalRankProfileStore::new(appender));

        let candidates: Arc<dyn crate::network::rest::canonical::rank::CandidateProvider> =
            Arc::new(MockRangeCandidateProvider { count: 5 });
        let services = Arc::new(RankServices::new(candidates));
        (store, services)
    }

    #[tokio::test]
    async fn install_persists_and_installs_into_live_registry() {
        let (store, services) = rank_pipeline();
        let dto = install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "alpha".to_string(),
                tenant: None,
                spec: VALID_SPEC.to_string(),
            },
        )
        .await
        .expect("install must succeed");
        assert_eq!(dto.name, "alpha");
        assert_eq!(dto.version, 1);
        assert_eq!(dto.spec, VALID_SPEC);
        assert!(store.get("alpha").await.unwrap().is_some());
        assert!(services.profile_registry.get("alpha").is_some());
    }

    #[tokio::test]
    async fn install_rejects_empty_name() {
        let (store, services) = rank_pipeline();
        let err = install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "  ".to_string(),
                tenant: None,
                spec: VALID_SPEC.to_string(),
            },
        )
        .await
        .expect_err("blank name must be rejected");
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn install_rejects_empty_spec() {
        let (store, services) = rank_pipeline();
        let err = install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "x".to_string(),
                tenant: None,
                spec: "   ".to_string(),
            },
        )
        .await
        .expect_err("blank spec must be rejected");
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn install_rejects_uncompilable_spec_without_persisting() {
        let (store, services) = rank_pipeline();
        let err = install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "broken".to_string(),
                tenant: None,
                spec: BROKEN_SPEC.to_string(),
            },
        )
        .await
        .expect_err("broken spec must be rejected");
        match err {
            ApiError::InvalidArgument(msg) => assert!(msg.contains("compile failed")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
        assert!(store.get("broken").await.unwrap().is_none());
        assert!(services.profile_registry.get("broken").is_none());
    }

    #[tokio::test]
    async fn get_returns_404_for_missing() {
        let (store, _services) = rank_pipeline();
        let err = get_rank_profile_inner(store.as_ref(), "ghost".to_string())
            .await
            .expect_err("missing profile must 404");
        assert!(matches!(err, ApiError::NotFound(_)));
    }

    #[tokio::test]
    async fn get_returns_installed_profile() {
        let (store, services) = rank_pipeline();
        install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "exists".to_string(),
                tenant: Some("tenant_a".into()),
                spec: VALID_SPEC.to_string(),
            },
        )
        .await
        .unwrap();

        let dto = get_rank_profile_inner(store.as_ref(), "exists".to_string())
            .await
            .unwrap();
        assert_eq!(dto.name, "exists");
        assert_eq!(dto.tenant.as_deref(), Some("tenant_a"));
        assert_eq!(dto.spec, VALID_SPEC);
    }

    #[tokio::test]
    async fn remove_clears_catalog_and_registry() {
        let (store, services) = rank_pipeline();
        install_rank_profile_inner(
            store.as_ref(),
            services.as_ref(),
            InstallRankProfileRequest {
                name: "doomed".to_string(),
                tenant: None,
                spec: VALID_SPEC.to_string(),
            },
        )
        .await
        .unwrap();
        assert!(store.get("doomed").await.unwrap().is_some());

        remove_rank_profile_inner(
            store.as_ref(),
            Some(services.as_ref()),
            "doomed".to_string(),
            false,
        )
        .await
        .unwrap();
        assert!(store.get("doomed").await.unwrap().is_none());
        assert!(services.profile_registry.get("doomed").is_none());
    }

    #[tokio::test]
    async fn remove_if_exists_is_noop_for_missing() {
        let (store, services) = rank_pipeline();
        remove_rank_profile_inner(
            store.as_ref(),
            Some(services.as_ref()),
            "ghost".to_string(),
            true,
        )
        .await
        .expect("if_exists must hide the missing profile");
    }

    #[tokio::test]
    async fn remove_without_if_exists_errors_for_missing() {
        let (store, services) = rank_pipeline();
        let err = remove_rank_profile_inner(
            store.as_ref(),
            Some(services.as_ref()),
            "ghost".to_string(),
            false,
        )
        .await
        .expect_err("missing profile must 404 without if_exists");
        assert!(matches!(err, ApiError::NotFound(_)));
    }
}
