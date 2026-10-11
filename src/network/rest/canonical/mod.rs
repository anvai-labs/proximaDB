/// ABAC policy-provisioning admin API (TD-ABAC control-plane). Auth-gated to
/// `SystemAdmin` or `ConfigureSystem`; `abac-policy`-only (default builds omit it).
#[cfg(feature = "abac-policy")]
pub mod abac_admin;
/// Phase 7.2.4: per-collection cache-affinity operator API.
pub mod affinity;
/// REST handlers for read-only collection analytics (Entanglement Index)
pub mod analytics;
/// REST handlers for Agentic Query Language (RUBICON)
pub mod aql;
#[cfg(feature = "enterprise-catalogs")]
/// REST handlers for enterprise catalog operations (Polaris, Delta Lake)
pub mod catalog;
/// REST handlers for document storage operations
pub mod document;
/// REST handlers for SKS entity operations
pub mod entities;
/// Core REST handlers for collections, vectors, search, and health
pub mod handlers;
/// REST handlers for hybrid (vector + BM25) search
pub mod hybrid;
/// Iceberg REST Catalog server (v1 spec) — Spark/Trino/DuckDB/PyIceberg compatible
pub mod iceberg_rest_catalog;
/// REST handler for the agent-memory write surface (TD-101)
pub mod memory;
/// REST handlers for unified multi-model query execution
pub mod multimodal_query;
/// REST handlers for Natural Language query translation (AV-SQL)
pub mod nl;
/// Phase 6: per-collection pinning control surface (turbopuffer
/// `/v1/namespaces/:ns/metadata`-style operator API).
pub mod pinning;
/// Slice 3 of tenant-pod-affinity: per-(tenant, collection) primary-pod
/// operator API. Auth-gated to `SystemAdmin` or `ConfigureSystem`.
pub mod primary_pod;
/// REST DTOs + handler for the multi-phase rank pipeline (R-7b)
pub mod rank;
/// Production `HybridSearchBackend` impl backing `RankServices` retrieval (R-7c.2)
pub mod rank_backend;
/// REST endpoints for the durable rank-profile catalog (R-7c.3 commit 5/5)
pub mod rank_profile;
