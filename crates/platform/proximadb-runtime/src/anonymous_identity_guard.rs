// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Guard: every use of the anonymous port identity is listed here with its
//! reason. A request path must pass the request tenant (`PortIdentity::for_tenant`
//! or `From<&TenantContext>`) so collection resolution is scoped to it; a new,
//! unlisted anonymous use fails this test.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// `file → (count, reason)`. Keep the reason specific.
    const ALLOWED: &[(&str, usize, &str)] = &[
        (
            "crates/binding/proximadb-embedded/src/lib.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/platform/proximadb-api/src/lib.rs",
            3,
            "unit tests of the API port wiring",
        ),
        (
            "crates/platform/proximadb-runtime/src/handlers.rs",
            4,
            "the deprecated non-tenant v1 handler (test callers only) and its tests",
        ),
        (
            "src/network/rest/canonical/rank_backend.rs",
            1,
            "vector leg runs after scope_collection resolved the collection within the \
             request tenant",
        ),
        (
            "src/query/facade/strategies/vector.rs",
            1,
            "internal execution outside a request scope (requests carry request_scope)",
        ),
        (
            "src/query/federated/execution/mod.rs",
            1,
            "internal execution outside a request scope (requests carry request_scope)",
        ),
        (
            "src/services/dml/mod.rs",
            1,
            "scan is scoped by the TenantContext argument; identity carries no subject",
        ),
        ("src/services/dml/tests.rs", 5, "unit tests"),
        (
            "src/services/record_store.rs",
            1,
            "get is scoped by the TenantContext argument; identity carries no subject",
        ),
    ];

    /// Same contract for the owned form's tenant-less default.
    const ALLOWED_OWNED_DEFAULT: &[(&str, usize, &str)] = &[
        (
            "src/datafusion/mod.rs",
            2,
            "create_session_context registers no vector/graph function (the identity is \
             unused); create_session_context_with_vector_ops has no production caller",
        ),
        (
            "src/datafusion/cross_modal.rs",
            2,
            "VectorSearchTableFunction::new (test constructor) and a unit test; production \
             registers with_identity",
        ),
    ];

    /// Unscoped table resolution (`CatalogManager::resolve_table`). Request
    /// paths resolve with `resolve_table_scoped` and the request tenant.
    const ALLOWED_UNSCOPED_RESOLVE: &[(&str, usize, &str)] = &[
        (
            "crates/binding/proximadb-embedded/src/lib.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/binding/proximadb-embedded/src/python_dataframe.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/binding/proximadb-embedded/src/tests.rs",
            1,
            "unit tests",
        ),
        (
            "crates/control/proximadb-catalog/src/fc_metamodel.rs",
            1,
            "metamodel name lookup within an explicit namespace (a different resolver)",
        ),
        (
            "crates/control/proximadb-catalog/src/manager.rs",
            1,
            "resolve_table_scoped's own base resolution before tenant prefixing",
        ),
        (
            "src/catalog/index_location_resolver.rs",
            1,
            "storage path resolution for a collection id already resolved by its caller",
        ),
        ("src/catalog/mod.rs", 5, "unit tests"),
        (
            "src/services/collection/manager.rs",
            1,
            "the internal collection lookup (collection_internal); request paths apply the \
             ownership rule on its result",
        ),
        ("src/services/ddl/mod.rs", 5, "unit tests"),
        ("src/services/dml/tests.rs", 21, "unit tests"),
        (
            "src/services/external_collection/service.rs",
            1,
            "unit test",
        ),
        (
            "src/services/snapshot/coordinator.rs",
            2,
            "discovery projections for a collection the caller already resolved within its \
             tenant, and background discovery",
        ),
        (
            "src/storage/persistence/write_ahead_log/mod.rs",
            1,
            "internal WAL/recovery collection lookup; no request reaches it",
        ),
    ];

    /// Direct catalog access (`CatalogManager::default_catalog`) bypasses the
    /// tenant-scoped resolver. Every use is listed; request-facing ones state how
    /// they scope to the request tenant.
    const ALLOWED_DEFAULT_CATALOG: &[(&str, usize, &str)] = &[
        (
            "crates/control/proximadb-catalog/src/iceberg_rest_service.rs",
            14,
            "Iceberg REST: every method maps the namespace into the request tenant's \
             (scoped_namespace) and checks visibility (visible_entry) before use",
        ),
        (
            "crates/control/proximadb-catalog/src/manager.rs",
            4,
            "the resolver itself",
        ),
        (
            "crates/control/proximadb-catalog/src/model_registry_service.rs",
            1,
            "model registry service owns tenant scoping of its own documents",
        ),
        ("src/catalog/mod.rs", 7, "unit tests"),
        (
            "src/cluster/primary_pod_registry.rs",
            5,
            "operator-only placement registry (internal)",
        ),
        ("src/database.rs", 2, "startup wiring"),
        (
            "src/network/rest/canonical/primary_pod.rs",
            6,
            "operator-only API (authorize_operator) over all tenants' placement, by design",
        ),
        ("src/network/shared_services.rs", 2, "startup wiring"),
        (
            "src/services/collection/manager.rs",
            16,
            "collection writer/reader internals; request paths apply the ownership rule",
        ),
        (
            "src/services/ddl/mod.rs",
            4,
            "namespace DDL scoped via tenant_scoped_namespace, and unit tests",
        ),
        ("src/services/discovery/executor.rs", 1, "unit test"),
        ("src/services/snapshot/coordinator.rs", 1, "unit test"),
        (
            "src/storage/persistence/write_ahead_log/mod.rs",
            2,
            "internal WAL/recovery catalog walk",
        ),
    ];

    /// Every route literal in the codebase with how it is held to the request
    /// tenant (an e2e test name, the scoping mechanism, or why it is not a
    /// tenant-facing route). A new route must be added here to pass.
    const ROUTE_INVENTORY: &[(&str, &str, &str)] = &[
        (
            "src/observability/ingestion/adapters/otlp.rs",
            "/v1/traces",
            "not mounted by default (OTLP adapter); writes the configured namespace",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/health",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/metrics",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/api/metrics",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/api/alerts",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/monitoring/dashboard/mod.rs",
            "/alerts",
            "not mounted (create_dashboard_router has no caller)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q1",
            "test router (inside a test module)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q2",
            "test router (inside a test module)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q3",
            "test router (inside a test module)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q4",
            "test router (inside a test module)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q5",
            "test router (inside a test module)",
        ),
        (
            "src/storage/multimodel/htap/router.rs",
            "q6",
            "test router (inside a test module)",
        ),
        (
            "src/network/metrics_service.rs",
            "/",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/metrics_service.rs",
            "/json",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/metrics_service.rs",
            "/prometheus",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/metrics_service.rs",
            "/metrics/prometheus",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/metrics_service.rs",
            "/health",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/metrics_service.rs",
            "/alerts",
            "operator: /metrics requires ViewSystemMetrics with auth on (no tenant data otherwise documented)",
        ),
        (
            "src/network/mcp/mod.rs",
            "/",
            "MCP listener, not mounted by default (mcp_port unset); unauthenticated-only, acts as its configured tenant",
        ),
        (
            "src/network/mcp/mod.rs",
            "/mcp",
            "MCP listener, not mounted by default (mcp_port unset); unauthenticated-only, acts as its configured tenant",
        ),
        (
            "src/network/rest/server.rs",
            "/mlflow-ui/",
            "gated off by default (MLflow UI) / legacy v1 path returns the migration envelope",
        ),
        (
            "src/network/rest/server.rs",
            "/api/v1/search",
            "gated off by default (MLflow UI) / legacy v1 path returns the migration envelope",
        ),
        (
            "src/network/rest/websocket.rs",
            "/v1/stream/insert/{collection}",
            "tenant: streaming coordinator never reads stored tenant data (no storage flush)",
        ),
        (
            "src/network/rest/websocket.rs",
            "/v1/stream/subscribe/{collection}",
            "tenant: streaming coordinator never reads stored tenant data (no storage flush)",
        ),
        (
            "src/network/rest/websocket.rs",
            "/v1/stream/status/{session_id}",
            "tenant: streaming coordinator never reads stored tenant data (no storage flush)",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}",
            "e2e the_default_tenant_sees_other_tenants_collections_as_missing",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/entities",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/entities/search",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/entities/{entity_id}",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/schema",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/records/batch",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/records/scan",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/records/{record_id}",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/search",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/changes",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/documents",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/query",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/query/explain",
            "tenant: explain resolves authority within the request scope",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/sql",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/model-registries",
            "tenant: model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/model-registries/{name}",
            "tenant: model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/model-registries/{name}/mutations",
            "tenant: model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/model-registries/{name}/resolve",
            "tenant: model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/timeseries/collections",
            "tenant: timeseries collections resolved within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/timeseries/collections/{collection_id}",
            "tenant: timeseries collections resolved within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/timeseries/collections/{collection_id}/ingest",
            "tenant: timeseries collections resolved within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/timeseries/collections/{collection_id}/query",
            "tenant: timeseries collections resolved within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/timeseries/collections/{collection_id}/aggregate",
            "tenant: timeseries collections resolved within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/graphs/{graph_id}/fusion-search",
            "tenant: graph scope composed from the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/graphs/{graph_id}/impact-analysis",
            "tenant: graph scope composed from the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/discovery-jobs",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/discovery-jobs/{job_id}",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/external-collections",
            "tenant: registry records the owning tenant; gated by PROXIMADB_EXTERNAL_TABLE_ROOTS",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/external-collections/{id}",
            "tenant: registry records the owning tenant; gated by PROXIMADB_EXTERNAL_TABLE_ROOTS",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/external-collections/{id}/build",
            "tenant: registry records the owning tenant; gated by PROXIMADB_EXTERNAL_TABLE_ROOTS",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/external-collections/{id}/search",
            "tenant: registry records the owning tenant; gated by PROXIMADB_EXTERNAL_TABLE_ROOTS",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/external-collections/{id}/refresh",
            "tenant: registry records the owning tenant; gated by PROXIMADB_EXTERNAL_TABLE_ROOTS",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/_diagnostics/collections/{collection_id}/route-health",
            "tenant: collection resolved within the request tenant, 404 on miss",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/_diagnostics/collections/{collection_id}/statistics",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/_diagnostics/collections/{collection_id}/recall-tune",
            "tenant: collection resolved within the request tenant, 404 on miss; admin-gated",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/_diagnostics/collections/{collection_id}/recluster",
            "tenant: collection resolved within the request tenant, 404 on miss; admin-gated",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/suspend",
            "tenant: ensure_collection_exists within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/collections/{collection_id}/resume",
            "tenant: ensure_collection_exists within the request tenant",
        ),
        (
            "src/network/rest/v2/mod.rs",
            "/_meta/capabilities",
            "public: no tenant data",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/search",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/{model_id}",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/{model_id}/params",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/{model_id}/tags",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/{model_id}/tags/{tag_key}",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/logged_models.rs",
            "/logged-models/{model_id}/artifacts/directories",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/create",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/get",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/get-by-name",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/search",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/delete",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/experiments/restore",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/create",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/get",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/update",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/search",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/delete",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/restore",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/log-parameter",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/log-metric",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/log-batch",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/set-tag",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/delete-tag",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/outputs",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/log-model",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/runs/log-inputs",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/metrics/get-history",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/traces/delete-traces",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/mod.rs",
            "/traces/{trace_id}/tags",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/registered-models/create",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/registered-models/get",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/registered-models/get-by-name",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/registered-models/search",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/registered-models/alias",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/model-versions/get",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/model-versions/search",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/model-versions/create",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/registry.rs",
            "/model-versions/transition-stage",
            "gated off by default (MLflow env gate); model registry service scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/get",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/batchGet",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/batchGetInfos",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/search",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/{trace_id}",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/{trace_id}/assessments",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/traces.rs",
            "/traces/{trace_id}/assessments/{assessment_id}",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/ui.rs",
            "/",
            "gated off by default (MLflow env gate); static UI",
        ),
        (
            "src/network/rest/mlflow/ui.rs",
            "/static-files/{*path}",
            "gated off by default (MLflow env gate); static UI",
        ),
        (
            "src/network/rest/mlflow/artifacts.rs",
            "/artifacts",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/mlflow/artifacts.rs",
            "/artifacts/{*path}",
            "gated off by default (MLflow env gate); MLflow run store scopes by request tenant",
        ),
        (
            "src/network/rest/canonical/memory.rs",
            "/ingest",
            "tenant: scope keyed by the authenticated request tenant",
        ),
        (
            "src/network/rest/canonical/memory.rs",
            "/consolidation/{session_id}",
            "tenant: scope keyed by the authenticated request tenant",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/config",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/oauth/tokens",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/properties",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/tables",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/tables/{table}",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/tables/{table}/metadata/{file}",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/register",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/iceberg_rest_catalog.rs",
            "/namespaces/{namespace}/views",
            "e2e iceberg_catalog_alter_truncate_and_restart_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/aql.rs",
            "/execute",
            "tenant: AQL scope = request tenant (sources resolve within it)",
        ),
        (
            "src/network/rest/canonical/aql.rs",
            "/audit/{query_id}",
            "tenant: AQL scope = request tenant (sources resolve within it)",
        ),
        (
            "src/network/rest/canonical/nl.rs",
            "/translate",
            "tenant: translation only; execution goes through the scoped query path",
        ),
        (
            "src/network/rest/canonical/entities.rs",
            "/v1/collections/{collection_id}/entities",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/canonical/entities.rs",
            "/v1/collections/{collection_id}/entities/{entity_id}",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/canonical/entities.rs",
            "/v1/collections/{collection_id}/entities/search",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/progressive/search/{collection_id}",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/catalog/table-routing",
            "e2e catalog_entries_introspection_and_lifecycle_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/catalog/table-write/explain",
            "e2e catalog_entries_introspection_and_lifecycle_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/collections/{collection_id}/branches/{branch}/merge",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/rank/search",
            "e2e search_resolution_is_scoped_to_the_request_tenant",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/collections/{collection_id}/pin",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/collections/pinning",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/collections/{collection_id}/affinity",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/collections/affinity",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/primary-pod/{tenant_id}/{collection_id}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/primary-pod",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/policy-bindings/{tenant}/{object_id}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/policy-bindings/{tenant}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/tenant-posture/{tenant}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/grants",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/grants/{owner_tenant}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/grants/{owner_tenant}/{grant_id}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/attribute-bindings",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/predicate-objects",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/abac/predicate-objects/{object_id}",
            "operator: authorize_operator (SystemAdmin/ConfigureSystem)",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/health",
            "public: liveness, no tenant data",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/health/live",
            "public: liveness, no tenant data",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/health/ready",
            "public: liveness, no tenant data",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v1/ping",
            "public: liveness, no tenant data",
        ),
        (
            "src/network/rest/canonical/handlers.rs",
            "/api/v2/ping",
            "public: liveness, no tenant data",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/{name}",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/{catalog}/namespaces",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/{catalog}/namespaces/{namespace}",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/{catalog}/tables",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/catalog.rs",
            "/{catalog}/tables/{table}",
            "not mounted by default (external catalogs feature off)",
        ),
        (
            "src/network/rest/canonical/analytics.rs",
            "/entanglement",
            "tenant: collection resolved within the request tenant",
        ),
        (
            "src/network/rest/canonical/analytics.rs",
            "/collections/{collection_id}/entanglement",
            "tenant: collection resolved within the request tenant",
        ),
        (
            "src/network/auth/middleware.rs",
            "/api/v1/secure",
            "test router (inside a test module)",
        ),
        (
            "src/network/auth/middleware.rs",
            "/api/v1/search",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/request_id.rs",
            "/api/v1/...",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/auth_failclosed.rs",
            "/api/v1/search",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/auth_failclosed.rs",
            "/health",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/tls.rs",
            "/api/v1/secure",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/tenant.rs",
            "/api/v1/collections",
            "test router (inside a test module)",
        ),
        (
            "src/network/middleware/tenant.rs",
            "/probe",
            "test router (inside a test module)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/health",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/events/{collection_id}",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/events",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/events/{event_id}/process",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/events/query",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/index/axis/eventlog/service_interface.rs",
            "/compact/check",
            "not mounted (no caller of create_router)",
        ),
        (
            "src/cluster/routing.rs",
            "coll-x",
            "test router (inside a test module)",
        ),
        (
            "src/cluster/routing.rs",
            "coll-y",
            "test router (inside a test module)",
        ),
        (
            "src/cluster/routing.rs",
            "coll-w",
            "test router (inside a test module)",
        ),
        (
            "src/cluster/routing.rs",
            "coll-exp",
            "test router (inside a test module)",
        ),
        (
            "crates/platform/proximadb-admin-ui/src/lib.rs",
            "/admin",
            "not mounted by default (admin UI gated off); operator surface",
        ),
        (
            "crates/platform/proximadb-admin-ui/src/lib.rs",
            "/dashboard",
            "not mounted by default (admin UI gated off); operator surface",
        ),
        (
            "crates/platform/proximadb-api/src/rest/deprecation.rs",
            "/relative",
            "test router (inside a test module)",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/execute",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/multi-model",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/federated",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/distributed",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/explain",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/prepare",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/execute/{statement_id}",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/prepared/{statement_id}",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs",
            "/prepared/stats",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}/documents",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}/documents/{id}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}/documents/batch",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}/documents/aggregate",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/document.rs",
            "/{collection}/indexes",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/logs/bulk",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/logs",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/logs/search",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/metrics/bulk",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/metrics",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/metrics/aggregate",
            "e2e diagnostics_observability_and_prepared_statements_are_tenant_scoped",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/metrics/promql",
            "returns 501 without touching data",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/traces/bulk",
            "returns 501 without touching data",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/observability.rs",
            "/namespaces/{namespace}/traces/search",
            "returns 501 without touching data",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/hybrid.rs",
            "/api/v2/hybrid/search",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/hybrid.rs",
            "/api/v2/hybrid/index",
            "e2e every_collection_surface_is_scoped_to_the_request_tenant",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/hybrid.rs",
            "/health/live",
            "test router (inside a test module)",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/hybrid.rs",
            "/health/ready",
            "test router (inside a test module)",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/schema",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/nodes",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/nodes/{id}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/nodes/{id}/neighbors",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/edges",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/edges/{id}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/traverse",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/walk",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/step",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/shortest-path",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/query/nodes",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/query/edges",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/query",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/rag",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/nodes/batch",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/edges/batch",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/stats",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/components",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/cycles",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/graphs/{graph_id}/constraints/unique",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/nodes",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/nodes/{id}",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/edges",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-api/src/rest/canonical/graph.rs",
            "/stats",
            "e2e tenant_owned_resources_are_reachable_only_by_their_owner",
        ),
        (
            "crates/platform/proximadb-network-middleware/src/rate_limit.rs",
            "/",
            "test router (inside a test module)",
        ),
        (
            "crates/platform/proximadb-network-middleware/src/rate_limit.rs",
            "/health",
            "test router (inside a test module)",
        ),
    ];

    /// Every route path literal (a string passed to `route`) under `src/` and `crates/platform/`, as
    /// `(file, path)` pairs.
    fn route_literals() -> std::collections::BTreeSet<(String, String)> {
        let root = workspace_root();
        let mut files = Vec::new();
        for top in ["src", "crates/platform"] {
            rust_files(&root.join(top), &mut files);
        }
        let mut found = std::collections::BTreeSet::new();
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let rel = file
                .strip_prefix(&root)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            let mut rest = text.as_str();
            let marker = format!(".{}(", "route");
            while let Some(at) = rest.find(&marker) {
                rest = &rest[at + marker.len()..];
                let trimmed = rest.trim_start();
                if let Some(literal) = trimmed.strip_prefix('"')
                    && let Some(end) = literal.find('"')
                {
                    found.insert((rel.clone(), literal[..end].to_string()));
                }
            }
        }
        found
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// Uses of the borrowed form's tenant-less default (`Default` is derived).
    /// Excludes the owned form, whose name ends with the same text.
    const ALLOWED_BORROWED_DEFAULT: &[(&str, usize, &str)] = &[(
        "src/services/dml/tests.rs",
        4,
        "unit tests of the identity carrier",
    )];

    /// Struct literals of either identity spelling `tenant_id: None`.
    const ALLOWED_TENANTLESS_LITERAL: &[(&str, usize, &str)] = &[];

    fn check(needle: &str, allowlist: &[(&str, usize, &str)]) {
        check_with(|text| text.matches(needle).count(), allowlist);
    }

    /// The borrowed form's `default()` constructor, excluding the owned form.
    fn borrowed_defaults(text: &str) -> usize {
        let needle = format!("PortIdentity::{}()", "default");
        text.match_indices(&needle)
            .filter(|(at, _)| !text[..*at].ends_with("Owned"))
            .count()
    }

    /// Struct literals `PortIdentity { .. }` / `OwnedPortIdentity { .. }` whose
    /// body sets `tenant_id: None` (whitespace-insensitive).
    fn tenantless_literals(text: &str) -> usize {
        let opener = format!("PortIdentity {}", "{");
        let mut count = 0;
        for (at, _) in text.match_indices(&opener) {
            // Skip the type definitions and impl headers.
            let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
            let line = &text[line_start..at];
            if line.contains("struct ") || line.contains("impl") || line.contains("for ") {
                continue;
            }
            let body_start = at + opener.len();
            let mut depth = 1usize;
            let mut end = body_start;
            for (offset, c) in text[body_start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = body_start + offset;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let body: String = text[body_start..end]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            if body.contains(&format!("tenant_id:{}", "None")) {
                count += 1;
            }
        }
        count
    }

    fn check_with(count_in: impl Fn(&str) -> usize, allowlist: &[(&str, usize, &str)]) {
        let root = workspace_root();
        let mut files = Vec::new();
        for top in ["src", "crates", "apps"] {
            rust_files(&root.join(top), &mut files);
        }
        assert!(
            !files.is_empty(),
            "guard found no sources under {}: the workspace layout changed",
            root.display()
        );
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let count = count_in(&text);
            if count > 0 {
                let rel = file
                    .strip_prefix(&root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| file.to_string_lossy().into_owned());
                found.insert(rel, count);
            }
        }
        // A guard with listed uses must still find them; one with none listed
        // passes on finding none.
        assert!(
            allowlist.is_empty() || !found.is_empty(),
            "guard matched nothing under {}: the pattern or layout changed",
            root.display()
        );
        let allowed: BTreeMap<String, usize> = allowlist
            .iter()
            .map(|(file, count, _)| (file.to_string(), *count))
            .collect();
        let mut problems = Vec::new();
        for (file, count) in &found {
            match allowed.get(file) {
                Some(limit) if count <= limit => {}
                Some(limit) => problems.push(format!(
                    "{file}: {count} uses, {limit} allowed — pass the request tenant instead"
                )),
                None => problems.push(format!(
                    "{file}: {count} unlisted use(s) — pass the request tenant, or list it with a reason"
                )),
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn anonymous_port_identity_uses_are_all_listed() {
        // Built so this file does not match itself.
        check(&format!("PortIdentity::{}()", "anonymous"), ALLOWED);
    }

    #[test]
    fn tenant_less_owned_identity_defaults_are_all_listed() {
        check(
            &format!("OwnedPortIdentity::{}()", "default"),
            ALLOWED_OWNED_DEFAULT,
        );
    }

    #[test]
    fn tenant_less_borrowed_identity_defaults_are_all_listed() {
        check_with(borrowed_defaults, ALLOWED_BORROWED_DEFAULT);
    }

    #[test]
    fn tenant_less_identity_literals_are_all_listed() {
        check_with(tenantless_literals, ALLOWED_TENANTLESS_LITERAL);
    }

    #[test]
    fn tenantless_literal_detector_catches_the_spellings() {
        let src = format!(
            "let a = PortIdentity {{ tenant_id: {n}, subject }};\n\
             let b = proximadb_runtime::OwnedPortIdentity {{\n    tenant_id:   {n},\n}};\n\
             let c = PortIdentity {{ tenant_id: Some(t) }};\n\
             pub struct PortIdentity {{ pub tenant_id: Option<u8> }}\n",
            n = "None"
        );
        assert_eq!(tenantless_literals(&src), 2);
        let defaults = format!(
            "PortIdentity::{d}(); OwnedPortIdentity::{d}();",
            d = "default"
        );
        assert_eq!(borrowed_defaults(&defaults), 1);
    }

    #[test]
    fn unscoped_table_resolutions_are_all_listed() {
        check(&format!(".resolve_{}(", "table"), ALLOWED_UNSCOPED_RESOLVE);
    }

    #[test]
    fn direct_default_catalog_access_is_all_listed() {
        check(
            &format!(".default_{}()", "catalog"),
            ALLOWED_DEFAULT_CATALOG,
        );
    }

    #[test]
    fn every_route_is_in_the_tenant_inventory() {
        let found = route_literals();
        assert!(
            !found.is_empty(),
            "route scan found nothing: layout changed"
        );
        let listed: std::collections::BTreeSet<(String, String)> = ROUTE_INVENTORY
            .iter()
            .map(|(file, path, _)| (file.to_string(), path.to_string()))
            .collect();
        let unlisted: Vec<_> = found.difference(&listed).collect();
        let stale: Vec<_> = listed.difference(&found).collect();
        assert!(
            unlisted.is_empty(),
            "routes missing from ROUTE_INVENTORY (state how each is held to the request \
             tenant): {unlisted:#?}"
        );
        assert!(
            stale.is_empty(),
            "ROUTE_INVENTORY lists routes that no longer exist: {stale:#?}"
        );
        for (file, path, coverage) in ROUTE_INVENTORY {
            assert!(!coverage.trim().is_empty(), "{file} {path}: empty coverage");
        }
    }
}
