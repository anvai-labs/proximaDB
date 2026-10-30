// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Transport-edge request scope: every REST request runs with its resolved
//! tenant as the query request scope (`crate::query::request_scope`), so the
//! query facade and federated executor resolve collections within it. A
//! request without a resolved tenant gets no scope, which the query layers
//! treat as deny.

use axum::{extract::Request, middleware::Next, response::Response};

/// Axum middleware: run the request in its tenant's scope. Must sit inside
/// the tenant middleware (which resolves the tenant).
pub async fn rest_request_scope(request: Request, next: Next) -> Response {
    let tenant = request
        .extensions()
        .get::<crate::network::middleware::tenant::TenantContext>()
        .map(|tenant| tenant.tenant_id.clone());
    match tenant {
        Some(tenant) => {
            crate::query::request_scope::with_request_tenant(tenant, next.run(request)).await
        }
        None => next.run(request).await,
    }
}
