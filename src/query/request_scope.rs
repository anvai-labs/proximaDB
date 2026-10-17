// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The tenant a query request acts as, carried through the query facade and
//! the federated executor for the duration of one request.
//!
//! The unified-query REST router sets it once per request
//! ([`with_request_tenant`]); the facade's vector strategy and the federated
//! executor read it ([`current_request_tenant`]) to resolve collections within
//! that tenant. Outside a request scope it is `None` (internal execution).
//! Federated execution runs on the request task (no spawns), so the scope
//! covers every node it evaluates.

tokio::task_local! {
    static REQUEST_TENANT: Option<String>;
}

/// Run `future` with `tenant` as the request tenant.
pub async fn with_request_tenant<F: std::future::Future>(
    tenant: Option<String>,
    future: F,
) -> F::Output {
    REQUEST_TENANT.scope(tenant, future).await
}

/// The request tenant of the current request scope, if any.
pub fn current_request_tenant() -> Option<String> {
    REQUEST_TENANT.try_with(Clone::clone).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scope_is_visible_inside_and_absent_outside() {
        assert_eq!(current_request_tenant(), None);
        let inside = with_request_tenant(Some("acme".to_string()), async {
            Box::pin(async { current_request_tenant() }).await
        })
        .await;
        assert_eq!(inside.as_deref(), Some("acme"));
        assert_eq!(current_request_tenant(), None);
    }
}
