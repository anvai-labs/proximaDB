// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The scope a query runs in, carried through the query facade and the
//! federated executor for the duration of one call.
//!
//! * [`RequestScope::Tenant`] — a request acting as a tenant; collections are
//!   resolved within it. Set at the transport edge (REST, gRPC) and at the
//!   adapter boundary from the caller's identity.
//! * [`RequestScope::Internal`] — explicitly internal execution (embedded
//!   single-process use, background work); collections are not scoped.
//!
//! No scope at all means **deny**: [`resolution_tenant`] answers a collection
//! as not found, so a path that forgot to establish a scope fails closed.
//! Federated execution runs on the calling task (no spawns), so the scope
//! covers every node it evaluates.

tokio::task_local! {
    static REQUEST_SCOPE: RequestScope;
}

/// The scope a query runs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestScope {
    /// A request acting as this tenant.
    Tenant(String),
    /// Explicitly internal execution; the reason is for audit/debug.
    Internal(&'static str),
}

/// Run `future` as a request acting as `tenant` (blank ⇒ the default tenant).
pub async fn with_request_tenant<F: std::future::Future>(
    tenant: impl AsRef<str>,
    future: F,
) -> F::Output {
    let tenant = proximadb_tenant::resolve_request_tenant(Some(tenant.as_ref()));
    REQUEST_SCOPE
        .scope(RequestScope::Tenant(tenant), future)
        .await
}

/// Run `future` as explicitly internal execution.
pub async fn with_internal_scope<F: std::future::Future>(
    reason: &'static str,
    future: F,
) -> F::Output {
    REQUEST_SCOPE
        .scope(RequestScope::Internal(reason), future)
        .await
}

/// Run `future` in the scope of a port identity: its tenant when it carries
/// one; otherwise the scope already in force; otherwise **no scope**, so
/// collection resolution inside denies. A tenant-less identity never becomes
/// internal on its own: internal execution is established explicitly with
/// [`with_internal_scope`].
pub async fn with_identity_scope<F: std::future::Future>(
    tenant: Option<&str>,
    future: F,
) -> F::Output {
    match tenant.filter(|tenant| !tenant.is_empty()) {
        Some(tenant) => with_request_tenant(tenant, future).await,
        None => future.await,
    }
}

/// The scope of the current call, if any.
pub fn current_scope() -> Option<RequestScope> {
    REQUEST_SCOPE.try_with(Clone::clone).ok()
}

/// The tenant to resolve `identifier` within: `Ok(Some(tenant))` for a
/// request, `Ok(None)` for internal execution, and — with no scope at all —
/// `Err(CollectionNotFound)` (fail closed).
pub fn resolution_tenant(identifier: &str) -> anyhow::Result<Option<String>> {
    match current_scope() {
        Some(RequestScope::Tenant(tenant)) => Ok(Some(tenant)),
        Some(RequestScope::Internal(_)) => Ok(None),
        None => {
            Err(proximadb_search_types::client_argument::CollectionNotFound::new(identifier).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scope_is_visible_inside_and_absent_outside() {
        assert_eq!(current_scope(), None);
        let inside =
            with_request_tenant("acme", async { Box::pin(async { current_scope() }).await }).await;
        assert_eq!(inside, Some(RequestScope::Tenant("acme".into())));
        assert_eq!(current_scope(), None);
    }

    #[tokio::test]
    async fn absent_scope_denies_and_internal_is_explicit() {
        let denied = resolution_tenant("docs");
        assert!(
            denied
                .as_ref()
                .err()
                .and_then(proximadb_search_types::client_argument::collection_not_found)
                .is_some(),
            "no scope must answer not found: {denied:?}"
        );
        let internal = with_internal_scope("test", async { resolution_tenant("docs") }).await;
        assert_eq!(internal.ok(), Some(None));
        let tenant = with_request_tenant(" ", async { resolution_tenant("docs") }).await;
        assert_eq!(
            tenant.ok(),
            Some(Some(proximadb_tenant::DEFAULT_TENANT.to_string()))
        );
    }

    #[tokio::test]
    async fn identity_scope_prefers_the_identity_then_the_outer_scope() {
        let outer = with_request_tenant("acme", async {
            with_identity_scope(None, async { current_scope() }).await
        })
        .await;
        assert_eq!(outer, Some(RequestScope::Tenant("acme".into())));
        let explicit = with_request_tenant("acme", async {
            with_identity_scope(Some("globex"), async { current_scope() }).await
        })
        .await;
        assert_eq!(explicit, Some(RequestScope::Tenant("globex".into())));
        // A tenant-less identity with no outer scope stays unscoped and denies.
        let bare =
            with_identity_scope(None, async { (current_scope(), resolution_tenant("docs")) }).await;
        assert_eq!(bare.0, None);
        assert!(bare.1.is_err(), "no scope must deny");
        let empty = with_identity_scope(Some(""), async { current_scope() }).await;
        assert_eq!(empty, None);
        let internal = with_internal_scope("test", async {
            with_identity_scope(None, async { current_scope() }).await
        })
        .await;
        assert!(matches!(internal, Some(RequestScope::Internal(_))));
    }
}
