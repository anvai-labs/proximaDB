// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Typed client-argument errors shared by every search surface
//! (TD-SECSCAN-1 §5).
//!
//! A search failure caused by the client's own argument — an empty,
//! non-finite or wrong-dimension query vector ([`InvalidQueryVector`]) or a
//! `top_k` above the server cap ([`TopKLimitExceeded`]) — must reach the
//! client as HTTP 400 / gRPC `InvalidArgument`, never as a 500 / `Internal`.
//! The search authorities raise these as typed errors; each request boundary
//! (REST, gRPC, AQL — including the API crate, which cannot see the root
//! crate) classifies through [`is_client_argument_error`] and reports
//! [`client_argument_message`]. Both walk the whole `anyhow` chain, so a
//! `.context(..)` wrapper between the authority and the boundary does not
//! hide the cause.

use crate::top_k_limit::TopKLimitExceeded;

/// A query vector the client must fix: empty, non-finite, or of the wrong
/// dimension for the collection. The message names the collection the way the
/// client knows it, never by its internal object id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidQueryVector(pub String);

impl std::fmt::Display for InvalidQueryVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidQueryVector {}

/// True when a search failure is the client's argument: an invalid query
/// vector or a `top_k` above the server cap. Walks the whole `anyhow` chain so
/// `.context(..)` wrappers don't hide it.
pub fn is_client_argument_error(err: &anyhow::Error) -> bool {
    client_argument_message(err).is_some()
}

/// The client-facing message of the typed client-argument cause in `err`'s
/// chain (the cause itself, without the internal context wrappers around it),
/// or `None` when the failure is not the client's argument.
pub fn client_argument_message(err: &anyhow::Error) -> Option<String> {
    err.chain().find_map(|cause| {
        if let Some(invalid) = cause.downcast_ref::<InvalidQueryVector>() {
            Some(invalid.to_string())
        } else {
            cause
                .downcast_ref::<TopKLimitExceeded>()
                .map(ToString::to_string)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn typed_cause_survives_context_wrappers() {
        let err = Err::<(), _>(InvalidQueryVector(
            "Query vector has dimension 3 but collection 'orders' expects dimension 4".into(),
        ))
        .context("vector search failed for fusion")
        .context("entity fusion search")
        .err();
        let Some(err) = err else {
            panic!("expected an error");
        };
        assert!(is_client_argument_error(&err));
        assert_eq!(
            client_argument_message(&err).as_deref(),
            Some("Query vector has dimension 3 but collection 'orders' expects dimension 4")
        );
    }

    #[test]
    fn top_k_limit_is_a_client_argument() {
        let err = anyhow::Error::new(TopKLimitExceeded {
            requested: 20_000,
            max: 10_000,
        })
        .context("search");
        assert!(
            client_argument_message(&err).is_some_and(|m| m.contains("top_k=20000")),
            "{err:#}"
        );
    }

    /// Genuine server failures stay server failures (500 / `Internal`), even
    /// when their text happens to mention a dimension.
    #[test]
    fn server_errors_are_not_client_arguments() {
        let err = anyhow::anyhow!("index segment has dimension 3, storage expects 4")
            .context("vector search failed");
        assert!(!is_client_argument_error(&err));
        assert_eq!(client_argument_message(&err), None);
    }
}
