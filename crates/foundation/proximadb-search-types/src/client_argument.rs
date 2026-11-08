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
/// dimension for the collection.
///
/// The message names the collection ONLY by an identifier the caller supplied
/// ([`Self::collection`]); when none is known it says "the collection". It is
/// never filled from the catalog: an error message echoes only what the
/// request contained. Authorities that rewrite the caller's identifier to the
/// internal object id attach the caller's identifier on the way out with
/// [`label_client_collection`].
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidQueryVector {
    /// The collection as the caller named it, if known.
    pub collection: Option<String>,
    /// What is wrong with the vector.
    pub problem: QueryVectorProblem,
}

/// What is wrong with a query vector.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryVectorProblem {
    /// No values.
    Empty,
    /// A NaN or infinite value.
    NonFinite { index: usize, value: f32 },
    /// The vector's dimension differs from the collection's.
    Dimension { actual: usize, expected: usize },
    /// Any other client-side defect, described verbatim.
    Other(String),
}

impl InvalidQueryVector {
    /// An invalid query vector for `collection` (as the caller named it).
    pub fn new(collection: Option<String>, problem: QueryVectorProblem) -> Self {
        Self {
            collection,
            problem,
        }
    }

    /// An opaque client-argument message (no collection label).
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(None, QueryVectorProblem::Other(message.into()))
    }

    fn label(&self) -> String {
        match &self.collection {
            Some(name) => format!("collection '{name}'"),
            None => "the collection".to_string(),
        }
    }
}

impl std::fmt::Display for InvalidQueryVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.problem {
            QueryVectorProblem::Empty => write!(f, "Query vector for {} is empty", self.label()),
            QueryVectorProblem::NonFinite { index, value } => write!(
                f,
                "Query vector for {} contains non-finite value at dimension {index}: {value}",
                self.label()
            ),
            QueryVectorProblem::Dimension { actual, expected } => write!(
                f,
                "Query vector has dimension {actual} but {} expects dimension {expected}",
                self.label()
            ),
            QueryVectorProblem::Other(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for InvalidQueryVector {}

/// Attach the identifier the CALLER supplied to an unlabelled
/// [`InvalidQueryVector`] anywhere in `err`'s chain. Used by search
/// authorities that rewrite the caller's collection name to the internal
/// object id before validating: the validator then (correctly) declines to
/// echo the id, and this restores the caller's own name. A label that is
/// already present is never overwritten, and nothing is read from the catalog.
pub fn label_client_collection(mut err: anyhow::Error, requested: &str) -> anyhow::Error {
    if !requested.is_empty()
        && let Some(invalid) = err.downcast_mut::<InvalidQueryVector>()
        && invalid.collection.is_none()
    {
        invalid.collection = Some(requested.to_string());
    }
    err
}

/// The addressed collection does not exist FOR THIS REQUEST: it is missing,
/// or it is not resolvable by the request's tenant. Both cases are this one
/// error with the same text, so a response never distinguishes them. Request
/// boundaries answer it as HTTP 404 / gRPC `NotFound` via
/// [`collection_not_found`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionNotFound {
    /// The collection as the request named it.
    pub identifier: String,
}

impl CollectionNotFound {
    /// A not-found error for the identifier the request used.
    pub fn new(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
        }
    }
}

impl std::fmt::Display for CollectionNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Collection '{}' not found", self.identifier)
    }
}

impl std::error::Error for CollectionNotFound {}

/// A collection drop the server refused or could not complete (catalog or
/// metadata delete failed). Distinct from [`CollectionNotFound`]: the
/// collection still exists, so a surface must report a failure — never
/// success, and never "does not exist" (which an idempotent client would
/// treat as done).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionDropFailed {
    /// The collection as the request named it.
    pub identifier: String,
    /// The service's machine-readable failure code (e.g.
    /// `CATALOG_DELETE_FAILED`).
    pub code: String,
}

impl std::fmt::Display for CollectionDropFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Collection '{}' could not be dropped ({}); it still exists, retry the drop",
            self.identifier, self.code
        )
    }
}

impl std::error::Error for CollectionDropFailed {}

/// The [`CollectionDropFailed`] anywhere in `err`'s chain, if any.
pub fn collection_drop_failed(err: &anyhow::Error) -> Option<&CollectionDropFailed> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<CollectionDropFailed>())
}

/// The [`CollectionNotFound`] anywhere in `err`'s chain, if any.
pub fn collection_not_found(err: &anyhow::Error) -> Option<&CollectionNotFound> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<CollectionNotFound>())
}

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
        let err = Err::<(), _>(InvalidQueryVector::new(
            Some("orders".into()),
            QueryVectorProblem::Dimension {
                actual: 3,
                expected: 4,
            },
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

    /// The caller's identifier is attached through context wrappers, only to
    /// an UNLABELLED error, and never replaces an existing label.
    #[test]
    fn label_client_collection_fills_only_a_missing_label() {
        let dimension = QueryVectorProblem::Dimension {
            actual: 3,
            expected: 4,
        };
        let unlabelled = anyhow::Error::new(InvalidQueryVector::new(None, dimension.clone()))
            .context("search failed");
        assert_eq!(
            client_argument_message(&unlabelled).as_deref(),
            Some("Query vector has dimension 3 but the collection expects dimension 4")
        );
        let labelled = label_client_collection(unlabelled, "orders");
        assert_eq!(
            client_argument_message(&labelled).as_deref(),
            Some("Query vector has dimension 3 but collection 'orders' expects dimension 4")
        );
        let relabelled = label_client_collection(labelled, "someone_else");
        assert!(
            client_argument_message(&relabelled).is_some_and(|m| m.contains("'orders'")),
            "an existing label must not be overwritten"
        );
        // Non-client errors pass through untouched.
        let server = label_client_collection(anyhow::anyhow!("disk full"), "orders");
        assert_eq!(server.to_string(), "disk full");
    }

    #[test]
    fn collection_not_found_is_found_through_context() {
        let err = anyhow::Error::new(CollectionNotFound::new("docs")).context("search failed");
        assert_eq!(
            collection_not_found(&err)
                .map(ToString::to_string)
                .as_deref(),
            Some("Collection 'docs' not found")
        );
        assert!(!is_client_argument_error(&err));
        assert!(collection_not_found(&anyhow::anyhow!("Collection docs not found")).is_none());
    }
}
