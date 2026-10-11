/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

//! REST endpoints for per-collection pinning (Phase 6 control surface).
//!
//! Matches the operator UX turbopuffer documents at
//! `/docs/pinning`: `PATCH /v1/namespaces/:namespace/metadata` returns
//! immediately with the new pin state; physical data movement happens
//! out of band (the access-pattern engine reads the registry on its
//! next evaluation).
//!
//! Routes:
//!
//! * `PATCH /api/v2/collections/:collection_id/pin` — set or clear a
//!   pin. Body: `{ "pinned": true, "target": "nvme_ssd", "replicas": 1 }`
//!   or `{ "pinned": false }`.
//! * `GET /api/v2/collections/:collection_id/pin` — read current pin
//!   state. 200 with state when pinned; 200 with `{"status":"unpinned"}`
//!   when not pinned.
//! * `GET /api/v2/collections/pinning` — list all pinned collections.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use crate::network::rest::canonical::handlers::AppState;
use crate::storage::collection_pinning::{CollectionPinTarget, PinState};

/// PATCH body. Two shapes supported:
///
/// ```json
/// { "pinned": true, "target": "nvme_ssd", "replicas": 2 }
/// { "pinned": false }
/// ```
#[derive(Debug, Deserialize)]
pub struct PinRequest {
    /// Whether to pin (`true`) or unpin (`false`).
    pub pinned: bool,
    /// Required when `pinned` is true; ignored otherwise.
    #[serde(default)]
    pub target: Option<CollectionPinTarget>,
    /// Optional when `pinned` is true; defaults to 1 if omitted.
    #[serde(default)]
    pub replicas: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PinResponse {
    Pinned {
        collection_id: String,
        target: CollectionPinTarget,
        replicas: u32,
        pinned_at_ns: i64,
    },
    Unpinned {
        collection_id: String,
        /// Echoed when the request unpinned a previously-pinned
        /// collection; helpful for operator audit logs.
        was_pinned: bool,
    },
}

impl PinResponse {
    fn pinned(collection_id: String, state: &PinState) -> Self {
        Self::Pinned {
            collection_id,
            target: state.target,
            replicas: state.replicas,
            pinned_at_ns: state.pinned_at_ns,
        }
    }
}

/// `PATCH /api/v2/collections/:collection_id/pin`
pub async fn patch_pin(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
    Json(body): Json<PinRequest>,
) -> Result<(StatusCode, Json<PinResponse>), (StatusCode, String)> {
    let registry = &state.pin_registry;

    if body.pinned {
        let target = body.target.ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "`target` is required when `pinned` is true".to_string(),
            )
        })?;
        let replicas = body.replicas.unwrap_or(1);
        let new_state = registry.pin(collection_id.clone(), target, replicas);
        Ok((
            StatusCode::OK,
            Json(PinResponse::pinned(collection_id, &new_state)),
        ))
    } else {
        let was_pinned = registry.unpin(&collection_id).is_some();
        Ok((
            StatusCode::OK,
            Json(PinResponse::Unpinned {
                collection_id,
                was_pinned,
            }),
        ))
    }
}

/// `GET /api/v2/collections/:collection_id/pin`
pub async fn get_pin(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Result<Json<PinResponse>, (StatusCode, String)> {
    let registry = &state.pin_registry;

    match registry.get(&collection_id) {
        Some(s) => Ok(Json(PinResponse::pinned(collection_id, &s))),
        None => Ok(Json(PinResponse::Unpinned {
            collection_id,
            was_pinned: false,
        })),
    }
}

#[derive(Debug, Serialize)]
pub struct PinListItem {
    pub collection_id: String,
    pub target: CollectionPinTarget,
    pub replicas: u32,
    pub pinned_at_ns: i64,
}

#[derive(Debug, Serialize)]
pub struct PinListResponse {
    pub count: usize,
    pub items: Vec<PinListItem>,
}

/// `GET /api/v2/collections/pinning` — operator-dashboard view of
/// every currently pinned collection.
pub async fn list_pins(
    State(state): State<AppState>,
) -> Result<Json<PinListResponse>, (StatusCode, String)> {
    let registry = &state.pin_registry;

    let items: Vec<PinListItem> = registry
        .list()
        .into_iter()
        .map(|(collection_id, state)| PinListItem {
            collection_id,
            target: state.target,
            replicas: state.replicas,
            pinned_at_ns: state.pinned_at_ns,
        })
        .collect();
    let count = items.len();
    Ok(Json(PinListResponse { count, items }))
}
