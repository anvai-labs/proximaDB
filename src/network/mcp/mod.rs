// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! In-process reference MCP transport (ADR-037 Decision 5).
//!
//! Exposes the reference MCP surface as another transport on the engine — an
//! HTTP JSON-RPC endpoint, bound **only when `api.mcp_port` is configured** (off
//! by default, so an un-configured deployment runs no MCP surface and no extra
//! process). Reuses the `proximadb-mcp` crate (protocol + tool catalog +
//! dispatch) over the in-process [`EngineBackend`], which calls the engine's
//! services directly.
//!
//! Framing: request/response JSON-RPC over `POST` (one message per request). SSE
//! / streamable-HTTP for server-initiated messages is a later refinement;
//! the data tools (list/describe/stats/explain/search) are request/response.

pub mod backend;

pub use backend::EngineBackend;

use axum::{Router, extract::State, response::Json, routing::post};
use proximadb_mcp::{JsonRpcRequest, handle_request};
use serde_json::{Value, json};
use std::sync::Arc;

/// Build the MCP HTTP router over the given backend (`POST /` and `POST /mcp`).
/// Build the MCP router. `security_coordinator` (TD-AUTH-FC): when present,
/// the SAME `auth_middleware_unified` REST uses is attached — MCP is a fully
/// privileged data-plane surface and no longer serves unauthenticated JSON-RPC
/// when a security subsystem is active. `None` ⇒ unauthenticated (dev
/// posture; `api.mcp_port` defaults off).
pub fn router(
    backend: Arc<EngineBackend>,
    security_coordinator: Option<std::sync::Arc<crate::security::SecurityCoordinator>>,
) -> Router {
    let router = Router::new()
        .route("/", post(rpc))
        .route("/mcp", post(rpc))
        .with_state(backend);
    match security_coordinator {
        Some(coordinator) => router.layer(axum::middleware::from_fn_with_state(
            coordinator,
            crate::network::auth::middleware::auth_middleware_unified,
        )),
        None => router,
    }
}

async fn rpc(State(backend): State<Arc<EngineBackend>>, body: String) -> Json<Value> {
    let req: JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            return Json(json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": { "code": -32700, "message": format!("parse error: {e}") }
            }));
        }
    };
    match handle_request(backend.as_ref(), req).await {
        Some(resp) => Json(serde_json::to_value(resp).unwrap_or_else(|_| {
            json!({ "jsonrpc": "2.0", "id": Value::Null, "error": { "code": -32603, "message": "serialize failed" } })
        })),
        // Notification (no id) → no response body.
        None => Json(Value::Null),
    }
}

/// Serve the MCP transport on `addr` until shutdown. Call only when an MCP port
/// is configured.
pub async fn serve(
    addr: std::net::SocketAddr,
    backend: Arc<EngineBackend>,
    security_coordinator: Option<std::sync::Arc<crate::security::SecurityCoordinator>>,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        "MCP reference surface listening on {addr} (MCP {}; auth: {})",
        proximadb_mcp::MCP_PROTOCOL_VERSION,
        if security_coordinator.is_some() {
            "coordinator"
        } else {
            "none (dev posture)"
        }
    );
    axum::serve(listener, router(backend, security_coordinator)).await?;
    Ok(())
}
