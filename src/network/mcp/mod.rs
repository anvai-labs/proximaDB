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

/// Build the local-only MCP router. Authenticated deployments are refused
/// until the tool port consumes request-bound tenant identity and permissions.
fn router(
    backend: Arc<EngineBackend>,
    security_coordinator: Option<std::sync::Arc<crate::security::SecurityCoordinator>>,
) -> anyhow::Result<Router> {
    anyhow::ensure!(
        security_coordinator.is_none(),
        "MCP has no request-bound authorization; disable api.mcp_port when security is enabled"
    );
    let router = Router::new()
        .route("/", post(rpc))
        .route("/mcp", post(rpc))
        .with_state(backend);
    Ok(router)
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
    proximadb_runtime::bootstrap_config::validate_trust_only_listener(
        "MCP",
        addr,
        security_coordinator.is_some(),
        false,
        &proximadb_tenant::TenantDeploymentMode::single_tenant_default(),
    )?;
    let router = router(backend, security_coordinator)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        "MCP local reference surface listening on {addr} (MCP {})",
        proximadb_mcp::MCP_PROTOCOL_VERSION,
    );
    axum::serve(listener, router).await?;
    Ok(())
}
