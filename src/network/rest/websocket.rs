/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! WebSocket streaming handler for real-time vector ingestion
//!
//! This module provides WebSocket endpoints for streaming vector data,
//! complementing the gRPC streaming service with a browser-friendly protocol.
//!
//! ## Endpoints
//!
//! - `GET /v1/stream/insert/:collection` - WebSocket upgrade for vector streaming
//! - `GET /v1/stream/subscribe/:collection` - WebSocket upgrade for live query results
//!
//! ## Message Protocol
//!
//! Messages use JSON for simplicity and browser compatibility:
//!
//! ### StreamInsert Request
//! ```json
//! {
//!     "type": "insert",
//!     "vectors": [
//!         {"id": "vec_1", "vector": [0.1, 0.2, ...], "metadata": {"key": "value"}}
//!     ],
//!     "sequence": 1
//! }
//! ```
//!
//! ### StreamInsert Response
//! ```json
//! {
//!     "type": "ack",
//!     "acked_sequences": [1],
//!     "backpressure": {"level": "none", "suggested_delay_ms": 0, "buffer_percent": 25},
//!     "vectors_buffered": 10,
//!     "vectors_dropped": 0
//! }
//! ```
//!
//! ### Subscribe Request
//! ```json
//! {
//!     "type": "subscribe",
//!     "vector": [0.1, 0.2, ...],
//!     "top_k": 10,
//!     "score_threshold": 0.8
//! }
//! ```
//!
//! ### Query Update (Server -> Client)
//! ```json
//! {
//!     "type": "update",
//!     "update_type": "insert",
//!     "results": [...],
//!     "position": 0,
//!     "timestamp": 1234567890
//! }
//! ```

use std::sync::Arc;

use axum::{
    Router,
    extract::{
        Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::streaming::{
    BackpressureLevel, SessionConfig, StreamConfig, StreamCoordinator, StreamId,
};

/// Safely serialize a message to JSON, logging errors without panicking
fn safe_serialize<T: Serialize>(msg: &T) -> Option<String> {
    match serde_json::to_string(msg) {
        Ok(json) => Some(json),
        Err(e) => {
            error!(error = %e, "Failed to serialize message to JSON");
            None
        }
    }
}

/// WebSocket streaming state shared across handlers
#[derive(Clone)]
pub struct WebSocketState {
    /// Stream coordinator for managing sessions
    pub coordinator: Arc<StreamCoordinator>,
}

impl WebSocketState {
    /// Create a new WebSocket state with default configuration
    pub fn new() -> Self {
        Self {
            coordinator: Arc::new(StreamCoordinator::new(StreamConfig::default())),
        }
    }

    /// Create a new WebSocket state with custom configuration
    pub fn with_config(config: StreamConfig) -> Self {
        Self {
            coordinator: Arc::new(StreamCoordinator::new(config)),
        }
    }

    /// Create a new WebSocket state with an existing coordinator
    pub fn with_coordinator(coordinator: Arc<StreamCoordinator>) -> Self {
        Self { coordinator }
    }
}

impl Default for WebSocketState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create WebSocket streaming routes
///
/// Returns an Axum router with WebSocket endpoints for vector streaming.
pub fn websocket_routes(state: WebSocketState) -> Router {
    Router::new()
        .route("/v1/stream/insert/{collection}", get(ws_insert_handler))
        .route(
            "/v1/stream/subscribe/{collection}",
            get(ws_subscribe_handler),
        )
        .route("/v1/stream/status/{session_id}", get(ws_status_handler))
        .with_state(state)
}

// ============================================================================
// Message Types
// ============================================================================

/// Client-to-server insert message
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Insert(InsertMessage),
    Subscribe(SubscribeMessage),
    Close,
    Ping,
}

/// Insert vectors message
#[derive(Debug, Deserialize)]
struct InsertMessage {
    vectors: Vec<VectorInput>,
    sequence: u64,
    #[serde(default)]
    #[allow(dead_code)]
    partition_key: Option<String>,
}

/// Vector input from client
#[derive(Debug, Deserialize)]
struct VectorInput {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Subscribe to live query results
#[derive(Debug, Deserialize)]
struct SubscribeMessage {
    #[allow(dead_code)]
    vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: u32,
    #[serde(default)]
    #[allow(dead_code)]
    score_threshold: f32,
    #[serde(default)]
    include_initial: bool,
}

fn default_top_k() -> u32 {
    10
}

/// Server-to-client response message
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    Ack(AckMessage),
    Error(ErrorMessage),
    Update(UpdateMessage),
    Heartbeat(HeartbeatMessage),
    SessionCreated(SessionCreatedMessage),
    Pong,
}

/// Acknowledgment message
#[derive(Debug, Serialize)]
struct AckMessage {
    acked_sequences: Vec<u64>,
    backpressure: BackpressureInfo,
    vectors_buffered: u32,
    vectors_dropped: u32,
}

/// Backpressure information
#[derive(Debug, Serialize)]
struct BackpressureInfo {
    level: String,
    suggested_delay_ms: u32,
    buffer_percent: u32,
}

impl From<BackpressureLevel> for BackpressureInfo {
    fn from(level: BackpressureLevel) -> Self {
        let level_str = match level {
            BackpressureLevel::None => "none",
            BackpressureLevel::Low => "low",
            BackpressureLevel::Medium => "medium",
            BackpressureLevel::High => "high",
            BackpressureLevel::Critical => "critical",
        };
        Self {
            level: level_str.to_string(),
            suggested_delay_ms: level.delay_ms(),
            buffer_percent: 0, // Set by caller
        }
    }
}

/// Error message
#[derive(Debug, Serialize)]
struct ErrorMessage {
    code: String,
    message: String,
}

/// Query update message
#[derive(Debug, Serialize)]
struct UpdateMessage {
    update_type: String,
    results: Vec<serde_json::Value>,
    position: u32,
    timestamp: u64,
    total_count: u32,
}

/// Heartbeat message
#[derive(Debug, Serialize)]
struct HeartbeatMessage {
    timestamp: u64,
}

/// Session created message
#[derive(Debug, Serialize)]
struct SessionCreatedMessage {
    session_id: String,
    buffer_size: u32,
    expires_in_seconds: u32,
}

// ============================================================================
// Handlers
// ============================================================================

/// WebSocket upgrade handler for vector insertion
async fn ws_insert_handler(
    ws: WebSocketUpgrade,
    Path(collection): Path<String>,
    State(state): State<WebSocketState>,
) -> impl IntoResponse {
    info!(collection = %collection, "WebSocket insert connection request");
    ws.on_upgrade(move |socket| handle_insert_socket(socket, collection, state))
}

/// Handle WebSocket connection for vector insertion
async fn handle_insert_socket(socket: WebSocket, collection: String, state: WebSocketState) {
    let (mut sender, mut receiver) = socket.split();

    // Create session
    let session_id = match state
        .coordinator
        .create_session(collection.clone(), SessionConfig::default())
        .await
    {
        Ok(id) => id,
        Err(e) => {
            error!(error = %e, "Failed to create streaming session");
            let error_msg = ServerMessage::Error(ErrorMessage {
                code: "SESSION_CREATE_FAILED".to_string(),
                message: e.to_string(),
            });
            if let Some(json) = safe_serialize(&error_msg) {
                let _ = sender.send(Message::Text(json.into())).await;
            }
            return;
        }
    };

    // Send session created message
    let session_info = state.coordinator.get_session_info(&session_id);
    let created_msg = ServerMessage::SessionCreated(SessionCreatedMessage {
        session_id: session_id.to_string(),
        buffer_size: session_info
            .as_ref()
            .map_or(10000, |s| s.buffer_capacity as u32),
        expires_in_seconds: state.coordinator.config().session_timeout.as_secs() as u32,
    });
    let session_created_json = match safe_serialize(&created_msg) {
        Some(json) => json,
        None => {
            error!("Failed to serialize SessionCreated message");
            state.coordinator.close_session(&session_id);
            return;
        }
    };

    if sender
        .send(Message::Text(session_created_json.into()))
        .await
        .is_err()
    {
        state.coordinator.close_session(&session_id);
        return;
    }

    info!(session_id = %session_id, collection = %collection, "WebSocket insert session created");

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Insert(msg)) => {
                        // Convert incoming JSON vectors to canonical ProximaRecord at protocol boundary
                        let records: Vec<proximadb_records::ProximaRecord> = msg
                            .vectors
                            .into_iter()
                            .map(|v| {
                                let dim = v.vector.len() as u32;
                                let mut props = proximadb_records::ProximaTree::new();
                                for (k, jv) in v.metadata {
                                    let pv = proximadb_records::conversions::json_to_proxima(&jv);
                                    props.insert(k, proximadb_records::ProximaTreeNode::Value(pv));
                                }
                                proximadb_records::ProximaRecord {
                                    oid: v.id,
                                    embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                                        "default", "vector", dim, v.vector,
                                    )],
                                    props,
                                    record_version: 1,
                                    created_at_ns: chrono::Utc::now()
                                        .timestamp_nanos_opt()
                                        .unwrap_or(0),
                                    updated_at_ns: chrono::Utc::now()
                                        .timestamp_nanos_opt()
                                        .unwrap_or(0),
                                    ..Default::default()
                                }
                            })
                            .collect();

                        // Push to coordinator
                        match state.coordinator.push_records(&session_id, records).await {
                            Ok(result) => {
                                let mut bp_info: BackpressureInfo = result.backpressure.into();
                                bp_info.buffer_percent = result.buffer_percent;

                                let ack = ServerMessage::Ack(AckMessage {
                                    acked_sequences: vec![msg.sequence],
                                    backpressure: bp_info,
                                    vectors_buffered: result.pushed as u32,
                                    vectors_dropped: result.dropped as u32,
                                });

                                if let Some(ack_json) = safe_serialize(&ack) {
                                    if sender.send(Message::Text(ack_json.into())).await.is_err() {
                                        break;
                                    }
                                } else {
                                    error!("Failed to serialize Ack message");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!(session_id = %session_id, error = %e, "Push failed");
                                let error_msg = ServerMessage::Error(ErrorMessage {
                                    code: "PUSH_FAILED".to_string(),
                                    message: e.to_string(),
                                });
                                if let Some(error_json) = safe_serialize(&error_msg) {
                                    let _ = sender.send(Message::Text(error_json.into())).await;
                                }
                            }
                        }
                    }
                    Ok(ClientMessage::Ping) => {
                        let pong = ServerMessage::Pong;
                        if let Some(pong_json) = safe_serialize(&pong) {
                            let _ = sender.send(Message::Text(pong_json.into())).await;
                        }
                    }
                    Ok(ClientMessage::Close) => {
                        debug!(session_id = %session_id, "Client requested close");
                        break;
                    }
                    Ok(_) => {
                        let error_msg = ServerMessage::Error(ErrorMessage {
                            code: "INVALID_MESSAGE".to_string(),
                            message: "Expected insert or close message".to_string(),
                        });
                        if let Some(error_json) = safe_serialize(&error_msg) {
                            let _ = sender.send(Message::Text(error_json.into())).await;
                        }
                    }
                    Err(e) => {
                        warn!(session_id = %session_id, error = %e, "Invalid message format");
                        let error_msg = ServerMessage::Error(ErrorMessage {
                            code: "PARSE_ERROR".to_string(),
                            message: e.to_string(),
                        });
                        if let Some(error_json) = safe_serialize(&error_msg) {
                            let _ = sender.send(Message::Text(error_json.into())).await;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                debug!(session_id = %session_id, "WebSocket closed by client");
                break;
            }
            Ok(Message::Ping(data)) => {
                let _ = sender.send(Message::Pong(data)).await;
            }
            Ok(_) => {}
            Err(e) => {
                error!(session_id = %session_id, error = %e, "WebSocket error");
                break;
            }
        }
    }

    // Cleanup
    state.coordinator.close_session(&session_id);
    info!(session_id = %session_id, "WebSocket insert session closed");
}

/// WebSocket upgrade handler for query subscription
async fn ws_subscribe_handler(
    ws: WebSocketUpgrade,
    Path(collection): Path<String>,
    State(state): State<WebSocketState>,
) -> impl IntoResponse {
    info!(collection = %collection, "WebSocket subscribe connection request");
    ws.on_upgrade(move |socket| handle_subscribe_socket(socket, collection, state))
}

/// Handle WebSocket connection for live query subscription
async fn handle_subscribe_socket(socket: WebSocket, collection: String, _state: WebSocketState) {
    let (mut sender, mut receiver) = socket.split();

    let subscription_id = StreamId::new().to_string();
    info!(subscription_id = %subscription_id, collection = %collection, "WebSocket subscription started");

    // Wait for subscribe message
    let subscribe_msg = loop {
        match receiver.next().await {
            Some(Ok(Message::Text(text))) => match serde_json::from_str::<ClientMessage>(&text) {
                Ok(ClientMessage::Subscribe(msg)) => break msg,
                Ok(ClientMessage::Close) => return,
                Ok(_) => {
                    let error_msg = ServerMessage::Error(ErrorMessage {
                        code: "EXPECTED_SUBSCRIBE".to_string(),
                        message: "First message must be a subscribe message".to_string(),
                    });
                    if let Some(error_json) = safe_serialize(&error_msg) {
                        let _ = sender.send(Message::Text(error_json.into())).await;
                    }
                }
                Err(e) => {
                    let error_msg = ServerMessage::Error(ErrorMessage {
                        code: "PARSE_ERROR".to_string(),
                        message: e.to_string(),
                    });
                    if let Some(error_json) = safe_serialize(&error_msg) {
                        let _ = sender.send(Message::Text(error_json.into())).await;
                    }
                }
            },
            Some(Ok(Message::Close(_))) | None => return,
            _ => {}
        }
    };

    debug!(
        subscription_id = %subscription_id,
        top_k = subscribe_msg.top_k,
        "Subscription parameters received"
    );

    // Send initial snapshot if requested
    if subscribe_msg.include_initial {
        let initial = ServerMessage::Update(UpdateMessage {
            update_type: "initial".to_string(),
            results: vec![], // Deferred: Perform initial query
            position: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            total_count: 0,
        });
        let initial_json = match safe_serialize(&initial) {
            Some(json) => json,
            None => {
                error!("Failed to serialize initial Update message");
                return;
            }
        };

        if sender
            .send(Message::Text(initial_json.into()))
            .await
            .is_err()
        {
            return;
        }
    }

    // Send periodic heartbeats while waiting for client messages
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        tokio::select! {
            // Heartbeat
            _ = interval.tick() => {
                let heartbeat = ServerMessage::Heartbeat(HeartbeatMessage {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0),
                });
                if let Some(heartbeat_json) = safe_serialize(&heartbeat) {
                    if sender.send(Message::Text(heartbeat_json.into())).await.is_err() {
                        break;
                    }
                } else {
                    error!("Failed to serialize Heartbeat message");
                    break;
                }
            }
            // Client message
            result = receiver.next() => {
                match result {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(ClientMessage::Close) = serde_json::from_str(&text) {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sender.send(Message::Pong(data)).await;
                    }
                    Some(Err(_)) | None => break,
                    _ => {}
                }
            }
        }
    }
    info!(subscription_id = %subscription_id, "WebSocket subscription closed");
}

/// WebSocket status handler (non-upgrade endpoint for session status)
async fn ws_status_handler(
    Path(session_id): Path<String>,
    State(state): State<WebSocketState>,
) -> impl IntoResponse {
    let stream_id = StreamId::from_string(session_id.clone());
    match state.coordinator.get_session_info(&stream_id) {
        Some(info) => {
            let status = serde_json::json!({
                "session_id": session_id,
                "collection": info.collection,
                "state": format!("{:?}", info.state),
                "records_received": info.records_received,
                "records_processed": info.records_processed,
                "buffer_length": info.buffer_len,
                "buffer_capacity": info.buffer_capacity,
                "age_seconds": info.age_secs,
                "idle_seconds": info.idle_secs,
            });
            axum::Json(status).into_response()
        }
        None => (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({
                "error": "Session not found",
                "session_id": session_id
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_state_creation() {
        let state = WebSocketState::new();
        assert_eq!(state.coordinator.session_count(), 0);
    }

    #[test]
    fn test_backpressure_info_conversion() {
        let info: BackpressureInfo = BackpressureLevel::None.into();
        assert_eq!(info.level, "none");
        assert_eq!(info.suggested_delay_ms, 0);

        let info: BackpressureInfo = BackpressureLevel::Critical.into();
        assert_eq!(info.level, "critical");
        assert_eq!(info.suggested_delay_ms, 1000);
    }

    #[test]
    fn test_client_message_parsing() {
        let json = r#"{"type": "insert", "vectors": [], "sequence": 1}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Insert(_)));

        let json = r#"{"type": "close"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Close));

        let json = r#"{"type": "ping"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Ping));
    }

    #[test]
    fn test_server_message_serialization() {
        let msg = ServerMessage::Ack(AckMessage {
            acked_sequences: vec![1, 2, 3],
            backpressure: BackpressureInfo {
                level: "none".to_string(),
                suggested_delay_ms: 0,
                buffer_percent: 25,
            },
            vectors_buffered: 100,
            vectors_dropped: 0,
        });

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ack\""));
        assert!(json.contains("\"acked_sequences\":[1,2,3]"));
    }

    #[test]
    fn test_subscribe_message_defaults() {
        let json = r#"{"type": "subscribe", "vector": [0.1, 0.2]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        if let ClientMessage::Subscribe(sub) = msg {
            assert_eq!(sub.top_k, 10); // default
            assert_eq!(sub.score_threshold, 0.0); // default
            assert!(!sub.include_initial); // default
        } else {
            panic!("Expected Subscribe message");
        }
    }
}
