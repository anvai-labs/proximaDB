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

//! REST API handlers for ProximaDB
//!
//! Provides HTTP/JSON endpoints that delegate to the unified service layer

pub use crate::api_handlers;
/// Canonical REST API handlers — the live `/api/v2` surface (collections,
/// vectors, graph, entities, rank, document, observability, multimodal, hybrid).
pub mod canonical;
/// Health and readiness check endpoints
pub mod health;
/// MLflow-compatible tracking wire (TD-MLOPS-1 slice 2) — default-OFF
/// compatibility surface mounted at /api/2.0/mlflow behind
/// PROXIMADB_MLFLOW_COMPAT_ENABLE.
pub mod mlflow;
/// OpenAPI spec-from-code aggregation + generator (TD-126 Phase 1).
pub mod openapi;
/// Progressive multi-stage search with explain endpoint
pub mod progressive_search_handler;
/// Proto-JSON bidirectional serialization for REST responses
pub mod proto_json;
/// REST HTTP server setup and route configuration
pub mod server;
/// V2 REST API handlers (ProximaRecord, typed fields, schema)
pub mod v2;
/// WebSocket support for real-time streaming
pub mod websocket;

pub use api_handlers::*;
pub use server::*;
// Re-export handlers from canonical
pub use canonical::handlers;
