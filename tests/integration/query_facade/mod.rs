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

//! # Query Facade Adapter Integration Tests
//!
//! This module contains integration tests for the `QueryFacadeAdapter` and
//! `UnifiedQueryFacade` that validate the unified query routing architecture.
//!
//! ## Test Organization
//!
//! - `test_vector_through_facade.rs`: Vector search routing tests
//! - `test_sql_through_facade.rs`: SQL query routing tests
//! - `test_graph_through_facade.rs`: Graph query routing tests
//! - `test_rest_grpc_parity.rs`: REST/gRPC parity tests (feature: unified-facade-routing)
//! - `test_explain_plan_consistency.rs`: Explain plan format consistency tests
//!
//! ## Feature Flag
//!
//! These tests validate the `unified-facade-routing` feature flag behavior.
//! When enabled, all queries route through `UnifiedQueryFacade` for consistent
//! execution. When disabled, legacy direct handler paths are used.
//!
//! ## Architecture Validation
//!
//! ```text
//! REST/gRPC Handler
//!        |
//!        v
//! QueryFacadeAdapter
//!        |
//!        v
//! UnifiedQueryFacade
//!        |
//!   +----+----+----+
//!   |    |    |    |
//!   v    v    v    v
//! Vector SQL Graph ...
//! Strategy Strategy Strategy
//! ```

pub mod test_graph_through_facade;
pub mod test_sql_through_facade;
pub mod test_vector_through_facade;

/// REST/gRPC parity tests - validates consistent results between API protocols
pub mod test_rest_grpc_parity;

/// Dual-transport RENDERING parity: real REST router vs real gRPC service over
/// ONE query facade (the file above tests facade idempotence — see its honest
/// scope note; this one exercises two genuinely different renderer stacks).
pub mod test_rest_grpc_rendering_parity;

/// Same-SQL transport parity: the REAL v2 router and the REAL v2 record
/// service over ONE SharedServices stack (the strongest pairing — both legs
/// converge on the same api_handlers.execute_sql).
pub mod test_sql_transport_rendering_parity;

/// Explain plan consistency tests - validates explain plan format across query types
pub mod test_explain_plan_consistency;
