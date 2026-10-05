//! # REST API v2 Contracts
//!
//! Version 2 REST request/response contracts for multimodal and agentic APIs.

pub mod agentic;
pub mod query;
pub mod schema;
pub mod sql;

pub use agentic::{
    AgentCheckpointRequest, AgentCheckpointResponse, AgentEventAppendRequest,
    AgentEventAppendResponse, AgentEventRecord, AgentEventReplayRequest, AgentMemoryItem,
    AgentMemoryPutRequest, AgentMemorySearchRequest, AgentMemorySearchResponse, AgenticApiError,
    AgenticErrorBody, AgenticRestBackend,
};
