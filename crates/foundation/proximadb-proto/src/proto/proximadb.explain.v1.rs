// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// This file is generated from proximadb/explain.proto
// DO NOT EDIT MANUALLY

// ExplainPlan provides a unified explain format across all APIs
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExplainPlan {
    #[prost(string, tag = "1")]
    pub plan_id: ::prost::alloc::string::String,
    #[prost(enumeration = "QueryType", tag = "2")]
    pub query_type: i32,
    #[prost(string, tag = "3")]
    pub query: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub optimized_query: ::prost::alloc::string::String,
    #[prost(message, repeated, tag = "5")]
    pub plan_nodes: ::prost::alloc::vec::Vec<PlanNode>,
    #[prost(message, optional, tag = "6")]
    pub execution_stats: ::core::option::Option<ExecutionStats>,
    #[prost(message, repeated, tag = "7")]
    pub warnings: ::prost::alloc::vec::Vec<ExplainWarning>,
    #[prost(message, optional, tag = "8")]
    pub metadata: ::core::option::Option<PlanMetadata>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum QueryType {
    QueryTypeUnknown = 0,
    QueryTypeSql = 1,
    QueryTypeUql = 2,
    QueryTypeVectorSearch = 3,
    QueryTypeGraphQuery = 4,
    QueryTypeDocumentQuery = 5,
    QueryTypeHybrid = 6,
    QueryTypeAggregation = 7,
}

impl QueryType {
    pub fn as_str_name(&self) -> &'static str {
        match self {
            QueryType::QueryTypeUnknown => "QUERY_TYPE_UNKNOWN",
            QueryType::QueryTypeSql => "QUERY_TYPE_SQL",
            QueryType::QueryTypeUql => "QUERY_TYPE_UQL",
            QueryType::QueryTypeVectorSearch => "QUERY_TYPE_VECTOR_SEARCH",
            QueryType::QueryTypeGraphQuery => "QUERY_TYPE_GRAPH_QUERY",
            QueryType::QueryTypeDocumentQuery => "QUERY_TYPE_DOCUMENT_QUERY",
            QueryType::QueryTypeHybrid => "QUERY_TYPE_HYBRID",
            QueryType::QueryTypeAggregation => "QUERY_TYPE_AGGREGATION",
        }
    }
    /// Creates an enum from the proto field names (round 8: the decode
    /// direction was simply absent — JSON/enum-name decoding of explain
    /// types never worked).
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "QUERY_TYPE_UNKNOWN" => Some(Self::QueryTypeUnknown),
            "QUERY_TYPE_SQL" => Some(Self::QueryTypeSql),
            "QUERY_TYPE_UQL" => Some(Self::QueryTypeUql),
            "QUERY_TYPE_VECTOR_SEARCH" => Some(Self::QueryTypeVectorSearch),
            "QUERY_TYPE_GRAPH_QUERY" => Some(Self::QueryTypeGraphQuery),
            "QUERY_TYPE_DOCUMENT_QUERY" => Some(Self::QueryTypeDocumentQuery),
            "QUERY_TYPE_HYBRID" => Some(Self::QueryTypeHybrid),
            "QUERY_TYPE_AGGREGATION" => Some(Self::QueryTypeAggregation),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PlanNode {
    #[prost(string, tag = "1")]
    pub node_id: ::prost::alloc::string::String,
    #[prost(enumeration = "NodeType", tag = "2")]
    pub node_type: i32,
    #[prost(string, tag = "3")]
    pub display_name: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub description: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "5")]
    pub parent_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag = "6")]
    pub child_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(double, tag = "7")]
    pub estimated_cost: f64,
    #[prost(int64, tag = "8")]
    pub estimated_rows: i64,
    #[prost(int64, tag = "9")]
    pub actual_rows: i64,
    #[prost(message, optional, tag = "10")]
    pub node_details: ::core::option::Option<NodeDetails>,
    #[prost(message, optional, tag = "11")]
    pub node_stats: ::core::option::Option<NodeStats>,
    #[prost(string, repeated, tag = "12")]
    pub hints: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum NodeType {
    NodeTypeUnknown = 0,
    NodeTypeScan = 1,
    NodeTypeIndexScan = 2,
    NodeTypeVectorIndexScan = 3,
    NodeTypeGraphScan = 4,
    NodeTypeDocumentScan = 5,
    NodeTypeFilter = 10,
    NodeTypeProject = 11,
    NodeTypeAggregate = 12,
    NodeTypeSort = 13,
    NodeTypeLimit = 14,
    NodeTypeJoin = 15,
    NodeTypeUnion = 16,
    NodeTypeDistinct = 17,
    NodeTypeVectorSearch = 20,
    NodeTypeDistanceCompute = 21,
    NodeTypeQuantization = 22,
    NodeTypeGraphTraversal = 30,
    NodeTypePatternMatch = 31,
    NodeTypePathFinding = 32,
    NodeTypeDistributedScan = 40,
    NodeTypeDistributedAggregate = 41,
    NodeTypeShuffle = 42,
    NodeTypeExchange = 43,
    NodeTypeHybridSearch = 50,
    NodeTypeFilterContract = 51,
    NodeTypeCandidateSet = 52,
    /// Multi-phase rank pipeline (RERANK SRF surface, R-7c.4c.1)
    NodeTypeRerank = 53,
    NodeTypeMock = 99,
}

impl NodeType {
    pub fn as_str_name(&self) -> &'static str {
        match self {
            NodeType::NodeTypeUnknown => "NODE_TYPE_UNKNOWN",
            NodeType::NodeTypeScan => "NODE_TYPE_SCAN",
            NodeType::NodeTypeIndexScan => "NODE_TYPE_INDEX_SCAN",
            NodeType::NodeTypeVectorIndexScan => "NODE_TYPE_VECTOR_INDEX_SCAN",
            NodeType::NodeTypeGraphScan => "NODE_TYPE_GRAPH_SCAN",
            NodeType::NodeTypeDocumentScan => "NODE_TYPE_DOCUMENT_SCAN",
            NodeType::NodeTypeFilter => "NODE_TYPE_FILTER",
            NodeType::NodeTypeProject => "NODE_TYPE_PROJECT",
            NodeType::NodeTypeAggregate => "NODE_TYPE_AGGREGATE",
            NodeType::NodeTypeSort => "NODE_TYPE_SORT",
            NodeType::NodeTypeLimit => "NODE_TYPE_LIMIT",
            NodeType::NodeTypeJoin => "NODE_TYPE_JOIN",
            NodeType::NodeTypeUnion => "NODE_TYPE_UNION",
            NodeType::NodeTypeDistinct => "NODE_TYPE_DISTINCT",
            NodeType::NodeTypeVectorSearch => "NODE_TYPE_VECTOR_SEARCH",
            NodeType::NodeTypeDistanceCompute => "NODE_TYPE_DISTANCE_COMPUTE",
            NodeType::NodeTypeQuantization => "NODE_TYPE_QUANTIZATION",
            NodeType::NodeTypeGraphTraversal => "NODE_TYPE_GRAPH_TRAVERSAL",
            NodeType::NodeTypePatternMatch => "NODE_TYPE_PATTERN_MATCH",
            NodeType::NodeTypePathFinding => "NODE_TYPE_PATH_FINDING",
            NodeType::NodeTypeDistributedScan => "NODE_TYPE_DISTRIBUTED_SCAN",
            NodeType::NodeTypeDistributedAggregate => "NODE_TYPE_DISTRIBUTED_AGGREGATE",
            NodeType::NodeTypeShuffle => "NODE_TYPE_SHUFFLE",
            NodeType::NodeTypeExchange => "NODE_TYPE_EXCHANGE",
            NodeType::NodeTypeHybridSearch => "NODE_TYPE_HYBRID_SEARCH",
            NodeType::NodeTypeFilterContract => "NODE_TYPE_FILTER_CONTRACT",
            NodeType::NodeTypeCandidateSet => "NODE_TYPE_CANDIDATE_SET",
            NodeType::NodeTypeRerank => "NODE_TYPE_RERANK",
            NodeType::NodeTypeMock => "NODE_TYPE_MOCK",
        }
    }
    /// Creates an enum from the proto field names (round 8: the decode
    /// direction was simply absent — JSON/enum-name decoding of explain
    /// types never worked).
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "NODE_TYPE_UNKNOWN" => Some(Self::NodeTypeUnknown),
            "NODE_TYPE_SCAN" => Some(Self::NodeTypeScan),
            "NODE_TYPE_INDEX_SCAN" => Some(Self::NodeTypeIndexScan),
            "NODE_TYPE_VECTOR_INDEX_SCAN" => Some(Self::NodeTypeVectorIndexScan),
            "NODE_TYPE_GRAPH_SCAN" => Some(Self::NodeTypeGraphScan),
            "NODE_TYPE_DOCUMENT_SCAN" => Some(Self::NodeTypeDocumentScan),
            "NODE_TYPE_FILTER" => Some(Self::NodeTypeFilter),
            "NODE_TYPE_PROJECT" => Some(Self::NodeTypeProject),
            "NODE_TYPE_AGGREGATE" => Some(Self::NodeTypeAggregate),
            "NODE_TYPE_SORT" => Some(Self::NodeTypeSort),
            "NODE_TYPE_LIMIT" => Some(Self::NodeTypeLimit),
            "NODE_TYPE_JOIN" => Some(Self::NodeTypeJoin),
            "NODE_TYPE_UNION" => Some(Self::NodeTypeUnion),
            "NODE_TYPE_DISTINCT" => Some(Self::NodeTypeDistinct),
            "NODE_TYPE_VECTOR_SEARCH" => Some(Self::NodeTypeVectorSearch),
            "NODE_TYPE_DISTANCE_COMPUTE" => Some(Self::NodeTypeDistanceCompute),
            "NODE_TYPE_QUANTIZATION" => Some(Self::NodeTypeQuantization),
            "NODE_TYPE_GRAPH_TRAVERSAL" => Some(Self::NodeTypeGraphTraversal),
            "NODE_TYPE_PATTERN_MATCH" => Some(Self::NodeTypePatternMatch),
            "NODE_TYPE_PATH_FINDING" => Some(Self::NodeTypePathFinding),
            "NODE_TYPE_DISTRIBUTED_SCAN" => Some(Self::NodeTypeDistributedScan),
            "NODE_TYPE_DISTRIBUTED_AGGREGATE" => Some(Self::NodeTypeDistributedAggregate),
            "NODE_TYPE_SHUFFLE" => Some(Self::NodeTypeShuffle),
            "NODE_TYPE_EXCHANGE" => Some(Self::NodeTypeExchange),
            "NODE_TYPE_HYBRID_SEARCH" => Some(Self::NodeTypeHybridSearch),
            "NODE_TYPE_FILTER_CONTRACT" => Some(Self::NodeTypeFilterContract),
            "NODE_TYPE_CANDIDATE_SET" => Some(Self::NodeTypeCandidateSet),
            "NODE_TYPE_RERANK" => Some(Self::NodeTypeRerank),
            "NODE_TYPE_MOCK" => Some(Self::NodeTypeMock),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeDetails {
    #[prost(message, optional, tag = "1")]
    pub scan: ::core::option::Option<ScanDetails>,
    #[prost(message, optional, tag = "2")]
    pub index_scan: ::core::option::Option<IndexScanDetails>,
    #[prost(message, optional, tag = "3")]
    pub vector_index_scan: ::core::option::Option<VectorIndexScanDetails>,
    #[prost(message, optional, tag = "4")]
    pub filter: ::core::option::Option<FilterDetails>,
    #[prost(message, optional, tag = "5")]
    pub project: ::core::option::Option<ProjectDetails>,
    #[prost(message, optional, tag = "6")]
    pub aggregate: ::core::option::Option<AggregateDetails>,
    #[prost(message, optional, tag = "7")]
    pub join: ::core::option::Option<JoinDetails>,
    #[prost(message, optional, tag = "8")]
    pub sort: ::core::option::Option<SortDetails>,
    #[prost(message, optional, tag = "9")]
    pub graph_scan: ::core::option::Option<GraphScanDetails>,
    #[prost(message, optional, tag = "10")]
    pub hybrid_search: ::core::option::Option<HybridSearchDetails>,
    #[prost(message, optional, tag = "11")]
    pub rerank: ::core::option::Option<RerankDetails>,
    #[prost(message, optional, tag = "100")]
    pub additional_metadata: ::core::option::Option<::prost_types::Struct>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ScanDetails {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub collection_id: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "3")]
    pub columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "4")]
    pub filter_pushed_down: ::prost::alloc::string::String,
    #[prost(int64, tag = "5")]
    pub estimated_bytes: i64,
    #[prost(bool, tag = "6")]
    pub is_parallel: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IndexScanDetails {
    #[prost(string, tag = "1")]
    pub index_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub index_id: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub index_type: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "4")]
    pub index_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "5")]
    pub index_condition: ::prost::alloc::string::String,
    #[prost(bool, tag = "6")]
    pub is_covering: bool,
    #[prost(int64, tag = "7")]
    pub estimated_entries: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VectorIndexScanDetails {
    #[prost(string, tag = "1")]
    pub index_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub index_type: ::prost::alloc::string::String,
    #[prost(int32, tag = "3")]
    pub vector_dimension: i32,
    #[prost(string, tag = "4")]
    pub distance_metric: ::prost::alloc::string::String,
    #[prost(int32, tag = "5")]
    pub top_k: i32,
    #[prost(int32, tag = "6")]
    pub ef: i32,
    #[prost(int32, tag = "7")]
    pub nprobe: i32,
    #[prost(bool, tag = "8")]
    pub has_filter: bool,
    #[prost(string, tag = "9")]
    pub filter_condition: ::prost::alloc::string::String,
    #[prost(string, tag = "10")]
    pub filter_strategy: ::prost::alloc::string::String,
    #[prost(double, tag = "11")]
    pub selectivity_estimate: f64,
    #[prost(bool, tag = "12")]
    pub early_pruning_enabled: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FilterDetails {
    #[prost(string, tag = "1")]
    pub filter_condition: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "2")]
    pub filter_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "3")]
    pub filter_type: ::prost::alloc::string::String,
    #[prost(double, tag = "4")]
    pub selectivity_estimate: f64,
    #[prost(bool, tag = "5")]
    pub is_sargable: bool,
    #[prost(string, tag = "6")]
    pub index_used: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProjectDetails {
    #[prost(string, repeated, tag = "1")]
    pub projected_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag = "2")]
    pub derived_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag = "3")]
    pub project_expressions: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AggregateDetails {
    #[prost(string, repeated, tag = "1")]
    pub group_by_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(message, repeated, tag = "2")]
    pub aggregate_functions: ::prost::alloc::vec::Vec<AggregateFunction>,
    #[prost(string, tag = "3")]
    pub aggregate_strategy: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AggregateFunction {
    #[prost(string, tag = "1")]
    pub function_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub column_name: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub alias: ::prost::alloc::string::String,
    #[prost(bool, tag = "4")]
    pub is_distinct: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct JoinDetails {
    #[prost(string, tag = "1")]
    pub join_type: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub join_condition: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "3")]
    pub join_columns: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "4")]
    pub join_algorithm: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub build_side: ::prost::alloc::string::String,
    #[prost(int64, tag = "6")]
    pub estimated_build_rows: i64,
    #[prost(int64, tag = "7")]
    pub estimated_probe_rows: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SortDetails {
    #[prost(message, repeated, tag = "1")]
    pub sort_keys: ::prost::alloc::vec::Vec<SortKey>,
    #[prost(string, tag = "2")]
    pub sort_algorithm: ::prost::alloc::string::String,
    #[prost(bool, tag = "3")]
    pub is_order_preserving: bool,
    #[prost(int64, tag = "4")]
    pub sort_memory_bytes: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SortKey {
    #[prost(string, tag = "1")]
    pub column_name: ::prost::alloc::string::String,
    #[prost(bool, tag = "2")]
    pub is_ascending: bool,
    #[prost(bool, tag = "3")]
    pub nulls_first: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphScanDetails {
    #[prost(string, tag = "1")]
    pub graph_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub graph_id: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub start_node: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "4")]
    pub edge_types: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag = "5")]
    pub node_labels: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "6")]
    pub traversal_direction: ::prost::alloc::string::String,
    #[prost(int32, tag = "7")]
    pub max_depth: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HybridSearchDetails {
    #[prost(string, tag = "1")]
    pub vector_index_name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub filter_condition: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub execution_strategy: ::prost::alloc::string::String,
    #[prost(double, tag = "4")]
    pub filter_selectivity: f64,
    #[prost(int32, tag = "5")]
    pub total_candidates: i32,
    #[prost(int32, tag = "6")]
    pub filtered_candidates: i32,
    #[prost(message, optional, tag = "7")]
    pub vector_metrics: ::core::option::Option<StrategyMetrics>,
    #[prost(message, optional, tag = "8")]
    pub filter_metrics: ::core::option::Option<StrategyMetrics>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StrategyMetrics {
    #[prost(double, tag = "1")]
    pub latency_ms: f64,
    #[prost(int32, tag = "2")]
    pub candidates_found: i32,
    #[prost(double, tag = "3")]
    pub recall: f64,
}

/// RerankDetails describes a multi-phase rank-pipeline (RERANK SRF)
/// plan node (R-7c.4c.1). `rank_profile` is empty on the
/// retrieval-only path. `executed_phase` is filled only with
/// ANALYZE.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RerankDetails {
    #[prost(string, tag = "1")]
    pub collection: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub query_text: ::prost::alloc::string::String,
    #[prost(int32, tag = "3")]
    pub k: i32,
    #[prost(string, tag = "4")]
    pub rank_profile: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub executed_phase: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeStats {
    #[prost(double, tag = "1")]
    pub wall_time_ms: f64,
    #[prost(double, tag = "2")]
    pub cpu_time_ms: f64,
    #[prost(double, tag = "3")]
    pub wait_time_ms: f64,
    #[prost(int64, tag = "4")]
    pub memory_bytes: i64,
    #[prost(int64, tag = "5")]
    pub peak_memory_bytes: i64,
    #[prost(int64, tag = "6")]
    pub spill_bytes: i64,
    #[prost(int64, tag = "7")]
    pub rows_in: i64,
    #[prost(int64, tag = "8")]
    pub rows_out: i64,
    #[prost(int64, tag = "9")]
    pub bytes_in: i64,
    #[prost(int64, tag = "10")]
    pub bytes_out: i64,
    #[prost(int32, tag = "11")]
    pub threads_used: i32,
    #[prost(double, tag = "12")]
    pub parallelism_efficiency: f64,
    #[prost(int64, tag = "13")]
    pub bytes_read_from_disk: i64,
    #[prost(int64, tag = "14")]
    pub bytes_read_from_cache: i64,
    #[prost(int64, tag = "15")]
    pub cache_hits: i64,
    #[prost(int64, tag = "16")]
    pub cache_misses: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionStats {
    #[prost(message, optional, tag = "1")]
    pub start_time: ::core::option::Option<::prost_types::Timestamp>,
    #[prost(message, optional, tag = "2")]
    pub end_time: ::core::option::Option<::prost_types::Timestamp>,
    #[prost(double, tag = "3")]
    pub total_wall_time_ms: f64,
    #[prost(double, tag = "4")]
    pub total_cpu_time_ms: f64,
    #[prost(int64, tag = "5")]
    pub peak_memory_bytes: i64,
    #[prost(double, tag = "6")]
    pub cpu_utilization_percent: f64,
    #[prost(int64, tag = "7")]
    pub total_rows_in: i64,
    #[prost(int64, tag = "8")]
    pub total_rows_out: i64,
    #[prost(int64, tag = "9")]
    pub total_bytes_in: i64,
    #[prost(int64, tag = "10")]
    pub total_bytes_out: i64,
    #[prost(double, tag = "11")]
    pub rows_per_second: f64,
    #[prost(double, tag = "12")]
    pub bytes_per_second: f64,
    #[prost(int64, tag = "13")]
    pub total_cache_hits: i64,
    #[prost(int64, tag = "14")]
    pub total_cache_misses: i64,
    #[prost(double, tag = "15")]
    pub cache_hit_rate: f64,
    #[prost(int32, tag = "16")]
    pub total_workers: i32,
    #[prost(int32, tag = "17")]
    pub active_workers: i32,
    #[prost(double, tag = "18")]
    pub worker_utilization_percent: f64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExplainWarning {
    #[prost(string, tag = "1")]
    pub warning_code: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub severity: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub message: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub suggestion: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "5")]
    pub affected_nodes: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PlanMetadata {
    #[prost(string, tag = "1")]
    pub optimizer_version: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub optimization_level: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "3")]
    pub optimization_rules_applied: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(int64, tag = "4")]
    pub optimization_time_ms: i64,
    #[prost(string, tag = "5")]
    pub execution_engine: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "6")]
    pub storage_engines_used: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "7")]
    pub query_language: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "8")]
    pub enabled_features: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, tag = "9")]
    pub plan_version: ::prost::alloc::string::String,
    #[prost(string, tag = "10")]
    pub plan_hash: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "100")]
    pub additional_metadata: ::core::option::Option<::prost_types::Struct>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExplainPlanRequest {
    #[prost(string, tag = "1")]
    pub query: ::prost::alloc::string::String,
    #[prost(enumeration = "QueryType", tag = "2")]
    pub query_type: i32,
    #[prost(enumeration = "ExplainFormat", tag = "3")]
    pub format: i32,
    #[prost(message, optional, tag = "4")]
    pub options: ::core::option::Option<ExplainOptions>,
    #[prost(message, optional, tag = "5")]
    pub session_params: ::core::option::Option<::prost_types::Struct>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ExplainFormat {
    ExplainFormatUnknown = 0,
    ExplainFormatJson = 1,
    ExplainFormatText = 2,
    ExplainFormatGraphviz = 3,
    ExplainFormatProtobuf = 4,
}

impl ExplainFormat {
    pub fn as_str_name(&self) -> &'static str {
        match self {
            ExplainFormat::ExplainFormatUnknown => "EXPLAIN_FORMAT_UNKNOWN",
            ExplainFormat::ExplainFormatJson => "EXPLAIN_FORMAT_JSON",
            ExplainFormat::ExplainFormatText => "EXPLAIN_FORMAT_TEXT",
            ExplainFormat::ExplainFormatGraphviz => "EXPLAIN_FORMAT_GRAPHVIZ",
            ExplainFormat::ExplainFormatProtobuf => "EXPLAIN_FORMAT_PROTOBUF",
        }
    }
    /// Creates an enum from the proto field names (round 8: the decode
    /// direction was simply absent — JSON/enum-name decoding of explain
    /// types never worked).
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "EXPLAIN_FORMAT_UNKNOWN" => Some(Self::ExplainFormatUnknown),
            "EXPLAIN_FORMAT_JSON" => Some(Self::ExplainFormatJson),
            "EXPLAIN_FORMAT_TEXT" => Some(Self::ExplainFormatText),
            "EXPLAIN_FORMAT_GRAPHVIZ" => Some(Self::ExplainFormatGraphviz),
            "EXPLAIN_FORMAT_PROTOBUF" => Some(Self::ExplainFormatProtobuf),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExplainOptions {
    #[prost(bool, tag = "1")]
    pub analyze: bool,
    #[prost(bool, tag = "2")]
    pub verbose: bool,
    #[prost(bool, tag = "3")]
    pub costs: bool,
    #[prost(bool, tag = "4")]
    pub timing: bool,
    #[prost(bool, tag = "5")]
    pub buffers: bool,
    #[prost(bool, tag = "6")]
    pub format: bool,
    #[prost(bool, tag = "7")]
    pub summary: bool,
    #[prost(bool, tag = "8")]
    pub warnings: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExplainPlanResponse {
    #[prost(message, optional, tag = "1")]
    pub plan: ::core::option::Option<ExplainPlan>,
    #[prost(string, tag = "2")]
    pub formatted_output: ::prost::alloc::string::String,
    #[prost(bool, tag = "3")]
    pub success: bool,
    #[prost(string, tag = "4")]
    pub error_message: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "5")]
    pub error_details: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
