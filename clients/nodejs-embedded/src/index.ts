/**
 * ProximaDB TypeScript SDK
 *
 * A modern, async-first TypeScript client for ProximaDB vector database.
 *
 * @example
 * ```typescript
 * import { connect, FilterBuilder } from "@proximadb/client";
 *
 * // Connect to ProximaDB
 * const client = connect("http://localhost:5678");
 *
 * // Create a collection
 * await client.createCollection("embeddings")
 *   .dimension(768)
 *   .engine(StorageEngine.Sst)
 *   .execute();
 *
 * // Insert vectors
 * await client.collection("embeddings")
 *   .insert()
 *   .id("vec_1")
 *   .vector([0.1, 0.2, ...])
 *   .meta("category", "tech")
 *   .execute();
 *
 * // Search with filters
 * const results = await client.collection("embeddings")
 *   .search()
 *   .vector(queryVector)
 *   .topK(10)
 *   .filterEq("category", "tech")
 *   .execute();
 *
 * // Graph operations
 * await client.createGraph("knowledge").execute();
 * await client.graph("knowledge")
 *   .addNode()
 *   .id("person_1")
 *   .label("Person")
 *   .property("name", "Alice")
 *   .execute();
 * ```
 *
 * Copyright 2025 ProximaDB Contributors
 * Licensed under the Apache License, Version 2.0
 */

// ============================================================================
// VERSION
// ============================================================================

// The package version. Kept equal to package.json `version` by
// scripts/version-sync.sh (`set` rewrites it, `check` fails on drift) and by
// tests/package_entry.test.ts.
export const VERSION = "0.5.0";

// ============================================================================
// TYPES
// ============================================================================

// Enums (runtime values)
export {
  DistanceMetric,
  StorageEngine,
  IndexType,
  FilterOp,
  LogicalOp,
  SearchMode,
  TraversalDirection,
  ErrorCode,
  ProximaDBError,
} from "./types";

// Type-only exports (interfaces and types)
export type {
  ProximaRecord,
  VectorRecord,
  SearchResult,
  CollectionConfig,
  CollectionStats,
  CollectionInfo,
  HealthStatus,
  Metadata,
  MetadataValue,
  JsonValue,
  Filter,
  FilterCondition,
  FilterGroup,
  FilterNode,
  GraphNode,
  GraphEdge,
  TraversalResult,
  GraphInfo,
  ClientConfig,
  SearchOptions,
  InsertOptions,
  UpdateOptions,
  DeleteResult,
  OperationMetrics,
  SearchResultBatch,
  SearchResultIterator,
  ProbeResponse,
  ColumnDefinition,
  SchemaDefinition,
  SchemaResponse,
  UpdateSchemaRequest,
  UpdateSchemaResponse,
  QueryLanguage,
  QueryRequest,
  ExplainQueryRequest,
  QueryResponse,
} from "./types";

// ============================================================================
// CLIENT
// ============================================================================

export {
  ProximaDBClient,
  ClientBuilder,
  connect,
  connectRest,
} from "./client";

// ============================================================================
// COLLECTION
// ============================================================================

export {
  CollectionHandle,
  CollectionBuilder,
  InsertBuilder,
  BatchInsertBuilder,
  UpdateBuilder,
} from "./collection";

// ============================================================================
// SEARCH
// ============================================================================

export {
  SearchBuilder,
  createSearchBuilder,
} from "./search";

// ============================================================================
// FILTER
// ============================================================================

export {
  FilterBuilder,
  // Convenience functions
  eq,
  ne,
  gt,
  lt,
  inList,
  range,
  andFilters,
  orFilters,
  filterToExpression,
} from "./filter";

// ============================================================================
// GRAPH
// ============================================================================

export {
  GraphHandle,
  GraphBuilder,
  NodeBuilder,
  EdgeBuilder,
  TraversalBuilder,
  // Convenience functions
  createNode,
  node,
  createEdge,
  edge,
} from "./graph";

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

import { ProximaDBClient } from "./client";
export default ProximaDBClient;
