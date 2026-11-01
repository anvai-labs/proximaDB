/**
 * ProximaDB TypeScript SDK - Type Definitions
 *
 * Copyright 2025 ProximaDB Contributors
 * Licensed under the Apache License, Version 2.0
 */

// ============================================================================
// ENUMS
// ============================================================================

/**
 * Distance metrics for vector similarity search
 */
export enum DistanceMetric {
  /** Cosine similarity (normalized dot product) */
  Cosine = "cosine",
  /** Euclidean (L2) distance */
  Euclidean = "euclidean",
  /** Dot product similarity */
  DotProduct = "dot_product",
  /** Manhattan (L1) distance */
  Manhattan = "manhattan",
  /** Hamming distance for binary vectors */
  Hamming = "hamming",
  /** Jaccard similarity for set-like vectors */
  Jaccard = "jaccard",
}

/**
 * Storage engines available in ProximaDB
 */
export enum StorageEngine {
  /** Write-optimized, real-time workloads */
  Sst = "sst",
  /** Locality-optimized with Hilbert curves */
  Helix = "helix",
  /** Columnar Parquet for analytics */
  Viper = "viper",
  /** Ultra-low latency for small datasets */
  Swift = "swift",
  /** Progressive columnar for mixed workloads */
  Nova = "nova",
  /** Adaptive row-group for dynamic workloads */
  Raptor = "raptor",
}

/**
 * Index types for vector search
 */
export enum IndexType {
  /** Hierarchical Navigable Small World */
  Hnsw = "hnsw",
  /** Inverted File Index */
  Ivf = "ivf",
  /** Locality Sensitive Hashing */
  Lsh = "lsh",
  /** Brute force (no index) */
  Flat = "flat",
  /** Product Quantization */
  Pq = "pq",
  /** Approximate Nearest Neighbors Oh Yeah */
  Annoy = "annoy",
}

/**
 * Filter operations for metadata filtering
 */
export enum FilterOp {
  /** Equal to */
  Eq = "equals",
  /** Not equal to */
  Ne = "not_equals",
  /** Greater than */
  Gt = "gt",
  /** Greater than or equal */
  Gte = "gte",
  /** Less than */
  Lt = "lt",
  /** Less than or equal */
  Lte = "lte",
  /** In list of values */
  In = "in",
  /** Not in list of values */
  NotIn = "not_in",
  /** Contains (for strings/arrays) */
  Contains = "contains",
  /** Starts with (for strings) */
  StartsWith = "starts_with",
  /** Ends with (for strings) */
  EndsWith = "ends_with",
  /** Field exists */
  Exists = "exists",
  /** Field is null */
  IsNull = "is_null",
}

/**
 * Logical operators for combining filter conditions
 */
export enum LogicalOp {
  /** All conditions must match */
  And = "and",
  /** Any condition must match */
  Or = "or",
}

/**
 * Search mode for controlling recall vs performance tradeoff
 */
export enum SearchMode {
  /** Exact search - 100% recall, searches all partitions */
  Exact = "exact",
  /** Approximate search - faster with ~95% recall */
  Approximate = "approximate",
  /** Adaptive search - auto-selects based on dataset size */
  Adaptive = "adaptive",
}

/**
 * Traversal direction for graph queries
 */
export enum TraversalDirection {
  /** Traverse outgoing edges */
  Outgoing = "outgoing",
  /** Traverse incoming edges */
  Incoming = "incoming",
  /** Traverse both directions */
  Both = "both",
}

// ============================================================================
// CORE INTERFACES
// ============================================================================

/**
 * Metadata value types supported by ProximaDB
 */
export type MetadataValue = string | number | boolean | string[] | number[];

/**
 * Metadata dictionary type
 */
export type Metadata = Record<string, MetadataValue>;

/**
 * JSON-compatible value type for filters
 */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * Canonical ProximaRecord payload for storing records with optional vectors.
 */
export interface ProximaRecord {
  /** Unique record identifier */
  id: string;
  /** Dense vector embedding */
  vector: number[];
  /** Rich record properties */
  props?: Record<string, JsonValue>;
  /** @deprecated Use props. */
  metadata?: Metadata;
  /** Creation timestamp in milliseconds */
  timestampMs?: number;
  /** Last update timestamp in milliseconds */
  updatedAtMs?: number;
  /** Expiration timestamp in milliseconds (TTL support) */
  expiresAtMs?: number;
  /** Version number for optimistic concurrency */
  version?: number;
  /** Original content that generated this vector (e.g., chunk text for RAG) */
  source?: string;
  /** Text fields for dedicated text storage.
   *
   * DISPLAY-SIDE alias only: the REST wire contract is snake_case
   * `text_fields` (TD-REC-1) and the transport stringifies bodies as-is —
   * write-path builders must emit `text_fields`, never this camelCase key.
   */
  textFields?: Array<{ name: string; content: string }>;
}

/**
 * @deprecated Use ProximaRecord. VectorRecord remains a compatibility alias.
 */
export type VectorRecord = ProximaRecord;

/**
 * Search result with score and optional metadata
 */
export interface SearchResult {
  /** Vector ID */
  id: string;
  /** Similarity score */
  score: number;
  /** Associated metadata (if requested) */
  metadata?: Metadata;
  /** Vector data (if requested) */
  vector?: number[];
  /** Result rank in the result set */
  rank?: number;
  /** Version number */
  version?: number;
  /** Original source content */
  source?: string;
}

/**
 * Collection configuration
 */
export interface CollectionConfig {
  /** Collection name (min 8 characters) */
  name: string;
  /** Vector dimension */
  dimension: number;
  /** Distance metric for similarity */
  distanceMetric?: DistanceMetric;
  /** Storage engine */
  storageEngine?: StorageEngine;
  /** Index type */
  indexType?: IndexType;
  /** Collection description */
  description?: string;
  /** Collection tags */
  tags?: string[];
}

/**
 * Collection statistics
 */
export interface CollectionStats {
  /** Number of vectors in the collection */
  vectorCount: number;
  /** Index size in bytes */
  indexSizeBytes: number;
  /** Data size in bytes */
  dataSizeBytes: number;
}

/**
 * Collection information
 */
export interface CollectionInfo {
  /** Collection ID */
  id: string;
  /** Collection name */
  name: string;
  /** Vector dimension */
  dimension: number;
  /** Distance metric */
  metric: string;
  /** Creation timestamp in milliseconds */
  createdAtMs: number;
  /** Last update timestamp in milliseconds */
  updatedAtMs: number;
  /** Number of vectors */
  vectorCount?: number;
  /** Whether the collection is indexed */
  indexed?: boolean;
}

/**
 * Health status response
 */
export interface HealthStatus {
  /** Server status */
  status: string;
  /** Server version */
  version: string;
  /** Uptime in seconds */
  uptimeSeconds: number;
  /** Service statuses */
  services: Record<string, string>;
  /** Timestamp in milliseconds */
  timestampMs: number;
}

// ============================================================================
// PROBE / SCHEMA / QUERY INTERFACES (v2 wire types — snake_case)
// ============================================================================

/**
 * Kubernetes-style liveness/readiness probe response.
 *
 * Matches the `ProbeResponse` schema in docs/openapi/proximadb-openapi.yaml.
 */
export interface ProbeResponse {
  /** Probe status (e.g. "ok", "ready") */
  status: string;
}

/**
 * Column definition for a typed schema.
 *
 * Field names are snake_case to wire-match the OpenAPI spec / Python SDK.
 */
export interface ColumnDefinition {
  /** Column name */
  name: string;
  /** Column data type (see OpenAPI enum: text, integer, float, ...) */
  data_type: string;
  /** Whether the column is nullable */
  nullable?: boolean;
  /** Whether the column is indexed */
  indexed?: boolean;
  /** Whether the column is filterable */
  filterable?: boolean;
  /** Maximum length (for text types) */
  max_length?: number;
  /** Decimal precision */
  precision?: number;
  /** Decimal scale */
  scale?: number;
  /** Vector dimension (for `vector` columns) */
  vector_dimension?: number;
}

/**
 * Typed schema definition for a collection.
 */
export interface SchemaDefinition {
  /** Column definitions */
  columns: ColumnDefinition[];
  /** Schema enforcement strategy */
  enforcement?: "strict" | "flexible" | "hybrid";
  /** Whether records may include fields beyond the declared columns */
  allow_additional_fields?: boolean;
}

/**
 * Response from GET /api/v2/collections/{id}/schema.
 */
export interface SchemaResponse {
  schema_id: string;
  schema_version: string;
  collection_id: string;
  schema: SchemaDefinition;
  created_at: string;
  updated_at?: string | null;
  parent_schema_id?: string | null;
}

/**
 * Request body for PUT /api/v2/collections/{id}/schema.
 *
 * Wire-encodes as a `SchemaDefinition` plus optional `force` flag.
 */
export interface UpdateSchemaRequest extends SchemaDefinition {
  /** Allow incompatible schema changes */
  force?: boolean;
}

/**
 * Response from PUT /api/v2/collections/{id}/schema.
 */
export interface UpdateSchemaResponse {
  schema_id: string;
  schema_version: string;
  previous_schema_id: string;
  changes: unknown[];
  warnings: unknown[];
  updated_at: string;
  [key: string]: unknown;
}

/**
 * Supported query languages for the shared query facade.
 */
export type QueryLanguage = "uql" | "aql" | "federated";

/**
 * Request body for POST /api/v2/query.
 */
export interface QueryRequest {
  /** Query language */
  language: QueryLanguage;
  /** Query text (AQL / UQL) */
  query: string;
  /** Optional bound parameters (ProximaValue-encoded) */
  parameters?: JsonValue[];
  /** Optional default collection */
  collection?: string | null;
  /** Optional row-limit cap */
  limit?: number | null;
}

/**
 * Request body for POST /api/v2/query/explain.
 */
export interface ExplainQueryRequest {
  /** Query language */
  language: QueryLanguage;
  /** Query text (AQL / UQL) */
  query: string;
  /** Optional default collection */
  collection?: string | null;
}

/**
 * Response from POST /api/v2/query and POST /api/v2/query/explain.
 *
 * Open shape per the OpenAPI contract — implementations typically return
 * records / total_count / metrics / plan / diagnostics depending on the
 * language and endpoint.
 */
export interface QueryResponse {
  [key: string]: unknown;
}

// ============================================================================
// FILTER INTERFACES
// ============================================================================

/**
 * A single filter condition
 */
export interface FilterCondition {
  /** Field name */
  field: string;
  /** Operation */
  operation: string;
  /** Value to compare (optional for exists/is_null) */
  value?: JsonValue;
}

/**
 * A filter group containing conditions and nested groups
 */
export interface FilterGroup {
  /** Logical operator for combining conditions */
  operator: string;
  /** List of conditions */
  conditions: FilterNode[];
}

/**
 * A node in the filter tree
 */
export type FilterNode = FilterCondition | FilterGroup;

/**
 * Compiled filter expression
 */
export interface Filter {
  /** Root filter group */
  operator: string;
  conditions: FilterNode[];
}

// ============================================================================
// GRAPH INTERFACES
// ============================================================================

/**
 * A graph node
 */
export interface GraphNode {
  /** Unique node identifier */
  id: string;
  /** Node label (type) */
  label?: string;
  /** Node properties */
  properties: Record<string, JsonValue>;
  /** Optional embedding vector for semantic operations */
  vector?: number[];
}

/**
 * A graph edge
 */
export interface GraphEdge {
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Relationship type */
  relationship: string;
  /** Edge properties */
  properties: Record<string, JsonValue>;
  /** Optional weight */
  weight?: number;
}

/**
 * Graph traversal result
 */
export interface TraversalResult {
  /** Nodes in the traversal path */
  nodes: GraphNode[];
  /** Edges in the traversal path */
  edges: GraphEdge[];
  /** Paths from start to each node */
  paths: string[][];
}

/**
 * Graph information
 */
export interface GraphInfo {
  /** Graph name */
  name: string;
  /** Number of nodes */
  nodeCount: number;
  /** Number of edges */
  edgeCount: number;
  /** Graph description */
  description?: string;
}

// ----------------------------------------------------------------------------
// Server-true graph payload shapes (OpenAPI: docs/openapi/proximadb-openapi.yaml)
//
// These mirror the wire-level types the server actually accepts. The legacy
// `GraphNode` / `GraphEdge` interfaces above remain as ergonomic SDK-facing
// types; the SDK builders now lower them into the spec types below.
// ----------------------------------------------------------------------------

/**
 * Embedding payload nested inside `NodeInput.embedding`.
 *
 * OpenAPI: `EmbeddingInput` — `{ vector, model_id?, modality? }`.
 */
export interface EmbeddingInput {
  vector: number[];
  model_id?: string;
  modality?: string;
}

/**
 * Node payload nested inside `CreateNodeRequest.node` and
 * `BatchCreateNodesRequest.nodes[]`.
 *
 * OpenAPI: `NodeInput` — `{ id, labels?, properties?, embedding? }`.
 */
export interface NodeInput {
  id: string;
  labels?: string[];
  properties?: Record<string, JsonValue>;
  embedding?: EmbeddingInput;
}

/**
 * Edge payload nested inside `CreateEdgeRequest.edge` and
 * `BatchCreateEdgesRequest.edges[]`.
 *
 * OpenAPI: `EdgeInput` — `{ id, from_node_id, to_node_id, edge_type,
 * properties?, weight? }`.
 */
export interface EdgeInput {
  id: string;
  from_node_id: string;
  to_node_id: string;
  edge_type: string;
  properties?: Record<string, JsonValue>;
  weight?: number;
}

// ============================================================================
// CLIENT CONFIGURATION
// ============================================================================

/**
 * Client configuration options
 */
export interface ClientConfig {
  /** Server URL */
  url: string;
  /** Request timeout in milliseconds */
  timeoutMs?: number;
  /** Number of retries for failed requests */
  maxRetries?: number;
  /** API key for authentication */
  apiKey?: string;
  /** Enable connection pooling */
  poolConnections?: boolean;
  /** Maximum idle connections in pool */
  maxIdleConnections?: number;
}

/**
 * Search request options
 */
export interface SearchOptions {
  /** Number of results to return */
  topK?: number;
  /** Filter expression string */
  filter?: string;
  /** Search mode */
  mode?: SearchMode;
  /** Number of partitions to probe (for approximate search) */
  nprobe?: number;
  /** Include vectors in results */
  includeVectors?: boolean;
  /** Include metadata in results */
  includeMetadata?: boolean;
  /** Minimum score threshold */
  minScore?: number;
  /** Request timeout in milliseconds */
  timeoutMs?: number;
}

/**
 * Insert operation options
 */
export interface InsertOptions {
  /** Timeout in milliseconds */
  timeoutMs?: number;
  /** Skip validation */
  skipValidation?: boolean;
}

/**
 * Update operation options
 */
export interface UpdateOptions {
  /** Replace all metadata instead of merging */
  replaceMetadata?: boolean;
  /** Timeout in milliseconds */
  timeoutMs?: number;
}

/**
 * Delete operation result
 */
export interface DeleteResult {
  /** Number of items deleted */
  deletedCount: number;
  /** Whether the operation succeeded */
  success: boolean;
  /** Optional message */
  message?: string;
}

/**
 * Operation metrics
 */
export interface OperationMetrics {
  /** Total items processed */
  totalProcessed: number;
  /** Successful count */
  successfulCount: number;
  /** Failed count */
  failedCount: number;
  /** Processing time in microseconds */
  processingTimeUs: number;
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/**
 * ProximaDB error codes
 */
export enum ErrorCode {
  /** Unknown error */
  Unknown = "UNKNOWN",
  /** Network error */
  Network = "NETWORK_ERROR",
  /** Authentication failed */
  AuthenticationFailed = "AUTHENTICATION_FAILED",
  /** Authorization denied */
  AuthorizationDenied = "AUTHORIZATION_DENIED",
  /** Rate limited */
  RateLimited = "RATE_LIMITED",
  /** Collection not found */
  CollectionNotFound = "COLLECTION_NOT_FOUND",
  /** Collection already exists */
  CollectionExists = "COLLECTION_EXISTS",
  /** Vector not found */
  VectorNotFound = "VECTOR_NOT_FOUND",
  /** Invalid vector dimension */
  InvalidDimension = "INVALID_DIMENSION",
  /** Invalid configuration */
  InvalidConfig = "INVALID_CONFIG",
  /** Invalid filter */
  InvalidFilter = "INVALID_FILTER",
  /** Timeout */
  Timeout = "TIMEOUT",
  /** Server error */
  ServerError = "SERVER_ERROR",
}

/**
 * ProximaDB error
 */
export class ProximaDBError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode = ErrorCode.Unknown,
    public readonly statusCode?: number,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "ProximaDBError";
    Object.setPrototypeOf(this, ProximaDBError.prototype);
  }
}

// ============================================================================
// STREAMING INTERFACES
// ============================================================================

/**
 * Streaming search result batch
 */
export interface SearchResultBatch {
  /** Results in this batch */
  results: SearchResult[];
  /** Whether there are more results */
  hasMore: boolean;
  /** Cursor for pagination */
  cursor?: string;
  /** Total count (if known) */
  total?: number;
}

/**
 * Async iterator for streaming search results
 */
export interface SearchResultIterator extends AsyncIterableIterator<SearchResult> {
  /** Get total count if available */
  getTotal(): number | undefined;
  /** Check if iteration is complete */
  isComplete(): boolean;
}
