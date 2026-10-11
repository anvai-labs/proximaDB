/**
 * ProximaDB TypeScript SDK - Client
 *
 * Main entry point for connecting to ProximaDB servers.
 *
 * Copyright 2025 ProximaDB Contributors
 * Licensed under the Apache License, Version 2.0
 */

import {
  ClientConfig,
  HealthStatus,
  CollectionInfo,
  GraphInfo,
  ProximaDBError,
  ErrorCode,
  ProbeResponse,
  SchemaResponse,
  UpdateSchemaRequest,
  UpdateSchemaResponse,
  QueryRequest,
  ExplainQueryRequest,
  QueryResponse,
  GraphNode,
  GraphEdge,
  TraversalResult,
  JsonValue,
} from "./types";
import {
  CollectionBuilder,
  CollectionHandle,
  CollectionHttpClient,
} from "./collection";
import { GraphBuilder, GraphHandle, GraphHttpClient } from "./graph";
import { createTransport, GeneratedClient } from "./transport";

/**
 * Default client configuration
 */
const DEFAULT_CONFIG: Required<ClientConfig> = {
  url: "http://localhost:5678",
  timeoutMs: 30000,
  maxRetries: 3,
  apiKey: "",
  poolConnections: true,
  maxIdleConnections: 10,
};

/**
 * HTTP response interface
 */
interface HttpResponse {
  ok: boolean;
  status: number;
  statusText: string;
  json(): Promise<unknown>;
  text(): Promise<string>;
}

interface GraphCollectionWire {
  graph_id?: unknown;
  name?: unknown;
  description?: unknown;
  node_count?: unknown;
  edge_count?: unknown;
  stats?: {
    total_nodes?: unknown;
    total_edges?: unknown;
  };
}

function graphInfoFromWire(value: unknown): GraphInfo {
  const graph =
    typeof value === "object" && value !== null
      ? (value as GraphCollectionWire)
      : {};
  const displayName = typeof graph.name === "string" ? graph.name : "";
  const graphId = typeof graph.graph_id === "string" ? graph.graph_id : "";
  const description =
    typeof graph.description === "string" ? graph.description : undefined;
  const topLevelNodeCount =
    typeof graph.node_count === "number" ? graph.node_count : undefined;
  const topLevelEdgeCount =
    typeof graph.edge_count === "number" ? graph.edge_count : undefined;
  const nestedNodeCount =
    typeof graph.stats?.total_nodes === "number"
      ? graph.stats.total_nodes
      : undefined;
  const nestedEdgeCount =
    typeof graph.stats?.total_edges === "number"
      ? graph.stats.total_edges
      : undefined;

  return {
    name: displayName.trim().length > 0 ? displayName : graphId,
    nodeCount: topLevelNodeCount ?? nestedNodeCount ?? 0,
    edgeCount: topLevelEdgeCount ?? nestedEdgeCount ?? 0,
    ...(description === undefined ? {} : { description }),
  };
}

function objectFromWire(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

/**
 * Wave-6 envelope truth: a 2xx graph body carries the canonical
 * `{success, data}` envelope. A 200 whose `data` member is absent is a
 * server-contract violation — a pre-wave-6 flat body or a `{success:false}`
 * body that slipped through as 200. Lowering such a body would fabricate an
 * empty node / traversal / list out of a real failure (the repo's
 * silent-Ok(failure) anti-pattern), so throw instead.
 */
function requireEnvelopeData(body: unknown, operation: string): unknown {
  const data = objectFromWire(body).data;
  if (data === undefined) {
    throw new ProximaDBError(
      `${operation}: 2xx response is missing the canonical {success, data} envelope` +
        " (server version mismatch or success:false body)",
      ErrorCode.ServerError,
    );
  }
  return data;
}

function graphNodeFromWire(value: unknown): GraphNode {
  const node = objectFromWire(value);
  const labels = Array.isArray(node.labels) ? node.labels : [];
  const embedding = objectFromWire(node.embedding);
  const vector = Array.isArray(embedding.vector)
    ? embedding.vector.filter(
        (entry): entry is number => typeof entry === "number",
      )
    : undefined;
  const label = labels.find(
    (entry): entry is string => typeof entry === "string",
  );

  return {
    id: typeof node.id === "string" ? node.id : "",
    ...(label === undefined ? {} : { label }),
    properties: objectFromWire(node.properties) as Record<string, JsonValue>,
    ...(vector === undefined ? {} : { vector }),
  };
}

function graphEdgeFromWire(value: unknown): GraphEdge {
  const edge = objectFromWire(value);
  const weight = typeof edge.weight === "number" ? edge.weight : undefined;
  return {
    source: typeof edge.from_node_id === "string" ? edge.from_node_id : "",
    target: typeof edge.to_node_id === "string" ? edge.to_node_id : "",
    relationship: typeof edge.edge_type === "string" ? edge.edge_type : "",
    properties: objectFromWire(edge.properties) as Record<string, JsonValue>,
    ...(weight === undefined ? {} : { weight }),
  };
}

function traversalResultFromWire(value: unknown): TraversalResult {
  const result = objectFromWire(value);
  const paths = Array.isArray(result.paths)
    ? result.paths.filter(
        (path): path is string[] =>
          Array.isArray(path) &&
          path.every((entry) => typeof entry === "string"),
      )
    : [];
  return {
    nodes: (Array.isArray(result.nodes) ? result.nodes : []).map(
      graphNodeFromWire,
    ),
    edges: (Array.isArray(result.edges) ? result.edges : []).map(
      graphEdgeFromWire,
    ),
    paths,
  };
}

/**
 * Minimal fetch-like function signature
 */
type FetchFunction = (url: string, init?: RequestInit) => Promise<HttpResponse>;

/**
 * Get the global fetch function
 */
function getFetch(): FetchFunction {
  if (typeof globalThis.fetch === "function") {
    return globalThis.fetch.bind(globalThis) as FetchFunction;
  }
  throw new Error(
    "No fetch implementation available. " +
      "Node.js 18+ includes fetch natively. " +
      "For older versions, install node-fetch.",
  );
}

/**
 * ProximaDB client for connecting to a remote server
 */
export class ProximaDBClient implements CollectionHttpClient, GraphHttpClient {
  private config: Required<ClientConfig>;
  private fetchFn: FetchFunction;
  /**
   * Generated typed REST transport (TD-126 Phase 4). Core collection / record
   * / search operations route through this; it owns URL/path/query/body
   * encoding per the OpenAPI spec. Auth, retries, and error mapping are
   * injected via the facade's own fetch (see `transportFetch`).
   */
  private gen: GeneratedClient;

  /**
   * Create a new client with the given URL
   */
  static connect(url: string): ProximaDBClient {
    return new ProximaDBClient({ url });
  }

  /**
   * Create a new client builder
   */
  static builder(): ClientBuilder {
    return new ClientBuilder();
  }

  /**
   * Create a client with configuration
   */
  constructor(config: Partial<ClientConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.fetchFn = getFetch();

    // Validate URL
    try {
      new URL(this.config.url);
    } catch {
      throw new ProximaDBError(
        "Invalid URL: " + this.config.url,
        ErrorCode.InvalidConfig,
      );
    }

    // Wire the generated typed transport. Every request the generated client
    // issues is funneled through `transportFetch`, the facade's own fetch that
    // applies bearer auth, retries, and error mapping — exactly as the legacy
    // generic `request<T>` path does.
    this.gen = createTransport(this.config.url, (req) =>
      this.transportFetch(req),
    );
  }

  /**
   * Transport `fetch` handed to the generated client.
   *
   * The generated client assembles a resolved `Request` (URL + method + body);
   * this normalizes it back to the facade's `(url, init)` retry/auth/error
   * pipeline so the wire plumbing and transport policy compose. The returned
   * value is a `Response`-shaped object the generated client parses; on an
   * unsuccessful response a `ProximaDBError` is thrown (matching the generic
   * path), so callers never see a non-2xx body.
   */
  private async transportFetch(req: Request): Promise<Response> {
    const method = (req.method || "GET").toUpperCase();
    let body: unknown = undefined;
    if (method !== "GET" && method !== "DELETE") {
      const text = await req.clone().text();
      if (text.length > 0) {
        body = JSON.parse(text);
      }
    }
    const data = await this.request<unknown>(method, req.url, body);
    // Re-wrap the already-parsed, validated body as a Response so the
    // generated client can read `.json()`/`.ok` without re-issuing I/O.
    return new Response(JSON.stringify(data ?? null), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  /**
   * Get the server URL
   */
  url(): string {
    return this.config.url;
  }

  /**
   * Get the client configuration
   */
  getConfig(): ClientConfig {
    return { ...this.config };
  }

  // =========================================================================
  // HTTP Methods
  // =========================================================================

  /**
   * Make a GET request
   */
  async get<T>(requestUrl: string): Promise<T> {
    return this.request<T>("GET", requestUrl);
  }

  /**
   * Make a POST request
   */
  async post<T>(requestUrl: string, body: unknown): Promise<T> {
    return this.request<T>("POST", requestUrl, body);
  }

  /**
   * Make a PUT request
   */
  async put<T>(requestUrl: string, body: unknown): Promise<T> {
    return this.request<T>("PUT", requestUrl, body);
  }

  /**
   * Make a DELETE request
   */
  async delete<T>(requestUrl: string): Promise<T> {
    return this.request<T>("DELETE", requestUrl);
  }

  /**
   * Make an HTTP request with retry logic
   */
  private async request<T>(
    method: string,
    requestUrl: string,
    body?: unknown,
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };

    if (this.config.apiKey) {
      headers["Authorization"] = "Bearer " + this.config.apiKey;
    }

    const buildInit = (signal?: AbortSignal): RequestInit => {
      const init: RequestInit = {
        method,
        headers,
        signal,
      };

      if (body !== undefined) {
        init.body = JSON.stringify(body);
      }

      return init;
    };

    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.timeoutMs,
      );
      try {
        const response = await this.fetchFn(
          requestUrl,
          buildInit(controller.signal),
        );
        return await this.handleResponse<T>(response);
      } catch (e: unknown) {
        lastError = e instanceof Error ? e : new Error(String(e));

        // Don't retry on client errors (4xx)
        if (lastError instanceof ProximaDBError) {
          if (
            lastError.statusCode &&
            lastError.statusCode >= 400 &&
            lastError.statusCode < 500
          ) {
            throw lastError;
          }
        }

        // Wait before retry with exponential backoff
        if (attempt < this.config.maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      } finally {
        clearTimeout(timeoutId);
      }
    }

    throw lastError ?? new Error("Request failed after retries");
  }

  /**
   * Handle HTTP response
   */
  private async handleResponse<T>(response: HttpResponse): Promise<T> {
    if (response.ok) {
      try {
        return (await response.json()) as T;
      } catch {
        throw new ProximaDBError(
          "Failed to parse response JSON",
          ErrorCode.ServerError,
          response.status,
        );
      }
    }

    const statusCode = response.status;
    let message = response.statusText;

    try {
      message = await response.text();
    } catch {
      // Use statusText if text() fails
    }

    switch (statusCode) {
      case 401:
        throw new ProximaDBError(
          message,
          ErrorCode.AuthenticationFailed,
          statusCode,
        );
      case 403:
        throw new ProximaDBError(
          message,
          ErrorCode.AuthorizationDenied,
          statusCode,
        );
      case 404:
        throw new ProximaDBError(
          message,
          ErrorCode.CollectionNotFound,
          statusCode,
        );
      case 409:
        throw new ProximaDBError(
          message,
          ErrorCode.CollectionExists,
          statusCode,
        );
      case 429:
        throw new ProximaDBError(message, ErrorCode.RateLimited, statusCode);
      default:
        throw new ProximaDBError(message, ErrorCode.ServerError, statusCode);
    }
  }

  // =========================================================================
  // Collection Operations
  // =========================================================================

  /**
   * Get a handle to a collection for fluent operations
   */
  collection(name: string): CollectionHandle {
    return new CollectionHandle(this, name);
  }

  /**
   * Create a collection builder
   */
  createCollection(name: string): CollectionBuilder {
    return new CollectionBuilder(this, name);
  }

  /**
   * Delete a collection
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: DELETE /api/v2/collections/{collection_id}
   * OpenAPI operationId: deleteCollection
   */
  async deleteCollection(name: string): Promise<void> {
    await this.gen.DELETE("/api/v2/collections/{collection_id}", {
      params: { path: { collection_id: name } },
    });
  }

  /**
   * List all collections
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /api/v2/collections
   * OpenAPI operationId: listCollections
   */
  async listCollections(): Promise<CollectionInfo[]> {
    const { data } = await this.gen.GET("/api/v2/collections", {});
    return (data?.collections ?? []) as unknown[] as CollectionInfo[];
  }

  /**
   * Get the schema for a collection
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /api/v2/collections/{collection_id}/schema
   * OpenAPI operationId: getCollectionSchema
   */
  async getCollectionSchema(collectionId: string): Promise<SchemaResponse> {
    const { data } = await this.gen.GET(
      "/api/v2/collections/{collection_id}/schema",
      { params: { path: { collection_id: collectionId } } },
    );
    return data as unknown as SchemaResponse;
  }

  /**
   * Update the schema for a collection
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: PUT /api/v2/collections/{collection_id}/schema
   * OpenAPI operationId: updateCollectionSchema
   */
  async updateCollectionSchema(
    collectionId: string,
    schema: UpdateSchemaRequest,
  ): Promise<UpdateSchemaResponse> {
    const { data } = await this.gen.PUT(
      "/api/v2/collections/{collection_id}/schema",
      {
        params: { path: { collection_id: collectionId } },
        body: schema as never,
      },
    );
    return data as unknown as UpdateSchemaResponse;
  }

  // =========================================================================
  // Query Facade (AQL / UQL)
  // =========================================================================

  /**
   * Execute an AQL or UQL query through the shared query facade
   *
   * Wire endpoint: POST /api/v2/query
   * OpenAPI operationId: executeQuery
   */
  async executeQuery(req: QueryRequest): Promise<QueryResponse> {
    const { data } = await this.gen.POST("/api/v2/query", {
      body: req as never,
    });
    return data as unknown as QueryResponse;
  }

  /**
   * Explain an AQL or UQL query through the shared query facade
   *
   * Wire endpoint: POST /api/v2/query/explain
   * OpenAPI operationId: explainQuery
   */
  async explainQuery(req: ExplainQueryRequest): Promise<QueryResponse> {
    const { data } = await this.gen.POST("/api/v2/query/explain", {
      body: req as never,
    });
    return data as unknown as QueryResponse;
  }

  // =========================================================================
  // Typed transport for core collection / record / search ops (TD-126 Phase 4)
  //
  // These route the builders' wire calls through the generated typed client.
  // The builders (collection.ts / search.ts) call these instead of hand-building
  // URL strings, so the core CRUD + search paths are spec-generated end to end.
  // =========================================================================

  /**
   * Create a collection.
   * Wire endpoint: POST /api/v2/collections (createCollection)
   */
  async createCollectionRequest(
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST("/api/v2/collections", {
      body: body as never,
    });
    return data;
  }

  /**
   * Get collection details.
   * Wire endpoint: GET /api/v2/collections/{collection_id} (getCollection)
   */
  async getCollectionRequest(collectionId: string): Promise<unknown> {
    const { data } = await this.gen.GET("/api/v2/collections/{collection_id}", {
      params: { path: { collection_id: collectionId } },
    });
    return data;
  }

  /**
   * Insert / upsert records in a batch.
   * Wire endpoint: POST /api/v2/collections/{collection_id}/records/batch (insertRecords)
   */
  async insertRecordsRequest(
    collectionId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST(
      "/api/v2/collections/{collection_id}/records/batch",
      {
        params: { path: { collection_id: collectionId } },
        body: body as never,
      },
    );
    return data;
  }

  /**
   * Get a single record by ID.
   * Wire endpoint: GET /api/v2/collections/{collection_id}/records/{record_id} (getRecord)
   */
  async getRecordRequest(
    collectionId: string,
    recordId: string,
    includeVector: boolean,
    includeText: boolean,
  ): Promise<unknown> {
    const { data } = await this.gen.GET(
      "/api/v2/collections/{collection_id}/records/{record_id}",
      {
        params: {
          path: { collection_id: collectionId, record_id: recordId },
          query: { include_vector: includeVector, include_text: includeText },
        },
      },
    );
    return data;
  }

  /**
   * Delete a single record by ID.
   * Wire endpoint: DELETE /api/v2/collections/{collection_id}/records/{record_id} (deleteRecord)
   */
  async deleteRecordRequest(
    collectionId: string,
    recordId: string,
  ): Promise<void> {
    await this.gen.DELETE(
      "/api/v2/collections/{collection_id}/records/{record_id}",
      {
        params: { path: { collection_id: collectionId, record_id: recordId } },
      },
    );
  }

  /**
   * Vector similarity search.
   * Wire endpoint: POST /api/v2/collections/{collection_id}/search (searchRecords)
   */
  async searchRecordsRequest(
    collectionId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST(
      "/api/v2/collections/{collection_id}/search",
      {
        params: { path: { collection_id: collectionId } },
        body: body as never,
      },
    );
    return data;
  }

  // =========================================================================
  // Graph Operations
  // =========================================================================

  /**
   * Get a handle to a graph for fluent operations
   */
  graph(name: string): GraphHandle {
    return new GraphHandle(this, name);
  }

  /**
   * Create a graph builder.
   *
   * Wire endpoint: POST /api/v2/graphs
   * OpenAPI operationId: createGraph
   *
   * The argument is the server-true `graph_id` (unique identifier). Use
   * `.name(...)` on the returned builder to set the optional display name.
   */
  createGraph(graphId: string): GraphBuilder {
    return new GraphBuilder(this, graphId);
  }

  /**
   * Delete a graph
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: DELETE /api/v2/graphs/{graph_id}
   * OpenAPI operationId: deleteGraph
   */
  async deleteGraph(name: string): Promise<void> {
    await this.gen.DELETE("/api/v2/graphs/{graph_id}", {
      params: { path: { graph_id: name } },
    });
  }

  /**
   * List all graphs
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /api/v2/graphs
   * OpenAPI operationId: listGraphs
   */
  async listGraphs(): Promise<GraphInfo[]> {
    const { data } = await this.gen.GET("/api/v2/graphs", {});
    // Wave-6 envelope truth: the wire body is {success, data} — the
    // collection array rides `data.data` (the old flat `.graphs` field
    // never existed on the wire). A body without the envelope throws
    // rather than fabricating an empty list.
    return (requireEnvelopeData(data, "listGraphs") as unknown[]).map(
      graphInfoFromWire,
    );
  }

  // =========================================================================
  // Typed transport for graph ops (TD-126 Phase 4)
  //
  // Seam methods routing the graph builders' / handle's wire calls through the
  // generated typed client, mirroring the core collection / record / search
  // seams above. The public graph API (graph.ts builders + handle) is
  // unchanged; only the wire dispatch moves off hand-built URL strings.
  // =========================================================================

  /**
   * Create a graph collection.
   * Wire endpoint: POST /api/v2/graphs (createGraph)
   */
  async createGraphRequest(body: Record<string, unknown>): Promise<unknown> {
    const { data } = await this.gen.POST("/api/v2/graphs", {
      body: body as never,
    });
    return data;
  }

  /**
   * Get a graph collection by id.
   * Wire endpoint: GET /api/v2/graphs/{graph_id} (getGraph)
   */
  async getGraphRequest(graphId: string): Promise<GraphInfo> {
    const { data } = await this.gen.GET("/api/v2/graphs/{graph_id}", {
      params: { path: { graph_id: graphId } },
    });
    return graphInfoFromWire(requireEnvelopeData(data, "getGraph"));
  }

  /**
   * Get graph statistics.
   * Wire endpoint: GET /api/v2/graphs/{graph_id}/stats (getGraphStats)
   */
  async getGraphStatsRequest(graphId: string): Promise<unknown> {
    const { data } = await this.gen.GET("/api/v2/graphs/{graph_id}/stats", {
      params: { path: { graph_id: graphId } },
    });
    return data;
  }

  /**
   * Create a node in a graph.
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/nodes (createNode)
   */
  async createNodeRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST("/api/v2/graphs/{graph_id}/nodes", {
      params: { path: { graph_id: graphId } },
      body: body as never,
    });
    return data;
  }

  /**
   * Create multiple nodes in a single call.
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/nodes/batch (batchCreateNodes)
   */
  async batchCreateNodesRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST(
      "/api/v2/graphs/{graph_id}/nodes/batch",
      {
        params: { path: { graph_id: graphId } },
        body: body as never,
      },
    );
    return data;
  }

  /**
   * Get a node by id.
   * Wire endpoint: GET /api/v2/graphs/{graph_id}/nodes/{node_id} (getNode)
   */
  async getNodeRequest(graphId: string, nodeId: string): Promise<GraphNode> {
    const { data } = await this.gen.GET(
      "/api/v2/graphs/{graph_id}/nodes/{node_id}",
      {
        params: { path: { graph_id: graphId, node_id: nodeId } },
      },
    );
    return graphNodeFromWire(requireEnvelopeData(data, "getNode"));
  }

  /**
   * Delete a node by id.
   * Wire endpoint: DELETE /api/v2/graphs/{graph_id}/nodes/{node_id} (deleteNode)
   */
  async deleteNodeRequest(graphId: string, nodeId: string): Promise<void> {
    await this.gen.DELETE("/api/v2/graphs/{graph_id}/nodes/{node_id}", {
      params: { path: { graph_id: graphId, node_id: nodeId } },
    });
  }

  /**
   * Create an edge in a graph.
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/edges (createEdge)
   */
  async createEdgeRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST("/api/v2/graphs/{graph_id}/edges", {
      params: { path: { graph_id: graphId } },
      body: body as never,
    });
    return data;
  }

  /**
   * Create multiple edges in a single call.
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/edges/batch (batchCreateEdges)
   */
  async batchCreateEdgesRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    const { data } = await this.gen.POST(
      "/api/v2/graphs/{graph_id}/edges/batch",
      {
        params: { path: { graph_id: graphId } },
        body: body as never,
      },
    );
    return data;
  }

  /**
   * Traverse a graph from a start node.
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/traverse (traverseGraph)
   */
  async traverseGraphRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<TraversalResult> {
    const { data } = await this.gen.POST("/api/v2/graphs/{graph_id}/traverse", {
      params: { path: { graph_id: graphId } },
      body: body as never,
    });
    return traversalResultFromWire(requireEnvelopeData(data, "traverseGraph"));
  }

  // =========================================================================
  // Health and Monitoring
  // =========================================================================

  /**
   * Check if the server is healthy
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /health
   * OpenAPI operationId: getHealth
   */
  async health(): Promise<HealthStatus> {
    const { data } = await this.gen.GET("/health", {});
    return data as unknown as HealthStatus;
  }

  /**
   * Kubernetes-style liveness probe
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /health/live
   * OpenAPI operationId: getLiveness
   */
  async healthLive(): Promise<ProbeResponse> {
    const { data } = await this.gen.GET("/health/live", {});
    return data as unknown as ProbeResponse;
  }

  /**
   * Kubernetes-style readiness probe
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /health/ready
   * OpenAPI operationId: getReadiness
   */
  async healthReady(): Promise<ProbeResponse> {
    const { data } = await this.gen.GET("/health/ready", {});
    return data as unknown as ProbeResponse;
  }

  /**
   * Check if the server is reachable
   */
  async ping(): Promise<boolean> {
    try {
      await this.health();
      return true;
    } catch {
      return false;
    }
  }
}

// ============================================================================
// CLIENT BUILDER
// ============================================================================

/**
 * Builder for creating a ProximaDB client with custom configuration
 */
export class ClientBuilder {
  private config: Partial<ClientConfig> = {};

  /**
   * Set the server URL
   */
  url(serverUrl: string): ClientBuilder {
    this.config.url = serverUrl;
    return this;
  }

  /**
   * Set the request timeout in milliseconds
   */
  timeoutMs(timeout: number): ClientBuilder {
    this.config.timeoutMs = timeout;
    return this;
  }

  /**
   * Set the maximum number of retries
   */
  maxRetries(retries: number): ClientBuilder {
    this.config.maxRetries = retries;
    return this;
  }

  /**
   * Set the API key for authentication
   */
  apiKey(key: string): ClientBuilder {
    this.config.apiKey = key;
    return this;
  }

  /**
   * Enable or disable connection pooling
   */
  poolConnections(enable: boolean): ClientBuilder {
    this.config.poolConnections = enable;
    return this;
  }

  /**
   * Set maximum idle connections in pool
   */
  maxIdleConnections(max: number): ClientBuilder {
    this.config.maxIdleConnections = max;
    return this;
  }

  /**
   * Build the ProximaDB client
   */
  build(): ProximaDBClient {
    return new ProximaDBClient(this.config);
  }

  /**
   * Connect to the server (alias for build)
   */
  connect(): ProximaDBClient {
    return this.build();
  }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Create a ProximaDB client connected to the given URL
 */
export function connect(
  url: string = "http://localhost:5678",
): ProximaDBClient {
  return ProximaDBClient.connect(url);
}

/**
 * Create a ProximaDB client with REST protocol
 */
export function connectRest(
  url: string = "http://localhost:5678",
): ProximaDBClient {
  return ProximaDBClient.connect(url);
}
