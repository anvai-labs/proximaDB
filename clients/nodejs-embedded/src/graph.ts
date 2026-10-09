/**
 * ProximaDB TypeScript SDK - Graph Operations
 *
 * Provides a fluent API for graph database operations including
 * node and edge management, traversal queries, and graph analytics.
 *
 * Copyright 2025 ProximaDB Contributors
 * Licensed under the Apache License, Version 2.0
 */

import {
  GraphNode,
  GraphEdge,
  TraversalResult,
  TraversalDirection,
  JsonValue,
  GraphInfo,
  NodeInput,
  EdgeInput,
  EmbeddingInput,
} from "./types";

/**
 * HTTP client interface for graph operations.
 *
 * Wire calls route through the typed `*Request` seam methods (TD-126 Phase 4),
 * which dispatch via the generated openapi-fetch transport. The generic
 * `get`/`delete` + `url()` remain ONLY for `deleteEdge`, whose 3-segment route
 * shape (`/edges/{source}/{target}/{rel}`) has no matching generated op (the
 * spec uses `/edges/{edge_id}`); see `GraphHandle.deleteEdge`.
 */
export interface GraphHttpClient {
  get<T>(url: string): Promise<T>;
  delete<T>(url: string): Promise<T>;
  url(): string;
  // Typed transport seams (route through the generated client).
  createGraphRequest(body: Record<string, unknown>): Promise<unknown>;
  getGraphRequest(graphId: string): Promise<GraphInfo>;
  getGraphStatsRequest(graphId: string): Promise<unknown>;
  createNodeRequest(graphId: string, body: Record<string, unknown>): Promise<unknown>;
  batchCreateNodesRequest(graphId: string, body: Record<string, unknown>): Promise<unknown>;
  getNodeRequest(graphId: string, nodeId: string): Promise<GraphNode>;
  deleteNodeRequest(graphId: string, nodeId: string): Promise<void>;
  createEdgeRequest(graphId: string, body: Record<string, unknown>): Promise<unknown>;
  batchCreateEdgesRequest(graphId: string, body: Record<string, unknown>): Promise<unknown>;
  traverseGraphRequest(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<TraversalResult>;
}

// ============================================================================
// NODE BUILDER
// ============================================================================

/**
 * Builder for creating graph nodes
 */
export class NodeBuilder {
  private handle: GraphHandle;
  private nodeId: string | null = null;
  private nodeLabel: string | null = null;
  private nodeProperties: Record<string, JsonValue> = {};
  private nodeVector: number[] | null = null;

  constructor(handle: GraphHandle) {
    this.handle = handle;
  }

  /**
   * Set the node ID
   */
  id(nodeId: string): NodeBuilder {
    this.nodeId = nodeId;
    return this;
  }

  /**
   * Set the node label (type)
   */
  label(nodeLabel: string): NodeBuilder {
    this.nodeLabel = nodeLabel;
    return this;
  }

  /**
   * Add a property to the node
   */
  property(key: string, value: JsonValue): NodeBuilder {
    this.nodeProperties[key] = value;
    return this;
  }

  /**
   * Add multiple properties at once
   */
  properties(props: Record<string, JsonValue>): NodeBuilder {
    Object.assign(this.nodeProperties, props);
    return this;
  }

  /**
   * Set the embedding vector for semantic operations
   */
  vector(vec: number[]): NodeBuilder {
    this.nodeVector = vec;
    return this;
  }

  /**
   * Execute the node addition.
   *
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/nodes
   * OpenAPI operationId: createNode
   *
   * Body shape (server-true, per `CreateNodeRequest`):
   *   `{ node: NodeInput }` — NodeInput = `{ id, labels?, properties?, embedding? }`.
   */
  async execute(): Promise<void> {
    if (this.nodeId === null) {
      throw new Error("Node ID is required");
    }

    const node: NodeInput = { id: this.nodeId };
    if (this.nodeLabel !== null) {
      node.labels = [this.nodeLabel];
    }
    if (Object.keys(this.nodeProperties).length > 0) {
      node.properties = this.nodeProperties;
    }
    if (this.nodeVector !== null) {
      const embedding: EmbeddingInput = { vector: this.nodeVector };
      node.embedding = embedding;
    }

    const request = { node };
    await this.handle.getClient().createNodeRequest(
      this.handle.getName(),
      request as unknown as Record<string, unknown>,
    );
  }

  /**
   * Build the node without executing
   */
  build(): GraphNode {
    if (this.nodeId === null) {
      throw new Error("Node ID is required");
    }

    return {
      id: this.nodeId,
      label: this.nodeLabel ?? undefined,
      properties: this.nodeProperties,
      vector: this.nodeVector ?? undefined,
    };
  }
}

// ============================================================================
// EDGE BUILDER
// ============================================================================

/**
 * Builder for creating graph edges
 */
export class EdgeBuilder {
  private handle: GraphHandle;
  private edgeId: string | null = null;
  private sourceNode: string | null = null;
  private targetNode: string | null = null;
  private relationshipType: string | null = null;
  private edgeProperties: Record<string, JsonValue> = {};
  private edgeWeight: number | null = null;

  constructor(handle: GraphHandle) {
    this.handle = handle;
  }

  /**
   * Set the edge ID (required by server `EdgeInput.id`).
   */
  id(edgeId: string): EdgeBuilder {
    this.edgeId = edgeId;
    return this;
  }

  /**
   * Set the source node ID
   */
  from(nodeId: string): EdgeBuilder {
    this.sourceNode = nodeId;
    return this;
  }

  /**
   * Set the target node ID
   */
  to(nodeId: string): EdgeBuilder {
    this.targetNode = nodeId;
    return this;
  }

  /**
   * Set the relationship / edge type (server field: `edge_type`).
   */
  relationship(relType: string): EdgeBuilder {
    this.relationshipType = relType;
    return this;
  }

  /**
   * Alias for `relationship(...)` matching the server-side field name.
   */
  edgeType(edgeType: string): EdgeBuilder {
    this.relationshipType = edgeType;
    return this;
  }

  /**
   * Add a property to the edge
   */
  property(key: string, value: JsonValue): EdgeBuilder {
    this.edgeProperties[key] = value;
    return this;
  }

  /**
   * Add multiple properties at once
   */
  properties(props: Record<string, JsonValue>): EdgeBuilder {
    Object.assign(this.edgeProperties, props);
    return this;
  }

  /**
   * Set the edge weight
   */
  weight(w: number): EdgeBuilder {
    this.edgeWeight = w;
    return this;
  }

  /**
   * Execute the edge addition.
   *
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/edges
   * OpenAPI operationId: createEdge
   *
   * Body shape (server-true, per `CreateEdgeRequest`):
   *   `{ edge: EdgeInput }` — EdgeInput = `{ id, from_node_id, to_node_id,
   *   edge_type, properties?, weight? }`.
   */
  async execute(): Promise<void> {
    if (this.edgeId === null) {
      throw new Error("Edge ID is required (call .id(...))");
    }
    if (this.sourceNode === null) {
      throw new Error("Source node ID is required");
    }
    if (this.targetNode === null) {
      throw new Error("Target node ID is required");
    }
    if (this.relationshipType === null) {
      throw new Error("Edge type / relationship is required");
    }

    const edge: EdgeInput = {
      id: this.edgeId,
      from_node_id: this.sourceNode,
      to_node_id: this.targetNode,
      edge_type: this.relationshipType,
    };
    if (Object.keys(this.edgeProperties).length > 0) {
      edge.properties = this.edgeProperties;
    }
    if (this.edgeWeight !== null) {
      edge.weight = this.edgeWeight;
    }

    const request = { edge };
    await this.handle.getClient().createEdgeRequest(
      this.handle.getName(),
      request as unknown as Record<string, unknown>,
    );
  }

  /**
   * Build the edge without executing
   */
  build(): GraphEdge {
    if (this.sourceNode === null) {
      throw new Error("Source node ID is required");
    }
    if (this.targetNode === null) {
      throw new Error("Target node ID is required");
    }
    if (this.relationshipType === null) {
      throw new Error("Relationship type is required");
    }

    return {
      source: this.sourceNode,
      target: this.targetNode,
      relationship: this.relationshipType,
      properties: this.edgeProperties,
      weight: this.edgeWeight ?? undefined,
    };
  }
}

// ============================================================================
// TRAVERSAL BUILDER
// ============================================================================

/**
 * Builder for graph traversal queries
 */
export class TraversalBuilder {
  private handle: GraphHandle;
  private startNodeId: string | null = null;
  private relationshipTypes: string[] = [];
  private nodeLabelsList: string[] = [];
  private traversalDirection: TraversalDirection = TraversalDirection.Outgoing;
  private maxDepthValue: number = 3;
  private limitValue: number | null = null;
  private algorithmValue: string | null = null;
  private filterExpr: string | null = null;

  constructor(handle: GraphHandle) {
    this.handle = handle;
  }

  /**
   * Set the starting node
   */
  start(nodeId: string): TraversalBuilder {
    this.startNodeId = nodeId;
    return this;
  }

  /**
   * Add a relationship type to follow
   */
  relationship(relType: string): TraversalBuilder {
    this.relationshipTypes.push(relType);
    return this;
  }

  /**
   * Add multiple relationship types
   */
  relationships(relTypes: string[]): TraversalBuilder {
    this.relationshipTypes.push(...relTypes);
    return this;
  }

  /**
   * Set the traversal direction
   */
  direction(dir: TraversalDirection): TraversalBuilder {
    this.traversalDirection = dir;
    return this;
  }

  /**
   * Traverse outgoing edges only
   */
  outgoing(): TraversalBuilder {
    this.traversalDirection = TraversalDirection.Outgoing;
    return this;
  }

  /**
   * Traverse incoming edges only
   */
  incoming(): TraversalBuilder {
    this.traversalDirection = TraversalDirection.Incoming;
    return this;
  }

  /**
   * Traverse both directions
   */
  both(): TraversalBuilder {
    this.traversalDirection = TraversalDirection.Both;
    return this;
  }

  /**
   * Set the maximum traversal depth
   */
  maxDepth(depth: number): TraversalBuilder {
    this.maxDepthValue = depth;
    return this;
  }

  /**
   * Set the maximum number of results
   */
  limit(lim: number): TraversalBuilder {
    this.limitValue = lim;
    return this;
  }

  /**
   * Add a label to filter visited nodes by.
   */
  nodeLabel(label: string): TraversalBuilder {
    this.nodeLabelsList.push(label);
    return this;
  }

  /**
   * Replace the set of node-label filters.
   */
  nodeLabels(labels: string[]): TraversalBuilder {
    this.nodeLabelsList = [...labels];
    return this;
  }

  /**
   * Select traversal algorithm (server accepts `bfs`, `dfs`, or `shortest_path`).
   */
  algorithm(name: string): TraversalBuilder {
    this.algorithmValue = name;
    return this;
  }

  /**
   * Add a filter expression for nodes (client-side helper, not sent on wire).
   */
  filter(filterStr: string): TraversalBuilder {
    this.filterExpr = filterStr;
    return this;
  }

  /**
   * Execute the traversal.
   *
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/traverse
   * OpenAPI operationId: traverseGraph
   *
   * Body shape (server-true, per `TraverseRequest`):
   *   `{ start_node_id, max_depth, edge_types?, node_labels?, algorithm?, limit? }`.
   *   Note: spec is flat (no `graph` wrapper — `graph_id` is the URL path
   *   parameter) and has no `direction` field; legacy `direction` /
   *   `start_node` / `relationships` keys are intentionally not sent.
   */
  async execute(): Promise<TraversalResult> {
    if (this.startNodeId === null) {
      throw new Error("Start node is required for traversal");
    }

    const request: Record<string, unknown> = {
      start_node_id: this.startNodeId,
      max_depth: this.maxDepthValue,
    };
    if (this.relationshipTypes.length > 0) {
      request.edge_types = this.relationshipTypes;
    }
    if (this.nodeLabelsList.length > 0) {
      request.node_labels = this.nodeLabelsList;
    }
    if (this.algorithmValue !== null) {
      request.algorithm = this.algorithmValue;
    }
    if (this.limitValue !== null) {
      request.limit = this.limitValue;
    }
    // Note: direction (this.traversalDirection) and filter (this.filterExpr)
    // are intentionally NOT sent — the server `TraverseRequest` does not
    // accept them. The setters remain for SDK ergonomic continuity.
    void this.traversalDirection;
    void this.filterExpr;

    return this.handle.getClient().traverseGraphRequest(
      this.handle.getName(),
      request,
    );
  }
}

// ============================================================================
// GRAPH HANDLE
// ============================================================================

/**
 * Handle to a graph for fluent operations
 */
export class GraphHandle {
  private client: GraphHttpClient;
  private graphName: string;

  constructor(client: GraphHttpClient, name: string) {
    this.client = client;
    this.graphName = name;
  }

  /**
   * Get the graph name
   */
  getName(): string {
    return this.graphName;
  }

  /**
   * Get the HTTP client (internal use)
   */
  getClient(): GraphHttpClient {
    return this.client;
  }

  /**
   * Start building a node addition
   */
  addNode(): NodeBuilder {
    return new NodeBuilder(this);
  }

  /**
   * Start building an edge addition
   */
  addEdge(): EdgeBuilder {
    return new EdgeBuilder(this);
  }

  /**
   * Start building a traversal query
   */
  traverse(): TraversalBuilder {
    return new TraversalBuilder(this);
  }

  /**
   * Add a batch of nodes.
   *
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/nodes/batch
   * OpenAPI operationId: batchCreateNodes
   *
   * Body shape (server-true, per `BatchCreateNodesRequest`):
   *   `{ nodes: [NodeInput, ...] }`. `graph_id` lives in the URL path.
   *
   * Accepts either spec-true `NodeInput[]` or legacy `GraphNode[]`; the
   * latter is normalized (singular `label` → `labels: [label]`,
   * `vector` → `embedding.vector`).
   */
  async addNodes(nodes: NodeInput[] | GraphNode[]): Promise<number> {
    const normalized: NodeInput[] = nodes.map(n => normalizeNodeInput(n));
    const request = { nodes: normalized };
    const response = (await this.client.batchCreateNodesRequest(
      this.graphName,
      request as unknown as Record<string, unknown>,
    )) as {
      data?: { created_count?: number; count?: number };
      created_count?: number;
      count?: number;
      added_count?: number;
    } | null;
    return (
      response?.data?.created_count ??
      response?.data?.count ??
      response?.created_count ??
      response?.count ??
      response?.added_count ??
      0
    );
  }

  /**
   * Add a batch of edges.
   *
   * Wire endpoint: POST /api/v2/graphs/{graph_id}/edges/batch
   * OpenAPI operationId: batchCreateEdges
   *
   * Body shape (server-true, per `BatchCreateEdgesRequest`):
   *   `{ edges: [EdgeInput, ...] }`. `graph_id` lives in the URL path.
   *
   * Accepts either spec-true `EdgeInput[]` or legacy `GraphEdge[]`; the
   * latter is normalized (`source` → `from_node_id`, `target` →
   * `to_node_id`, `relationship` → `edge_type`). Legacy `GraphEdge` has no
   * `id` field; callers MUST upgrade to `EdgeInput` for batch use (the
   * normalizer throws if `id` is missing).
   */
  async addEdges(edges: EdgeInput[] | GraphEdge[]): Promise<number> {
    const normalized: EdgeInput[] = edges.map(e => normalizeEdgeInput(e));
    const request = { edges: normalized };
    const response = (await this.client.batchCreateEdgesRequest(
      this.graphName,
      request as unknown as Record<string, unknown>,
    )) as {
      data?: { created_count?: number; count?: number };
      created_count?: number;
      count?: number;
      added_count?: number;
    } | null;
    return (
      response?.data?.created_count ??
      response?.data?.count ??
      response?.created_count ??
      response?.count ??
      response?.added_count ??
      0
    );
  }

  /**
   * Get a node by ID
   */
  async getNode(nodeId: string): Promise<GraphNode | null> {
    try {
      return await this.client.getNodeRequest(this.graphName, nodeId);
    } catch (e: unknown) {
      // The facade maps a 404 to a ProximaDBError (statusCode 404, message =
      // server body). Treat "not found" as a null result, preserving the
      // pre-rebase behavior; match on status or the body text.
      const status = (e as { statusCode?: number }).statusCode;
      if (status === 404) {
        return null;
      }
      if (e instanceof Error && e.message.includes("404")) {
        return null;
      }
      throw e;
    }
  }

  /**
   * Delete a node by ID
   */
  async deleteNode(nodeId: string): Promise<void> {
    await this.client.deleteNodeRequest(this.graphName, nodeId);
  }

  // deleteEdge is intentionally LEFT facade-built (generic URL path): the
  // generated REST client has no matching typed op for this 3-segment route
  // shape. The OpenAPI spec / generated client model edge deletion as
  // `DELETE /api/v2/graphs/{graph_id}/edges/{edge_id}` (single `edge_id`),
  // whereas this SDK's public signature is `(source, target, relationship)` →
  // `/edges/{source}/{target}/{relationship}`. Until the SDK signature is
  // reconciled to `{edge_id}` (separate change), this cannot route through the
  // generated client. Tracked separately.
  /**
   * Delete an edge
   */
  async deleteEdge(source: string, target: string, relationship: string): Promise<void> {
    const url = this.client.url() + "/api/v2/graphs/" + this.graphName + "/edges/" + source + "/" + target + "/" + relationship;
    await this.client.delete<unknown>(url);
  }

  /**
   * Get graph statistics.
   *
   * Routed through the generated typed transport (TD-126 Phase 4).
   * Wire endpoint: GET /api/v2/graphs/{graph_id} (getGraph)
   */
  async info(): Promise<GraphInfo> {
    return this.client.getGraphRequest(this.graphName);
  }
}

// ============================================================================
// GRAPH BUILDER
// ============================================================================

/**
 * Builder for creating graphs
 */
export class GraphBuilder {
  private client: GraphHttpClient;
  private graphId: string;
  private graphDisplayName: string | null = null;
  private graphDescription: string | null = null;

  constructor(client: GraphHttpClient, graphId: string) {
    this.client = client;
    this.graphId = graphId;
  }

  /**
   * Set an optional human-readable name (defaults server-side to graph_id).
   */
  name(displayName: string): GraphBuilder {
    this.graphDisplayName = displayName;
    return this;
  }

  /**
   * Set the graph description
   */
  description(desc: string): GraphBuilder {
    this.graphDescription = desc;
    return this;
  }

  /**
   * Execute the graph creation.
   *
   * Wire endpoint: POST /api/v2/graphs
   * OpenAPI operationId: createGraph
   *
   * Body shape (server-true, per `CreateGraphRequest`):
   *   `{ graph_id, name?, description? }`.
   */
  async execute(): Promise<void> {
    const request: Record<string, unknown> = { graph_id: this.graphId };
    if (this.graphDisplayName !== null) {
      request.name = this.graphDisplayName;
    }
    if (this.graphDescription !== null) {
      request.description = this.graphDescription;
    }

    await this.client.createGraphRequest(request);
  }
}

// ============================================================================
// INTERNAL: legacy -> spec normalizers
// ============================================================================

function normalizeNodeInput(input: NodeInput | GraphNode): NodeInput {
  // Distinguish legacy `GraphNode` (has singular `label` and/or `vector`) from
  // spec-true `NodeInput` (has `labels[]` / `embedding`). Field absence is
  // fine — `NodeInput` only requires `id`.
  const out: NodeInput = { id: input.id };
  // Inline structural check avoids a hard cross-cast.
  const legacy = input as GraphNode;
  const spec = input as NodeInput;
  if (spec.labels !== undefined) {
    out.labels = spec.labels;
  } else if (legacy.label !== undefined) {
    out.labels = [legacy.label];
  }
  if (spec.properties !== undefined) {
    out.properties = spec.properties;
  } else if (legacy.properties !== undefined) {
    out.properties = legacy.properties;
  }
  if (spec.embedding !== undefined) {
    out.embedding = spec.embedding;
  } else if (legacy.vector !== undefined) {
    out.embedding = { vector: legacy.vector };
  }
  return out;
}

function normalizeEdgeInput(input: EdgeInput | GraphEdge): EdgeInput {
  const spec = input as EdgeInput;
  const legacy = input as GraphEdge;
  // Spec edge has `id` (required), legacy has none. If absent on a legacy
  // value, surface a hard error — server `EdgeInput.id` is required.
  if (spec.id === undefined) {
    throw new Error(
      "addEdges: edge.id is required by EdgeInput (legacy GraphEdge has no id; pass an EdgeInput)",
    );
  }
  const out: EdgeInput = {
    id: spec.id,
    from_node_id: spec.from_node_id ?? legacy.source,
    to_node_id: spec.to_node_id ?? legacy.target,
    edge_type: spec.edge_type ?? legacy.relationship,
  };
  if (spec.properties !== undefined) {
    out.properties = spec.properties;
  } else if (legacy.properties !== undefined) {
    out.properties = legacy.properties;
  }
  if (spec.weight !== undefined) {
    out.weight = spec.weight;
  } else if (legacy.weight !== undefined) {
    out.weight = legacy.weight;
  }
  return out;
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Create a new GraphNode
 */
export function createNode(id: string): GraphNode {
  return {
    id,
    properties: {},
  };
}

/**
 * Create a new GraphNode with builder pattern
 */
export function node(id: string): {
  label: (l: string) => ReturnType<typeof node>;
  property: (k: string, v: JsonValue) => ReturnType<typeof node>;
  vector: (v: number[]) => ReturnType<typeof node>;
  build: () => GraphNode;
} {
  const n: GraphNode = { id, properties: {} };
  return {
    label: (l: string) => { n.label = l; return node(id); },
    property: (k: string, v: JsonValue) => { n.properties[k] = v; return node(id); },
    vector: (v: number[]) => { n.vector = v; return node(id); },
    build: () => n,
  };
}

/**
 * Create a new GraphEdge
 */
export function createEdge(source: string, target: string, relationship: string): GraphEdge {
  return {
    source,
    target,
    relationship,
    properties: {},
  };
}

/**
 * Create a new GraphEdge with builder pattern
 */
export function edge(source: string, target: string, relationship: string): {
  property: (k: string, v: JsonValue) => ReturnType<typeof edge>;
  weight: (w: number) => ReturnType<typeof edge>;
  build: () => GraphEdge;
} {
  const e: GraphEdge = { source, target, relationship, properties: {} };
  return {
    property: (k: string, v: JsonValue) => { e.properties[k] = v; return edge(source, target, relationship); },
    weight: (w: number) => { e.weight = w; return edge(source, target, relationship); },
    build: () => e,
  };
}
