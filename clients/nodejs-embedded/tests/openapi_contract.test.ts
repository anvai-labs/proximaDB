/**
 * OpenAPI contract gate for the TypeScript REST SDK.
 *
 * For every covered SDK method this test:
 *   1. Stubs `globalThis.fetch` to capture the outbound request and return a
 *      programmed minimal-valid response.
 *   2. Calls the SDK method with a sample input.
 *   3. Asserts the captured URL/method/body against the corresponding
 *      OpenAPI operation defined in
 *      `docs/openapi/proximadb-openapi.yaml`.
 *
 * No server is started; drift between the SDK and the contract therefore
 * fails this test deterministically. Mirrors the Python contract gate at
 * `clients/python/tests/unit/test_openapi_contract.py`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as YAML from "yaml";
import { ProximaDBClient } from "../src/client";

// ---------------------------------------------------------------------------
// Spec loading
// ---------------------------------------------------------------------------

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const SPEC_PATH = path.join(
  REPO_ROOT,
  "docs",
  "openapi",
  "proximadb-openapi.yaml",
);

type AnyObj = Record<string, unknown>;

function loadSpec(): AnyObj {
  if (!fs.existsSync(SPEC_PATH)) {
    throw new Error(`OpenAPI spec missing at ${SPEC_PATH}`);
  }
  return YAML.parse(fs.readFileSync(SPEC_PATH, "utf-8")) as AnyObj;
}

const SPEC: AnyObj = loadSpec();

function resolveRef(spec: AnyObj, ref: string): AnyObj {
  if (!ref.startsWith("#/")) {
    throw new Error(`Unsupported $ref form: ${ref}`);
  }
  let node: unknown = spec;
  for (const part of ref.slice(2).split("/")) {
    node = (node as AnyObj)[part];
    if (node === undefined) {
      throw new Error(`Cannot resolve ref ${ref} (broke at "${part}")`);
    }
  }
  return node as AnyObj;
}

function operationFor(
  spec: AnyObj,
  pathTemplate: string,
  method: string,
): AnyObj {
  const paths = spec.paths as AnyObj;
  const pathItem = paths[pathTemplate] as AnyObj | undefined;
  if (!pathItem) {
    throw new Error(`Path ${pathTemplate} not in OpenAPI spec`);
  }
  const op = pathItem[method.toLowerCase()] as AnyObj | undefined;
  if (!op) {
    throw new Error(`${method} ${pathTemplate} not defined in spec`);
  }
  return op;
}

function requestBodySchema(spec: AnyObj, operation: AnyObj): AnyObj | null {
  const body = operation.requestBody as AnyObj | undefined;
  if (!body) return null;
  const content = body.content as AnyObj;
  const json = content["application/json"] as AnyObj;
  let schema = json.schema as AnyObj;
  if (schema.$ref) {
    schema = resolveRef(spec, schema.$ref as string);
  }
  return schema;
}

/**
 * Lightweight schema-level check: confirm every top-level `required` key
 * present in the (resolved, possibly allOf-merged) request schema is also
 * present in the actual payload.
 */
function collectRequiredKeys(spec: AnyObj, schema: AnyObj): string[] {
  const keys: string[] = [];
  const visit = (s: AnyObj): void => {
    if (s.$ref) {
      visit(resolveRef(spec, s.$ref as string));
      return;
    }
    if (Array.isArray(s.allOf)) {
      for (const sub of s.allOf as AnyObj[]) {
        visit(sub);
      }
    }
    if (Array.isArray(s.required)) {
      for (const k of s.required as string[]) {
        if (!keys.includes(k)) keys.push(k);
      }
    }
  };
  visit(schema);
  return keys;
}

// ---------------------------------------------------------------------------
// Capturing fetch stub
// ---------------------------------------------------------------------------

interface Captured {
  url: string | null;
  method: string | null;
  body: AnyObj | null;
}

function installFetchStub(responseBody: unknown, status = 200): Captured {
  const captured: Captured = { url: null, method: null, body: null };

  const stub = vi.fn(async (url: string, init?: RequestInit) => {
    captured.url = url;
    captured.method = (init?.method ?? "GET").toUpperCase();
    captured.body =
      init?.body !== undefined && init?.body !== null
        ? (JSON.parse(String(init.body)) as AnyObj)
        : null;
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: "OK",
      json: async () => responseBody,
      text: async () => JSON.stringify(responseBody),
    };
  });

  // The client captures `globalThis.fetch` at construction time, so we must
  // patch the global before constructing the client in each test.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).fetch = stub;

  return captured;
}

function makeClient(): ProximaDBClient {
  return new ProximaDBClient({
    url: "http://contract.test",
    maxRetries: 0,
  });
}

const ORIGINAL_FETCH = globalThis.fetch;

beforeEach(() => {
  // ensure a clean slate each test
});

afterEach(() => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).fetch = ORIGINAL_FETCH;
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Per-operation contract checks (6 new v2 ops)
// ---------------------------------------------------------------------------

describe("OpenAPI contract gate", () => {
  it("applies timeout signals to generated transport requests", async () => {
    vi.useFakeTimers();
    const abortStates: boolean[] = [];
    const stub = vi.fn(
      async (_url: string, init?: RequestInit): Promise<Response> =>
        new Promise((_resolve, reject) => {
          const signal = init?.signal;
          expect(signal).toBeDefined();
          signal?.addEventListener("abort", () => {
            abortStates.push(signal.aborted);
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = stub;

    try {
      const client = new ProximaDBClient({
        url: "http://contract.test",
        timeoutMs: 5,
        maxRetries: 0,
      });

      const request = expect(client.healthLive()).rejects.toThrow("aborted");
      await vi.advanceTimersByTimeAsync(5);

      await request;
      expect(abortStates).toEqual([true]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("healthLive() matches GET /health/live (getLiveness)", async () => {
    const captured = installFetchStub({ status: "ok" });
    const client = makeClient();

    const probe = await client.healthLive();

    const op = operationFor(SPEC, "/health/live", "get");
    expect(op.operationId).toBe("getLiveness");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe("http://contract.test/health/live");
    expect(captured.body).toBeNull();
    expect(probe.status).toBe("ok");
  });

  it("healthReady() matches GET /health/ready (getReadiness)", async () => {
    const captured = installFetchStub({ status: "ready" });
    const client = makeClient();

    const probe = await client.healthReady();

    const op = operationFor(SPEC, "/health/ready", "get");
    expect(op.operationId).toBe("getReadiness");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe("http://contract.test/health/ready");
    expect(captured.body).toBeNull();
    expect(probe.status).toBe("ready");
  });

  it("getCollectionSchema() matches GET /api/v2/collections/{id}/schema (getCollectionSchema)", async () => {
    const captured = installFetchStub({
      schema_id: "sch_1",
      schema_version: "v1",
      collection_id: "col_abc",
      schema: { columns: [{ name: "title", data_type: "text" }] },
      created_at: "2026-05-23T00:00:00Z",
    });
    const client = makeClient();

    const resp = await client.getCollectionSchema("col_abc");

    const op = operationFor(
      SPEC,
      "/api/v2/collections/{collection_id}/schema",
      "get",
    );
    expect(op.operationId).toBe("getCollectionSchema");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe(
      "http://contract.test/api/v2/collections/col_abc/schema",
    );
    expect(captured.body).toBeNull();
    expect(resp.schema_id).toBe("sch_1");
  });

  it("updateCollectionSchema() matches PUT /api/v2/collections/{id}/schema (updateCollectionSchema)", async () => {
    const captured = installFetchStub({
      schema_id: "sch_2",
      schema_version: "v2",
      previous_schema_id: "sch_1",
      changes: [],
      warnings: [],
      updated_at: "2026-05-23T00:00:00Z",
    });
    const client = makeClient();

    const requestBody = {
      columns: [{ name: "title", data_type: "text" }],
      enforcement: "strict" as const,
      force: true,
    };
    const resp = await client.updateCollectionSchema("col_abc", requestBody);

    const op = operationFor(
      SPEC,
      "/api/v2/collections/{collection_id}/schema",
      "put",
    );
    expect(op.operationId).toBe("updateCollectionSchema");
    expect(captured.method).toBe("PUT");
    expect(captured.url).toBe(
      "http://contract.test/api/v2/collections/col_abc/schema",
    );

    // Body must satisfy the spec's required top-level keys.
    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("columns");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    expect(captured.body!.force).toBe(true);
    expect(resp.schema_id).toBe("sch_2");
  });

  it("executeQuery() matches POST /api/v2/query (executeQuery)", async () => {
    const captured = installFetchStub({
      records: [],
      total_count: 0,
    });
    const client = makeClient();

    const resp = await client.executeQuery({
      language: "aql",
      query: "MATCH (n) RETURN n LIMIT 1",
      collection: "col_abc",
    });

    const op = operationFor(SPEC, "/api/v2/query", "post");
    expect(op.operationId).toBe("executeQuery");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/query");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("language");
    expect(requiredKeys).toContain("query");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    expect(resp).toBeDefined();
  });

  it("explainQuery() matches POST /api/v2/query/explain (explainQuery)", async () => {
    const captured = installFetchStub({
      plan: { node: "Scan" },
    });
    const client = makeClient();

    const resp = await client.explainQuery({
      language: "uql",
      query: "SELECT * FROM col_abc LIMIT 1",
    });

    const op = operationFor(SPEC, "/api/v2/query/explain", "post");
    expect(op.operationId).toBe("explainQuery");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/query/explain");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("language");
    expect(requiredKeys).toContain("query");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    expect(resp).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // Graph operations (v2)
  //
  // The SDK was migrated from /api/v1/graphs/* (which never matched a real
  // server route) to /api/v2/graphs/* in a prior change, and the four graph
  // POST body shapes were reconciled against the live server handler
  // (`crates/platform/proximadb-api/src/rest/v1/graph.rs`) and OpenAPI spec
  // on 2026-05-30 (TD-095 closeout). All graph contract checks below assert
  // both URL/method and (where applicable) server-true body shape.
  // -------------------------------------------------------------------------

  // TD-095 (resolved 2026-05-30): SDK `GraphBuilder.execute()` now posts
  // `{ graph_id, name?, description? }` to match `CreateGraphRequest`.
  it("createGraph(...).execute() matches POST /api/v2/graphs (createGraph)", async () => {
    const captured = installFetchStub({ graph_id: "g1", success: true });
    const client = makeClient();

    await client.createGraph("g1").description("test graph").execute();

    const op = operationFor(SPEC, "/api/v2/graphs", "post");
    expect(op.operationId).toBe("createGraph");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("graph_id");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    expect(captured.body!.graph_id).toBe("g1");
  });

  it("listGraphs() matches GET /api/v2/graphs (listGraphs)", async () => {
    const captured = installFetchStub({
      success: true,
      data: [
        {
          graph_id: "g1",
          name: "",
          description: "test graph",
          stats: { total_nodes: 3, total_edges: 2 },
        },
      ],
    });
    const client = makeClient();

    const resp = await client.listGraphs();

    const op = operationFor(SPEC, "/api/v2/graphs", "get");
    expect(op.operationId).toBe("listGraphs");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs");
    expect(captured.body).toBeNull();
    expect(resp).toEqual([
      {
        name: "g1",
        nodeCount: 3,
        edgeCount: 2,
        description: "test graph",
      },
    ]);
  });

  it("graph(name).info() matches GET /api/v2/graphs/{graph_id} (getGraph)", async () => {
    const captured = installFetchStub({
      success: true,
      data: {
        graph_id: "g1",
        name: "",
        description: "test graph",
        stats: { total_nodes: 4, total_edges: 3 },
      },
    });
    const client = makeClient();

    const info = await client.graph("g1").info();

    const op = operationFor(SPEC, "/api/v2/graphs/{graph_id}", "get");
    expect(op.operationId).toBe("getGraph");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1");
    expect(captured.body).toBeNull();
    expect(info).toEqual({
      name: "g1",
      nodeCount: 4,
      edgeCount: 3,
      description: "test graph",
    });
  });

  it("deleteGraph(name) matches DELETE /api/v2/graphs/{graph_id} (deleteGraph)", async () => {
    const captured = installFetchStub({ deleted: true, name: "g1" });
    const client = makeClient();

    await client.deleteGraph("g1");

    const op = operationFor(SPEC, "/api/v2/graphs/{graph_id}", "delete");
    expect(op.operationId).toBe("deleteGraph");
    expect(captured.method).toBe("DELETE");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1");
    expect(captured.body).toBeNull();
  });

  // TD-095 (resolved 2026-05-30): SDK `NodeBuilder.execute()` now posts the
  // singleton wrapper `{ node: NodeInput }` per `CreateNodeRequest`.
  it("addNode(...).execute() matches POST /api/v2/graphs/{graph_id}/nodes (createNode)", async () => {
    const captured = installFetchStub({ id: "n1" });
    const client = makeClient();

    await client
      .graph("g1")
      .addNode()
      .id("n1")
      .label("Person")
      .property("name", "Alice")
      .execute();

    const op = operationFor(SPEC, "/api/v2/graphs/{graph_id}/nodes", "post");
    expect(op.operationId).toBe("createNode");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1/nodes");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("node");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
  });

  it("getNode(id) matches GET /api/v2/graphs/{graph_id}/nodes/{node_id} (getNode)", async () => {
    const captured = installFetchStub({
      success: true,
      data: {
        id: "n1",
        labels: ["Person", "Employee"],
        properties: { active: true },
        embedding: { vector: [0.5, 0.25] },
        created_at: "2026-09-05T00:00:00Z",
        updated_at: "2026-09-05T00:00:00Z",
      },
    });
    const client = makeClient();

    const node = await client.graph("g1").getNode("n1");

    const op = operationFor(
      SPEC,
      "/api/v2/graphs/{graph_id}/nodes/{node_id}",
      "get",
    );
    expect(op.operationId).toBe("getNode");
    expect(captured.method).toBe("GET");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1/nodes/n1");
    expect(captured.body).toBeNull();
    expect(node).toEqual({
      id: "n1",
      label: "Person",
      properties: { active: true },
      vector: [0.5, 0.25],
    });
  });

  // Envelope negative controls: a 2xx body without the canonical
  // {success, data} envelope is a server-contract violation — the SDK must
  // throw, never fabricate an empty node / traversal / list out of a real
  // failure (the silent-Ok(failure) anti-pattern).
  it("getNode() throws on a 200 flat body without the envelope", async () => {
    // Pre-wave-6 flat body: node fields at the top level, no `data` member.
    installFetchStub({ id: "n1", labels: ["Person"], properties: {} });
    const client = makeClient();

    await expect(client.graph("g1").getNode("n1")).rejects.toThrow(
      /missing the canonical \{success, data\} envelope/,
    );
  });

  it("getNode() throws on a 200 success:false body", async () => {
    installFetchStub({
      success: false,
      error: { code: "INTERNAL", message: "boom" },
    });
    const client = makeClient();

    await expect(client.graph("g1").getNode("n1")).rejects.toThrow(
      /missing the canonical \{success, data\} envelope/,
    );
  });

  it("traverse() throws on a 200 body without the envelope", async () => {
    installFetchStub({ nodes: [], edges: [] });
    const client = makeClient();

    await expect(
      client.graph("g1").traverse().start("n1").maxDepth(2).execute(),
    ).rejects.toThrow(/missing the canonical \{success, data\} envelope/);
  });

  it("listGraphs() throws on a 200 body without the envelope", async () => {
    installFetchStub({ graphs: [] });
    const client = makeClient();

    await expect(client.listGraphs()).rejects.toThrow(
      /missing the canonical \{success, data\} envelope/,
    );
  });

  it("deleteNode(id) matches DELETE /api/v2/graphs/{graph_id}/nodes/{node_id} (deleteNode)", async () => {
    const captured = installFetchStub({ deleted: true, id: "n1" });
    const client = makeClient();

    await client.graph("g1").deleteNode("n1");
    const op = operationFor(
      SPEC,
      "/api/v2/graphs/{graph_id}/nodes/{node_id}",
      "delete",
    );
    expect(op.operationId).toBe("deleteNode");
    expect(captured.method).toBe("DELETE");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1/nodes/n1");
    expect(captured.body).toBeNull();
  });

  // TD-095 (resolved 2026-05-30): SDK `EdgeBuilder.execute()` now posts the
  // singleton wrapper `{ edge: EdgeInput }` per `CreateEdgeRequest`, with
  // server-true keys `id` / `from_node_id` / `to_node_id` / `edge_type`.
  it("addEdge(...).execute() matches POST /api/v2/graphs/{graph_id}/edges (createEdge)", async () => {
    const captured = installFetchStub({ id: "e1" });
    const client = makeClient();

    await client
      .graph("g1")
      .addEdge()
      .id("e1")
      .from("n1")
      .to("n2")
      .relationship("KNOWS")
      .execute();

    const op = operationFor(SPEC, "/api/v2/graphs/{graph_id}/edges", "post");
    expect(op.operationId).toBe("createEdge");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1/edges");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("edge");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    // EdgeInput inner shape: server-true keys present.
    const edge = (captured.body as Record<string, unknown>).edge as Record<
      string,
      unknown
    >;
    expect(edge.id).toBe("e1");
    expect(edge.from_node_id).toBe("n1");
    expect(edge.to_node_id).toBe("n2");
    expect(edge.edge_type).toBe("KNOWS");
  });

  // TD-095 (resolved 2026-05-30): SDK `TraversalBuilder.execute()` now
  // posts the flat `TraverseRequest` shape: `start_node_id` (not
  // `start_node`), `edge_types` (not `relationships`), no `graph` wrapper,
  // no `direction`.
  it("traverse(...).execute() matches POST /api/v2/graphs/{graph_id}/traverse (traverseGraph)", async () => {
    const captured = installFetchStub({
      success: true,
      data: {
        nodes: [
          {
            id: "n1",
            labels: ["Person"],
            properties: { active: true },
            created_at: "2026-09-05T00:00:00Z",
            updated_at: "2026-09-05T00:00:00Z",
          },
        ],
        edges: [
          {
            id: "e1",
            from_node_id: "n1",
            to_node_id: "n2",
            edge_type: "KNOWS",
            properties: { since: 2020 },
            weight: 0.75,
            created_at: "2026-09-05T00:00:00Z",
            updated_at: "2026-09-05T00:00:00Z",
          },
        ],
        paths: [["n1", "n2"]],
      },
    });
    const client = makeClient();

    const result = await client
      .graph("g1")
      .traverse()
      .start("n1")
      .maxDepth(2)
      .execute();

    const op = operationFor(SPEC, "/api/v2/graphs/{graph_id}/traverse", "post");
    expect(op.operationId).toBe("traverseGraph");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe("http://contract.test/api/v2/graphs/g1/traverse");

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("start_node_id");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    // Server-true flat shape: no `graph`, no `start_node`, no `direction`.
    expect(captured.body!.start_node_id).toBe("n1");
    expect(captured.body!.max_depth).toBe(2);
    expect(captured.body).not.toHaveProperty("graph");
    expect(captured.body).not.toHaveProperty("start_node");
    expect(captured.body).not.toHaveProperty("direction");
    expect(result).toEqual({
      nodes: [{ id: "n1", label: "Person", properties: { active: true } }],
      edges: [
        {
          source: "n1",
          target: "n2",
          relationship: "KNOWS",
          properties: { since: 2020 },
          weight: 0.75,
        },
      ],
      paths: [["n1", "n2"]],
    });
  });

  // -------------------------------------------------------------------------
  // Batch graph ops (added 2026-05-30 alongside TD-095 SDK rewrite).
  // -------------------------------------------------------------------------

  it("addNodes(...) matches POST /api/v2/graphs/{graph_id}/nodes/batch (batchCreateNodes)", async () => {
    const captured = installFetchStub({
      success: true,
      data: {
        created_count: 1,
        updated_count: 0,
        failed_count: 1,
        results: [],
        errors: [{ id: "n2", error: "rejected" }],
      },
    });
    const client = makeClient();

    const added = await client.graph("g1").addNodes([
      { id: "n1", labels: ["Person"], properties: { name: "Alice" } },
      { id: "n2", labels: ["Person"] },
    ]);

    const op = operationFor(
      SPEC,
      "/api/v2/graphs/{graph_id}/nodes/batch",
      "post",
    );
    expect(op.operationId).toBe("batchCreateNodes");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe(
      "http://contract.test/api/v2/graphs/g1/nodes/batch",
    );

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("nodes");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    // No `graph` wrapper — graph_id is the URL path parameter.
    expect(captured.body).not.toHaveProperty("graph");
    const nodes = captured.body!.nodes as Array<Record<string, unknown>>;
    expect(nodes).toHaveLength(2);
    expect(nodes[0].id).toBe("n1");
    expect(added).toBe(1);
  });

  it("addEdges(...) matches POST /api/v2/graphs/{graph_id}/edges/batch (batchCreateEdges)", async () => {
    const captured = installFetchStub({
      success: true,
      data: {
        created_count: 0,
        updated_count: 0,
        failed_count: 1,
        results: [],
        errors: [{ id: "e1", error: "rejected" }],
      },
    });
    const client = makeClient();

    const added = await client.graph("g1").addEdges([
      {
        id: "e1",
        from_node_id: "n1",
        to_node_id: "n2",
        edge_type: "KNOWS",
      },
    ]);

    const op = operationFor(
      SPEC,
      "/api/v2/graphs/{graph_id}/edges/batch",
      "post",
    );
    expect(op.operationId).toBe("batchCreateEdges");
    expect(captured.method).toBe("POST");
    expect(captured.url).toBe(
      "http://contract.test/api/v2/graphs/g1/edges/batch",
    );

    const reqSchema = requestBodySchema(SPEC, op);
    expect(reqSchema).not.toBeNull();
    const requiredKeys = collectRequiredKeys(SPEC, reqSchema!);
    expect(requiredKeys).toContain("edges");
    for (const key of requiredKeys) {
      expect(captured.body).not.toBeNull();
      expect(Object.keys(captured.body!)).toContain(key);
    }
    expect(captured.body).not.toHaveProperty("graph");
    const edges = captured.body!.edges as Array<Record<string, unknown>>;
    expect(edges).toHaveLength(1);
    expect(edges[0].id).toBe("e1");
    expect(edges[0].from_node_id).toBe("n1");
    expect(edges[0].to_node_id).toBe("n2");
    expect(edges[0].edge_type).toBe("KNOWS");
    expect(added).toBe(0);
  });
});
