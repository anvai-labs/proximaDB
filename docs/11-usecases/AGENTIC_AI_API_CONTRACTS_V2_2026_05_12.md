# Agentic AI API Contracts V2

Date: 2026-05-12

This design turns the agentic backing-store MVP into concrete contracts while
staying aligned with `roadmap/MULTIMODAL_OVERHAUL_SPEC_2026_05_08.adoc` and
`docs/history/planning/WORKSPACE_REFACTOR_PLAN_2026_05_07.adoc`. It was checked
against current LangGraph persistence docs and BaseStore reference material on
2026-05-12. The implementation strategy is contract-first: keep compatibility
facades stable during the workspace migration, then extract the same contracts
into modality/runtime crates when the build split lands.

## MVP Principle

ProximaDB should be usable as one embedded or server-backed agent state engine:
documents for JSON state, vectors for semantic retrieval, graph for topology,
relational/SQL for typed metadata, events for replay/audit, and observability
for traces/logs/metrics. All lanes should read and write `ProximaRecord` once
the multimodal record envelope is fully landed.

The MVP must not overclaim. Vector and graph paths are the strongest today;
document, relational, event, and observability paths remain experimental until
the supported-surface docs and durability tests say otherwise.

## Canonical Storage Lanes

| Lane | Contract | Primary surface |
| --- | --- | --- |
| Long-term memory | namespaced JSON KV, filters, timestamps, semantic search | Python `ProximaBaseStore`; later `/api/v2/stores/{store}/items` |
| Checkpoints | thread state, checkpoint history, pending writes, delete thread | Python `ProximaCheckpointSaver`; later `/api/v2/checkpoints/{thread_id}` |
| Vectors | dense/sparse/hybrid upsert/search with metadata filters | existing vector API, `VECTOR_SEARCH`, pgwire vector operators |
| Documents | JSON/JSONB CRUD, indexed JSON paths, schema evolution | document API, SQL JSON operators |
| Graph | nodes/edges, labels, properties, Cypher traversal | graph API, `GRAPH_QUERY`, Cypher endpoint |
| Relational | typed rows, joins, projection, ordering, pagination | SQL frontend and pgwire |
| Events | append-only stream, expected version, replay, snapshots | event API and `EVENTS` logical operator |
| Observability | logs, metrics, traces, span lookup | existing observability API and logical operators |

## Entity And Fusion Boundary

The SKS/entity shape is a convenience contract, not a new storage lane. An
entity spans existing primitives:

| Entity concern | Backing primitive | API owner |
| --- | --- | --- |
| Stable semantic identity and typed fields | `ProximaRecord` id + `props` | record/collection APIs |
| Current topology and explicit relations | graph node + graph edges | graph APIs and `GRAPH_QUERY` |
| One or more embeddings | vector-bearing records / embedding cells | record/vector APIs and `VECTOR_SEARCH` |
| Source chunks and evidence | document records / text fields | document APIs and `DOCUMENT_QUERY` |
| Provenance, temporal validity, replay | event/history records | event/logical query path |
| Cross-modal ranking | `oid`-keyed fusion seam | query/fusion layer |

Do not introduce a v2 entity store with its own WAL, path layout, or recovery
rules. If a v2 `EntityService` is exposed, it should be an orchestration facade:

- `UpsertEntity`: write/update the graph node, vector-bearing records, document
  provenance, and optional temporal/event records through their owning services.
- `GetEntity`: assemble the view from graph + record/vector + document/event
  lookups.
- `SearchEntities`: seed from vector search, apply record/document predicates,
  optionally expand through graph, then fuse/rank via the shared `oid` seam.

This avoids the wrong seam: separate "entity storage" that duplicates graph
nodes, vector records, document metadata, and temporal state.

## Server API V2

The v2 server surface should be additive and map to the existing service layer:

- `POST /api/v2/records/{collection}`: upsert `ProximaRecord` envelopes.
- `GET /api/v2/records/{collection}/{id}`: fetch by stable record id.
- `POST /api/v2/query`: execute logical algebra plans or surface-language text.
- `POST /api/v2/stores/{store}/items`: put namespaced agent memory.
- `GET /api/v2/stores/{store}/items`: get/search memory by namespace, key, filter, or semantic query.
- `POST /api/v2/checkpoints/{thread_id}`: persist checkpoint.
- `GET /api/v2/checkpoints/{thread_id}`: latest/history lookup.
- `POST /api/v2/events/{stream}`: append with expected stream version.
- `GET /api/v2/events/{stream}`: replay stream or snapshot range.

gRPC and Arrow Flight should expose the same request/response shapes. Arrow
Flight is the preferred high-volume scan and result transport; REST and gRPC are
control-plane and common SDK paths.

## Query Surface

The logical algebra stays canonical. Surface languages lower into the same plan:

- SQL/pgwire for relational, vector, document, observability, and event scans.
- Cypher for graph traversal and graph pattern matching.
- JSON/JSONB operators for document fields: `->`, `->>`, `@>`, `?`, JSON path
  predicates, and `jsonb_path_exists`-style filters.
- Cross-modal functions: `VECTOR_SEARCH`, `DOCUMENT_QUERY`, `GRAPH_QUERY`,
  `LOGS`, `METRICS`, `TRACES`, `EVENTS`, `CHECKPOINTS`, and `STORE_SEARCH`.

Minimum pgwire examples:

```sql
SELECT id, title
FROM docs
WHERE metadata->>'tenant' = 'acme'
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector
LIMIT 10;

SELECT *
FROM DOCUMENT_QUERY('profiles', '$.skills[*] ? (@ == "rust")');

SELECT *
FROM GRAPH_QUERY('MATCH (s:Symbol)-[:CALLS]->(t) WHERE s.name = $name RETURN t');
```

## SDK And Embedded Contract

The Python SDK should expose three levels:

- Low-level adapter methods: vector/document/graph/SQL/event/observability CRUD.
- Typed multimodal records: `ProximaRecord`, `ProximaType`, `ProximaValue`, and
  query result models from `models_v2.py`.
- Agentic helpers: `ProximaBaseStore`, `ProximaCheckpointSaver`, and a lightweight
  mapper/session for dataclasses or Pydantic models.
- Event helpers: `ProximaEventStore` provides append, replay, snapshot, and
  optimistic stream version checks over document storage until native event v2
  endpoints land.
- LangGraph alignment: checkpointer `put`, `put_writes`, `get_tuple`, `list`,
  async shims, and store `put`, `get`, `search`, `delete`, `list_namespaces`
  with filters, pagination, and optional semantic search.

The mapper should be embedded-first for MVP:

- `session.upsert(model)` writes a document or `ProximaRecord`.
- `session.get(Model, id)` fetches and validates.
- `session.query(Model).where(...).limit(...)` lowers to document/SQL filters.
- `session.vector_search(Model, field, vector, filter=...)` composes vector plus
  document fetch.
- `session.link(src, edge, dst, **properties)` writes graph relationships.

A full SQLAlchemy dialect is useful later, but MVP value comes faster from a
typed mapper over Rust-backed embedded calls.

## Workspace Migration Boundary

Do not create new crates only for this MVP. While workspace v2 is in flight:

- Keep root compatibility modules stable.
- Add SDK and docs first.
- Put Rust contract shims behind existing query, document, graph, network, and
  embedded facades.
- Extract to future crates only when the workspace split provides the target
  `Contracts -> Modality Runtime -> Cross-Model Query Runtime -> Platform`
  boundaries.

## Implementation Sequence

1. Land Python agent store/checkpoint helpers with fake-adapter unit tests.
2. Add embedded smoke tests that exercise document-backed memory and checkpoint
   replay against `proximadb_embedded`.
3. Add event-store adapter and a Victor embedded provider.
4. Add lightweight mapper/session over document + vector + graph methods.
5. Tighten pgwire JSON/vector syntax tests and Cypher lowering tests.
6. Add one cross-modal embedded test: memory document + vector hit + graph edge
   + checkpoint + event + trace, queried through SDK helpers and one unified
   query path.

## External References

- LangGraph persistence docs: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph BaseStore reference: https://reference.langchain.com/javascript/langchain-langgraph/index/BaseStore
- LangGraph SDK namespace listing reference: https://reference.langchain.com/python/langgraph-sdk/_async/store/StoreClient/list_namespaces
