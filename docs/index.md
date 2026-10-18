# ProximaDB

**A SQL-first context engine on object storage** — a Postgres-compatible database in which vectors, JSON documents and relationships are first-class, so applications and agents retrieve, transact, analyse and relate over one governed record. What is supported today is stated exactly in the [support contract](https://github.com/anvai-labs/proximaDB/blob/main/docs/SUPPORTED_SURFACE.adoc).

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
  C[Client] --> P[":5678<br/>REST · gRPC · Arrow Flight"]
  C --> Q[":5433<br/>Postgres wire"]
  P --> R{Query router}
  Q --> R
  R -->|vector / ANN| V["SST · HELIX · NOVA<br/>(VIPER: deprecated, existing collections only)"]
  R -->|relational / OLAP| D[DataFusion over Parquet]
  R -->|graph| G[ORION]
  V --> S[(Object storage)]
  D --> S
  G --> S
  style R fill:#3498db,color:#fff
  style S fill:#95a5a6,color:#fff
```

---

## Start here

| | |
|---|---|
| **[Quick Start](01-quick-start/index.md)** | Install, run your first query, and understand the moving parts |
| **[Guides](02-guides/index.md)** | Vector search, index selection, multi-model joins |
| **[Concepts](05-concepts/index.md)** | Quantization, the query planner, the unified WAL |
| **[API Reference](03-api-reference/index.md)** | REST, gRPC, Arrow Flight and pgwire surfaces |
| **[Operations](04-operations/index.md)** | Running and releasing ProximaDB |

## What makes it different

**SQL is the front door.** Point any Postgres client at port 5433 on a locally run server (the
published Docker image keeps pgwire on its loopback until authenticated exposure lands in v0.5).
Vector search
(`vector_search` and the pgvector `<->` operator) and JSON operators work inside
SQL; transactions with Postgres semantics are being built behind default-off
gates.

**One query surface, several engines.** You submit a query and the engine picks
the execution path — DataFusion's vectorised kernels for analytical scans over
Parquet, purpose-built engines for vector/ANN work, ORION for graph traversal.
You do not choose the engine; the planner does, and it discloses what it chose.

**Co-located vectors and metadata.** Vectors, payload and the coarse index live
in one object, so a filtered vector search prunes before it ranks and the
payload comes back without a second round trip. See
[Vector Index Selection & Benchmarking](02-guides/vector-index-selection-and-benchmarking.md)
for when that wins and when a separate index is the better shape — with
measurements you can reproduce on your own data.

**Designed against the cost of object storage.** Round-trips dominate, not CPU,
so the storage format, the codecs, the caches and the planner are tuned together
against measured per-query I/O rather than component microbenchmarks.

## Choosing a wire protocol

| Surface | Port | Use it for |
|---|---|---|
| Postgres wire | 5433 | SQL from any Postgres client or BI tool |
| REST | 5678 | Ergonomic HTTP, SDKs |
| gRPC | 5678 | Typed RPC, low overhead |
| Arrow Flight | 5678 | Bulk / columnar transfer, zero-copy (canonical `vector_search` is Supported; general transport is experimental) |

All three of REST, gRPC and Arrow Flight are multiplexed on one port — they
coexist rather than competing.

## Where it is going

One plan, by release: v0.5 makes the SQL front door correct and durable, v0.6
brings vectors and context fully inside SQL, v0.7 measures cost and performance
on real object stores, and v1.0 makes pgwire SQL and transactions part of the
supported promise. See the
[roadmap](https://github.com/anvai-labs/proximaDB/blob/main/docs/_internal/roadmap/STRATEGIC_ROADMAP.adoc)
and the
[direction decision](https://github.com/anvai-labs/proximaDB/blob/main/docs/12-design/adr/ADR-096-product-direction-sql-first-context-engine.adoc).

!!! note "Supported surface"
    Code presence is broader than the supported product surface. For
    supported / beta / experimental status of a given feature, see
    `docs/SUPPORTED_SURFACE.adoc` in the repository.
