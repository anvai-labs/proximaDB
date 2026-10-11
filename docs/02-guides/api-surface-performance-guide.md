# API Surface and Performance Selection Guide

This guide explains which ProximaDB API surface to use for common insert and query workloads. The benchmark numbers below are from the local Python embedded release wheel on macOS arm64, Python 3.12, scale 200, dimension 64, three isolated runs.

Source artifact: `artifacts/python_embedded_modalities_search_sql_uql_cypher_2026_05_19.json`.

## Quick Choice

| Workload | Choose | Why |
|---|---|---|
| Highest-throughput in-process vector or record insert | Python embedded native batch APIs | Avoids network and protocol overhead; preserves `ProximaRecord` shape. |
| Bulk dataframe/table ingest | Arrow embedded or Arrow Flight (experimental) | Columnar transfer, good for analytics pipelines and ETL. |
| Lowest-latency in-process vector search from Python | NumPy-native search | Avoids Python list/object conversion on the query vector. |
| Application-facing semantic search over server | REST/gRPC or SQL `VECTOR_SEARCH` | Stable multi-client surface with server-side policy/catalog enforcement. |
| Cross-model query composition | SQL extensions or UQL | One query can combine vector, document, graph, and observability sources. |
| Graph pattern queries | Cypher via `execute_cypher` or SQL `GRAPH_QUERY` | Natural graph syntax lowered into the shared graph/query stack. |
| Operational SQL clients | PostgreSQL wire | Works with SQL tools and pgvector-style clients. |
| High-throughput analytical reads | Arrow Flight / Flight SQL | Streaming columnar result transport. |

## Measured Embedded Paths

These are in-process Python embedded measurements. They are useful for choosing SDK call paths, not for estimating networked REST/gRPC latency.

| Method | API Shape | Median Throughput | Median Time | Use When | Avoid When |
|---|---|---:|---:|---|---|
| `insert_numpy(collection, ids, vectors, metadata)` | Legacy vector batch | 89.2k rows/s | 11.21 ms / 1k rows | You have pure vector batches and need compatibility with older vector APIs. | New multi-model code should prefer `ProximaRecord` shaped APIs. |
| `insert_records_profiled(... ProximaRecord ...)` vector batch | Native `ProximaRecord` dense batch | 75.1k rows/s | 13.31 ms / 1k rows | You want canonical records, rich props, and modern storage shape. | If you only need legacy vector compatibility. |
| `insert_records_profiled(... document records ...)` | Document records as `ProximaRecord` | 24.8k rows/s | 8.07 ms / 200 rows | Document/JSON helpers should emit canonical records. | If using the old document facade only for quick prototypes. |
| `insert_records_profiled(... graph node records ...)` | Graph nodes as `ProximaRecord` | 25.3k rows/s | 7.90 ms / 200 rows | Canonical graph-node state, rebuildable projections, PAX/record layout alignment. | If you need graph facade conveniences like automatic edge traversal. |
| `insert_records_profiled(... observability records ...)` | Logs/events as `ProximaRecord` | 22.8k rows/s | 8.78 ms / 200 rows | Canonical observability ingestion with rich typed props. | If you need specialized log/metric facade helpers. |
| `insert_arrow(db, collection, table)` | Arrow embedded | 82.5k rows/s | 12.12 ms / 1k rows | ETL, pandas/Arrow handoff, columnar pipelines. | If row-level canonical record helpers are simpler. |
| SQL single-row insert loop | Embedded SQL DML | 160 rows/s | 1.25 s / 200 rows | Compatibility tests, admin scripts, tiny writes. | High-throughput ingest. Use batch insert or native records. |
| SQL multi-row insert | Embedded SQL DML batch | 20.1k rows/s | 9.96 ms / 200 rows | SQL-shaped OLTP loads where records arrive in rows. | Large vector/record ingestion where native batches are available. |

## Measured Search and Query Paths

| Method | API Shape | Median Throughput | Median Time | Use When | Notes |
|---|---|---:|---:|---|---|
| `search_numpy` profiled | NumPy-native vector search | 32.4k queries/s | 1.54 ms / 50 searches | Fastest in-process Python vector search path. | Native search body was about 0.024 ms median per search in this benchmark; Python loop timing dominates the reported batch time. |
| Generic `db.search(..., query=list/array)` | Python generic vector search | 652 queries/s | 76.73 ms / 50 searches | Simple scripts and compatibility. | Slower because it uses generic Python object/list conversion. |
| Record-wire `search_numpy` profiled | NumPy-native search after canonical record insert | 26.6k queries/s | 1.88 ms / 50 searches | Fast record-shaped ingestion plus fast vector search. | Same native search path, with data inserted through `ProximaRecord`. |
| Record-wire generic `db.search` | Generic search after record insert | 339 queries/s | 147.50 ms / 50 searches | Compatibility checks. | Use NumPy-native search for performance. |
| SQL `VECTOR_SEARCH(...)` | Embedded SQL extension | 13.0k queries/s | 3.84 ms / 50 searches | SQL users, server-side query composition, pgwire/REST parity. | About 2x slower than native profiled search, but much more composable. |
| UQL `execute_unified_query(...)` | Unified query facade | 11.5k queries/s | 4.37 ms / 50 searches | Cross-model query routing and language-neutral plans. | Slightly slower than SQL extension due to unified result shaping. |
| Cypher `execute_cypher(...)` | Lowers to SQL `GRAPH_QUERY(...)` | 4.1k queries/s | 12.07 ms / 50 queries | Graph pattern queries and graph-native user workflows. | Use graph traversal API for simple fixed-depth traversals. |
| Document indexed path query | Document facade | 2.25k queries/s | 22.23 ms / 50 queries | JSON path lookup and document-centric workflows. | Prefer cataloged projections/indexes for heavy analytical scans. |
| Observability log query | Observability facade | 79.8k queries/s | 0.63 ms / 50 queries | Recent log lookup and operational filtering. | Retention and rollup policy should be cataloged for long windows. |

## Protocol Endpoints

| Surface | Default Endpoint | Primary Use | Strength | Tradeoff |
|---|---|---|---|---|
| Python embedded | In-process wheel | Local apps, notebooks, agent runtimes, benchmarks | No network hop; fastest developer loop. | Process-local deployment and Python ABI packaging. |
| REST/gRPC unified | `localhost:5678` | General application APIs, health, admin, Iceberg REST under `/iceberg/v1` | Broad client compatibility, one server port. | JSON/HTTP or gRPC framing overhead. |
| Dedicated gRPC | `localhost:5679` when unified mode is off | High-throughput service-to-service RPC | Typed protobuf contracts and streaming. | Separate port only in multi-port mode. |
| PostgreSQL wire | `localhost:5433` (loopback-bound) | SQL tools, pgvector-style clients, BI compatibility | Works with existing SQL clients. | SQL parsing and row protocol overhead. Host-local only: under the trust auth default the listener is admitted on loopback only, so a port published from a container reaches nothing — serving it off loopback needs pgwire SCRAM (TD-PGWIRE-AUTH-1) plus `[api] pg_port` (TD-PGWIRE-CONTAINER-1). |
| Arrow Flight / Flight SQL | `localhost:5680` | Analytical reads/writes, columnar streaming | High-throughput Arrow-native transport. | Requires Arrow Flight clients. |
| Iceberg REST Catalog | `localhost:5678/iceberg/v1` | Spark, Trino, DuckDB, Flink, PyIceberg catalog access | Open table interoperability without a custom connector. | Catalog/projection surface; not the hot record write path. |

## Which One To Choose

### Inserts

| If You Have | Use | Rationale |
|---|---|---|
| NumPy matrix of embeddings only | `insert_numpy` or native record batch | `insert_numpy` is fastest for legacy vector-only loads; native record batch is the target for new APIs. |
| Rich records with typed props, labels, document/graph/observability fields | `insert_proxima_records` / `insert_records_profiled` | Preserves `ProximaRecord`/`ProximaValue` as the API contract. |
| Arrow table or dataframe-style batch | `insert_arrow` / Arrow Flight | Columnar handoff avoids row-by-row Python work. |
| SQL row data | Multi-row SQL insert | Batches amortize parser/planner overhead. |
| One-off admin insert | SQL single-row insert | Simpler to write, not a performance path. |

### Searches and Queries

| If You Need | Use | Rationale |
|---|---|---|
| Fast vector search inside Python | `search_numpy` | Avoids generic object conversion and keeps query vectors contiguous. |
| Vector search from SQL clients | `VECTOR_SEARCH(...)` | Composable and compatible with pgwire/SQL workflows. |
| Vector + document + graph + observability planning | `execute_unified_query` / UQL | One facade over shared logical query planning. |
| Graph pattern syntax | `execute_cypher` or `GRAPH_QUERY(...)` | Cypher is clearer for graph users and still lowers into the shared query/storage stack. |
| Simple graph neighborhood walk | `traverse_graph` | Lower overhead than Cypher for fixed traversal operations. |
| JSON path lookup | Document query facade or `DOCUMENT_QUERY(...)` | Uses document-oriented semantics over canonical record storage/projections. |
| Logs, metrics, traces | Observability facade or SQL extensions | Keeps operational query semantics explicit. |

## Practical Guidance

Prefer canonical `ProximaRecord`/`ProximaValue` APIs for new SDK and embedded work. Legacy vector-only paths remain useful as compatibility and performance baselines, but they should not define new storage or API semantics.

Use language facades for user ergonomics, not durable authority:

| Language Facade | Lowers To | Use For |
|---|---|---|
| SQL `VECTOR_SEARCH` | Shared vector/query service | SQL and cross-model composition. |
| SQL `GRAPH_QUERY` | Shared graph/query service | Graph patterns inside SQL. |
| `execute_cypher` | SQL `GRAPH_QUERY` | Python graph users who expect Cypher. |
| UQL `execute_unified_query` | Shared logical plan | Multi-model planning and language-neutral query routing. |
| Document/observability helpers | Canonical records plus projections | Domain-specific ergonomics. |

## Benchmark Caveats

The measured numbers are local embedded timings from one machine and should be used for relative path selection. Networked REST/gRPC/pgwire/Arrow Flight performance depends on payload size, client language, TLS, serialization, batching, and server configuration.

For production benchmarking, measure the exact path you will deploy:

```bash
/Users/vijaysingh/code/.venv/bin/python \
  clients/python-embedded/benchmarks/baseline_modalities.py \
  --scale 200 \
  --dimension 64 \
  --runs 3 \
  --json-out artifacts/python_embedded_modalities_search_sql_uql_cypher_2026_05_19.json
```
