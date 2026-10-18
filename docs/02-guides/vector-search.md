# Vector Search Guide

**Semantic search in milliseconds**

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
  A[Query Vector] --> B[Quantizer]
  B --> C[Distance Engine]
  C --> D[Top K Results]

  D --> E[Post-Filter]
  E --> F[Re-Rank]

  style B fill:#3498db,color:#fff
  style C fill:#e74c3c,color:#fff
  style F fill:#9b59b6,color:#fff
```

---

## Overview

Vector search finds similar items using embedding distance:

In the canonical API, vectors are embedding fields on `ProximaRecord` payloads.
REST v1 vector-shaped endpoints remain only as deprecated compatibility facades.

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Cosine** | `1 - (A·B / ‖A‖‖B‖)` | Semantic similarity |
| **L2** | `sqrt(Σ(A-B)²)` | Euclidean distance |
| **Dot Product** | `A·B` | Normalized vectors |

---

## Quick Example

### Python SDK

```python
from proximadb_sdk import ProximaDBClient, ProximaRecord, Protocol

client = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)

client.create_collection("products", dimension=3, storage_engine="sst")
client.insert_records("products", [
    ProximaRecord(id="sku-1", vector=[0.1, 0.2, 0.3])
        .set_flexible("category", "Electronics")
        .set_flexible("price", 79.0),
    ProximaRecord(id="sku-2", vector=[0.3, 0.2, 0.1])
        .set_flexible("category", "Books")
        .set_flexible("price", 15.0),
])

# Simple search
results = client.search("products", vector=[0.1, 0.2, 0.3], top_k=10)

# With an equality filter: use metadata_filter=, not filter= (TD-SDKERR-1)
results = client.search(
    "products", vector=[0.1, 0.2, 0.3], top_k=10, metadata_filter={"category": "Electronics"}
)
```

### REST API

```bash
curl -X POST http://localhost:5678/api/v2/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "filters": [{"field": "category", "op": "eq", "value": "Electronics"}]
  }'
```

---

## Advanced Filtering

### Metadata Filters

The Python SDK's `metadata_filter` supports equality only. For ranges and set membership, use the
REST API's typed `filters`; conditions are combined with AND (there is no OR). Operators: `eq`,
`neq`, `gt`, `gte`, `lt`, `lte`, `in`, `between` (with `value_upper`), `contains`,
`starts_with` and `ends_with`.

```bash
# Price range and category membership
curl -X POST http://localhost:5678/api/v2/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "filters": [
      {"field": "price", "op": "gte", "value": 10},
      {"field": "price", "op": "lte", "value": 100},
      {"field": "category", "op": "in", "value": ["Electronics", "Computers"]}
    ]
  }'
```

### How filters are applied

Today the vector engines apply property filters to the ANN candidates (post-filtering). When a
filtered search returns fewer than `top_k` results, the REST response carries
`predicate_shortfall` (`requested_k`, `returned_k`, `oversample_pool`,
`"ann_filtering_mode": "post_filter"`). Ask for a larger `top_k` and trim in the client.
Filter-aware ANN is tracked by TD-064.

```python
# Over-fetch, then keep the first 10 matches
results = client.search(
    "products", vector=query_vector, top_k=100, metadata_filter={"status": "active"}
)[:10]
```

---

## Hybrid Search

### Vector + Keyword

`client.hybrid_search` returns no results today: it fails inside the SDK and returns an empty
result instead of raising (TD-SDKERR-1). The call shape, for when it is fixed:

```python
# Combine semantic and keyword search (reciprocal rank fusion by default)
results = client.hybrid_search(
    "products",
    text_query="wireless headphones",
    query_vector=query_vector,
    top_k=10,
)
```

### Weighted Scoring

The SDK has no server-side scoring hook. Re-rank in the client:

```python
def prop(record, name, default=0.0):
    value = record.metadata.get(name, default)
    # Over REST, properties arrive as {"type": ..., "value": ...} cells (TD-SDKERR-1)
    return value.get("value", default) if isinstance(value, dict) else value

results = client.search("products", vector=query_vector, top_k=50)
ranked = sorted(
    results, key=lambda r: r.score * 0.8 + float(prop(r, "popularity")) * 0.2, reverse=True
)[:10]
```

---

## Batch Operations

### Bulk Insert

```python
# Efficient bulk insert through the record API: one call, many records
rows = [("1", [0.1, 0.2, 0.3], "A"), ("2", [0.3, 0.2, 0.1], "B")]
client.insert_records("products", [
    ProximaRecord(id=rid, vector=vec).set_flexible("name", name) for rid, vec, name in rows
])
```

### Bulk Search

```python
# Search multiple queries at once: one result list per query
results = client.search_batch(
    "products",
    [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
    top_k=10,
)
```

---

## Performance Tuning

### Engine Selection

```python
# SST: the Supported engine (the Python SDK's default; use storage_engine=, not engine=)
client.create_collection("events", dimension=384, storage_engine="sst")

# HELIX (Beta): locality-ordered embeddings; measure recall against SST first
client.create_collection("locations", dimension=384, storage_engine="helix")
```

### Index Parameters

The Python SDK's `create_collection` does not apply index or quantization settings today: it
accepts them and silently drops them (TD-SDKENG-1). Set them through the REST API, which stores
them (check with `GET /api/v2/collections/<name>`):

```bash
curl -X POST http://localhost:5678/api/v2/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products_hnsw",
    "dimension": 384,
    "engine": "sst",
    "distance_metric": "cosine",
    "index_configs": [{
      "algorithm": "hnsw",
      "is_primary": true,
      "hnsw_config": {"m": 16, "ef_construction": 200, "ef_search": 100}
    }]
  }'
```

`m` is graph connectivity, `ef_construction` build-time accuracy and `ef_search` query-time
accuracy (higher is more accurate and slower). `search()` takes no per-query index parameters.

### Quantization

Quantization is on by default, tiered by dimension: under 64 dimensions `minimal`, 64 to 511
`smart_defaults`, 512 and up `aggressive` (so typical 768- or 1536-dimension embeddings are
aggressively quantized unless you say otherwise). Choose a strategy explicitly, and create an
unquantized baseline to measure recall against:

```bash
# Strategies: smart_defaults, minimal, aggressive, custom_levels (the default depends on dimension)
curl -X POST http://localhost:5678/api/v2/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "products_aggressive", "dimension": 384, "engine": "sst",
       "quantization": {"enabled": true, "strategy": "aggressive"}}'

# Unquantized baseline
curl -X POST http://localhost:5678/api/v2/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "products_f32", "dimension": 384, "engine": "sst",
       "quantization": {"enabled": false}}'
```

---

## Monitoring

### Search Latency

```python
import time

start = time.time()
results = client.search("products", vector=query_vector, top_k=10)
latency = (time.time() - start) * 1000
print(f"Search latency: {latency:.2f}ms")
```

### Metrics Endpoint

```bash
curl -sL http://localhost:5678/metrics/prometheus | grep proximadb_search_latency_seconds
```

---

## Best Practices

### 1. Choose the Right Metric

| Metric | When to Use |
|--------|-------------|
| **Cosine** | Text embeddings, normalized vectors |
| **L2** | Image embeddings, unnormalized |
| **Dot Product** | Pre-normalized vectors (fastest) |

### 2. Filter Selectively

```python
# Good: a selective condition
results = client.search(
    "products",
    vector=query_vector,
    metadata_filter={"category": "Electronics"},
    top_k=10
)

# A filter that matches almost everything adds cost without narrowing the result;
# leave it out
results = client.search("products", vector=query_vector, top_k=10)
```

### 3. Batch When Possible

```python
# ids, vectors and props are parallel lists from your pipeline
# Good: one call for the batch
client.insert_records("products", [
    ProximaRecord(id=str(i), vector=v, flexible_fields=p) for i, v, p in zip(ids, vectors, props)
])

# Avoid: one call per record
for i, v in zip(ids, vectors):
    client.insert_records("products", [ProximaRecord(id=str(i), vector=v)])  # Slow
```

### 4. Use Right Engine

| Workload | Engine |
|----------|--------|
| Real-time ingest | SST |
| Spatial / locality queries | HELIX (Beta) |
| Analytical SQL | Relational table + `MATERIALIZE` (DataFusion), not a vector engine |
| Mixed read/write | NOVA (Beta) |

> **SWIFT and RAPTOR are experimental** (off by default behind the `experimental-engines`
> cargo feature) and are not supported — don't target them for production. For small or
> dynamic workloads use SST or NOVA. See
> [Storage Engines](../05-concepts/storage-engines.adoc).

---

## Common Patterns


### Recommendation Engine

```bash
# User-based collaborative filtering: exclude purchased items
curl -X POST http://localhost:5678/api/v2/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "top_k": 20,
       "filters": [{"field": "category", "op": "neq", "value": "purchased"}]}'
```

### Deduplication

```bash
# Find near-duplicates of a document, excluding the document itself
curl -X POST http://localhost:5678/api/v2/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "top_k": 10,
       "filters": [{"field": "checksum", "op": "neq", "value": "<doc checksum>"}]}'
# A score above about 0.95 suggests a duplicate
```

### Face Recognition

This assumes a `faces` collection exists. Searching a collection that does not exist currently
leaves the SDK client unusable for the rest of the session (TD-SDKERR-1).

```python
def identify(face_embedding, threshold=0.8):
    """Return the closest verified face's record id, or None."""
    results = client.search(
        "faces", vector=face_embedding, top_k=5, metadata_filter={"verified": True}
    )
    if results and results[0].score > threshold:
        return results[0].id
    return None
```

---

## Next Steps

- [Storage Engines](../05-concepts/storage-engines.adoc) - Engine internals
- [Multi-Model Joins](./multi-model-joins.md) - Combine vectors + documents
- [Graph API](../03-api-reference/graph.adoc) - Add relationships
- [API Reference](../03-api-reference/index.md) - Complete API docs

---

*Need help?* [GitHub Issues](https://github.com/anvai-labs/proximaDB/issues)
