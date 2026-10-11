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
from proximadb_sdk import ProximaDBClient, ProximaRecord

client = ProximaDBClient(url="http://localhost:5678")

client.insert_records("products", [
    ProximaRecord(id="sku-1", vector=[0.1, 0.2, 0.3]).set_flexible("category", "Electronics")
])

# Simple search
results = client.search(
    collection="products",
    vector=[0.1, 0.2, 0.3],
    top_k=10
)

# With filter
results = client.search(
    collection="products",
    vector=[0.1, 0.2, 0.3],
    top_k=10,
    filter={"category": "Electronics", "price": {"$lt": 100}}
)
```

### REST API

```bash
curl -X POST http://localhost:5678/api/v2/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "filter": {
      "category": "Electronics"
    }
  }'
```

---

## Advanced Filtering

### Metadata Filters

```python
# Exact match
results = client.search(
    collection="products",
    vector=query_vector,
    top_k=10,
    filter={"status": "active"}
)

# Range queries
results = client.search(
    collection="products",
    vector=query_vector,
    top_k=10,
    filter={
        "price": {"$gte": 10, "$lte": 100},
        "rating": {"$gt": 4.0}
    }
)

# Logical operators
results = client.search(
    collection="products",
    vector=query_vector,
    top_k=10,
    filter={
        "$or": [
            {"category": "Electronics"},
            {"category": "Computers"}
        ],
        "stock": {"$gt": 0}
    }
)
```

### Pre-filter vs Post-filter

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart TB
    subgraph Pre["Pre-Filter (Default)"]
        A1[Filter Candidates] --> A2[Vector Search]
    end

    subgraph Post["Post-Filter"]
        B1[Vector Search] --> B2[Filter Results]
    end

    style A2 fill:#27ae60,color:#fff
    style B1 fill:#e67e22,color:#fff
```

**Pre-filter** (default): Filter before search
- ✅ Faster for selective filters
- ✅ Lower compute cost
- ❌ May miss results if filter removes nearest neighbors

**Post-filter**: Search then filter
- ✅ Guaranteed recall
- ❌ Slower for non-selective filters
- ❌ May return fewer than k results

```python
# Pre-filter (default)
results = client.search("products", vector=query_vector, top_k=10, filter={"status": "active"})

# Post-filter
results = client.search(
    "products",
    vector=query_vector,
    top_k=100,  # Search more, filter after
    filter={"status": "active"},
    filter_mode="post"
)
```

---

## Hybrid Search

### Vector + Keyword

```python
# Combine semantic and keyword search
results = client.hybrid_search(
    collection="products",
    query_vector=[...],
    query_text="wireless headphones",
    alpha=0.7,  # 0.7 vector, 0.3 keyword
    top_k=10
)
```

### Weighted Scoring

```python
# Custom scoring function
results = client.search(
    collection="products",
    vector=query_vector,
    top_k=10,
    score_function=lambda score, metadata: (
        score * 0.8 + metadata.get("popularity", 0) * 0.2
    )
)
```

---

## Batch Operations

### Bulk Insert

```python
# Efficient bulk insert through the record API
client.insert_records("products", [
    ProximaRecord(id="1", vector=v1).set_flexible("name", "A"),
    ProximaRecord(id="2", vector=v2).set_flexible("name", "B"),
])
```

### Bulk Search

```python
# Search multiple queries at once
results = client.batch_search(
    collection="products",
    query_vectors=[[...], [...], ...],
    top_k=10
)
```

---

## Performance Tuning

### Engine Selection

```python
# SST: Write-heavy, real-time
client.create_collection("events", dimension=384, engine="sst")

# HELIX: Read-optimized, spatial
client.create_collection("locations", dimension=384, engine="helix")

# VIPER: Analytics workloads
client.create_collection("analytics", dimension=384, engine="viper")
```

### Index Parameters

```python
client.create_collection(
    name="products",
    dimension=384,
    metric="cosine",
    index_params={
        "M": 16,      # HNSW connectivity
        "ef_construction": 200  # Build-time accuracy
    }
)

# Search-time accuracy
results = client.search(
    "products",
    vector=query_vector,
    top_k=10,
    search_params={"ef": 100}  # Higher = more accurate, slower
)
```

### Quantization

```python
# Reduce memory with quantization
client.create_collection(
    name="products",
    dimension=384,
    quantization={
        "type": "product",  # PQ quantization
        "bits": 8  # 8 bits per dimension
    }
)
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
curl http://localhost:5678/metrics | grep vector_search
```

---

## Best Practices

### 1. Choose the Right Metric

| Metric | When to Use |
|--------|-------------|
| **Cosine** | Text embeddings, normalized vectors |
| **L2** | Image embeddings, unnormalized |
| **Dot Product** | Pre-normalized vectors (fastest) |

### 2. Filter Before Search

```python
# Good: Pre-filter selective condition
results = client.search(
    "products",
    vector=query_vector,
    filter={"category": "Electronics"},  # Reduces search space
    top_k=10
)

# Avoid: Post-filter non-selective
results = client.search(
    "products",
    vector=query_vector,
    filter={"status": "active"},  # 99% of data
    filter_mode="post",
    top_k=10
)
```

### 3. Batch When Possible

```python
# Good: Bulk insert
client.insert_records("products", [
    ProximaRecord(id=str(id_), vector=vector, flexible_fields=props)
    for id_, vector, props in zip(ids, large_list, metadata)
])

# Avoid: Loop insert
for v, id in zip(vectors, ids):
    client.insert_records("products", [ProximaRecord(id=str(id), vector=v)])  # Slow!
```

### 4. Use Right Engine

| Workload | Engine |
|----------|--------|
| Real-time ingest | SST |
| Spatial / locality queries | HELIX |
| Columnar analytics | VIPER |
| Mixed read/write | NOVA |

> **SWIFT and RAPTOR are experimental** (off by default behind the `experimental-engines`
> cargo feature) and are not v0.2-supported — don't target them for production. For small or
> dynamic workloads use SST or NOVA. See
> [Storage Engines](../05-concepts/storage-engines.adoc).

---

## Common Patterns

### Recommendation Engine

```python
# User-based collaborative filtering
user_vector = get_user_embedding(user_id)
results = client.search(
    "products",
    vector=user_vector,
    top_k=20,
    filter={"category": {"$ne": "purchased"}}
)
```

### Deduplication

```python
# Find near-duplicates
results = client.search(
    "documents",
    vector=document_vector,
    top_k=10,
    filter={"checksum": {"$ne": doc_checksum}}
)
# If score > 0.95, likely duplicate
```

### Face Recognition

```python
# Find similar faces
results = client.search(
    "faces",
    vector=face_embedding,
    top_k=5,
    filter={"verified": True}
)
if results[0].score > 0.8:
    return results[0].metadata["user_id"]
```

---

## Next Steps

- [Storage Engines](../05-concepts/storage-engines.adoc) - Engine internals
- [Multi-Model Joins](./multi-model-joins.md) - Combine vectors + documents
- [Graph API](../03-api-reference/graph.adoc) - Add relationships
- [API Reference](../03-api-reference/index.md) - Complete API docs

---

*Need help?* [GitHub Issues](https://github.com/anvai-labs/proximaDB/issues)
