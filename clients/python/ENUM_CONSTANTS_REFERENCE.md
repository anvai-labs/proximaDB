# ProximaDB Python SDK - Enum Constants Quick Reference

This guide shows how to use readable enum constants instead of magic numbers when working with the ProximaDB Python SDK.

---

## Distance Metric Types

Use `DistanceMetricType` when creating collections or configuring search:

```python
from proximadb.models import DistanceMetricType

# ✅ Readable and self-documenting
client.create_collection(
    name="my_collection",
    dimension=1536,
    distance_metric=DistanceMetricType.COSINE  # Clear: cosine similarity
)

# ❌ Avoid magic numbers
client.create_collection(
    distance_metric=1  # What does 1 mean? Unclear!
)
```

### Available Distance Metrics

| Constant | Value | Use Case |
|----------|-------|----------|
| `DistanceMetricType.UNSPECIFIED` | 0 | Auto-select (let ProximaDB choose) |
| `DistanceMetricType.COSINE` | 1 | Text embeddings, normalized vectors (most common) |
| `DistanceMetricType.EUCLIDEAN` | 2 | Spatial data, image embeddings |
| `DistanceMetricType.DOT_PRODUCT` | 3 | Recommendation systems, ranking |
| `DistanceMetricType.MANHATTAN` | 5 | Grid-based distances, city block distance |
| `DistanceMetricType.HAMMING` | 4 | Binary vectors, error detection |
| `DistanceMetricType.JACCARD` | 6 | Set similarity, document comparison |
| `DistanceMetricType.CHEBYSHEV` | 8 | Max difference, chessboard distance |
| `DistanceMetricType.CANBERRA` | 9 | Sensitive to small changes near zero |
| `DistanceMetricType.MINKOWSKI` | 10 | Generalized distance metric |
| `DistanceMetricType.ANGULAR` | 7 | Angular distance, rotation-invariant |
| `DistanceMetricType.BRAY_CURTIS` | 11 | Ecological data, compositional data |
| `DistanceMetricType.HELLINGER` | 12 | Probability distributions |
| `DistanceMetricType.CUSTOM` | 13 | Custom distance function |

---

## Index Types

Use `IndexType` when configuring indexing algorithms:

```python
from proximadb.models import IndexType
from proximadb.v1 import collection_types_pb2 as v1_collection

# ✅ Create index with readable constant
index_config = v1_collection.IndexConfig(
    index_name="my_primary_index",
    algorithm=IndexType.HNSW,  # Clear: using HNSW algorithm
    is_primary=True
)

# ❌ Avoid magic numbers
index_config = v1_collection.IndexConfig(
    algorithm=1  # What algorithm is 1? Unclear!
)
```

### Available Index Types

| Constant | Value | Description | Best For |
|----------|-------|-------------|----------|
| `IndexType.UNSPECIFIED` | 0 | Auto-select index | Let ProximaDB choose optimal index |
| `IndexType.HNSW` | 1 | Hierarchical Navigable Small World | High recall, fast approximate search |
| `IndexType.IVF` | 2 | Inverted File Index | Large-scale datasets, memory efficiency |
| `IndexType.PQ` | 3 | Product Quantization | Memory-constrained environments |
| `IndexType.FLAT` | 4 | Exact search (no index) | Small datasets, exact results required |
| `IndexType.ANNOY` | 5 | Approximate Nearest Neighbors Oh Yeah | Static datasets, read-heavy workloads |
| `IndexType.LSH` | 6 | Locality Sensitive Hashing | High-dimensional sparse vectors |

---

## Storage Engine Types

Use `StorageEngineType` when creating collections:

```python
from proximadb.models import StorageEngineType

# ✅ Explicit engine selection
client.create_collection(
    name="analytics_collection",
    dimension=768,
    storage_engine=StorageEngineType.VIPER  # Clear: using VIPER for analytics
)

# ✅ Let ProximaDB auto-select best engine
client.create_collection(
    name="auto_collection",
    dimension=768,
    storage_engine=StorageEngineType.UNSPECIFIED  # Auto-select
)

# ❌ Avoid magic numbers
client.create_collection(
    storage_engine=1  # What engine is 1? Unclear!
)
```

### Available Storage Engines

| Constant | Value | Engine | Best For |
|----------|-------|--------|----------|
| `StorageEngineType.UNSPECIFIED` | 0 | Auto-select | Let ProximaDB choose optimal engine |
| `StorageEngineType.VIPER` | 1 | VIPER | Analytics, batch operations, compression |
| `StorageEngineType.SST` | 2 | SST | Real-time queries, frequent updates |
| `StorageEngineType.NOVA` | 3 | NOVA | Mixed workloads, progressive search |
| `StorageEngineType.HELIX` | 4 | HELIX | Spatial locality, range queries |
| `StorageEngineType.SWIFT` | 5 | SWIFT | Low-latency operations |
| `StorageEngineType.RAPTOR` | 6 | RAPTOR | Dynamic workloads, adaptive optimization |

---

## Complete Examples

### Example 1: Create Collection for Text Embeddings

```python
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
from proximadb.models import DistanceMetricType, StorageEngineType

client = ProximaDBSyncGrpcClient("localhost:5679")

# Create collection optimized for text embeddings
collection_id = client.create_collection(
    name="text_embeddings",
    dimension=1536,  # OpenAI ada-002 dimension
    distance_metric=DistanceMetricType.COSINE,  # Best for normalized embeddings
    storage_engine=StorageEngineType.UNSPECIFIED  # Auto-select
)
```

### Example 2: Create Collection with HNSW Index

```python
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
from proximadb.models import DistanceMetricType, IndexType, StorageEngineType
from proximadb.v1 import collection_types_pb2 as v1_collection

client = ProximaDBSyncGrpcClient("localhost:5679")

# Create HNSW index configuration
hnsw_index = v1_collection.IndexConfig(
    index_name="primary_hnsw",
    algorithm=IndexType.HNSW,
    is_primary=True
)

# Create collection with explicit index
collection_id = client.create_collection(
    name="fast_search_collection",
    dimension=768,
    distance_metric=DistanceMetricType.COSINE,
    storage_engine=StorageEngineType.VIPER,
    index_configs=[hnsw_index]
)
```

### Example 3: Analytics Workload

```python
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
from proximadb.models import DistanceMetricType, StorageEngineType

client = ProximaDBSyncGrpcClient("localhost:5679")

# Create collection optimized for analytics
collection_id = client.create_collection(
    name="analytics_vectors",
    dimension=512,
    distance_metric=DistanceMetricType.EUCLIDEAN,
    storage_engine=StorageEngineType.VIPER  # Columnar format for analytics
)
```

### Example 4: Low-Latency Real-Time Search

```python
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
from proximadb.models import DistanceMetricType, StorageEngineType

client = ProximaDBSyncGrpcClient("localhost:5679")

# Create collection optimized for low latency
collection_id = client.create_collection(
    name="realtime_search",
    dimension=384,
    distance_metric=DistanceMetricType.COSINE,
    storage_engine=StorageEngineType.SWIFT  # Low-latency row-based storage
)
```

---

## Migration Guide

### Old Code (Magic Numbers)

```python
# ❌ Hard to understand and maintain
client.create_collection(
    name="my_collection",
    dimension=1536,
    distance_metric=1,  # What metric?
    storage_engine=1    # What engine?
)

index_config = IndexConfig(
    algorithm=1,  # What algorithm?
    is_primary=True
)
```

### New Code (Readable Constants)

```python
# ✅ Self-documenting and type-safe
from proximadb.models import DistanceMetricType, IndexType, StorageEngineType

client.create_collection(
    name="my_collection",
    dimension=1536,
    distance_metric=DistanceMetricType.COSINE,  # Clear!
    storage_engine=StorageEngineType.VIPER      # Clear!
)

index_config = IndexConfig(
    algorithm=IndexType.HNSW,  # Clear!
    is_primary=True
)
```

---

## IDE Autocomplete

These enum constants provide excellent IDE support:

```python
from proximadb.models import DistanceMetricType

# Type "DistanceMetricType." and your IDE will show all options:
# - UNSPECIFIED
# - COSINE
# - EUCLIDEAN
# - DOT_PRODUCT
# ... etc.

metric = DistanceMetricType.  # <-- IDE autocomplete kicks in here!
```

---

## Type Safety

The integer-based enums provide type safety while maintaining gRPC compatibility:

```python
from proximadb.models import DistanceMetricType

# These are actual integers
assert DistanceMetricType.COSINE == 1  # True
assert isinstance(DistanceMetricType.COSINE, int)  # True

# But with IDE support and type checking
def create_collection(distance_metric: DistanceMetricType):
    # mypy will catch errors here
    pass

create_collection(DistanceMetricType.COSINE)  # ✅ Works
create_collection(1)  # ⚠️ Type checker warning (but still works)
create_collection(999)  # ⚠️ Type checker error
```

---

## Best Practices

1. **Always use enum constants** instead of magic numbers
2. **Import explicitly** from `proximadb.models`
3. **Use UNSPECIFIED** when you want ProximaDB to auto-select
4. **Document your choice** if using a specific distance metric or engine
5. **Test different configurations** to find optimal performance

---

## Common Patterns

### Pattern 1: Auto-Select Everything

```python
from proximadb.models import DistanceMetricType, StorageEngineType

# Let ProximaDB choose optimal settings
client.create_collection(
    name="auto_collection",
    dimension=768,
    distance_metric=DistanceMetricType.UNSPECIFIED,
    storage_engine=StorageEngineType.UNSPECIFIED
)
```

### Pattern 2: Explicit Configuration

```python
from proximadb.models import DistanceMetricType, IndexType, StorageEngineType
from proximadb.v1 import collection_types_pb2 as v1_collection

# Explicit control over all settings
hnsw_index = v1_collection.IndexConfig(
    index_name="primary",
    algorithm=IndexType.HNSW,
    is_primary=True
)

client.create_collection(
    name="explicit_collection",
    dimension=1536,
    distance_metric=DistanceMetricType.COSINE,
    storage_engine=StorageEngineType.VIPER,
    index_configs=[hnsw_index]
)
```

### Pattern 3: Conditional Engine Selection

```python
from proximadb.models import StorageEngineType

# Choose engine based on workload
workload = "analytics"  # or "realtime"

engine = (
    StorageEngineType.VIPER if workload == "analytics"
    else StorageEngineType.SWIFT
)

client.create_collection(
    name=f"{workload}_collection",
    dimension=768,
    storage_engine=engine
)
```

---

## Troubleshooting

### Issue: "TypeError: expected int, got DistanceMetricType"

**Solution:** The enum constants ARE integers, this shouldn't happen. Check your proto imports:

```python
# ✅ Correct v1 proto import
from proximadb.v1 import collection_types_pb2

# ❌ Old proto import (deleted)
from proximadb import proximadb_pb2  # This will fail
```

### Issue: "ImportError: cannot import name 'DistanceMetricType'"

**Solution:** Make sure you're importing from the correct module:

```python
# ✅ Correct import
from proximadb.models import DistanceMetricType

# ❌ Wrong module
from proximadb import DistanceMetricType  # Not exported at top level
```

### Issue: IDE not showing autocomplete

**Solution:** Make sure you have the latest SDK version and type stubs installed:

```bash
pip install --upgrade proximadb
# or for development:
cd clients/python && pip install -e .
```

---

## Additional Resources

- **Main Documentation:** See `MIGRATION_V1_PROTO_SUMMARY.md` for complete migration guide
- **Test Examples:** See `tests/unit/test_grpc_sync_collections.py` for usage examples
- **Demo Application:** See `examples/complete_workflow_demo.py` for end-to-end example

---

**Last Updated:** October 2025
**SDK Version:** 0.2.0+
**Status:** Production Ready ✅
