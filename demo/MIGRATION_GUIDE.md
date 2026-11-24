# ProximaDB Demo Migration Guide

This guide helps migrate demo code from the old SDK API to the new proto-based gRPC/REST API structure.

## Quick Reference: Old vs New API

### Client Initialization

**Old Pattern (Deprecated):**
```python
from proximadb import connect_rest, ProximaDBClient, Protocol

# REST client
client = connect_rest("http://localhost:5678")

# gRPC client
client = ProximaDBClient(
    protocol=Protocol.GRPC,
    url="http://localhost:5678",
    grpc_url="localhost:5679"
)
```

**New Pattern (Current):**
```python
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
from proximadb.protocols.rest_sync import ProximaDBSyncRestClient

# gRPC client (recommended)
client = ProximaDBSyncGrpcClient(
    "localhost:5679",
    enable_compression=False
)

# REST client
client = ProximaDBSyncRestClient("http://localhost:5678")
```

### Collection Creation

**Old Pattern:**
```python
from proximadb import CollectionConfig, DistanceMetric

config = CollectionConfig(
    name="my_collection",
    dimension=1536,
    distance_metric=DistanceMetric.COSINE
)
client.create_collection(config)
```

**New Pattern:**
```python
# gRPC
result = client.create_collection(
    name="my_collection",
    dimension=1536,
    distance_metric=1,  # 1 = cosine, 2 = euclidean, 3 = dot_product
    storage_engine=0    # 0 = auto-select
)

# REST
import requests
response = requests.post(
    "http://localhost:5678/v1/collections",
    json={
        "name": "my_collection",
        "dimension": 1536,
        "distance_metric": "cosine"  # REST uses string names
    }
)
```

### Vector Insertion

**Old Pattern:**
```python
from proximadb import VectorRecord

vectors = [
    VectorRecord(
        id="vec_1",
        values=[0.1, 0.2, ...],
        metadata={"key": "value"}
    )
]
client.insert(collection_name, vectors)
```

**New Pattern:**
```python
# gRPC - uses simple dictionaries
vectors = [
    {
        "id": "vec_1",
        "vector": [0.1, 0.2, ...],
        "metadata": {"key": "value"}
    }
]
client.insert_vectors(collection_name, vectors)

# REST
response = requests.post(
    f"http://localhost:5678/v1/collections/{collection_name}/vectors",
    json={
        "vectors": [
            {
                "id": "vec_1",
                "values": [0.1, 0.2, ...],
                "metadata": {"key": "value"}
            }
        ]
    }
)
```

### Vector Search

**Old Pattern:**
```python
from proximadb import SearchRequest

request = SearchRequest(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"category": "example"}
)
results = client.search(request)
```

**New Pattern:**
```python
# gRPC
results = client.search_vectors(
    collection_id="my_collection",
    query_vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"category": "example"}
)

# Access results (dataclass attributes)
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")

# REST
response = requests.post(
    f"http://localhost:5678/v1/collections/my_collection/search",
    json={
        "query_vector": [0.1, 0.2, ...],
        "top_k": 10,
        "filter": {"category": "example"}
    }
)
results = response.json()

# Access results (dictionary)
for result in results.get("results", results):
    print(f"ID: {result['id']}, Score: {result['score']}")
```

### Collection Info

**Old Pattern:**
```python
info = client.get_collection_info(collection_name)
print(f"Vectors: {info.vector_count}")
```

**New Pattern:**
```python
# gRPC
info = client.get_collection(collection_name)
print(f"Vectors: {info.get('vector_count', 0)}")

# REST
response = requests.get(f"http://localhost:5678/v1/collections/{collection_name}")
info = response.json()
print(f"Vectors: {info.get('vector_count', 0)}")
```

## Distance Metric Mapping

### gRPC API (uses integer codes)
- `0` = Auto-select
- `1` = Cosine
- `2` = Euclidean (L2)
- `3` = Dot Product

### REST API (uses string names)
- `"cosine"` = Cosine similarity
- `"euclidean"` = Euclidean distance
- `"dot_product"` = Dot product similarity

## Storage Engine Selection

### gRPC API (integer codes)
- `0` = Auto-select (recommended)
- `1` = SST (write-optimized)
- `2` = SST (write-optimized)
- `3` = NOVA (progressive search)
- `4` = SWIFT (low-latency)
- `5` = RAPTOR (adaptive)
- `6` = HELIX (locality-optimized)

### REST API (string names)
- `"auto"` = Auto-select
- `"sst"`, `"nova"`, `"nova"`, `"swift"`, `"raptor"`, `"helix"` = Specific engines

## Common Migration Tasks

### 1. Update Imports
```python
# Remove old imports
# from proximadb import connect_rest, ProximaDBClient, Protocol
# from proximadb import CollectionConfig, DistanceMetric, VectorRecord, SearchRequest

# Add new imports
from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient
# or
from proximadb.protocols.rest_sync import ProximaDBSyncRestClient
```

### 2. Update Client Initialization
```python
# Old
# client = connect_rest("http://localhost:5678")

# New
client = ProximaDBSyncGrpcClient("localhost:5679", enable_compression=False)
```

### 3. Update Collection Creation
```python
# Old
# config = CollectionConfig(name="test", dimension=128, distance_metric=DistanceMetric.COSINE)
# client.create_collection(config)

# New
client.create_collection(
    name="test",
    dimension=128,
    distance_metric=1,  # 1 = cosine
    storage_engine=0    # 0 = auto
)
```

### 4. Update Vector Operations
```python
# Old
# vectors = [VectorRecord(id="1", values=[...], metadata={})]
# client.insert(collection_name, vectors)

# New
vectors = [{"id": "1", "vector": [...], "metadata": {}}]
client.insert_vectors(collection_name, vectors)
```

### 5. Update Search Operations
```python
# Old
# request = SearchRequest(collection_name="test", query_vector=[...], top_k=10)
# results = client.search(request)

# New
results = client.search_vectors(
    collection_id="test",
    query_vector=[...],
    top_k=10
)
```

## Working Example: Complete Workflow

See `/tmp/complete_workflow_demo.py` for a full working example that demonstrates:
1. Client connection (gRPC)
2. Collection creation
3. Batch vector insertion
4. Vector search
5. Metrics retrieval
6. Dashboard integration

```python
#!/usr/bin/env python3
import sys
import random
sys.path.insert(0, 'clients/python/src')

from proximadb.protocols.grpc_sync import ProximaDBSyncGrpcClient

def generate_random_vector(dimension: int) -> list:
    vec = [random.gauss(0, 1) for _ in range(dimension)]
    magnitude = sum(x**2 for x in vec) ** 0.5
    return [x / magnitude for x in vec]

def main():
    # Connect
    client = ProximaDBSyncGrpcClient("localhost:5679", enable_compression=False)

    # Create collection
    client.create_collection(
        name="demo_collection",
        dimension=128,
        distance_metric=1,  # cosine
        storage_engine=0    # auto
    )

    # Insert vectors
    vectors = [
        {
            "id": f"vec_{i}",
            "vector": generate_random_vector(128)
        }
        for i in range(100)
    ]
    client.insert_vectors("demo_collection", vectors)

    # Search
    results = client.search_vectors(
        collection_id="demo_collection",
        query_vector=generate_random_vector(128),
        top_k=10
    )

    for i, result in enumerate(results[:5], 1):
        print(f"{i}. ID: {result.id}, Score: {result.score:.6f}")

    client.close()

if __name__ == "__main__":
    main()
```

## REST API Alternative: Simple Workflow

See `/tmp/simple_rest_workflow_demo.py` for a REST-only example:

```python
import requests
import random

def generate_random_vector(dimension: int) -> list:
    vec = [random.gauss(0, 1) for _ in range(dimension)]
    magnitude = sum(x**2 for x in vec) ** 0.5
    return [x / magnitude for x in vec]

base_url = "http://localhost:5678"

# Health check
response = requests.get(f"{base_url}/health")
print(f"Health: {response.json()}")

# Create collection
response = requests.post(
    f"{base_url}/v1/collections",
    json={
        "name": "rest_demo",
        "dimension": 128,
        "distance_metric": "cosine"
    }
)
print(f"Collection created: {response.json()}")

# Insert vectors
vectors = [
    {
        "id": f"vec_{i}",
        "values": generate_random_vector(128)
    }
    for i in range(100)
]

response = requests.post(
    f"{base_url}/v1/collections/rest_demo/vectors",
    json={"vectors": vectors}
)
print(f"Vectors inserted: {response.status_code}")

# Search
response = requests.post(
    f"{base_url}/v1/collections/rest_demo/search",
    json={
        "query_vector": generate_random_vector(128),
        "top_k": 10
    }
)
results = response.json()
print(f"Search results: {len(results)} found")
```

## Troubleshooting

### Import Errors
**Error**: `ModuleNotFoundError: No module named 'proximadb.protocols'`

**Solution**: Ensure you're using the correct import path and the SDK is installed:
```bash
cd clients/python
pip install -e .
```

### Connection Errors
**Error**: `grpc._channel._InactiveRpcError: ... UNAVAILABLE`

**Solution**: Ensure ProximaDB server is running:
```bash
cargo run --bin proximadb-server
# Server starts on:
# - REST: http://localhost:5678
# - gRPC: localhost:5679
```

### Proto Version Errors
**Error**: `AttributeError: 'module' object has no attribute 'VectorRecord'`

**Solution**: Don't import proto classes directly. Use dictionaries for gRPC operations:
```python
# Don't do this
# from proximadb.proximadb.v1.vector_pb2 import VectorRecord

# Do this instead
vectors = [{"id": "1", "vector": [...]}]  # Simple dict
```

### Distance Metric Confusion
**Error**: `Invalid distance_metric value`

**Solution**: Use correct format for your protocol:
- gRPC: Use integers (1 = cosine, 2 = euclidean, 3 = dot_product)
- REST: Use strings ("cosine", "euclidean", "dot_product")

## Demo Files Migration Status

### High Priority (Fixed First)
- [ ] `quickstart/basic_demo.py` - Main demo for new users
- [ ] `load_data.py` - Docker startup script
- [ ] `quickstart/unified_rest_api_demo.py` - REST API showcase
- [ ] `quickstart/feature_showcase.py` - Feature demonstration

### Medium Priority
- [ ] `progressive_search_demo.py` - Progressive quantization demo
- [ ] `showcases/quantization_showcase.py` - Quantization levels
- [ ] `showcases/engine_comparison_demo.py` - Engine performance comparison

### Lower Priority (Specialized Demos)
- [ ] `validation/engine_validation.py`
- [ ] `validation/quantization_validation.py`
- [ ] Other showcase demos

## Additional Resources

- **Working Examples**:
  - `/tmp/complete_workflow_demo.py` - Full gRPC workflow
  - `/tmp/simple_rest_workflow_demo.py` - Full REST workflow

- **API Documentation**:
  - REST API: See `/docs/api/rest_api.md`
  - gRPC API: See `proto/proximadb.proto`

- **Server Endpoints**:
  - REST API: `http://localhost:5678/v1/*`
  - gRPC API: `localhost:5679`
  - Dashboard: `http://localhost:5678/dashboard`
  - Metrics: `http://localhost:5678/metrics/json`

## Migration Checklist

For each demo file:
- [ ] Update imports to new SDK structure
- [ ] Update client initialization
- [ ] Convert CollectionConfig to direct create_collection() call
- [ ] Convert VectorRecord to dictionary format
- [ ] Convert SearchRequest to direct search_vectors() call
- [ ] Update distance metric format (integer for gRPC, string for REST)
- [ ] Test end-to-end workflow
- [ ] Verify output matches expected behavior
- [ ] Update any documentation/comments
