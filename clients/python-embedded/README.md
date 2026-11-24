# ProximaDB Embedded

**In-process ProximaDB bindings for Python**

ProximaDB Embedded provides direct in-process access to ProximaDB's Rust services without opening REST, gRPC, Arrow Flight, or PostgreSQL wire ports. It is the local/embedded entry point for applications that need fast vector, document, graph, SQL/UQL, and observability operations in the same process.

## Features

- **No Local Port Required**: Direct in-process calls into shared Rust services and facades
- **Multi-Disk Support**: Configure multiple storage locations with weighted distribution
- **SIMD Acceleration**: Automatic AVX2/NEON vector operation optimization
- **Full Persistence**: Write-ahead logging with configurable sync modes
- **NumPy Integration**: Fast NumPy ingestion paths that avoid Python list conversion
- **Context Manager**: Automatic resource cleanup

## Embedded vs Remote Protocols

Embedded mode should not start a local server just to call ProximaDB from Python. The PyO3 module owns an in-process database instance and routes operations directly to the same Rust collection, vector, catalog, document, graph, observability, SQL, and unified-query services used by server mode.

Use remote protocols when another process or host needs access:

- **Arrow Flight**: high-throughput remote columnar/bulk exchange.
- **gRPC**: low-latency remote lifecycle and vector operations.
- **REST**: human-friendly and operational HTTP workflows.
- **PostgreSQL wire**: SQL clients, BI tools, SQLAlchemy/JDBC/ODBC compatibility.

For embedded Python benchmarks and local agent runtimes, prefer direct PyO3 calls. Use gRPC or Arrow Flight only when benchmarking the network/protocol surface itself.

## Installation

```bash
pip install proximadb_embedded
```

### Build from Source

Requires Rust toolchain and maturin:

```bash
# Install maturin
/Users/vijaysingh/code/.venv/bin/python -m pip install maturin

# Build and install
cd clients/python-embedded
# Build from the REPO ROOT — its pyproject.toml points maturin at the
# binding crate (crates/binding/proximadb-embedded). The old command here
# passed `--features python,pylib`; `pylib` no longer exists and the build
# could never work (#1675).
cd ../.. && maturin develop --release --features python
```

Canonical import:

```python
import proximadb_embedded
```

## Quick Start

```python
import numpy as np
from proximadb_embedded import ProximaDB, DiskConfig

# Simple single-disk setup
db = ProximaDB(data_dirs="./my_database")

# Or multi-disk with weighted distribution
db = ProximaDB(
    data_dirs=[
        DiskConfig("/nvme/data", weight=2),  # Fast SSD - gets 2x data
        DiskConfig("/hdd/data", weight=1),   # Slower HDD
    ],
    metadata_dir="/nvme/metadata",
    cache_size_mb=2048,
)

# Create a collection
db.create_collection("embeddings", dimension=768, engine="sst")

# Insert vectors with NumPy
vectors = np.random.rand(10000, 768).astype(np.float32)
ids = [f"doc_{i}" for i in range(10000)]
metadata = [{"category": "A" if i % 2 == 0 else "B"} for i in range(10000)]

db.insert("embeddings", ids=ids, vectors=vectors, metadata=metadata)

# Search
query = np.random.rand(768).astype(np.float32)
results = db.search("embeddings", query=query, top_k=10)

for r in results:
    print(f"{r.id}: score={r.score:.4f}")

# Flush to ensure durability
db.flush()
```

## API Reference

### ProximaDB

Main database class for embedded operations.

```python
ProximaDB(
    data_dirs=None,           # Path or list of DiskConfig
    metadata_dir=None,        # Metadata storage path
    cache_size_mb=512,        # Cache size in MB
    default_engine="sst",     # Storage engine type
    enable_wal=True,          # Enable write-ahead logging
    wal_sync_mode="batch",    # WAL sync: "immediate", "batch", "async"
)
```

**Methods:**

- `create_collection(name, dimension, engine=None)` - Create a new collection
- `delete_collection(name)` - Delete a collection
- `get_collection(name)` - Get collection info
- `list_collections()` - List all collections
- `insert(collection, ids, vectors, metadata=None)` - Insert vectors
- `search(collection, query, top_k=10, filter=None)` - Search for similar vectors
- `flush()` - Flush pending writes to disk
- `stats()` - Get storage statistics

### DiskConfig

Configuration for multi-disk storage.

```python
DiskConfig(
    path="/data/proximadb",  # Storage directory path
    weight=1,                # Weight for data distribution
    tags=["hot", "ssd"],     # Optional tags
)
```

### Storage Engines

ProximaDB supports multiple storage engines optimized for different workloads:

| Engine | Best For | Characteristics |
|--------|----------|-----------------|
| `sst` | OLTP, real-time | Hybrid columnar, fast queries (default) |
| `nova` | Mixed workloads, analytics | Hybrid quantized, columnar |
| `swift` | Hot data | Hierarchical blocks, low latency |
| `raptor` | Adaptive | Matrix-optimized, workload learning |
| `helix` | High-dimensional | PCA + Hilbert clustering |

## Multi-Disk Configuration

ProximaDB can distribute data across multiple disks with weighted allocation:

```python
from proximadb_embedded import ProximaDB, DiskConfig

# Configure disks with different weights
disks = [
    DiskConfig("/nvme1/data", weight=3, tags=["hot", "ssd"]),
    DiskConfig("/nvme2/data", weight=3, tags=["hot", "ssd"]),
    DiskConfig("/hdd1/data", weight=1, tags=["cold", "hdd"]),
]

db = ProximaDB(
    data_dirs=disks,
    metadata_dir="/nvme1/metadata",  # Metadata on fast disk
)
```

Data is distributed based on weights:
- NVMe disks (weight=3 each) get 75% of data
- HDD (weight=1) gets 25% of data

## Context Manager

ProximaDB supports context manager for automatic cleanup:

```python
with ProximaDB(data_dirs="./data") as db:
    db.create_collection("test", dimension=128)
    # ... operations ...
# Automatically flushed on exit
```

## Performance Tips

1. **Use NumPy arrays**: Pass `np.float32` arrays directly for zero-copy transfer
2. **Batch inserts**: Insert vectors in batches of 1000-10000 for best throughput
3. **Choose the right engine**: Use `sst` for real-time, `nova` for analytics/mixed workloads (`viper` was REMOVED — ADR-093)
4. **Enable batch WAL**: Use `wal_sync_mode="batch"` for better write throughput
5. **Size cache appropriately**: Larger cache = better read performance

## Comparison with Client SDK

| Feature | Embedded | Client SDK |
|---------|----------|------------|
| Hot path | Direct PyO3 service calls | REST/gRPC/Arrow Flight/pgwire |
| Network overhead | None | Protocol dependent |
| Deployment | In-process | Separate server |
| Scaling | Single node | Multi-node |
| Use case | Local apps, agents, notebooks, embedded runtimes | Distributed systems and external clients |

## License

Apache License 2.0

## Links

- [ProximaDB Repository](https://github.com/anvai-labs/proximaDB)
- [Documentation](https://github.com/anvai-labs/proximaDB#readme)
- [Issues](https://github.com/anvai-labs/proximaDB/issues)
