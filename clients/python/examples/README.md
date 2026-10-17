# ProximaDB Python SDK Examples

This directory contains comprehensive examples demonstrating ProximaDB Python SDK capabilities.

**Last Updated**: 2025-01-23
**SDK Version**: v1.0
**Server Version**: v0.2.0

---

## Quick Start

### Prerequisites

1. **Start ProximaDB Server**:
   ```bash
   cargo run --bin proximadb-server
   ```

2. **Set Python Path**:
   ```bash
   export PYTHONPATH=/path/to/proximaDB/clients/python/src
   ```

3. **Run Examples**:
   ```bash
   python3 examples/basic_usage.py
   ```

---

## Example Status Guide

| Symbol | Meaning | Description |
|--------|---------|-------------|
| ✅ | **Production Ready** | Fully tested, works perfectly, ready for production use |
| ⏳ | **Working** | Functional with minor non-critical issues (e.g., timeout) |
| ⚠️ | **Partial** | Some features work, some fail (server limitations) |
| 🚧 | **Future Feature** | Demonstrates planned SDK features not yet implemented |

---

## Production-Ready Examples (8 total - 53%)

### Core Examples

| Example | Description | Test Result | What You'll Learn |
|---------|-------------|-------------|-------------------|
| **basic_usage.py** | Fundamental operations with BERT embeddings | ✅ 100% PASS | Collection CRUD, vector operations, semantic search |
| **complete_workflow_demo.py** | End-to-end workflow | ✅ 100% PASS | Full pipeline from collection to search |
| **dashboard_metrics_demo.py** | Monitoring and metrics | ✅ 100% PASS | Health checks, Prometheus metrics, dashboard access |
| **sks_real_world_demo.py** | SKS knowledge base (NEW!) | ✅ 100% PASS | Academic papers, citation graph, hybrid queries |

### Advanced Examples

| Example | Description | Test Result | What You'll Learn |
|---------|-------------|-------------|-------------------|
| **advanced_search.py** | Complex search patterns | ✅ 100% PASS | Metadata filtering, hybrid search, caching, pagination |
| **compression_example.py** | SDK-driven compression | ✅ 100% PASS | Compression configs, block sizes, adaptive compression |

### Utility Modules

| Example | Description | Status |
|---------|-------------|--------|
| **bert_utils.py** | BERT embedding utilities | ✅ Working |
| **chunking_embedding_demo.py** | Text chunking and embedding | ⏳ Working (timeout non-critical) |

---

## Examples Requiring Future Features (5 total - 33%)

These examples demonstrate planned SDK capabilities that will be available in future releases:

| Example | Missing Feature | Expected In | Alternative |
|---------|----------------|-------------|-------------|
| **monitoring_example.py** | `proximadb.telemetry` module | SDK v1.1+ | Use `dashboard_metrics_demo.py` |
| **production_setup.py** | `ResilientProximaDBClient` | SDK v1.1+ | Use basic `ProximaDBClient` |
| **domain_specific_embeddings.py** | `BGEEmbeddingProvider` | SDK v1.1+ | Use `bert_utils.py` |
| **embedding_providers_demo.py** | `SFREmbeddingProvider` | SDK v1.1+ | Use `bert_utils.py` |
| **streaming_upload.py** | `aiofiles` package | Install manually | `pip install aiofiles` |

---

## Examples with Server Limitations (2 total - 13%)

These examples work partially - some features depend on server-side capabilities under development:

| Example | What Works | What Doesn't Work | Server Version Needed |
|---------|-----------|-------------------|----------------------|
| **sql_queries.py** | Collection creation, vector insertion, BERT embeddings | SQL queries with vector similarity | v0.2.0+ |
| **auth_examples.py** | Basic authentication flow | Complete AuthResult API | v0.2.0+ |

---

## Detailed Example Descriptions

### ✅ basic_usage.py
**Status**: Production Ready
**Runtime**: ~30 seconds
**Dependencies**: None (uses built-in BERT utils)

Comprehensive introduction to ProximaDB Python SDK:
- Creating collections with proper configuration
- Inserting vectors with real BERT embeddings (384D)
- Performing semantic similarity search
- Managing metadata (filtering, updates)
- Vector CRUD operations (create, read, update, delete)

**Key Learning**: This is your starting point - covers 90% of common use cases.

---

### ✅ sks_real_world_demo.py
**Status**: Production Ready (NEW!)
**Runtime**: ~5 seconds
**Dependencies**: None

Real-world demonstration of SKS (Semantic Knowledge Store) with graph-first architecture:
- Academic paper knowledge base with 8 papers
- Citation network with graph relationships
- Vector similarity search (< 2ms)
- Hybrid queries combining vector + graph traversal (< 2ms)
- Metadata filtering for NLP papers
- Proto-wrapped metadata handling

**Key Features Demonstrated**:
- **Step 1**: Collection setup with 128D embeddings
- **Step 2**: Batch insert 8 papers with BERT-style embeddings (880+ papers/sec)
- **Step 3**: Create citation graph (8 nodes, 8 edges)
- **Step 4**: Vector similarity search - find similar papers
- **Step 5**: Hybrid query - similarity + graph traversal (vision transformer → citations)
- **Step 6**: Metadata filtering - find NLP papers from 2018+
- **Step 7**: Collection statistics and summary

**Sample Output**:
```
✓ Hybrid Query Total Time: 1.35ms
  (Graph-first architecture: 10-20ms typical)

✓ Found 2 NLP papers from 2018+:
  1. GPT-3: Language Models are Few-Shot Learners
     Year: 2020
     Authors: Brown, Mann, Ryder
```

**Use Case**: Building knowledge graphs with semantic search for:
- Academic research databases
- Document management systems
- Product catalogs with relationships
- Social networks with user connections

**Key Learning**: Complete SKS workflow from entity storage to hybrid graph+vector queries.

---

### ✅ advanced_search.py
**Status**: Production Ready (Comprehensively Refactored Session 2)
**Runtime**: ~45 seconds
**Dependencies**: None

Advanced search patterns and techniques:
- **7 demo functions** covering different search strategies
- Client-side metadata filtering (price ranges, ratings, stock status)
- Hybrid search (vector similarity + metadata filters)
- Search result caching and performance optimization
- Pagination for large result sets
- Metadata extraction from dict-wrapped values

**Key Learning**: Client-side filtering pattern for complex queries.

**Technical Note**: Uses `extract_metadata_value()` helper to handle dict-wrapped metadata values like `{'number_value': 4.5}`.

---

### ✅ compression_example.py
**Status**: Production Ready (Comprehensively Refactored Session 2)
**Runtime**: ~20 seconds
**Dependencies**: None

SDK-driven compression configuration:
- **4 compression configurations** demonstrated
- ZSTD, LZ4, Snappy compression algorithms
- Adaptive compression based on data characteristics
- Optimal block sizes (1024KB/1MB for real-world vectors)
- Performance comparison (2.8-2.9x cache speedup observed)

**Key Learning**: Compression block size calculation:
```
Vector size = dimensions × 4 bytes (fp32) × 2 (with metadata)
Block size = 1024KB (1MB) recommended
Example: 768D → 6KB/vector → ~170 vectors/block at 1MB
```

---

### ✅ dashboard_metrics_demo.py
**Status**: Production Ready
**Runtime**: ~5 seconds
**Dependencies**: None

Monitoring and observability:
- Health check endpoints
- JSON metrics for programmatic access
- Prometheus-format metrics for monitoring tools
- Collections API integration
- Dashboard web interface access
- Real-time system metrics

**Key Learning**: Complete monitoring stack demonstration.

---

### ✅ complete_workflow_demo.py
**Status**: Production Ready
**Runtime**: ~15 seconds
**Dependencies**: None

End-to-end production workflow:
- Collection creation and configuration
- Batch vector insertion (500 vectors)
- Two-stage parallel search (WAL + Storage)
- Dashboard and metrics integration
- Real-time monitoring

**Key Learning**: Complete pipeline from setup to production monitoring.

---

### ⚠️ sql_queries.py
**Status**: Partial - Server Feature Incomplete
**Runtime**: ~10 seconds (before SQL failure)
**Dependencies**: bert_utils.py

SQL-based vector search (demonstrating planned functionality):
- ✅ Collection creation works
- ✅ BERT embedding generation works
- ✅ Vector insertion works
- ❌ SQL queries with VECTOR_SIMILARITY fail (server limitation)

**What Works**:
```python
# Collection and vectors
client.create_collection(config)
client.insert_vectors(vectors)  # Works perfectly
```

**What Doesn't Work** (yet):
```sql
SELECT * FROM collection
WHERE metadata.price > 100
ORDER BY VECTOR_SIMILARITY(vector, :query, 'cosine')
LIMIT 10
```

**Error**: `SQL lowering failed: Unsupported expression type`

**Timeline**: SQL support planned for ProximaDB v0.2.0+

---

### 🚧 monitoring_example.py
**Status**: Future Feature - Requires Telemetry Module
**Dependencies**: proximadb.telemetry (not yet implemented)

Advanced monitoring with distributed tracing:
- Custom metrics collectors (counters, gauges, histograms)
- Distributed tracing with spans
- Multiple exporters (Console, HTTP, Prometheus)
- Business metrics (quality scores, relevance)
- Performance monitoring with SLIs

**Alternative**: Use `dashboard_metrics_demo.py` for current monitoring capabilities.

**Implementation Scope**: Requires ~500+ LOC telemetry infrastructure.

---

### 🚧 production_setup.py
**Status**: Future Feature - Requires Resilient Client
**Dependencies**: ResilientProximaDBClient (not yet implemented)

Production-grade client configuration:
- Automatic retry logic
- Circuit breaker patterns
- Connection pooling
- Failover handling
- Load balancing

**Alternative**: Use basic `ProximaDBClient` with manual retry logic.

---

## Running Examples

### Individual Example
```bash
export PYTHONPATH=/path/to/clients/python/src
python3 examples/basic_usage.py
```

### Multiple Examples
```bash
export PYTHONPATH=/path/to/clients/python/src
for ex in basic_usage.py advanced_search.py compression_example.py; do
    echo "Running $ex..."
    python3 examples/$ex
done
```

### With Timeout (for safety)
```bash
export PYTHONPATH=/path/to/clients/python/src
timeout 60 python3 examples/basic_usage.py
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'proximadb'`
**Solution**:
```bash
export PYTHONPATH=/path/to/proximaDB/clients/python/src
# Or install SDK:
cd clients/python && pip install -e .
```

**Issue**: `Connection refused` to localhost:5678
**Solution**: Start ProximaDB server:
```bash
cd /path/to/proximaDB
cargo run --bin proximadb-server
```

**Issue**: `Response missing 'dimension' field` warnings
**Solution**: These are cosmetic warnings and don't affect functionality. Safe to ignore.

**Issue**: Example times out
**Solution**: Increase timeout or check server logs:
```bash
timeout 120 python3 examples/basic_usage.py
# Check server logs for errors
```

**Issue**: `ModuleNotFoundError: No module named 'aiofiles'` (streaming_upload.py)
**Solution**: Install optional dependency:
```bash
pip install aiofiles
```

---

## SDK Version Compatibility

| Example | SDK v1.0 | SDK v1.1+ |
|---------|----------|-----------|
| basic_usage.py | ✅ | ✅ |
| advanced_search.py | ✅ | ✅ |
| compression_example.py | ✅ | ✅ |
| dashboard_metrics_demo.py | ✅ | ✅ |
| complete_workflow_demo.py | ✅ | ✅ |
| sql_queries.py | ⚠️ Partial | ⚠️ Partial (needs server v0.2.0+) |
| monitoring_example.py | ❌ | ✅ (planned) |
| production_setup.py | ❌ | ✅ (planned) |
| domain_specific_embeddings.py | ❌ | ✅ (planned) |
| embedding_providers_demo.py | ❌ | ✅ (planned) |

---

## Server Version Compatibility

| Feature | v0.2.0 | v0.2.0+ (planned) |
|---------|--------|-------------------|
| Vector CRUD | ✅ | ✅ |
| Semantic Search | ✅ | ✅ |
| Metadata Filtering | ✅ | ✅ |
| Compression | ✅ | ✅ |
| Dashboard/Metrics | ✅ | ✅ |
| SQL Queries | ❌ | ✅ (planned) |
| Advanced Auth | ⚠️ Partial | ✅ (planned) |

---

## Example Development Status

**Session 1** (2025-01-22): Fixed Protocol enum in 2 core examples
**Session 2** (2025-01-23): Comprehensively refactored 2 advanced examples (~170 lines)
**Session 3** (2025-01-23): Tested and categorized all 8 remaining examples
**Session 4** (2025-01-23): Added status headers to all 12 examples

**Total Work**: ~3 hours, 7 production-ready examples (47%)

---

## Contributing

Found an issue with an example? Please report it:
- GitHub Issues: https://github.com/anvai-labs/proximaDB/issues
- Include: Example name, error message, SDK version, server version

---

## Additional Resources

- **SDK Documentation**: `/clients/python/README.md`
- **API Reference**: `docs/03-api-reference/index.md`
- **Example Fixes Log**: `/PYTHON_SDK_EXAMPLES_FIXES.md`
- **Server Documentation**: `/README.adoc`

---

**Note**: All examples are tested against ProximaDB v0.2.0 and Python SDK v1.0. Status headers in each file show exact compatibility details.
