# ProximaDB Benchmarks

Comprehensive benchmark suite for ProximaDB performance validation and competitor comparison.

## Overview

This benchmark suite provides:

1. **ANN-Benchmarks Integration** - Standardized benchmarks following the ANN-benchmarks methodology
2. **Competitor Comparisons** - via the vendored VectorDBBench (see below; the old fabricated harness is removed)
3. **Storage Engine Benchmarks** - Performance tests for all 6 ProximaDB storage engines
4. **Index Type Comparisons** - HNSW, IVF, Annoy, Flat, LSH, PQ, DiskANN

## Quick Start

### Prerequisites

```bash
# Install Rust (1.88+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install ProximaDB
cargo install --path .

# Download benchmark datasets
mkdir -p /data/ann_benchmarks
cd /data/ann_benchmarks

# SIFT (1M vectors, 128D, L2)
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5

# GIST (1M vectors, 960D, L2)
wget http://ann-benchmarks.com/gist-960-euclidean.hdf5

# MNIST (60K vectors, 784D, L2)
wget http://ann-benchmarks.com/mnist-784-euclidean.hdf5
```

### Running ANN-Benchmarks

```bash
# HNSW on SIFT dataset
cargo run -p proximadb-ann-bench -- \
  --dataset sift \
  --algorithm hnsw \
  --m 16 \
  --ef-construction 200 \
  --ef-search 100 \
  --runs 1000

# IVF on GIST dataset
cargo run -p proximadb-ann-bench -- \
  --dataset gist \
  --algorithm ivf \
  --nlist 100 \
  --nprobe 10 \
  --runs 100

# DiskANN on DEEP1B dataset
cargo run -p proximadb-ann-bench -- \
  --dataset deep1b \
  --algorithm diskann \
  --r 32 \
  --l 50 \
  --runs 100
```

### Competitor comparisons

The former `proximadb-vectordb-compare` harness has been **removed**. Despite
its CLI accepting `--competitors qdrant,weaviate,milvus,pinecone` and this
README instructing readers to start those databases in Docker, the harness
never connected to any of them — every competitor function returned
hardcoded "representative results" (fixed QPS/recall/build-time literals).
A comparison harness that fabricates its comparison is worse than none: it
manufactures confidence exactly where measurement is supposed to live, and
it is part of how an 8× vector-read gap to LanceDB went unnoticed until an
external consumer measured it (#1690).

For real competitor comparisons use upstream
[VectorDBBench](https://github.com/zilliztech/VectorDBBench) (real clients,
including LanceDB; writing a ProximaDB client for it is the missing piece —
an earlier vendored copy was removed from this tree years ago), and register
every resulting claim in `BENCHMARK_EVIDENCE.toml` with the exact command and
environment, per the evidence-ledger rules.

## Results

### ANN-Benchmarks Results

Results are automatically exported to CSV and JSON:

- `results_<dataset>_<algorithm>.csv` - CSV format for analysis
- `results_<dataset>_<algorithm>.json` - JSON format for programmatic use

Example CSV output:

```csv
dataset,algorithm,k,build_time_secs,index_size_bytes,memory_usage_bytes,avg_qps,median_qps,p95_qps,p99_qps,recall_at_k,avg_latency_ms,median_latency_ms,p95_latency_ms,p99_latency_ms
sift,hnsw,10,142.5,1034837382,2147483648,5234,5757,4187,3140,0.948,0.191,0.174,0.239,0.319
```

### Performance Documentation

See `docs/benchmarks/PERFORMANCE.md` for:
- Detailed benchmark results
- Comparison with competitors
- Hardware specifications
- Methodology and reproducibility

## Benchmark Datasets

### Standard Datasets

| Dataset | Vectors | Dimensions | Metric | Download |
|---------|---------|------------|--------|----------|
| SIFT | 1,000,000 | 128 | L2 | http://ann-benchmarks.com/sift-128-euclidean.hdf5 |
| GIST | 1,000,000 | 960 | L2 | http://ann-benchmarks.com/gist-960-euclidean.hdf5 |
| MNIST | 60,000 | 784 | L2 | http://ann-benchmarks.com/mnist-784-euclidean.hdf5 |
| DEEP1B | 1,000,000,000 | 96 | Angular | Contact ANN-benchmarks |

### Synthetic Datasets

For quick testing, synthetic datasets can be generated:

```bash
# Generate 100K random vectors (128D)
python scripts/generate_vectors.py --count 100000 --dimensions 128 --output /data/test_100k.hdf5
```

## Algorithms

### Supported Algorithms

| Algorithm | Description | Parameters |
|-----------|-------------|------------|
| HNSW | Hierarchical Navigable Small World | M, ef_construction, ef_search |
| IVF | Inverted File Index | nlist, nprobe |
| Annoy | Approximate Nearest Neighbors Oh Yeah | n_trees, search_k |
| Flat | Exact search (brute force) | None |
| LSH | Locality Sensitive Hashing | n_bits, n_probes |
| PQ | Product Quantization | n_subvectors |
| DiskANN | Graph-based SSD-optimized index | R, L |

### Parameter Tuning

#### HNSW Parameters

```bash
# High recall (95%+)
--m 32 --ef-construction 400 --ef-search 200

# Balanced speed/recall (90%)
--m 16 --ef-construction 200 --ef-search 100

# High speed (80% recall)
--m 12 --ef-construction 100 --ef-search 50
```

#### IVF Parameters

```bash
# High recall
--nlist 1000 --nprobe 100

# Balanced
--nlist 100 --nprobe 10

# High speed
--nlist 50 --nprobe 5
```

## Storage Engine Benchmarks

To benchmark individual storage engines:

```bash
# SST engine
cargo test --bench sst_engine -- --nocapture


# All engines
cargo test --benches -- --nocapture
```

## Reproducibility

To reproduce published benchmark results:

```bash
# Use exact same parameters
cargo run -p proximadb-ann-bench -- \
  --dataset sift \
  --algorithm hnsw \
  --m 16 \
  --ef-construction 200 \
  --ef-search 100 \
  --runs 1000 \
  --seed 42  # Fixed random seed
```

Results should match within ±5% variance due to:
- System load
- CPU frequency scaling
- Cache effects
- Background processes

## Continuous Benchmarking

For CI/CD integration:

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, benchmark]
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          cargo run -p proximadb-ann-bench -- --dataset sift --algorithm hnsw
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results_*.json
```

## Contributing

When adding new benchmarks:

1. Follow ANN-benchmarks methodology
2. Document hardware specifications
3. Provide reproducibility instructions
4. Include comparison with baseline
5. Submit to ANN-benchmarks leaderboard

## References

- ANN-Benchmarks: https://ann-benchmarks.com
- VectorDBBench: https://github.com/qdrant/vector-db-benchmark
- HNSW Paper: https://arxiv.org/abs/1603.09320
- DiskANN Paper: https://arxiv.org/abs/1905.02235

## License

Apache-2.0

## Contact

For questions or issues:
- GitHub: https://github.com/anvai-labs/proximaDB/issues
- Email: vijay@anvaiops.com
