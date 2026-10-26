# ProximaDB Feature Toggles

**Version**: 0.2.0  
**Last Updated**: 2026-05-01  
**Purpose**: Document all feature flags, their purposes, and production readiness criteria

## Overview

ProximaDB uses feature flags to control compilation of different components, optimize binary size, and separate stable from experimental functionality. This document provides a comprehensive reference for all available feature flags.

## Feature Flag Categories

### 1. Default Features (Always On)

These features are enabled by default and represent the core ProximaDB functionality:

- **`sql_frontend`** - SQL query parsing and execution engine
- **`graph-first-sks`** - Graph-first architecture with ORION-backed entity store
- **`unified-facade-routing`** - Unified query facade for consistent execution

**Production Status**: ✅ Production Ready  
**Test Coverage**: 80%+  
**Backward Compatibility**: Guaranteed

### 2. Optional Surfaces

These features control optional API surfaces and are kept off by default to minimize binary size:

- **`tenant_access`** - Multi-tenant access control

(The former `sales_endpoints` and `executive_intel` features were removed with
the commercial-surface externalization — boundary-spec phase B4, #437; the
former `ai_endpoints` executive-dashboard/BI surface was removed by TD-OSS-2.)

**Production Status**: ⚠️ Beta  
**Test Coverage**: 60-70%  
**Backward Compatibility**: Best effort, may change between releases

**How to Enable**:
```bash
cargo build --features tenant_access
```

### 3. Experimental Features

⚠️ **WARNING**: Experimental features are incomplete and may cause panics or data loss.

- **`experimental-engines`** - SWIFT and RAPTOR storage engines (archived research engines)
- **`simd-experimental`** - Experimental SIMD optimizations (unstable)
- **`avx512`** - AVX-512 intrinsics (requires nightly Rust)

**Production Status**: ❌ Experimental - NOT for production use  
**Test Coverage**: 30-50%  
**Backward Compatibility**: NOT guaranteed  
**Known Issues**:
- SWIFT/RAPTOR: Archived research engines, superseded by SST/VIPER

**How to Enable**:
```bash
cargo build --features experimental-engines
```

**Graph note**: PULSAR and QUASAR graph engine flags have been removed. Use ORION; distributed placement is a relational substrate concern and tiering is catalog/storage policy.

### 4. Distributed Features

Features for distributed multi-node deployments:

- **`distributed`** - Basic distributed operations
- **`cluster`** - Full cluster mode with Raft consensus, replication, health services
- **`standalone`** - Standalone single-node mode

**Production Status**: ⚠️ Beta (cluster), ✅ Production (standalone)  
**Test Coverage**: 70% (cluster), 90%+ (standalone)  
**Backward Compatibility**: Guaranteed for standalone, evolving for cluster

**How to Enable**:
```bash
# Cluster mode
cargo build --features cluster

# Standalone mode (default for single-node)
cargo build --features standalone
```

### 5. Storage Backend Features

Optional storage backend integrations:

- **`rocksdb`** - RocksDB metadata backend (alternative to built-in)
- **`aws`** - AWS S3 cloud storage support
- **`azure`** - Azure Blob Storage support
- **`gcp`** - Google Cloud Storage support
- **`cloud-full`** - All cloud backends (AWS + Azure + GCP)

**Production Status**: ✅ Production Ready  
**Test Coverage**: 75-85%  
**Backward Compatibility**: Guaranteed

**How to Enable**:
```bash
# Individual cloud backends
cargo build --features aws
cargo build --features azure
cargo build --features gcp

# All cloud backends
cargo build --features cloud-full

# RocksDB backend
cargo build --features rocksdb
```

### 6. Network and Integration Features

Network protocols and external system integrations:

- **`network-rest`** - REST API endpoints (HTTP/HTTPS)
- **`cdc-kafka`** - Change Data Capture to Kafka integration
- **`llm-joins`** - LLM-driven semantic join modes (block-batched joins)
- **`compile_protobuf`** - Protobuf compilation for distributed gRPC

**Production Status**: ✅ Production Ready (network-rest, cdc-kafka), ⚠️ Beta (llm-joins)  
**Test Coverage**: 80-90% (network-rest, cdc-kafka), 60% (llm-joins)  
**Backward Compatibility**: Guaranteed

**How to Enable**:
```bash
cargo build --features network-rest,cdc-kafka
cargo build --features llm-joins
```

### 7. Embedded Language Bindings

Language-specific bindings for embedded database mode:

- **`python`** - Python bindings via PyO3 with zero-copy NumPy support
- **`java`** - Java bindings via JNI
- **`nodejs`** - Node.js bindings via NAPI-RS
- **`c_ffi`** - C FFI for Go CGO and other C-compatible languages
- **`embedded-all`** - All embedded language bindings

**Production Status**: ✅ Production Ready (Python), ⚠️ Beta (Java, Node.js, C FFI)  
**Test Coverage**: 85%+ (Python), 50-70% (others)  
**Backward Compatibility**: Guaranteed for Python, evolving for others

**How to Enable**:
```bash
# Python bindings (most mature)
cargo build --features python

# All language bindings
cargo build --features embedded-all
```

**Python cdylib Output**:
- For PyO3 builds, use: `cargo build --features python,pylib`
- The `pylib` feature enables cdylib crate-type for shared library generation
- See ADR-006 for rationale on feature-gated cdylib

### 8. Enterprise Catalog Features

Enterprise data catalog integrations:

- **`unity-catalog`** - Databricks Unity Catalog integration
- **`polaris-catalog`** - Apache Polaris (Iceberg REST Catalog) integration
- **`delta-lake`** - Delta Lake table format support
- **`enterprise-catalogs`** - All enterprise catalogs (Unity + Polaris + Delta)

**Production Status**: ✅ Production Ready  
**Test Coverage**: 75-80%  
**Backward Compatibility**: Guaranteed

**How to Enable**:
```bash
# Individual catalogs
cargo build --features unity-catalog
cargo build --features polaris-catalog
cargo build --features delta-lake

# All enterprise catalogs
cargo build --features enterprise-catalogs
```

## Production Readiness Criteria

Features must meet the following criteria to be considered "Production Ready":

1. **Test Coverage**: ≥80% for core features, ≥70% for optional features
2. **Performance**: Meets performance benchmarks documented in docs/SUPPORTED_SURFACE.adoc
3. **Security**: Passes security audit for relevant components
4. **Documentation**: Complete API documentation and usage examples
5. **Backward Compatibility**: Clear migration path for breaking changes
6. **Stability**: No unresolved critical bugs or known panics
7. **Operational Readiness**: Monitoring, logging, and debugging support

## Feature Maturity Levels

- **Alpha**: Experimental, breaking changes likely, limited testing
- **Beta**: Feature complete, limited production use, backward compatibility best effort
- **Production Ready**: Extensively tested, backward compatibility guaranteed, fully documented

## Feature Development Lifecycle

1. **Experimental** → Initial development, may be incomplete or unstable
2. **Beta** → Feature complete, testing in progress, limited production use
3. **Production Ready** → Meets all production readiness criteria
4. **Deprecated** → Superseded by better alternatives, will be removed
5. **Archived** → Removed from active development, moved to archive/

## Feature Flag Combinations

### Recommended Combinations

**Development/Testing**:
```bash
cargo build --features sql_frontend,graph-first-sks,unified-facade-routing
```

**Production Single-Node**:
```bash
cargo build --features python,network-rest,cdc-kafka
```

**Production Cluster**:
```bash
cargo build --features cluster,cloud-full,network-rest
```

**Full Enterprise**:
```bash
cargo build --features cluster,enterprise-catalogs,network-rest,cdc-kafka,python
```

**Experimental Development**:
```bash
cargo build --features experimental-engines
```

### Incompatible Combinations

- `experimental-engines` + `cluster` - Experimental engines not cluster-aware
- `standalone` + `cluster` - Mutually exclusive deployment modes
- `avx512` + stable Rust - Requires nightly compiler

## Feature-Specific Notes

### Python/PyO3 Features

The `python` feature enables Python bindings but does NOT change the crate-type. To build a Python shared library (cdylib), also enable the `pylib` feature:

```bash
# Python bindings with rlib (default, for testing)
cargo build --features python

# Python bindings with cdylib (for PyO3 shared library)
cargo build --features python,pylib
```

### Graph Engine Features

**ORION** (default, recommended):
- Always available, no feature flag needed
- Production-ready, most mature (253+ tests)
- In-memory CSR format with WAL persistence

**Retired names**:
- `PULSAR` and `QUASAR` are no longer Cargo features or graph engine modules.
- Former PULSAR behavior maps to relational distributed planning requirements.
- Former QUASAR behavior maps to xCatalog/storage projection policy requirements.

### CDC Connector Features

**Kafka Integration**:
- Enable with `cdc-kafka` feature
- Production-ready, comprehensive Kafka producer/consumer support

**Native CDC Connectors** (PostgreSQL, MySQL, MongoDB):
- Enable with `experimental-cdc-connectors` feature
- ⚠️ Experimental: partial implementations without full network I/O
- Recommendation: Use Debezium with Kafka sink for production MySQL/MongoDB CDC

## Debugging Feature Flags

### Verbose Feature Compilation

To see which features are being compiled:

```bash
cargo build --features python,cluster 2>&1 | grep "cargo:rust-cdylib"
```

### Feature Dependency Resolution

To see feature dependencies:

```bash
cargo tree --features python,pylib
```

### Feature-Specific Unit Tests

To run tests for specific features:

```bash
# Test only Python bindings
cargo test --features python --lib

# Test cluster features
cargo test --features cluster --lib

# Test with experimental features
cargo test --features experimental-engines --lib
```

## Migration Guide

### Upgrading Feature Flags

When feature flags change between versions:

1. **Review** this document for changes in the release notes
2. **Update** build scripts to use new feature flag combinations
3. **Test** with new features before deploying to production
4. **Monitor** for deprecation warnings in build output

### Deprecated Features

Deprecated features will emit warnings during compilation:

```
warning: The 'experimental-engines' feature is deprecated and will be removed in v0.3.0
```

Action required before next upgrade.

## Related Documentation

- **docs/SUPPORTED_SURFACE.adoc** - Feature status and production readiness matrix
- **../04-operations/production-readiness.adoc** - Detailed production readiness criteria
- **ADR-006** - Infrastructure and platform decisions (including test binary fix)
- **TECHNICAL_DEBT.adoc** - Known gaps and limitations

## Feature Request Process

To request new features or changes to existing features:

1. Check existing technical debt in TECHNICAL_DEBT.adoc
2. Review strategic roadmap in docs/current/ROADMAP.adoc
3. Submit GitHub issue with feature proposal
4. Include production readiness criteria and timeline

## Summary

ProximaDB's feature flag system provides:
- ✅ **Modular compilation** - Only build what you need
- ✅ **Clear separation** - Stable vs experimental functionality  
- ✅ **Production safety** - Experimental features opt-in only
- ✅ **Binary size optimization** - Exclude unnecessary components
- ✅ **Flexibility** - Customize for your deployment scenario

For production deployments, use recommended feature combinations and avoid experimental features unless you're specifically testing or developing them.
