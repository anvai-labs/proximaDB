# Experimental Engines Status - SWIFT & RAPTOR

## Overview

This document describes the current status of ProximaDB's experimental storage engines and provides guidance for users and developers.

## Experimental Engines

### SWIFT (Storage With Indexed Fast Traversal)

**Status**: ⚠️ **INCOMPLETE - Not Production Ready**

**Feature Flag**: `experimental-engines`

**Description**: Hierarchical storage engine with three-tier architecture (SuperBlock → DataBlock → Records)

**Current Implementation**:
- ✅ Core hierarchical structure implemented
- ✅ Basic read/write operations functional
- ✅ Progressive search pipeline stages
- ✅ Unified reader integration
- ❌ 30+ TODO items for complete functionality
- ❌ Limited testing coverage (41 tests)

**Known Limitations**:
1. Incomplete batch operations
2. Missing SuperBlock cache optimization
3. Limited hierarchical search capabilities
4. No tenant isolation enforcement
5. Inefficient metadata serialization

**Recommended Use**: None - use SST, VIPER, HELIX, or NOVA instead

### RAPTOR (Row-Aligned Predicated Tensor Optimized Repository)

**Status**: ⚠️ **EXPERIMENTAL - Not Production Ready**

**Feature Flag**: `experimental-engines`

**Description**: Adaptive storage engine with Matrix Trinity architecture (P² + K² + P×K)

**Current Implementation**:
- ✅ Matrix Trinity architecture core
- ✅ Adaptive row group sizing
- ✅ Consolidated compaction
- ✅ Progressive search stages
- ❌ 35+ TODO items for optimization
- ❌ Limited testing coverage (23 tests)

**Known Limitations**:
1. Incomplete adaptive learning
2. Missing workload pattern detection
3. No centroid optimization
4. Limited matrix compression
5. Experimental clustering algorithms

**Recommended Use**: Research and development only - use SST, VIPER, HELIX, or NOVA for production

## Production Alternatives

### For Hierarchical Use Cases (SWIFT alternative):
- **SST**: Sorted String Table with efficient range queries
- **NOVA**: Columnar analytics with zone maps and predicate pushdown
- **Custom application-level hierarchy**: Build hierarchy using multiple collections

### For Adaptive Workloads (RAPTOR alternative):
- **VIPER**: Vector storage with Proxima encoding and compression
- **HELIX**: High-dimensional data with PCA dimension reduction
- **SST**: Efficient for static workloads with good performance

## Migration Guide

If you're currently using experimental engines, here's how to migrate:

### From SWIFT to SST:
```rust,ignore
// Instead of:
let swift_engine = SwiftEngine::new(config).await?;
swift_engine.create_department_hierarchy(&org_structure).await?;

// Use multiple SST collections with application-level hierarchy:
let marketing_collection = sst_engine.create_collection("marketing").await?;
let sales_collection = sst_engine.create_collection("sales").await?;
// Use collection naming conventions for hierarchy
```

### From RAPTOR to VIPER:
```rust,ignore
// Instead of:
let raptor_engine = RaptorEngine::new(config).await?;
raptor_engine.enable_adaptive_mode(true).await?;

// Use VIPER with optimized configuration:
let viper_config = ViperConfig::optimized_for_workload(&your_workload);
let viper_engine = ViperEngine::new(viper_config).await?;
```

## Development Status

### Completion Roadmap

**SWIFT Engine** (Estimated 2-3 months):
1. Complete batch operations (2 weeks)
2. Implement SuperBlock caching (1 week)
3. Add hierarchical search optimization (2 weeks)
4. Improve metadata serialization (1 week)
5. Enhance testing coverage (2 weeks)
6. Performance optimization (2 weeks)
7. Documentation and examples (1 week)

**RAPTOR Engine** (Estimated 3-4 months):
1. Implement adaptive learning algorithms (3 weeks)
2. Add workload pattern detection (2 weeks)
3. Optimize centroid computation (2 weeks)
4. Improve matrix compression (2 weeks)
5. Complete clustering implementation (3 weeks)
6. Enhance testing coverage (2 weeks)
7. Performance benchmarking (2 weeks)
8. Documentation and examples (1 week)

### Contribution Guidelines

If you want to help complete these engines:

1. **Start with tests**: Add comprehensive test coverage for existing functionality
2. **Focus on core features**: Complete essential operations before optimizations
3. **Document assumptions**: Clearly mark experimental behavior
4. **Measure performance**: Add benchmarks to track improvements
5. **Follow patterns**: Use consistent patterns with production engines

## Safety Guarantees

### Current Protections:
- ✅ Feature-gated behind `experimental-engines` flag
- ✅ Clear warnings in engine headers
- ✅ Not enabled by default
- ✅ Separated from production code paths
- ✅ Documented in technical debt tracker

### Missing Protections:
- ❌ No runtime deprecation warnings
- ❌ No migration tools
- ❌ No feature parity matrix
- ❌ No performance comparison data

## Recommendations

### For Users:
1. **Do not use** SWIFT or RAPTOR in production
2. **Use** SST, VIPER, HELIX, or NOVA instead
3. **Monitor** this document for completion status updates
4. **Test** production engines for your specific workload

### For Developers:
1. **Focus** on completing production engine features first
2. **Evaluate** whether experimental engine features are truly unique
3. **Consider** integrating valuable features into production engines
4. **Document** completion progress in this file

### For Project Leads:
1. **Assess** whether SWIFT and RAPTOR should be completed or deprecated
2. **Allocate** resources based on strategic value
3. **Set** clear milestones for completion or removal
4. **Communicate** timeline to users and developers

## Timeline

**Current Status**: Under Evaluation (2026-04-03)

**Decision Deadline**: 2026-05-01 (4 weeks)

**Options**:
1. **Complete**: Allocate resources to finish both engines (5-7 months)
2. **Deprecate**: Remove experimental engines and focus on production engines
3. **Hybrid**: Complete unique features, integrate into production engines, deprecate rest

**Next Review**: 2026-04-17 (2 weeks)

---

*Last Updated: 2026-04-03*
*Status: Under Evaluation*
*For questions: https://github.com/vijaysingh1992/proximadb/issues*

## Update 2026-10-04 (ADR-094 program)

RAPTOR executed ADR-093-style **stage-3 deletion**: `src/storage/engines/raptor/`,
`crates/storage/proximadb-raptor-engine`, `crates/storage/proximadb-raptor-common`,
the four factory dispatch arms/constructors, and `tests/raptor_recall_test.rs` are
removed (~21.6K LOC). The `StorageEngineType::Raptor` proto variant and the
`StorageEngine::Raptor`/`FormatType::Raptor` enum variants remain for wire compat
(stage-2 rule); factory construction now always errors with a migrate hint, and
hybrid-strategy collections fall back to SST. The migration target above is
corrected accordingly (VIPER is itself deprecated — ADR-093).
