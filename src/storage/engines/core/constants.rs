//! Centralized constants for all storage engines
//!
//! This module consolidates all constants used across storage engines to:
//! - Avoid duplication
//! - Ensure consistency
//! - Simplify maintenance
//! - Provide a single source of truth

/// Common constants used across all storage engines
pub mod common {
    /// Default cache size in MB (used by all engines)
    pub const DEFAULT_CACHE_SIZE_MB: usize = 1024;

    /// Default compression level for ZSTD (balanced performance)
    pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

    /// Default bloom filter false positive rate
    pub const DEFAULT_BLOOM_FPP: f64 = 0.01;

    /// Default prefetch size for cloud I/O (MB)
    pub const DEFAULT_PREFETCH_SIZE_MB: usize = 32;

    /// Default buffer pool size (MB)
    pub const DEFAULT_BUFFER_POOL_SIZE_MB: usize = 512;

    /// Maximum parallel read operations
    pub const DEFAULT_MAX_PARALLEL_READS: usize = 8;

    /// Default SIMD lane width for vectorized operations
    pub const DEFAULT_SIMD_LANES: usize = 16;

    /// Compaction trigger when this many files exist
    pub const COMPACTION_THRESHOLD_FILES: usize = 4;

    /// Minimum file size before compaction (MB)
    pub const COMPACTION_MIN_SIZE_MB: usize = 1;
}

/// SST engine specific constants
pub mod sst {
    /// SST file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"SST3";

    /// Current SST file format version
    pub const VERSION: u32 = 3;

    /// Default block size (16KB)
    pub const DEFAULT_BLOCK_SIZE: usize = 16384;

    /// Default target file size (64MB)
    pub const DEFAULT_TARGET_FILE_SIZE: usize = 64 * 1024 * 1024;
}

/// NOVA engine specific constants
pub mod nova {
    /// NOVA file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"NOVA";

    /// Current NOVA file format version
    pub const VERSION: u32 = 1;

    /// Default quantization levels
    pub const DEFAULT_QUANTIZATION_LEVELS: usize = 4;

    /// Progressive search batch size
    pub const DEFAULT_PROGRESSIVE_BATCH_SIZE: usize = 100;
}

/// SWIFT engine specific constants
pub mod swift {
    /// SWIFT file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"SWFT";

    /// Current SWIFT file format version
    pub const VERSION: u32 = 1;

    /// Default superblock size
    pub const DEFAULT_SUPERBLOCK_SIZE: usize = 256;

    /// Proxima encoding bits
    pub const DEFAULT_PROXIMA_BITS: u8 = 16;
}

/// RAPTOR engine specific constants
pub mod raptor {
    /// RAPTOR file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"RPTR";

    /// Current RAPTOR file format version
    pub const VERSION: u32 = 1;

    /// RAPTOR file extension
    pub const FILE_EXTENSION: &str = "raptor";

    /// Default row group size optimized for k<10 queries
    pub const DEFAULT_ROWGROUP_SIZE: usize = 1024;

    /// Minimum row group size for meaningful clustering benefit
    pub const MIN_ROWGROUP_SIZE: usize = 512;

    /// Default P² matrix dimension (vectors per rowgroup)
    pub const DEFAULT_P_DIMENSION: usize = 1024;

    /// Maximum K² matrix dimension (number of centroids)
    pub const MAX_K_DIMENSION: usize = 10000;

    /// P×K matrix compression threshold
    pub const PXK_COMPRESSION_THRESHOLD: usize = 100;
}

/// PRISM engine specific constants
pub mod prism {
    /// PRISM file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"PRSM";

    /// Current PRISM file format version
    pub const VERSION: u32 = 1;

    /// Default resolution levels for multi-resolution quantization
    pub const DEFAULT_RESOLUTION_LEVELS: usize = 3;

    /// Memory budget per resolution level (MB)
    pub const MEMORY_BUDGET_PER_LEVEL_MB: usize = 256;
}

/// HELIX engine specific constants
pub mod helix {
    /// HELIX file magic constant (4 bytes)
    pub const MAGIC: [u8; 4] = *b"HELX";

    /// Current HELIX file format version
    pub const VERSION: u32 = 1;

    /// Default spiral pattern parameters
    pub const DEFAULT_SPIRAL_DEPTH: usize = 8;

    /// Time window for time-series optimization (seconds)
    pub const DEFAULT_TIME_WINDOW_SECONDS: u64 = 3600;
}

/// Quantization constants used across multiple engines
pub mod quantization {
    /// Default bits per key for quantization
    pub const DEFAULT_BITS_PER_KEY: u32 = 10;

    /// Default codebook size for PQ
    pub const DEFAULT_CODEBOOK_SIZE: usize = 256;

    /// Minimum vectors required for quantization
    pub const MIN_VECTORS_FOR_QUANTIZATION: usize = 1000;
}

/// Block size configuration constants
/// These define vector-count based block sizing (more intuitive for vector DBs)
/// To convert to bytes: bytes = vector_count * dimension * sizeof(f32) + metadata_overhead
pub mod block_sizes {
    // ====== Vector-count based block sizes (preferred) ======

    /// Default vectors per block for SST engine (16KB at 384D = ~10 vectors)
    pub const SST_DEFAULT_VECTORS_PER_BLOCK: usize = 128;

    /// Default vectors per block for HELIX Proxima blocks
    pub const HELIX_DEFAULT_VECTORS_PER_BLOCK: usize = 128;

    /// Default vectors per SWIFT superblock
    pub const SWIFT_DEFAULT_VECTORS_PER_SUPERBLOCK: usize = 256;


    /// Default vectors per NOVA progressive block
    pub const NOVA_DEFAULT_VECTORS_PER_BLOCK: usize = 100;

    /// Default vectors per RAPTOR row group (optimized for k<10 queries)
    pub const RAPTOR_DEFAULT_VECTORS_PER_ROWGROUP: usize = 512;

    // ====== Minimum and Maximum clamping values ======

    /// Minimum vectors per block (for all engines)
    pub const MIN_VECTORS_PER_BLOCK: usize = 16;

    /// Maximum vectors per block (prevent memory issues)
    pub const MAX_VECTORS_PER_BLOCK: usize = 8192;

    /// Minimum row group size for meaningful clustering
    pub const MIN_ROWGROUP_SIZE: usize = 256;

    /// Maximum row group size
    pub const MAX_ROWGROUP_SIZE: usize = 16384;

    // ====== Bytes-based sizes (for I/O optimization) ======

    /// Default block size in bytes for SST (16KB)
    pub const SST_DEFAULT_BLOCK_SIZE_BYTES: usize = 16 * 1024;

    /// Default page size for Parquet (256KB - balanced for embeddings)

    /// Target file size for SST files (64MB)
    pub const SST_TARGET_FILE_SIZE_BYTES: usize = 64 * 1024 * 1024;

    /// Target superblock size for SWIFT (1GB)
    pub const SWIFT_TARGET_SUPERBLOCK_SIZE_BYTES: usize = 1024 * 1024 * 1024;

    /// Minimum file size before compaction (1MB)
    pub const MIN_FILE_SIZE_FOR_COMPACTION_BYTES: usize = 1024 * 1024;

    /// Helper: Estimate bytes per vector for a given dimension
    #[inline]
    pub const fn bytes_per_vector(dimension: usize) -> usize {
        // 4 bytes per f32 + ~64 bytes metadata overhead (id, metadata hash, etc.)
        dimension * 4 + 64
    }

    /// Helper: Calculate block size in bytes from vector count
    #[inline]
    pub const fn block_size_bytes(vectors_per_block: usize, dimension: usize) -> usize {
        vectors_per_block * bytes_per_vector(dimension)
    }
}

/// Block pruning configuration constants
/// These control when block-level pruning is applied vs full scan
pub mod pruning {
    /// Minimum blocks before pruning is cost-effective (SST/HELIX/SWIFT)
    pub const MIN_BLOCKS_FOR_PRUNING: usize = 100;

    /// Minimum files before file-level pruning is effective
    pub const MIN_FILES_FOR_PRUNING: usize = 10;

    /// Default pruning distance threshold multiplier
    /// Blocks within best_distance * threshold are scanned
    pub const DEFAULT_PRUNING_THRESHOLD: f32 = 1.5;

    /// Pruning efficiency threshold (skip if >90% blocks would be pruned)
    pub const PRUNING_EFFICIENCY_THRESHOLD: f32 = 0.9;
}

/// Common vector dimensions for optimization defaults
pub mod dimensions {
    /// OpenAI text-embedding-ada-002 dimension
    pub const OPENAI_ADA_002: usize = 1536;

    /// OpenAI text-embedding-3-small dimension
    pub const OPENAI_3_SMALL: usize = 1536;

    /// OpenAI text-embedding-3-large dimension
    pub const OPENAI_3_LARGE: usize = 3072;

    /// Sentence transformers all-MiniLM-L6-v2 dimension
    pub const SENTENCE_TRANSFORMERS_MINI: usize = 384;

    /// BERT base model dimension
    pub const BERT_BASE: usize = 768;

    /// Default dimension when not specified
    pub const DEFAULT: usize = SENTENCE_TRANSFORMERS_MINI;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_consistency() {
        // Ensure all magic constants are 4 bytes
        assert_eq!(sst::MAGIC.len(), 4);
        assert_eq!(nova::MAGIC.len(), 4);
        assert_eq!(swift::MAGIC.len(), 4);
        assert_eq!(raptor::MAGIC.len(), 4);
        assert_eq!(prism::MAGIC.len(), 4);
        assert_eq!(helix::MAGIC.len(), 4);

        // Ensure common values are reasonable
        assert!(common::DEFAULT_CACHE_SIZE_MB > 0);
        assert!(common::DEFAULT_COMPRESSION_LEVEL >= 1 && common::DEFAULT_COMPRESSION_LEVEL <= 22);
        assert!(common::DEFAULT_BLOOM_FPP > 0.0 && common::DEFAULT_BLOOM_FPP < 1.0);

        // Ensure RAPTOR constants are consistent
        assert!(raptor::DEFAULT_ROWGROUP_SIZE >= raptor::MIN_ROWGROUP_SIZE);
    }
}
