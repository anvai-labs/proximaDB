// Engine type constants for consistent identification across the system.

pub const ENGINE_SST: &str = "sst";
pub const SST_MAGIC: [u8; 4] = *b"SST1";

pub const ENGINE_SWIFT: &str = "swift";
pub const SWIFT_MAGIC: [u8; 4] = *b"SWFT";

pub const ENGINE_NOVA: &str = "nova";
pub const NOVA_MAGIC: [u8; 4] = *b"NOVA";

pub const ENGINE_RAPTOR: &str = "raptor";
pub const RAPTOR_MAGIC: [u8; 4] = *b"RPTR";

pub const ENGINE_COLUMNAR: &str = "columnar";

pub const ENGINE_HELIX: &str = "helix";
pub const HELIX_MAGIC: [u8; 4] = *b"HELX";
pub const HELIX_FILE_EXT: &str = ".helix";

pub const SST_FILE_EXT: &str = ".sst";
pub const SWIFT_FILE_EXT: &str = ".swift";
pub const NOVA_FILE_EXT: &str = ".parquet";
pub const RAPTOR_FILE_EXT: &str = ".raptor";
pub const PRISM_FILE_EXT: &str = ".prism";

pub const METADATA_EXT: &str = ".meta";
pub const STATS_EXT: &str = ".stats";
pub const INDEX_EXT: &str = ".idx";
pub const BLOOM_FILTER_EXT: &str = ".bloom";

// ProximaBlocks sizing constants (SST/Swift/Helix)
pub const DEFAULT_BLOCK_METADATA_OVERHEAD_BYTES: usize = 200;
pub const DEFAULT_TARGET_BLOCK_SIZE_BYTES: usize = 3 * 1024 * 1024;
pub const MIN_TARGET_BLOCK_SIZE_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_TARGET_BLOCK_SIZE_BYTES: usize = 4 * 1024 * 1024;
/// Hard per-block ROW bound for the PAX segment writer. The normal block cut is
/// byte-driven off a *modeled* per-row estimate; this cap bounds the per-block
/// stripe-assembly transient (row buffers + per-column stripes materialized at
/// cut time) even when the model under-counts — e.g. the coalesced layout hoists
/// vectors out of blocks, so a two-level (IVF-probe) compaction cell can
/// otherwise grow a block far past the intended geometry. Slightly above the
/// largest byte-derived rows-per-block (4 MiB / 200 B ≈ 21k target-capped at
/// ~15.7k for the 3 MiB default), so healthy paths are unchanged.
pub const MAX_BLOCK_ROWS: usize = 16 * 1024;
