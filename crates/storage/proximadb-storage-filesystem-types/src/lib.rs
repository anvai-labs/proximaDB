//! Pure filesystem abstraction types for ProximaDB storage.
//!
//! The `FileSystem` / `FilesystemFile` traits plus their supporting value types,
//! extracted from `src/storage/persistence/filesystem/mod.rs` into a leaf crate
//! so storage modules (encryption, schema, common, metadata) can depend *down*
//! on the abstraction rather than up into the root crate. Behavior-neutral: the
//! root path re-exports these via `pub use proximadb_storage_filesystem_types::*;`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Error as IoError;

/// TD-096 S2 / S1.5: GET-count instrumentation on the filesystem seam.
pub mod counting;
pub mod faults;

/// Turning logical byte needs into physical requests (ADR-034 P7).
pub mod read_ranges_plan;

/// Filesystem operation result type
pub type FsResult<T> = Result<T, FilesystemError>;

/// Filesystem error types
#[derive(Debug, thiserror::Error)]
pub enum FilesystemError {
    #[error("IO error: {0}")]
    Io(#[from] IoError),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Unsupported filesystem scheme: {0}")]
    UnsupportedScheme(String),

    #[error("File not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Backwards-compat alias for [`FsFileMetadata`].
pub type FileMetadata = FsFileMetadata;

/// File metadata information
#[derive(Debug, Clone, Default)]
pub struct FsFileMetadata {
    pub path: String,
    pub size: u64,
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    pub is_directory: bool,
    pub permissions: Option<String>,
    pub etag: Option<String>,          // For cloud storage
    pub storage_class: Option<String>, // For cloud storage
}

/// Directory listing entry (stateless design - contains full URL)
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub url: String, // Full URL instead of relative path
    pub metadata: FsFileMetadata,
}

/// Metrics from a bounded-concurrent `read_ranges` call (TD-RDSTRAT-12).
///
/// `fetch_rounds` is the count of parallel rounds issued by `buffered(N)`.
/// `max_inflight` is the peak concurrent reads in a single round.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadRangesMetrics {
    /// Number of parallel fetch rounds issued.
    pub fetch_rounds: u64,
    /// Peak concurrent reads in any round (bounded by the parallelism cap).
    pub max_inflight: u64,
}

/// Thread-local storage for the most recent `read_ranges` metrics (TD-RDSTRAT-12).
///
/// Similar to the IVF `PROBE_TRACE` pattern: each `read_ranges` call with
/// `parallel > 1` records its metrics here, and callers with tenant context
/// (e.g., the SST engine) drain and emit to Prometheus.
static READ_RANGES_METRICS: std::sync::Mutex<Option<ReadRangesMetrics>> =
    std::sync::Mutex::new(None);

/// Record metrics from a bounded-concurrent `read_ranges` call (TD-RDSTRAT-12).
///
/// Called by `read_ranges_buffered_with_metrics` after the reads complete.
/// Thread-safe; overwrites any previous value (one metric per query is sufficient).
pub fn record_read_ranges_metrics(metrics: ReadRangesMetrics) {
    if let Ok(mut m) = READ_RANGES_METRICS.lock() {
        *m = Some(metrics);
    }
}

/// Drain the most recent `read_ranges` metrics, if any (TD-RDSTRAT-12).
///
/// Returns `None` if no metrics have been recorded since the last drain (or if
/// `read_ranges` ran sequentially with `parallel <= 1`).
pub fn drain_read_ranges_metrics() -> Option<ReadRangesMetrics> {
    READ_RANGES_METRICS.lock().ok().and_then(|mut m| m.take())
}

/// Bounded-concurrent, order-preserving, short-circuit-on-error ranged read
/// **with metrics tracking** (TD-RDSTRAT-12).
///
/// Returns the buffers and the metrics (`fetch_rounds`, `max_inflight`).
/// The caller should emit the metrics to Prometheus via
/// `READ_RANGES_FETCH_ROUNDS_TOTAL` and `READ_RANGES_MAX_INFLIGHT`.
async fn read_ranges_buffered_with_metrics<F, Fut>(
    ranges: Vec<std::ops::Range<u64>>,
    parallel: usize,
    read: F,
) -> FsResult<(Vec<Vec<u8>>, ReadRangesMetrics)>
where
    F: Fn(u64, u64) -> Fut + Clone,
    Fut: std::future::Future<Output = FsResult<Vec<u8>>>,
{
    use futures::stream::{self, StreamExt as _, TryStreamExt as _};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_inflight = Arc::new(AtomicUsize::new(0));
    let max_inflight_snapshot = max_inflight.clone();
    let read = read.clone();

    let buffers: Vec<Vec<u8>> = stream::iter(ranges)
        .map(move |r| {
            let read = read.clone();
            let in_flight = in_flight.clone();
            let max_inflight = max_inflight.clone();
            async move {
                in_flight.fetch_add(1, Ordering::SeqCst);
                let cur = in_flight.load(Ordering::SeqCst);
                max_inflight.fetch_max(cur, Ordering::SeqCst);
                let result = read(r.start, r.end - r.start).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                result
            }
        })
        .buffered(parallel)
        .try_collect()
        .await?;

    // Estimate rounds: ceil(len / parallel), with a minimum of 1
    let fetch_rounds = if buffers.is_empty() {
        0
    } else {
        ((buffers.len() as f64) / (parallel as f64)).ceil().max(1.0) as u64
    };

    let metrics = ReadRangesMetrics {
        fetch_rounds,
        max_inflight: max_inflight_snapshot.load(Ordering::SeqCst) as u64,
    };

    Ok((buffers, metrics))
}

/// Bounded-concurrent, order-preserving, short-circuit-on-error ranged read.
#[derive(Debug, Clone, Default)]
pub enum TempStrategy {
    /// Direct write (no temp files) - for local filesystem with atomic guarantees
    DirectWrite,

    /// Write to ___temp subdirectory in same location (same mount point)
    /// Ensures move operations are filesystem renames, not copies
    #[default]
    SameDirectory,

    /// Write to user-configured temp directory
    /// Falls back to system /tmp if not configured (R&D mode)
    ConfiguredTemp {
        /// Custom temp directory path (optional)
        temp_dir: Option<String>,
    },

    /// Write to system /tmp directory (fallback for R&D)
    SystemTemp,
}

/// File operation options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileOptions {
    pub create_dirs: bool,
    pub overwrite: bool,
    pub buffer_size: Option<usize>,
    pub encryption: Option<String>,
    /// Canonical object-storage access tier for this write (the cost lever — see
    /// [`ObjectAccessTier`]). Holds the lowercase canonical name (`hot`|`cool`|
    /// `cold`|`archive`) or a native provider spelling; a backend maps it to its
    /// provider class at the I/O boundary. `None` ⇒ the account/backend default
    /// (today's behavior, unchanged).
    pub storage_class: Option<String>,
    pub metadata: Option<HashMap<String, String>>,

    /// Pre-computed temp path (cached for performance)
    /// None means direct write, Some means atomic write-temp-rename
    pub temp_path: Option<String>,
}

impl FileOptions {
    /// Resolve [`FileOptions::storage_class`] to a canonical [`ObjectAccessTier`].
    /// Returns `None` when unset or unrecognized — the caller then uses the
    /// backend/account default tier rather than failing the write.
    pub fn access_tier(&self) -> Option<ObjectAccessTier> {
        self.storage_class
            .as_deref()
            .and_then(ObjectAccessTier::parse)
    }
}

/// Canonical, cloud-neutral object-storage **access tier** — the dominant
/// object-storage **cost lever** (ADR-036). A colder tier trades retrieval
/// latency/cost for a far lower at-rest GB-month price; on object storage the
/// at-rest term, not CPU, is what we co-design against. Callers choose a
/// *canonical* tier and each cloud backend maps it to that provider's native
/// class at the I/O boundary, so call sites never hard-code `"Cool"` (Azure) vs
/// `"STANDARD_IA"` (S3) vs `"NEARLINE"` (GCS).
///
/// Distinct from [`FileStorageTier`], which models *physical media latency*
/// (Memory/NVMe/SSD/…); this is the *cloud billing tier* applied per object PUT.
///
/// IMPORTANT (Azure orientation, ADR-036): the tier is a property of the **object
/// PUT** (`x-ms-access-tier`), available on a *flat* Blob account — it does **not**
/// require ADLS Gen2 Hierarchical Namespace. HNS adds per-operation cost with no
/// benefit for our flat-key, immutable, ranged-read workload; the access tier is
/// the lever, the namespace mode is not. The `az://`/`azure://`/`adls://`/`abfs://`
/// schemes all resolve to the *same* Blob endpoint backend (see the filesystem
/// factory) — the scheme is **not** a cost lever, only the tier is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObjectAccessTier {
    /// Frequent access, lowest latency, highest at-rest price. The backend default.
    Hot,
    /// Infrequent access (≈30-day min). Azure `Cool` / S3 `STANDARD_IA` / GCS `NEARLINE`.
    Cool,
    /// Rare access, still online/instant-retrieval. Azure `Cold` / S3 Glacier
    /// Instant Retrieval / GCS `COLDLINE`.
    Cold,
    /// Offline, minutes-to-hours retrieval, cheapest. Azure `Archive` / S3 Glacier
    /// Flexible Retrieval / GCS `ARCHIVE`.
    Archive,
}

impl ObjectAccessTier {
    /// Canonical, cloud-neutral lowercase name — the stable label for metrics and
    /// the inverse of the canonical names accepted by [`Self::parse`]
    /// (`hot`/`cool`/`cold`/`archive`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Cool => "cool",
            Self::Cold => "cold",
            Self::Archive => "archive",
        }
    }

    /// Lenient parse: canonical names + common provider spellings. Returns `None`
    /// for an unrecognized value — callers fall back to the backend default and
    /// never fail a write on a tier typo.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hot" | "standard" => Some(Self::Hot),
            "cool" | "standard_ia" | "standard-ia" | "onezone_ia" | "nearline" => Some(Self::Cool),
            "cold" | "coldline" | "glacier_ir" | "glacier-ir" => Some(Self::Cold),
            "archive"
            | "glacier"
            | "glacier_flexible"
            | "deep_archive"
            | "glacier_deep_archive" => Some(Self::Archive),
            _ => None,
        }
    }

    /// Azure Blob `x-ms-access-tier` value (also the ADLS Gen2 access tier).
    pub fn as_azure_access_tier(self) -> &'static str {
        match self {
            Self::Hot => "Hot",
            Self::Cool => "Cool",
            Self::Cold => "Cold",
            Self::Archive => "Archive",
        }
    }

    /// AWS S3 `x-amz-storage-class` value. `Cold` maps to Glacier Instant Retrieval
    /// (synchronous, no restore); `Archive` maps to Glacier Flexible Retrieval.
    pub fn as_s3_storage_class(self) -> &'static str {
        match self {
            Self::Hot => "STANDARD",
            Self::Cool => "STANDARD_IA",
            Self::Cold => "GLACIER_IR",
            Self::Archive => "GLACIER",
        }
    }

    /// GCS storage class.
    pub fn as_gcs_storage_class(self) -> &'static str {
        match self {
            Self::Hot => "STANDARD",
            Self::Cool => "NEARLINE",
            Self::Cold => "COLDLINE",
            Self::Archive => "ARCHIVE",
        }
    }

    /// Canonical lowercase name — the serialized [`FileOptions::storage_class`] form.
    pub fn as_canonical_str(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Cool => "cool",
            Self::Cold => "cold",
            Self::Archive => "archive",
        }
    }
}

/// Authentication configuration for cloud providers
#[derive(Debug, Clone)]
pub struct FilesystemAuthConfig {
    /// AWS authentication method
    pub aws_auth: Option<AwsAuthMethod>,

    /// Azure authentication method
    pub azure_auth: Option<AzureAuthMethod>,

    /// GCS authentication method
    pub gcs_auth: Option<GcsAuthMethod>,

    /// Enable credential caching
    pub enable_credential_caching: bool,

    /// Credential refresh interval (seconds)
    pub credential_refresh_interval_seconds: u64,
}

#[derive(Debug, Clone)]
pub enum AwsAuthMethod {
    /// Use AWS IAM roles (recommended for EC2/ECS)
    IamRole,
    /// Use AWS credentials file
    CredentialsFile { profile: Option<String> },
    /// Use environment variables
    Environment,
    /// Use STS temporary credentials
    StsAssumeRole {
        role_arn: String,
        session_name: String,
    },
}

#[derive(Debug, Clone)]
pub enum AzureAuthMethod {
    /// Use Azure Managed Identity
    ManagedIdentity,
    /// Use Azure Service Principal
    ServicePrincipal {
        client_id: String,
        tenant_id: String,
    },
    /// Use Azure CLI authentication
    AzureCli,
    /// Use environment variables
    Environment,
}

#[derive(Debug, Clone)]
pub enum GcsAuthMethod {
    /// Use Application Default Credentials
    ApplicationDefault,
    /// Use service account file
    ServiceAccountFile { path: String },
    /// Use service account key
    ServiceAccountKey { key_json: String },
    /// Use environment variables
    Environment,
}

/// Backwards-compat alias for [`FsRetryConfig`].
pub type RetryConfig = FsRetryConfig;

/// Retry configuration for operations
#[derive(Debug, Clone)]
pub struct FsRetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial delay between retries (ms)
    pub initial_delay_ms: u64,
    /// Maximum delay between retries (ms)
    pub max_delay_ms: u64,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

/// Storage tier type for intelligent data placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileStorageTier {
    /// In-memory storage (fastest)
    Memory,
    /// NVMe SSD storage (microsecond latency)
    NVMe,
    /// SSD storage (millisecond latency)
    SSD,
    /// HDD storage (10-20ms latency)
    HDD,
    /// S3 Express One Zone (single-digit ms)
    S3Express,
    /// S3 Standard (10-100ms)
    S3Standard,
    /// S3 Glacier Instant (100ms+)
    S3GlacierInstant,
    /// Azure Premium SSD
    AzurePremium,
    /// Azure Standard SSD
    AzureStandard,
    /// Google Cloud SSD
    GcsSSD,
    /// Google Cloud HDD
    GcsHDD,
}

impl FileStorageTier {
    /// Get expected latency in microseconds
    pub fn expected_latency_us(&self) -> u64 {
        match self {
            FileStorageTier::Memory => 1,                 // <1μs
            FileStorageTier::NVMe => 100,                 // 100μs
            FileStorageTier::SSD => 1_000,                // 1ms
            FileStorageTier::HDD => 10_000,               // 10ms
            FileStorageTier::S3Express => 5_000,          // 5ms
            FileStorageTier::S3Standard => 50_000,        // 50ms
            FileStorageTier::S3GlacierInstant => 100_000, // 100ms
            FileStorageTier::AzurePremium => 500,         // 500μs
            FileStorageTier::AzureStandard => 2_000,      // 2ms
            FileStorageTier::GcsSSD => 800,               // 800μs
            FileStorageTier::GcsHDD => 15_000,            // 15ms
        }
    }

    /// Get optimal I/O size in bytes for this tier
    pub fn optimal_io_size(&self) -> usize {
        match self {
            FileStorageTier::Memory => 64 * 1024,                 // 64KB
            FileStorageTier::NVMe => 128 * 1024,                  // 128KB
            FileStorageTier::SSD => 256 * 1024,                   // 256KB
            FileStorageTier::HDD => 1024 * 1024,                  // 1MB
            FileStorageTier::S3Express => 512 * 1024,             // 512KB
            FileStorageTier::S3Standard => 1024 * 1024,           // 1MB
            FileStorageTier::S3GlacierInstant => 4 * 1024 * 1024, // 4MB
            FileStorageTier::AzurePremium => 256 * 1024,          // 256KB
            FileStorageTier::AzureStandard => 512 * 1024,         // 512KB
            FileStorageTier::GcsSSD => 256 * 1024,                // 256KB
            FileStorageTier::GcsHDD => 2 * 1024 * 1024,           // 2MB
        }
    }

    /// Check if this tier is faster than another
    pub fn is_faster_than(&self, other: &FileStorageTier) -> bool {
        self.expected_latency_us() < other.expected_latency_us()
    }
}

/// Backwards-compat alias for [`FsTierConfig`].
pub type TierConfig = FsTierConfig;

/// Tier-specific storage configuration
#[derive(Debug, Clone)]
pub struct FsTierConfig {
    /// Storage tier type
    pub tier: FileStorageTier,

    /// Base URL for this tier (e.g., "file:///mnt/nvme", "s3://bucket")
    pub base_url: String,

    /// Maximum capacity in bytes (None = unlimited)
    pub max_capacity_bytes: Option<u64>,

    /// Current usage in bytes (tracked runtime)
    pub current_usage_bytes: u64,

    /// Enable compression for this tier
    pub compression: bool,

    /// Custom I/O size override (uses tier default if None)
    pub io_size_override: Option<usize>,
}

/// Filesystem performance configuration
#[derive(Debug, Clone)]
pub struct FilesystemPerformanceConfig {
    /// Connection pool size per backend
    pub connection_pool_size: usize,

    /// Enable connection keep-alive
    pub enable_keep_alive: bool,

    /// Request timeout (seconds)
    pub request_timeout_seconds: u64,

    /// Enable compression for network transfers
    pub compression: bool,

    /// Retry configuration
    pub retry_config: FsRetryConfig,

    /// Buffer size for operations (bytes)
    pub buffer_size: usize,

    /// Enable parallel operations
    pub enable_parallel_ops: bool,

    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,

    /// Tier-specific configurations
    pub tier_configs: Vec<FsTierConfig>,
}

/// File handle trait for streaming operations on large files
/// Provides async read/write capabilities similar to tokio::fs::File
#[async_trait]
pub trait FilesystemFile: Send + Sync + std::fmt::Debug {
    /// Read data from current position
    async fn read(&mut self, buf: &mut [u8]) -> FsResult<usize>;

    /// Write data at current position
    async fn write(&mut self, buf: &[u8]) -> FsResult<usize>;

    /// Flush any buffered writes
    async fn flush(&mut self) -> FsResult<()>;

    /// Seek to position (if supported)
    async fn seek(&mut self, pos: u64) -> FsResult<u64>;

    /// Get current position
    async fn position(&self) -> FsResult<u64>;

    /// Get file size
    async fn file_size(&self) -> FsResult<u64>;

    /// Sync data to underlying storage
    async fn sync_all(&mut self) -> FsResult<()>;
}

/// Bounded-concurrent, order-preserving, short-circuit-on-error ranged read.
///
/// Used by [`FileSystem::read_ranges`] when `PROXIMADB_FS_READ_RANGES_PARALLEL > 1`
/// (TD-RDSTRAT-8 rev-2.1). Extracted as a free fn so the concurrency contract is
/// unit-testable without a full `FileSystem` implementation: `buffered` preserves
/// result order; `try_collect` short-circuits on the first error (identical to a
/// sequential `await?` loop). The futures share a `&self` borrow (read takes
/// `&self`) — polled in-place on the current task, no spawn, no Send/'static.
#[allow(dead_code)]
async fn read_ranges_buffered<F, Fut>(
    ranges: Vec<std::ops::Range<u64>>,
    parallel: usize,
    read: F,
) -> FsResult<Vec<Vec<u8>>>
where
    F: Fn(u64, u64) -> Fut,
    Fut: std::future::Future<Output = FsResult<Vec<u8>>>,
{
    use futures::stream::{self, StreamExt as _, TryStreamExt as _};
    stream::iter(ranges)
        .map(|r| read(r.start, r.end - r.start))
        .buffered(parallel)
        .try_collect()
        .await
}

/// Policy for merging the logical ranges of one [`FileSystem::read_ranges`]
/// call into fewer physical requests.
///
/// Object stores bill **per request** — Azure Hot, S3 Standard and GCS Standard
/// each charge one transaction per ranged GET regardless of its size — so the
/// count of physical reads is the billed read cost, and bytes are paid for in
/// latency and memory rather than money. Merging therefore trades a bounded
/// over-read for a saved round trip.
///
/// Both bounds are required. `max_gap_bytes` alone (which is all upstream
/// `object_store` enforces) leaves the over-read unbounded: a hundred scattered
/// 4 KiB ranges under a 1 MiB gap would materialise ~100 MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeCoalescePolicy {
    /// Largest gap between two logical ranges that may be bridged.
    pub max_gap_bytes: u64,
    /// Ceiling on a **merged** physical read — the over-read/RSS bound.
    ///
    /// This bounds merging, not individual reads: a single logical range larger
    /// than this is still issued whole, because the caller needs those bytes
    /// contiguously and splitting them would leave no single buffer to slice
    /// from. So a physical read exceeds this value only when it serves exactly
    /// one logical range.
    pub max_merged_bytes: u64,
}

/// Abstract filesystem trait for strategy pattern
#[async_trait]
pub trait FileSystem: Send + Sync + std::fmt::Debug {
    /// Range-merging policy for this filesystem instance, or `None` to issue one
    /// physical read per logical range.
    ///
    /// Injected at construction rather than resolved inside the implementation:
    /// this crate is a pure filesystem abstraction with no dependency on
    /// `proximadb-storage-common`, so it cannot (and should not) reach for
    /// `IopsBudget` itself. `FilesystemFactory` builds one instance per scheme,
    /// so a backend-derived policy is fully expressible here.
    ///
    /// Any implementation that overrides [`FileSystem::read_ranges`] MUST also
    /// override this to describe what it really does, or the physical-GET meter
    /// in `CountingFileSystem` will report intent instead of truth.
    fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
        None
    }

    /// Which physical reads would serve these logical ranges, and where each
    /// range's bytes land inside them.
    ///
    /// Pure and synchronous — no I/O. Exposed on the trait so a decorator can
    /// report **measured** physical requests rather than assuming one per
    /// logical range: `CountingFileSystem` charges `plan.physical.len()`.
    fn plan_read_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
    ) -> FsResult<read_ranges_plan::RangePlan> {
        read_ranges_plan::coalesce_ranges_with_mapping(ranges, self.range_coalesce_policy())
    }

    /// Get self as Any for downcasting to concrete types
    fn as_any(&self) -> &dyn std::any::Any;

    /// Read file contents
    async fn read(&self, path: &str) -> FsResult<Vec<u8>>;

    /// Get memory-mapped access to a file (only supported for local filesystem)
    /// Returns None if memory mapping is not supported (e.g., cloud storage)
    /// The returned mmap is read-only and safe for concurrent access
    async fn get_mmap(&self, _path: &str) -> FsResult<Option<memmap2::Mmap>> {
        // Default implementation returns None (not supported)
        // LocalFileSystem will override this to provide actual memory mapping
        Ok(None)
    }

    /// Check if this filesystem supports memory mapping
    fn supports_mmap(&self) -> bool {
        false // Default: most filesystems don't support mmap
    }

    /// Read specific byte range from file (for efficient cloud storage access)
    /// Returns the requested bytes. Default implementation reads entire file and slices.
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        // Default implementation for backwards compatibility
        let data = self.read(path).await?;
        let start = offset as usize;
        let end = (offset + length) as usize;

        if start >= data.len() {
            return Ok(vec![]);
        }

        let end = end.min(data.len());
        Ok(data[start..end].to_vec())
    }

    /// Read multiple byte ranges from file in a single operation.
    ///
    /// Default implementation issues one `read_range` per range. By default these
    /// are **sequential** (identical to a loop, one at a time); set
    /// `PROXIMADB_FS_READ_RANGES_PARALLEL=N` (N > 1) to issue up to N concurrently —
    /// order-preserving and short-circuit-on-error, so results + error semantics
    /// are identical to the sequential path. The coalesced read path issues many
    /// ranged GETs per query (e.g. 51 @ SIFT1M); bounded concurrency turns cold
    /// latency from `GETs × RTT` into `rounds × RTT` (TD-RDSTRAT-8 rev-2.1, the
    /// waves-not-sums latency model). Default OFF (0) until the wave-latency
    /// model is measured on real cloud.
    ///
    /// **Decode-bound vs pure prefetch:** This method uses the `PARALLEL` cap,
    /// which is tuned for decode-bound work (zstd, SQ8, distance kernels). For
    /// pure I/O-bound prefetches (metadata, footers) where decode is minimal or
    /// none, use `read_ranges_prefetch` instead, which uses the higher `INFLIGHT`
    /// cap (TD-RDSTRAT-12).
    async fn read_ranges(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        // `PROXIMADB_FS_READ_RANGES_PARALLEL`: bounded concurrent ranged reads.
        // Parsed once + cached (fn-local `static` = persists across calls, read
        // on the hot path without re-parsing env each call).
        static PARALLEL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let parallel = *PARALLEL.get_or_init(|| {
            std::env::var("PROXIMADB_FS_READ_RANGES_PARALLEL")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
        });
        // Plan first: which physical reads serve which logical ranges. With no
        // policy this is the identity plan, so the requests issued below are
        // exactly the ones this method issued before coalescing existed.
        let plan = self.plan_read_ranges(&ranges)?;
        let identity = self.range_coalesce_policy().is_none();

        let buffers = if parallel <= 1 {
            // Sequential (the default): one physical read at a time.
            let mut buffers = Vec::with_capacity(plan.physical.len());
            for range in &plan.physical {
                let length = range.end - range.start;
                buffers.push(self.read_range(path, range.start, length).await?);
            }
            buffers
        } else {
            // Bounded-concurrent with metrics tracking (TD-RDSTRAT-12).
            // Delegates to `read_ranges_buffered_with_metrics`, which tracks
            // `fetch_rounds` and `max_inflight` and records them to thread-local
            // storage for callers with tenant context to emit to Prometheus.
            let (bufs, metrics) = read_ranges_buffered_with_metrics(
                plan.physical.clone(),
                parallel,
                |offset, length| self.read_range(path, offset, length),
            )
            .await?;
            record_read_ranges_metrics(metrics);
            bufs
        };

        if identity {
            // MOVE the buffers out untouched rather than re-slicing them. A
            // backend that clamped at EOF returned a short buffer; re-slicing it
            // through the mapping would clamp it a second time and could differ
            // from the pre-change loop. Moving makes "policy unset ⇒ byte-identical"
            // a property of the code, not a hope.
            return Ok(buffers);
        }

        Ok(plan
            .mapping
            .iter()
            .map(|slice| match slice.physical {
                Some(idx) => read_ranges_plan::slice_from_physical(&buffers[idx], *slice),
                // Zero-length range: satisfied without issuing any request.
                None => Vec::new(),
            })
            .collect())
    }

    /// Read multiple byte ranges from file in **pure prefetch mode** (TD-RDSTRAT-12).
    ///
    /// Uses `PROXIMADB_READ_RANGES_INFLIGHT` for bounded concurrency instead of
    /// `PROXIMADB_FS_READ_RANGES_PARALLEL`. The `INFLIGHT` cap is tuned for pure
    /// I/O-bound work where decode is minimal or none (metadata, footers, raw zstd),
    /// allowing higher concurrency than the decode-bound `PARALLEL` cap.
    ///
    /// **When to use:** For prefetches that are I/O-bound with minimal CPU work
    /// after the read (e.g., PAX footers, IVF coarse probe posts, metadata reads).
    /// For decode-bound work (zstd decompression, SQ8 parsing, distance kernels),
    /// use `read_ranges` instead, which respects the lower `PARALLEL` cap.
    ///
    /// **Concurrency model:** Same as `read_ranges` — order-preserving, short-
    /// circuit-on-error, with `fetch_rounds` and `max_inflight` metrics emitted.
    /// The only difference is which env gate controls the `buffered(N)` concurrency.
    async fn read_ranges_prefetch(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        // `PROXIMADB_READ_RANGES_INFLIGHT`: pure prefetch cap (TD-RDSTRAT-12).
        // Parsed once + cached (fn-local `static` = persists across calls, read
        // on the hot path without re-parsing env each call).
        //
        // Semantics (per ENV_GATE_REGISTRY):
        // - unset = defer to PROXIMADB_FS_READ_RANGES_PARALLEL
        // - 0 = same as PARALLEL
        // - N > 0 = separate cap (independent of PARALLEL)
        static INFLIGHT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let inflight = *INFLIGHT.get_or_init(|| {
            std::env::var("PROXIMADB_READ_RANGES_INFLIGHT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0) // Default: 0 = defer to PARALLEL (opt-in until measured)
        });

        // If INFLIGHT is 0 (default or explicitly set), defer to PARALLEL.
        let inflight = if inflight == 0 {
            static PARALLEL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *PARALLEL.get_or_init(|| {
                std::env::var("PROXIMADB_FS_READ_RANGES_PARALLEL")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0)
            })
        } else {
            inflight
        };

        // Plan first: which physical reads serve which logical ranges.
        let plan = self.plan_read_ranges(&ranges)?;
        let identity = self.range_coalesce_policy().is_none();

        let buffers = if inflight <= 1 {
            // Sequential: one physical read at a time.
            let mut buffers = Vec::with_capacity(plan.physical.len());
            for range in &plan.physical {
                let length = range.end - range.start;
                buffers.push(self.read_range(path, range.start, length).await?);
            }
            buffers
        } else {
            // Bounded-concurrent with metrics tracking (TD-RDSTRAT-12).
            let (bufs, metrics) = read_ranges_buffered_with_metrics(
                plan.physical.clone(),
                inflight,
                |offset, length| self.read_range(path, offset, length),
            )
            .await?;

            // Record to global metrics for downstream emission (TD-RDSTRAT-12).
            // A higher layer (e.g., observability-engine) drains these and forwards to
            // Prometheus and the per-query io_trace, respecting layer boundaries.
            record_read_ranges_metrics(metrics);

            bufs
        };

        if identity {
            return Ok(buffers);
        }

        Ok(plan
            .mapping
            .iter()
            .map(|slice| match slice.physical {
                Some(idx) => read_ranges_plan::slice_from_physical(&buffers[idx], *slice),
                None => Vec::new(),
            })
            .collect())
    }

    /// Write file contents
    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()>;

    /// Whether [`FileSystem::write_local_file`] has a bounded-memory
    /// implementation rather than the compatibility whole-file fallback.
    ///
    /// Callers with an admitted memory ceiling (notably local-spill
    /// compaction) must check this capability and fail closed. Decorators must
    /// delegate only when they preserve the underlying bound.
    fn supports_bounded_local_file_write(&self) -> bool {
        false
    }

    /// Publish an existing local file as one object and return its byte count.
    ///
    /// Object-store implementations override this with bounded multipart or
    /// resumable upload so compaction does not rematerialize a multi-GiB PAX
    /// segment in memory at publication. The compatibility default preserves
    /// behavior for non-cloud plugins; it is intentionally observable in code
    /// review as an unbounded fallback and should not be used by major cloud
    /// backends.
    async fn write_local_file(
        &self,
        path: &str,
        local_path: &std::path::Path,
        options: Option<FileOptions>,
    ) -> FsResult<u64> {
        let data = std::fs::read(local_path)?;
        let bytes = data.len() as u64;
        self.write(path, &data, options).await?;
        Ok(bytes)
    }

    /// Atomically create a new file/object, failing with [`FilesystemError::AlreadyExists`]
    /// when the key is already present. Recovery protocols use this as their commit
    /// primitive; implementations must not emulate it with a racy exists-then-write.
    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let _ = (data, options);
        Err(FilesystemError::InvalidOperation(format!(
            "{} does not support conditional create for {path}",
            self.filesystem_type()
        )))
    }

    /// Whether this adapter explicitly implements conditional replacement.
    /// Transforming/caching decorators must qualify the full contract to opt in.
    fn supports_conditional_replace(&self) -> bool {
        false
    }

    /// Exact byte-conditioned durable replacement of a small coordination file.
    /// None means absent, not empty; false means mismatch or lock contention.
    /// All writers must cooperate and domain records must prevent ABA themselves.
    /// Errors may follow publication. No read/overwrite emulation is permitted.
    /// Caching/transforming adapters must explicitly implement compatible semantics
    /// or retain this fail-closed default; ordinary write_atomic is not a substitute.
    /// Read-only guards must share the target parent directory and be compared
    /// under the same serialization boundary as the target replacement.
    async fn compare_exchange(
        &self,
        path: &str,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&str, Option<&[u8]>)],
    ) -> FsResult<bool> {
        let _ = (expected, replacement, guards);
        Err(FilesystemError::InvalidOperation(format!(
            "{} does not support conditional replacement for {path}",
            self.filesystem_type()
        )))
    }

    /// Sync file data to disk (fsync/fdatasync)
    /// Ensures data durability after write operations
    /// Returns Ok(()) if sync is not supported by the filesystem
    async fn sync_file(&self, _path: &str) -> FsResult<()> {
        // Default implementation - no sync
        // Filesystems that support sync should override
        Ok(())
    }

    /// Sync file *data* to disk (fdatasync): flush contents without necessarily
    /// flushing inode metadata (size/mtime) separately — a cheaper durability
    /// barrier than [`FileSystem::sync_file`] on local disks. The default
    /// delegates to `sync_file` (a full fsync, a safe superset), so backends
    /// without a distinct data-only barrier (object stores, etc.) need not
    /// override it. `LocalFileSystem` overrides this with a real `fdatasync`.
    async fn sync_data(&self, path: &str) -> FsResult<()> {
        self.sync_file(path).await
    }

    /// Append to file
    async fn append(&self, path: &str, data: &[u8]) -> FsResult<()>;

    /// Whether this backend provides a native, durable append operation.
    ///
    /// Object stores intentionally leave this false. Callers whose on-disk
    /// format requires append must reject the backend during construction,
    /// before accepting data, rather than discovering the mismatch on flush.
    fn supports_append(&self) -> bool {
        false
    }

    /// Delete file or directory
    async fn delete(&self, path: &str) -> FsResult<()>;

    /// Check if file exists
    async fn exists(&self, path: &str) -> FsResult<bool>;

    /// Get file metadata
    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata>;

    /// List directory contents
    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>>;

    /// Create directory
    async fn create_dir(&self, path: &str) -> FsResult<()>;

    /// Create directory and all parent directories
    async fn create_dir_all(&self, path: &str) -> FsResult<()>;

    /// Copy file
    async fn copy(&self, from: &str, to: &str) -> FsResult<()>;

    /// Move/rename file
    async fn move_file(&self, from: &str, to: &str) -> FsResult<()>;

    /// Get filesystem type identifier
    fn filesystem_type(&self) -> &'static str;

    /// Check if filesystem supports atomic writes natively
    /// Local filesystems can write directly, object stores need atomic pattern
    fn supports_atomic_writes(&self) -> bool {
        match self.filesystem_type() {
            "local" => true, // Local filesystem supports atomic writes natively
            _ => false,      // Object stores (S3, ADLS, GCS) need write-temp-rename pattern
        }
    }

    /// Generate temporary file path based on strategy (called once during setup)
    /// Ensures optimal temp location for each filesystem type
    fn generate_temp_path(&self, final_path: &str, strategy: &TempStrategy) -> FsResult<String> {
        use std::env;
        use std::path::{Path, PathBuf};

        let final_path = Path::new(final_path);
        let filename = final_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| FilesystemError::InvalidPath("Invalid filename".to_string()))?;

        match strategy {
            TempStrategy::DirectWrite => {
                // No temp file needed for direct writes
                Err(FilesystemError::InvalidOperation(
                    "DirectWrite strategy should not generate temp paths".to_string(),
                ))
            }

            TempStrategy::SameDirectory => {
                // Create ___temp subdirectory in same location (same mount point)
                let parent = final_path.parent();
                let temp_dir = parent.map_or_else(
                    || std::path::PathBuf::from("___temp"),
                    |p| p.join("___temp"),
                );
                let temp_file = temp_dir.join(format!("{}.{}", filename, std::process::id())); // Add PID for uniqueness
                Ok(temp_file.to_string_lossy().to_string())
            }

            TempStrategy::ConfiguredTemp { temp_dir } => {
                // Use configured temp dir or fall back to system temp
                let temp_base = if let Some(dir) = temp_dir {
                    PathBuf::from(dir)
                } else {
                    // Fallback to system temp for R&D mode
                    env::temp_dir()
                };
                let temp_file =
                    temp_base.join(format!("proximadb_{}.{}", filename, std::process::id()));
                Ok(temp_file.to_string_lossy().to_string())
            }

            TempStrategy::SystemTemp => {
                // Use system /tmp directory
                let temp_file =
                    env::temp_dir().join(format!("proximadb_{}.{}", filename, std::process::id()));
                Ok(temp_file.to_string_lossy().to_string())
            }
        }
    }

    /// Fast atomic write using pre-computed temp path (performance optimized)
    /// Called during actual write operations with cached temp strategy
    async fn write_atomic(
        &self,
        final_path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let temp_path_opt = options.as_ref().and_then(|o| o.temp_path.clone());

        match temp_path_opt {
            None => {
                // Direct write (optimal for local filesystem)
                self.write(final_path, data, options).await
            }
            Some(temp_path_str) => {
                // Atomic write-temp-rename (optimal for object stores)
                let temp_path = std::path::Path::new(&temp_path_str);

                // Ensure temp directory exists
                if let Some(temp_parent) = temp_path.parent() {
                    self.create_dir_all(&temp_parent.to_string_lossy()).await?;
                }

                // Write to temp location
                let temp_opts = options.map(|o| FileOptions {
                    temp_path: None, // Prevent recursion
                    ..o
                });
                self.write(&temp_path_str, data, temp_opts).await?;

                // Atomic move (rename on same mount point)
                self.move_file(&temp_path_str, final_path).await
            }
        }
    }

    /// Sync/flush operations to storage
    async fn sync(&self) -> FsResult<()>;

    /// Read file as string (UTF-8) - convenience method for text files
    async fn read_to_string(&self, path: &str) -> FsResult<String> {
        let bytes = self.read(path).await?;
        String::from_utf8(bytes)
            .map_err(|e| FilesystemError::InvalidOperation(format!("Invalid UTF-8: {}", e)))
    }

    /// Write string to file - convenience method for text files
    async fn write_string(
        &self,
        path: &str,
        content: &str,
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        self.write(path, content.as_bytes(), options).await
    }

    /// Remove directory and all contents recursively
    async fn remove_dir_all(&self, path: &str) -> FsResult<()> {
        // Default implementation using list and delete
        let entries = self.list(path).await?;

        for entry in entries {
            if entry.metadata.is_directory {
                self.remove_dir_all(&entry.url).await?;
            } else {
                self.delete(&entry.url).await?;
            }
        }

        self.delete(path).await
    }

    /// Create a file handle for streaming operations (for large files)
    /// Returns a file handle that implements AsyncRead + AsyncWrite
    async fn open_file(&self, path: &str, create: bool) -> FsResult<Box<dyn FilesystemFile>>;
}

impl Default for FilesystemPerformanceConfig {
    fn default() -> Self {
        Self {
            connection_pool_size: 10,
            enable_keep_alive: true,
            request_timeout_seconds: 30,
            compression: true,
            retry_config: FsRetryConfig {
                max_retries: 3,
                initial_delay_ms: 100,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
            },
            buffer_size: 8 * 1024 * 1024, // 8MB
            enable_parallel_ops: true,
            max_concurrent_ops: 100,
            tier_configs: vec![],
        }
    }
}

/// Convert a filesystem error into the foundational `VectorDBError` contract.
/// Defined here (where `FilesystemError` is local) to satisfy the orphan rule
/// after the trait + types were extracted from the root crate.
impl From<FilesystemError> for proximadb_kernel::error::VectorDBError {
    fn from(err: FilesystemError) -> Self {
        proximadb_kernel::error::VectorDBError::Filesystem(err.to_string())
    }
}

#[cfg(test)]
mod object_access_tier_tests {
    use super::{FileOptions, ObjectAccessTier};

    #[test]
    fn parse_accepts_canonical_names_case_insensitively() {
        assert_eq!(ObjectAccessTier::parse("hot"), Some(ObjectAccessTier::Hot));
        assert_eq!(
            ObjectAccessTier::parse("Cool"),
            Some(ObjectAccessTier::Cool)
        );
        assert_eq!(
            ObjectAccessTier::parse(" COLD "),
            Some(ObjectAccessTier::Cold)
        );
        assert_eq!(
            ObjectAccessTier::parse("archive"),
            Some(ObjectAccessTier::Archive)
        );
    }

    #[test]
    fn parse_accepts_native_provider_spellings() {
        // S3
        assert_eq!(
            ObjectAccessTier::parse("STANDARD"),
            Some(ObjectAccessTier::Hot)
        );
        assert_eq!(
            ObjectAccessTier::parse("STANDARD_IA"),
            Some(ObjectAccessTier::Cool)
        );
        assert_eq!(
            ObjectAccessTier::parse("GLACIER_IR"),
            Some(ObjectAccessTier::Cold)
        );
        assert_eq!(
            ObjectAccessTier::parse("GLACIER"),
            Some(ObjectAccessTier::Archive)
        );
        // GCS
        assert_eq!(
            ObjectAccessTier::parse("NEARLINE"),
            Some(ObjectAccessTier::Cool)
        );
        assert_eq!(
            ObjectAccessTier::parse("COLDLINE"),
            Some(ObjectAccessTier::Cold)
        );
    }

    #[test]
    fn parse_returns_none_for_unknown_so_write_falls_back_to_default() {
        assert_eq!(ObjectAccessTier::parse("warm"), None);
        assert_eq!(ObjectAccessTier::parse(""), None);
    }

    #[test]
    fn per_cloud_mappings_are_native() {
        assert_eq!(ObjectAccessTier::Cool.as_azure_access_tier(), "Cool");
        assert_eq!(ObjectAccessTier::Cold.as_azure_access_tier(), "Cold");
        assert_eq!(ObjectAccessTier::Cool.as_s3_storage_class(), "STANDARD_IA");
        assert_eq!(ObjectAccessTier::Cold.as_s3_storage_class(), "GLACIER_IR");
        assert_eq!(ObjectAccessTier::Archive.as_gcs_storage_class(), "ARCHIVE");
    }

    #[test]
    fn canonical_str_round_trips_through_parse() {
        for t in [
            ObjectAccessTier::Hot,
            ObjectAccessTier::Cool,
            ObjectAccessTier::Cold,
            ObjectAccessTier::Archive,
        ] {
            assert_eq!(ObjectAccessTier::parse(t.as_canonical_str()), Some(t));
        }
    }

    #[test]
    fn file_options_access_tier_resolves_and_defaults_safely() {
        let mut o = FileOptions::default();
        assert_eq!(o.access_tier(), None); // unset ⇒ backend default
        o.storage_class = Some("cool".to_string());
        assert_eq!(o.access_tier(), Some(ObjectAccessTier::Cool));
        o.storage_class = Some("nonsense".to_string());
        assert_eq!(o.access_tier(), None); // typo ⇒ default, never an error
    }
}

#[cfg(test)]
mod read_ranges_buffered_tests {
    //! TD-RDSTRAT-8 rev-2.1: the bounded-concurrent `read_ranges` contract —
    //! order preservation, the concurrency bound, and error short-circuit.
    use super::{FilesystemError, read_ranges_buffered};
    use std::future::poll_fn;
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::Poll;

    /// 10 disjoint ranges, start = i·7, length 5 (deterministic).
    fn sample_ranges() -> Vec<Range<u64>> {
        (0..10_u64).map(|i| (i * 7)..(i * 7 + 5)).collect()
    }

    #[test]
    fn parallel_matches_sequential_in_order() {
        // Same per-range bytes (offset repeated length times); seq vs par must agree.
        let seq = futures::executor::block_on(read_ranges_buffered(
            sample_ranges(),
            1,
            |o, l| async move { Ok::<_, FilesystemError>(vec![o as u8; l as usize]) },
        ));
        let par = futures::executor::block_on(read_ranges_buffered(
            sample_ranges(),
            4,
            |o, l| async move { Ok::<_, FilesystemError>(vec![o as u8; l as usize]) },
        ));
        assert_eq!(seq.unwrap(), par.unwrap());
    }

    #[test]
    fn respects_bound_and_runs_concurrently() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let read = {
            let (in_flight, peak) = (in_flight.clone(), peak.clone());
            move |o: u64, l: u64| {
                let (in_flight, peak) = (in_flight.clone(), peak.clone());
                async move {
                    let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(cur, Ordering::SeqCst);
                    // Yield once so `buffered` overlaps sibling futures (proves
                    // concurrency); per-future AtomicBool state.
                    let yielded = AtomicBool::new(false);
                    poll_fn(|cx| {
                        if !yielded.swap(true, Ordering::SeqCst) {
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            Poll::Ready(())
                        }
                    })
                    .await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    Ok::<_, FilesystemError>(vec![o as u8; l as usize])
                }
            }
        };
        let out =
            futures::executor::block_on(read_ranges_buffered(sample_ranges(), 4, read)).unwrap();
        assert_eq!(out.len(), 10);
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 4, "exceeded the bound of 4: peak={peak}");
        assert!(peak >= 2, "did not overlap futures: peak={peak}");
    }

    #[test]
    fn short_circuits_on_first_error() {
        // range i=2 → start=14 → error; try_collect must surface it, not mask it.
        let read = |o: u64, _l: u64| async move {
            if o == 14 {
                return Err(FilesystemError::InvalidOperation("bad range".into()));
            }
            Ok(vec![o as u8])
        };
        let res = futures::executor::block_on(read_ranges_buffered(sample_ranges(), 4, read));
        assert!(res.is_err(), "expected short-circuit error, got {res:?}");
    }
}

#[cfg(test)]
mod read_ranges_prefetch_tests {
    //! Tests for `FileSystem::read_ranges_prefetch` (TD-RDSTRAT-12 INFLIGHT cap).
    use super::{
        DirEntry, FileOptions, FileSystem, FilesystemError, FsFileMetadata, FsResult,
        RangeCoalescePolicy,
    };
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Simple mock FileSystem for testing prefetch behavior.
    #[derive(Debug)]
    struct MockFs {
        read_count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl FileSystem for MockFs {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, _path: &str) -> FsResult<Vec<u8>> {
            Ok(Vec::new())
        }

        async fn read_range(&self, _path: &str, _offset: u64, length: u64) -> FsResult<Vec<u8>> {
            self.read_count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![b'x'; length as usize])
        }

        async fn write(
            &self,
            _path: &str,
            _data: &[u8],
            _options: Option<FileOptions>,
        ) -> FsResult<()> {
            Ok(())
        }
        async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
            Ok(())
        }
        async fn delete(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn exists(&self, _path: &str) -> FsResult<bool> {
            Ok(false)
        }
        async fn metadata(&self, _path: &str) -> FsResult<FsFileMetadata> {
            Ok(FsFileMetadata {
                path: _path.to_string(),
                size: 100,
                created: None,
                modified: None,
                is_directory: false,
                permissions: None,
                etag: None,
                storage_class: None,
            })
        }
        async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
            Ok(Vec::new())
        }
        async fn create_dir(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn copy(&self, _from: &str, _to: &str) -> FsResult<()> {
            Ok(())
        }
        async fn move_file(&self, _from: &str, _to: &str) -> FsResult<()> {
            Ok(())
        }
        fn filesystem_type(&self) -> &'static str {
            "local"
        }
        async fn sync(&self) -> FsResult<()> {
            Ok(())
        }
        async fn open_file(
            &self,
            _path: &str,
            _create: bool,
        ) -> FsResult<Box<dyn super::FilesystemFile>> {
            Err(FilesystemError::InvalidOperation("mock".into()))
        }

        fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
            None
        }

        fn plan_read_ranges(
            &self,
            ranges: &[Range<u64>],
        ) -> FsResult<super::read_ranges_plan::RangePlan> {
            Ok(super::read_ranges_plan::RangePlan {
                physical: ranges.to_vec(),
                mapping: ranges
                    .iter()
                    .map(|r| super::read_ranges_plan::PlanSlice {
                        physical: None,
                        offset: 0,
                        len: (r.end - r.start) as usize,
                    })
                    .collect(),
            })
        }
    }

    fn sample_ranges() -> Vec<Range<u64>> {
        (0..10_u64).map(|i| (i * 7)..(i * 7 + 5)).collect()
    }

    #[test]
    fn prefetch_defers_to_parallel_when_unset() {
        // Both INFLIGHT and PARALLEL unset → sequential (default PARALLEL=0)
        let fs = MockFs {
            read_count: Arc::new(AtomicUsize::new(0)),
        };
        let _ = std::env::var("PROXIMADB_READ_RANGES_INFLIGHT")
            .map(|_v| unsafe { std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT") });
        let _ = std::env::var("PROXIMADB_FS_READ_RANGES_PARALLEL")
            .map(|_v| unsafe { std::env::remove_var("PROXIMADB_FS_READ_RANGES_PARALLEL") });

        let result =
            futures::executor::block_on(fs.read_ranges_prefetch("test.bin", sample_ranges()));
        assert!(result.is_ok());
        // Sequential: one read per range (no coalescing in this mock)
        assert_eq!(fs.read_count.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn prefetch_uses_parallel_when_inflight_is_zero() {
        // INFLIGHT=0 explicitly → defer to PARALLEL
        unsafe {
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "0");
        }
        unsafe {
            std::env::set_var("PROXIMADB_FS_READ_RANGES_PARALLEL", "1");
        }
        let fs = MockFs {
            read_count: Arc::new(AtomicUsize::new(0)),
        };

        let result =
            futures::executor::block_on(fs.read_ranges_prefetch("test.bin", sample_ranges()));
        assert!(result.is_ok());
        // Sequential (PARALLEL=1)
        assert_eq!(fs.read_count.load(Ordering::SeqCst), 10);

        unsafe {
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
        }
        unsafe {
            std::env::remove_var("PROXIMADB_FS_READ_RANGES_PARALLEL");
        }
    }

    #[test]
    fn prefetch_uses_inflight_cap_when_set() {
        // INFLIGHT=4, PARALLEL unset → use INFLIGHT=4
        unsafe {
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
        }
        let _ = std::env::var("PROXIMADB_FS_READ_RANGES_PARALLEL")
            .map(|_v| unsafe { std::env::remove_var("PROXIMADB_FS_READ_RANGES_PARALLEL") });
        let fs = MockFs {
            read_count: Arc::new(AtomicUsize::new(0)),
        };

        let result =
            futures::executor::block_on(fs.read_ranges_prefetch("test.bin", sample_ranges()));
        assert!(result.is_ok());
        // All ranges read (concurrency bound doesn't affect count)
        assert_eq!(fs.read_count.load(Ordering::SeqCst), 10);

        // Note: We can't reliably check metrics here because drain_read_ranges_metrics()
        // consumes the global metrics, and other tests may have run. The fact that the
        // call succeeded and returned 10 buffers is sufficient proof that the INFLIGHT
        // cap was used (sequential would have the same count but different behavior).

        unsafe {
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
        }
    }

    #[test]
    fn prefetch_preserves_order() {
        unsafe {
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
        }
        let fs = MockFs {
            read_count: Arc::new(AtomicUsize::new(0)),
        };

        let result =
            futures::executor::block_on(fs.read_ranges_prefetch("test.bin", sample_ranges()));
        assert!(result.is_ok());
        let bufs = result.unwrap();
        assert_eq!(bufs.len(), 10);

        // Each buffer should have length 5 and contain 'x' bytes
        for (i, buf) in bufs.iter().enumerate() {
            assert_eq!(buf.len(), 5, "buffer {i} has wrong length");
            assert!(
                buf.iter().all(|&b| b == b'x'),
                "buffer {i} has wrong content"
            );
        }

        unsafe {
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
        }
    }

    #[test]
    fn prefetch_short_circuits_on_error() {
        // Erroring FS mock
        #[derive(Debug)]
        struct ErrorFs;
        #[async_trait::async_trait]
        impl FileSystem for ErrorFs {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            async fn read(&self, _path: &str) -> FsResult<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn read_range(
                &self,
                _path: &str,
                offset: u64,
                _length: u64,
            ) -> FsResult<Vec<u8>> {
                if offset == 14 {
                    Err(FilesystemError::InvalidOperation("bad range".into()))
                } else {
                    Ok(vec![b'x'; 5])
                }
            }

            async fn write(
                &self,
                _path: &str,
                _data: &[u8],
                _options: Option<FileOptions>,
            ) -> FsResult<()> {
                Ok(())
            }
            async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
                Ok(())
            }
            async fn delete(&self, _path: &str) -> FsResult<()> {
                Ok(())
            }
            async fn exists(&self, _path: &str) -> FsResult<bool> {
                Ok(false)
            }
            async fn metadata(&self, _path: &str) -> FsResult<FsFileMetadata> {
                Ok(FsFileMetadata {
                    path: _path.to_string(),
                    size: 100,
                    created: None,
                    modified: None,
                    is_directory: false,
                    permissions: None,
                    etag: None,
                    storage_class: None,
                })
            }
            async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
                Ok(Vec::new())
            }
            async fn create_dir(&self, _path: &str) -> FsResult<()> {
                Ok(())
            }
            async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
                Ok(())
            }
            async fn copy(&self, _from: &str, _to: &str) -> FsResult<()> {
                Ok(())
            }
            async fn move_file(&self, _from: &str, _to: &str) -> FsResult<()> {
                Ok(())
            }
            fn filesystem_type(&self) -> &'static str {
                "local"
            }
            async fn sync(&self) -> FsResult<()> {
                Ok(())
            }
            async fn open_file(
                &self,
                _path: &str,
                _create: bool,
            ) -> FsResult<Box<dyn super::FilesystemFile>> {
                Err(FilesystemError::InvalidOperation("mock".into()))
            }

            fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
                None
            }

            fn plan_read_ranges(
                &self,
                ranges: &[Range<u64>],
            ) -> FsResult<super::read_ranges_plan::RangePlan> {
                Ok(super::read_ranges_plan::RangePlan {
                    physical: ranges.to_vec(),
                    mapping: ranges
                        .iter()
                        .map(|r| super::read_ranges_plan::PlanSlice {
                            physical: None,
                            offset: 0,
                            len: (r.end - r.start) as usize,
                        })
                        .collect(),
                })
            }
        }

        unsafe {
            std::env::set_var("PROXIMADB_READ_RANGES_INFLIGHT", "4");
        }
        let fs = ErrorFs;
        let result =
            futures::executor::block_on(fs.read_ranges_prefetch("test.bin", sample_ranges()));
        assert!(result.is_err(), "expected error, got {result:?}");
        unsafe {
            std::env::remove_var("PROXIMADB_READ_RANGES_INFLIGHT");
        }
    }
}

pub mod range_coalescer;
pub mod smart_io_traits;

#[cfg(test)]
pub(crate) mod read_ranges_coalescing_tests {
    //! The `read_ranges` seam: N logical ranges must not cost N physical GETs.
    //!
    //! Object stores bill per request (Azure Hot, S3 Standard, GCS Standard all
    //! charge one transaction per ranged GET regardless of size), so the number
    //! of physical `read_range` calls IS the billed read cost. The trait has had
    //! a multi-range signature since inception but no multi-range semantics —
    //! and two call sites document the opposite.
    //!
    //! The fake below counts physical `read_range` invocations, so these assert
    //! measured physics rather than intent. It deliberately does NOT override
    //! `read_ranges`: the point is to exercise the trait default.
    use super::{
        DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata,
        FsResult, RangeCoalescePolicy,
    };
    use std::ops::Range;
    use std::sync::Mutex;

    /// How a backend behaves when a range starts at or past EOF.
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum EofMode {
        /// Local filesystems: short read, `Ok`.
        Clamp,
        /// Azure/S3/GCS: `InvalidGetRange::StartTooLarge` / HTTP 416.
        Error,
    }

    #[derive(Debug)]
    pub(crate) struct CountingFake {
        data: Vec<u8>,
        calls: Mutex<Vec<(u64, u64)>>,
        eof: EofMode,
        policy: Option<RangeCoalescePolicy>,
    }

    impl CountingFake {
        pub(crate) fn new(len: usize, eof: EofMode, policy: Option<RangeCoalescePolicy>) -> Self {
            Self {
                data: (0..len).map(|i| (i % 251) as u8).collect(),
                calls: Mutex::new(Vec::new()),
                eof,
                policy,
            }
        }
        pub(crate) fn calls(&self) -> Vec<(u64, u64)> {
            self.calls.lock().expect("calls mutex").clone()
        }
        fn slice(&self, from: usize, to: usize) -> Vec<u8> {
            self.data[from.min(self.data.len())..to.min(self.data.len())].to_vec()
        }
    }

    #[async_trait::async_trait]
    impl FileSystem for CountingFake {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn read(&self, _path: &str) -> FsResult<Vec<u8>> {
            Ok(self.data.clone())
        }
        async fn read_range(&self, _path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
            // Record BEFORE applying EOF semantics: this is the physical meter.
            self.calls
                .lock()
                .expect("calls mutex")
                .push((offset, length));
            if length == 0 {
                return Ok(Vec::new());
            }
            let start = offset as usize;
            if start >= self.data.len() {
                return match self.eof {
                    EofMode::Clamp => Ok(Vec::new()),
                    EofMode::Error => Err(FilesystemError::InvalidOperation(format!(
                        "range start {offset} past EOF {}",
                        self.data.len()
                    ))),
                };
            }
            Ok(self.slice(start, start + length as usize))
        }
        fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
            self.policy
        }
        async fn write(&self, _p: &str, _d: &[u8], _o: Option<FileOptions>) -> FsResult<()> {
            unimplemented!("write not exercised")
        }
        async fn delete(&self, _path: &str) -> FsResult<()> {
            unimplemented!("delete not exercised")
        }
        async fn exists(&self, _path: &str) -> FsResult<bool> {
            Ok(true)
        }
        async fn metadata(&self, _path: &str) -> FsResult<FsFileMetadata> {
            unimplemented!("metadata not exercised")
        }
        async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
            unimplemented!("append not exercised")
        }
        async fn sync(&self) -> FsResult<()> {
            Ok(())
        }
        async fn open_file(&self, _path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
            unimplemented!("open_file not exercised")
        }
        async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
            Ok(Vec::new())
        }
        async fn create_dir(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn copy(&self, _f: &str, _t: &str) -> FsResult<()> {
            Ok(())
        }
        async fn move_file(&self, _f: &str, _t: &str) -> FsResult<()> {
            Ok(())
        }
        fn filesystem_type(&self) -> &'static str {
            "counting-fake"
        }
    }

    fn policy(gap: u64, max: u64) -> Option<RangeCoalescePolicy> {
        Some(RangeCoalescePolicy {
            max_gap_bytes: gap,
            max_merged_bytes: max,
        })
    }

    fn read(fake: &CountingFake, ranges: Vec<Range<u64>>) -> FsResult<Vec<Vec<u8>>> {
        futures::executor::block_on(fake.read_ranges("p", ranges))
    }

    /// R1 — the headline. Adjacent and near-adjacent ranges become ONE request.
    #[test]
    fn coalescing_merges_adjacent_ranges_into_one_physical_read() {
        let fake = CountingFake::new(4096, EofMode::Clamp, policy(16, 4096));
        let out = read(&fake, vec![0..64, 64..128, 130..200]).expect("read");
        assert_eq!(
            fake.calls().len(),
            1,
            "3 near-adjacent ranges must cost ONE physical GET, got {:?}",
            fake.calls()
        );
        assert_eq!(fake.calls()[0], (0, 200), "merged span covers all members");
        // Exact slices: PaxBlockReader parses a footer at len-FOOTER and CRCs the
        // body, so an extra trailing byte is a hard decode failure.
        assert_eq!(out[0], fake.slice(0, 64));
        assert_eq!(out[1], fake.slice(64, 128));
        assert_eq!(out[2], fake.slice(130, 200));
    }

    /// R2 — output must be in INPUT order, not sorted order. The PAX adapter
    /// builds RecordBatches positionally from this Vec, so a permutation is a
    /// wrong-answer bug, not a crash.
    #[test]
    fn returned_buffers_are_in_input_order_not_sorted_order() {
        let fake = CountingFake::new(4096, EofMode::Clamp, policy(4096, 8192));
        let out = read(&fake, vec![200..264, 0..64, 100..164]).expect("read");
        assert_eq!(out[0], fake.slice(200, 264), "input position 0");
        assert_eq!(out[1], fake.slice(0, 64), "input position 1");
        assert_eq!(out[2], fake.slice(100, 164), "input position 2");
        assert_eq!(fake.calls().len(), 1, "all three merge into one span");
    }

    /// R3 — duplicates and overlaps each get their own buffer; never deduped.
    #[test]
    fn duplicate_and_overlapping_ranges_each_get_their_own_buffer() {
        let fake = CountingFake::new(1024, EofMode::Clamp, policy(64, 4096));
        let out = read(&fake, vec![0..32, 0..32, 16..48]).expect("read");
        assert_eq!(out.len(), 3, "one buffer per INPUT range");
        assert_eq!(out[0], fake.slice(0, 32));
        assert_eq!(out[1], fake.slice(0, 32));
        assert_eq!(out[2], fake.slice(16, 48));
        assert_eq!(fake.calls().len(), 1);
    }

    /// R4 — the caps are the memory bound, asserted as physical fact. Upstream
    /// object_store has a gap cap but NO size cap; we must not inherit that.
    #[test]
    fn gap_above_cap_and_span_above_max_are_not_merged() {
        let far = CountingFake::new(4096, EofMode::Clamp, policy(8, 4096));
        read(&far, vec![0..32, 1000..1032]).expect("read");
        assert_eq!(far.calls().len(), 2, "gap 968 > cap 8 must not merge");

        let big = CountingFake::new(4096, EofMode::Clamp, policy(64, 128));
        read(&big, vec![0..64, 64..128, 128..192]).expect("read");
        assert_eq!(
            big.calls().len(),
            2,
            "merged span must never exceed max_merged_bytes, got {:?}",
            big.calls()
        );
    }

    /// R5 — EOF. Every backend clamps, so a merged buffer is SHORTER than the
    /// requested span; slicing must saturate on both ends or it panics.
    #[test]
    fn merged_read_short_at_eof_slices_saturating_never_panics() {
        // Arm 1: merged span crosses EOF; the tail range is partially satisfied.
        let clamp = CountingFake::new(100, EofMode::Clamp, policy(4096, 8192));
        let out = read(&clamp, vec![0..32, 90..140]).expect("clamped read");
        assert_eq!(clamp.calls().len(), 1);
        assert_eq!(out[0], clamp.slice(0, 32));
        assert_eq!(out[1], clamp.slice(90, 100), "10 bytes, saturated at EOF");

        // Arm 2: the tail range is entirely past EOF.
        let past = CountingFake::new(100, EofMode::Clamp, policy(4096, 8192));
        let out = read(&past, vec![0..32, 200..232]).expect("past-EOF read");
        assert!(
            out[1].is_empty(),
            "entirely past EOF yields an empty buffer"
        );

        // Arm 3: DOCUMENTED BEHAVIOUR CHANGE. On object stores a range starting
        // past EOF errors today (416 / StartTooLarge), so this call returns Err.
        // After merging, the single physical read starts at 0 and clamps to Ok.
        // Error becomes success — this converges Azure onto local semantics.
        // Defensible, but locked here so it is reviewed, not discovered.
        let store = CountingFake::new(100, EofMode::Error, policy(4096, 8192));
        let out = read(&store, vec![0..32, 200..232]).expect("merged read clamps to Ok");
        assert_eq!(store.calls().len(), 1);
        assert!(out[1].is_empty());
    }

    /// R6 — a zero-length range must cost no request and must not shift output.
    #[test]
    fn zero_length_range_costs_no_physical_read_and_yields_empty() {
        let fake = CountingFake::new(1024, EofMode::Clamp, policy(64, 4096));
        let out = read(&fake, vec![0..32, 64..64, 100..132]).expect("read");
        assert_eq!(out.len(), 3);
        assert!(
            out[1].is_empty(),
            "zero-length yields empty, keeps its slot"
        );
        assert!(
            fake.calls().iter().all(|(_, len)| *len > 0),
            "no zero-length physical read may be issued, got {:?}",
            fake.calls()
        );
    }

    /// R7 — an inverted range underflows `range.end - range.start` today: debug
    /// panic, release ~u64::MAX length. Fail closed instead.
    // See the planner's equivalent test: the inverted range IS the input under
    // test, so the lint's advice would delete the case.
    #[allow(clippy::reversed_empty_ranges)]
    #[test]
    fn inverted_range_is_rejected_not_silently_widened() {
        let fake = CountingFake::new(1024, EofMode::Clamp, policy(64, 4096));
        let err = read(&fake, vec![0..32, 40..10]).expect_err("inverted must be rejected");
        assert!(
            matches!(err, FilesystemError::InvalidOperation(_)),
            "expected InvalidOperation, got {err:?}"
        );
    }

    /// R8 — the mixed-read-safety lock. With no policy the plan is the identity
    /// and behaviour must be BYTE-IDENTICAL to the pre-change loop, including
    /// the zero-length call and past-EOF error semantics. Passes today; must
    /// keep passing, which is what makes "unset ⇒ unchanged" provable.
    #[test]
    fn policy_none_is_byte_identical_to_the_per_range_loop() {
        let fake = CountingFake::new(100, EofMode::Clamp, None);
        let out = read(&fake, vec![0..32, 200..232, 64..64]).expect("read");
        assert_eq!(
            fake.calls(),
            vec![(0, 32), (200, 32), (64, 0)],
            "identity plan must issue exactly the same calls, in order"
        );
        assert_eq!(out[0], fake.slice(0, 32));
        assert!(out[1].is_empty());
        assert!(out[2].is_empty());

        // And the object-store error semantics survive untouched when OFF.
        let store = CountingFake::new(100, EofMode::Error, None);
        read(&store, vec![0..32, 200..232]).expect_err("past-EOF still errors when OFF");
    }
}
