//! Deterministic, bounded local-scratch primitives for canonical compaction.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use proximadb_block_format::{
    FlatRow, PaxBlockReader, sq8_codes_offset, sq8_decode_codes, sq8_region_header_len,
};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use proximadb_storage_common::segment_layout::{
    SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::core::search::mvcc_resolution::{effective_version, is_append_only_oid};
use crate::storage::engines::sst::error::{Result, SstError};
use crate::storage::persistence::filesystem::FilesystemFactory;

const RUN_MAGIC: &[u8; 8] = b"PXSPLRUN";
const RUN_FORMAT_VERSION: u16 = 1;
const FRAME_HEADER_BYTES: u64 = 8;
const MAX_FRAME_BYTES: u64 = 1 << 30;
const OWNER_DIRECTORY_PREFIX: &str = "proximadb-compaction-owner-";
const TASK_MANIFEST_VERSION: u16 = 1;

#[derive(Debug, Serialize)]
struct SpillOwnerManifest {
    format_version: u16,
    process_id: u32,
    process_start_time_seconds: u64,
    owner_id: String,
}

/// Process-lifetime scratch namespace. Every compaction manager in the same
/// process resolves to the same directory. Liveness is an advisory filesystem
/// lock, not a PID assumption, so PID namespaces and reuse cannot make one
/// process delete another process's active work.
#[derive(Debug)]
pub(crate) struct SpillScratchOwner {
    inner: std::sync::Arc<SpillScratchOwnerInner>,
    reclaimed_owner_directories: usize,
}

#[derive(Debug)]
struct SpillScratchOwnerInner {
    path: PathBuf,
    _lease: File,
}

impl SpillScratchOwner {
    pub(crate) fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        let root = root.canonicalize()?;
        let mut registry = match spill_owner_registry().lock() {
            Ok(registry) => registry,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(owner) = registry.get(&root).and_then(std::sync::Weak::upgrade) {
            return Ok(Self {
                inner: owner,
                reclaimed_owner_directories: 0,
            });
        }

        let cleanup_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".owner-cleanup.lock"))?;
        if !lock_file_exclusive(&cleanup_lock, true)? {
            return Err(SstError::Compaction(
                "blocking spill cleanup lock unexpectedly returned unlocked".to_string(),
            ));
        }
        let reclaimed_owner_directories = cleanup_stale_owner_directories(&root)?;
        let system = sysinfo::System::new_all();
        let process_id = std::process::id();
        let process_start_time_seconds = system
            .process(sysinfo::Pid::from_u32(process_id))
            .map(sysinfo::Process::start_time)
            .ok_or_else(|| {
                SstError::Compaction(
                    "cannot resolve current process start time for spill ownership".to_string(),
                )
            })?;
        let owner_id = uuid::Uuid::new_v4().simple().to_string();
        let path = root.join(format!(
            "{OWNER_DIRECTORY_PREFIX}{process_id}-{process_start_time_seconds}-{owner_id}"
        ));
        std::fs::create_dir(&path)?;
        let initialize = || -> Result<File> {
            let lease = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(path.join(".lease"))?;
            if !lock_file_exclusive(&lease, true)? {
                return Err(SstError::Compaction(
                    "blocking spill owner lease unexpectedly returned unlocked".to_string(),
                ));
            }
            write_json_atomic(
                &path.join("owner.json"),
                &SpillOwnerManifest {
                    format_version: TASK_MANIFEST_VERSION,
                    process_id,
                    process_start_time_seconds,
                    owner_id,
                },
            )?;
            Ok(lease)
        };
        let lease = match initialize() {
            Ok(lease) => lease,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&path);
                return Err(error);
            }
        };
        let inner = std::sync::Arc::new(SpillScratchOwnerInner {
            path,
            _lease: lease,
        });
        registry.insert(root, std::sync::Arc::downgrade(&inner));
        Ok(Self {
            inner,
            reclaimed_owner_directories,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.inner.path
    }

    pub(crate) fn reclaimed_owner_directories(&self) -> usize {
        self.reclaimed_owner_directories
    }
}

fn spill_owner_registry()
-> &'static std::sync::Mutex<HashMap<PathBuf, std::sync::Weak<SpillScratchOwnerInner>>> {
    static REGISTRY: std::sync::OnceLock<
        std::sync::Mutex<HashMap<PathBuf, std::sync::Weak<SpillScratchOwnerInner>>>,
    > = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn cleanup_stale_owner_directories(root: &Path) -> Result<usize> {
    let mut reclaimed = 0usize;
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !name.starts_with(OWNER_DIRECTORY_PREFIX) {
            continue;
        }
        let lease_path = entry.path().join(".lease");
        let lease = match OpenOptions::new().read(true).write(true).open(&lease_path) {
            Ok(lease) => Some(lease),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if let Some(lease) = &lease
            && !lock_file_exclusive(lease, false)?
        {
            continue;
        }
        std::fs::remove_dir_all(entry.path())?;
        reclaimed = reclaimed.saturating_add(1);
    }
    Ok(reclaimed)
}

#[cfg(unix)]
fn lock_file_exclusive(file: &File, blocking: bool) -> Result<bool> {
    use std::os::fd::AsRawFd;

    let operation = if blocking {
        libc::LOCK_EX
    } else {
        libc::LOCK_EX | libc::LOCK_NB
    };
    // SAFETY: `file` owns a valid descriptor for this call's duration. `flock`
    // does not retain pointers or mutate Rust-managed memory.
    let result = unsafe { libc::flock(file.as_raw_fd(), operation) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if !blocking && error.kind() == std::io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(error.into())
    }
}

#[cfg(not(unix))]
fn lock_file_exclusive(_file: &File, _blocking: bool) -> Result<bool> {
    Err(SstError::Compaction(
        "local-spill ownership requires advisory file locks on this platform".to_string(),
    ))
}

#[derive(Debug, Serialize)]
struct SpillTaskSource {
    path: String,
    size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct SpillTaskManifest {
    format_version: u16,
    collection_object_id: u64,
    source_level: u8,
    sources: Vec<SpillTaskSource>,
    declared_input_bytes: u64,
    output_path: String,
    reserved_memory_bytes: u64,
    reserved_scratch_bytes: u64,
    phase: String,
}

/// Deterministic task scratch namespace plus its auditable phase manifest.
/// Drop reclaims the complete subtree; process-owner recovery handles crashes.
#[derive(Debug)]
pub(crate) struct SpillTaskScratch {
    path: PathBuf,
    manifest: SpillTaskManifest,
}

impl SpillTaskScratch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        owner_root: &Path,
        collection_object_id: u64,
        source_level: u8,
        input_paths: &[String],
        declared_input_bytes: u64,
        output_path: &str,
        reserved_memory_bytes: u64,
        reserved_scratch_bytes: u64,
    ) -> Result<Self> {
        let fingerprint =
            spill_task_fingerprint(collection_object_id, source_level, input_paths, output_path);
        let path = owner_root.join(format!(
            "task-{collection_object_id}-{source_level}-{fingerprint:016x}"
        ));
        std::fs::create_dir(&path).map_err(|error| {
            SstError::Compaction(format!(
                "create deterministic spill task directory {}: {error}",
                path.display()
            ))
        })?;
        let task = Self {
            path,
            manifest: SpillTaskManifest {
                format_version: TASK_MANIFEST_VERSION,
                collection_object_id,
                source_level,
                sources: input_paths
                    .iter()
                    .cloned()
                    .map(|path| SpillTaskSource {
                        path,
                        size_bytes: None,
                    })
                    .collect(),
                declared_input_bytes,
                output_path: output_path.to_string(),
                reserved_memory_bytes,
                reserved_scratch_bytes,
                phase: "admitted".to_string(),
            },
        };
        if let Err(error) = task.persist_manifest() {
            let _ = std::fs::remove_dir_all(&task.path);
            return Err(error);
        }
        Ok(task)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn update_phase(&mut self, phase: &str, source_sizes: Option<&[u64]>) -> Result<()> {
        if let Some(source_sizes) = source_sizes {
            if source_sizes.len() != self.manifest.sources.len() {
                return Err(SstError::Compaction(format!(
                    "spill manifest has {} sources but {} observed sizes",
                    self.manifest.sources.len(),
                    source_sizes.len()
                )));
            }
            for (source, size) in self.manifest.sources.iter_mut().zip(source_sizes) {
                source.size_bytes = Some(*size);
            }
        }
        self.manifest.phase = phase.to_string();
        self.persist_manifest()
    }

    fn persist_manifest(&self) -> Result<()> {
        write_json_atomic(&self.path.join("task.json"), &self.manifest)
    }
}

impl Drop for SpillTaskScratch {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %self.path.display(),
                %error,
                "Failed to reclaim compaction spill task directory"
            );
        }
    }
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        SstError::Compaction(format!(
            "serialize spill manifest {}: {error}",
            path.display()
        ))
    })?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn spill_task_fingerprint(
    collection_object_id: u64,
    source_level: u8,
    input_paths: &[String],
    output_path: &str,
) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(PRIME);
    };
    update(&collection_object_id.to_le_bytes());
    update(&[source_level]);
    for path in input_paths {
        update(&(path.len() as u64).to_le_bytes());
        update(path.as_bytes());
    }
    update(&(output_path.len() as u64).to_le_bytes());
    update(output_path.as_bytes());
    hash
}

/// A canonical record plus its stable position in the compaction input stream.
///
/// `ProximaRecord::schema_version` is intentionally skipped by serde because
/// the durable WAL has its own version envelope. Scratch runs are ephemeral,
/// but must preserve precision-aware records within a running process, so this
/// wrapper carries the field explicitly and restores it after decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SpillRecord {
    source_ordinal: u64,
    schema_version: u8,
    record: ProximaRecord,
}

impl SpillRecord {
    pub(crate) fn new(source_ordinal: u64, record: ProximaRecord) -> Self {
        Self {
            source_ordinal,
            schema_version: record.schema_version,
            record,
        }
    }

    fn restore_schema_version(mut self) -> Self {
        self.record.schema_version = self.schema_version;
        self
    }

    pub(crate) fn into_record(self) -> ProximaRecord {
        self.record
    }
}

/// Auditable resource-shape statistics for one external MVCC pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExternalMvccStats {
    pub(crate) input_records: u64,
    pub(crate) output_records: u64,
    pub(crate) initial_run_count: usize,
    pub(crate) merge_pass_count: usize,
    pub(crate) max_open_runs: usize,
    pub(crate) peak_run_buffer_bytes: u64,
    pub(crate) scratch_bytes_written: u64,
}

/// Owns the task scratch directory. Dropping this value reclaims every run,
/// including after a downstream error or cancellation.
pub(crate) struct ExternalMvccOutput {
    task_directory: tempfile::TempDir,
    output_path: PathBuf,
    stats: ExternalMvccStats,
}

impl ExternalMvccOutput {
    pub(crate) fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub(crate) fn stats(&self) -> &ExternalMvccStats {
        &self.stats
    }

    pub(crate) fn for_each_record(
        &self,
        mut visitor: impl FnMut(SpillRecord) -> Result<()>,
    ) -> Result<()> {
        let mut reader = FramedRunReader::open(&self.output_path)?;
        while let Some(record) = reader.read_next()? {
            visitor(record)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn task_path(&self) -> &Path {
        self.task_directory.path()
    }

    #[cfg(test)]
    fn read_records(&self) -> Result<Vec<SpillRecord>> {
        let mut reader = FramedRunReader::open(&self.output_path)?;
        let mut records = Vec::new();
        while let Some(record) = reader.read_next()? {
            records.push(record);
        }
        Ok(records)
    }
}

/// Resolve an input stream through deterministic bounded external runs.
///
/// `max_run_buffer_bytes` bounds each in-memory sort batch by a conservative
/// serialized-size estimate. `max_merge_fan_in` bounds both open file handles
/// and one-record-per-run merge memory. When there are more runs, sorted
/// intermediate passes collapse them before the final MVCC pass.
pub(crate) fn resolve_external_mvcc<I>(
    scratch_root: &Path,
    records: I,
    max_run_buffer_bytes: u64,
    max_merge_fan_in: usize,
    current_timestamp_ns: i64,
) -> Result<ExternalMvccOutput>
where
    I: IntoIterator<Item = Result<SpillRecord>>,
{
    let mut builder = ExternalMvccBuilder::new(
        scratch_root,
        max_run_buffer_bytes,
        max_merge_fan_in,
        current_timestamp_ns,
    )?;
    for record in records {
        builder.push(record?)?;
    }
    builder.finish()
}

/// Incremental front-end for the external MVCC pipeline. Async ranged readers
/// push one decoded block at a time without first materializing the corpus.
pub(crate) struct ExternalMvccBuilder {
    task_directory: tempfile::TempDir,
    max_run_buffer_bytes: u64,
    max_merge_fan_in: usize,
    current_timestamp_ns: i64,
    stats: ExternalMvccStats,
    runs: Vec<PathBuf>,
    buffer: Vec<SpillRecord>,
    buffered_bytes: u64,
}

impl ExternalMvccBuilder {
    pub(crate) fn new(
        scratch_root: &Path,
        max_run_buffer_bytes: u64,
        max_merge_fan_in: usize,
        current_timestamp_ns: i64,
    ) -> Result<Self> {
        if max_run_buffer_bytes == 0 {
            return Err(SstError::InvalidArgument(
                "spill MVCC run buffer must be greater than zero".to_string(),
            ));
        }
        if max_merge_fan_in < 2 {
            return Err(SstError::InvalidArgument(
                "spill MVCC merge fan-in must be at least two".to_string(),
            ));
        }

        Ok(Self {
            task_directory: tempfile::Builder::new()
                .prefix("proximadb-compaction-spill-")
                .tempdir_in(scratch_root)?,
            max_run_buffer_bytes,
            max_merge_fan_in,
            current_timestamp_ns,
            stats: ExternalMvccStats::default(),
            runs: Vec::new(),
            buffer: Vec::new(),
            buffered_bytes: 0,
        })
    }

    pub(crate) fn push(&mut self, record: SpillRecord) -> Result<()> {
        let record_bytes = estimated_buffer_bytes(&record)?;
        if !self.buffer.is_empty()
            && self.buffered_bytes.saturating_add(record_bytes) > self.max_run_buffer_bytes
        {
            self.flush_run()?;
        }
        self.buffered_bytes = self.buffered_bytes.saturating_add(record_bytes);
        self.stats.peak_run_buffer_bytes =
            self.stats.peak_run_buffer_bytes.max(self.buffered_bytes);
        self.stats.input_records = self.stats.input_records.saturating_add(1);
        self.buffer.push(record);
        Ok(())
    }

    fn flush_run(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = self
            .task_directory
            .path()
            .join(format!("oid-pass-0000-run-{:06}.pxrun", self.runs.len()));
        self.stats.scratch_bytes_written = self
            .stats
            .scratch_bytes_written
            .saturating_add(write_sorted_run(&path, &mut self.buffer)?);
        self.runs.push(path);
        self.buffered_bytes = 0;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<ExternalMvccOutput> {
        self.flush_run()?;
        self.stats.initial_run_count = self.runs.len();

        let mut pass = 0usize;
        while self.runs.len() > self.max_merge_fan_in {
            pass = pass.saturating_add(1);
            let mut next_runs = Vec::with_capacity(self.runs.len().div_ceil(self.max_merge_fan_in));
            for (group_index, group) in self.runs.chunks(self.max_merge_fan_in).enumerate() {
                let output = self
                    .task_directory
                    .path()
                    .join(format!("oid-pass-{pass:04}-run-{group_index:06}.pxrun"));
                self.stats.max_open_runs = self.stats.max_open_runs.max(group.len());
                self.stats.scratch_bytes_written = self
                    .stats
                    .scratch_bytes_written
                    .saturating_add(merge_sorted_runs(group, &output)?);
                next_runs.push(output);
            }
            for old_run in &self.runs {
                std::fs::remove_file(old_run)?;
            }
            self.runs = next_runs;
        }
        self.stats.merge_pass_count = pass;

        let output_path = self.task_directory.path().join("mvcc-winners.pxrun");
        self.stats.max_open_runs = self.stats.max_open_runs.max(self.runs.len());
        let (bytes_written, output_records) =
            write_mvcc_winners(&self.runs, &output_path, self.current_timestamp_ns)?;
        self.stats.scratch_bytes_written = self
            .stats
            .scratch_bytes_written
            .saturating_add(bytes_written);
        self.stats.output_records = output_records;
        for old_run in &self.runs {
            std::fs::remove_file(old_run)?;
        }

        Ok(ExternalMvccOutput {
            task_directory: self.task_directory,
            output_path,
            stats: self.stats,
        })
    }
}

/// Read port used by spill compaction. Keeping it smaller than `FileSystem`
/// makes the one-block-at-a-time contract directly testable with an in-memory
/// range source while production delegates to the scheme-aware factory.
#[async_trait]
pub(crate) trait CompactionRangeSource: Send + Sync {
    async fn size(&self, path: &str) -> Result<u64>;
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> Result<Vec<u8>>;
    async fn read_all(&self, path: &str) -> Result<Vec<u8>>;
}

#[async_trait]
impl CompactionRangeSource for FilesystemFactory {
    async fn size(&self, path: &str) -> Result<u64> {
        Ok(self.metadata(path).await?.size)
    }

    async fn read_range(&self, path: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        Ok(FilesystemFactory::read_range(self, path, offset, length).await?)
    }

    async fn read_all(&self, path: &str) -> Result<Vec<u8>> {
        Ok(FilesystemFactory::read(self, path).await?)
    }
}

/// Physical shape of the compaction input pass. `largest_range_bytes` is the
/// concrete assertion that coalesced inputs were never whole-object decoded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RangedInputStats {
    pub(crate) input_files: usize,
    pub(crate) source_sizes: Vec<u64>,
    pub(crate) coalesced_files: usize,
    pub(crate) legacy_whole_file_fallbacks: usize,
    pub(crate) block_range_reads: u64,
    pub(crate) sq8_range_reads: u64,
    pub(crate) largest_range_bytes: u64,
}

impl RangedInputStats {
    fn observe_range(&mut self, length: u64) {
        self.largest_range_bytes = self.largest_range_bytes.max(length);
    }
}

/// Range-decode immutable inputs into external MVCC runs.
///
/// Coalesced PAX reads its header and footer once, then alternates one Region-D
/// block with only that block's fixed-stride Region-B SQ8 rows. Legacy inputs
/// remain mixed-read-safe through the canonical whole-file router, but only one
/// legacy file is resident at a time. This fallback is observable and is not a
/// claim of bounded memory for obsolete formats.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn resolve_external_mvcc_from_segments(
    source: &dyn CompactionRangeSource,
    input_paths: &[String],
    scratch_root: &Path,
    max_run_buffer_bytes: u64,
    max_merge_fan_in: usize,
    current_timestamp_ns: i64,
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
    deleted_positions: &[HashSet<u32>],
) -> Result<(ExternalMvccOutput, RangedInputStats)> {
    let mut builder = ExternalMvccBuilder::new(
        scratch_root,
        max_run_buffer_bytes,
        max_merge_fan_in,
        current_timestamp_ns,
    )?;
    let mut ordinal = 0u64;
    let mut stats = RangedInputStats {
        input_files: input_paths.len(),
        ..RangedInputStats::default()
    };

    for (input_index, path) in input_paths.iter().enumerate() {
        let deleted_rows = deleted_positions.get(input_index);
        let size = source.size(path).await?;
        stats.source_sizes.push(size);
        // TD-PAXRG-1 collapse: the single 88 B header form — fetch the full
        // prefix (a 72-byte probe predates the collapse and now fails parse).
        let prefix_len = size.min(SEG_HEADER_PREFIX_LEN as u64);
        let prefix = source.read_range(path, 0, prefix_len).await?;
        stats.observe_range(prefix_len);

        match SegmentHeaderPrefix::parse(&prefix) {
            Ok(header) => {
                read_coalesced_segment_into_builder(
                    source,
                    path,
                    size,
                    &header,
                    embedding_model_ids,
                    user_column_keys,
                    tenant_ctx,
                    deleted_rows,
                    &mut ordinal,
                    &mut builder,
                    &mut stats,
                )
                .await?;
                stats.coalesced_files = stats.coalesced_files.saturating_add(1);
            }
            Err(_) => {
                let bytes = source.read_all(path).await?;
                let records = crate::storage::engines::sst::segment_format::read_segment_records(
                    &bytes,
                    embedding_model_ids,
                    user_column_keys,
                    tenant_ctx,
                )?;
                stats.legacy_whole_file_fallbacks =
                    stats.legacy_whole_file_fallbacks.saturating_add(1);
                for (row, record) in records.into_iter().enumerate() {
                    if u32::try_from(row).ok().is_some_and(|position| {
                        deleted_rows.is_some_and(|set| set.contains(&position))
                    }) {
                        continue;
                    }
                    builder.push(SpillRecord::new(ordinal, record))?;
                    ordinal = ordinal.saturating_add(1);
                }
            }
        }
    }

    Ok((builder.finish()?, stats))
}

#[allow(clippy::too_many_arguments)]
async fn read_coalesced_segment_into_builder(
    source: &dyn CompactionRangeSource,
    path: &str,
    segment_size: u64,
    header: &SegmentHeaderPrefix,
    embedding_model_ids: &[String],
    user_column_keys: &[String],
    tenant_ctx: Option<&str>,
    deleted_rows: Option<&HashSet<u32>>,
    ordinal: &mut u64,
    builder: &mut ExternalMvccBuilder,
    stats: &mut RangedInputStats,
) -> Result<()> {
    validate_extent("footer", header.footer_off, header.footer_len, segment_size)?;
    let footer_bytes = source
        .read_range(path, header.footer_off, header.footer_len)
        .await?;
    stats.observe_range(header.footer_len);
    let footer = SegmentFooterIndex::parse(&footer_bytes)?;
    if footer.rabitq_off != header.rabitq_off
        || footer.rabitq_len != header.rabitq_len
        || footer.sq8_off != header.sq8_off
        || footer.sq8_len != header.sq8_len
    {
        return Err(SstError::Compaction(format!(
            "spill input {path} header/footer region extents disagree"
        )));
    }
    if footer.has_f32_tier {
        return Err(SstError::Compaction(format!(
            "spill input {path} has an exact-f32 tier; bounded exact-tier projection is required before spill"
        )));
    }
    if footer.embed_count > 1 {
        return Err(SstError::Compaction(format!(
            "spill input {path} has {} embedding columns; bounded multi-embedding projection is not implemented",
            footer.embed_count
        )));
    }

    let (validity, codes_base, sq8_params) = if footer.sq8_len > 0 {
        validate_extent("SQ8", footer.sq8_off, footer.sq8_len, segment_size)?;
        let row_count = usize::try_from(footer.row_count).map_err(|_| {
            SstError::Compaction(format!("spill input {path} row count exceeds usize"))
        })?;
        let dim = usize::try_from(footer.embed_dim).map_err(|_| {
            SstError::Compaction(format!("spill input {path} dimension exceeds usize"))
        })?;
        let bitmap_len = proximadb_block_format::coalesced_sq8::bitmap_len(row_count);
        let codes_offset = sq8_codes_offset(row_count);
        let expected_len = codes_offset
            .checked_add(row_count.checked_mul(dim).ok_or_else(|| {
                SstError::Compaction(format!("spill input {path} SQ8 size overflows usize"))
            })?)
            .ok_or_else(|| {
                SstError::Compaction(format!("spill input {path} SQ8 extent overflows usize"))
            })?;
        if expected_len as u64 > footer.sq8_len {
            return Err(SstError::Compaction(format!(
                "spill input {path} SQ8 region is truncated: expected {expected_len}, got {}",
                footer.sq8_len
            )));
        }
        let bitmap_offset = footer
            .sq8_off
            .saturating_add(sq8_region_header_len() as u64);
        let validity = source
            .read_range(path, bitmap_offset, bitmap_len as u64)
            .await?;
        stats.observe_range(bitmap_len as u64);
        (
            validity,
            footer.sq8_off.saturating_add(codes_offset as u64),
            Some(
                proximadb_block_format::coalesced_sq8::params_from_min_scale(
                    footer.sq8_min,
                    footer.sq8_scale,
                ),
            ),
        )
    } else {
        (Vec::new(), 0, None)
    };

    let mut global_row = 0usize;
    for block in &footer.blocks {
        validate_extent(
            "Region D block",
            block.offset,
            block.size as u64,
            segment_size,
        )?;
        let block_bytes = source
            .read_range(path, block.offset, block.size as u64)
            .await?;
        stats.block_range_reads = stats.block_range_reads.saturating_add(1);
        stats.observe_range(block.size as u64);
        let reader = PaxBlockReader::open(&block_bytes)?;
        let mut records = FlatRow::from_block_reader(&reader)?
            .into_iter()
            .map(|flat| flat.into_record(embedding_model_ids, user_column_keys, tenant_ctx))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if records.len() != block.row_count as usize {
            return Err(SstError::Compaction(format!(
                "spill input {path} block row count mismatch: footer={}, decoded={}",
                block.row_count,
                records.len()
            )));
        }

        if let Some(params) = &sq8_params {
            let dim = footer.embed_dim as usize;
            let code_len = records.len().checked_mul(dim).ok_or_else(|| {
                SstError::Compaction(format!("spill input {path} block SQ8 size overflows"))
            })?;
            let code_offset = codes_base
                .saturating_add((global_row as u64).saturating_mul(footer.embed_dim as u64));
            let codes = source
                .read_range(path, code_offset, code_len as u64)
                .await?;
            stats.sq8_range_reads = stats.sq8_range_reads.saturating_add(1);
            stats.observe_range(code_len as u64);
            if codes.len() != code_len {
                return Err(SstError::Compaction(format!(
                    "spill input {path} returned a short SQ8 block range"
                )));
            }
            for (row_in_block, record) in records.iter_mut().enumerate() {
                let row = global_row.saturating_add(row_in_block);
                let present = validity
                    .get(row >> 3)
                    .is_some_and(|byte| (byte >> (row & 7)) & 1 == 1);
                if present && record.embeddings.is_empty() {
                    let start = row_in_block.saturating_mul(dim);
                    let end = start.saturating_add(dim);
                    let vector = sq8_decode_codes(&codes[start..end], params);
                    record.embeddings.push(EmbeddingCell::new_fp32(
                        "",
                        "dense",
                        footer.embed_dim,
                        vector,
                    ));
                }
            }
        }

        for (row_in_block, record) in records.into_iter().enumerate() {
            let source_row = global_row.saturating_add(row_in_block);
            if u32::try_from(source_row)
                .ok()
                .is_some_and(|position| deleted_rows.is_some_and(|set| set.contains(&position)))
            {
                continue;
            }
            builder.push(SpillRecord::new(*ordinal, record))?;
            *ordinal = (*ordinal).saturating_add(1);
        }
        global_row = global_row.saturating_add(block.row_count as usize);
    }
    if global_row as u64 != footer.row_count {
        return Err(SstError::Compaction(format!(
            "spill input {path} footer rows={} but block table rows={global_row}",
            footer.row_count
        )));
    }
    Ok(())
}

fn validate_extent(name: &str, offset: u64, length: u64, segment_size: u64) -> Result<()> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| SstError::Compaction(format!("spill input {name} extent overflows u64")))?;
    if end > segment_size {
        return Err(SstError::Compaction(format!(
            "spill input {name} extent {offset}..{end} exceeds segment size {segment_size}"
        )));
    }
    Ok(())
}

fn estimated_buffer_bytes(record: &SpillRecord) -> Result<u64> {
    let serialized = bincode::serialized_size(record)?;
    // The decoded form owns strings, vectors, and property nodes in addition
    // to the serialized payload. Two payload widths plus the root struct is a
    // conservative, deterministic accounting unit for run admission.
    Ok(serialized
        .saturating_mul(2)
        .saturating_add(std::mem::size_of::<SpillRecord>() as u64))
}

fn spill_order(left: &SpillRecord, right: &SpillRecord) -> Ordering {
    left.record
        .oid
        .cmp(&right.record.oid)
        .then_with(|| effective_version(&left.record).cmp(&effective_version(&right.record)))
        .then_with(|| left.record.created_at_ns.cmp(&right.record.created_at_ns))
        .then_with(|| left.record.updated_at_ns.cmp(&right.record.updated_at_ns))
        .then_with(|| left.source_ordinal.cmp(&right.source_ordinal))
}

fn write_sorted_run(path: &Path, records: &mut Vec<SpillRecord>) -> Result<u64> {
    records.sort_unstable_by(spill_order);
    let mut writer = FramedRunWriter::create(path)?;
    for record in records.drain(..) {
        writer.write_record(&record)?;
    }
    writer.finish()
}

#[derive(Debug)]
struct HeapEntry {
    run_index: usize,
    record: SpillRecord,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.run_index == other.run_index && spill_order(&self.record, &other.record).is_eq()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse the record ordering so the global
        // minimum is popped first. Run index is a deterministic final tie.
        spill_order(&other.record, &self.record).then_with(|| other.run_index.cmp(&self.run_index))
    }
}

fn open_merge_heap(paths: &[PathBuf]) -> Result<(Vec<FramedRunReader>, BinaryHeap<HeapEntry>)> {
    let mut readers = Vec::with_capacity(paths.len());
    let mut heap = BinaryHeap::with_capacity(paths.len());
    for (run_index, path) in paths.iter().enumerate() {
        let mut reader = FramedRunReader::open(path)?;
        if let Some(record) = reader.read_next()? {
            heap.push(HeapEntry { run_index, record });
        }
        readers.push(reader);
    }
    Ok((readers, heap))
}

fn advance_run(
    readers: &mut [FramedRunReader],
    heap: &mut BinaryHeap<HeapEntry>,
    run_index: usize,
) -> Result<()> {
    if let Some(record) = readers[run_index].read_next()? {
        heap.push(HeapEntry { run_index, record });
    }
    Ok(())
}

fn merge_sorted_runs(paths: &[PathBuf], output: &Path) -> Result<u64> {
    let (mut readers, mut heap) = open_merge_heap(paths)?;
    let mut writer = FramedRunWriter::create(output)?;
    while let Some(entry) = heap.pop() {
        let run_index = entry.run_index;
        writer.write_record(&entry.record)?;
        advance_run(&mut readers, &mut heap, run_index)?;
    }
    writer.finish()
}

struct VersionGroup {
    oid: String,
    expected_version: u64,
    blocked_by_gap: bool,
    deleted: bool,
    winner: Option<SpillRecord>,
}

impl VersionGroup {
    fn begin(record: &SpillRecord) -> Self {
        let start = effective_version(&record.record);
        Self {
            oid: record.record.oid.clone(),
            expected_version: start,
            blocked_by_gap: start > 1,
            deleted: false,
            winner: None,
        }
    }

    fn push(&mut self, record: SpillRecord, current_timestamp_ns: i64) {
        if is_expired(&record.record, current_timestamp_ns) {
            self.deleted = true;
            self.winner = None;
            return;
        }
        if self.deleted || self.blocked_by_gap {
            return;
        }
        let version = effective_version(&record.record);
        if version == self.expected_version {
            self.winner = Some(record);
            self.expected_version = self.expected_version.saturating_add(1);
        } else if version > self.expected_version {
            // Match MvccResolver: retain the last contiguous winner and ignore
            // every version after the first gap.
            self.blocked_by_gap = true;
        }
        // version < expected is a later duplicate of an already selected
        // version, ordered after it by timestamp/ordinal, so it is ignored.
    }

    fn finish(self) -> Option<SpillRecord> {
        if self.deleted { None } else { self.winner }
    }
}

fn write_mvcc_winners(
    paths: &[PathBuf],
    output: &Path,
    current_timestamp_ns: i64,
) -> Result<(u64, u64)> {
    let (mut readers, mut heap) = open_merge_heap(paths)?;
    let mut writer = FramedRunWriter::create(output)?;
    let mut group: Option<VersionGroup> = None;
    let mut output_records = 0u64;

    while let Some(entry) = heap.pop() {
        let run_index = entry.run_index;
        let record = entry.record;
        let is_new_group = group
            .as_ref()
            .is_some_and(|current| current.oid != record.record.oid);
        if is_new_group && let Some(winner) = group.take().and_then(VersionGroup::finish) {
            writer.write_record(&winner)?;
            output_records = output_records.saturating_add(1);
        }

        if is_append_only_oid(&record.record.oid) {
            if !is_expired(&record.record, current_timestamp_ns) {
                writer.write_record(&record)?;
                output_records = output_records.saturating_add(1);
            }
        } else {
            if group.is_none() {
                group = Some(VersionGroup::begin(&record));
            }
            if let Some(current) = group.as_mut() {
                current.push(record, current_timestamp_ns);
            }
        }
        advance_run(&mut readers, &mut heap, run_index)?;
    }

    if let Some(winner) = group.and_then(VersionGroup::finish) {
        writer.write_record(&winner)?;
        output_records = output_records.saturating_add(1);
    }
    Ok((writer.finish()?, output_records))
}

fn is_expired(record: &ProximaRecord, current_timestamp_ns: i64) -> bool {
    record
        .valid_to_ns
        .is_some_and(|valid_to| valid_to < current_timestamp_ns)
}

/// Resource and physical-shape evidence for the external IVF ordering pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExternalClusterStats {
    pub(crate) input_records: u64,
    pub(crate) usable_records: u64,
    pub(crate) training_records: u64,
    pub(crate) cell_count: usize,
    pub(crate) initial_run_count: usize,
    pub(crate) merge_pass_count: usize,
    pub(crate) max_open_runs: usize,
    pub(crate) peak_run_buffer_bytes: u64,
    pub(crate) scratch_bytes_written: u64,
    pub(crate) estimated_training_peak_bytes: u64,
}

/// Owns cell-ordered winners and their persisted A0 model. The scratch tree is
/// reclaimed after the PAX writer has consumed the run or on any later error.
#[derive(Debug)]
pub(crate) struct ExternalClusterOutput {
    task_directory: tempfile::TempDir,
    output_path: PathBuf,
    model: proximadb_storage_common::coarse_directory::CoarseModel,
    stats: ExternalClusterStats,
}

impl ExternalClusterOutput {
    pub(crate) fn model(&self) -> &proximadb_storage_common::coarse_directory::CoarseModel {
        &self.model
    }

    pub(crate) fn stats(&self) -> &ExternalClusterStats {
        &self.stats
    }

    pub(crate) fn for_each_record(
        &self,
        mut visitor: impl FnMut(ProximaRecord) -> Result<()>,
    ) -> Result<()> {
        let mut reader = FramedRunReader::open(&self.output_path)?;
        while let Some(record) = reader.read_value::<CellSpillRecord>()? {
            visitor(record.restore_schema_version().spill.record)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn task_path(&self) -> &Path {
        self.task_directory.path()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CellSpillRecord {
    cell_rank: u32,
    pc1: f32,
    spill: SpillRecord,
}

impl CellSpillRecord {
    fn restore_schema_version(mut self) -> Self {
        self.spill = self.spill.restore_schema_version();
        self
    }
}

fn cell_order(left: &CellSpillRecord, right: &CellSpillRecord) -> Ordering {
    left.cell_rank
        .cmp(&right.cell_rank)
        .then_with(|| left.pc1.total_cmp(&right.pc1))
        .then_with(|| left.spill.source_ordinal.cmp(&right.spill.source_ordinal))
}

fn first_f32_embedding(record: &ProximaRecord) -> Option<&[f32]> {
    match record.embeddings.first().map(|embedding| &embedding.values) {
        Some(EmbeddingValues::Fp32(vector)) if !vector.is_empty() => Some(vector),
        _ => None,
    }
}

/// Conservative peak for the existing full covariance PCA plus low-dimensional
/// sample/k-means buffers. This is checked before `IncrementalPCA::new`, so an
/// oversized dimension fails admission rather than first allocating memory.
fn estimated_ivf_training_peak_bytes(
    dim: usize,
    shape: crate::storage::engines::sst::block_cluster::IvfTrainingShape,
) -> Option<u64> {
    let dim = u64::try_from(dim).ok()?;
    let components = u64::try_from(shape.n_components).ok()?;
    let samples = u64::try_from(shape.train_sample).ok()?;
    let cells = u64::try_from(shape.k).ok()?;
    let covariance_peak = dim.checked_mul(dim)?.checked_mul(16)?;
    let component_bytes = components.checked_mul(dim)?.checked_mul(8)?;
    let sample_coordinates = samples.checked_mul(components)?.checked_mul(4)?;
    let kmeans_working = samples
        .checked_mul(12)?
        .checked_add(cells.checked_mul(components)?.checked_mul(16)?)?;
    covariance_peak
        .checked_add(component_bytes)?
        .checked_add(sample_coordinates)?
        .checked_add(kmeans_working)?
        .checked_add(8 * 1024 * 1024)
}

/// Train persisted-f32 PCA/IVF through repeatable scans of the MVCC winner run,
/// then externally order every record by `(Hilbert cell, PC1, source ordinal)`.
/// Raw high-dimensional samples and the full coordinate/assignment arrays are
/// never retained. The configured working-memory bound is enforced before PCA.
pub(crate) fn cluster_external_mvcc(
    winners: ExternalMvccOutput,
    scratch_root: &Path,
    max_run_buffer_bytes: u64,
    max_merge_fan_in: usize,
    max_working_memory_bytes: u64,
    destination_url: Option<&str>,
) -> Result<ExternalClusterOutput> {
    use crate::storage::engines::core::formats::proximablocks::spatial_clustering::IncrementalPCA;
    use crate::storage::engines::sst::block_cluster::{
        IvfPcaProjection, finish_ivf_probe_classifier, ivf_training_shape,
    };

    let mut dim = None;
    let mut usable_rows = 0usize;
    winners.for_each_record(|item| {
        if let Some(vector) = first_f32_embedding(&item.record) {
            if let Some(expected) = dim {
                if vector.len() != expected {
                    return Err(SstError::Compaction(format!(
                        "spill IVF dimension {} differs from established dimension {expected}",
                        vector.len()
                    )));
                }
            } else {
                dim = Some(vector.len());
            }
            usable_rows = usable_rows.checked_add(1).ok_or_else(|| {
                SstError::Compaction("spill IVF usable row count exceeds usize".to_string())
            })?;
        }
        Ok(())
    })?;
    let dim = dim.ok_or_else(|| {
        SstError::Compaction("spill IVF requires at least one f32 embedding".to_string())
    })?;
    let shape = ivf_training_shape(usable_rows, dim, destination_url).ok_or_else(|| {
        SstError::Compaction(format!(
            "spill IVF requires at least 64 usable rows; found {usable_rows}"
        ))
    })?;
    let estimated_training_peak_bytes =
        estimated_ivf_training_peak_bytes(dim, shape).ok_or_else(|| {
            SstError::Compaction("spill IVF memory estimate overflows u64".to_string())
        })?;
    // TD-PAXDIR-1 D3: classify with the collection's shared model when the
    // compaction set one and no retrain trigger fires. No training happens
    // then, so the training-memory bound applies only to the training arm.
    let (classifier, frozen_radii_sq, trained_on) = match shared_spill_classifier(
        &winners,
        dim,
        usable_rows,
        destination_url,
    )? {
        Some(shared) => shared,
        None => {
            if estimated_training_peak_bytes > max_working_memory_bytes {
                return Err(SstError::Compaction(format!(
                    "spill IVF training needs an estimated {estimated_training_peak_bytes} bytes, above the admitted {max_working_memory_bytes} byte working set"
                )));
            }
            let mut pca = IncrementalPCA::new(dim, shape.n_components);
            let mut usable_index = 0usize;
            winners.for_each_record(|item| {
                if let Some(vector) = first_f32_embedding(&item.record) {
                    if usable_index.is_multiple_of(shape.sample_step) {
                        pca.add_sample(vector);
                    }
                    usable_index = usable_index.saturating_add(1);
                }
                Ok(())
            })?;
            pca.finalize();
            let projection = IvfPcaProjection::from_finalized(&pca).ok_or_else(|| {
                SstError::Compaction("spill IVF PCA did not produce a projection".to_string())
            })?;

            let mut sample_coords = Vec::with_capacity(shape.train_sample);
            usable_index = 0;
            winners.for_each_record(|item| {
                if let Some(vector) = first_f32_embedding(&item.record) {
                    if usable_index.is_multiple_of(shape.sample_step) {
                        sample_coords.push(projection.project(vector));
                    }
                    usable_index = usable_index.saturating_add(1);
                }
                Ok(())
            })?;
            let classifier = finish_ivf_probe_classifier(projection, shape, &sample_coords)
                .ok_or_else(|| {
                    SstError::Compaction("spill IVF k-means training failed".to_string())
                })?;
            let trained_on = sample_coords.len() as u64;
            (classifier, None, trained_on)
        }
    };
    let cell_count = classifier.cell_count();
    let mut cell_rows = vec![0u64; cell_count];
    let mut radii_sq = vec![0f32; cell_count];
    let mut builder =
        ExternalCellBuilder::new(scratch_root, max_run_buffer_bytes, max_merge_fan_in)?;
    winners.for_each_record(|spill| {
        let (cell_rank, pc1) = if let Some(vector) = first_f32_embedding(&spill.record) {
            let assignment = classifier.classify(vector).ok_or_else(|| {
                SstError::Compaction("spill IVF could not classify a validated vector".to_string())
            })?;
            cell_rows[assignment.source_cell] = cell_rows[assignment.source_cell].saturating_add(1);
            radii_sq[assignment.source_cell] =
                radii_sq[assignment.source_cell].max(assignment.distance_sq);
            (
                u32::try_from(assignment.cell_rank).map_err(|_| {
                    SstError::Compaction("spill IVF cell rank exceeds u32".to_string())
                })?,
                assignment.pc1,
            )
        } else {
            (u32::MAX, 0.0)
        };
        builder.push(CellSpillRecord {
            cell_rank,
            pc1,
            spill,
        })
    })?;
    // TD-PAXDIR-1 D3: a shared-model output keeps the space's trained radii.
    let radii_sq = frozen_radii_sq.unwrap_or(radii_sq);
    let model = classifier
        .finish_model(&cell_rows, &radii_sq)
        .ok_or_else(|| {
            SstError::Compaction("spill IVF model dimensions disagree with assignments".to_string())
        })?;
    builder.finish(
        model,
        winners.stats.output_records,
        usable_rows as u64,
        trained_on,
        estimated_training_peak_bytes,
    )
}

/// TD-PAXDIR-1 D3: the collection's shared model as a spill classifier, or
/// `None` when none is set for this write or a retrain trigger fires.
/// Returns the classifier, the space's trained radii (squared) for the
/// output to keep, and 0 rows trained. The
/// dimension and cell-count checks run before any pass over the rows; drift
/// is then measured on a strided sample (one extra sequential pass over the
/// winner run, far cheaper than the PCA and k-means passes it replaces).
fn shared_spill_classifier(
    winners: &ExternalMvccOutput,
    dim: usize,
    usable_rows: usize,
    destination_url: Option<&str>,
) -> Result<
    Option<(
        crate::storage::engines::sst::block_cluster::IvfProbeClassifier,
        Option<Vec<f32>>,
        u64,
    )>,
> {
    use crate::storage::engines::sst::block_cluster::{
        IvfProbeClassifier, SHARED_MODEL_DRIFT_SAMPLE, SharedModelVerdict,
        current_collection_model, outside_trained_radius, shared_model_drift,
        shared_model_precheck,
    };
    let Some(space) = current_collection_model() else {
        return Ok(None);
    };
    let retrain = |verdict: SharedModelVerdict| {
        tracing::info!(
            ?verdict,
            rows = usable_rows,
            "PAX shared collection model retrained (spill)"
        );
    };
    let verdict = shared_model_precheck(&space, usable_rows, dim, destination_url);
    if verdict != SharedModelVerdict::Accept {
        retrain(verdict);
        return Ok(None);
    }
    let model = &space.model;
    let Some(classifier) = IvfProbeClassifier::from_collection_model(model) else {
        return Ok(None);
    };
    let step = usable_rows.div_ceil(SHARED_MODEL_DRIFT_SAMPLE).max(1);
    let (mut sampled, mut outside, mut usable_index) = (0usize, 0usize, 0usize);
    winners.for_each_record(|item| {
        if let Some(vector) = first_f32_embedding(&item.record) {
            if usable_index.is_multiple_of(step)
                && let Some(assignment) = classifier.classify(vector)
            {
                sampled += 1;
                outside += usize::from(outside_trained_radius(model, &assignment));
            }
            usable_index = usable_index.saturating_add(1);
        }
        Ok(())
    })?;
    let verdict = shared_model_drift(outside, sampled);
    if verdict != SharedModelVerdict::Accept {
        retrain(verdict);
        return Ok(None);
    }
    Ok(Some((
        classifier,
        Some(crate::storage::engines::sst::block_cluster::frozen_radii_sq(model)),
        0,
    )))
}

struct ExternalCellBuilder {
    task_directory: tempfile::TempDir,
    max_run_buffer_bytes: u64,
    max_merge_fan_in: usize,
    runs: Vec<PathBuf>,
    buffer: Vec<CellSpillRecord>,
    buffered_bytes: u64,
    stats: ExternalClusterStats,
}

impl ExternalCellBuilder {
    fn new(
        scratch_root: &Path,
        max_run_buffer_bytes: u64,
        max_merge_fan_in: usize,
    ) -> Result<Self> {
        if max_run_buffer_bytes == 0 || max_merge_fan_in < 2 {
            return Err(SstError::InvalidArgument(
                "spill IVF run buffer must be positive and fan-in at least two".to_string(),
            ));
        }
        Ok(Self {
            task_directory: tempfile::Builder::new()
                .prefix("proximadb-compaction-cells-")
                .tempdir_in(scratch_root)?,
            max_run_buffer_bytes,
            max_merge_fan_in,
            runs: Vec::new(),
            buffer: Vec::new(),
            buffered_bytes: 0,
            stats: ExternalClusterStats::default(),
        })
    }

    fn push(&mut self, record: CellSpillRecord) -> Result<()> {
        let record_bytes = estimated_buffer_bytes(&record.spill)?.saturating_add(32);
        if !self.buffer.is_empty()
            && self.buffered_bytes.saturating_add(record_bytes) > self.max_run_buffer_bytes
        {
            self.flush_run()?;
        }
        self.buffered_bytes = self.buffered_bytes.saturating_add(record_bytes);
        self.stats.peak_run_buffer_bytes =
            self.stats.peak_run_buffer_bytes.max(self.buffered_bytes);
        self.buffer.push(record);
        Ok(())
    }

    fn flush_run(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = self
            .task_directory
            .path()
            .join(format!("cell-pass-0000-run-{:06}.pxrun", self.runs.len()));
        self.stats.scratch_bytes_written = self
            .stats
            .scratch_bytes_written
            .saturating_add(write_sorted_cell_run(&path, &mut self.buffer)?);
        self.runs.push(path);
        self.buffered_bytes = 0;
        Ok(())
    }

    fn finish(
        mut self,
        model: proximadb_storage_common::coarse_directory::CoarseModel,
        input_records: u64,
        usable_records: u64,
        training_records: u64,
        estimated_training_peak_bytes: u64,
    ) -> Result<ExternalClusterOutput> {
        self.flush_run()?;
        self.stats.initial_run_count = self.runs.len();
        let mut pass = 0usize;
        while self.runs.len() > self.max_merge_fan_in {
            pass = pass.saturating_add(1);
            let mut next = Vec::with_capacity(self.runs.len().div_ceil(self.max_merge_fan_in));
            for (group_index, group) in self.runs.chunks(self.max_merge_fan_in).enumerate() {
                let path = self
                    .task_directory
                    .path()
                    .join(format!("cell-pass-{pass:04}-run-{group_index:06}.pxrun"));
                self.stats.max_open_runs = self.stats.max_open_runs.max(group.len());
                self.stats.scratch_bytes_written = self
                    .stats
                    .scratch_bytes_written
                    .saturating_add(merge_cell_runs(group, &path)?);
                next.push(path);
            }
            for path in &self.runs {
                std::fs::remove_file(path)?;
            }
            self.runs = next;
        }
        self.stats.merge_pass_count = pass;
        self.stats.max_open_runs = self.stats.max_open_runs.max(self.runs.len());
        let output_path = self.task_directory.path().join("cell-ordered.pxrun");
        self.stats.scratch_bytes_written = self
            .stats
            .scratch_bytes_written
            .saturating_add(merge_cell_runs(&self.runs, &output_path)?);
        for path in &self.runs {
            std::fs::remove_file(path)?;
        }
        self.stats.input_records = input_records;
        self.stats.usable_records = usable_records;
        self.stats.training_records = training_records;
        self.stats.cell_count = model.cell_rows.len();
        self.stats.estimated_training_peak_bytes = estimated_training_peak_bytes;
        Ok(ExternalClusterOutput {
            task_directory: self.task_directory,
            output_path,
            model,
            stats: self.stats,
        })
    }
}

fn write_sorted_cell_run(path: &Path, records: &mut Vec<CellSpillRecord>) -> Result<u64> {
    records.sort_unstable_by(cell_order);
    let mut writer = FramedRunWriter::create(path)?;
    for record in records.drain(..) {
        writer.write_value(&record)?;
    }
    writer.finish()
}

#[derive(Debug)]
struct CellHeapEntry {
    run_index: usize,
    record: CellSpillRecord,
}

impl PartialEq for CellHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.run_index == other.run_index && cell_order(&self.record, &other.record).is_eq()
    }
}

impl Eq for CellHeapEntry {}

impl PartialOrd for CellHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CellHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        cell_order(&other.record, &self.record).then_with(|| other.run_index.cmp(&self.run_index))
    }
}

fn merge_cell_runs(paths: &[PathBuf], output: &Path) -> Result<u64> {
    let mut readers = Vec::with_capacity(paths.len());
    let mut heap = BinaryHeap::with_capacity(paths.len());
    for (run_index, path) in paths.iter().enumerate() {
        let mut reader = FramedRunReader::open(path)?;
        if let Some(record) = reader.read_value::<CellSpillRecord>()? {
            heap.push(CellHeapEntry { run_index, record });
        }
        readers.push(reader);
    }
    let mut writer = FramedRunWriter::create(output)?;
    while let Some(entry) = heap.pop() {
        let run_index = entry.run_index;
        writer.write_value(&entry.record)?;
        if let Some(record) = readers[run_index].read_value::<CellSpillRecord>()? {
            heap.push(CellHeapEntry { run_index, record });
        }
    }
    writer.finish()
}

struct FramedRunWriter {
    writer: BufWriter<File>,
    bytes_written: u64,
}

impl FramedRunWriter {
    fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().create_new(true).write(true).open(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(RUN_MAGIC)?;
        writer.write_all(&RUN_FORMAT_VERSION.to_le_bytes())?;
        Ok(Self {
            writer,
            bytes_written: RUN_MAGIC.len() as u64 + 2,
        })
    }

    fn write_record(&mut self, record: &SpillRecord) -> Result<()> {
        self.write_value(record)
    }

    fn write_value(&mut self, value: &impl Serialize) -> Result<()> {
        let payload = bincode::serialize(value)?;
        let length = u32::try_from(payload.len()).map_err(|_| {
            SstError::Compaction(format!(
                "spill record frame exceeds u32 length: {} bytes",
                payload.len()
            ))
        })?;
        let checksum = crc32fast::hash(&payload);
        self.writer.write_all(&length.to_le_bytes())?;
        self.writer.write_all(&checksum.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.bytes_written = self
            .bytes_written
            .saturating_add(FRAME_HEADER_BYTES)
            .saturating_add(payload.len() as u64);
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(self.bytes_written)
    }
}

struct FramedRunReader {
    reader: BufReader<File>,
}

impl FramedRunReader {
    fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|error| {
            SstError::Compaction(format!(
                "truncated spill run header {}: {error}",
                path.display()
            ))
        })?;
        if &magic != RUN_MAGIC {
            return Err(SstError::Compaction(format!(
                "invalid spill run magic in {}",
                path.display()
            )));
        }
        let mut version = [0u8; 2];
        reader.read_exact(&mut version).map_err(|error| {
            SstError::Compaction(format!(
                "truncated spill run version {}: {error}",
                path.display()
            ))
        })?;
        let version = u16::from_le_bytes(version);
        if version != RUN_FORMAT_VERSION {
            return Err(SstError::Compaction(format!(
                "unsupported spill run version {version} in {}",
                path.display()
            )));
        }
        Ok(Self { reader })
    }

    fn read_next(&mut self) -> Result<Option<SpillRecord>> {
        Ok(self
            .read_value::<SpillRecord>()?
            .map(SpillRecord::restore_schema_version))
    }

    fn read_value<T: DeserializeOwned>(&mut self) -> Result<Option<T>> {
        let mut header = [0u8; 8];
        let read = self.reader.read(&mut header[..1])?;
        if read == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut header[1..]).map_err(|error| {
            SstError::Compaction(format!("truncated spill frame header: {error}"))
        })?;
        let length = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as u64;
        if length > MAX_FRAME_BYTES {
            return Err(SstError::Compaction(format!(
                "spill frame length {length} exceeds {MAX_FRAME_BYTES} byte safety limit"
            )));
        }
        let expected_checksum = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let mut payload = vec![0u8; length as usize];
        self.reader.read_exact(&mut payload).map_err(|error| {
            SstError::Compaction(format!("truncated spill frame payload: {error}"))
        })?;
        let actual_checksum = crc32fast::hash(&payload);
        if actual_checksum != expected_checksum {
            return Err(SstError::Compaction(format!(
                "spill frame checksum mismatch: expected {expected_checksum:#010x}, got {actual_checksum:#010x}"
            )));
        }
        Ok(Some(bincode::deserialize::<T>(&payload)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::search::mvcc_resolution::MvccResolver;
    use proximadb_block_format::{BlockCompression, BlockMode, VectorQuant};
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
    use proximadb_storage_common::pax_block::PaxSegmentWriter;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    fn record(oid: &str, version: u64, created_at_ns: i64) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            record_version: version,
            created_at_ns,
            updated_at_ns: created_at_ns,
            ..ProximaRecord::default()
        }
    }

    fn spill(records: Vec<ProximaRecord>) -> Vec<SpillRecord> {
        records
            .into_iter()
            .enumerate()
            .map(|(ordinal, record)| SpillRecord::new(ordinal as u64, record))
            .collect()
    }

    fn vector_record(oid: &str, seed: usize) -> ProximaRecord {
        let values = (0..8)
            .map(|dimension| (seed * 17 + dimension * 3) as f32 * 0.01)
            .collect::<Vec<_>>();
        ProximaRecord {
            oid: oid.to_string(),
            created_at_ns: seed as i64 + 1,
            updated_at_ns: seed as i64 + 1,
            embeddings: vec![EmbeddingCell::new_fp32(
                "model_0",
                "dense",
                values.len() as u32,
                values,
            )],
            ..ProximaRecord::default()
        }
    }

    #[test]
    fn startup_cleanup_reclaims_only_unlocked_spill_owners() {
        let root = tempfile::tempdir().expect("scratch root");
        let stale = root
            .path()
            .join(format!("{OWNER_DIRECTORY_PREFIX}111-10-stale"));
        let live = root
            .path()
            .join(format!("{OWNER_DIRECTORY_PREFIX}222-20-live"));
        let unrelated = root.path().join("operator-owned");
        std::fs::create_dir_all(stale.join("task-stale")).expect("stale owner");
        std::fs::create_dir_all(live.join("task-live")).expect("live owner");
        std::fs::create_dir_all(&unrelated).expect("unrelated directory");
        File::create(stale.join(".lease")).expect("stale lease");
        let live_lease = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(live.join(".lease"))
            .expect("live lease");
        assert!(
            lock_file_exclusive(&live_lease, true).expect("lock live lease"),
            "blocking lease acquisition must succeed"
        );

        let reclaimed = cleanup_stale_owner_directories(root.path()).expect("cleanup stale owners");

        assert_eq!(reclaimed, 1);
        assert!(!stale.exists(), "unlocked spill scratch must be reclaimed");
        assert!(live.exists(), "a locked process owner must be preserved");
        assert!(unrelated.exists(), "unknown operator data must be ignored");
    }

    #[test]
    fn spill_owner_is_reused_for_the_same_process_and_root() {
        let root = tempfile::tempdir().expect("scratch root");
        let first = SpillScratchOwner::open(root.path()).expect("first owner");
        let second = SpillScratchOwner::open(root.path()).expect("reused owner");

        assert_eq!(first.path(), second.path());
        assert_eq!(second.reclaimed_owner_directories(), 0);
    }

    #[tokio::test]
    async fn cancellation_drops_task_guard_and_reclaims_scratch() {
        let root = tempfile::tempdir().expect("scratch root");
        let scratch_root = root.path().to_path_buf();
        let (path_tx, path_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let task = SpillTaskScratch::begin(
                &scratch_root,
                42,
                1,
                &["az://bucket/a.pax".to_string()],
                100,
                "az://bucket/merged.pax",
                128,
                400,
            )
            .expect("begin cancellable task");
            path_tx
                .send(task.path().to_path_buf())
                .expect("publish task path");
            std::future::pending::<()>().await;
        });
        let task_path = path_rx.await.expect("receive task path");
        assert!(task_path.exists());

        handle.abort();
        let join_error = handle.await.expect_err("task must be cancelled");

        assert!(join_error.is_cancelled());
        assert!(
            !task_path.exists(),
            "cancelling the future must synchronously drop and reclaim its task guard"
        );
    }

    #[test]
    fn deterministic_task_manifest_blocks_duplicates_and_reclaims_on_drop() {
        let root = tempfile::tempdir().expect("scratch root");
        let inputs = vec![
            "az://bucket/a.pax".to_string(),
            "az://bucket/b.pax".to_string(),
        ];
        let mut task = SpillTaskScratch::begin(
            root.path(),
            42,
            1,
            &inputs,
            300,
            "az://bucket/merged.pax",
            128,
            1_200,
        )
        .expect("begin task");
        let task_path = task.path().to_path_buf();
        assert!(
            SpillTaskScratch::begin(
                root.path(),
                42,
                1,
                &inputs,
                300,
                "az://bucket/merged.pax",
                128,
                1_200,
            )
            .is_err(),
            "the same immutable task cannot execute twice in one process"
        );
        task.update_phase("mvcc-resolved", Some(&[100, 200]))
            .expect("update manifest");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(task.path().join("task.json")).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(manifest["phase"], "mvcc-resolved");
        assert_eq!(manifest["sources"][0]["size_bytes"], 100);
        assert_eq!(manifest["sources"][1]["size_bytes"], 200);

        drop(task);
        assert!(
            !task_path.exists(),
            "task drop must reclaim its whole subtree"
        );
        let restarted = SpillTaskScratch::begin(
            root.path(),
            42,
            1,
            &inputs,
            300,
            "az://bucket/merged.pax",
            128,
            1_200,
        )
        .expect("a reclaimed task identity may be scheduled again");
        drop(restarted);
    }

    struct InMemoryRangeSource {
        bytes: Vec<u8>,
        range_reads: AtomicU64,
        whole_reads: AtomicU64,
    }

    #[async_trait]
    impl CompactionRangeSource for InMemoryRangeSource {
        async fn size(&self, _path: &str) -> Result<u64> {
            Ok(self.bytes.len() as u64)
        }

        async fn read_range(&self, _path: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
            self.range_reads.fetch_add(1, AtomicOrdering::Relaxed);
            let start = usize::try_from(offset)
                .map_err(|_| SstError::Storage("test range offset exceeds usize".to_string()))?;
            let requested_end = offset.saturating_add(length);
            let end = usize::try_from(requested_end)
                .unwrap_or(usize::MAX)
                .min(self.bytes.len());
            Ok(self.bytes.get(start..end).unwrap_or_default().to_vec())
        }

        async fn read_all(&self, _path: &str) -> Result<Vec<u8>> {
            self.whole_reads.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(self.bytes.clone())
        }
    }

    #[test]
    fn external_mvcc_matches_canonical_resolver_across_multiple_runs() {
        let now_ns = 1_000;
        let mut expired = record("deleted", 1, 1);
        expired.valid_to_ns = Some(now_ns - 1);
        let records = vec![
            record("a", 2, 20),
            record("gap", 3, 30),
            record("a", 1, 10),
            record("a", 2, 21),
            expired,
            record("", 1, 40),
            record("", 1, 41),
            record("b", 0, 50),
        ];
        let expected =
            MvccResolver::with_timestamp_ns(now_ns).resolve_sorted_batch(records.clone());
        let scratch = tempfile::tempdir().expect("scratch tempdir");

        let output = resolve_external_mvcc(
            scratch.path(),
            spill(records).into_iter().map(Ok),
            600,
            2,
            now_ns,
        )
        .expect("external MVCC");
        let actual: Vec<_> = output
            .read_records()
            .expect("read output")
            .into_iter()
            .map(|item| item.record)
            .collect();

        assert_eq!(actual, expected);
        assert!(output.stats().initial_run_count > 1);
        assert!(output.stats().merge_pass_count > 0);
        assert!(output.stats().max_open_runs <= 2);
    }

    #[test]
    fn output_is_byte_deterministic_across_run_sizes_and_fan_in() {
        let records = (0..40)
            .rev()
            .map(|i| record(&format!("oid-{i:03}"), 1, i))
            .collect::<Vec<_>>();
        let scratch_a = tempfile::tempdir().expect("scratch a");
        let scratch_b = tempfile::tempdir().expect("scratch b");

        let a = resolve_external_mvcc(
            scratch_a.path(),
            spill(records.clone()).into_iter().map(Ok),
            700,
            2,
            10_000,
        )
        .expect("external MVCC a");
        let b = resolve_external_mvcc(
            scratch_b.path(),
            spill(records).into_iter().map(Ok),
            2_000,
            8,
            10_000,
        )
        .expect("external MVCC b");

        assert_eq!(
            std::fs::read(a.output_path()).expect("read a"),
            std::fs::read(b.output_path()).expect("read b")
        );
    }

    #[test]
    fn checksum_corruption_fails_closed() {
        let scratch = tempfile::tempdir().expect("scratch tempdir");
        let output = resolve_external_mvcc(
            scratch.path(),
            spill(vec![record("a", 1, 1)]).into_iter().map(Ok),
            1_024,
            2,
            10_000,
        )
        .expect("external MVCC");
        let mut bytes = std::fs::read(output.output_path()).expect("read run");
        let last = bytes.len().checked_sub(1).expect("nonempty run");
        bytes[last] ^= 0xff;
        std::fs::write(output.output_path(), bytes).expect("corrupt run");

        let error = output.read_records().expect_err("corruption must fail");
        assert!(error.to_string().contains("checksum"));
    }

    #[test]
    fn output_drop_reclaims_task_scratch() {
        let scratch = tempfile::tempdir().expect("scratch tempdir");
        let task_path = {
            let output = resolve_external_mvcc(
                scratch.path(),
                spill(vec![record("a", 1, 1)]).into_iter().map(Ok),
                1_024,
                2,
                10_000,
            )
            .expect("external MVCC");
            output.task_path().to_path_buf()
        };
        assert!(!task_path.exists());
    }

    #[tokio::test]
    async fn coalesced_inputs_are_ranged_and_apply_deletion_positions() {
        let segment_dir = tempfile::tempdir().expect("segment tempdir");
        let segment_path = segment_dir.path().join("input.pax");
        let records = (0..64)
            .rev()
            .map(|i| vector_record(&format!("oid-{i:03}"), i))
            .collect::<Vec<_>>();
        let mut writer = PaxSegmentWriter::new(
            &segment_path,
            BlockMode::Pax,
            BlockCompression::Lz4,
            "collection",
            0,
            1,
            Some(512),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_rerank_quant(VectorQuant::Sq8)
        .with_coalesced_rabitq(true);
        for record in &records {
            writer.add_record(record).expect("add record");
        }
        writer.finish().expect("finish segment");
        let bytes = std::fs::read(&segment_path).expect("read segment");
        let source = InMemoryRangeSource {
            bytes: bytes.clone(),
            range_reads: AtomicU64::new(0),
            whole_reads: AtomicU64::new(0),
        };
        let scratch = tempfile::tempdir().expect("scratch tempdir");
        let deleted = HashSet::from([0u32]);

        let (output, input_stats) = resolve_external_mvcc_from_segments(
            &source,
            &["az://container/input.pax".to_string()],
            scratch.path(),
            2_048,
            3,
            10_000,
            &["model_0".to_string()],
            &[],
            None,
            &[deleted],
        )
        .await
        .expect("ranged external MVCC");
        let actual = output
            .read_records()
            .expect("read winners")
            .into_iter()
            .map(|record| record.record)
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), records.len() - 1);
        assert!(
            actual.iter().all(|record| record.oid != "oid-063"),
            "the deletion vector must be applied to the input's physical row position"
        );
        assert!(actual.windows(2).all(|pair| pair[0].oid < pair[1].oid));
        assert!(actual.iter().all(|record| {
            record
                .embeddings
                .first()
                .is_some_and(|embedding| embedding.dim == 8)
        }));
        assert_eq!(input_stats.coalesced_files, 1);
        assert_eq!(input_stats.legacy_whole_file_fallbacks, 0);
        assert!(input_stats.block_range_reads > 1);
        assert_eq!(
            source.whole_reads.load(AtomicOrdering::Relaxed),
            0,
            "coalesced compaction must never whole-object decode"
        );
        assert!(source.range_reads.load(AtomicOrdering::Relaxed) > 4);
        assert!(input_stats.largest_range_bytes < bytes.len() as u64);
    }

    #[test]
    fn external_ivf_and_disk_writer_match_in_memory_pax_bytes() {
        let records = (0..128)
            .rev()
            .map(|index| vector_record(&format!("oid-{index:03}"), index))
            .collect::<Vec<_>>();
        let mut canonical_records = records.clone();
        canonical_records.sort_by(|left, right| left.oid.cmp(&right.oid));
        let canonical_plan = crate::storage::engines::sst::block_cluster::cluster_plan_ivf_probe(
            &canonical_records,
            0,
            None,
        )
        .expect("canonical IVF plan");

        let root = tempfile::tempdir().expect("test root");
        let canonical_path = root.path().join("canonical.pax");
        let mut canonical_writer = PaxSegmentWriter::new(
            &canonical_path,
            BlockMode::Pax,
            BlockCompression::Lz4,
            "collection",
            0,
            1,
            Some(1_024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_rerank_quant(VectorQuant::Sq8)
        .with_coalesced_rabitq(true)
        .with_expected_rows(canonical_records.len())
        .with_two_level(canonical_plan.model.clone());
        for row in canonical_plan.plan.order {
            canonical_writer
                .add_record(&canonical_records[row])
                .expect("canonical row");
        }
        canonical_writer.finish().expect("canonical segment");

        let winners = resolve_external_mvcc(
            root.path(),
            spill(records).into_iter().map(Ok),
            2_048,
            2,
            10_000,
        )
        .expect("MVCC winners");
        let clustered =
            cluster_external_mvcc(winners, root.path(), 2_048, 2, 64 * 1024 * 1024, None)
                .expect("external IVF");
        assert_eq!(clustered.model(), &canonical_plan.model);
        assert_eq!(clustered.stats().input_records, 128);
        assert_eq!(clustered.stats().usable_records, 128);
        assert!(clustered.stats().initial_run_count > 1);

        let spill_root = root.path().join("pax-scratch");
        std::fs::create_dir(&spill_root).expect("spill root");
        let spill_path = root.path().join("spill.pax");
        let mut spill_writer = PaxSegmentWriter::new(
            &spill_path,
            BlockMode::Pax,
            BlockCompression::Lz4,
            "collection",
            0,
            1,
            Some(1_024),
        )
        .with_quant(VectorQuant::RaBitQ)
        .with_rerank_quant(VectorQuant::Sq8)
        .with_coalesced_rabitq(true)
        .with_expected_rows(clustered.stats().input_records as usize)
        .with_two_level(clustered.model().clone())
        .with_local_spill(&spill_root)
        .expect("disk writer");
        clustered
            .for_each_record(|record| {
                spill_writer.add_record(&record)?;
                Ok(())
            })
            .expect("stream ordered records");
        spill_writer.finish().expect("spill segment");

        assert_eq!(
            std::fs::read(canonical_path).expect("canonical bytes"),
            std::fs::read(spill_path).expect("spill bytes")
        );
        let task_path = clustered.task_path().to_path_buf();
        drop(clustered);
        assert!(!task_path.exists());
        assert!(
            std::fs::read_dir(spill_root)
                .expect("read spill root")
                .next()
                .is_none()
        );
    }

    /// TD-PAXDIR-1 D3: an output from the same distribution reuses the
    /// collection's shared model (identical PCA and centroids, so the segment
    /// lands in the same cell space); data that drifted outside the model's
    /// trained radii retrains instead.
    #[test]
    fn external_ivf_reuses_the_shared_model_and_retrains_on_drift() {
        use crate::storage::engines::sst::block_cluster::{
            CollectionModelScope, cluster_plan_ivf_probe,
        };
        let records = |offset: usize, shift: f32| {
            (0..128)
                .map(|index| {
                    let mut record = vector_record(&format!("oid-{index:03}"), index + offset);
                    if let Some(proximadb_records::EmbeddingValues::Fp32(values)) =
                        record.embeddings.first_mut().map(|e| &mut e.values)
                    {
                        values.iter_mut().for_each(|v| *v += shift);
                    }
                    record
                })
                .collect::<Vec<_>>()
        };
        let model = cluster_plan_ivf_probe(&records(0, 0.0), 0, None)
            .expect("train the shared model")
            .model;
        let shared = std::sync::Arc::new(
            proximadb_storage_common::centroid_directory::SharedCellSpace {
                covered_rows: model.rows_covered(),
                model,
            },
        );
        let cluster = |input: Vec<ProximaRecord>| {
            let root = tempfile::tempdir().expect("test root");
            let winners = resolve_external_mvcc(
                root.path(),
                spill(input).into_iter().map(Ok),
                2_048,
                2,
                10_000,
            )
            .expect("MVCC winners");
            let _scope = CollectionModelScope::enter(Some(std::sync::Arc::clone(&shared)));
            cluster_external_mvcc(winners, root.path(), 2_048, 2, 64 * 1024 * 1024, None)
                .expect("cluster")
                .model()
                .clone()
        };
        // The same rows: every row lies within its trained radius.
        let reused = cluster(records(0, 0.0));
        assert_eq!(reused.pca_mean, shared.model.pca_mean);
        assert_eq!(reused.pca_components, shared.model.pca_components);
        assert_eq!(reused.centroids, shared.model.centroids);
        assert_eq!(reused.cell_rows.iter().sum::<u64>(), 128);

        let drifted = cluster(records(0, 50.0));
        assert_ne!(drifted.pca_mean, shared.model.pca_mean, "drift retrains");
    }

    #[test]
    fn external_ivf_rejects_training_before_exceeding_working_memory() {
        let records = (0..64)
            .map(|index| vector_record(&format!("oid-{index:03}"), index))
            .collect::<Vec<_>>();
        let root = tempfile::tempdir().expect("test root");
        let winners = resolve_external_mvcc(
            root.path(),
            spill(records).into_iter().map(Ok),
            4_096,
            4,
            10_000,
        )
        .expect("MVCC winners");
        let error = cluster_external_mvcc(winners, root.path(), 4_096, 4, 1024, None)
            .expect_err("tiny working set must reject PCA");
        assert!(error.to_string().contains("above the admitted"));
    }

    #[test]
    fn openai_embedding_partition_training_fits_default_spill_working_set() {
        for (rows, dim) in [(11_184_810usize, 1_536usize), (5_592_405, 3_072)] {
            let shape =
                crate::storage::engines::sst::block_cluster::ivf_training_shape(rows, dim, None)
                    .expect("billion-scale partition shape");
            let estimate = estimated_ivf_training_peak_bytes(dim, shape).expect("memory estimate");
            assert!(
                estimate <= 512 * 1024 * 1024,
                "dim={dim} estimate={estimate}"
            );
        }
    }
}
