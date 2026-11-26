//! TD-PAXDIR-1 D1: publish the collection-global PAX centroid directory.
//!
//! After a compaction has published its output and retired its inputs, the
//! collection's directory (`centroids-<gen>.pxcd`, format in
//! [`proximadb_storage_common::centroid_directory`]) is brought up to date
//! with the live segment set:
//!
//! 1. LIST the data directory, as discovery does: live segments are the
//!    `*.pax` objects not pending retirement (TD-COMPACT-13).
//! 2. Load the newest valid directory as the base; keep its entries for
//!    segments that are still live with the same size.
//! 3. Read the Region A0 of every other live segment (header prefix, then A0:
//!    two ranged reads, off the query path). Segments without a usable A0 are
//!    recorded as examined-but-uncovered, so later publishes skip them.
//!    TD-PAXDIR-1 D6 (`PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS`, default
//!    ON since D8): a covered segment also gets a **decode key** (its Region-A header
//!    and footer: two more ranged reads, and its listed ETag), kept only when
//!    the A0 the directory rebuilds from it is byte-identical to the
//!    segment's. Entries carried from the base keep their key; a carried
//!    keyless entry is retried for one while keys are written.
//!    TD-PAXMETA-1 (ADR-098 D2): an UNCOVERED segment that is coalesced PAX
//!    (the untrained flush layout, or a trained segment whose dimension
//!    mismatches the directory) also gets a key — the zero-encoded untrained
//!    form, built from its header prefix and footer — so collection-level
//!    questions (row counts, exact-authority) read no footers. Legacy
//!    non-coalesced objects stay keyless.
//! 4. Write generation `g + 1` **create-only**. Losing the race to another
//!    publisher reloads and retries; repeated losses give up, since the next
//!    compaction repairs the directory.
//! 5. Delete generations older than the previous one (readers holding an older
//!    generation stay correct: every entry is validated against the live set).
//!
//! Gate: `PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY` (default ON since D8;
//! `0/false/off/no` disables). A publish
//! failure is logged and never fails the compaction: the directory is a
//! projection, and without one queries use the per-segment cascade.

use std::collections::HashMap;

use proximadb_storage_common::centroid_directory::{
    DirectorySegment, LiveSegment, PaxCentroidDirectory, SegmentDecodeKey, object_name,
    parse_object_name, plan_reconcile_keeping,
};
use proximadb_storage_common::coarse_directory::CoarseDirectory;
use proximadb_storage_common::segment_layout::{
    SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix, is_coalesced_segment,
};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem, FilesystemError};

use crate::storage::engines::sst::retirement_ledger;

/// Create-only publish attempts before giving up to a concurrent publisher.
const PUBLISH_ATTEMPTS: usize = 3;

// WRITE gate: compaction publishes the collection centroid directory.
proximadb_config::env_kill_switch! {
    /// Kill-switch gate; default ON (0|false|off|no disables).
    pub fn centroid_directory_write_enabled() { gate: "PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY", default: true }
}

// TD-PAXDIR-1 D6 WRITE gate: publishes add decode keys for the segments they
// load (two more ranged reads each) and write the v2 format. Default ON
// since D8; off, the directory is the D1 format, byte for byte.
proximadb_config::env_kill_switch! {
    /// Kill-switch gate; default ON (0|false|off|no disables).
    pub fn directory_decode_keys_write_enabled() { gate: "PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", default: true }
}

// TD-PAXMETA-1 (ADR-098 D2) WRITE gate: a flush that published new segments
// also refreshes the collection directory — the same create-only reconcile
// the compaction publish runs — so flush-written (uncovered) segments appear
// with their untrained decode keys instead of waiting for the next
// compaction. Default ON since the request-count flip evidence
// (`paxmeta1_directory_at_flush_request_counts_azurite`: cold collection-level
// metadata GETs 20→0 / 40→0 on Azurite, same recall). Each flush pays ~3 GETs
// (2 ranged + the directory) + 1 LIST (+1 PUT, +1 DELETE once two generations
// exist) off the query path, inline but bounded by
// [`directory_at_flush_timeout`]; `0/false/off/no` is the kill switch. Not memoized (`env_kill_switch!`) so
// tests can flip it mid-process.
proximadb_config::env_kill_switch! {
    /// Kill-switch gate; default ON (0|false|off|no disables).
    pub fn directory_at_flush_write_enabled() { gate: "PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH", default: true }
}

/// Default bound on the flush-time publish (ms). The publish runs inline on
/// the flush path (LIST + directory GET + per-changed-segment ranged GETs +
/// PUT + DELETE, up to [`PUBLISH_ATTEMPTS`] create-only attempts), so a slow
/// or hung object-store call must not stall the flush; a few seconds is far
/// above a healthy publish's handful of round trips. Latency unmeasured.
pub const DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS: u64 = 5_000;

/// The flush-time publish bound: `PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH_TIMEOUT_MS`
/// as a positive integer of milliseconds; unset, `0`, an off spelling or a
/// typo keep [`DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS`] (the bound itself can
/// never be switched off — `PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH=0` is the
/// way to take the publish off the flush path). Read on every call.
pub fn directory_at_flush_timeout() -> std::time::Duration {
    let raw = std::env::var("PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH_TIMEOUT_MS").ok();
    let ms = crate::services::storage_format::default_on_gate_value(
        raw.as_deref(),
        DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS,
    )
    .unwrap_or(DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS);
    std::time::Duration::from_millis(ms)
}

/// [`publish_centroid_directory`] bounded by `bound`. `None` when the bound
/// elapsed: the in-flight publish is dropped. Every step is create-only or a
/// delete of a superseded generation, so an interrupted publish leaves at
/// most a stale or extra generation — or, on a backend without atomic PUT
/// (local fs), a truncated newest object, which readers treat as no
/// directory (footer path) and the next publish supersedes.
pub(crate) async fn publish_centroid_directory_bounded(
    fs: &dyn FileSystem,
    dir_url: &str,
    bound: std::time::Duration,
) -> Option<anyhow::Result<Option<u64>>> {
    tokio::time::timeout(bound, publish_centroid_directory(fs, dir_url))
        .await
        .ok()
}

/// Directory objects in a listing, newest generation first.
pub(crate) fn directory_objects_newest_first(entries: &[DirEntry]) -> Vec<(u64, &DirEntry)> {
    let mut objects: Vec<(u64, &DirEntry)> = entries
        .iter()
        .filter(|e| !e.metadata.is_directory)
        .filter_map(|e| parse_object_name(&e.name).map(|generation| (generation, e)))
        .collect();
    objects.sort_by_key(|o| std::cmp::Reverse(o.0));
    objects
}

/// What [`read_segment_a0_with_header`] found: a trained segment's A0, or no
/// usable A0 — carrying the header prefix it already read (when it read a
/// full one) so the untrained-key attach does not GET it again (TD-PAXMETA-1
/// review L1: an untrained flush segment costs 2 ranged GETs, not 3).
enum SegmentA0Read {
    Trained(Box<SegmentA0>),
    Untrained { prefix: Option<Vec<u8>> },
}

/// A segment's Region A0 as the publisher reads it.
pub(crate) struct SegmentA0 {
    pub(crate) a0: CoarseDirectory,
    /// The raw A0 bytes (the decode-key rebuild must reproduce them).
    a0_bytes: Vec<u8>,
    header: SegmentHeaderPrefix,
    prefix: Vec<u8>,
}

/// Read and parse a segment's Region A0. `Ok(None)` when the segment has no
/// usable A0 (a non-coalesced layout, an untrained segment, or a corrupt
/// directory); `Err` only for I/O failures.
pub(crate) async fn read_segment_a0(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
) -> Result<Option<CoarseDirectory>, FilesystemError> {
    Ok(
        match read_segment_a0_with_header(fs, url, size_bytes).await? {
            SegmentA0Read::Trained(segment) => Some(segment.a0),
            SegmentA0Read::Untrained { .. } => None,
        },
    )
}

async fn read_segment_a0_with_header(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
) -> Result<SegmentA0Read, FilesystemError> {
    let prefix_len = (SEG_HEADER_PREFIX_LEN as u64).min(size_bytes);
    if prefix_len < SEG_HEADER_PREFIX_LEN as u64 {
        return Ok(SegmentA0Read::Untrained { prefix: None });
    }
    let prefix = fs.read_range(url, 0, prefix_len).await?;
    let Ok(header) = SegmentHeaderPrefix::parse(&prefix) else {
        return Ok(SegmentA0Read::Untrained {
            prefix: Some(prefix),
        });
    };
    if header.a0_len == 0 || !within(size_bytes, header.a0_off, header.a0_len) {
        return Ok(SegmentA0Read::Untrained {
            prefix: Some(prefix),
        });
    }
    let a0_bytes = fs.read_range(url, header.a0_off, header.a0_len).await?;
    Ok(match CoarseDirectory::parse(&a0_bytes) {
        Ok(a0) => SegmentA0Read::Trained(Box::new(SegmentA0 {
            a0,
            a0_bytes,
            header,
            prefix,
        })),
        Err(_) => SegmentA0Read::Untrained {
            prefix: Some(prefix),
        },
    })
}

fn within(size_bytes: u64, off: u64, len: u64) -> bool {
    off.checked_add(len).is_some_and(|end| end <= size_bytes)
}

/// TD-PAXDIR-1 D6: read the rest of a covered segment's query-path metadata
/// (the Region-A header and the footer) and build its decode key. `None` when
/// a read fails or the parts disagree: the segment stays covered without a
/// key and its queries read their own metadata.
async fn read_decode_key(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
    etag: Option<&str>,
    segment: &SegmentA0,
) -> Option<SegmentDecodeKey> {
    let header = &segment.header;
    let hdr_len = proximadb_block_format::region_header_len(segment.a0.model.dim) as u64;
    if !within(size_bytes, header.rabitq_off, hdr_len)
        || !within(size_bytes, header.footer_off, header.footer_len)
    {
        return None;
    }
    let read = async {
        let rabitq_header = fs.read_range(url, header.rabitq_off, hdr_len).await?;
        let footer = fs
            .read_range(url, header.footer_off, header.footer_len)
            .await?;
        Ok::<_, FilesystemError>((rabitq_header, footer))
    };
    let (rabitq_header, footer) = match read.await {
        Ok(parts) => parts,
        Err(error) => {
            tracing::debug!(%url, %error, "segment metadata unreadable; no decode key");
            return None;
        }
    };
    let footer = SegmentFooterIndex::parse(&footer).ok()?;
    SegmentDecodeKey::from_segment(&segment.prefix, &segment.a0, &rabitq_header, &footer, etag)
        .map_err(|error| tracing::debug!(%url, %error, "segment has no decode key"))
        .ok()
}

/// Keys the published directory may keep: none when the write gate is off
/// (so turning it off returns the directory to the D1 v1 format, carried
/// keys included), and otherwise only keys whose recorded ETag still matches
/// the segment's listing (a carried key for a rewritten object is dropped
/// rather than refused by every reader).
fn retain_current_decode_keys(
    dir: &mut PaxCentroidDirectory,
    decode_keys: bool,
    listed: &HashMap<String, (String, Option<String>)>,
) {
    for segment in &mut dir.segments {
        if !(decode_keys && key_matches_listing(segment, listed)) {
            segment.decode_key = None;
        }
    }
}

/// TD-PAXMETA-1 (ADR-098 D2): read an UNCOVERED segment's footer and build its
/// zero-encoded decode key — the coalesced-but-untrained segments a flush
/// writes, and trained segments whose dimension mismatches the directory. Two
/// ranged reads (header prefix, footer) — one when the caller passes the
/// `prefix` it already read. `None` when the object is not
/// coalesced PAX or any part disagrees: the segment stays uncovered and
/// keyless, and its readers keep the per-segment footer path.
async fn read_uncovered_decode_key(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
    etag: Option<&str>,
    prefix: Option<Vec<u8>>,
) -> Option<SegmentDecodeKey> {
    let prefix_len = (SEG_HEADER_PREFIX_LEN as u64).min(size_bytes);
    if prefix_len < SEG_HEADER_PREFIX_LEN as u64 {
        return None;
    }
    let prefix = match prefix {
        Some(prefix) => prefix,
        None => match fs.read_range(url, 0, prefix_len).await {
            Ok(prefix) => prefix,
            Err(error) => {
                tracing::debug!(%url, %error, "segment header unreadable; no untrained key");
                return None;
            }
        },
    };
    if !is_coalesced_segment(&prefix) {
        return None; // legacy block layout: never keyable
    }
    let Ok(header) = SegmentHeaderPrefix::parse(&prefix) else {
        return None;
    };
    if header.footer_len == 0 || !within(size_bytes, header.footer_off, header.footer_len) {
        return None;
    }
    let footer = match fs
        .read_range(url, header.footer_off, header.footer_len)
        .await
    {
        Ok(footer) => footer,
        Err(error) => {
            tracing::debug!(%url, %error, "segment footer unreadable; no untrained key");
            return None;
        }
    };
    let footer = SegmentFooterIndex::parse(&footer).ok()?;
    SegmentDecodeKey::from_untrained_segment(&prefix, &footer, etag)
        .map_err(|error| tracing::debug!(%url, %error, "segment has no untrained decode key"))
        .ok()
}

/// Best-effort untrained-key attach for a segment just recorded as uncovered:
/// a failure leaves it keyless (readers use the footer path), never errors
/// the publish.
async fn attach_untrained_key(
    builder: &mut proximadb_storage_common::centroid_directory::DirectoryBuilder,
    fs: &dyn FileSystem,
    name: &str,
    url: &str,
    size_bytes: u64,
    etag: Option<&str>,
    decode_keys: bool,
    prefix: Option<Vec<u8>>,
) {
    if !decode_keys {
        return;
    }
    if let Some(key) = read_uncovered_decode_key(fs, url, size_bytes, etag, prefix).await
        && let Err(error) = builder.set_decode_key(name, key)
    {
        tracing::debug!(%url, %error, "untrained decode key refused; publishing keyless");
    }
}

/// Whether `segment`'s decode key records the ETag its listing reports.
fn key_matches_listing(
    segment: &DirectorySegment,
    listed: &HashMap<String, (String, Option<String>)>,
) -> bool {
    segment.decode_key.as_ref().is_some_and(|key| {
        listed
            .get(&segment.name)
            .is_some_and(|(_, etag)| key.matches_object(etag.as_deref()))
    })
}

/// Drop every freshly added key whose rebuilt A0 is not byte-identical to
/// the segment's own (e.g. two models sharing a `model_id`): a key must never
/// describe a segment differently from its bytes.
fn verify_decode_keys(dir: &mut PaxCentroidDirectory, a0_by_name: &HashMap<String, Vec<u8>>) {
    for idx in 0..dir.segments.len() {
        let Some(expected) = a0_by_name.get(&dir.segments[idx].name) else {
            continue;
        };
        if dir.segments[idx].decode_key.is_none() {
            continue;
        }
        let rebuilt = dir.segment_decode(idx).map(|d| d.a0);
        if rebuilt.as_ref().ok() != Some(expected) {
            tracing::warn!(
                segment = %dir.segments[idx].name,
                "decode key does not reproduce the segment's A0; publishing without it"
            );
            dir.segments[idx].decode_key = None;
        }
    }
}

/// The first directory among `objects` (`(generation, url)`, newest first)
/// that reads, parses and records its listed generation; corrupt or
/// unreadable generations are skipped. Shared by the publisher (an older
/// valid base still saves A0 reads) and the query-side reader (a corrupt
/// newest generation falls back instead of disabling the directory).
/// `resident` serves a resident copy without a GET (TD-TIER-1 SC4-5; the
/// query path), otherwise every object is read from the store (publish).
/// Concrete owned inputs and a plain flag, not a generic iterator or an
/// async closure: either makes the compaction worker future not `Send`.
pub(crate) async fn first_valid_directory(
    fs: &dyn FileSystem,
    objects: &[(u64, String)],
    resident: bool,
) -> Option<PaxCentroidDirectory> {
    for (generation, url) in objects {
        let (generation, url) = (*generation, url.as_str());
        let read = if resident {
            crate::storage::engines::sst::segment_io::ResidentRangeRead::read_resident(fs, url)
                .await
        } else {
            fs.read(url).await
        };
        match read {
            Ok(bytes) => match PaxCentroidDirectory::parse(&bytes) {
                Ok(dir) if dir.generation == generation => return Some(dir),
                Ok(_) => tracing::warn!(%url, "centroid directory generation mismatch"),
                Err(error) => tracing::warn!(%url, %error, "corrupt centroid directory"),
            },
            Err(error) => tracing::warn!(%url, %error, "unreadable centroid directory"),
        }
    }
    None
}

/// Newest directory in `objects` that reads and parses.
async fn load_newest_valid(
    fs: &dyn FileSystem,
    objects: &[(u64, &DirEntry)],
) -> Option<PaxCentroidDirectory> {
    let objects: Vec<(u64, String)> = objects.iter().map(|(g, e)| (*g, e.url.clone())).collect();
    first_valid_directory(fs, &objects, false).await
}

/// The newest valid directory of the collection data directory `dir_url`
/// (one LIST plus one read), or `None` when there is none or it is unreadable.
pub(crate) async fn load_newest_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> Option<PaxCentroidDirectory> {
    let entries = fs.list(dir_url.trim_end_matches('/')).await.ok()?;
    load_newest_valid(fs, &directory_objects_newest_first(&entries)).await
}

async fn object_size(fs: &dyn FileSystem, entry: &DirEntry) -> Option<u64> {
    if entry.metadata.size > 0 {
        return Some(entry.metadata.size);
    }
    fs.metadata(&entry.url).await.ok().map(|m| m.size)
}

/// Bring the directory of the collection data directory `dir_url` up to date.
/// Returns the generation written, or `None` when nothing changed (or a
/// concurrent publisher kept winning).
pub async fn publish_centroid_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> anyhow::Result<Option<u64>> {
    let dir_url = dir_url.trim_end_matches('/');
    for _ in 0..PUBLISH_ATTEMPTS {
        let entries = fs.list(dir_url).await?;
        let pending = retirement_ledger::pending_paths();
        let mut live = Vec::new();
        let mut urls: HashMap<String, (String, Option<String>)> = HashMap::new();
        for entry in &entries {
            if entry.metadata.is_directory
                || !entry.name.ends_with(".pax")
                || pending.contains(&entry.url)
            {
                continue;
            }
            let Some(size_bytes) = object_size(fs, entry).await else {
                continue; // vanished between LIST and stat: not live
            };
            urls.insert(
                entry.name.clone(),
                (entry.url.clone(), entry.metadata.etag.clone()),
            );
            live.push(LiveSegment {
                name: entry.name.clone(),
                size_bytes,
            });
        }
        let objects = directory_objects_newest_first(&entries);
        let newest_generation = objects.first().map(|(g, _)| *g);
        let base = load_newest_valid(fs, &objects).await;
        let decode_keys = directory_decode_keys_write_enabled();
        // D6: a carried entry whose key no longer matches its object (a
        // rewrite in place) is reloaded, cells included. ADR-042: so is a key
        // carried from a v2 base (no filter facts) — a one-time re-read of
        // that segment's header and footer, after which the v3 entry is
        // carried like any other.
        let mut plan = plan_reconcile_keeping(base.as_ref(), &live, |segment| {
            segment.decode_key.as_ref().is_none_or(|key| {
                key_matches_listing(segment, &urls) && (key.filter_facts.is_some() || !decode_keys)
            })
        });
        // A base older than the newest object (the newest was corrupt) is
        // republished even if it matches the live set, and so is one whose
        // keys the write gate no longer allows (turning the gate off must
        // return the directory to v1 without waiting for another change).
        let base_is_newest = base.as_ref().map(|d| d.generation) == newest_generation;
        let keys_to_drop = !decode_keys
            && base
                .as_ref()
                .is_some_and(|d| d.segments.iter().any(|s| s.decode_key.is_some()));
        // Post-#2535 review M1: a carried KEYLESS entry (a pre-key base, a
        // transient footer-read failure, a key gate toggled off then on) is
        // kept — its coverage is still valid — but, while keys are written,
        // retried for a key on every publish instead of being carried keyless
        // forever. Only keyless entries are re-read; an entry that still
        // cannot be keyed (legacy layout, a failed read) stays keyless and,
        // when nothing else changed, no new generation is written.
        let to_upgrade: Vec<DirectorySegment> = match (&base, decode_keys) {
            (Some(base), true) => {
                let reloading: std::collections::HashSet<&str> =
                    plan.to_load.iter().map(|s| s.name.as_str()).collect();
                base.segments
                    .iter()
                    .filter(|s| {
                        s.decode_key.is_none()
                            && urls.contains_key(&s.name)
                            && !reloading.contains(s.name.as_str())
                    })
                    .cloned()
                    .collect()
            }
            _ => Vec::new(),
        };
        if !plan.changed && base_is_newest && !keys_to_drop && to_upgrade.is_empty() {
            return Ok(None);
        }
        let mut a0_by_name = HashMap::new();
        for segment in &to_upgrade {
            let (url, etag) = &urls[&segment.name];
            if segment.is_covered() {
                if let Ok(SegmentA0Read::Trained(read)) =
                    read_segment_a0_with_header(fs, url, segment.size_bytes).await
                    && let Some(key) =
                        read_decode_key(fs, url, segment.size_bytes, etag.as_deref(), &read).await
                    && plan.builder.set_decode_key(&segment.name, key).is_ok()
                {
                    a0_by_name.insert(segment.name.clone(), read.a0_bytes);
                }
            } else {
                attach_untrained_key(
                    &mut plan.builder,
                    fs,
                    &segment.name,
                    url,
                    segment.size_bytes,
                    etag.as_deref(),
                    decode_keys,
                    None,
                )
                .await;
            }
        }
        for segment in &plan.to_load {
            let (url, etag) = &urls[&segment.name];
            match read_segment_a0_with_header(fs, url, segment.size_bytes).await {
                Ok(SegmentA0Read::Trained(read)) => {
                    if let Err(error) =
                        plan.builder
                            .add_covered(&segment.name, segment.size_bytes, &read.a0)
                    {
                        tracing::debug!(%url, %error, "segment not addressable by the directory");
                        plan.builder
                            .add_uncovered(&segment.name, segment.size_bytes);
                        // TD-PAXMETA-1: a trained segment the directory cannot
                        // address (dimension mismatch) still gets its
                        // collection-level facts published.
                        attach_untrained_key(
                            &mut plan.builder,
                            fs,
                            &segment.name,
                            url,
                            segment.size_bytes,
                            etag.as_deref(),
                            decode_keys,
                            Some(read.prefix.clone()),
                        )
                        .await;
                    } else if decode_keys
                        && let Some(key) =
                            read_decode_key(fs, url, segment.size_bytes, etag.as_deref(), &read)
                                .await
                        && plan.builder.set_decode_key(&segment.name, key).is_ok()
                    {
                        a0_by_name.insert(segment.name.clone(), read.a0_bytes);
                    }
                }
                Ok(SegmentA0Read::Untrained { prefix }) => {
                    plan.builder
                        .add_uncovered(&segment.name, segment.size_bytes);
                    // TD-PAXMETA-1: a coalesced-but-untrained segment (the
                    // flush layout) carries its footer facts in the directory.
                    attach_untrained_key(
                        &mut plan.builder,
                        fs,
                        &segment.name,
                        url,
                        segment.size_bytes,
                        etag.as_deref(),
                        decode_keys,
                        prefix,
                    )
                    .await;
                }
                // Transient read failure: leave the segment out of the stamp
                // entirely, so readers scan it per segment and the next
                // publish retries it.
                Err(error) => tracing::warn!(%url, %error, "segment A0 unreadable; not stamped"),
            }
        }
        let generation = newest_generation.map_or(1, |g| g.saturating_add(1));
        let mut dir = plan.builder.build(generation)?;
        verify_decode_keys(&mut dir, &a0_by_name);
        retain_current_decode_keys(&mut dir, decode_keys, &urls);
        let upgraded = {
            let by_name = dir.segment_indices_by_name();
            to_upgrade.iter().any(|old| {
                by_name
                    .get(old.name.as_str())
                    .and_then(|&idx| dir.segments.get(idx))
                    .is_some_and(|s| s.decode_key.is_some())
            })
        };
        if !plan.changed && base_is_newest && !keys_to_drop && !upgraded {
            return Ok(None); // every keyless entry is still unkeyable
        }
        let bytes = dir.to_bytes()?;
        let url = format!("{dir_url}/{}", object_name(generation));
        // TD-TIER-1 SC5 residual: control objects (the centroid directory)
        // keep an EXPLICIT Hot tier so a cold-tier bucket policy never
        // transitions-lifecycle-moves them — they are read on every query.
        let options = crate::storage::persistence::filesystem::FileOptions::for_access_tier(Some(
            proximadb_storage_filesystem_types::ObjectAccessTier::Hot,
        ));
        match fs.write_if_absent(&url, &bytes, options).await {
            Ok(()) => {
                for (old, entry) in &objects {
                    // TD-TIER-1 SC4-5: g-1 is kept remotely for readers but is
                    // no longer what queries read — drop its resident copy so
                    // it stops charging the grant (review N-b).
                    if old.saturating_add(1) == generation {
                        crate::storage::engines::sst::segment_io::forget_object(&entry.url);
                    }
                    if old.saturating_add(1) < generation {
                        match fs.delete(&entry.url).await {
                            // TD-TIER-1 SC4-5: a collected generation's resident
                            // copy must not keep charging the grant.
                            Ok(()) => {
                                crate::storage::engines::sst::segment_io::forget_object(&entry.url)
                            }
                            Err(error) => tracing::debug!(
                                url = %entry.url,
                                %error,
                                "centroid directory GC deferred"
                            ),
                        }
                    }
                }
                return Ok(Some(generation));
            }
            Err(FilesystemError::AlreadyExists(_)) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    tracing::debug!(
        dir_url,
        "centroid directory publish lost every race; next compaction repairs"
    );
    Ok(None)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::storage::engines::sst::segment_format::write_pax_segment_compacted;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_block_format::VectorQuant;
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use std::sync::Arc;

    const DIM: usize = 16;

    fn enable_trained_layout() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
        }
    }

    fn records(seed: usize, n: usize) -> Vec<ProximaRecord> {
        (0..n)
            .map(|i| {
                let vector = (0..DIM)
                    .map(|d| ((((i + seed) * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect::<Vec<f32>>();
                let mut r = ProximaRecord {
                    oid: format!("s{seed}-{i}"),
                    tenant_id: "t".into(),
                    created_at_ns: 1000 + i as i64,
                    updated_at_ns: 1000 + i as i64,
                    ..Default::default()
                };
                r.embeddings
                    .push(EmbeddingCell::new_fp32("", "dense", DIM as u32, vector));
                r
            })
            .collect()
    }

    pub(crate) fn write_segment(dir: &std::path::Path, name: &str, seed: usize) {
        write_pax_segment_compacted(
            &dir.join(name),
            &records(seed, 300),
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .expect("write trained segment");
    }

    async fn local_fs() -> LocalFileSystem {
        LocalFileSystem::new(LocalConfig::default()).await.unwrap()
    }

    /// Parsable directories, newest first.
    async fn newest(fs: &dyn FileSystem, dir: &str) -> Vec<(u64, PaxCentroidDirectory)> {
        let entries = fs.list(dir).await.unwrap();
        let mut out = Vec::new();
        for (generation, entry) in directory_objects_newest_first(&entries) {
            let bytes = fs.read(&entry.url).await.unwrap();
            if let Ok(dir) = PaxCentroidDirectory::parse(&bytes) {
                out.push((generation, dir));
            }
        }
        out
    }

    #[tokio::test]
    async fn publish_covers_exactly_the_live_trained_segments() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        std::fs::write(tmp.path().join("l0.pax"), b"not a coalesced segment").unwrap();
        std::fs::write(tmp.path().join("retire-pending.json"), b"{}").unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();

        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        let published = newest(&fs, &dir_url).await;
        assert_eq!(published.len(), 1);
        let dir = &published[0].1;
        let covered: Vec<&str> = dir
            .covered_segments()
            .map(|(_, s)| s.name.as_str())
            .collect();
        assert_eq!(covered, ["a.pax", "b.pax"]);
        let uncovered: Vec<&str> = dir
            .segments
            .iter()
            .filter(|s| !s.is_covered())
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(uncovered, ["l0.pax"], "examined but not addressable");
        assert_eq!(dir.dim as usize, DIM);
        for (idx, segment) in dir.covered_segments() {
            assert_eq!(segment.rows_covered, 300);
            assert!(dir.cells.iter().any(|c| c.segment_idx as usize == idx));
        }

        // Every cell's extents equal the segment's own A0.
        for (idx, segment) in dir.covered_segments() {
            let url = tmp.path().join(&segment.name).to_string_lossy().to_string();
            let a0 = read_segment_a0(&fs, &url, segment.size_bytes)
                .await
                .unwrap()
                .unwrap();
            for cell in dir.cells.iter().filter(|c| c.segment_idx as usize == idx) {
                assert_eq!(cell.entry, a0.cells[cell.a0_cell as usize]);
            }
        }

        // Nothing changed: no new generation.
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }

    /// TD-PAXDIR-1 D6: every covered segment is published with a decode key
    /// that rebuilds its header, Region A0, Region-A header and the footer's
    /// ANN fields exactly as the segment stores them.
    #[tokio::test]
    async fn published_decode_keys_reproduce_the_segment_metadata() {
        enable_trained_layout();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.covered_segments().count(), 2, "both segments covered");
        for (idx, segment) in dir.covered_segments() {
            // Read the segment's own metadata through the FileSystem seam,
            // as ranged reads — the same way the query path would.
            let url = format!("{dir_url}/{}", segment.name);
            let read = |off: u64, len: u64| fs.read_range(&url, off, len);
            let prefix = read(0, SEG_HEADER_PREFIX_LEN as u64).await.unwrap();
            let header = SegmentHeaderPrefix::parse(&prefix).unwrap();
            let decoded = dir
                .segment_decode(idx)
                .expect("covered segments carry a key");
            assert_eq!(decoded.header_prefix, prefix);
            assert_eq!(
                decoded.a0,
                read(header.a0_off, header.a0_len).await.unwrap()
            );
            let hdr_len = proximadb_block_format::region_header_len(DIM as u32) as u64;
            assert_eq!(
                decoded.rabitq_header,
                read(header.rabitq_off, hdr_len).await.unwrap()
            );
            let real = SegmentFooterIndex::parse(
                &read(header.footer_off, header.footer_len).await.unwrap(),
            )
            .unwrap();
            let ann = SegmentFooterIndex::parse(&decoded.ann_footer).unwrap();
            assert_eq!(ann.row_count, real.row_count);
            assert_eq!(
                (ann.sq8_min.to_bits(), ann.sq8_scale.to_bits()),
                (real.sq8_min.to_bits(), real.sq8_scale.to_bits())
            );
            assert_eq!(ann.embed_dim, real.embed_dim);
            assert_eq!(ann.embed_count, real.embed_count);
            assert_eq!(ann.embed_quant_tag, real.embed_quant_tag);
            assert_eq!(ann.has_f32_tier, real.has_f32_tier);
            assert_eq!(
                (ann.rabitq_off, ann.rabitq_len, ann.sq8_off, ann.sq8_len),
                (real.rabitq_off, real.rabitq_len, real.sq8_off, real.sq8_len)
            );
            assert_eq!(
                (ann.a0_off, ann.a0_len, ann.c_off, ann.c_len),
                (real.a0_off, real.a0_len, real.c_off, real.c_len)
            );
            assert_eq!(ann.blocks, real.blocks);
            assert_eq!(ann.block_stats, real.block_stats);
        }
    }

    /// Turning the decode-key write gate off returns the next publish to the
    /// D1 (v1) format, even for keys carried from a keyed base.
    #[tokio::test]
    async fn turning_the_key_gate_off_drops_carried_keys() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        let keyed = fs
            .read(&format!("{dir_url}/{}", object_name(1)))
            .await
            .unwrap();
        // ADR-042 (2026-11-21): a keyed directory is written as v3 (the key
        // carries the filter facts; breaking change accepted pre-GA).
        assert_eq!(keyed[4], 3, "v3 with a key");

        // No live change: turning the gate off alone republishes as v1,
        // once; a later change keeps v1.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "0") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        let v1 = fs
            .read(&format!("{dir_url}/{}", object_name(2)))
            .await
            .unwrap();
        assert_eq!(v1[4], 1, "the gate alone returns the directory to v1");
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None,
            "and only once"
        );
        write_segment(tmp.path(), "b.pax", 7);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3)
        );
        let bytes = fs
            .read(&format!("{dir_url}/{}", object_name(3)))
            .await
            .unwrap();
        assert_eq!(bytes[4], 1, "back to v1");
        let dir = PaxCentroidDirectory::parse(&bytes).unwrap();
        assert_eq!(dir.covered_segments().count(), 2);
        assert!(dir.segments.iter().all(|s| s.decode_key.is_none()));
    }

    /// A carried key whose recorded ETag no longer matches the listing means
    /// the object was rewritten in place: the segment is reloaded (cells and
    /// key), not carried. The local backend lists no ETag, so the newest
    /// generation is planted with a key recording one.
    #[tokio::test]
    async fn a_stale_etag_reloads_the_segment() {
        enable_trained_layout();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let mut planted = newest(&fs, &dir_url).await.remove(0).1;
        planted.generation = 2;
        planted.segments[0].decode_key.as_mut().unwrap().object_etag = Some("old".into());
        fs.write_if_absent(
            &format!("{dir_url}/{}", object_name(2)),
            &planted.to_bytes().unwrap(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3),
            "a stale key is a change"
        );
        let dir = &newest(&fs, &dir_url).await[0].1;
        let key = dir.segments[0].decode_key.as_ref().expect("reloaded key");
        assert!(key.matches_object(None), "rebuilt from the listed object");
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }

    /// With the decode-key write gate off, a publish writes the D1 (v1)
    /// format byte for byte: no keys, no extra segment reads.
    #[tokio::test]
    async fn without_the_key_gate_the_directory_is_the_d1_format() {
        enable_trained_layout();
        // Default ON since D8 — disable via the kill-switch for this arm.
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "0") };
        assert!(!directory_decode_keys_write_enabled());
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let bytes = fs
            .read(&format!("{dir_url}/{}", object_name(1)))
            .await
            .unwrap();
        assert_eq!(bytes[4], 1, "v1");
        let dir = PaxCentroidDirectory::parse(&bytes).unwrap();
        assert_eq!(dir.covered_segments().count(), 1);
        assert!(dir.segments.iter().all(|s| s.decode_key.is_none()));
    }

    #[tokio::test]
    async fn publish_tracks_compaction_and_garbage_collects_old_generations() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );

        // A compaction folds a + b into c.
        write_segment(tmp.path(), "c.pax", 3);
        std::fs::remove_file(tmp.path().join("a.pax")).unwrap();
        std::fs::remove_file(tmp.path().join("b.pax")).unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        let covered: Vec<String> = newest(&fs, &dir_url).await[0]
            .1
            .covered_segments()
            .map(|(_, s)| s.name.clone())
            .collect();
        assert_eq!(covered, ["c.pax"]);

        write_segment(tmp.path(), "d.pax", 5);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3)
        );
        let generations: Vec<u64> = newest(&fs, &dir_url)
            .await
            .iter()
            .map(|(g, _)| *g)
            .collect();
        assert_eq!(generations, [3, 2], "keeps the newest two generations");
    }

    #[tokio::test]
    async fn a_corrupt_newest_generation_is_superseded_by_a_rebuild() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        std::fs::write(tmp.path().join(object_name(4)), b"garbage").unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(5)
        );
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.generation, 5);
        assert_eq!(dir.covered_segments().count(), 1);
    }

    #[tokio::test]
    async fn losing_the_create_only_race_retries_on_the_winner() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        write_segment(tmp.path(), "b.pax", 7);
        // Two publishers race on the same change: exactly one generation is
        // written for it, and the loser finds nothing left to do.
        let (first, second) = tokio::join!(
            publish_centroid_directory(&fs, &dir_url),
            publish_centroid_directory(&fs, &dir_url)
        );
        let mut written: Vec<u64> = [first.unwrap(), second.unwrap()]
            .into_iter()
            .flatten()
            .collect();
        written.sort();
        assert_eq!(written, [2]);
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.covered_segments().count(), 2);
    }

    #[test]
    fn write_gate_defaults_on_with_kill_switch() {
        // Default ON since D8; the gate is a kill-switch.
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY") };
        assert!(centroid_directory_write_enabled());
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY", "0") };
        assert!(!centroid_directory_write_enabled());
        // Restore for the shared-process `cargo test --lib` runner.
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY") };
    }

    /// TD-PAXMETA-1: the flush-publish gate is a default-ON kill switch —
    /// unset (and unrecognized values) publish at flush; the kill-switch set
    /// `0|false|off|no` (case-insensitive, whitespace-tolerant) disables it.
    #[test]
    #[serial_test::serial]
    fn at_flush_gate_defaults_on_with_kill_switch() {
        // SAFETY: #[serial] keeps the gate private to this test in the
        // shared `cargo test --lib` process; restored at the end.
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH") };
        assert!(directory_at_flush_write_enabled(), "default ON");
        for on in ["1", "true", "on", "yes", "bogus"] {
            unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH", on) };
            assert!(directory_at_flush_write_enabled(), "{on} keeps it on");
        }
        for off in ["0", "false", "off", "no", " OFF "] {
            unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH", off) };
            assert!(!directory_at_flush_write_enabled(), "{off} kills it");
        }
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH") };
    }

    /// A filesystem whose LIST never completes (a hung object-store call).
    #[derive(Debug)]
    struct StallingListFs {
        inner: Arc<LocalFileSystem>,
    }

    proximadb_storage_filesystem_types::forward_filesystem_impl! {
        impl FileSystem for StallingListFs => inner;
        forward [
            as_any, filesystem_type, supports_mmap, write,
            supports_bounded_local_file_write, write_local_file, write_if_absent, append,
            supports_append, supports_conditional_replace, delete, exists, metadata,
            create_dir, create_dir_all, copy, move_file, sync, sync_file, sync_data, open_file,
        ];
        {
            async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
                self.inner.read(path).await
            }

            async fn read_range(
                &self,
                path: &str,
                offset: u64,
                length: u64,
            ) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
                self.inner.read_range(path, offset, length).await
            }

            async fn list(
                &self,
                _path: &str,
            ) -> proximadb_storage_filesystem_types::FsResult<Vec<DirEntry>> {
                std::future::pending().await
            }
        }
    }

    /// TD-PAXMETA-1 review M1: the flush-time publish is bounded — a publish
    /// whose object-store call hangs returns `None` at the bound instead of
    /// stalling the flush. Mutation check: replacing the `timeout` in
    /// `publish_centroid_directory_bounded` with a plain await makes the
    /// outer guard below fire (the test fails instead of hanging forever).
    #[tokio::test]
    async fn a_hung_flush_publish_returns_at_the_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let fs = StallingListFs {
            inner: Arc::new(local_fs().await),
        };
        let dir = tmp.path().to_str().unwrap().to_string();
        let bound = std::time::Duration::from_millis(50);
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            publish_centroid_directory_bounded(&fs, &dir, bound),
        )
        .await
        .expect("the bounded publish must return well before the 10 s guard");
        assert!(outcome.is_none(), "a hung publish reports the timeout");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "returned at the bound, not after the hang: {:?}",
            started.elapsed()
        );
        // Positive control: the same (non-stalling) backend under the default
        // bound completes — the `None` above is the stall, not the wrapper.
        let healthy = local_fs().await;
        let outcome = publish_centroid_directory_bounded(
            &healthy,
            &dir,
            std::time::Duration::from_millis(DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS),
        )
        .await;
        assert!(
            outcome.is_some(),
            "a healthy publish finishes inside the bound"
        );
    }

    /// The bound's env knob: positive milliseconds; everything else keeps the
    /// default (the bound itself cannot be switched off).
    #[test]
    #[serial_test::serial]
    fn at_flush_timeout_knob_parses_positive_millis() {
        const KNOB: &str = "PROXIMADB_PAX_WRITE_DIRECTORY_AT_FLUSH_TIMEOUT_MS";
        let default = std::time::Duration::from_millis(DEFAULT_DIRECTORY_AT_FLUSH_TIMEOUT_MS);
        // SAFETY: #[serial]; restored at the end.
        unsafe { std::env::remove_var(KNOB) };
        assert_eq!(directory_at_flush_timeout(), default);
        unsafe { std::env::set_var(KNOB, "250") };
        assert_eq!(
            directory_at_flush_timeout(),
            std::time::Duration::from_millis(250)
        );
        for keep in ["0", "off", "typo", ""] {
            unsafe { std::env::set_var(KNOB, keep) };
            assert_eq!(directory_at_flush_timeout(), default, "{keep:?}");
        }
        unsafe { std::env::remove_var(KNOB) };
    }

    /// Post-#2535 review M1: a keyless base (written with the key gate off,
    /// as a pre-key directory or a transient key-read failure leaves it) is
    /// upgraded by the next publish once keys are written, though no segment
    /// changed; an entry that still cannot be keyed does not force a new
    /// generation. Negative control: with the gate off the keyless base is
    /// carried with zero segment reads.
    #[tokio::test]
    async fn a_keyless_base_entry_is_upgraded_to_a_keyed_one() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        crate::storage::engines::sst::segment_format::write_pax_segment(
            &tmp.path().join("l0.pax"),
            &records(3, 300),
            "col",
            1,
            VectorQuant::RaBitQ,
            None,
            None,
        )
        .expect("write untrained segment");
        std::fs::write(tmp.path().join("garbage.pax"), b"not a segment").unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "0") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        let keyless = &newest(&fs, &dir_url).await[0].1;
        assert!(keyless.segments.iter().all(|s| s.decode_key.is_none()));

        // Negative control: gate still off ⇒ nothing to upgrade, no reads.
        let counters =
            Arc::new(proximadb_storage_filesystem_types::counting::GetCounters::default());
        let counting = proximadb_storage_filesystem_types::counting::CountingFileSystem::new(
            Arc::new(local_fs().await),
            counters.clone(),
        );
        assert_eq!(
            publish_centroid_directory(&counting, &dir_url)
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            counters
                .range_reads
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "gate off: keyless entries are carried without segment reads"
        );

        // Gate on: the unchanged live set is republished with keys attached.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2),
            "keyless entries are retried, not carried keyless forever"
        );
        let keyed = &newest(&fs, &dir_url).await[0].1;
        let a = keyed.segments.iter().find(|s| s.name == "a.pax").unwrap();
        assert!(a.is_covered(), "coverage is carried, not dropped");
        assert!(a.decode_key.is_some(), "covered entry gained its key");
        let l0 = keyed.segments.iter().find(|s| s.name == "l0.pax").unwrap();
        assert!(l0.decode_key.is_some(), "untrained entry gained its key");
        let garbage = keyed
            .segments
            .iter()
            .find(|s| s.name == "garbage.pax")
            .unwrap();
        assert!(
            garbage.decode_key.is_none(),
            "unkeyable entry stays keyless"
        );

        // Only the unkeyable entry is left keyless: it is retried, but no new
        // generation is written for it.
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }

    /// TD-PAXMETA-1 (ADR-098 D2): an untrained (flush-layout) coalesced
    /// segment is published UNCOVERED but WITH its zero-encoded decode key,
    /// whose footer facts equal the segment's own; a non-coalesced object
    /// stays keyless. A second publish finds nothing changed.
    #[tokio::test]
    async fn untrained_coalesced_segments_carry_zero_encoded_keys() {
        enable_trained_layout();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        let tmp = tempfile::tempdir().unwrap();
        // Flush-style L0: coalesced RaBitQ (the default write layout), no A0
        // trained — the exact shape the flush path publishes.
        crate::storage::engines::sst::segment_format::write_pax_segment(
            &tmp.path().join("l0.pax"),
            &records(0, 300),
            "col",
            1,
            VectorQuant::RaBitQ,
            None,
            None,
        )
        .expect("write untrained segment");
        std::fs::write(tmp.path().join("garbage.pax"), b"not a segment").unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();

        // Review L1: the untrained segment costs exactly 2 ranged reads (header
        // prefix once, footer) — the prefix the A0 probe read is reused, not
        // fetched again (was 3). The garbage object is shorter than a prefix
        // and costs none.
        let counters =
            Arc::new(proximadb_storage_filesystem_types::counting::GetCounters::default());
        let counting = proximadb_storage_filesystem_types::counting::CountingFileSystem::new(
            Arc::new(local_fs().await),
            counters.clone(),
        );
        assert_eq!(
            publish_centroid_directory(&counting, &dir_url)
                .await
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            counters
                .range_reads
                .load(std::sync::atomic::Ordering::Relaxed),
            2,
            "untrained segment: prefix + footer, the prefix read once"
        );
        let dir = &newest(&fs, &dir_url).await[0].1;
        let l0 = dir
            .segments
            .iter()
            .find(|s| s.name == "l0.pax")
            .expect("untrained segment stamped");
        assert!(!l0.is_covered(), "no A0 ⇒ not addressable");
        let key = l0.decode_key.as_ref().expect("untrained decode key");
        assert!(
            key.rabitq_header.is_empty() && key.a0_seed == 0 && key.a0_trained_on == 0,
            "zero-encoded A0 fields"
        );
        // The key's facts equal the segment's own footer.
        let bytes = fs.read(&format!("{dir_url}/l0.pax")).await.unwrap();
        let real = SegmentFooterIndex::locate_in_segment(&bytes)
            .expect("footer locates")
            .expect("coalesced footer present");
        assert_eq!(key.row_count, real.row_count);
        assert_eq!(key.row_count, 300);
        assert_eq!(key.has_f32_tier, real.has_f32_tier);
        assert_eq!(key.embed_quant_tag, real.embed_quant_tag);
        assert_eq!(key.blocks, real.blocks);
        assert_eq!(key.block_stats, real.block_stats);
        // A legacy/garbage object stays uncovered AND keyless.
        let garbage = dir
            .segments
            .iter()
            .find(|s| s.name == "garbage.pax")
            .expect("garbage stamped");
        assert!(!garbage.is_covered() && garbage.decode_key.is_none());

        // Nothing changed: no new generation.
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }
}
