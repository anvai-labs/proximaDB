//! TD-SEAM-12 hazard 2: the durable branch merge-base.
//!
//! A branch merge needs the main-line history up to the branch point (the
//! merge base, `graph::merge`). Truncating the canonical WAL would delete it,
//! so before each truncation this store keeps, in the WAL itself, a **base
//! image** per branched `(tenant, collection)`: the latest untagged upsert per
//! oid as of `through = T`, where `T` is one below the oldest retained
//! branch-tagged entry. Truncation then stops at `T + 1`, and a merge computes
//! its base as image(T) plus the retained untagged upserts in `(T, M]`, which
//! is exactly what it computes from the full history.
//!
//! The rows live in the synthetic `__proxima_branch_bases__` collection, one
//! **generation** per checkpoint:
//!
//! * one `marks` row: every collection whose history some truncation removed,
//!   with the first sequence it lost, so a merge of a collection branched only
//!   after that refuses instead of using an incomplete base;
//! * `image` rows: an image in chunks;
//! * a `header`, written LAST, naming the generation complete.
//!
//! Replay uses the newest complete generation; rows of a generation with no
//! header (a checkpoint that crashed part-way) are ignored. A checkpoint is
//! written only when something changed or the previous one would hold the
//! truncation floor, and runs under the store's write gate.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use serde::{Deserialize, Serialize};

use crate::graph::merge::MergeBaseImage;
use crate::services::canonical_wal::{FramedTableWalAppender, TenantCollection};
use crate::services::record_store::TableWalAppender;

/// Synthetic collection holding the merge-base rows in the shared canonical WAL.
pub const BRANCH_BASES_COLLECTION_ID: &str = "__proxima_branch_bases__";

/// Records per image row.
const IMAGE_CHUNK_RECORDS: usize = 256;
/// Rows per WAL append (one sync each).
const ROWS_PER_APPEND: usize = 64;

const KIND_MARKS: &str = "marks";
const KIND_IMAGE: &str = "image";
const KIND_HEADER: &str = "header";

/// A merge cannot get a complete main-line base: the history it needs was
/// truncated and no image covers it. Surfaced as HTTP 410.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeBaseTruncated(pub String);

impl std::fmt::Display for MergeBaseTruncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "branch merge base is truncated: {}", self.0)
    }
}

impl std::error::Error for MergeBaseTruncated {}

/// Where a merge takes its main-line base from.
#[derive(Debug, Clone, Copy)]
pub enum MergeBase<'a> {
    /// The retained WAL holds the collection's whole history.
    Retained,
    /// The image, plus the retained history above its `through`.
    Image(&'a MergeBaseImage),
}

#[derive(Debug, Serialize, Deserialize)]
struct HeaderPayload {
    through: u64,
    /// Rows of this generation before the header.
    rows: u64,
    /// `retained_from` when the first generation was written: history below
    /// it was truncated before anything tracked it (`0` = nothing was).
    tracked_from: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImagePayload {
    tenant: String,
    collection: String,
    records: Vec<ProximaRecord>,
    /// [`MergeBaseImage::forks`]; carried by an image's first row.
    #[serde(default)]
    forks: BTreeMap<String, u64>,
}

/// One truncation mark: history of `(tenant, collection)` from `first` on
/// was removed.
#[derive(Debug, Serialize, Deserialize)]
struct MarkPayload {
    tenant: String,
    collection: String,
    first: u64,
}

/// The newest complete generation, replayed from the WAL.
#[derive(Debug, Clone, Default)]
pub struct BranchBases {
    /// Generation of the newest complete checkpoint; `0` = none.
    generation: u64,
    /// Highest generation on any row, complete or not: the next checkpoint
    /// uses one above it, so it never mixes with a crashed one's rows.
    max_generation: u64,
    /// Sequence of the first row of `generation`.
    first_row_seq: Option<u64>,
    /// See [`HeaderPayload::tracked_from`].
    tracked_from: u64,
    /// `(tenant, collection)` → first sequence some truncation removed of it.
    truncated: BTreeMap<TenantCollection, u64>,
    images: BTreeMap<TenantCollection, MergeBaseImage>,
}

#[derive(Default)]
struct PendingGeneration {
    rows: u64,
    first_row_seq: Option<u64>,
    truncated: BTreeMap<TenantCollection, u64>,
    images: BTreeMap<TenantCollection, (Vec<ProximaRecord>, BTreeMap<String, u64>)>,
}

impl BranchBases {
    /// Rebuild from canonical WAL entries; entries of other collections are
    /// skipped. Fails on an unreadable row, or when the newest header names a
    /// generation whose rows are not all present.
    pub fn replay(entries: &[CanonicalWalEntry]) -> Result<Self> {
        let mut out = Self::default();
        let mut pending: BTreeMap<u64, PendingGeneration> = BTreeMap::new();
        let mut newest_header = 0;
        for entry in entries {
            let CanonicalOperation::RecordUpsert {
                collection_id,
                record,
                ..
            } = &entry.operation
            else {
                continue;
            };
            if collection_id != BRANCH_BASES_COLLECTION_ID {
                continue;
            }
            let (kind, generation, payload) = decode_row(record)
                .with_context(|| format!("merge-base row at sequence {}", entry.sequence_number))?;
            out.max_generation = out.max_generation.max(generation);
            let generation_rows = pending.entry(generation).or_default();
            let seq = entry.sequence_number;
            generation_rows.first_row_seq =
                Some(generation_rows.first_row_seq.map_or(seq, |f| f.min(seq)));
            match kind.as_str() {
                KIND_MARKS => {
                    generation_rows.rows += 1;
                    let marks: Vec<MarkPayload> = rmp_serde::from_slice(&payload)
                        .context("decoding a merge-base marks row")?;
                    generation_rows.truncated = marks
                        .into_iter()
                        .map(|m| {
                            let key = TenantCollection {
                                tenant: m.tenant,
                                collection: m.collection,
                            };
                            (key, m.first)
                        })
                        .collect();
                }
                KIND_IMAGE => {
                    generation_rows.rows += 1;
                    let chunk: ImagePayload = rmp_serde::from_slice(&payload)
                        .context("decoding a merge-base image row")?;
                    let (records, forks) = generation_rows
                        .images
                        .entry(TenantCollection {
                            tenant: chunk.tenant,
                            collection: chunk.collection,
                        })
                        .or_default();
                    records.extend(chunk.records);
                    forks.extend(chunk.forks);
                }
                KIND_HEADER => {
                    let header: HeaderPayload = rmp_serde::from_slice(&payload)
                        .context("decoding a merge-base header row")?;
                    newest_header = newest_header.max(generation);
                    let Some(complete) = pending.remove(&generation) else {
                        continue;
                    };
                    if complete.rows != header.rows || generation < out.generation {
                        // An older generation whose first rows were truncated
                        // once a newer one superseded it.
                        continue;
                    }
                    pending.retain(|g, _| *g > generation);
                    out.generation = generation;
                    out.first_row_seq = complete.first_row_seq;
                    out.tracked_from = header.tracked_from;
                    out.truncated = complete.truncated;
                    out.images = complete
                        .images
                        .into_iter()
                        .map(|(key, (records, forks))| {
                            let image = MergeBaseImage {
                                through: header.through,
                                records: records
                                    .into_iter()
                                    .map(|record| (record.oid.clone(), record))
                                    .collect(),
                                forks,
                            };
                            (key, image)
                        })
                        .collect();
                }
                other => bail!("unknown merge-base row kind '{other}'"),
            }
        }
        if newest_header > out.generation {
            bail!(
                "merge-base generation {newest_header} has a header but not all its rows; the \
                 canonical WAL is damaged"
            );
        }
        Ok(out)
    }

    /// The image of `collection`, if the newest generation holds one.
    pub fn image(&self, collection: &TenantCollection) -> Option<&MergeBaseImage> {
        self.images.get(collection)
    }

    /// Where a merge of `collection` takes its base from, given the first
    /// sequence the WAL still holds (`0` = never truncated). Refuses when the
    /// history the base needs is gone and no image covers it, or when it
    /// cannot be told whether it is gone.
    pub fn base_for(
        &self,
        collection: &TenantCollection,
        retained_from: u64,
    ) -> std::result::Result<MergeBase<'_>, MergeBaseTruncated> {
        if retained_from == 0 {
            return Ok(MergeBase::Retained);
        }
        if let Some(image) = self.images.get(collection) {
            if image.through.saturating_add(1) < retained_from {
                return Err(MergeBaseTruncated(format!(
                    "the image of {}/{} runs through {} but the WAL starts at {retained_from}",
                    collection.tenant, collection.collection, image.through
                )));
            }
            return Ok(MergeBase::Image(image));
        }
        if self.generation == 0 || self.tracked_from > 0 {
            return Err(MergeBaseTruncated(format!(
                "canonical WAL history before {} was truncated before merge bases were tracked",
                if self.generation == 0 {
                    retained_from
                } else {
                    self.tracked_from
                }
            )));
        }
        if let Some(first) = self.truncated.get(collection)
            && *first < retained_from
        {
            return Err(MergeBaseTruncated(format!(
                "history of collection '{}' (tenant '{}') from sequence {first} was truncated \
                 before it was branched; register the branch before truncation (BranchRef)",
                collection.collection, collection.tenant
            )));
        }
        Ok(MergeBase::Retained)
    }
}

fn prop_string(record: &ProximaRecord, key: &str) -> Result<String> {
    match record.props.get(key) {
        Some(ProximaTreeNode::Value(ProximaValue::String(s))) => Ok(s.clone()),
        _ => bail!("missing string prop '{key}'"),
    }
}

fn decode_row(record: &ProximaRecord) -> Result<(String, u64, Vec<u8>)> {
    let kind = prop_string(record, "kind")?;
    let generation = match record.props.get("generation") {
        Some(ProximaTreeNode::Value(ProximaValue::UInt64(g))) => *g,
        _ => bail!("missing prop 'generation'"),
    };
    let payload = match record.props.get("payload") {
        Some(ProximaTreeNode::Value(ProximaValue::Binary(b))) => b.clone(),
        _ => bail!("missing prop 'payload'"),
    };
    Ok((kind, generation, payload))
}

fn encode_row(kind: &str, generation: u64, index: usize, payload: Vec<u8>) -> CanonicalOperation {
    let mut props = HashMap::new();
    props.insert(
        "kind".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(kind.to_string())),
    );
    props.insert(
        "generation".to_string(),
        ProximaTreeNode::Value(ProximaValue::UInt64(generation)),
    );
    props.insert(
        "payload".to_string(),
        ProximaTreeNode::Value(ProximaValue::Binary(payload)),
    );
    CanonicalOperation::RecordUpsert {
        collection_id: BRANCH_BASES_COLLECTION_ID.to_string(),
        record: Box::new(ProximaRecord {
            oid: format!("{kind}:{generation}:{index}"),
            variation_id: Some(BRANCH_BASES_COLLECTION_ID.to_string()),
            props,
            ..Default::default()
        }),
        projections: Vec::new(),
    }
}

/// Keeps the branch merge-base images and truncation marks for the shared
/// canonical WAL; [`Self::checkpoint`] is the truncator's floor source.
pub struct BranchBaseStore {
    wal: Arc<FramedTableWalAppender>,
    /// `None` until the first checkpoint replays the store's rows. The mutex
    /// is the write gate: one checkpoint at a time, so a generation's rows
    /// never interleave with another's.
    state: tokio::sync::Mutex<Option<BranchBases>>,
}

impl BranchBaseStore {
    pub fn new(wal: Arc<FramedTableWalAppender>) -> Self {
        Self {
            wal,
            state: tokio::sync::Mutex::new(None),
        }
    }

    /// Make the merge-base durable for a truncation run with WAL head `head`
    /// and return the floor it imposes: entries from the oldest retained
    /// branch-tagged record up (merges read them), and the newest generation's
    /// rows. On the way:
    ///
    /// * marks every collection the truncation may strip;
    /// * images every branched `(tenant, collection)` whose history is
    ///   complete, through one below the oldest branch-tagged record;
    /// * arms the WAL's branch fence with the branched set it saw, so a
    ///   collection branched before the truncation runs refuses it.
    ///
    /// Fails — refusing the run — when the WAL cannot be read or appended, or
    /// an image no longer reaches the retained history.
    pub async fn checkpoint(&self, head: u64) -> Result<Option<u64>> {
        let mut guard = self.state.lock().await;
        if guard.is_none() {
            let rows = self
                .wal
                .read_entries_since(BRANCH_BASES_COLLECTION_ID, 0)
                .await
                .context("reading the merge-base rows")?;
            *guard = Some(BranchBases::replay(&rows)?);
        }
        let state = guard.as_mut().context("merge-base state not loaded")?;

        let summary = self.wal.branch_summary();
        let retained_from = self.wal.retained_from();
        let floor = summary
            .oldest_branch_seq
            .unwrap_or(u64::MAX)
            .min(head.saturating_add(1));
        let through = floor.saturating_sub(1);

        let below = self.wal.collections_below(floor);
        let mut truncated = state.truncated.clone();
        for (collection, first) in &below {
            if collection.collection == BRANCH_BASES_COLLECTION_ID {
                continue; // its own rows: superseded by every generation
            }
            let mark = truncated.entry(collection.clone()).or_insert(*first);
            *mark = (*mark).min(*first);
        }

        let mut images = BTreeMap::new();
        for collection in &summary.collections {
            let (mut image, fresh) = match state.base_for(collection, retained_from) {
                Ok(MergeBase::Image(image)) if image.through > through => bail!(
                    "the merge-base image of {}/{} runs through {}, past the new base {through}",
                    collection.tenant,
                    collection.collection,
                    image.through
                ),
                Ok(MergeBase::Image(image)) => (image.clone(), false),
                Ok(MergeBase::Retained) => (MergeBaseImage::default(), true),
                // Its base is already gone: it stays unimaged and its merges
                // refuse (410).
                Err(_) => continue,
            };
            if fresh || image.through < through {
                let entries: Vec<CanonicalWalEntry> = self
                    .wal
                    .read_entries_since(&collection.collection, image.through)
                    .await
                    .with_context(|| {
                        format!(
                            "reading {}/{} for its merge-base image",
                            collection.tenant, collection.collection
                        )
                    })?
                    .into_iter()
                    .filter(|entry| is_entry_of(entry, collection))
                    .collect();
                image.advance(&entries, through);
            }
            images.insert(collection.clone(), image);
        }

        let changed = state.generation == 0
            || truncated != state.truncated
            || images != state.images
            // The previous rows would be the binding floor: move them up.
            || state.first_row_seq.is_some_and(|first| first < floor);
        if changed {
            let tracked_from = if state.generation == 0 {
                retained_from
            } else {
                state.tracked_from
            };
            let generation = state.max_generation + 1;
            // Claimed before writing: a write that fails part-way leaves rows
            // of `generation` without a header, and the retry must not reuse
            // the number (replay would count both attempts' rows against one
            // header and refuse every later replay).
            state.max_generation = generation;
            let first_row_seq = self
                .write_generation(generation, through, tracked_from, &truncated, &images)
                .await?;
            *state = BranchBases {
                generation,
                max_generation: generation,
                first_row_seq: Some(first_row_seq),
                tracked_from,
                truncated,
                images,
            };
        }

        // Everything below the floor is now recorded (this store's own rows
        // need no mark: every generation supersedes them).
        self.wal
            .fence_branches(summary.collections, below.into_keys().collect());
        Ok(Some(
            state.first_row_seq.map_or(floor, |first| first.min(floor)),
        ))
    }

    /// Append one generation — marks, image chunks, then the header — and
    /// return the sequence of its first row.
    async fn write_generation(
        &self,
        generation: u64,
        through: u64,
        tracked_from: u64,
        truncated: &BTreeMap<TenantCollection, u64>,
        images: &BTreeMap<TenantCollection, MergeBaseImage>,
    ) -> Result<u64> {
        let marks: Vec<MarkPayload> = truncated
            .iter()
            .map(|(key, first)| MarkPayload {
                tenant: key.tenant.clone(),
                collection: key.collection.clone(),
                first: *first,
            })
            .collect();
        let mut rows = vec![encode_row(
            KIND_MARKS,
            generation,
            0,
            rmp_serde::to_vec_named(&marks).context("encoding merge-base marks")?,
        )];
        for (collection, image) in images {
            let mut records: Vec<&ProximaRecord> = image.records.values().collect();
            records.sort_by(|a, b| a.oid.cmp(&b.oid));
            // An empty image still gets a row: its absence would read as
            // "not imaged".
            let mut chunks: Vec<&[&ProximaRecord]> = records.chunks(IMAGE_CHUNK_RECORDS).collect();
            if chunks.is_empty() {
                chunks.push(&[]);
            }
            for (n, chunk) in chunks.into_iter().enumerate() {
                let payload = ImagePayload {
                    tenant: collection.tenant.clone(),
                    collection: collection.collection.clone(),
                    records: chunk.iter().map(|r| (*r).clone()).collect(),
                    forks: if n == 0 {
                        image.forks.clone()
                    } else {
                        BTreeMap::new()
                    },
                };
                let index = rows.len();
                rows.push(encode_row(
                    KIND_IMAGE,
                    generation,
                    index,
                    rmp_serde::to_vec_named(&payload).context("encoding a merge-base image")?,
                ));
            }
        }
        let header = HeaderPayload {
            through,
            rows: rows.len() as u64,
            tracked_from,
        };
        let header = encode_row(
            KIND_HEADER,
            generation,
            rows.len(),
            rmp_serde::to_vec_named(&header).context("encoding a merge-base header")?,
        );

        let mut first = None;
        let mut rows = rows.into_iter().peekable();
        while rows.peek().is_some() {
            let batch: Vec<CanonicalOperation> = rows.by_ref().take(ROWS_PER_APPEND).collect();
            let written = self
                .wal
                .append_operations(batch, None)
                .await
                .context("appending merge-base rows")?;
            first = first.or(written.first().map(|e| e.sequence_number));
        }
        // Last, and only once every row is durable.
        self.wal
            .append_operations(vec![header], None)
            .await
            .context("appending the merge-base header")?;
        first.context("a merge-base generation always has a marks row")
    }
}

/// Whether `entry` is a record operation of `collection` — upserts, deletes
/// and partition drops all shape its main-line state.
fn is_entry_of(entry: &CanonicalWalEntry, collection: &TenantCollection) -> bool {
    let entry_collection = match &entry.operation {
        CanonicalOperation::RecordUpsert { collection_id, .. }
        | CanonicalOperation::RecordDelete { collection_id, .. }
        | CanonicalOperation::RecordPartitionDrop { collection_id } => collection_id,
        _ => return false,
    };
    *entry_collection == collection.collection
        && proximadb_tenant::resolve_request_tenant(entry.tenant_id.as_deref()) == collection.tenant
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upsert(collection: &str, oid: &str, branch: Option<&str>, name: &str) -> CanonicalOperation {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(name.to_string())),
        );
        CanonicalOperation::RecordUpsert {
            collection_id: collection.to_string(),
            record: Box::new(ProximaRecord {
                oid: oid.to_string(),
                branch_id: branch.map(str::to_string),
                props,
                ..Default::default()
            }),
            projections: Vec::new(),
        }
    }

    async fn append(wal: &FramedTableWalAppender, op: CanonicalOperation) -> Result<u64> {
        Ok(wal.append_operations(vec![op], None).await?[0].sequence_number)
    }

    fn key(collection: &str) -> TenantCollection {
        TenantCollection::new(None, collection)
    }

    async fn replayed(wal: &FramedTableWalAppender) -> Result<BranchBases> {
        BranchBases::replay(&wal.read_entries().await?)
    }

    /// `c` gets main writes in sealed segments, then branch `x` at seq 4.
    async fn branched_wal(dir: &tempfile::TempDir) -> Result<Arc<FramedTableWalAppender>> {
        let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("c.wal")).await?);
        wal.enable_segment_rotation(1);
        append(&wal, upsert("c", "n1", None, "v0")).await?;
        append(&wal, upsert("c", "n2", None, "v0")).await?;
        append(&wal, upsert("c", "n1", None, "v1")).await?;
        append(&wal, upsert("c", "n1", Some("x"), "left")).await?;
        Ok(wal)
    }

    #[tokio::test]
    async fn checkpoint_images_a_branched_collection_and_floors_at_its_branch() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = branched_wal(&dir).await?;
        let store = BranchBaseStore::new(wal.clone());

        let floor = store.checkpoint(wal.last_sequence()).await?;
        assert_eq!(floor, Some(4), "the oldest branch-tagged entry stays");
        wal.truncate_below(4).await?;
        assert_eq!(wal.retained_from(), 4);

        let bases = replayed(&wal).await?;
        let image = bases.image(&key("c")).expect("c is branched");
        assert_eq!(image.through, 3);
        assert_eq!(image.records.len(), 2);
        assert!(matches!(
            image.records["n1"].props.get("name"),
            Some(ProximaTreeNode::Value(ProximaValue::String(v))) if v == "v1"
        ));
        assert!(matches!(
            bases.base_for(&key("c"), wal.retained_from()),
            Ok(MergeBase::Image(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn an_unchanged_checkpoint_appends_nothing() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = branched_wal(&dir).await?;
        let store = BranchBaseStore::new(wal.clone());
        store.checkpoint(wal.last_sequence()).await?;
        let head = wal.last_sequence();
        assert_eq!(store.checkpoint(head).await?, Some(4));
        assert_eq!(wal.last_sequence(), head);

        // A restarted store replays its rows and still has nothing to add.
        let restarted = BranchBaseStore::new(wal.clone());
        assert_eq!(restarted.checkpoint(head).await?, Some(4));
        assert_eq!(wal.last_sequence(), head);
        Ok(())
    }

    #[tokio::test]
    async fn without_branches_the_marks_move_up_with_the_head() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("c.wal")).await?);
        wal.enable_segment_rotation(1);
        append(&wal, upsert("c", "n1", None, "v0")).await?;
        let store = BranchBaseStore::new(wal.clone());

        let head = wal.last_sequence();
        let floor = store.checkpoint(head).await?.expect("rows were written");
        assert!(
            floor > head,
            "only this generation's own rows constrain the floor"
        );
        wal.truncate_below(floor).await?;

        // The next run re-appends, so its old rows never pin the WAL.
        append(&wal, upsert("d", "n1", None, "v0")).await?;
        let head = wal.last_sequence();
        let next = store.checkpoint(head).await?.expect("rows were written");
        assert!(next > head);
        wal.truncate_below(next).await?;

        // `c` lost history before it was ever branched: its merge refuses.
        let bases = replayed(&wal).await?;
        let refused = bases.base_for(&key("c"), wal.retained_from()).unwrap_err();
        assert!(refused.0.contains("'c'"), "{refused}");
        assert!(matches!(
            bases.base_for(&key("fresh"), wal.retained_from()),
            Ok(MergeBase::Retained)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn replay_ignores_rows_of_a_generation_without_its_header() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = branched_wal(&dir).await?;
        let store = BranchBaseStore::new(wal.clone());
        store.checkpoint(wal.last_sequence()).await?;
        let complete = replayed(&wal).await?;
        assert_eq!(complete.generation, 1);

        // A crashed checkpoint: generation 2's rows, no header.
        let crashed = ImagePayload {
            tenant: key("c").tenant,
            collection: "c".to_string(),
            records: vec![ProximaRecord {
                oid: "bogus".to_string(),
                ..Default::default()
            }],
            forks: BTreeMap::new(),
        };
        append(
            &wal,
            encode_row(
                KIND_MARKS,
                2,
                0,
                rmp_serde::to_vec_named(&Vec::<MarkPayload>::new())?,
            ),
        )
        .await?;
        append(
            &wal,
            encode_row(KIND_IMAGE, 2, 1, rmp_serde::to_vec_named(&crashed)?),
        )
        .await?;

        let bases = replayed(&wal).await?;
        assert_eq!(bases.generation, 1);
        assert_eq!(bases.image(&key("c")), complete.image(&key("c")));
        assert_eq!(bases.max_generation, 2);

        // The next generation numbers past the crashed one.
        let head = wal.last_sequence();
        append(&wal, upsert("e", "n9", Some("y"), "late")).await?;
        let store = BranchBaseStore::new(wal.clone());
        store.checkpoint(head + 1).await?;
        assert_eq!(replayed(&wal).await?.generation, 3);
        Ok(())
    }

    #[tokio::test]
    async fn a_header_without_all_its_rows_fails_replay() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = FramedTableWalAppender::open(dir.path().join("c.wal")).await?;
        let header = HeaderPayload {
            through: 3,
            rows: 2,
            tracked_from: 0,
        };
        append(
            &wal,
            encode_row(
                KIND_MARKS,
                1,
                0,
                rmp_serde::to_vec_named(&Vec::<MarkPayload>::new())?,
            ),
        )
        .await?;
        append(
            &wal,
            encode_row(KIND_HEADER, 1, 2, rmp_serde::to_vec_named(&header)?),
        )
        .await?;
        let error = replayed(&wal).await.unwrap_err();
        assert!(
            format!("{error:#}").contains("not all its rows"),
            "{error:#}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn truncation_before_tracking_refuses_unimaged_merges() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("c.wal")).await?);
        wal.enable_segment_rotation(1);
        append(&wal, upsert("c", "n1", None, "v0")).await?;
        append(&wal, upsert("c", "n2", None, "v0")).await?;
        // Truncated by something that kept no merge base.
        wal.truncate_below(2).await?;
        append(&wal, upsert("c", "n1", Some("x"), "left")).await?;

        assert!(
            BranchBases::default()
                .base_for(&key("c"), wal.retained_from())
                .is_err(),
            "no generation at all: whether history is missing is unknown"
        );
        let store = BranchBaseStore::new(wal.clone());
        store.checkpoint(wal.last_sequence()).await?;
        let bases = replayed(&wal).await?;
        assert!(
            bases.image(&key("c")).is_none(),
            "its base is already incomplete"
        );
        assert!(bases.base_for(&key("c"), wal.retained_from()).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn an_empty_image_survives_replay_and_marks_are_per_tenant() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("c.wal")).await?);
        wal.enable_segment_rotation(1);
        // t2's `c` has main-line history; t1's `c` only a branch write.
        wal.append_operations(vec![upsert("c", "n1", None, "v0")], Some("t2".into()))
            .await?;
        wal.append_operations(vec![upsert("c", "n1", Some("x"), "b")], Some("t1".into()))
            .await?;
        let store = BranchBaseStore::new(wal.clone());
        assert_eq!(store.checkpoint(wal.last_sequence()).await?, Some(2));
        wal.truncate_below(2).await?;

        let bases = replayed(&wal).await?;
        let t1 = TenantCollection::new(Some("t1"), "c");
        let image = bases.image(&t1).expect("an empty image is still an image");
        assert!(image.records.is_empty());
        assert!(matches!(
            bases.base_for(&t1, wal.retained_from()),
            Ok(MergeBase::Image(_))
        ));
        // t2's `c` lost history and has no image; t3's `c` never had any.
        let t2 = TenantCollection::new(Some("t2"), "c");
        assert!(bases.base_for(&t2, wal.retained_from()).is_err());
        assert!(matches!(
            bases.base_for(&TenantCollection::new(Some("t3"), "c"), wal.retained_from()),
            Ok(MergeBase::Retained)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn a_partition_drop_reaches_the_image() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("c.wal")).await?);
        wal.enable_segment_rotation(1);
        append(&wal, upsert("c", "n1", None, "v0")).await?;
        // The drop sits alone in its segment: the segment summary must still
        // index it under `c`, or the image read skips it.
        append(
            &wal,
            CanonicalOperation::RecordPartitionDrop {
                collection_id: "c".to_string(),
            },
        )
        .await?;
        append(&wal, upsert("c", "n2", None, "v0")).await?;
        append(&wal, upsert("c", "n1", Some("x"), "left")).await?;
        let store = BranchBaseStore::new(wal.clone());
        assert_eq!(store.checkpoint(wal.last_sequence()).await?, Some(4));
        wal.truncate_below(4).await?;

        let bases = replayed(&wal).await?;
        let image = bases.image(&key("c")).expect("c is branched");
        assert_eq!(
            image.records.keys().collect::<Vec<_>>(),
            vec!["n2"],
            "n1 was dropped before the branch point"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_checkpoint_never_reuses_its_generation() -> Result<()> {
        use std::time::{Duration, Instant};
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("c.wal");
        drop(branched_wal(&dir).await?);
        let fence = Arc::new(crate::services::writer_fence::WriterFence::default());
        let wal = Arc::new(
            FramedTableWalAppender::open(&path)
                .await?
                .with_writer_fence(fence.clone()),
        );
        let store = BranchBaseStore::new(wal.clone());

        // An expired lease fails the write.
        fence.enforce_lease(Instant::now(), Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(store.checkpoint(wal.last_sequence()).await.is_err());

        // The retry writes the next generation, and replay stays sound.
        fence.renew_lease(Instant::now(), Duration::from_secs(600));
        store.checkpoint(wal.last_sequence()).await?;
        let bases = replayed(&wal).await?;
        assert_eq!(bases.generation, 2);
        assert!(bases.image(&key("c")).is_some());
        Ok(())
    }
}
