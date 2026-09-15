//! Inclusive committed offsets in the existing JSON format. Publication checks
//! prior offset bytes and current lease bytes under one directory lock.
//! No multi-file write transaction, new authority, or format migration.

use crate::error::QueueError;
use crate::fs::QueueFs;
use crate::topic::PartitionId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OffsetMeta {
    pub group: String,
    pub committed_offset: u64,
}
const META_FILE: &str = "offset.meta";

/// Retain canonical paths; reject rather than silently sanitize aliases.
/// Existing ASCII spelling is preserved; acquisition rejects directory case aliases.
/// Noncanonical historical names require explicit operator migration.
pub(crate) fn validate_group(group: &str) -> crate::Result<()> {
    if group.is_empty()
        || group.len() > 255
        || group.starts_with(['.', '_'])
        || group.ends_with(['.', '_'])
        || !group
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
    {
        return Err(QueueError::Persistence(format!(
            "noncanonical consumer group {group:?}; use exact ASCII without edge dots/underscores"
        )));
    }
    Ok(())
}

// Callers validate before I/O. No lossy remapping.
pub(crate) fn group_dir_name(group: &str) -> String {
    group.to_string()
}

fn meta_path(root: &Path, topic: &str, partition: PartitionId, group: &str) -> PathBuf {
    root.join(topic)
        .join(partition.to_string())
        .join(group)
        .join(META_FILE)
}

fn decode(bytes: &[u8], group: &str) -> crate::Result<u64> {
    let parsed: OffsetMeta = serde_json::from_slice(bytes)
        .map_err(|e| QueueError::Persistence(format!("offset_store parse: {e}")))?;
    if parsed.group != group {
        return Err(QueueError::Persistence(
            "offset metadata group identity mismatch".into(),
        ));
    }
    Ok(parsed.committed_offset)
}

async fn read_bytes(fs: &Arc<dyn QueueFs>, path: &Path) -> crate::Result<Option<Vec<u8>>> {
    match fs.read(path).await {
        Ok(bytes) => Ok(Some(bytes)),
        Err(QueueError::NotFound(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

/// One inclusive contiguous ACK watermark, guarded by its admitted owner's bytes.
/// Expiry is eligibility at admission, not at rename. Takeover shares this lock.
pub async fn commit(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder: &str,
    committed_offset: u64,
) -> crate::Result<()> {
    validate_group(group)?;
    crate::topic::validate_topic(topic)?;
    let path = meta_path(root, topic, partition, group);
    for _ in 0..8 {
        let (lease_path, lease_bytes) =
            crate::leases::owned_bytes(fs, root, topic, partition, group, holder).await?;
        let observed = read_bytes(fs, &path).await?;
        let previous = observed
            .as_deref()
            .map(|bytes| decode(bytes, group))
            .transpose()?;
        let body = OffsetMeta {
            group: group.into(),
            committed_offset: previous.map_or(committed_offset, |v| v.max(committed_offset)),
        };
        let bytes =
            serde_json::to_vec(&body).map_err(|e| QueueError::Persistence(e.to_string()))?;
        if fs
            .compare_exchange(
                &path,
                observed.as_deref(),
                &bytes,
                &[(lease_path.as_path(), Some(lease_bytes.as_slice()))],
            )
            .await
            .map_err(|e| QueueError::PublicationIndeterminate(e.to_string()))?
        {
            return Ok(());
        }
        // Renewal changes lease bytes too; reread/revalidate, never reuse the guard.
        tokio::task::yield_now().await;
    }
    Err(QueueError::Persistence(
        "offset publication contention retry budget exhausted".into(),
    ))
}

pub async fn read(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
) -> crate::Result<Option<u64>> {
    validate_group(group)?;
    crate::topic::validate_topic(topic)?;
    read_bytes(fs, &meta_path(root, topic, partition, group))
        .await?
        .as_deref()
        .map(|bytes| decode(bytes, group))
        .transpose()
}

/// Every registered group participates, including lease-only groups with no ACK.
/// None progress blocks reaping/recovery skipping. Failed reads/listing propagate.
pub async fn read_all_groups(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
) -> crate::Result<Vec<(String, Option<u64>)>> {
    crate::topic::validate_topic(topic)?;
    let directory = root.join(topic).join(partition.to_string());
    let mut groups = HashSet::new();
    for entry in fs.list(&directory).await? {
        let name = entry
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| QueueError::Persistence("invalid queue entry name".into()))?;
        if matches!(name, "offset.meta" | "lease.meta") {
            let parent = entry
                .parent()
                .ok_or_else(|| QueueError::Persistence("metadata without group".into()))?;
            if parent.parent() != Some(directory.as_path()) {
                continue;
            }
            let group = parent
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| QueueError::Persistence("invalid metadata group".into()))?;
            validate_group(group)?;
            groups.insert(group.to_string());
        } else if fs.metadata(&entry).await?.is_directory {
            if entry.parent() != Some(directory.as_path()) {
                continue;
            }
            let children = fs.list(&entry).await?;
            if children.iter().any(|p| {
                matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("offset.meta" | "lease.meta")
                )
            }) {
                validate_group(name)?;
                groups.insert(name.to_string());
            }
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let offset = read(fs, root, topic, partition, &group).await?;
        out.push((group, offset));
    }
    Ok(out)
}
