//! Cross-process partition leases — prevents two consumer instances IN THE SAME
//! GROUP (running on different replicas) from competing for the same
//! `(topic, partition)` and producing duplicate work.
//!
//! ## Mechanism
//!
//! Leases are scoped per consumer group: each `(group, topic, partition)` has
//! a `lease.meta` file at `{queue_root}/{topic}/{partition}/{group}/lease.meta`
//! containing `{holder_id, expires_at_unix_nanos}`. Two DIFFERENT groups can
//! each hold the same partition (pub/sub fan-out); only consumers in the SAME
//! group compete. Acquisition uses exact byte-conditioned publication through
//! `QueueFs::compare_exchange`. LocalFs reuses the runtime file-lock primitive.
//!
//! ## Acquisition flow
//!
//! 1. Read the existing `lease.meta` (if any). If it exists, is not
//!    expired, and belongs to someone else → `LeaseConflict`.
//! 2. Validate ownership/expiry policy and serialize the replacement.
//! 3. Replace only if the exact inspected state still exists. A mismatch or
//!    lock contender returns conflict; unavailable/corrupt state fails closed.
//!
//! Every writer must use the conditional path; legacy rename-only workers must
//! be drained during rollout. Offset publication guards the exact lease bytes
//! under the same lock. Neither operation fences arbitrary external effects.
//!
//! ## Renewal
//!
//! `Consumer::subscribe` spawns a background renewer task that calls
//! `renew()` every `lease_duration / 2` to keep the lease alive while
//! the consumer is running. On `Consumer` drop the renewer is
//! signalled for best-effort owner-conditioned release; explicit shutdown awaits
//! that release. Runtime termination may prevent release, so expiry is necessary.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::QueueError;
use crate::fs::QueueFs;
use crate::topic::PartitionId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaseMeta {
    pub holder_id: String,
    /// Unix-epoch nanoseconds. Compared against `now_unix_nanos()` to
    /// determine eligibility using the caller's wall clock. This does not prove
    /// exclusion of in-flight work or strict expiration at publication time.
    pub expires_at_unix_nanos: u128,
}

const LEASE_FILE: &str = "lease.meta";

pub(crate) fn lease_path(root: &Path, topic: &str, partition: PartitionId, group: &str) -> PathBuf {
    root.join(topic)
        .join(partition.to_string())
        .join(crate::offset_store::group_dir_name(group))
        .join(LEASE_FILE)
}

fn now_unix_nanos() -> crate::Result<u128> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|e| QueueError::Persistence(format!("lease clock before epoch: {e}")))
}

enum LeaseUpdate {
    Acquire(Duration),
    Renew(Duration),
}

/// Acquire (or take over an expired) lease on `(group, topic, partition)` for
/// `holder_id`. Returns `LeaseConflict` if a non-expired lease for THIS group
/// is held by someone else. Different groups never conflict (pub/sub fan-out).
pub async fn try_acquire(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder_id: &str,
    lease_duration: Duration,
) -> crate::Result<()> {
    update(
        fs,
        root,
        topic,
        partition,
        group,
        holder_id,
        LeaseUpdate::Acquire(lease_duration),
    )
    .await
}

async fn update(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder_id: &str,
    update: LeaseUpdate,
) -> crate::Result<()> {
    crate::offset_store::validate_group(group)?;
    crate::topic::validate_topic(topic)?;
    let (lease_duration, renewal) = match update {
        LeaseUpdate::Acquire(duration) => (duration, false),
        LeaseUpdate::Renew(duration) => (duration, true),
    };
    if lease_duration.is_zero() || holder_id.is_empty() {
        return Err(QueueError::Persistence(
            "lease requires nonempty holder and positive duration".into(),
        ));
    }
    let path = lease_path(root, topic, partition, group);
    if let Some(parent) = path.parent() {
        crate::topic::ensure_exact_directory(fs, parent).await?;
    }
    for _ in 0..8 {
        let observed = match fs.read(&path).await {
            Ok(bytes) => Some(bytes),
            Err(QueueError::NotFound(_)) => None,
            Err(error) => return Err(error),
        };
        let now = now_unix_nanos()?;
        let existing = observed
            .as_deref()
            .map(serde_json::from_slice::<LeaseMeta>)
            .transpose()
            .map_err(|e| QueueError::Persistence(format!("corrupt lease metadata: {e}")))?;
        let conflict = |holder: String| QueueError::LeaseConflict {
            topic: topic.to_string(),
            partition,
            holder,
        };
        match existing {
            Some(existing) => {
                if existing.holder_id.is_empty() {
                    return Err(QueueError::Persistence(
                        "corrupt lease: empty holder".into(),
                    ));
                }
                let rejected = if renewal {
                    existing.holder_id != holder_id || existing.expires_at_unix_nanos <= now
                } else {
                    existing.holder_id != holder_id && existing.expires_at_unix_nanos > now
                };
                if rejected {
                    return Err(conflict(existing.holder_id));
                }
            }
            None if renewal => return Err(conflict("unowned".into())),
            None => {}
        }
        let new_meta = LeaseMeta {
            holder_id: holder_id.to_string(),
            expires_at_unix_nanos: now
                .checked_add(lease_duration.as_nanos())
                .ok_or_else(|| QueueError::Persistence("lease deadline overflow".into()))?,
        };
        let bytes = serde_json::to_vec(&new_meta)
            .map_err(|e| QueueError::Persistence(format!("lease serialize: {e}")))?;
        if !fs
            .compare_exchange(&path, observed.as_deref(), &bytes, &[])
            .await?
        {
            tokio::task::yield_now().await;
            continue;
        }
        return Ok(());
    }
    Err(QueueError::Persistence(
        "lease update contention retry budget exhausted".into(),
    ))
}

/// Authoritative admission read, not proof of strict expiry at a later rename.
/// A guarded progress write must retain these exact bytes as its read precondition.
pub(crate) async fn owned_bytes(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder: &str,
) -> crate::Result<(PathBuf, Vec<u8>)> {
    crate::offset_store::validate_group(group)?;
    crate::topic::validate_topic(topic)?;
    let path = lease_path(root, topic, partition, group);
    let bytes = fs.read(&path).await?;
    let lease: LeaseMeta = serde_json::from_slice(&bytes)
        .map_err(|e| QueueError::Persistence(format!("corrupt lease metadata: {e}")))?;
    if lease.holder_id != holder || lease.expires_at_unix_nanos <= now_unix_nanos()? {
        return Err(QueueError::LeaseConflict {
            topic: topic.into(),
            partition,
            holder: lease.holder_id,
        });
    }
    Ok((path, bytes))
}

/// Owner-conditioned tombstone, never delete authority. A displaced owner cannot
/// release a successor; uncertain errors propagate and the existing lease expires.
pub(crate) async fn release(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder: &str,
) -> crate::Result<()> {
    crate::offset_store::validate_group(group)?;
    crate::topic::validate_topic(topic)?;
    let path = lease_path(root, topic, partition, group);
    for _ in 0..8 {
        let bytes = match fs.read(&path).await {
            Ok(bytes) => bytes,
            Err(QueueError::NotFound(_)) => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut lease: LeaseMeta = serde_json::from_slice(&bytes)
            .map_err(|e| QueueError::Persistence(format!("corrupt release metadata: {e}")))?;
        if lease.holder_id != holder || lease.expires_at_unix_nanos == 0 {
            return Ok(());
        }
        lease.expires_at_unix_nanos = 0;
        let replacement =
            serde_json::to_vec(&lease).map_err(|e| QueueError::Persistence(e.to_string()))?;
        if fs
            .compare_exchange(&path, Some(&bytes), &replacement, &[])
            .await?
        {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
    Err(QueueError::Persistence(
        "lease release contention retry budget exhausted".into(),
    ))
}

/// Refresh a currently valid lease held by this holder. Missing, expired or
/// superseded ownership is a conflict, never implicit reacquisition.
pub async fn renew(
    fs: &Arc<dyn QueueFs>,
    root: &Path,
    topic: &str,
    partition: PartitionId,
    group: &str,
    holder_id: &str,
    lease_duration: Duration,
) -> crate::Result<()> {
    update(
        fs,
        root,
        topic,
        partition,
        group,
        holder_id,
        LeaseUpdate::Renew(lease_duration),
    )
    .await
}
