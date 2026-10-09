//! Topic / partition routing.

use std::hash::Hasher;

pub type PartitionId = u32;

/// Queue topics are exact path components, never relative or absolute paths.
pub(crate) fn validate_topic(topic: &str) -> crate::Result<()> {
    if topic.is_empty()
        || topic.len() > 255
        || matches!(topic, "." | "..")
        || topic.ends_with('.')
        || !topic
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
    {
        return Err(crate::QueueError::Persistence(format!(
            "invalid queue topic component {topic:?}"
        )));
    }
    Ok(())
}

/// Preserve historical spelling, rejecting actual case aliases rather than
/// lowercasing identities. Check before creating, then verify the persisted
/// spelling after creation so case-insensitive aliases cannot pass a race.
pub(crate) async fn ensure_exact_directory(
    fs: &std::sync::Arc<dyn crate::fs::QueueFs>,
    directory: &std::path::Path,
) -> crate::Result<()> {
    let parent = directory
        .parent()
        .ok_or_else(|| crate::QueueError::Persistence("queue directory has no parent".into()))?;
    let name = directory
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| crate::QueueError::Persistence("invalid queue directory identity".into()))?;
    for create in [false, true] {
        if create {
            fs.create_dir_all(directory).await?;
        }
        let entries = match fs.list(parent).await {
            Ok(entries) => entries,
            Err(crate::QueueError::NotFound(_)) if !create => Vec::new(),
            Err(error) => return Err(error),
        };
        let mut exact = false;
        for entry in entries {
            let first = entry
                .strip_prefix(parent)
                .ok()
                .and_then(|p| p.components().next());
            if let Some(std::path::Component::Normal(actual)) = first
                && let Some(actual) = actual.to_str()
            {
                if actual != name && actual.eq_ignore_ascii_case(name) {
                    return Err(crate::QueueError::Persistence(format!(
                        "queue directory {name:?} aliases existing identity {actual:?}"
                    )));
                }
                exact |= actual == name;
            }
        }
        if exact {
            return Ok(());
        }
    }
    Err(crate::QueueError::Persistence(format!(
        "queue filesystem did not list the exact directory identity {directory:?}"
    )))
}

/// Deterministic partition selector. Same `tenant_id` always lands on the
/// same partition for a given `partition_count`, preserving per-tenant FIFO.
///
/// Uses `xxhash3` (`twox-hash` crate) modulo partition count. xxhash gives
/// uniform distribution on string inputs with negligible per-call cost.
pub fn partition_for(tenant_id: &str, partition_count: u32) -> PartitionId {
    debug_assert!(partition_count > 0, "partition_count must be > 0");
    let mut hasher = twox_hash::XxHash64::with_seed(0);
    hasher.write(tenant_id.as_bytes());
    (hasher.finish() % partition_count as u64) as PartitionId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_tenant_always_lands_on_same_partition() {
        for n in [1, 4, 16, 64] {
            let a = partition_for("tenant-acme", n);
            let b = partition_for("tenant-acme", n);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn distribution_spreads_reasonably_across_partitions() {
        // 1024 distinct tenants over 16 partitions — every partition should
        // see at least one tenant. (Uniform distribution would give ~64 each
        // by pigeonhole; we just guard against pathological hashing.)
        let pc = 16u32;
        let mut counts = vec![0usize; pc as usize];
        for i in 0..1024u32 {
            let t = format!("tenant-{i}");
            counts[partition_for(&t, pc) as usize] += 1;
        }
        for (idx, c) in counts.iter().enumerate() {
            assert!(
                *c > 0,
                "partition {idx} got no tenants — hash distribution broken"
            );
        }
    }
}
