//! The Iceberg v2 discovery window for engine-owned immutable generations.

use serde_json::{Value, json};

use crate::iceberg::{FormatVersion, SnapshotCommit, name_mapping, schema_json};
use crate::retention::window_floor;

// The wire contract is the standard `snapshots` array. This property tells our
// collector how archived copies of an older head were built.
pub(crate) const WINDOW_PROPERTY: &str = "proximadb.snapshot-history";
pub(crate) const WINDOW_POLICY: &str = "generation-window-v1";

pub(crate) fn has_generation_window(metadata: &Value) -> bool {
    metadata["format-version"] == 2
        && metadata["properties"][WINDOW_PROPERTY].as_str() == Some(WINDOW_POLICY)
}

/// Recognize only the engine's generation-named list in this table's own
/// metadata directory. A similarly named file under a different root proves
/// nothing about this table's retention.
pub(crate) fn list_generation(root: &str, id: i64, list: &str) -> Option<u64> {
    let directory = format!("{}/metadata/snap-{id}-v", root.trim_end_matches('/'));
    let digits = list.strip_prefix(&directory)?.strip_suffix("-list.avro")?;
    if digits.len() != 20 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Copy descriptors from the already fetched parent. Each published manifest
/// list is a complete snapshot. Only old publications with the same v2 schema
/// and partition spec can use this head's schema and name mapping. Stop at a
/// discontinuity so a wider configuration cannot revive an expired pin.
pub(crate) fn retained_snapshots(
    commit: &SnapshotCommit<'_>,
    parent: Option<&Value>,
    generation: u64,
    retained_generations: u64,
) -> Option<Vec<Value>> {
    if commit.format_version != FormatVersion::V2
        || !commit.location.contains("://")
        || list_generation(
            commit.location,
            commit.snapshot_id,
            commit.manifest_list_location,
        ) != Some(generation)
    {
        return None;
    }

    let mut retained = Vec::new();
    let compatible = parent.filter(|doc| {
        doc["format-version"] == 2
            && doc["table-uuid"] == commit.table_uuid
            && doc["location"] == commit.location
            && doc["schemas"] == json!([schema_json(commit.fields)])
            && doc["current-schema-id"] == 0
            && doc["default-spec-id"] == 0
            && doc["partition-specs"] == json!([{"spec-id": 0, "fields": []}])
            && doc["properties"]["schema.name-mapping.default"] == name_mapping(commit.fields)
    });
    if let Some(doc) = compatible
        && let Some(snapshots) = doc["snapshots"].as_array()
    {
        // Older engine publications listed a single current snapshot. Adopt
        // that parent without reading the retired `metadata-log` documents.
        let limit = if has_generation_window(doc) || snapshots.len() == 1 {
            snapshots.len()
        } else {
            0
        };
        let floor = window_floor(generation, retained_generations);
        let mut expected_id = doc["current-snapshot-id"].as_i64();
        let mut newer_id = commit.snapshot_id;
        for (snapshot, version) in snapshots
            .iter()
            .rev()
            .take(limit)
            .zip((floor..generation).rev())
        {
            let (Some(id), Some(list)) = (
                snapshot["snapshot-id"].as_i64(),
                snapshot["manifest-list"].as_str(),
            ) else {
                break;
            };
            if Some(id) != expected_id
                || id < 0
                || id >= newer_id
                || snapshot["schema-id"] != 0
                || snapshot["timestamp-ms"].as_i64().is_none()
                || list_generation(commit.location, id, list) != Some(version)
            {
                break;
            }
            retained.push(snapshot.clone());
            expected_id = snapshot["parent-snapshot-id"].as_i64();
            newer_id = id;
        }
        retained.reverse();
    }
    Some(retained)
}
