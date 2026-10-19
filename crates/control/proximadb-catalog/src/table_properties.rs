//! Table properties owned by the system.
//!
//! A catalog entry's table properties carry, besides user settings, the record
//! of a collection (`asset.*`, `collection.*`, `vector.*`, `stats.*`, `owner`,
//! `tags`). Those are written only by the collection writer; client-facing
//! table creation (SQL DDL, the Iceberg REST catalog) must never set them.
//! [`COLLECTION_PROVENANCE_PROPERTY`] marks an entry written by the collection
//! writer, and is itself reserved.

use std::collections::HashMap;

/// Set on every catalog entry the collection writer creates or updates.
pub const COLLECTION_PROVENANCE_PROPERTY: &str = "collection.provenance";
/// The only value of [`COLLECTION_PROVENANCE_PROPERTY`].
pub const COLLECTION_PROVENANCE_VALUE: &str = "collection-service";

/// Key prefixes reserved for system-written table properties.
pub const RESERVED_TABLE_PROPERTY_PREFIXES: &[&str] =
    &["asset.", "collection.", "vector.", "stats."];
/// Exact keys reserved for system-written table properties.
pub const RESERVED_TABLE_PROPERTY_KEYS: &[&str] = &["owner", "tags"];

/// Whether `key` is a system-written table property a client may not set.
/// Compared case-insensitively.
pub fn is_reserved_table_property(key: &str) -> bool {
    let key = key.trim().to_ascii_lowercase();
    RESERVED_TABLE_PROPERTY_KEYS.contains(&key.as_str())
        || RESERVED_TABLE_PROPERTY_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
}

/// Refuse client-supplied table properties that include a reserved key.
pub fn ensure_no_reserved_table_properties(
    properties: &HashMap<String, String>,
) -> anyhow::Result<()> {
    let mut reserved: Vec<&str> = properties
        .keys()
        .map(String::as_str)
        .filter(|key| is_reserved_table_property(key))
        .collect();
    if reserved.is_empty() {
        return Ok(());
    }
    reserved.sort_unstable();
    anyhow::bail!(
        "table properties {} are reserved and cannot be set",
        reserved.join(", ")
    )
}

/// Drop reserved keys from client-supplied table properties.
pub fn without_reserved_table_properties(
    properties: HashMap<String, String>,
) -> HashMap<String, String> {
    properties
        .into_iter()
        .filter(|(key, _)| !is_reserved_table_property(key))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_keys_are_refused_and_user_keys_pass() {
        for key in [
            "asset.kind",
            "Asset.Capability.Vector",
            "collection.id",
            COLLECTION_PROVENANCE_PROPERTY,
            "vector.dimension",
            "stats.row_count",
            "owner",
            " TAGS ",
        ] {
            assert!(is_reserved_table_property(key), "{key}");
        }
        for key in ["workload_profile", "storage_layout", "engine", "comment"] {
            assert!(!is_reserved_table_property(key), "{key}");
        }
        let mut props = HashMap::new();
        props.insert("engine".to_string(), "x".to_string());
        assert!(ensure_no_reserved_table_properties(&props).is_ok());
        props.insert("collection.id".to_string(), "1".to_string());
        assert!(ensure_no_reserved_table_properties(&props).is_err());
        assert_eq!(
            without_reserved_table_properties(props)
                .keys()
                .collect::<Vec<_>>(),
            vec!["engine"]
        );
    }
}
