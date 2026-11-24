/// Storage path utilities for consistent path construction across all engines
///
/// # Multi-Tenant Storage Isolation
///
/// All path construction methods support optional `tenant_id` for multi-tenant isolation:
///
/// - **Single-tenant** (tenant_id=None): `{base_url}/{collection_id}/...`
/// - **Multi-tenant** (tenant_id=Some): `{base_url}/tenants/{tenant_id}/{collection_id}/...`
///
/// The "tenants/" prefix is used to clearly separate tenant-isolated data from shared resources.
pub struct StoragePath;

impl StoragePath {
    /// Builds the tenant prefix for a path
    /// Returns "{base_url}/tenants/{tenant_id}" if tenant_id is provided, otherwise just "{base_url}"
    #[inline]
    fn tenant_prefix(base_url: &str, tenant_id: Option<&str>) -> String {
        match tenant_id {
            Some(tid) if !tid.is_empty() => format!("{}/tenants/{}", base_url, tid),
            _ => base_url.to_string(),
        }
    }

    /// Constructs the data directory path for a collection (tenant-aware)
    ///
    /// # Format
    /// - Single-tenant: `{base_url}/{collection_id}/data`
    /// - Multi-tenant: `{base_url}/tenants/{tenant_id}/{collection_id}/data`
    ///
    /// This is the standard location where all storage engines should store their data files.
    /// Using this helper ensures consistency across SST, NOVA, NOVA, SWIFT, RAPTOR, and HELIX engines.
    ///
    /// # Arguments
    /// * `base_url` - The base storage URL (e.g., "file:///path/to/storage" or "s3://bucket/path")
    /// * `tenant_id` - Optional tenant ID for multi-tenant isolation
    /// * `collection_id` - The collection identifier
    ///
    /// # Returns
    /// The full path to the collection's data directory
    pub fn collection_data_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> String {
        format!(
            "{}/{}/data",
            Self::tenant_prefix(base_url, tenant_id),
            collection_id
        )
    }

    /// Constructs the data directory path for a collection (single-tenant, backward compatible)
    /// Format: {base_url}/{collection_id}/data
    pub fn collection_data_path(base_url: &str, collection_id: &str) -> String {
        Self::collection_data_path_with_tenant(base_url, None, collection_id)
    }

    /// Constructs the WAL directory path for a collection (tenant-aware)
    /// Format: {base_url}/[tenants/{tenant_id}/]{collection_id}/wal
    pub fn collection_wal_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> String {
        format!(
            "{}/{}/wal",
            Self::tenant_prefix(base_url, tenant_id),
            collection_id
        )
    }

    /// Constructs the WAL directory path for a collection (single-tenant)
    /// Format: {base_url}/{collection_id}/wal
    pub fn collection_wal_path(base_url: &str, collection_id: &str) -> String {
        Self::collection_wal_path_with_tenant(base_url, None, collection_id)
    }

    /// Constructs the index directory path for a collection (tenant-aware)
    /// Format: {base_url}/[tenants/{tenant_id}/]{collection_id}/indexes
    pub fn collection_index_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> String {
        format!(
            "{}/{}/indexes",
            Self::tenant_prefix(base_url, tenant_id),
            collection_id
        )
    }

    /// Constructs the index directory path for a collection (single-tenant)
    /// Format: {base_url}/{collection_id}/indexes
    pub fn collection_index_path(base_url: &str, collection_id: &str) -> String {
        Self::collection_index_path_with_tenant(base_url, None, collection_id)
    }

    /// Constructs the metadata directory path for a collection (tenant-aware)
    /// Format: {base_url}/[tenants/{tenant_id}/]{collection_id}/metadata
    pub fn collection_metadata_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> String {
        format!(
            "{}/{}/metadata",
            Self::tenant_prefix(base_url, tenant_id),
            collection_id
        )
    }

    /// Constructs the metadata directory path for a collection (single-tenant)
    /// Format: {base_url}/{collection_id}/metadata
    pub fn collection_metadata_path(base_url: &str, collection_id: &str) -> String {
        Self::collection_metadata_path_with_tenant(base_url, None, collection_id)
    }

    /// Constructs the compaction staging directory path (tenant-aware)
    /// Format: {base_url}/[tenants/{tenant_id}/]{collection_id}/compaction_staging
    pub fn collection_compaction_staging_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> String {
        format!(
            "{}/{}/compaction_staging",
            Self::tenant_prefix(base_url, tenant_id),
            collection_id
        )
    }

    /// Constructs the compaction staging directory path (single-tenant)
    /// Format: {base_url}/{collection_id}/compaction_staging
    pub fn collection_compaction_staging_path(base_url: &str, collection_id: &str) -> String {
        Self::collection_compaction_staging_path_with_tenant(base_url, None, collection_id)
    }

    /// Constructs a file path within the data directory (tenant-aware)
    /// Format: {base_url}/[tenants/{tenant_id}/]{collection_id}/data/{filename}
    pub fn data_file_path_with_tenant(
        base_url: &str,
        tenant_id: Option<&str>,
        collection_id: &str,
        filename: &str,
    ) -> String {
        format!(
            "{}/{}",
            Self::collection_data_path_with_tenant(base_url, tenant_id, collection_id),
            filename
        )
    }

    /// Constructs a file path within the data directory (single-tenant)
    /// Format: {base_url}/{collection_id}/data/{filename}
    pub fn data_file_path(base_url: &str, collection_id: &str, filename: &str) -> String {
        Self::data_file_path_with_tenant(base_url, None, collection_id, filename)
    }

    /// Constructs the tenant root directory path
    /// Format: {base_url}/tenants/{tenant_id}
    pub fn tenant_root_path(base_url: &str, tenant_id: &str) -> String {
        format!("{}/tenants/{}", base_url, tenant_id)
    }

    /// Constructs the tenant collections list directory
    /// Format: {base_url}/tenants/{tenant_id}/_collections
    pub fn tenant_collections_path(base_url: &str, tenant_id: &str) -> String {
        format!("{}/tenants/{}/_collections", base_url, tenant_id)
    }

    /// Helper to find a marker pattern in a path, avoiding false matches with directory names.
    /// Uses rfind to find the last occurrence (closest to end of path).
    fn find_marker(path: &str, marker: &str) -> Option<usize> {
        // Find the last occurrence of the marker
        // The marker should not be part of "tenants" or base path
        let mut search_start = 0;

        // Skip past "tenants" directory if present to avoid false matches
        if let Some(tenants_pos) = path.find("/tenants/") {
            // Look for marker after the tenants/{tenant_id}/ section
            let after_tenants = tenants_pos + "/tenants/".len();
            if let Some(next_slash) = path[after_tenants..].find('/') {
                search_start = after_tenants + next_slash;
            }
        }

        // Search from search_start for the marker
        path[search_start..]
            .rfind(marker)
            .map(|pos| search_start + pos)
    }

    /// Parses a full path to extract base URL, optional tenant ID, and collection ID
    ///
    /// # Supported formats
    /// - Single-tenant: `{base_url}/{collection_id}/data/...` → (base_url, None, collection_id)
    /// - Multi-tenant: `{base_url}/tenants/{tenant_id}/{collection_id}/data/...` → (base_url, Some(tenant_id), collection_id)
    ///
    /// # Returns
    /// Tuple of (base_url, tenant_id, collection_id)
    pub fn parse_collection_path_with_tenant(
        full_path: &str,
    ) -> Option<(String, Option<String>, String)> {
        // Find the data/wal/indexes marker - must be followed by '/' or end of path
        // We use patterns with trailing slash to avoid matching directory names like /data/
        let marker_pos = Self::find_marker(full_path, "/data/")
            .or_else(|| Self::find_marker(full_path, "/wal/"))
            .or_else(|| Self::find_marker(full_path, "/indexes/"))
            .or_else(|| Self::find_marker(full_path, "/metadata/"))
            .or_else(|| Self::find_marker(full_path, "/compaction_staging/"))
            // Also check for markers at end of path (no trailing content)
            .or_else(|| {
                for marker in &[
                    "/data",
                    "/wal",
                    "/indexes",
                    "/metadata",
                    "/compaction_staging",
                ] {
                    if full_path.ends_with(marker) {
                        return Some(full_path.len() - marker.len());
                    }
                }
                None
            })?;

        let base_and_collection = &full_path[..marker_pos];

        // Check for tenant path pattern: .../tenants/{tenant_id}/{collection_id}
        if let Some(tenants_pos) = base_and_collection.find("/tenants/") {
            let after_tenants = &base_and_collection[tenants_pos + "/tenants/".len()..];
            if let Some(slash_pos) = after_tenants.find('/') {
                let tenant_id = &after_tenants[..slash_pos];
                let collection_id = &after_tenants[slash_pos + 1..];
                let base_url = &base_and_collection[..tenants_pos];
                return Some((
                    base_url.to_string(),
                    Some(tenant_id.to_string()),
                    collection_id.to_string(),
                ));
            }
        }

        // Single-tenant pattern: {base_url}/{collection_id}
        if let Some(last_slash) = base_and_collection.rfind('/') {
            let base_url = &base_and_collection[..last_slash];
            let collection_id = &base_and_collection[last_slash + 1..];
            return Some((base_url.to_string(), None, collection_id.to_string()));
        }

        None
    }

    /// Parses a full path to extract the base URL and collection ID (backward compatible)
    /// Expects format: {base_url}/{collection_id}/...
    pub fn parse_collection_path(full_path: &str) -> Option<(String, String)> {
        Self::parse_collection_path_with_tenant(full_path).map(|(base, tenant, collection)| {
            // For backward compatibility, include tenant in base_url if present
            if let Some(tid) = tenant {
                (format!("{}/tenants/{}", base, tid), collection)
            } else {
                (base, collection)
            }
        })
    }

    /// Ensures the path uses forward slashes consistently (important for URLs)
    pub fn normalize_path(path: &str) -> String {
        path.replace('\\', "/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Single-tenant tests (backward compatible) ==========

    #[test]
    fn test_collection_data_path() {
        assert_eq!(
            StoragePath::collection_data_path("file:///storage", "test_collection"),
            "file:///storage/test_collection/data"
        );

        assert_eq!(
            StoragePath::collection_data_path("s3://bucket/path", "my-collection"),
            "s3://bucket/path/my-collection/data"
        );
    }

    #[test]
    fn test_parse_collection_path() {
        let (base, collection) =
            StoragePath::parse_collection_path("file:///storage/test_collection/data/file.sst")
                .unwrap();
        assert_eq!(base, "file:///storage");
        assert_eq!(collection, "test_collection");
    }

    #[test]
    fn test_data_file_path() {
        assert_eq!(
            StoragePath::data_file_path("file:///storage", "collection1", "level0_001.sst"),
            "file:///storage/collection1/data/level0_001.sst"
        );
    }

    // ========== Multi-tenant tests ==========

    #[test]
    fn test_tenant_data_path() {
        // With tenant
        assert_eq!(
            StoragePath::collection_data_path_with_tenant(
                "file:///storage",
                Some("tenant1"),
                "test_collection"
            ),
            "file:///storage/tenants/tenant1/test_collection/data"
        );

        // Without tenant (None)
        assert_eq!(
            StoragePath::collection_data_path_with_tenant(
                "file:///storage",
                None,
                "test_collection"
            ),
            "file:///storage/test_collection/data"
        );

        // Empty tenant treated as None
        assert_eq!(
            StoragePath::collection_data_path_with_tenant(
                "file:///storage",
                Some(""),
                "test_collection"
            ),
            "file:///storage/test_collection/data"
        );
    }

    #[test]
    fn test_tenant_wal_path() {
        assert_eq!(
            StoragePath::collection_wal_path_with_tenant(
                "s3://bucket",
                Some("acme-corp"),
                "vectors"
            ),
            "s3://bucket/tenants/acme-corp/vectors/wal"
        );
    }

    #[test]
    fn test_tenant_index_path() {
        assert_eq!(
            StoragePath::collection_index_path_with_tenant(
                "file:///data",
                Some("enterprise"),
                "embeddings"
            ),
            "file:///data/tenants/enterprise/embeddings/indexes"
        );
    }

    #[test]
    fn test_tenant_data_file_path() {
        assert_eq!(
            StoragePath::data_file_path_with_tenant(
                "file:///storage",
                Some("tenant-x"),
                "col1",
                "level0.sst"
            ),
            "file:///storage/tenants/tenant-x/col1/data/level0.sst"
        );
    }

    #[test]
    fn test_parse_tenant_path() {
        // Multi-tenant path
        let (base, tenant, collection) = StoragePath::parse_collection_path_with_tenant(
            "file:///storage/tenants/acme-corp/my-collection/data/file.sst",
        )
        .unwrap();
        assert_eq!(base, "file:///storage");
        assert_eq!(tenant, Some("acme-corp".to_string()));
        assert_eq!(collection, "my-collection");

        // Single-tenant path (no tenant)
        let (base, tenant, collection) = StoragePath::parse_collection_path_with_tenant(
            "file:///storage/my-collection/data/file.sst",
        )
        .unwrap();
        assert_eq!(base, "file:///storage");
        assert_eq!(tenant, None);
        assert_eq!(collection, "my-collection");
    }

    #[test]
    fn test_tenant_root_path() {
        assert_eq!(
            StoragePath::tenant_root_path("file:///storage", "customer-123"),
            "file:///storage/tenants/customer-123"
        );
    }

    #[test]
    fn test_tenant_collections_path() {
        assert_eq!(
            StoragePath::tenant_collections_path("s3://bucket", "org-456"),
            "s3://bucket/tenants/org-456/_collections"
        );
    }

    #[test]
    fn test_parse_various_markers() {
        // Test WAL path parsing
        let (base, tenant, collection) = StoragePath::parse_collection_path_with_tenant(
            "file:///data/tenants/t1/c1/wal/segment.log",
        )
        .unwrap();
        assert_eq!(base, "file:///data");
        assert_eq!(tenant, Some("t1".to_string()));
        assert_eq!(collection, "c1");

        // Test indexes path parsing
        let (base, tenant, collection) = StoragePath::parse_collection_path_with_tenant(
            "s3://bucket/path/my-collection/indexes/hnsw.idx",
        )
        .unwrap();
        assert_eq!(base, "s3://bucket/path");
        assert_eq!(tenant, None);
        assert_eq!(collection, "my-collection");
    }
}
