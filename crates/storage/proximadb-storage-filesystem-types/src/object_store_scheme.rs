//! The object-store URL schemes, in one place (TD-IOBUDGET-1: a scheme alias
//! added to one list but not another makes two decisions about the same URL
//! disagree). The IOPS budget (`proximadb-storage-common::iops_budget`) and
//! the ranged-read wave width ([`crate::FileSystem::read_ranges_prefetch`])
//! both resolve through [`ObjectStoreScheme::of`].

/// A remote object store, where every request pays a network round trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectStoreScheme {
    /// Azure Blob / ABFS / ADLS family.
    Azure,
    /// AWS S3 and HTTP(S) (no hard range limit).
    S3,
    /// Google Cloud Storage.
    Gcs,
}

impl ObjectStoreScheme {
    /// The object store a URL addresses, or `None` for local paths (`file://`,
    /// `minio://`, scheme-less) and unknown schemes.
    pub fn of(path: &str) -> Option<Self> {
        match path
            .split_once("://")
            .map(|(s, _)| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("azure" | "abfs" | "adls" | "az") => Some(Self::Azure),
            Some("s3" | "http" | "https") => Some(Self::S3),
            Some("gs" | "gcs") => Some(Self::Gcs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ObjectStoreScheme;

    #[test]
    fn object_store_schemes_resolve_and_local_paths_do_not() {
        assert_eq!(
            ObjectStoreScheme::of("s3://b/k"),
            Some(ObjectStoreScheme::S3)
        );
        assert_eq!(
            ObjectStoreScheme::of("ABFS://c/k"),
            Some(ObjectStoreScheme::Azure)
        );
        assert_eq!(
            ObjectStoreScheme::of("gs://b/k"),
            Some(ObjectStoreScheme::Gcs)
        );
        for local in ["file:///tmp/x", "minio://x", "/tmp/x", "weird://x"] {
            assert_eq!(ObjectStoreScheme::of(local), None, "{local}");
        }
    }
}
