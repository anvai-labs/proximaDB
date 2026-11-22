//! Catalog binding for a pinned NDJSON file and its byte-range manifest.

#[cfg(all(unix, feature = "datafusion-integration"))]
use proximadb_catalog::CatalogStorageLayout;

#[cfg(feature = "datafusion-integration")]
pub(crate) const MANIFEST_DIGEST: &str = "ndjson_manifest_sha256";

#[cfg(all(unix, feature = "datafusion-integration"))]
pub(crate) struct PinnedNdjsonLocation {
    pub root: std::path::PathBuf,
    pub file: object_store::path::Path,
    pub manifest_digest: [u8; 32],
}

#[cfg(all(unix, feature = "datafusion-integration"))]
impl PinnedNdjsonLocation {
    pub(crate) fn from_layout(layout: &CatalogStorageLayout) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let uri = layout
            .location
            .as_deref()
            .context("external NDJSON requires a file URI")?;
        let url = url::Url::parse(uri)?;
        ensure!(
            url.scheme() == "file"
                && url.host_str().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.as_str() == uri
                && url.path().ends_with(".jsonl")
                && url
                    .path()
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"/-_.~".contains(&b)),
            "external NDJSON requires a canonical local file URI ending in .jsonl"
        );
        let root = super::external_location_root(uri)
            .context("external NDJSON is outside the operator allowlist")?;
        let file = uri
            .strip_prefix(&root)
            .and_then(|path| path.strip_prefix('/'))
            .context("external NDJSON requires a file beneath an allowlisted directory")?;
        let file = object_store::path::Path::parse(file)?;
        let root = url::Url::parse(&root)?.to_file_path().map_err(|_| {
            anyhow::anyhow!("external NDJSON allowlist root must be a local directory URI")
        })?;
        let mut manifest_digest = [0; 32];
        let digest = layout
            .properties
            .get(MANIFEST_DIGEST)
            .context("external NDJSON requires ndjson_manifest_sha256")?;
        hex::decode_to_slice(digest, &mut manifest_digest)
            .context("ndjson_manifest_sha256 must be a 64-digit hexadecimal SHA-256 digest")?;
        Ok(Self {
            root,
            file,
            manifest_digest,
        })
    }
}
