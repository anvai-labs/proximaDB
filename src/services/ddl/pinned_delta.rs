//! Catalog binding for a local, explicitly versioned Delta table.

use proximadb_catalog::{CatalogAuthorityMode, CatalogPhysicalFormat, CatalogStorageLayout};

pub(crate) fn is_external_delta(layout: &CatalogStorageLayout) -> bool {
    layout.physical_format == CatalogPhysicalFormat::Delta
        && matches!(
            layout.authority,
            CatalogAuthorityMode::ExternalAuthoritative
                | CatalogAuthorityMode::ImportedSnapshot
                | CatalogAuthorityMode::FederatedRead
        )
}

#[cfg(all(unix, feature = "datafusion-integration"))]
pub(crate) const DELTA_VERSION: &str = "delta_version";

#[cfg(all(unix, feature = "datafusion-integration"))]
pub(crate) struct PinnedDeltaLocation {
    pub root: std::path::PathBuf,
    pub relative: object_store::path::Path,
    pub version: u64,
}

#[cfg(all(unix, feature = "datafusion-integration"))]
impl PinnedDeltaLocation {
    pub(crate) fn from_layout(layout: &CatalogStorageLayout) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let uri = layout
            .location
            .as_deref()
            .context("external Delta requires a table URI")?;
        let url = url::Url::parse(uri)?;
        ensure!(
            url.scheme() == "file"
                && url.host_str().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path() != "/"
                && url.as_str() == uri
                && !uri.ends_with('/')
                && url
                    .path()
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"/-_.~".contains(&b)),
            "external Delta requires a canonical local directory URI"
        );
        let root = super::external_location_root(uri)
            .context("external Delta table is outside the operator allowlist")?;
        let relative = object_store::path::Path::parse(
            uri.strip_prefix(&root)
                .context("external Delta table is outside the operator allowlist")?,
        )?;
        let root = url::Url::parse(&root)?.to_file_path().map_err(|_| {
            anyhow::anyhow!("external Delta allowlist root must be a local directory URI")
        })?;
        let version = layout
            .properties
            .get(DELTA_VERSION)
            .context("external Delta requires delta_version")?
            .parse::<u64>()
            .context("delta_version must be an unsigned integer")?;
        Ok(Self {
            root,
            relative,
            version,
        })
    }
}
