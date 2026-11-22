//! Catalog binding for an immutable external Iceberg metadata document.

#[cfg(feature = "datafusion-integration")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "datafusion-integration")]
use object_store::path::Path;
#[cfg(feature = "datafusion-integration")]
use proximadb_catalog::CatalogStorageLayout;
#[cfg(feature = "datafusion-integration")]
use url::Url;

#[cfg(feature = "datafusion-integration")]
pub(crate) const METADATA_LOCATION: &str = "iceberg_metadata_location";
#[cfg(feature = "datafusion-integration")]
pub(crate) const DELTA_VERSION: &str = "converted_delta_version";

#[cfg(feature = "datafusion-integration")]
#[derive(Clone)]
pub(crate) struct PinnedIcebergLocation {
    pub table_root: String,
    pub metadata_uri: String,
    pub expected_delta_version: Option<u64>,
}

#[cfg(feature = "datafusion-integration")]
impl PinnedIcebergLocation {
    pub(crate) fn from_layout(layout: &CatalogStorageLayout) -> Result<Self> {
        let root = layout
            .location
            .as_deref()
            .ok_or_else(|| anyhow!("external Iceberg requires a table root"))?;
        let metadata = layout
            .properties
            .get(METADATA_LOCATION)
            .ok_or_else(|| anyhow!("external Iceberg requires {METADATA_LOCATION}"))?;
        let root = root.trim_end_matches('/');
        let url = Url::parse(root)?;
        #[cfg(not(unix))]
        if url.scheme() == "file" {
            bail!("confined local Iceberg reads require Unix");
        }
        if !matches!(url.scheme(), "file" | "s3" | "gs" | "gcs")
            || (url.scheme() == "file" && (url.host_str().is_some() || url.path() == "/"))
            || (url.scheme() != "file" && url.host_str().is_none())
            || url.query().is_some()
            || url.fragment().is_some()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.port().is_some()
            || !url
                .path()
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/-_.~=".contains(&b))
            || (url.as_str().trim_end_matches('/') != root)
        {
            bail!("external Iceberg requires a canonical file, S3 or GCS table URI");
        }
        let key = metadata
            .strip_prefix(root)
            .and_then(|suffix| suffix.strip_prefix('/'))
            .filter(|key| {
                key.ends_with(".metadata.json")
                    && key
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"/-_.~=".contains(&b))
            })
            .ok_or_else(|| {
                anyhow!(
                    "{METADATA_LOCATION} must name a .metadata.json object under the table root"
                )
            })?;
        if Path::parse(key)?.as_ref() != key {
            bail!("{METADATA_LOCATION} must be a canonical object URI");
        }
        if super::external_location_root(root).is_none() {
            bail!("external Iceberg table root is outside the operator allowlist");
        }
        let expected_delta_version = layout
            .properties
            .get(DELTA_VERSION)
            .map(|version| version.parse::<u64>())
            .transpose()
            .map_err(|_| anyhow!("{DELTA_VERSION} must be an unsigned integer"))?;
        Ok(Self {
            table_root: root.to_owned(),
            metadata_uri: metadata.clone(),
            expected_delta_version,
        })
    }
}
