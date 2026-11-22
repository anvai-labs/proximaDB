//! Catalog classification shared by DDL, DML, and execution for pinned sources.

use proximadb_catalog::{CatalogAuthorityMode, CatalogPhysicalFormat, CatalogStorageLayout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinnedExternalFormat {
    Iceberg,
    Ndjson,
    Delta,
}

impl PinnedExternalFormat {
    pub(crate) fn for_layout(layout: &CatalogStorageLayout) -> Option<Self> {
        if !matches!(
            layout.authority,
            CatalogAuthorityMode::ExternalAuthoritative
                | CatalogAuthorityMode::ImportedSnapshot
                | CatalogAuthorityMode::FederatedRead
        ) {
            return None;
        }
        match layout.physical_format {
            CatalogPhysicalFormat::Iceberg => Some(Self::Iceberg),
            CatalogPhysicalFormat::Json => Some(Self::Ndjson),
            CatalogPhysicalFormat::Delta => Some(Self::Delta),
            _ => None,
        }
    }

    pub(crate) fn snapshot_semantics(self) -> &'static str {
        match self {
            Self::Iceberg => "pinned-iceberg-snapshot",
            Self::Ndjson => "pinned-ndjson-manifest",
            Self::Delta => "pinned-delta-version",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Iceberg => "Iceberg",
            Self::Ndjson => "NDJSON",
            Self::Delta => "Delta",
        }
    }

    pub(super) fn copy_options_and_validate(
        self,
        layout: &mut CatalogStorageLayout,
        schema: &proximadb_catalog::CatalogTableSchema,
    ) -> anyhow::Result<()> {
        #[cfg(not(feature = "datafusion-integration"))]
        {
            let _ = (layout, schema);
            anyhow::bail!("external {} requires datafusion-integration", self.name());
        }
        #[cfg(feature = "datafusion-integration")]
        {
            use super::{pinned_delta, pinned_iceberg, pinned_ndjson};

            let keys: &[&str] = match self {
                Self::Iceberg => &[
                    pinned_iceberg::METADATA_LOCATION,
                    pinned_iceberg::DELTA_VERSION,
                ],
                Self::Ndjson => &[pinned_ndjson::MANIFEST_DIGEST],
                Self::Delta => &[pinned_delta::DELTA_VERSION],
            };
            for &key in keys {
                if let Some(value) = schema.properties.get(key) {
                    layout.properties.insert(key.into(), value.clone());
                }
            }
            let _location = self.parse_location(layout)?;
            #[cfg(unix)]
            if self == Self::Ndjson {
                crate::datafusion::engine_adapters::pinned_ndjson::ndjson_schema(&schema.columns)?;
            }
            Ok(())
        }
    }

    #[cfg(feature = "datafusion-integration")]
    pub(crate) fn parse_location(
        self,
        layout: &CatalogStorageLayout,
    ) -> anyhow::Result<PinnedExternalLocation> {
        use anyhow::Context;

        let result = match self {
            Self::Iceberg => super::pinned_iceberg::PinnedIcebergLocation::from_layout(layout)
                .map(PinnedExternalLocation::Iceberg),
            #[cfg(unix)]
            Self::Ndjson => super::pinned_ndjson::PinnedNdjsonLocation::from_layout(layout)
                .map(PinnedExternalLocation::Ndjson),
            #[cfg(unix)]
            Self::Delta => super::pinned_delta::PinnedDeltaLocation::from_layout(layout)
                .map(PinnedExternalLocation::Delta),
            #[cfg(not(unix))]
            Self::Ndjson | Self::Delta => anyhow::bail!("external {} requires Unix", self.name()),
        };
        result.with_context(|| format!("external {} binding", self.name()))
    }
}

#[cfg(feature = "datafusion-integration")]
pub(crate) enum PinnedExternalLocation {
    Iceberg(super::pinned_iceberg::PinnedIcebergLocation),
    #[cfg(unix)]
    Ndjson(super::pinned_ndjson::PinnedNdjsonLocation),
    #[cfg(unix)]
    Delta(super::pinned_delta::PinnedDeltaLocation),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_and_authority_jointly_select_the_pinned_route() {
        for (physical_format, pinned) in [
            (
                CatalogPhysicalFormat::Iceberg,
                Some(PinnedExternalFormat::Iceberg),
            ),
            (
                CatalogPhysicalFormat::Json,
                Some(PinnedExternalFormat::Ndjson),
            ),
            (
                CatalogPhysicalFormat::Delta,
                Some(PinnedExternalFormat::Delta),
            ),
            (CatalogPhysicalFormat::Parquet, None),
        ] {
            for (authority, eligible) in [
                (CatalogAuthorityMode::ExternalAuthoritative, true),
                (CatalogAuthorityMode::ImportedSnapshot, true),
                (CatalogAuthorityMode::FederatedRead, true),
                (CatalogAuthorityMode::ProximaAuthoritative, false),
                (CatalogAuthorityMode::ProjectionPublication, false),
            ] {
                let layout = CatalogStorageLayout {
                    authority,
                    physical_format: physical_format.clone(),
                    ..Default::default()
                };
                let result = PinnedExternalFormat::for_layout(&layout);
                assert_eq!(result, if eligible { pinned } else { None }, "{layout:?}");
            }
        }
    }
}
