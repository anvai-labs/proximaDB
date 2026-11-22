//! Bind a cataloged external table before constructing its execution adapter.

use anyhow::{Context, ensure};
use proximadb_catalog::{CatalogColumn, CatalogTableSchema};

use super::pinned_iceberg;
use super::pinned_table::PinnedTable;
use crate::query::execution::engine::ExecutionControls;
use crate::services::ddl::pinned_external::{PinnedExternalFormat, PinnedExternalLocation};

pub(crate) async fn open_table(
    schema: &CatalogTableSchema,
    controls: &ExecutionControls,
) -> anyhow::Result<PinnedTable> {
    controls.check_cancelled()?;
    let mut layouts = schema.storage_layouts.iter().filter_map(|layout| {
        PinnedExternalFormat::for_layout(layout).map(|format| (format, layout))
    });
    let (format, layout) = layouts.next().context("expected one pinned layout")?;
    ensure!(layouts.next().is_none(), "expected one pinned layout");

    let bound = match format.parse_location(layout)? {
        PinnedExternalLocation::Iceberg(location) => {
            pinned_iceberg::open_reader(location, &schema.columns)
                .await
                .map(PinnedTable::new)
        }
        #[cfg(unix)]
        PinnedExternalLocation::Ndjson(location) => {
            super::pinned_ndjson::PinnedNdjsonReader::open(location, &schema.columns)
                .await
                .map(PinnedTable::from_source)
        }
        #[cfg(unix)]
        PinnedExternalLocation::Delta(location) => super::pinned_delta::PinnedDeltaReader::open(
            location,
            &schema.columns,
            controls.clone(),
        )
        .await
        .map(PinnedTable::from_source),
    };
    controls.check_cancelled()?;
    bound
}

pub(super) fn validate_external_schema(
    schema: &arrow_schema::Schema,
    columns: &[CatalogColumn],
) -> anyhow::Result<()> {
    if schema.fields().len() != columns.len()
        || schema.fields().iter().zip(columns).any(|(field, column)| {
            use arrow_schema::DataType;
            field.name() != &column.name
                || !matches!(
                    field.data_type(),
                    DataType::Boolean
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Utf8
                        | DataType::Binary
                        | DataType::Date32
                        | DataType::Decimal128(_, _)
                )
                || crate::query::execution::datafusion_engine::arrow_type_to_proxima(
                    field.data_type(),
                ) != column.data_type
        })
    {
        anyhow::bail!(
            "catalog columns must match the supported pinned schema in name, type and order"
        );
    }
    Ok(())
}
