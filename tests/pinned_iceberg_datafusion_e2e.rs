#![cfg(feature = "datafusion-integration")]

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::prelude::{SessionConfig, SessionContext};
use object_store::{memory::InMemory, path::Path};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use proximadb::datafusion::engine_adapters::pinned_iceberg::PinnedIcebergTable;
use proximadb_iceberg_engine::{
    IcebergObjectStoreBridge, deletion_vector::PositionDeletes, iceberg::IcebergField,
};
use proximadb_object_store::ProximaObjectStore;

#[tokio::test]
async fn datafusion_queries_project_and_count_pinned_iceberg_rows() {
    let bridge = IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(InMemory::new())))
        .with_root_url("memory://warehouse");
    let old = Path::from("table/data/old.parquet");
    let new = Path::from("table/data/new.parquet");
    for (path, name, values) in [
        (&old, "prior_name", vec![10, 11]),
        (&new, "value", vec![20, 21]),
    ] {
        let field = Field::new(name, DataType::Int64, false)
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), "7".to_string())]);
        let schema = Arc::new(Schema::new(vec![field]));
        let rows =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
        writer.write(&rows).unwrap();
        writer.close().unwrap();
        bridge
            .object_store()
            .put(path, Bytes::from(bytes))
            .await
            .unwrap();
    }

    let mut deletes = PositionDeletes::new();
    deletes.insert(1);
    let deletes = bridge
        .write_deletion_vectors(
            &Path::from("table/data/delete.puffin"),
            &[(old.clone(), &deletes)],
        )
        .await
        .unwrap();
    let version = bridge
        .publish_iceberg_files_with_deletes(
            &[old, new],
            &deletes,
            None,
            "table/metadata",
            "eeff0011-2233-4455-6677-8899aabbccdd",
            &bridge.object_uri("table"),
            &[IcebergField {
                id: 7,
                name: "value".into(),
                type_name: "long".into(),
                required: true,
            }],
            100,
            1_700_000_000_000,
        )
        .await
        .unwrap();
    let uri = bridge.metadata_file_uri("table/metadata", version);
    let reader = bridge
        .open_pinned_iceberg_reader(&Path::from("table"), &uri, None)
        .await
        .unwrap();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
    ctx.register_table("pinned", Arc::new(PinnedIcebergTable::new(reader)))
        .unwrap();

    for (query, expected) in [
        ("SELECT value FROM pinned ORDER BY value", vec![10, 20, 21]),
        ("SELECT count(*) FROM pinned", vec![3]),
        (
            "SELECT value FROM pinned WHERE value >= 11 ORDER BY value LIMIT 1",
            vec![20],
        ),
    ] {
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let actual: Vec<_> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(actual, expected, "{query}");
    }
}
