// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! FOREIGN KEY reference resolution shared by every enforcement site: the DML
//! child-insert probe and ON DELETE actions (`services::dml`) and the native
//! INSERT…SELECT executor (`query::table_write_executor`).
//!
//! TD-RELIDEQ-2: a FOREIGN KEY may reference the parent's declared primary key
//! (whose encoded values ARE the parent row's `oid`) or exactly one of its
//! UNIQUE column sets (whose values live in `props`) — the SQL rule. Before,
//! every site assumed "referenced columns = parent key = oid", and the
//! `id`/`record_id` fallback let an FK to a keyless parent through that check:
//! child values were then probed as parent oids (always generated), so valid
//! children were rejected and ON DELETE actions matched nothing.

use anyhow::Result;
use proximadb_catalog::CatalogTableSchema;
use proximadb_catalog::relational::encode_primary_key_tuple;
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};

use crate::services::record_store::{
    TableRecordGetRequest, TableRecordStore, UniqueCandidateSet, proxima_value_to_unique_text,
    record_unique_tuple, schema_declared_primary_key_columns, schema_oid_key_column,
    schema_unique_column_sets,
};
use crate::storage::tenant::context::TenantContext;
#[cfg(feature = "abac-policy")]
use proximadb_abac::ReadContext;

/// How a FOREIGN KEY's referenced columns identify a parent row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferencedKey {
    /// The parent's declared primary key, in order: the row's `oid`.
    PrimaryKey,
    /// One of the parent's UNIQUE column sets, in order: values in `props`.
    Unique,
}

/// Which key `references_columns` names on `parent`, or `None` when it names
/// neither the declared primary key nor a UNIQUE set (not enforceable).
pub(crate) fn resolve_referenced_key(
    parent: &CatalogTableSchema,
    references_columns: &[String],
) -> Option<ReferencedKey> {
    if references_columns.is_empty() {
        return None;
    }
    if schema_declared_primary_key_columns(parent).as_slice() == references_columns {
        return Some(ReferencedKey::PrimaryKey);
    }
    schema_unique_column_sets(parent)
        .iter()
        .any(|set| set.as_slice() == references_columns)
        .then_some(ReferencedKey::Unique)
}

/// The values of `columns` in `record.props`, or `None` when any is NULL,
/// absent or non-scalar — MATCH SIMPLE: such a child references nothing, and
/// such a parent row can be referenced by nothing.
pub(crate) fn record_key_values(
    record: &ProximaRecord,
    columns: &[String],
) -> Option<Vec<ProximaValue>> {
    columns
        .iter()
        .map(|column| match record.props.get(column) {
            Some(ProximaTreeNode::Value(value)) if !matches!(value, ProximaValue::Null) => {
                Some(value.clone())
            }
            _ => None,
        })
        .collect()
}

/// Separator for a UNIQUE reference's tuple when it is matched as one string.
const TUPLE_SEPARATOR: &str = "\u{1f}";

/// Child FK `values` in the parent's UNIQUE-index tuple form — exactly what
/// [`record_unique_tuple`] yields for a parent row holding them: the parent's
/// oid-key column as its `oid` (the encoded key), every other column as unique
/// text. One form for the existence probe, ON DELETE and ON UPDATE, so a value
/// the probe accepts is a value the actions match.
fn unique_reference_tuple(
    values: &[ProximaValue],
    references_columns: &[String],
    parent_schema: &CatalogTableSchema,
) -> Option<Vec<String>> {
    let oid_key = schema_oid_key_column(parent_schema);
    values
        .iter()
        .zip(references_columns)
        .map(|(value, column)| {
            if oid_key
                .as_deref()
                .is_some_and(|key| key.eq_ignore_ascii_case(column))
            {
                encode_primary_key_tuple(std::slice::from_ref(value)).ok()
            } else {
                Some(proxima_value_to_unique_text(value))
            }
        })
        .collect()
}

/// A child row's FK `values` as the text its parent's referenced key is
/// matched in: the encoded key (= the parent `oid`) for a primary-key
/// reference, the parent's unique-index tuple for a UNIQUE reference.
pub(crate) fn child_reference_text(
    values: &[ProximaValue],
    key: ReferencedKey,
    references_columns: &[String],
    parent_schema: &CatalogTableSchema,
) -> Option<String> {
    match key {
        ReferencedKey::PrimaryKey => encode_primary_key_tuple(values).ok(),
        ReferencedKey::Unique => unique_reference_tuple(values, references_columns, parent_schema)
            .map(|tuple| tuple.join(TUPLE_SEPARATOR)),
    }
}

/// A parent row's referenced key in the same text as
/// [`child_reference_text`]: its `oid`, or its unique-index tuple.
pub(crate) fn parent_referenced_key_text(
    parent_row: &ProximaRecord,
    key: ReferencedKey,
    references_columns: &[String],
    parent_schema: &CatalogTableSchema,
) -> Option<String> {
    match key {
        ReferencedKey::PrimaryKey => Some(parent_row.oid.clone()),
        ReferencedKey::Unique => record_unique_tuple(
            parent_row,
            references_columns,
            schema_oid_key_column(parent_schema).as_deref(),
        )
        .map(|tuple| tuple.join(TUPLE_SEPARATOR)),
    }
}

/// Whether a live parent row holds `values` in its referenced key. A
/// primary-key reference is a point lookup on the encoded `oid`; a UNIQUE
/// reference asks the parent's unique index (`check_unique_conflict`, the same
/// probe that enforces the UNIQUE constraint).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn referenced_parent_exists(
    record_store: &dyn TableRecordStore,
    parent_schema: &CatalogTableSchema,
    parent_table_id: &str,
    key: ReferencedKey,
    references_columns: &[String],
    values: &[ProximaValue],
    tenant_context: Option<&TenantContext>,
    #[cfg(feature = "abac-policy")] read_context: &ReadContext,
) -> Result<bool> {
    match key {
        ReferencedKey::PrimaryKey => Ok(record_store
            .get_by_key(
                parent_schema,
                TableRecordGetRequest {
                    table_id: parent_table_id.to_string(),
                    key: encode_primary_key_tuple(values)?,
                    include_vector: false,
                    include_props: false,
                },
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await?
            .is_some()),
        ReferencedKey::Unique => {
            let Some(tuple) = unique_reference_tuple(values, references_columns, parent_schema)
            else {
                return Ok(false);
            };
            let set = UniqueCandidateSet {
                columns: references_columns.to_vec(),
                candidates: std::iter::once(tuple).collect(),
            };
            Ok(record_store
                .check_unique_conflict(
                    parent_schema,
                    parent_table_id,
                    schema_oid_key_column(parent_schema).as_deref(),
                    std::slice::from_ref(&set),
                    &std::collections::HashSet::new(),
                    tenant_context,
                )
                .await?
                .is_some())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::ColumnConstraint;

    fn schema(primary_key: &[&str], unique: &[&[&str]]) -> CatalogTableSchema {
        let mut schema = CatalogTableSchema::new("parent");
        schema.primary_key = primary_key.iter().map(|c| c.to_string()).collect();
        schema.relational_capabilities.constraints = unique
            .iter()
            .map(|set| ColumnConstraint::Unique {
                columns: set.iter().map(|c| c.to_string()).collect(),
            })
            .collect();
        schema
    }

    fn cols(names: &[&str]) -> Vec<String> {
        names.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn resolves_the_declared_key_or_a_unique_set_only() {
        let keyed = schema(&["id"], &[&["email"]]);
        assert_eq!(
            resolve_referenced_key(&keyed, &cols(&["id"])),
            Some(ReferencedKey::PrimaryKey)
        );
        assert_eq!(
            resolve_referenced_key(&keyed, &cols(&["email"])),
            Some(ReferencedKey::Unique)
        );
        assert_eq!(resolve_referenced_key(&keyed, &cols(&["name"])), None);

        // A keyless table's `id` is an ordinary column: referenceable only
        // through a UNIQUE constraint, never as the oid.
        assert_eq!(
            resolve_referenced_key(&schema(&[], &[]), &cols(&["id"])),
            None
        );
        assert_eq!(
            resolve_referenced_key(&schema(&[], &[&["id"]]), &cols(&["id"])),
            Some(ReferencedKey::Unique)
        );

        // Composite: the whole key, in order.
        let composite = schema(&["region", "id"], &[]);
        assert_eq!(
            resolve_referenced_key(&composite, &cols(&["region", "id"])),
            Some(ReferencedKey::PrimaryKey)
        );
        assert_eq!(resolve_referenced_key(&composite, &cols(&["region"])), None);
        assert_eq!(
            resolve_referenced_key(&composite, &cols(&["id", "region"])),
            None
        );
    }

    #[test]
    fn parent_key_text_reads_the_oid_or_the_props_values() {
        let mut row = ProximaRecord {
            oid: "generated-oid".to_string(),
            ..ProximaRecord::default()
        };
        row.props.insert(
            "id".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int32(7)),
        );
        let id = cols(&["id"]);
        let keyless = schema(&[], &[&["id"]]);
        assert_eq!(
            parent_referenced_key_text(&row, ReferencedKey::PrimaryKey, &id, &keyless),
            Some("generated-oid".to_string())
        );
        // A UNIQUE reference reads props, and a child holding the same value
        // — even at another integer width — matches it.
        let parent_text = parent_referenced_key_text(&row, ReferencedKey::Unique, &id, &keyless);
        assert!(parent_text.is_some());
        for child_value in [ProximaValue::Int32(7), ProximaValue::Int64(7)] {
            assert_eq!(
                child_reference_text(&[child_value], ReferencedKey::Unique, &id, &keyless),
                parent_text
            );
        }
        row.props
            .insert("id".to_string(), ProximaTreeNode::Value(ProximaValue::Null));
        assert_eq!(
            parent_referenced_key_text(&row, ReferencedKey::Unique, &id, &keyless),
            None
        );
    }
}
