// Extended aggregation: cross-collection $lookup (left-outer-join).
//
// TD-106: the previously-here `AggregationExpression` evaluator and the
// `StdDevAccumulator`/`compute_stddev` helpers were unwired (no production
// callers; the wired aggregation pipeline in `aggregation.rs` already provides
// computed fields and the accumulators). They were a parallel, proto-`SqlValue`
// expression model and were removed as dead/divergent surface. The `$lookup`
// join machinery below is the one wired export; it remains a legitimate
// `SqlObject` edge (it queries the still-`SqlObject`-shaped foreign collection).

use anyhow::Result;

use crate::proto::proximadb_v1::{
    SqlArray, SqlObject, SqlValue, sql_value::Value as SqlValueVariant,
};

// =============================================================================
// LOOKUP STAGE (cross-collection join)
// =============================================================================

/// Configuration for a $lookup stage that joins documents from another
/// collection, analogous to a SQL left-outer-join.
#[derive(Debug, Clone)]
pub struct LookupConfig {
    /// The foreign collection name to join against.
    pub from_collection: String,
    /// Field path in the local document whose value is matched.
    pub local_field: String,
    /// Field path in the foreign document whose value is matched.
    pub foreign_field: String,
    /// Name of the new array field added to each local document containing
    /// the matched foreign documents.
    pub output_field: String,
}

/// Trait abstracting how foreign documents are fetched during a lookup.
///
/// Implementations can query the document service, an in-memory cache, or
/// a test stub.
pub trait LookupFetcher: Send + Sync {
    /// Return all documents in `collection` where the value at `field_path`
    /// equals `match_value`.
    fn fetch_matching(
        &self,
        collection: &str,
        field_path: &str,
        match_value: &SqlValue,
    ) -> Result<Vec<SqlObject>>;
}

/// Execute a lookup (left-outer-join) stage.
///
/// For each document in `documents`, fetches matching foreign docs via
/// `fetcher` and appends them as an array in `config.output_field`.
pub fn execute_lookup(
    documents: &[SqlObject],
    config: &LookupConfig,
    fetcher: &dyn LookupFetcher,
) -> Result<Vec<SqlObject>> {
    let mut results = Vec::with_capacity(documents.len());

    for doc in documents {
        let mut out = doc.clone();

        let local_val = doc.fields.get(&config.local_field).cloned();

        let matched = if let Some(lv) = &local_val {
            fetcher.fetch_matching(&config.from_collection, &config.foreign_field, lv)?
        } else {
            Vec::new()
        };

        // Convert matched docs to an array of ObjectValues
        let arr_values: Vec<SqlValue> = matched
            .into_iter()
            .map(|obj| SqlValue {
                value: Some(SqlValueVariant::ObjectValue(obj)),
            })
            .collect();

        out.fields.insert(
            config.output_field.clone(),
            SqlValue {
                value: Some(SqlValueVariant::ArrayValue(SqlArray { values: arr_values })),
            },
        );

        results.push(out);
    }

    Ok(results)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn int_val(i: i64) -> SqlValue {
        SqlValue {
            value: Some(SqlValueVariant::Int64Value(i)),
        }
    }

    fn string_val(s: &str) -> SqlValue {
        SqlValue {
            value: Some(SqlValueVariant::StringValue(s.to_string())),
        }
    }

    fn make_doc(fields: Vec<(&str, SqlValue)>) -> SqlObject {
        SqlObject {
            fields: fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }
    }

    /// Equality used by the test fetcher (cross-type numeric aware), matching
    /// the document-filter semantics.
    fn sql_values_equal(a: &SqlValue, b: &SqlValue) -> bool {
        match (&a.value, &b.value) {
            (Some(SqlValueVariant::NullValue(_)), Some(SqlValueVariant::NullValue(_))) => true,
            (Some(SqlValueVariant::BoolValue(va)), Some(SqlValueVariant::BoolValue(vb))) => {
                va == vb
            }
            (Some(SqlValueVariant::Int64Value(va)), Some(SqlValueVariant::Int64Value(vb))) => {
                va == vb
            }
            (Some(SqlValueVariant::NumberValue(va)), Some(SqlValueVariant::NumberValue(vb))) => {
                (va - vb).abs() < f64::EPSILON
            }
            (Some(SqlValueVariant::StringValue(va)), Some(SqlValueVariant::StringValue(vb))) => {
                va == vb
            }
            // TD-PROTO-2: JSONB compares by canonical decode — byte equality
            // would be encoding-fragile, and the wildcard made it always
            // false.
            (Some(SqlValueVariant::JsonbValue(va)), Some(SqlValueVariant::JsonbValue(vb))) => {
                proximadb_data_model::ProximaValue::jsonb_to_json_lossy(va)
                    == proximadb_data_model::ProximaValue::jsonb_to_json_lossy(vb)
            }
            (Some(SqlValueVariant::Int64Value(va)), Some(SqlValueVariant::NumberValue(vb))) => {
                (*va as f64 - vb).abs() < f64::EPSILON
            }
            (Some(SqlValueVariant::NumberValue(va)), Some(SqlValueVariant::Int64Value(vb))) => {
                (va - *vb as f64).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    /// In-memory stub fetcher for testing
    struct TestFetcher {
        foreign_docs: Vec<SqlObject>,
    }

    impl LookupFetcher for TestFetcher {
        fn fetch_matching(
            &self,
            _collection: &str,
            field_path: &str,
            match_value: &SqlValue,
        ) -> Result<Vec<SqlObject>> {
            Ok(self
                .foreign_docs
                .iter()
                .filter(|d| {
                    d.fields
                        .get(field_path)
                        .is_some_and(|v| sql_values_equal(v, match_value))
                })
                .cloned()
                .collect())
        }
    }

    #[test]
    fn test_lookup_basic() {
        // Local: orders with customer_id
        let orders = vec![
            make_doc(vec![
                ("order_id", int_val(1)),
                ("customer_id", int_val(100)),
            ]),
            make_doc(vec![
                ("order_id", int_val(2)),
                ("customer_id", int_val(200)),
            ]),
            make_doc(vec![
                ("order_id", int_val(3)),
                ("customer_id", int_val(100)),
            ]),
        ];

        // Foreign: customers
        let customers = vec![
            make_doc(vec![("cid", int_val(100)), ("name", string_val("Alice"))]),
            make_doc(vec![("cid", int_val(200)), ("name", string_val("Bob"))]),
        ];

        let fetcher = TestFetcher {
            foreign_docs: customers,
        };

        let config = LookupConfig {
            from_collection: "customers".into(),
            local_field: "customer_id".into(),
            foreign_field: "cid".into(),
            output_field: "customer_info".into(),
        };

        let results = execute_lookup(&orders, &config, &fetcher).unwrap();
        assert_eq!(results.len(), 3);

        // Order 1 (customer 100 = Alice) should have 1 match
        let info = results[0].fields.get("customer_info").unwrap();
        if let Some(SqlValueVariant::ArrayValue(arr)) = &info.value {
            assert_eq!(arr.values.len(), 1);
        } else {
            panic!("Expected ArrayValue for customer_info");
        }

        // Order 2 (customer 200 = Bob) should have 1 match
        let info2 = results[1].fields.get("customer_info").unwrap();
        if let Some(SqlValueVariant::ArrayValue(arr)) = &info2.value {
            assert_eq!(arr.values.len(), 1);
        } else {
            panic!("Expected ArrayValue");
        }
    }

    #[test]
    fn test_lookup_no_match() {
        let orders = vec![make_doc(vec![
            ("order_id", int_val(1)),
            ("customer_id", int_val(999)),
        ])];

        let fetcher = TestFetcher {
            foreign_docs: vec![make_doc(vec![
                ("cid", int_val(100)),
                ("name", string_val("Alice")),
            ])],
        };

        let config = LookupConfig {
            from_collection: "customers".into(),
            local_field: "customer_id".into(),
            foreign_field: "cid".into(),
            output_field: "customer_info".into(),
        };

        let results = execute_lookup(&orders, &config, &fetcher).unwrap();
        let info = results[0].fields.get("customer_info").unwrap();
        if let Some(SqlValueVariant::ArrayValue(arr)) = &info.value {
            assert!(arr.values.is_empty());
        } else {
            panic!("Expected empty ArrayValue");
        }
    }

    #[test]
    fn test_lookup_missing_local_field() {
        // Document without the local_field gets an empty array
        let orders = vec![make_doc(vec![("order_id", int_val(1))])];

        let fetcher = TestFetcher {
            foreign_docs: vec![],
        };

        let config = LookupConfig {
            from_collection: "customers".into(),
            local_field: "customer_id".into(),
            foreign_field: "cid".into(),
            output_field: "customer_info".into(),
        };

        let results = execute_lookup(&orders, &config, &fetcher).unwrap();
        let info = results[0].fields.get("customer_info").unwrap();
        if let Some(SqlValueVariant::ArrayValue(arr)) = &info.value {
            assert!(arr.values.is_empty());
        } else {
            panic!("Expected empty ArrayValue");
        }
    }
}
