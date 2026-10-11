use prost::Message;
use proximadb::proto::proximadb_v1 as v1;

#[test]
fn grpc_roundtrip_sqlvalue_prost() {
    let row = v1::SqlRow {
        fields: vec![
            v1::SqlRowField {
                key: "s".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::StringValue("hello".into())),
                }),
            },
            v1::SqlRowField {
                key: "f64".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::NumberValue(3.5)),
                }),
            },
            v1::SqlRowField {
                key: "b".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::BoolValue(true)),
                }),
            },
            v1::SqlRowField {
                key: "i64".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::Int64Value(42)),
                }),
            },
            v1::SqlRowField {
                key: "bytes".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::BytesValue(vec![1, 2, 3, 4])),
                }),
            },
            v1::SqlRowField {
                key: "jsonb".into(),
                value: Some(v1::SqlValue {
                    // Canonical MessagePack JSONB payload (types.proto tag 9).
                    value: Some(v1::sql_value::Value::JsonbValue(
                        proximadb_data_model::ProximaValue::to_jsonb_vec(
                            &serde_json::json!({"jsonb": true, "n": 1}),
                        )
                        .unwrap(),
                    )),
                }),
            },
            v1::SqlRowField {
                key: "null".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::NullValue(0)),
                }),
            },
            v1::SqlRowField {
                key: "arr".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::ArrayValue(v1::SqlArray {
                        values: vec![
                            v1::SqlValue {
                                value: Some(v1::sql_value::Value::NumberValue(1.0)),
                            },
                            v1::SqlValue {
                                value: Some(v1::sql_value::Value::StringValue("x".into())),
                            },
                        ],
                    })),
                }),
            },
            v1::SqlRowField {
                key: "obj".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::ObjectValue(v1::SqlObject {
                        fields: {
                            let mut m = std::collections::BTreeMap::new();
                            m.insert(
                                "k".into(),
                                v1::SqlValue {
                                    value: Some(v1::sql_value::Value::BoolValue(false)),
                                },
                            );
                            m.into_iter().collect()
                        },
                    })),
                }),
            },
        ],
        similarity: Some(0.99),
    };
    let resp = v1::ExecuteQueryResponse {
        rows: vec![row],
        rows_scanned: 1,
        rows_returned: 1,
        execution_time_ms: 5,
        columns: vec!["s".into()],
        column_types: vec!["string".into()],
    };

    let mut buf = Vec::new();
    resp.encode(&mut buf).unwrap();
    let decoded = v1::ExecuteQueryResponse::decode(&*buf).unwrap();
    assert_eq!(decoded.rows.len(), 1);
    assert_eq!(decoded.rows[0].fields.len(), 9);
    // The JSONB variant must survive the binary wire (TD-PROTO-2: it decoded
    // to None before the mirror gained the tag-9 arm).
    assert!(
        matches!(
            decoded.rows[0].fields[5].value.as_ref().and_then(|v| v.value.as_ref()),
            Some(v1::sql_value::Value::JsonbValue(b)) if !b.is_empty()
        ),
        "jsonb field must round-trip as tag-9 bytes"
    );
}

#[test]
fn rest_roundtrip_sqlvalue_json() {
    let row = v1::SqlRow {
        fields: vec![
            v1::SqlRowField {
                key: "n".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::NumberValue(7.0)),
                }),
            },
            v1::SqlRowField {
                key: "o".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::ObjectValue(v1::SqlObject {
                        fields: {
                            let mut m = std::collections::BTreeMap::new();
                            m.insert(
                                "a".into(),
                                v1::SqlValue {
                                    value: Some(v1::sql_value::Value::Int64Value(1)),
                                },
                            );
                            m.into_iter().collect()
                        },
                    })),
                }),
            },
            v1::SqlRowField {
                key: "jsonb".into(),
                value: Some(v1::SqlValue {
                    value: Some(v1::sql_value::Value::JsonbValue(vec![
                        0x81, 0xA1, b'j', 0x01,
                    ])),
                }),
            },
        ],
        similarity: None,
    };
    let resp = v1::ExecuteQueryResponse {
        rows: vec![row],
        rows_scanned: 1,
        rows_returned: 1,
        execution_time_ms: 1,
        columns: vec![],
        column_types: vec![],
    };

    // Wrap in REST envelope and JSON round-trip
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Wrapper {
        data: v1::ExecuteQueryResponse,
        success: bool,
    }
    let w = Wrapper {
        data: resp,
        success: true,
    };
    let json = serde_json::to_string(&w).unwrap();
    let back: Wrapper = serde_json::from_str(&json).unwrap();
    assert_eq!(back.data.rows.len(), 1);
    assert_eq!(back.data.rows[0].fields[0].key, "n");
    // The JSONB variant must survive the custom serde (base64 `jsonb_value`
    // key both ways) — a representation change here breaks persisted payloads
    // and SDKs, so this is the format gate (mandate #8).
    assert!(
        matches!(
            back.data.rows[0].fields[2].value.as_ref().and_then(|v| v.value.as_ref()),
            Some(v1::sql_value::Value::JsonbValue(b)) if b == &[0x81, 0xA1, b'j', 0x01]
        ),
        "jsonb field must round-trip through the REST/serde representation"
    );
    assert!(
        json.contains("\"jsonb_value\""),
        "serde wire key for the JSONB variant is pinned: {json}"
    );
}
