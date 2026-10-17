/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use proximadb_data_model::MemoryType;
use proximadb_records::ProximaRecord;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[test]
fn test_memory_type_assignment() {
    let mut record = ProximaRecord::default();
    record.memory_type = Some(MemoryType::Decision);

    assert_eq!(record.memory_type, Some(MemoryType::Decision));

    let json = serde_json::to_string(&record).unwrap();
    let back: ProximaRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.memory_type, Some(MemoryType::Decision));
}

#[test]
fn test_conflict_resolution_lww() {
    let mut r1 = ProximaRecord::default();
    r1.oid = "mem1".to_string();
    r1.updated_at_ns = 1000;
    r1.memory_type = Some(MemoryType::Fact);

    let mut r2 = ProximaRecord::default();
    r2.oid = "mem1".to_string();
    r2.updated_at_ns = 2000;
    r2.memory_type = Some(MemoryType::Decision);

    let resolved = r1.resolve_conflict(&r2);
    assert_eq!(resolved.memory_type, Some(MemoryType::Decision));
    assert_eq!(resolved.updated_at_ns, 2000);
}

// Mocking AQL behavior for unit-style verification of the filtering logic
#[tokio::test]
async fn test_aql_type_filtering_logic() {
    use async_trait::async_trait;
    use proximadb::query::aql::executor::AqlExecutor;
    use proximadb::query::aql::{
        AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlResult, AqlSource, AqlValue,
        AqlWhere, AuditContext, AuditOp, DataModel,
    };
    use std::sync::Arc;

    struct MockSource;
    #[async_trait]
    impl AqlSource for MockSource {
        fn model(&self) -> DataModel {
            DataModel::Document
        }
        async fn execute(
            &self,
            _query: &AqlQuery,
            _ctx: &mut AuditContext,
        ) -> proximadb::query::aql::Result<AqlResult> {
            let mut rows = Vec::new();

            let mut row1 = HashMap::new();
            row1.insert("id".to_string(), AqlValue::String("1".to_string()));
            row1.insert(
                "memory_type".to_string(),
                AqlValue::String("fact".to_string()),
            );
            rows.push(row1);

            let mut row2 = HashMap::new();
            row2.insert("id".to_string(), AqlValue::String("2".to_string()));
            row2.insert(
                "memory_type".to_string(),
                AqlValue::String("decision".to_string()),
            );
            rows.push(row2);

            Ok(AqlResult { rows, frame_id: 1 })
        }
    }

    let mut executor = AqlExecutor::new();
    executor.register_source("mem".to_string(), Arc::new(MockSource));

    let query = AqlQuery {
        find: AqlFind {
            projections: vec![AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: AqlFrom::Source {
            name: "mem".to_string(),
            alias: None,
        },
        where_clause: AqlWhere {
            predicate: Some(AqlPredicate::TypeMatch {
                memory_type: MemoryType::Decision,
            }),
        },
    };

    let (result, trail) = executor.execute_as_internal(query).await.unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].get("id").unwrap(),
        &AqlValue::String("2".to_string())
    );

    assert!(trail.frames.iter().any(|f| matches!(
        f.op,
        AuditOp::TypeMatch {
            memory_type: MemoryType::Decision
        }
    )));
}

#[tokio::test]
async fn test_aql_jsonb_filtering() {
    use async_trait::async_trait;
    use proximadb::query::aql::executor::AqlExecutor;
    use proximadb::query::aql::{
        AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlResult, AqlSource, AqlValue,
        AqlWhere, AuditContext, DataModel,
    };
    use serde_json::json;
    use std::sync::Arc;

    struct MockJsonSource;
    #[async_trait]
    impl AqlSource for MockJsonSource {
        fn model(&self) -> DataModel {
            DataModel::Document
        }
        async fn execute(
            &self,
            _query: &AqlQuery,
            _ctx: &mut AuditContext,
        ) -> proximadb::query::aql::Result<AqlResult> {
            let mut rows = Vec::new();

            let mut row1 = HashMap::new();
            row1.insert("id".to_string(), AqlValue::String("1".to_string()));
            row1.insert(
                "data".to_string(),
                AqlValue::Jsonb(json!({"priority": "high", "active": true})),
            );
            rows.push(row1);

            let mut row2 = HashMap::new();
            row2.insert("id".to_string(), AqlValue::String("2".to_string()));
            row2.insert(
                "data".to_string(),
                AqlValue::Jsonb(json!({"priority": "low", "active": false})),
            );
            rows.push(row2);

            Ok(AqlResult { rows, frame_id: 1 })
        }
    }

    let mut executor = AqlExecutor::new();
    executor.register_source("json_mem".to_string(), Arc::new(MockJsonSource));

    let query = AqlQuery {
        find: AqlFind {
            projections: vec![AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: AqlFrom::Source {
            name: "json_mem".to_string(),
            alias: None,
        },
        where_clause: AqlWhere {
            predicate: Some(AqlPredicate::Equals {
                field: "data".to_string(),
                value: AqlValue::Jsonb(json!({"priority": "high", "active": true})),
            }),
        },
    };

    let (result, _) = executor.execute_as_internal(query).await.unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].get("id").unwrap(),
        &AqlValue::String("1".to_string())
    );
}

#[tokio::test]
async fn test_aql_type_filtering_reduces_topic_bleed() {
    use async_trait::async_trait;
    use proximadb::query::aql::executor::AqlExecutor;
    use proximadb::query::aql::{
        AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlResult, AqlSource, AqlValue,
        AqlWhere, AuditContext, DataModel,
    };
    use std::sync::Arc;

    struct MockTopicSource;

    #[async_trait]
    impl AqlSource for MockTopicSource {
        fn model(&self) -> DataModel {
            DataModel::Document
        }

        async fn execute(
            &self,
            _query: &AqlQuery,
            _ctx: &mut AuditContext,
        ) -> proximadb::query::aql::Result<AqlResult> {
            let mut rows = Vec::new();

            let mut row1 = HashMap::new();
            row1.insert("id".to_string(), AqlValue::String("pref1".to_string()));
            row1.insert(
                "memory_type".to_string(),
                AqlValue::String("preference".to_string()),
            );
            row1.insert(
                "text".to_string(),
                AqlValue::String("release notes should mention deprecation".to_string()),
            );
            rows.push(row1);

            let mut row2 = HashMap::new();
            row2.insert("id".to_string(), AqlValue::String("fact1".to_string()));
            row2.insert(
                "memory_type".to_string(),
                AqlValue::String("fact".to_string()),
            );
            row2.insert(
                "text".to_string(),
                AqlValue::String("release notes should mention deprecation".to_string()),
            );
            rows.push(row2);

            let mut row3 = HashMap::new();
            row3.insert("id".to_string(), AqlValue::String("decision1".to_string()));
            row3.insert(
                "memory_type".to_string(),
                AqlValue::String("decision".to_string()),
            );
            row3.insert(
                "text".to_string(),
                AqlValue::String("release notes should mention deprecation".to_string()),
            );
            rows.push(row3);

            Ok(AqlResult { rows, frame_id: 1 })
        }
    }

    let mut executor = AqlExecutor::new();
    executor.register_source("topic_mem".to_string(), Arc::new(MockTopicSource));

    let query_without_type = AqlQuery {
        find: AqlFind {
            projections: vec![AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: AqlFrom::Source {
            name: "topic_mem".to_string(),
            alias: None,
        },
        where_clause: AqlWhere {
            predicate: Some(AqlPredicate::Contains {
                field: "text".to_string(),
                value: AqlValue::String("release notes".to_string()),
            }),
        },
    };

    let (unfiltered, _) = executor
        .execute_as_internal(query_without_type)
        .await
        .unwrap();
    assert_eq!(unfiltered.rows.len(), 3);

    let query_with_type = AqlQuery {
        find: AqlFind {
            projections: vec![AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: AqlFrom::Source {
            name: "topic_mem".to_string(),
            alias: None,
        },
        where_clause: AqlWhere {
            predicate: Some(AqlPredicate::And {
                lhs: Box::new(AqlPredicate::TypeMatch {
                    memory_type: MemoryType::Decision,
                }),
                rhs: Box::new(AqlPredicate::Contains {
                    field: "text".to_string(),
                    value: AqlValue::String("release notes".to_string()),
                }),
            }),
        },
    };

    let (filtered, _) = executor.execute_as_internal(query_with_type).await.unwrap();
    assert_eq!(filtered.rows.len(), 1);
    assert_eq!(
        filtered.rows[0].get("id"),
        Some(&AqlValue::String("decision1".to_string()))
    );
}

#[tokio::test]
async fn test_aql_type_filtering_preserves_intent_recall() {
    use async_trait::async_trait;
    use proximadb::query::aql::executor::AqlExecutor;
    use proximadb::query::aql::{
        AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlResult, AqlSource, AqlValue,
        AqlWhere, AuditContext, DataModel,
    };
    use std::sync::Arc;

    struct MockRecallSource;

    #[async_trait]
    impl AqlSource for MockRecallSource {
        fn model(&self) -> DataModel {
            DataModel::Document
        }

        async fn execute(
            &self,
            _query: &AqlQuery,
            _ctx: &mut AuditContext,
        ) -> proximadb::query::aql::Result<AqlResult> {
            let rows = vec![
                [
                    ("id", AqlValue::String("pref-1".to_string())),
                    ("memory_type", AqlValue::String("preference".to_string())),
                    ("text", AqlValue::String("use fast path index".to_string())),
                ],
                [
                    ("id", AqlValue::String("decision-1".to_string())),
                    ("memory_type", AqlValue::String("decision".to_string())),
                    ("text", AqlValue::String("use fast path index".to_string())),
                ],
                [
                    ("id", AqlValue::String("decision-2".to_string())),
                    ("memory_type", AqlValue::String("decision".to_string())),
                    (
                        "text",
                        AqlValue::String("enable fallback safety".to_string()),
                    ),
                ],
                [
                    ("id", AqlValue::String("decision-3".to_string())),
                    ("memory_type", AqlValue::String("decision".to_string())),
                    ("text", AqlValue::String("use fast path index".to_string())),
                ],
            ]
            .into_iter()
            .map(|items| {
                let mut row = HashMap::new();
                for (k, v) in items {
                    row.insert(k.to_string(), v);
                }
                row
            })
            .collect();

            Ok(AqlResult { rows, frame_id: 1 })
        }
    }

    let mut executor = AqlExecutor::new();
    executor.register_source("recall_mem".to_string(), Arc::new(MockRecallSource));

    let decision_context_query = AqlQuery {
        find: AqlFind {
            projections: vec![AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: AqlFrom::Source {
            name: "recall_mem".to_string(),
            alias: None,
        },
        where_clause: AqlWhere {
            predicate: Some(AqlPredicate::And {
                lhs: Box::new(AqlPredicate::TypeMatch {
                    memory_type: MemoryType::Decision,
                }),
                rhs: Box::new(AqlPredicate::Contains {
                    field: "text".to_string(),
                    value: AqlValue::String("fast path".to_string()),
                }),
            }),
        },
    };

    let (decision_rows, _) = executor
        .execute_as_internal(decision_context_query)
        .await
        .unwrap();
    assert_eq!(
        decision_rows.rows.len(),
        2,
        "should return all matching decision memories"
    );
    let returned_ids: Vec<&str> = decision_rows
        .rows
        .iter()
        .filter_map(|row| match row.get("id") {
            Some(AqlValue::String(id)) => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert!(returned_ids.contains(&"decision-1"));
    assert!(returned_ids.contains(&"decision-3"));
    assert!(!returned_ids.contains(&"pref-1"));
}

#[tokio::test]
async fn test_aql_type_filtering_integration_full_path() {
    use proximadb::proto::proximadb_v1::sql_value::Value as SqlValueData;
    use proximadb::proto::proximadb_v1::{DocumentCollectionConfig, SqlObject, SqlValue};
    use proximadb::query::aql;
    use proximadb::query::aql::executor::AqlExecutor;
    use proximadb::query::aql::sources::document::DocumentAqlSource;
    use proximadb::storage::document::DocumentService;
    use proximadb::storage::engines::sst::SstEngine;
    use std::collections::HashMap;

    let collection_id = format!("typed_mem_full_aql_{}", Uuid::new_v4().simple());

    let storage_engine = Arc::new(SstEngine::new().await.unwrap());
    let document_service = Arc::new(DocumentService::new(storage_engine));

    document_service
        .create_collection(
            &collection_id,
            DocumentCollectionConfig {
                name: collection_id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let mut fact = HashMap::new();
    fact.insert(
        "memory_type".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(MemoryType::Fact.to_string())),
        },
    );
    fact.insert(
        "text".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(
                "fact-level context that should be filtered out".to_string(),
            )),
        },
    );
    let mut decision = HashMap::new();
    decision.insert(
        "memory_type".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(MemoryType::Decision.to_string())),
        },
    );
    decision.insert(
        "text".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(
                "decision-level instruction for next steps".to_string(),
            )),
        },
    );
    let mut error = HashMap::new();
    error.insert(
        "memory_type".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(MemoryType::Error.to_string())),
        },
    );
    error.insert(
        "text".to_string(),
        SqlValue {
            value: Some(SqlValueData::StringValue(
                "error note should not match decision".to_string(),
            )),
        },
    );

    document_service
        .insert_document(&collection_id, Some("mem-fact"), SqlObject { fields: fact })
        .await
        .unwrap();
    document_service
        .insert_document(
            &collection_id,
            Some("mem-decision"),
            SqlObject { fields: decision },
        )
        .await
        .unwrap();
    document_service
        .insert_document(
            &collection_id,
            Some("mem-error"),
            SqlObject { fields: error },
        )
        .await
        .unwrap();

    let mut executor = AqlExecutor::new();
    executor.register_source(
        collection_id.clone(),
        Arc::new(DocumentAqlSource::new(document_service)),
    );

    let query = aql::AqlQuery {
        find: aql::AqlFind {
            projections: vec![aql::AqlProjection {
                field: "*".to_string(),
                alias: None,
            }],
        },
        from: aql::AqlFrom::Source {
            name: collection_id.clone(),
            alias: None,
        },
        where_clause: aql::AqlWhere {
            predicate: Some(aql::AqlPredicate::TypeMatch {
                memory_type: MemoryType::Decision,
            }),
        },
    };

    let (result, trail) = executor.execute_as_internal(query).await.unwrap();

    assert_eq!(
        result.rows.len(),
        1,
        "only decision memory should match type filter"
    );

    let row = result.rows.first().expect("a matching row should exist");
    assert_eq!(
        row.get("memory_type"),
        Some(&aql::AqlValue::String(MemoryType::Decision.to_string()))
    );
    assert_eq!(
        row.get("_id"),
        Some(&aql::AqlValue::String("mem-decision".to_string()))
    );

    assert!(
        trail.frames
            .iter()
            .any(|frame| matches!(frame.op, aql::AuditOp::TypeMatch { memory_type } if memory_type == MemoryType::Decision)),
        "audit trail should include type-match predicate frame for document-filtering"
    );
}
