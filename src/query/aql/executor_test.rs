#[cfg(test)]
mod tests {
    use crate::query::aql::executor::AqlExecutor;
    use crate::query::aql::{
        AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlValue, AqlWhere, DataModel,
        JoinType,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_aql_executor_join_audit() {
        // 1. Setup mock/dummy services (simplified)
        // Since we are testing the executor's coordination and audit trail,
        // we can use the real services with empty/dummy state if they don't panic.
        // Actually, it's better to create a mock AqlSource for pure executor testing.

        struct MockSource {
            model: DataModel,
        }

        #[async_trait::async_trait]
        impl crate::query::aql::AqlSource for MockSource {
            fn model(&self) -> DataModel {
                self.model
            }
            async fn execute(
                &self,
                _q: &AqlQuery,
                ctx: &mut crate::query::aql::AuditContext,
            ) -> crate::query::aql::Result<crate::query::aql::AqlResult> {
                let mut row = HashMap::new();
                row.insert("id".to_string(), AqlValue::String("test_id".to_string()));

                let frame_id = ctx.push_frame(crate::query::aql::AuditFrame {
                    frame_id: 0,
                    source: self.model,
                    op: crate::query::aql::AuditOp::Scan {
                        source: "mock".to_string(),
                    },
                    filters_pushed: vec![],
                    filters_post: vec![],
                    records_scanned: 1,
                    records_returned: 1,
                    wall_time_us: 100,
                    error: None,
                    redaction_count: 0,
                });

                Ok(crate::query::aql::AqlResult {
                    rows: vec![row],
                    frame_id,
                })
            }
        }

        let mut executor = AqlExecutor::new();
        executor.register_source(
            "vec".to_string(),
            Arc::new(MockSource {
                model: DataModel::Vector,
            }),
        );
        executor.register_source(
            "grp".to_string(),
            Arc::new(MockSource {
                model: DataModel::Graph,
            }),
        );

        // 2. Construct Join Query
        let query = AqlQuery {
            find: AqlFind {
                projections: vec![AqlProjection {
                    field: "*".to_string(),
                    alias: None,
                }],
            },
            from: AqlFrom::Join {
                left: Box::new(AqlFrom::Source {
                    name: "vec".to_string(),
                    alias: None,
                }),
                right: Box::new(AqlFrom::Source {
                    name: "grp".to_string(),
                    alias: None,
                }),
                on: AqlPredicate::Equals {
                    field: "id".to_string(),
                    value: AqlValue::String("test".to_string()),
                },
                join_type: JoinType::Inner,
            },
            where_clause: AqlWhere { predicate: None },
        };

        // 3. Execute
        let (result, trail) = executor.execute(query).await.unwrap();

        // 4. Verify
        assert_eq!(result.rows.len(), 1);
        assert_eq!(trail.frames.len(), 3); // 2 scans + 1 join

        assert_eq!(trail.frames[0].source, DataModel::Vector);
        assert_eq!(trail.frames[1].source, DataModel::Graph);
        assert_eq!(trail.frames[2].source, DataModel::Relational);

        if let crate::query::aql::AuditOp::Join {
            left_frame,
            right_frame,
            ..
        } = &trail.frames[2].op
        {
            assert_eq!(*left_frame, trail.frames[0].frame_id);
            assert_eq!(*right_frame, trail.frames[1].frame_id);
        } else {
            panic!("Last frame should be a join");
        }

        assert!(matches!(
            trail.outcome,
            crate::query::aql::AuditOutcome::Success
        ));
    }

    // ---------------------------------------------------------------------
    // TD-SECSCAN-1 §5 residual: a vector-valued `Equals` on a query-vector
    // field is the SEARCH INPUT, consumed by the vector / memory source. It is
    // stripped from the WHERE post-filter only where the source can have
    // consumed it (top level / AND conjunct). Under `Not` / `Or` it cannot be
    // a query vector, so it is rejected (InvalidInput → HTTP 400) instead of
    // `Not{..}` dropping every row and `Or{..}` always being true.
    // ---------------------------------------------------------------------

    /// Returns two rows: `kind = "a"` and `kind = "b"`.
    struct TwoRowSource;

    #[async_trait::async_trait]
    impl crate::query::aql::AqlSource for TwoRowSource {
        fn model(&self) -> DataModel {
            DataModel::Vector
        }
        async fn execute(
            &self,
            _q: &AqlQuery,
            _ctx: &mut crate::query::aql::AuditContext,
        ) -> crate::query::aql::Result<crate::query::aql::AqlResult> {
            let rows = ["a", "b"]
                .into_iter()
                .map(|kind| {
                    let mut row = HashMap::new();
                    row.insert("id".to_string(), AqlValue::String(format!("id-{kind}")));
                    row.insert("kind".to_string(), AqlValue::String(kind.to_string()));
                    row
                })
                .collect();
            Ok(crate::query::aql::AqlResult { rows, frame_id: 0 })
        }
    }

    fn vector_eq() -> AqlPredicate {
        AqlPredicate::Equals {
            field: "vector".to_string(),
            value: AqlValue::Vector(vec![1.0, 0.0]),
        }
    }

    fn kind_eq(kind: &str) -> AqlPredicate {
        AqlPredicate::Equals {
            field: "kind".to_string(),
            value: AqlValue::String(kind.to_string()),
        }
    }

    async fn run_where(
        predicate: AqlPredicate,
    ) -> crate::query::aql::Result<Vec<HashMap<String, AqlValue>>> {
        run_where_on(Arc::new(TwoRowSource), predicate).await
    }

    async fn run_where_on(
        source: Arc<dyn crate::query::aql::AqlSource>,
        predicate: AqlPredicate,
    ) -> crate::query::aql::Result<Vec<HashMap<String, AqlValue>>> {
        let mut executor = AqlExecutor::new();
        executor.register_source("vec".to_string(), source);
        let query = AqlQuery {
            find: AqlFind {
                projections: vec![AqlProjection {
                    field: "*".to_string(),
                    alias: None,
                }],
            },
            from: AqlFrom::Source {
                name: "vec".to_string(),
                alias: None,
            },
            where_clause: AqlWhere {
                predicate: Some(predicate),
            },
        };
        executor.execute(query).await.map(|(result, _)| result.rows)
    }

    fn kinds(rows: &[HashMap<String, AqlValue>]) -> Vec<String> {
        let mut kinds: Vec<String> = rows
            .iter()
            .filter_map(|row| match row.get("kind") {
                Some(AqlValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        kinds.sort();
        kinds
    }

    #[tokio::test]
    async fn consumed_query_vector_at_top_level_keeps_every_row() {
        let rows = run_where(vector_eq()).await.expect("top-level vector");
        assert_eq!(kinds(&rows), vec!["a", "b"]);
    }

    #[tokio::test]
    async fn consumed_query_vector_as_and_conjunct_leaves_the_rest_of_the_filter() {
        let rows = run_where(AqlPredicate::And {
            lhs: Box::new(vector_eq()),
            rhs: Box::new(kind_eq("a")),
        })
        .await
        .expect("AND conjunct");
        assert_eq!(kinds(&rows), vec!["a"]);
    }

    #[tokio::test]
    async fn query_vector_under_not_is_rejected_as_invalid_input() {
        let err = run_where(AqlPredicate::Not {
            inner: Box::new(vector_eq()),
        })
        .await
        .err();
        assert!(
            matches!(
                err,
                Some(proximadb_kernel::error::ProximaDBError::InvalidInput(ref m))
                    if m.contains("vector")
            ),
            "Not{{Equals vector}} must be a client error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn query_vector_under_or_is_rejected_as_invalid_input() {
        let err = run_where(AqlPredicate::Or {
            lhs: Box::new(vector_eq()),
            rhs: Box::new(kind_eq("a")),
        })
        .await
        .err();
        assert!(
            matches!(
                err,
                Some(proximadb_kernel::error::ProximaDBError::InvalidInput(ref m))
                    if m.contains("vector")
            ),
            "Or{{Equals vector, X}} must be a client error, got {err:?}"
        );
        // Nested deeper under Or (inside an And) is still not consumable.
        let err = run_where(AqlPredicate::Or {
            lhs: Box::new(AqlPredicate::And {
                lhs: Box::new(vector_eq()),
                rhs: Box::new(kind_eq("a")),
            }),
            rhs: Box::new(kind_eq("b")),
        })
        .await
        .err();
        assert!(
            matches!(
                err,
                Some(proximadb_kernel::error::ProximaDBError::InvalidInput(_))
            ),
            "Or{{And{{Equals vector, X}}, Y}} must be a client error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn scalar_equals_on_a_vector_field_still_filters() {
        // Only a VECTOR-valued Equals is the query input; a scalar Equals on
        // the same field name is an ordinary filter (no row has it ⇒ none).
        let rows = run_where(AqlPredicate::Equals {
            field: "vector".to_string(),
            value: AqlValue::String("x".to_string()),
        })
        .await
        .expect("scalar equals");
        assert!(rows.is_empty(), "scalar Equals must filter, got {rows:?}");
        // ...and under Not / Or it is NOT rejected.
        let rows = run_where(AqlPredicate::Not {
            inner: Box::new(AqlPredicate::Equals {
                field: "vector".to_string(),
                value: AqlValue::String("x".to_string()),
            }),
        })
        .await
        .expect("scalar equals under Not");
        assert_eq!(kinds(&rows), vec!["a", "b"]);
    }

    #[tokio::test]
    async fn query_vector_under_not_of_and_is_rejected_as_invalid_input() {
        let err = run_where(AqlPredicate::Not {
            inner: Box::new(AqlPredicate::And {
                lhs: Box::new(vector_eq()),
                rhs: Box::new(kind_eq("a")),
            }),
        })
        .await
        .err();
        assert!(
            matches!(
                err,
                Some(proximadb_kernel::error::ProximaDBError::InvalidInput(_))
            ),
            "Not{{And{{Equals vector, X}}}} must be a client error, got {err:?}"
        );
    }

    /// Counts `execute` calls, so a test can prove a rejected predicate never
    /// runs the (possibly expensive) source search.
    #[derive(Default)]
    struct CountingSource {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl crate::query::aql::AqlSource for CountingSource {
        fn model(&self) -> DataModel {
            DataModel::Vector
        }
        async fn execute(
            &self,
            _q: &AqlQuery,
            _ctx: &mut crate::query::aql::AuditContext,
        ) -> crate::query::aql::Result<crate::query::aql::AqlResult> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(crate::query::aql::AqlResult {
                rows: Vec::new(),
                frame_id: 0,
            })
        }
    }

    #[tokio::test]
    async fn rejected_query_vector_never_runs_the_source() {
        let source = Arc::new(CountingSource::default());
        let rejected = run_where_on(
            source.clone(),
            AqlPredicate::Not {
                inner: Box::new(vector_eq()),
            },
        )
        .await;
        assert!(rejected.is_err());
        assert_eq!(
            source.calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the source must not run for a rejected predicate"
        );
        // Control: an accepted predicate does run it.
        run_where_on(source.clone(), vector_eq())
            .await
            .expect("accepted predicate");
        assert_eq!(source.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
