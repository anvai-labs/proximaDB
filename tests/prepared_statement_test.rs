//! Integration Tests for Prepared Statements
//!
//! Tests the PreparedStatementCache functionality for the "parse once, execute many"
//! pattern that is essential for high-performance agentic AI workloads.
//!
//! Run with: `cargo test --test prepared_statement_test`

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use proximadb::query::prepared::{
    ParameterValue, PreparedStatementCache, PreparedStatementConfig, PreparedStatementError,
    PreparedStatementId,
};

// ================================
// Statement Preparation Tests
// ================================

/// Test basic statement preparation
#[test]
fn test_prepare_simple_statement() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test WHERE id = $1").unwrap();

    assert!(!id.is_empty());
    assert!(id.starts_with("stmt_"));
    assert!(cache.exists(&id));
    assert_eq!(cache.len(), 1);
}

/// Test preparing multiple statements
#[test]
fn test_prepare_multiple_statements() {
    let cache = PreparedStatementCache::with_defaults();

    let id1 = cache.prepare("SELECT * FROM test WHERE id = $1").unwrap();
    let id2 = cache
        .prepare("SELECT * FROM users WHERE name = $1")
        .unwrap();
    let id3 = cache
        .prepare("SELECT * FROM products WHERE price > $1")
        .unwrap();

    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_eq!(cache.len(), 3);

    assert!(cache.exists(&id1));
    assert!(cache.exists(&id2));
    assert!(cache.exists(&id3));
}

/// Test preparing a statement with vector search syntax
#[test]
fn test_prepare_vector_search_statement() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM VECTOR_SEARCH($1, $2, 10)")
        .unwrap();

    let statement = cache.get(&id).unwrap();
    assert_eq!(statement.parameter_count(), 2);
}

/// Test preparing a statement with no parameters
#[test]
fn test_prepare_no_parameters() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test LIMIT 10").unwrap();

    let statement = cache.get(&id).unwrap();
    assert_eq!(statement.parameter_count(), 0);
}

/// Test preparing a statement with many parameters
#[test]
fn test_prepare_many_parameters() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM t WHERE a=$1 AND b=$2 AND c=$3 AND d=$4 AND e=$5")
        .unwrap();

    let statement = cache.get(&id).unwrap();
    assert_eq!(statement.parameter_count(), 5);
}

/// Test preparing a statement with duplicate parameter references
#[test]
fn test_prepare_duplicate_parameters() {
    let cache = PreparedStatementCache::with_defaults();

    // $1 appears twice, $2 once
    let id = cache
        .prepare("SELECT * FROM t WHERE col1 = $1 AND col2 = $1 AND col3 = $2")
        .unwrap();

    let statement = cache.get(&id).unwrap();
    // Should only count unique parameters
    assert_eq!(statement.parameter_count(), 2);
}

/// Test parameter gap detection (missing $2)
#[test]
fn test_parameter_gap_error() {
    let cache = PreparedStatementCache::with_defaults();

    let result = cache.prepare("SELECT * FROM t WHERE a = $1 AND b = $3");

    assert!(result.is_err());
    match result {
        Err(PreparedStatementError::InvalidParameter(msg)) => {
            assert!(msg.contains("Missing parameter $2"));
        }
        _ => panic!("Expected InvalidParameter error for gap"),
    }
}

/// Test that parameters inside string literals are ignored
#[test]
fn test_parameters_in_strings_ignored() {
    let cache = PreparedStatementCache::with_defaults();

    // $1 in string should be ignored, only the real $1 counts
    let id = cache
        .prepare("SELECT * FROM t WHERE name = 'test$2literal' AND id = $1")
        .unwrap();

    let statement = cache.get(&id).unwrap();
    assert_eq!(statement.parameter_count(), 1);
}

// ================================
// Parameter Binding Tests
// ================================

/// Test binding integer parameters
#[test]
fn test_bind_integer_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test WHERE id = $1").unwrap();

    let sql = cache.execute_sql(&id, &[ParameterValue::Int(42)]).unwrap();

    assert!(sql.contains("42"));
    assert!(!sql.contains("$1"));
}

/// Test binding string parameters
#[test]
fn test_bind_string_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test WHERE name = $1").unwrap();

    let sql = cache
        .execute_sql(&id, &[ParameterValue::String("Alice".to_string())])
        .unwrap();

    assert!(sql.contains("'Alice'"));
    assert!(!sql.contains("$1"));
}

/// Test binding float parameters
#[test]
fn test_bind_float_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM test WHERE price > $1")
        .unwrap();

    let sql = cache
        .execute_sql(&id, &[ParameterValue::Float(19.99)])
        .unwrap();

    assert!(sql.contains("19.99"));
}

/// Test binding boolean parameters
#[test]
fn test_bind_boolean_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM test WHERE active = $1")
        .unwrap();

    let sql_true = cache
        .execute_sql(&id, &[ParameterValue::Bool(true)])
        .unwrap();
    assert!(sql_true.contains("true"));

    let sql_false = cache
        .execute_sql(&id, &[ParameterValue::Bool(false)])
        .unwrap();
    assert!(sql_false.contains("false"));
}

/// Test binding null parameter
#[test]
fn test_bind_null_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM test WHERE value IS $1")
        .unwrap();

    let sql = cache.execute_sql(&id, &[ParameterValue::Null]).unwrap();

    assert!(sql.contains("NULL"));
}

/// Test binding vector parameters
#[test]
fn test_bind_vector_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM VECTOR_SEARCH($1, $2, 10)")
        .unwrap();

    let sql = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::String("embeddings".to_string()),
                ParameterValue::Vector(vec![0.1, 0.2, 0.3, 0.4]),
            ],
        )
        .unwrap();

    assert!(sql.contains("'embeddings'"));
    // QUOTED vector splice (bare '[...]' was the round-15/20 churn — the
    // substring form passed under both spellings).
    assert!(sql.contains("'[0.1,0.2,0.3,0.4]'"));
}

/// Test binding JSON parameter
#[test]
fn test_bind_json_parameter() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM DOCUMENT_QUERY($1, $2)")
        .unwrap();

    let json_value = serde_json::json!({"category": "electronics", "price": 100});

    let sql = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::String("products".to_string()),
                ParameterValue::Json(json_value),
            ],
        )
        .unwrap();

    // STRICT pin of the Json splice shape (substring checks passed under
    // every prior spelling the branch cycled through): a JSON OBJECT
    // splices as QUOTED compact JSON text.
    assert!(
        sql.contains("'{\"category\":\"electronics\""),
        "json container param must splice as quoted compact JSON, got: {sql}"
    );
}

/// Test binding multiple mixed-type parameters
#[test]
fn test_bind_multiple_mixed_types() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM t WHERE id = $1 AND name = $2 AND price = $3 AND active = $4")
        .unwrap();

    let sql = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::Int(123),
                ParameterValue::String("Widget".to_string()),
                ParameterValue::Float(9.99),
                ParameterValue::Bool(true),
            ],
        )
        .unwrap();

    assert!(sql.contains("123"));
    assert!(sql.contains("'Widget'"));
    assert!(sql.contains("9.99"));
    assert!(sql.contains("true"));
}

/// Test parameter count mismatch - too few parameters
#[test]
fn test_parameter_count_mismatch_too_few() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM t WHERE a = $1 AND b = $2")
        .unwrap();

    let result = cache.execute_sql(&id, &[ParameterValue::Int(1)]);

    assert!(result.is_err());
    match result {
        Err(PreparedStatementError::ParameterCountMismatch { expected, actual }) => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        _ => panic!("Expected ParameterCountMismatch error"),
    }
}

/// Test parameter count mismatch - too many parameters
#[test]
fn test_parameter_count_mismatch_too_many() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM t WHERE a = $1").unwrap();

    let result = cache.execute_sql(
        &id,
        &[
            ParameterValue::Int(1),
            ParameterValue::String("extra".into()),
        ],
    );

    assert!(result.is_err());
    match result {
        Err(PreparedStatementError::ParameterCountMismatch { expected, actual }) => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 2);
        }
        _ => panic!("Expected ParameterCountMismatch error"),
    }
}

/// Test string with special characters that need escaping
#[test]
fn test_bind_string_with_quotes() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM t WHERE name = $1").unwrap();

    let sql = cache
        .execute_sql(&id, &[ParameterValue::String("O'Brien".to_string())])
        .unwrap();

    // Single quotes should be escaped as ''
    assert!(sql.contains("O''Brien"));
}

// ================================
// Statement Execution Tests
// ================================

/// Test executing same statement with different parameters (parse once, execute many)
#[test]
fn test_execute_same_statement_different_params() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM users WHERE user_id = $1 AND status = $2")
        .unwrap();

    // Execute with first set of parameters
    let sql1 = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::Int(1),
                ParameterValue::String("active".to_string()),
            ],
        )
        .unwrap();

    // Execute with second set of parameters
    let sql2 = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::Int(2),
                ParameterValue::String("inactive".to_string()),
            ],
        )
        .unwrap();

    // Execute with third set of parameters
    let sql3 = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::Int(999),
                ParameterValue::String("pending".to_string()),
            ],
        )
        .unwrap();

    // All should produce different SQL
    assert_ne!(sql1, sql2);
    assert_ne!(sql2, sql3);

    // But all should have the same structure
    assert!(sql1.contains("user_id ="));
    assert!(sql2.contains("user_id ="));
    assert!(sql3.contains("user_id ="));

    // Statement should still exist after multiple executions
    assert!(cache.exists(&id));
}

/// Test statement reuse preserves the parsed query
#[test]
fn test_statement_reuse_preserves_parsed_query() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache
        .prepare("SELECT * FROM VECTOR_SEARCH($1, $2, 10)")
        .unwrap();

    // Get the statement and verify parsed query is available
    let stmt1 = cache.get(&id).unwrap();
    let plan1 = cache.get_plan(&id).unwrap();

    // Execute with parameters
    let _ = cache
        .execute_sql(
            &id,
            &[
                ParameterValue::String("collection".to_string()),
                ParameterValue::Vector(vec![0.1, 0.2]),
            ],
        )
        .unwrap();

    // Get again and verify it's the same parsed query (Arc should be same)
    let stmt2 = cache.get(&id).unwrap();
    let plan2 = cache.get_plan(&id).unwrap();

    // The original SQL should be identical
    assert_eq!(stmt1.original_sql, stmt2.original_sql);
    // The Arc pointers should point to the same data
    assert!(Arc::ptr_eq(&plan1, &plan2));
}

/// Test getting parsed query and plan from cache
#[test]
fn test_get_parsed_query_and_plan() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test WHERE id = $1").unwrap();

    // Should be able to get the parsed query
    let parsed = cache.get_parsed_query(&id);
    assert!(parsed.is_ok());

    // Should be able to get the optimized plan
    let plan = cache.get_plan(&id);
    assert!(plan.is_ok());
}

/// Test execute on non-existent statement
#[test]
fn test_execute_nonexistent_statement() {
    let cache = PreparedStatementCache::with_defaults();

    let result = cache.execute_sql("nonexistent_id", &[]);

    assert!(result.is_err());
    match result {
        Err(PreparedStatementError::NotFound(id)) => {
            assert_eq!(id, "nonexistent_id");
        }
        _ => panic!("Expected NotFound error"),
    }
}

// ================================
// Statement Invalidation Tests
// ================================

/// Test dropping a prepared statement
#[test]
fn test_drop_statement() {
    let cache = PreparedStatementCache::with_defaults();

    let id = cache.prepare("SELECT * FROM test").unwrap();
    assert!(cache.exists(&id));

    cache.drop_statement(&id).unwrap();
    assert!(!cache.exists(&id));

    // Trying to drop again should fail
    let result = cache.drop_statement(&id);
    assert!(matches!(result, Err(PreparedStatementError::NotFound(_))));
}

/// Test clearing all statements
#[test]
fn test_clear_all_statements() {
    let cache = PreparedStatementCache::with_defaults();

    // Prepare several statements
    cache.prepare("SELECT 1").unwrap();
    cache.prepare("SELECT 2").unwrap();
    cache.prepare("SELECT 3").unwrap();

    assert_eq!(cache.len(), 3);

    cache.clear();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Test invalidation for collection by finding and removing matching statements
#[test]
fn test_invalidate_for_collection() {
    let cache = PreparedStatementCache::with_defaults();

    // Prepare statements for different collections
    let id1 = cache
        .prepare("SELECT * FROM products WHERE id = $1")
        .unwrap();
    let id2 = cache
        .prepare("SELECT * FROM products WHERE price > $1")
        .unwrap();
    let id3 = cache
        .prepare("SELECT * FROM users WHERE name = $1")
        .unwrap();
    let id4 = cache
        .prepare("SELECT * FROM orders WHERE user_id = $1")
        .unwrap();

    assert_eq!(cache.len(), 4);

    // Manually invalidate statements for "products" collection
    // (simulating what invalidate_for_collection would do)
    let product_ids: Vec<String> = vec![id1.clone(), id2.clone()];
    for id in product_ids {
        let stmt = cache.get(&id);
        if let Ok(s) = stmt
            && s.original_sql.contains("products")
        {
            cache.drop_statement(&id).ok();
        }
    }

    // products statements should be gone
    assert!(!cache.exists(&id1));
    assert!(!cache.exists(&id2));

    // users and orders statements should remain
    assert!(cache.exists(&id3));
    assert!(cache.exists(&id4));
}

/// Test that dropping non-existent statement returns proper error
#[test]
fn test_drop_nonexistent_statement() {
    let cache = PreparedStatementCache::with_defaults();

    let result = cache.drop_statement("does_not_exist");

    assert!(matches!(result, Err(PreparedStatementError::NotFound(_))));
}

// ================================
// TTL Expiration Tests
// ================================

/// Test statement TTL expiration
#[test]
fn test_statement_ttl_expiration() {
    let config = PreparedStatementConfig {
        max_statements: 100,
        default_ttl: Duration::from_millis(50),
        enable_cleanup: false,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    let id = cache.prepare("SELECT * FROM test").unwrap();
    assert!(cache.exists(&id));

    // Statement should be accessible immediately
    assert!(cache.get(&id).is_ok());

    // Wait for TTL to expire
    thread::sleep(Duration::from_millis(100));

    // Statement should now be expired
    let result = cache.get(&id);
    assert!(matches!(result, Err(PreparedStatementError::Expired(_))));

    // After expired access, statement should be removed from cache
    assert!(!cache.exists(&id));
}

/// Test custom TTL per statement
#[test]
fn test_custom_statement_ttl() {
    let cache = PreparedStatementCache::with_defaults();

    // Prepare with very short TTL
    let short_id = cache
        .prepare_with_ttl("SELECT 1", Duration::from_millis(10))
        .unwrap();

    // Prepare with longer TTL
    let long_id = cache
        .prepare_with_ttl("SELECT 2", Duration::from_secs(3600))
        .unwrap();

    // Wait for short TTL to expire
    thread::sleep(Duration::from_millis(50));

    // Short TTL statement should be expired
    let short_result = cache.get(&short_id);
    assert!(matches!(
        short_result,
        Err(PreparedStatementError::Expired(_))
    ));

    // Long TTL statement should still be valid
    let long_result = cache.get(&long_id);
    assert!(long_result.is_ok());
}

/// Test cleanup of expired statements
#[test]
fn test_cleanup_expired_statements() {
    let config = PreparedStatementConfig {
        max_statements: 100,
        default_ttl: Duration::from_millis(10),
        enable_cleanup: true,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    // Prepare several statements
    cache.prepare("SELECT 1").unwrap();
    cache.prepare("SELECT 2").unwrap();
    cache.prepare("SELECT 3").unwrap();

    assert_eq!(cache.len(), 3);

    // Wait for expiration
    thread::sleep(Duration::from_millis(50));

    // Trigger manual cleanup
    let cleaned = cache.cleanup_expired();

    assert_eq!(cleaned, 3);
    assert!(cache.is_empty());
}

/// Test that accessing a statement refreshes its last_accessed time
#[test]
fn test_access_refreshes_ttl() {
    let config = PreparedStatementConfig {
        max_statements: 100,
        default_ttl: Duration::from_millis(100),
        enable_cleanup: false,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    let id = cache.prepare("SELECT * FROM test").unwrap();

    // Access the statement periodically to refresh TTL
    for _ in 0..5 {
        thread::sleep(Duration::from_millis(30));
        // Each get() call should refresh the last_accessed time
        let result = cache.get(&id);
        assert!(
            result.is_ok(),
            "Statement should not expire with frequent access"
        );
    }

    // After 5 accesses over 150ms, the statement should still be valid
    // because each access refreshes the TTL
    assert!(cache.exists(&id));
}

// ================================
// Max Statements and LRU Eviction Tests
// ================================

/// Test max_statements limit
#[test]
fn test_max_statements_limit() {
    let config = PreparedStatementConfig {
        max_statements: 3,
        default_ttl: Duration::from_secs(3600),
        enable_cleanup: false,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    // Fill up the cache
    cache.prepare("SELECT 1").unwrap();
    cache.prepare("SELECT 2").unwrap();
    cache.prepare("SELECT 3").unwrap();

    assert_eq!(cache.len(), 3);

    // Trying to add a 4th statement should fail (cache is full)
    let result = cache.prepare("SELECT 4");

    assert!(result.is_err());
    match result {
        Err(PreparedStatementError::CacheFull(max)) => {
            assert_eq!(max, 3);
        }
        _ => panic!("Expected CacheFull error"),
    }
}

/// Test that expired statements are evicted to make room
#[test]
fn test_expired_eviction_makes_room() {
    let config = PreparedStatementConfig {
        max_statements: 2,
        default_ttl: Duration::from_millis(10),
        enable_cleanup: false,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    // Fill up the cache
    cache.prepare("SELECT 1").unwrap();
    cache.prepare("SELECT 2").unwrap();

    // Wait for expiration
    thread::sleep(Duration::from_millis(50));

    // Now a new prepare should succeed because cleanup_expired is called
    let result = cache.prepare("SELECT 3");
    assert!(
        result.is_ok(),
        "Should succeed after expired statements are cleaned up"
    );
}

/// Test cache statistics
#[test]
fn test_cache_statistics() {
    let config = PreparedStatementConfig {
        max_statements: 100,
        default_ttl: Duration::from_secs(3600),
        enable_cleanup: false,
        cleanup_interval: Duration::from_secs(300),
    };
    let cache = PreparedStatementCache::new(config);

    // Empty cache stats
    let stats = cache.stats();
    assert_eq!(stats.cached_statements, 0);
    assert_eq!(stats.max_statements, 100);
    assert_eq!(stats.total_access_count, 0);

    // Add some statements
    let id1 = cache.prepare("SELECT 1").unwrap();
    let id2 = cache.prepare("SELECT 2").unwrap();

    // Access statements multiple times
    cache.get(&id1).unwrap();
    cache.get(&id1).unwrap();
    cache.get(&id2).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.cached_statements, 2);
    assert_eq!(stats.total_access_count, 3);
}

// ================================
// Concurrent Access Tests
// ================================

/// Test concurrent access to the cache
#[test]
fn test_concurrent_access() {
    let cache = Arc::new(PreparedStatementCache::with_defaults());
    let mut handles = vec![];

    // Spawn multiple threads that prepare and execute statements
    for i in 0..10 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            let id = cache_clone
                .prepare(&format!("SELECT * FROM t{} WHERE id = $1", i))
                .unwrap();

            // Execute multiple times
            for j in 0..5 {
                let _ = cache_clone.execute_sql(&id, &[ParameterValue::Int(j)]);
            }

            id
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    let ids: Vec<PreparedStatementId> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All statements should exist
    for id in &ids {
        assert!(cache.exists(id));
    }

    assert_eq!(cache.len(), 10);
}

/// Test concurrent prepare and drop
#[test]
fn test_concurrent_prepare_and_drop() {
    let cache = Arc::new(PreparedStatementCache::with_defaults());
    let mut handles = vec![];

    // Spawn threads that prepare statements
    for i in 0..5 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || cache_clone.prepare(&format!("SELECT {}", i)).unwrap());
        handles.push(handle);
    }

    let ids: Vec<PreparedStatementId> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Now spawn threads that drop some statements while others access them
    let mut drop_handles = vec![];
    for (i, id) in ids.into_iter().enumerate() {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            if i % 2 == 0 {
                let _ = cache_clone.drop_statement(&id);
            } else {
                let _ = cache_clone.get(&id);
            }
        });
        drop_handles.push(handle);
    }

    for handle in drop_handles {
        handle.join().unwrap();
    }

    // Cache should be in a consistent state
    assert!(cache.len() <= 5);
}

// ================================
// ParameterValue Conversion Tests
// ================================

/// Test From implementations for ParameterValue
#[test]
fn test_parameter_value_from_conversions() {
    // From &str
    let pv: ParameterValue = "hello".into();
    assert!(matches!(pv, ParameterValue::String(s) if s == "hello"));

    // From String
    let pv: ParameterValue = String::from("world").into();
    assert!(matches!(pv, ParameterValue::String(s) if s == "world"));

    // From i64
    let pv: ParameterValue = 42i64.into();
    assert!(matches!(pv, ParameterValue::Int(42)));

    // From f64
    let pv: ParameterValue = 3.5f64.into();
    assert!(matches!(pv, ParameterValue::Float(f) if (f - 3.5).abs() < 0.001));

    // From bool
    let pv: ParameterValue = true.into();
    assert!(matches!(pv, ParameterValue::Bool(true)));

    // From Vec<f32>
    let pv: ParameterValue = vec![0.1f32, 0.2f32].into();
    assert!(matches!(pv, ParameterValue::Vector(_)));

    // From serde_json::Value
    let pv: ParameterValue = serde_json::json!({"key": "value"}).into();
    assert!(matches!(pv, ParameterValue::Json(_)));
}

/// Test to_sql_string for all parameter types
#[test]
fn test_parameter_value_to_sql_string() {
    assert_eq!(
        ParameterValue::String("test".to_string()).to_sql_string(),
        "'test'"
    );
    assert_eq!(ParameterValue::Int(123).to_sql_string(), "123");
    assert_eq!(ParameterValue::Float(45.67).to_sql_string(), "45.67");
    assert_eq!(ParameterValue::Bool(true).to_sql_string(), "true");
    assert_eq!(ParameterValue::Bool(false).to_sql_string(), "false");
    assert_eq!(ParameterValue::Null.to_sql_string(), "NULL");
    assert_eq!(
        ParameterValue::Vector(vec![1.0, 2.0, 3.0]).to_sql_string(),
        "'[1,2,3]'"
    );
}

// ================================
// Edge Cases and Error Handling
// ================================

/// Test preparing empty SQL
#[test]
fn test_prepare_empty_sql() {
    let cache = PreparedStatementCache::with_defaults();

    // Empty SQL should still be parseable (though might fail at parse stage)
    // The actual behavior depends on the parser
    let result = cache.prepare("");

    // Either it succeeds with 0 params or fails during parsing
    match result {
        Ok(id) => {
            let stmt = cache.get(&id).unwrap();
            assert_eq!(stmt.parameter_count(), 0);
        }
        Err(PreparedStatementError::ParseError(_)) => {
            // This is also acceptable
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
    }
}

/// Test preparing very long SQL
#[test]
fn test_prepare_long_sql() {
    let cache = PreparedStatementCache::with_defaults();

    // Generate a long SQL with many parameters
    let mut sql = "SELECT * FROM t WHERE ".to_string();
    let mut conditions: Vec<String> = Vec::new();
    for i in 1..=50 {
        conditions.push(format!("col{} = ${}", i, i));
    }
    sql.push_str(&conditions.join(" AND "));

    let result = cache.prepare(&sql);
    assert!(result.is_ok());

    let id = result.unwrap();
    let stmt = cache.get(&id).unwrap();
    assert_eq!(stmt.parameter_count(), 50);
}

/// Test ID uniqueness
#[test]
fn test_statement_id_uniqueness() {
    let cache = PreparedStatementCache::with_defaults();
    let mut ids: std::collections::HashSet<PreparedStatementId> = std::collections::HashSet::new();

    // Prepare many statements and verify all IDs are unique
    for i in 0..100 {
        let id = cache.prepare(&format!("SELECT {}", i)).unwrap();
        assert!(ids.insert(id.clone()), "ID {} should be unique", id);
    }

    assert_eq!(ids.len(), 100);
}

/// Test default configuration values
#[test]
fn test_default_config() {
    let config = PreparedStatementConfig::default();

    assert_eq!(config.max_statements, 1000);
    assert_eq!(config.default_ttl, Duration::from_secs(3600));
    assert!(config.enable_cleanup);
    assert_eq!(config.cleanup_interval, Duration::from_secs(300));
}

/// Test cache is_empty method
#[test]
fn test_cache_is_empty() {
    let cache = PreparedStatementCache::with_defaults();

    assert!(cache.is_empty());

    cache.prepare("SELECT 1").unwrap();
    assert!(!cache.is_empty());

    cache.clear();
    assert!(cache.is_empty());
}
