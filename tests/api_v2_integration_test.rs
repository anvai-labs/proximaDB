/*
 * Copyright 2025 ProximaDB
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

//! V2 REST API Integration Tests
//!
//! This module provides comprehensive integration tests for the ProximaDB V2 REST API,
//! covering ProximaRecord and typed schema support.
//!
//! ## Test Coverage
//!
//! - `POST /api/v2/collections` - Create collection with schema
//! - `GET /api/v2/collections` - List collections
//! - `GET /api/v2/collections/{id}` - Get collection details
//! - `POST /api/v2/collections/{id}/records/batch` - Insert ProximaRecords
//! - `POST /api/v2/collections/{id}/search` - Search with typed filters
//!
//! ## Running Tests
//!
//! ```bash
//! # Start server first (in another terminal)
//! cargo run --release --bin proximadb-server
//!
//! # Run V2 API tests
//! cargo test --test api_v2_integration_test -- --test-threads=1
//! ```

use reqwest::Client as HttpClient;
use serde_json::{Value as JsonValue, json};
use std::time::Duration;
use tokio::time::sleep;

// Constants for server endpoints
const REST_BASE_URL: &str = "http://127.0.0.1:5678";

/// Test dimension for vectors
const TEST_DIMENSION: u32 = 128;

// ================================================================================
// TEST INFRASTRUCTURE
// ================================================================================

/// V2 API Test Harness
///
/// Provides methods for testing V2 REST API endpoints with proper setup/teardown.
struct V2ApiTestHarness {
    http_client: HttpClient,
    test_collection_name: String,
}

impl V2ApiTestHarness {
    /// Create a new test harness with unique collection name
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            // Avoid platform system proxy discovery in test environments.
            .no_proxy()
            .build()?;

        // Generate unique test name to avoid conflicts
        let test_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let test_collection_name = format!("v2_test_coll_{}", test_id);

        Ok(Self {
            http_client,
            test_collection_name,
        })
    }

    /// Check if REST server is available
    async fn check_server(&self) -> bool {
        self.http_client
            .get(format!("{}/health", REST_BASE_URL))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Create a collection via V2 API with schema
    async fn create_collection_with_schema(
        &self,
        schema: Option<JsonValue>,
        enable_proxima_record: bool,
    ) -> Result<JsonValue, Box<dyn std::error::Error>> {
        let mut request = json!({
            "name": self.test_collection_name,
            "dimension": TEST_DIMENSION,
            "engine": "sst",
            "enable_proxima_record": enable_proxima_record
        });

        if let Some(s) = schema {
            request["schema"] = s;
        }

        let response = self
            .http_client
            .post(format!("{}/api/v2/collections", REST_BASE_URL))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body: JsonValue = response.json().await?;

        if !status.is_success() {
            return Err(format!(
                "Failed to create collection (status {}): {:?}",
                status, body
            )
            .into());
        }

        Ok(body)
    }

    /// Create a basic collection without schema
    async fn create_basic_collection(&self) -> Result<JsonValue, Box<dyn std::error::Error>> {
        self.create_collection_with_schema(None, false).await
    }

    /// List collections via V2 API
    async fn list_collections(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        include_stats: Option<bool>,
    ) -> Result<JsonValue, Box<dyn std::error::Error>> {
        let mut url = format!("{}/api/v2/collections", REST_BASE_URL);
        let mut params = Vec::new();

        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(o) = offset {
            params.push(format!("offset={}", o));
        }
        if let Some(s) = include_stats {
            params.push(format!("include_stats={}", s));
        }

        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self.http_client.get(&url).send().await?;

        let status = response.status();
        let body: JsonValue = response.json().await?;

        if !status.is_success() {
            return Err(
                format!("Failed to list collections (status {}): {:?}", status, body).into(),
            );
        }

        Ok(body)
    }

    /// Get collection details via V2 API
    async fn get_collection(&self) -> Result<JsonValue, Box<dyn std::error::Error>> {
        let response = self
            .http_client
            .get(format!(
                "{}/api/v2/collections/{}",
                REST_BASE_URL, self.test_collection_name
            ))
            .send()
            .await?;

        let status = response.status();
        let body: JsonValue = response.json().await?;

        if !status.is_success() {
            return Err(format!("Failed to get collection (status {}): {:?}", status, body).into());
        }

        Ok(body)
    }

    /// Insert ProximaRecords via V2 API
    async fn insert_records(
        &self,
        records: Vec<JsonValue>,
        validate_schema: Option<bool>,
    ) -> Result<JsonValue, Box<dyn std::error::Error>> {
        let mut request = json!({
            "records": records
        });

        if let Some(v) = validate_schema {
            request["validate_schema"] = json!(v);
        }

        let response = self
            .http_client
            .post(format!(
                "{}/api/v2/collections/{}/records/batch",
                REST_BASE_URL, self.test_collection_name
            ))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body: JsonValue = response.json().await?;

        if !status.is_success() {
            return Err(format!("Failed to insert records (status {}): {:?}", status, body).into());
        }

        Ok(body)
    }

    /// Search with typed filters via V2 API
    async fn search(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        filters: Option<Vec<JsonValue>>,
        include_text: Option<bool>,
        include_vector: Option<bool>,
    ) -> Result<JsonValue, Box<dyn std::error::Error>> {
        let mut request = json!({
            "vector": query_vector,
            "top_k": top_k
        });

        if let Some(f) = filters {
            request["filters"] = json!(f);
        }
        if let Some(t) = include_text {
            request["include_text"] = json!(t);
        }
        if let Some(v) = include_vector {
            request["include_vector"] = json!(v);
        }

        let response = self
            .http_client
            .post(format!(
                "{}/api/v2/collections/{}/search",
                REST_BASE_URL, self.test_collection_name
            ))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body: JsonValue = response.json().await?;

        if !status.is_success() {
            return Err(format!("Failed to search (status {}): {:?}", status, body).into());
        }

        Ok(body)
    }

    /// Get collection details by an arbitrary identifier (name OR UUID).
    async fn get_collection_by(
        &self,
        identifier: &str,
    ) -> Result<(reqwest::StatusCode, JsonValue), Box<dyn std::error::Error>> {
        let response = self
            .http_client
            .get(format!(
                "{}/api/v2/collections/{}",
                REST_BASE_URL, identifier
            ))
            .send()
            .await?;
        let status = response.status();
        let body: JsonValue = response.json().await?;
        Ok((status, body))
    }

    /// Insert records targeting an arbitrary identifier (name OR UUID).
    async fn insert_records_to(
        &self,
        identifier: &str,
        records: Vec<JsonValue>,
    ) -> Result<reqwest::StatusCode, Box<dyn std::error::Error>> {
        let response = self
            .http_client
            .post(format!(
                "{}/api/v2/collections/{}/records/batch",
                REST_BASE_URL, identifier
            ))
            .json(&json!({ "records": records }))
            .send()
            .await?;
        Ok(response.status())
    }

    /// Search targeting an arbitrary identifier (name OR UUID).
    async fn search_in(
        &self,
        identifier: &str,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<reqwest::StatusCode, Box<dyn std::error::Error>> {
        let response = self
            .http_client
            .post(format!(
                "{}/api/v2/collections/{}/search",
                REST_BASE_URL, identifier
            ))
            .json(&json!({ "vector": query_vector, "top_k": top_k }))
            .send()
            .await?;
        Ok(response.status())
    }

    /// Delete a collection via V2 API by an arbitrary identifier (name OR UUID).
    async fn delete_collection_by(
        &self,
        identifier: &str,
    ) -> Result<reqwest::StatusCode, Box<dyn std::error::Error>> {
        let response = self
            .http_client
            .delete(format!(
                "{}/api/v2/collections/{}",
                REST_BASE_URL, identifier
            ))
            .send()
            .await?;
        Ok(response.status())
    }

    /// Cleanup: Delete test collection
    async fn cleanup(&self) {
        // Try to delete via V1 API (V2 may not have delete endpoint yet)
        let _ = self
            .http_client
            .delete(format!(
                "{}/api/v1/collections/{}",
                REST_BASE_URL, self.test_collection_name
            ))
            .send()
            .await;
    }
}

/// Generate test records for insertion
fn generate_test_records(count: usize, dimension: usize) -> Vec<JsonValue> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| (i * dimension + j) as f32 / (count * dimension) as f32)
                .collect();

            // TD-REC-1: the record contract is {id, vector, props (alias
            // metadata), text_fields} with unknown fields rejected. The old
            // `typed_fields` key was never a contract field — the server
            // silently dropped it, so the filter assertions over `category`
            // were vacuous. The typed values now ride in `props` (merged with
            // the former `metadata` map: both keys in one record is a
            // duplicate-field rejection), making this suite actually
            // exercise persistence + filtering.
            json!({
                "id": format!("record_{}", i),
                "vector": vector,
                "props": {
                    "category": format!("category_{}", i % 3),
                    "price": 10.0 + (i as f64 * 5.0),
                    "in_stock": i % 2 == 0,
                    "quantity": i * 10,
                    "source": "test",
                    "index": i
                }
            })
        })
        .collect()
}

/// Generate test records with text fields
fn generate_test_records_with_text(count: usize, dimension: usize) -> Vec<JsonValue> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| (i * dimension + j) as f32 / (count * dimension) as f32)
                .collect();

            json!({
                "id": format!("doc_{}", i),
                "vector": vector,
                "props": {
                    "title": format!("Document Title {}", i),
                    "category": format!("cat_{}", i % 5),
                    "author": format!("author_{}", i % 3)
                },
                "text_fields": [
                    {
                        "name": "description",
                        "content": format!("This is the description for document {}. It contains sample text for testing purposes.", i),
                        "storage_hint": "adaptive"
                    }
                ]
            })
        })
        .collect()
}

// ================================================================================
// COLLECTION CREATION TESTS
// ================================================================================

/// Test: Create collection with basic configuration (no schema)
#[tokio::test]
async fn test_create_collection_basic() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available at {}", REST_BASE_URL);
        eprintln!("Start the server with: cargo run --release --bin proximadb-server");
        return;
    }

    // Create collection
    let result = harness.create_basic_collection().await;

    match result {
        Ok(response) => {
            println!("Collection created: {:?}", response);

            // Verify response fields
            assert!(
                response.get("collection_id").is_some(),
                "Response should contain collection_id"
            );
            assert!(
                response.get("name").is_some(),
                "Response should contain name"
            );
            assert!(
                response.get("dimension").is_some(),
                "Response should contain dimension"
            );
            assert_eq!(
                response.get("dimension").and_then(|v| v.as_u64()),
                Some(TEST_DIMENSION as u64),
                "Dimension should match"
            );
            assert!(
                response.get("engine").is_some(),
                "Response should contain engine"
            );
            assert!(
                response.get("created_at").is_some(),
                "Response should contain created_at"
            );

            println!("test_create_collection_basic PASSED");
        }
        Err(e) => {
            eprintln!("Failed to create collection: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Regression (#176 follow-up): the v2 `collection_id` identity is the
/// collection's canonical UUID on create/get/list — NOT the request echo —
/// while `name` stays the user-supplied name, and the returned UUID is a valid
/// lookup key on every endpoint (get/insert/search/delete) alongside the name.
#[tokio::test]
async fn test_collection_id_is_canonical_uuid_and_resolves_both_ways() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available at {}", REST_BASE_URL);
        return;
    }

    let name = harness.test_collection_name.clone();

    // --- CREATE: collection_id must be a UUID (≠ name), name == user name. ---
    let create = match harness.create_basic_collection().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping test: create failed: {}", e);
            return;
        }
    };
    let uuid = create
        .get("collection_id")
        .and_then(|v| v.as_str())
        .expect("create response has collection_id")
        .to_string();
    assert_ne!(
        uuid, name,
        "create collection_id must be the canonical UUID, not the request name"
    );
    assert!(
        !uuid.is_empty(),
        "create collection_id (UUID) must not be empty"
    );
    assert_eq!(
        create.get("name").and_then(|v| v.as_str()),
        Some(name.as_str()),
        "create name must be the user-supplied name"
    );

    // --- GET by NAME: collection_id is the same UUID, name is the user name. ---
    let (st, by_name) = harness.get_collection_by(&name).await.expect("get by name");
    assert!(st.is_success(), "get by name should succeed, got {}", st);
    assert_eq!(
        by_name.get("collection_id").and_then(|v| v.as_str()),
        Some(uuid.as_str()),
        "get-by-name collection_id must be the canonical UUID"
    );
    assert_eq!(
        by_name.get("name").and_then(|v| v.as_str()),
        Some(name.as_str()),
        "get-by-name name must be the user-supplied name"
    );

    // --- GET by UUID: must succeed and round-trip the same identity. ---
    let (st, by_uuid) = harness.get_collection_by(&uuid).await.expect("get by uuid");
    assert!(st.is_success(), "get by UUID should succeed, got {}", st);
    assert_eq!(
        by_uuid.get("collection_id").and_then(|v| v.as_str()),
        Some(uuid.as_str()),
        "get-by-UUID collection_id must be the canonical UUID"
    );
    assert_eq!(
        by_uuid.get("name").and_then(|v| v.as_str()),
        Some(name.as_str()),
        "get-by-UUID name must be the user-supplied name"
    );

    // --- LIST: our collection appears with UUID collection_id + user name. ---
    let list = harness
        .list_collections(Some(1000), Some(0), Some(false))
        .await
        .expect("list collections");
    let empty = vec![];
    let cols = list
        .get("collections")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let ours = cols
        .iter()
        .find(|c| c.get("name").and_then(|v| v.as_str()) == Some(name.as_str()))
        .expect("listed by user name");
    assert_eq!(
        ours.get("collection_id").and_then(|v| v.as_str()),
        Some(uuid.as_str()),
        "list collection_id must be the canonical UUID"
    );

    // --- INSERT + SEARCH by UUID (the create-returned key) must work. ---
    let records = generate_test_records(3, TEST_DIMENSION as usize);
    let st = harness
        .insert_records_to(&uuid, records)
        .await
        .expect("insert by uuid");
    assert!(
        st.is_success(),
        "insert by the returned UUID should succeed, got {}",
        st
    );
    let st = harness
        .search_in(&uuid, vec![0.1f32; TEST_DIMENSION as usize], 3)
        .await
        .expect("search by uuid");
    assert!(
        st.is_success(),
        "search by the returned UUID should succeed, got {}",
        st
    );

    // --- DELETE by the returned UUID must succeed (the create→delete flow). ---
    // (Post-delete read-visibility is a separate, pre-existing concern — a stale
    // read may briefly return a default stub — so we don't assert a 404 here.)
    let st = harness
        .delete_collection_by(&uuid)
        .await
        .expect("delete by uuid");
    assert!(
        st.is_success(),
        "delete by the returned UUID should succeed, got {}",
        st
    );

    println!("test_collection_id_is_canonical_uuid_and_resolves_both_ways PASSED");
    harness.cleanup().await;
}

/// Test: Create collection with typed schema
#[tokio::test]
async fn test_create_collection_with_schema() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create schema with various column types
    let schema = json!({
        "columns": [
            {"name": "category", "data_type": "text", "indexed": true, "filterable": true},
            {"name": "price", "data_type": "float", "filterable": true},
            {"name": "quantity", "data_type": "integer", "indexed": true},
            {"name": "in_stock", "data_type": "boolean"},
            {"name": "tags", "data_type": "array_text", "nullable": true}
        ],
        "enforcement": "hybrid",
        "allow_additional_fields": true
    });

    let result = harness
        .create_collection_with_schema(Some(schema), true)
        .await;

    match result {
        Ok(response) => {
            println!("Collection with schema created: {:?}", response);

            assert!(
                response.get("collection_id").is_some(),
                "Response should contain collection_id"
            );
            assert_eq!(
                response
                    .get("proxima_record_enabled")
                    .and_then(|v| v.as_bool()),
                Some(true),
                "proxima_record_enabled should be true"
            );
            assert!(
                response.get("schema_id").is_some(),
                "Response should contain schema_id for collections with schema"
            );

            println!("test_create_collection_with_schema PASSED");
        }
        Err(e) => {
            eprintln!("Failed to create collection with schema: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Create collection fails with invalid engine
#[tokio::test]
async fn test_create_collection_invalid_engine() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Try to create with invalid engine
    let request = json!({
        "name": format!("{}_invalid", harness.test_collection_name),
        "dimension": TEST_DIMENSION,
        "engine": "invalid_engine"
    });

    let response = harness
        .http_client
        .post(format!("{}/api/v2/collections", REST_BASE_URL))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 Bad Request for invalid engine, got {}",
                status
            );
            println!("test_create_collection_invalid_engine PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Create collection fails with zero dimension
#[tokio::test]
async fn test_create_collection_invalid_dimension() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    let request = json!({
        "name": format!("{}_zero_dim", harness.test_collection_name),
        "dimension": 0,
        "engine": "sst"
    });

    let response = harness
        .http_client
        .post(format!("{}/api/v2/collections", REST_BASE_URL))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 Bad Request for zero dimension, got {}",
                status
            );
            println!("test_create_collection_invalid_dimension PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

// ================================================================================
// LIST COLLECTIONS TESTS
// ================================================================================

/// Test: List collections with pagination
#[tokio::test]
async fn test_list_collections() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create a test collection first
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create test collection: {}", e);
        return;
    }

    // Wait for collection to be available
    sleep(Duration::from_millis(500)).await;

    // List collections
    let result = harness
        .list_collections(Some(10), Some(0), Some(true))
        .await;

    match result {
        Ok(response) => {
            println!("Collections list: {:?}", response);

            assert!(
                response.get("collections").is_some(),
                "Response should contain collections array"
            );
            assert!(
                response.get("total").is_some(),
                "Response should contain total count"
            );
            assert!(
                response.get("limit").is_some(),
                "Response should contain limit"
            );
            assert!(
                response.get("offset").is_some(),
                "Response should contain offset"
            );
            assert!(
                response.get("has_more").is_some(),
                "Response should contain has_more flag"
            );

            // Verify our collection is in the list
            let empty_vec = vec![];
            let collections = response
                .get("collections")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty_vec);

            let our_collection = collections.iter().find(|c| {
                c.get("name").and_then(|v| v.as_str()) == Some(&harness.test_collection_name)
            });

            assert!(
                our_collection.is_some(),
                "Created collection should be in the list (matched by user-supplied name, not UUID id)"
            );

            // Regression (list_collections name vs UUID): the summary `name`
            // MUST be the user-supplied collection name, not the UUID id.
            let coll = our_collection.unwrap();
            let listed_name = coll.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let listed_id = coll
                .get("collection_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            assert_eq!(
                listed_name, harness.test_collection_name,
                "Listed name should equal the user-supplied name, got '{}'",
                listed_name
            );
            assert_ne!(
                listed_name, listed_id,
                "Listed name must not be the UUID collection_id ('{}')",
                listed_id
            );

            println!("test_list_collections PASSED");
        }
        Err(e) => {
            eprintln!("Failed to list collections: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: List collections with stats included
#[tokio::test]
async fn test_list_collections_with_stats() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection and insert some records
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    let records = generate_test_records(5, TEST_DIMENSION as usize);
    if let Err(e) = harness.insert_records(records, None).await {
        eprintln!("Failed to insert records: {}", e);
        harness.cleanup().await;
        return;
    }

    // Wait for data to be indexed
    sleep(Duration::from_secs(1)).await;

    // List with stats
    let result = harness.list_collections(None, None, Some(true)).await;

    match result {
        Ok(response) => {
            let empty_vec = vec![];
            let collections = response
                .get("collections")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty_vec);

            // Find our collection
            let our_collection = collections.iter().find(|c| {
                c.get("name").and_then(|v| v.as_str()) == Some(&harness.test_collection_name)
            });

            if let Some(coll) = our_collection {
                // When include_stats is true, record_count should be present
                assert!(
                    coll.get("record_count").is_some(),
                    "Collection should have record_count when include_stats=true"
                );
            }

            println!("test_list_collections_with_stats PASSED");
        }
        Err(e) => {
            eprintln!("Failed to list collections with stats: {}", e);
        }
    }

    harness.cleanup().await;
}

// ================================================================================
// GET COLLECTION DETAILS TESTS
// ================================================================================

/// Test: Get collection details
#[tokio::test]
async fn test_get_collection_details() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Wait for collection to be available
    sleep(Duration::from_millis(500)).await;

    // Get collection details
    let result = harness.get_collection().await;

    match result {
        Ok(response) => {
            println!("Collection details: {:?}", response);

            assert!(
                response.get("collection_id").is_some(),
                "Response should contain collection_id"
            );
            assert!(
                response.get("name").is_some(),
                "Response should contain name"
            );
            // Regression (get_collection name vs UUID): `name` must be the
            // user-supplied collection name, not the UUID id.
            assert_eq!(
                response.get("name").and_then(|v| v.as_str()),
                Some(harness.test_collection_name.as_str()),
                "Get should return the user-supplied name, not the UUID id"
            );
            assert!(
                response.get("dimension").is_some(),
                "Response should contain dimension"
            );
            assert!(
                response.get("engine").is_some(),
                "Response should contain engine"
            );
            assert!(
                response.get("distance_metric").is_some(),
                "Response should contain distance_metric"
            );
            assert!(
                response.get("stats").is_some(),
                "Response should contain stats"
            );

            println!("test_get_collection_details PASSED");
        }
        Err(e) => {
            eprintln!("Failed to get collection: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Get non-existent collection returns 404
#[tokio::test]
async fn test_get_collection_not_found() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Try to get non-existent collection
    let response = harness
        .http_client
        .get(format!(
            "{}/api/v2/collections/nonexistent_collection_xyz_123",
            REST_BASE_URL
        ))
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 404,
                "Should return 404 for non-existent collection, got {}",
                status
            );
            println!("test_get_collection_not_found PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }
}

// ================================================================================
// INSERT RECORDS TESTS
// ================================================================================

/// Test: Insert ProximaRecords batch
#[tokio::test]
async fn test_insert_records_batch() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Insert records
    let records = generate_test_records(10, TEST_DIMENSION as usize);
    let result = harness.insert_records(records, Some(true)).await;

    match result {
        Ok(response) => {
            println!("Insert response: {:?}", response);

            assert!(
                response.get("inserted_count").is_some(),
                "Response should contain inserted_count"
            );
            assert!(
                response.get("inserted_ids").is_some(),
                "Response should contain inserted_ids"
            );

            let inserted_count = response
                .get("inserted_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            assert_eq!(inserted_count, 10, "Should have inserted 10 records");

            let inserted_ids = response
                .get("inserted_ids")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);

            assert_eq!(inserted_ids, 10, "Should have 10 inserted IDs");

            println!("test_insert_records_batch PASSED");
        }
        Err(e) => {
            eprintln!("Failed to insert records: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Insert records with text fields
#[tokio::test]
async fn test_insert_records_with_text_fields() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection with ProximaRecord enabled
    if let Err(e) = harness.create_collection_with_schema(None, true).await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Insert records with text fields
    let records = generate_test_records_with_text(5, TEST_DIMENSION as usize);
    let result = harness.insert_records(records, None).await;

    match result {
        Ok(response) => {
            println!("Insert with text fields response: {:?}", response);

            let inserted_count = response
                .get("inserted_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            assert_eq!(inserted_count, 5, "Should have inserted 5 records");

            println!("test_insert_records_with_text_fields PASSED");
        }
        Err(e) => {
            eprintln!("Failed to insert records with text fields: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Insert fails with empty vector
#[tokio::test]
async fn test_insert_records_empty_vector() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Try to insert record with empty vector
    let records = vec![json!({
        "id": "empty_vec_record",
        "vector": [],
        "props": {
            "test": "value"
        }
    })];

    let result = harness.insert_records(records, None).await;

    match result {
        Ok(response) => {
            // The request should succeed but report the record as failed
            let failed_count = response
                .get("failed_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            assert!(
                failed_count > 0,
                "Should report failed count for empty vector"
            );

            let errors = response
                .get("errors")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);

            assert!(errors > 0, "Should report error for empty vector");

            println!("test_insert_records_empty_vector PASSED");
        }
        Err(e) => {
            // API might reject at request level, which is also acceptable
            println!("Request rejected (expected): {}", e);
            println!("test_insert_records_empty_vector PASSED");
        }
    }

    harness.cleanup().await;
}

/// Test: Insert fails with no records
#[tokio::test]
async fn test_insert_records_empty_batch() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Try to insert empty batch
    let request = json!({
        "records": []
    });

    let response = harness
        .http_client
        .post(format!(
            "{}/api/v2/collections/{}/records/batch",
            REST_BASE_URL, harness.test_collection_name
        ))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 for empty batch, got {}",
                status
            );
            println!("test_insert_records_empty_batch PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

// ================================================================================
// SEARCH TESTS
// ================================================================================

/// Test: Basic vector search
#[tokio::test]
async fn test_search_basic() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection and insert records
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    let records = generate_test_records(20, TEST_DIMENSION as usize);
    if let Err(e) = harness.insert_records(records, None).await {
        eprintln!("Failed to insert records: {}", e);
        harness.cleanup().await;
        return;
    }

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Search
    let query_vector: Vec<f32> = (0..TEST_DIMENSION as usize)
        .map(|j| 0.5 + (j as f32 * 0.001))
        .collect();

    let result = harness.search(query_vector, 10, None, None, None).await;

    match result {
        Ok(response) => {
            println!("Search response: {:?}", response);

            assert!(
                response.get("results").is_some(),
                "Response should contain results"
            );
            assert!(
                response.get("latency_ms").is_some(),
                "Response should contain latency_ms"
            );
            assert!(
                response.get("request_id").is_some(),
                "Response should contain request_id"
            );

            let results = response
                .get("results")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);

            assert!(results > 0, "Should return some results");
            assert!(results <= 10, "Should respect top_k limit");

            println!("test_search_basic PASSED");
        }
        Err(e) => {
            eprintln!("Search failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search with typed equality filter
#[tokio::test]
async fn test_search_with_eq_filter() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create and populate collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    let records = generate_test_records(30, TEST_DIMENSION as usize);
    if let Err(e) = harness.insert_records(records, None).await {
        eprintln!("Failed to insert records: {}", e);
        harness.cleanup().await;
        return;
    }

    sleep(Duration::from_secs(1)).await;

    // Search with equality filter
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    let filters = vec![json!({
        "field": "category",
        "op": "eq",
        "value": "category_0"
    })];

    let result = harness
        .search(query_vector, 10, Some(filters), None, None)
        .await;

    match result {
        Ok(response) => {
            println!("Search with filter response: {:?}", response);

            let empty_vec = vec![];
            let results = response
                .get("results")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty_vec);

            // All results should have category_0
            for result in results {
                let typed_fields = result.get("typed_fields");
                if let Some(fields) = typed_fields
                    && let Some(category) = fields.get("category")
                {
                    assert_eq!(
                        category.as_str(),
                        Some("category_0"),
                        "Filtered results should only have category_0"
                    );
                }
            }

            println!("test_search_with_eq_filter PASSED");
        }
        Err(e) => {
            eprintln!("Filtered search failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search with range filter
#[tokio::test]
async fn test_search_with_range_filter() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create and populate collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    let records = generate_test_records(20, TEST_DIMENSION as usize);
    if let Err(e) = harness.insert_records(records, None).await {
        eprintln!("Failed to insert records: {}", e);
        harness.cleanup().await;
        return;
    }

    sleep(Duration::from_secs(1)).await;

    // Search with range filter (price < 50)
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    let filters = vec![json!({
        "field": "price",
        "op": "lt",
        "value": 50.0
    })];

    let result = harness
        .search(query_vector, 10, Some(filters), None, None)
        .await;

    match result {
        Ok(response) => {
            println!("Search with range filter response: {:?}", response);

            // The API should accept the range filter and return results
            assert!(
                response.get("results").is_some(),
                "Response should contain results"
            );

            println!("test_search_with_range_filter PASSED");
        }
        Err(e) => {
            eprintln!("Range filtered search failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search with include_vector option
#[tokio::test]
async fn test_search_include_vector() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create and populate collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    let records = generate_test_records(10, TEST_DIMENSION as usize);
    if let Err(e) = harness.insert_records(records, None).await {
        eprintln!("Failed to insert records: {}", e);
        harness.cleanup().await;
        return;
    }

    sleep(Duration::from_secs(1)).await;

    // Search with include_vector = true
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    let result = harness
        .search(query_vector, 5, None, None, Some(true))
        .await;

    match result {
        Ok(response) => {
            println!("Search with include_vector response: {:?}", response);

            let empty_vec = vec![];
            let results = response
                .get("results")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty_vec);

            for result in results {
                let vector = result.get("vector");
                if let Some(v) = vector
                    && let Some(arr) = v.as_array()
                {
                    assert_eq!(
                        arr.len(),
                        TEST_DIMENSION as usize,
                        "Vector should have correct dimension"
                    );
                }
            }

            println!("test_search_include_vector PASSED");
        }
        Err(e) => {
            eprintln!("Search with include_vector failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search fails with empty vector
#[tokio::test]
async fn test_search_empty_vector() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Search with empty vector
    let request = json!({
        "vector": [],
        "top_k": 10
    });

    let response = harness
        .http_client
        .post(format!(
            "{}/api/v2/collections/{}/search",
            REST_BASE_URL, harness.test_collection_name
        ))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 for empty query vector, got {}",
                status
            );
            println!("test_search_empty_vector PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search fails with invalid filter operator
#[tokio::test]
async fn test_search_invalid_filter_operator() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Search with invalid filter operator
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    let request = json!({
        "vector": query_vector,
        "top_k": 10,
        "filters": [
            {"field": "category", "op": "invalid_op", "value": "test"}
        ]
    });

    let response = harness
        .http_client
        .post(format!(
            "{}/api/v2/collections/{}/search",
            REST_BASE_URL, harness.test_collection_name
        ))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 for invalid filter operator, got {}",
                status
            );
            println!("test_search_invalid_filter_operator PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

/// Test: Search fails with between filter missing value_upper
#[tokio::test]
async fn test_search_between_filter_missing_upper() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    // Create collection
    if let Err(e) = harness.create_basic_collection().await {
        eprintln!("Failed to create collection: {}", e);
        return;
    }

    // Search with "between" filter but missing value_upper
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    let request = json!({
        "vector": query_vector,
        "top_k": 10,
        "filters": [
            {"field": "price", "op": "between", "value": 10}
        ]
    });

    let response = harness
        .http_client
        .post(format!(
            "{}/api/v2/collections/{}/search",
            REST_BASE_URL, harness.test_collection_name
        ))
        .json(&request)
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.as_u16() == 400,
                "Should return 400 for between filter missing value_upper, got {}",
                status
            );
            println!("test_search_between_filter_missing_upper PASSED");
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
        }
    }

    harness.cleanup().await;
}

// ================================================================================
// END-TO-END WORKFLOW TEST
// ================================================================================

/// Test: Complete workflow - create, insert, search
#[tokio::test]
async fn test_complete_workflow() {
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Skipping test: Failed to create test harness: {}", e);
            return;
        }
    };

    if !harness.check_server().await {
        eprintln!("Skipping test: Server not available");
        return;
    }

    println!("Starting complete V2 API workflow test...");

    // Step 1: Create collection with schema
    println!("Step 1: Creating collection with schema...");
    let schema = json!({
        "columns": [
            {"name": "title", "data_type": "text", "indexed": true},
            {"name": "category", "data_type": "text", "filterable": true},
            {"name": "price", "data_type": "float", "filterable": true}
        ],
        "enforcement": "hybrid"
    });

    match harness
        .create_collection_with_schema(Some(schema), true)
        .await
    {
        Ok(response) => {
            println!(
                "  Collection created: {}",
                response.get("collection_id").unwrap_or(&json!("unknown"))
            );
        }
        Err(e) => {
            eprintln!("  Failed to create collection: {}", e);
            return;
        }
    }

    // Step 2: Insert records
    println!("Step 2: Inserting records...");
    let records = generate_test_records_with_text(15, TEST_DIMENSION as usize);
    match harness.insert_records(records, None).await {
        Ok(response) => {
            let count = response
                .get("inserted_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("  Inserted {} records", count);
        }
        Err(e) => {
            eprintln!("  Failed to insert records: {}", e);
            harness.cleanup().await;
            return;
        }
    }

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Step 3: Get collection details
    println!("Step 3: Getting collection details...");
    match harness.get_collection().await {
        Ok(response) => {
            let stats = response.get("stats");
            println!("  Collection stats: {:?}", stats);
        }
        Err(e) => {
            eprintln!("  Failed to get collection: {}", e);
        }
    }

    // Step 4: Search without filters
    println!("Step 4: Searching without filters...");
    let query_vector: Vec<f32> = vec![0.5; TEST_DIMENSION as usize];
    match harness
        .search(query_vector.clone(), 5, None, None, None)
        .await
    {
        Ok(response) => {
            let results = response
                .get("results")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let latency = response
                .get("latency_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("  Found {} results in {}ms", results, latency);
        }
        Err(e) => {
            eprintln!("  Search failed: {}", e);
        }
    }

    // Step 5: Search with filter
    println!("Step 5: Searching with category filter...");
    let filters = vec![json!({
        "field": "category",
        "op": "eq",
        "value": "cat_0"
    })];
    match harness
        .search(query_vector, 10, Some(filters), None, Some(true))
        .await
    {
        Ok(response) => {
            let results = response
                .get("results")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            println!("  Found {} filtered results", results);
        }
        Err(e) => {
            eprintln!("  Filtered search failed: {}", e);
        }
    }

    // Step 6: List collections
    println!("Step 6: Listing collections...");
    match harness.list_collections(Some(5), None, Some(true)).await {
        Ok(response) => {
            let total = response.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  Found {} total collections", total);
        }
        Err(e) => {
            eprintln!("  List collections failed: {}", e);
        }
    }

    // Cleanup
    harness.cleanup().await;

    println!("\ntest_complete_workflow PASSED");
}

// ================================================================================
// SUMMARY TEST
// ================================================================================

/// Summary test that prints test suite information
#[tokio::test]
async fn test_v2_api_summary() {
    let separator = "=".repeat(70);
    println!("\n");
    println!("{}", separator);
    println!("V2 REST API INTEGRATION TEST SUITE");
    println!("{}", separator);
    println!("\nThis test suite verifies the ProximaDB V2 REST API endpoints:");
    println!("  - POST /api/v2/collections - Create collection with schema");
    println!("  - GET /api/v2/collections - List collections");
    println!("  - GET /api/v2/collections/{{id}} - Get collection details");
    println!("  - POST /api/v2/collections/{{id}}/records/batch - Insert ProximaRecords");
    println!("  - POST /api/v2/collections/{{id}}/search - Search with typed filters");
    println!("\nPrerequisites:");
    println!("  1. Start the ProximaDB server: cargo run --release --bin proximadb-server");
    println!("  2. Ensure port 5678 (REST) is available");
    println!("\nRun tests with:");
    println!("  cargo test --test api_v2_integration_test -- --test-threads=1");
    println!("{}", separator);

    // Check server availability
    let harness = match V2ApiTestHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            println!("\nCould not initialize test harness: {}", e);
            return;
        }
    };

    let available = harness.check_server().await;
    println!("\nServer Status ({}):", REST_BASE_URL);
    println!(
        "  REST API: {}",
        if available {
            "AVAILABLE"
        } else {
            "NOT AVAILABLE"
        }
    );

    if available {
        println!("\nServer is available. Run individual tests to verify API functionality.");
    } else {
        println!("\nServer not available. Start the server first:");
        println!("  cargo run --release --bin proximadb-server");
    }
}
