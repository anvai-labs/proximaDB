//! TD-101 sub-slice e2e — `POST /api/v2/memory/ingest` over the real REST
//! transport. Boots an in-process ProximaDB with a free REST port, creates a
//! collection, and posts an agent turn.
//!
//! CI reality: extraction + consolidation need an LLM backend, which the test
//! deployment does not configure (no API key). So this test asserts the
//! WIRING + availability guard: the route is mounted and returns a structured
//! "LLM backend required" error (HTTP 501) rather than 404/panic. If an LLM
//! key is present in the environment, the happy path (HTTP 200 + applied
//! actions) is asserted instead. The deterministic correctness of the
//! orchestration + parsing + record-building is covered by the unit tests in
//! `src/services/agent_memory.rs`.
//!
//! One ProximaDB boot per process (global WAL manifest is a set-once singleton).

use serde_json::json;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

/// Does the environment carry any LLM provider key the engine could use?
fn llm_configured() -> bool {
    ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"]
        .iter()
        .any(|k| std::env::var(k).map(|v| !v.is_empty()).unwrap_or(false))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_ingest_route_is_mounted_and_guarded() {
    let server = TestPgServer::start().await.expect("server start");
    let (http, base) = server.rest_client().expect("client");

    let collection = server.unique_name("mem_ingest");
    let _ = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({"name": collection, "dimension": 384, "engine": "sst"}))
        .send()
        .await;

    let resp = http
        .post(format!("{base}/api/v2/memory/ingest"))
        .json(&json!({
            "collection": collection,
            "tenant_id": "default",
            "actor": "test-agent",
            "session_id": "sess-e2e",
            "user": "I prefer dark mode and live in NYC.",
            "assistant": "Noted — dark mode on, based in New York."
        }))
        .send()
        .await
        .expect("ingest request sent");

    let status = resp.status();
    // The route must exist (never 404) and never 5xx-crash the server.
    assert_ne!(status.as_u16(), 404, "route should be mounted");

    if llm_configured() {
        assert!(
            status.is_success(),
            "with an LLM backend, ingest should succeed; got {status}"
        );
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert!(
            body.get("applied").is_some(),
            "response carries applied actions"
        );
    } else {
        // No LLM backend in CI: the availability guard returns 501 Not Implemented.
        assert_eq!(
            status.as_u16(),
            501,
            "without an LLM backend the guard should return 501, got {status}"
        );
    }
}
