//! Cross-surface unification: a write on ONE protocol must be visible to every surface.
//!
//! Proves the convergence fix — REST/gRPC `DmlService` and the pgwire direct-write path now
//! share ONE `DirectWalTableRecordStore` (`SharedServices.canonical_record_store`) instead of
//! divergent instances (the old vector-compatibility stub on REST/gRPC vs the WAL store on
//! pgwire). Here we INSERT over pgwire, then read the change-feed off the SHARED store the
//! REST/gRPC `DmlService` is built on, and see the row.
//!
//!   cargo test --test cross_surface_unification_e2e -- --nocapture

use std::time::Duration;

use proximadb::services::record_store::TableRecordStore;
use tokio::time::sleep;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_write_is_visible_on_the_shared_canonical_store() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // Write over pgwire.
    let _ = client.simple_query("DROP TABLE IF EXISTS uacct").await;
    client
        .simple_query("CREATE TABLE uacct (id INT PRIMARY KEY, bal INT)")
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO uacct (id, bal) VALUES (1, 100), (2, 200)")
        .await
        .expect("insert");
    sleep(Duration::from_millis(200)).await;

    // Read the change-feed off the SHARED canonical store — the SAME instance the REST/gRPC
    // DmlService is built on (SharedServices.canonical_record_store). If the surfaces were
    // still divergent (pgwire WAL store vs REST/gRPC vector-compat stub), this would be empty.
    let store = server
        .db()
        .canonical_record_store()
        .expect("shared canonical record store must exist when config is provided");

    let changes = store
        .read_changes_since("uacct", 0)
        .await
        .expect("read changes");

    assert_eq!(
        changes.len(),
        2,
        "pgwire INSERT of 2 rows must be visible on the shared store the REST/gRPC \
         DmlService reads from; got {changes:?}"
    );
    assert!(
        changes
            .iter()
            .all(|c| c.op == "upsert" && c.collection == "uacct")
    );
    let keys: Vec<&str> = changes.iter().map(|c| c.key.as_str()).collect();
    assert!(keys.contains(&"1") && keys.contains(&"2"), "keys: {keys:?}");
    eprintln!("✓ cross-surface unified: pgwire write visible on the shared REST/gRPC store");
}
