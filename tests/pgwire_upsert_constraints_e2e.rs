//! TD-DML-1: UPSERT (`INSERT … ON CONFLICT … DO UPDATE`) must enforce the SAME
//! relational integrity constraints as INSERT — over pgwire, the mandated
//! conformance surface.
//!
//! Before the fix `DmlService::execute_upsert` wrote its rows straight through
//! `write_mutations`, skipping the secondary-UNIQUE probe and the FOREIGN KEY
//! existence check, so a declared UNIQUE / FK could be violated by any UPSERT.
//! Every violation must be rejected with the PostgreSQL SQLSTATE a driver keys
//! on (23505 unique_violation / 23514 check_violation / 23503
//! foreign_key_violation) and must leave the table unchanged; a valid UPSERT
//! must still update the existing row.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

use std::time::Duration;

use tokio_postgres::{Client, SimpleQueryMessage};

/// All rows of `SELECT <cols> FROM <table> ORDER BY id`, each rendered as a
/// `|`-joined string (NULL as `∅`) — a whole-table fingerprint used to prove a
/// rejected statement left the table unchanged.
async fn snapshot(client: &Client, table: &str, cols: &[&str]) -> Vec<String> {
    let rows = client
        .simple_query(&format!(
            "SELECT {} FROM {table} ORDER BY id",
            cols.join(", ")
        ))
        .await
        .expect("snapshot SELECT");
    rows.iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some(
                (0..cols.len())
                    .map(|i| row.get(i).unwrap_or("∅").to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            _ => None,
        })
        .collect()
}

/// Run `sql`, require it to fail, and return `(sqlstate, message)` from the
/// server `DbError` (the pgwire client's own Display is a generic "db error").
async fn expect_db_error(client: &Client, sql: &str) -> (String, String) {
    let err = client
        .simple_query(sql)
        .await
        .err()
        .unwrap_or_else(|| panic!("statement must be rejected: {sql}"));
    let db = err
        .as_db_error()
        .unwrap_or_else(|| panic!("expected a server DbError for `{sql}`, got: {err:?}"));
    (db.code().code().to_string(), db.message().to_string())
}

fn suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// UNIQUE: an UPSERT that would give a NEW row an existing secondary-UNIQUE
/// value is rejected 23505 and writes nothing; an UPSERT that *replaces* a row
/// keeping its own unique value is not a self-conflict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_upsert_enforces_unique_constraint() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await;
    let t = format!("ups_uniq_{}", suffix());

    client
        .simple_query(&format!(
            "CREATE TABLE {t} (id BIGINT PRIMARY KEY, email VARCHAR, UNIQUE (email))"
        ))
        .await
        .expect("CREATE");
    client
        .simple_query(&format!(
            "INSERT INTO {t} (id, email) VALUES (1, 'a@x'), (2, 'b@x')"
        ))
        .await
        .expect("seed INSERT");
    let before = snapshot(&client, &t, &["id", "email"]).await;
    assert_eq!(before, vec!["1|a@x", "2|b@x"], "seed state");

    // New key, colliding secondary UNIQUE value → 23505, nothing written.
    let (code, msg) = expect_db_error(
        &client,
        &format!(
            "INSERT INTO {t} (id, email) VALUES (3, 'a@x') \
             ON CONFLICT (id) DO UPDATE SET email = excluded.email"
        ),
    )
    .await;
    assert_eq!(code, "23505", "UNIQUE violation SQLSTATE; message: {msg}");
    assert!(msg.contains("unique"), "specific message expected: {msg}");
    assert_eq!(
        snapshot(&client, &t, &["id", "email"]).await,
        before,
        "a rejected UPSERT must leave the table unchanged"
    );

    // Existing key moved onto ANOTHER row's unique value → 23505 too.
    let (code, msg) = expect_db_error(
        &client,
        &format!(
            "INSERT INTO {t} (id, email) VALUES (2, 'a@x') \
             ON CONFLICT (id) DO UPDATE SET email = excluded.email"
        ),
    )
    .await;
    assert_eq!(code, "23505", "UNIQUE violation SQLSTATE; message: {msg}");
    assert_eq!(snapshot(&client, &t, &["id", "email"]).await, before);

    // Replacing a row while keeping its own unique value is NOT a conflict.
    client
        .simple_query(&format!(
            "INSERT INTO {t} (id, email) VALUES (1, 'a@x') \
             ON CONFLICT (id) DO UPDATE SET email = excluded.email"
        ))
        .await
        .expect("self-replacing UPSERT must not self-conflict");
    assert_eq!(snapshot(&client, &t, &["id", "email"]).await, before);
}

/// CHECK: an UPSERT whose row fails a CHECK is rejected 23514 — on both the
/// insert arm (new key) and the update arm (existing key) — and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_upsert_enforces_check_constraint() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await;
    let t = format!("ups_chk_{}", suffix());

    client
        .simple_query(&format!(
            "CREATE TABLE {t} (id BIGINT PRIMARY KEY, qty BIGINT, CHECK (qty >= 0))"
        ))
        .await
        .expect("CREATE");
    client
        .simple_query(&format!("INSERT INTO {t} (id, qty) VALUES (1, 5)"))
        .await
        .expect("seed INSERT");
    let before = snapshot(&client, &t, &["id", "qty"]).await;
    assert_eq!(before, vec!["1|5"], "seed state");

    for (key, arm) in [(1, "update arm"), (2, "insert arm")] {
        let (code, msg) = expect_db_error(
            &client,
            &format!(
                "INSERT INTO {t} (id, qty) VALUES ({key}, -1) \
                 ON CONFLICT (id) DO UPDATE SET qty = excluded.qty"
            ),
        )
        .await;
        assert_eq!(
            code, "23514",
            "CHECK violation SQLSTATE ({arm}); message: {msg}"
        );
        assert!(msg.contains("CHECK"), "specific message expected: {msg}");
        assert_eq!(
            snapshot(&client, &t, &["id", "qty"]).await,
            before,
            "a rejected UPSERT ({arm}) must leave the table unchanged"
        );
    }
}

/// FOREIGN KEY: an UPSERT referencing a parent row that does not exist is
/// rejected 23503 — on both arms — and writes nothing; a valid UPSERT (existing
/// parent) still updates the child row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_upsert_enforces_foreign_key_and_valid_upsert_updates() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await;
    let s = suffix();
    let parent = format!("ups_par_{s}");
    let child = format!("ups_chl_{s}");

    client
        .simple_query(&format!(
            "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
        ))
        .await
        .expect("CREATE parent");
    client
        .simple_query(&format!(
            "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, note VARCHAR, \
             FOREIGN KEY (pid) REFERENCES {parent}(id))"
        ))
        .await
        .expect("CREATE child");
    client
        .simple_query(&format!(
            "INSERT INTO {parent} (id, name) VALUES ('p1', 'Alice'), ('p2', 'Bob')"
        ))
        .await
        .expect("seed parents");
    client
        .simple_query(&format!(
            "INSERT INTO {child} (id, pid, note) VALUES ('c1', 'p1', 'v1')"
        ))
        .await
        .expect("seed child");
    let cols = ["id", "pid", "note"];
    let before = snapshot(&client, &child, &cols).await;
    assert_eq!(before, vec!["c1|p1|v1"], "seed state");

    for (key, arm) in [("c1", "update arm"), ("c2", "insert arm")] {
        let (code, msg) = expect_db_error(
            &client,
            &format!(
                "INSERT INTO {child} (id, pid, note) VALUES ('{key}', 'p_missing', 'bad') \
                 ON CONFLICT (id) DO UPDATE SET pid = excluded.pid, note = excluded.note"
            ),
        )
        .await;
        assert_eq!(
            code, "23503",
            "FK violation SQLSTATE ({arm}); message: {msg}"
        );
        assert!(
            msg.contains("FOREIGN KEY"),
            "specific message expected: {msg}"
        );
        assert_eq!(
            snapshot(&client, &child, &cols).await,
            before,
            "a rejected UPSERT ({arm}) must leave the table unchanged"
        );
    }

    // A valid UPSERT still updates the existing row (re-pointed to p2) …
    client
        .simple_query(&format!(
            "INSERT INTO {child} (id, pid, note) VALUES ('c1', 'p2', 'v2') \
             ON CONFLICT (id) DO UPDATE SET pid = excluded.pid, note = excluded.note"
        ))
        .await
        .expect("valid UPSERT (update arm)");
    // … and inserts a new one.
    client
        .simple_query(&format!(
            "INSERT INTO {child} (id, pid, note) VALUES ('c2', 'p1', 'n2') \
             ON CONFLICT (id) DO UPDATE SET pid = excluded.pid, note = excluded.note"
        ))
        .await
        .expect("valid UPSERT (insert arm)");
    assert_eq!(
        snapshot(&client, &child, &cols).await,
        vec!["c1|p2|v2", "c2|p1|n2"],
        "a valid UPSERT must update the existing row and insert the new one"
    );
}
