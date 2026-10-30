// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Native (non-Parquet) relational scan — read-after-write & dead-record ratchet
//! (TD-REL-SCAN-1).
//!
//! Exercises the production native relational read path end-to-end through the
//! PostgreSQL wire protocol, never bypassing pgwire or the engine selector. The
//! table is **never** `MATERIALIZE`d, so it stays PAX/record-backed
//! (non-Parquet): SELECTs route via `ComputeScheduler::route_select` to the
//! **Native (Volcano)** backend →
//! `DmlService::scan_table_relational` →
//! `DirectWalTableRecordStore` → the per-(tenant, collection)
//! `MemtableRecordStorage` partition (`enable_direct_record_writes` defaults ON).
//!
//! This locks in the TD-REL-SCAN-1 conclusion that the relational native scan is
//! storage-inclusive and dead-filtered *by construction* (single never-evicted
//! authoritative tier + `matches_record`/`is_visible_at` dead filter + physical
//! memtable DELETE), so it can neither (a) miss a live row nor (b) return a
//! dead/tombstoned row. It is the native sibling of `pgwire_olap_delta_merge_e2e`
//! (which covers the Parquet/MATERIALIZE OLAP read-merge instead).
//!
//! Run with the real server-side cause behind any opaque pgwire `db error`:
//!   RUST_LOG=proximadb=debug cargo nextest run \
//!     --test pgwire_relational_native_read_after_write_e2e -- --nocapture

use tokio_postgres::{Client, SimpleQueryMessage};

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
async fn exec(client: &Client, sql: &str) {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {}", pg_server::db_err(&e)));
}

/// Run a single-cell scalar SELECT and return the one cell as text.
async fn scalar(client: &Client, sql: &str) -> String {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)));
    for m in &messages {
        if let SimpleQueryMessage::Row(r) = m {
            return r
                .get(0)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "NULL".to_string());
        }
    }
    panic!("query `{sql}` returned no row");
}

fn num(s: &str) -> i64 {
    s.parse::<i64>()
        .unwrap_or_else(|_| panic!("expected integer scalar, got `{s}`"))
}

/// Collect the first column of every returned row as i64 (simple-query renders
/// each cell as text). Used to enumerate the surviving `id`s of a scan.
async fn column_i64(client: &Client, sql: &str) -> Vec<i64> {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)));
    let mut out = Vec::new();
    for m in &messages {
        if let SimpleQueryMessage::Row(r) = m
            && let Some(cell) = r.get(0)
        {
            out.push(
                cell.parse::<i64>()
                    .unwrap_or_else(|_| panic!("non-integer id cell `{cell}` from `{sql}`")),
            );
        }
    }
    out
}

/// A native (non-materialized) relational table over pgwire must (a) return
/// EVERY inserted live row — no row "missed" — and (b) never surface a deleted
/// row, while (c) reflecting updates and (d) allowing a previously-deleted key to
/// be re-inserted (no tombstone resurrection). All SELECTs are aggregate/ordered
/// shapes that engage the relational engine → `scan_table_relational` → the
/// DirectWal memtable scan under test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_relational_scan_read_after_write_and_dead_filter() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    // Unique table name so parallel/repeat runs never collide in the catalog.
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let t = format!("rel_raw_{suffix}");

    exec(
        &client,
        &format!("CREATE TABLE {t} (id INT PRIMARY KEY, v INT)"),
    )
    .await;
    // NOTE: intentionally NO `ALTER TABLE ... MATERIALIZE` — the table stays
    // PAX/record-backed (non-Parquet), so reads take the native DirectWal path.

    // --- (a) Read-after-write completeness: insert N rows, expect ALL back. ---
    // N=64 is comfortably larger than any single-batch boundary, so a "flushed
    // rows dropped" bug (the documents() failure mode) would surface as a short
    // COUNT/SUM here. v = id * 10 gives a known closed-form sum.
    const N: i64 = 64;
    for id in 1..=N {
        exec(
            &client,
            &format!("INSERT INTO {t} (id, v) VALUES ({id}, {})", id * 10),
        )
        .await;
    }
    let expected_sum_all: i64 = (1..=N).map(|id| id * 10).sum();
    assert_eq!(
        num(&scalar(&client, &format!("SELECT COUNT(*) FROM {t}")).await),
        N,
        "read-after-write: every inserted live row must be scanned (none missed)"
    );
    assert_eq!(
        num(&scalar(&client, &format!("SELECT SUM(v) FROM {t}")).await),
        expected_sum_all,
        "read-after-write: SUM over all live rows"
    );
    let all_ids = column_i64(
        &client,
        &format!("SELECT id FROM {t} GROUP BY id ORDER BY id ASC"),
    )
    .await;
    assert_eq!(
        all_ids,
        (1..=N).collect::<Vec<_>>(),
        "read-after-write: ordered enumeration returns exactly the inserted ids"
    );

    // --- (b) Delete visibility / dead-record filter: drop the upper half. ---
    // Range predicate (not modulo — the relational frontend doesn't parse `%`).
    const CUT: i64 = 32;
    exec(&client, &format!("DELETE FROM {t} WHERE id > {CUT}")).await;
    let surviving: Vec<i64> = (1..=CUT).collect();
    let expected_sum_surviving: i64 = surviving.iter().map(|id| id * 10).sum();
    assert_eq!(
        num(&scalar(&client, &format!("SELECT COUNT(*) FROM {t}")).await),
        surviving.len() as i64,
        "delete: deleted rows must be invisible to the scan (no tombstone leak)"
    );
    assert_eq!(
        num(&scalar(&client, &format!("SELECT SUM(v) FROM {t}")).await),
        expected_sum_surviving,
        "delete: SUM must exclude deleted rows"
    );
    let after_delete = column_i64(
        &client,
        &format!("SELECT id FROM {t} GROUP BY id ORDER BY id ASC"),
    )
    .await;
    assert_eq!(
        after_delete, surviving,
        "delete: exactly the surviving ids (1..={CUT}) remain — no deleted id resurfaces, none dropped"
    );

    // --- (c) Update reflected on the very next read (write invalidates cache). ---
    // Bump id=1 from v=10 to v=1000; SUM shifts by exactly +990.
    exec(&client, &format!("UPDATE {t} SET v = 1000 WHERE id = 1")).await;
    assert_eq!(
        num(&scalar(&client, &format!("SELECT SUM(v) FROM {t}")).await),
        expected_sum_surviving + 990,
        "update: the new value is visible on the next read"
    );

    // --- (d) No tombstone resurrection: re-insert a previously-deleted key. ---
    // id=50 was deleted (physical memtable remove); re-inserting must succeed and
    // reappear with its NEW value — never a stale/tombstoned copy.
    let reinserted = CUT + 18; // 50 — was in the deleted upper half
    exec(
        &client,
        &format!("INSERT INTO {t} (id, v) VALUES ({reinserted}, 22)"),
    )
    .await;
    let final_ids = column_i64(
        &client,
        &format!("SELECT id FROM {t} GROUP BY id ORDER BY id ASC"),
    )
    .await;
    let mut expect_final = surviving.clone();
    expect_final.push(reinserted);
    expect_final.sort_unstable();
    assert_eq!(
        final_ids, expect_final,
        "re-insert of a deleted key must reappear exactly once (no tombstone resurrection)"
    );
    assert_eq!(
        num(&scalar(
            &client,
            &format!("SELECT v FROM {t} WHERE id = {reinserted}")
        )
        .await),
        22,
        "re-inserted key carries its new value"
    );
}
