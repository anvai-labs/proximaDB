// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-RERANK-PGWIRE-1: documents the REAL `RERANK(...)` contract over pgwire.
//!
//! `RERANK()` is a gRPC/federated-only capability today (`QueryFacadeAdapter`,
//! constructed only from `src/network/grpc/*`) — a comment in
//! `src/query/facade/adapter.rs` previously claimed pgwire clients could
//! `SELECT * FROM RERANK(...)`, which was false: pgwire has zero RERANK
//! dispatch anywhere. This test pins the actual, current, SAFE behavior —
//! pgwire declines rather than silently returning wrong data — so a future
//! change to the legacy-path comma guard or the UDTF-reachability gap
//! (TD-XMODAL-10) can't regress this into a silent-wrong-answer path without
//! a test noticing.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
#[test]
fn rerank_over_pgwire_declines_not_silently_wrong() {
    std::thread::Builder::new()
        .name("rerank-pgwire-contract-e2e".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt")
                .block_on(body())
        })
        .expect("spawn")
        .join()
        .expect("panic");
}

async fn body() {
    let server = TestPgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // Neither shape must return a success with zero (or any) rows — that
    // would be the silent-wrong-answer failure mode. Both must be a real
    // error, i.e. the client can tell "not supported" from "no matches".
    for sql in [
        "SELECT * FROM RERANK('coll', '[1,2,3]', 10)",
        "SELECT * FROM rerank(candidates, 'model_x')",
    ] {
        let err = client
            .simple_query(sql)
            .await
            .expect_err("RERANK() over pgwire must not silently succeed");
        let msg = pg_server::db_err(&err);
        // SQLSTATE 0A000 (feature not supported) either way, whatever the
        // proximate guard — this test pins SAFETY (an honest error), not the
        // exact wording of a message that legitimately belongs to a shared,
        // separately-scoped guard (TD-XMODAL-10 / the stale TD-REL-LOWER-1
        // citation) this PR does not touch.
        assert!(
            msg.starts_with("[0A000]"),
            "RERANK query {sql:?} should decline with 0A000, got: {msg}"
        );
    }
}
