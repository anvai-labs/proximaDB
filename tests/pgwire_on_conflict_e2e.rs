//! TD-DML-3: `INSERT … ON CONFLICT` honours its conflict TARGET and ACTION
//! with PostgreSQL semantics — over pgwire, the mandated conformance surface.
//!
//! Before the fix the conflict was always resolved on the PRIMARY KEY, `DO
//! UPDATE SET …` assignments were not evaluated (the whole proposed row replaced
//! the existing one), `DO NOTHING` replaced instead of skipping, and a conflict
//! on a non-PK target failed closed with 23505. Each test pins one PostgreSQL
//! behaviour and reads the table back afterwards (read-after-write).

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
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

/// The `CommandComplete` row count of a successful statement.
async fn affected(client: &Client, sql: &str) -> u64 {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("statement failed: {sql}: {e:?}"));
    messages
        .iter()
        .find_map(|msg| match msg {
            SimpleQueryMessage::CommandComplete(n) => Some(*n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no CommandComplete for `{sql}`"))
}

/// `id BIGINT PRIMARY KEY, email UNIQUE, counter CHECK (>= 0), note`, seeded
/// with (1,'a',10,'n1') and (2,'b',20,'n2').
async fn seeded_table(client: &Client, prefix: &str) -> String {
    let t = format!("{prefix}_{}", suffix());
    client
        .simple_query(&format!(
            "CREATE TABLE {t} (id BIGINT PRIMARY KEY, email VARCHAR, counter BIGINT, \
             note VARCHAR, UNIQUE (email), CHECK (counter >= 0))"
        ))
        .await
        .expect("CREATE");
    client
        .simple_query(&format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'a', 10, 'n1'), (2, 'b', 20, 'n2')"
        ))
        .await
        .expect("seed INSERT");
    t
}

const COLS: [&str; 4] = ["id", "email", "counter", "note"];

/// DO NOTHING skips the conflicting row (leaving the existing row unchanged),
/// still inserts the non-conflicting rows of the same statement, and reports
/// only the inserted rows — on the PK, on a UNIQUE column, with no target (any
/// unique constraint), and for a key repeated within the statement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_do_nothing_skips_conflicting_rows() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let t = seeded_table(&client, "oc_nothing").await;

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'z', 99, 'clobber'), (3, 'c', 30, 'n3') \
             ON CONFLICT (id) DO NOTHING"
        ),
    )
    .await;
    assert_eq!(n, 1, "only the non-conflicting row is inserted");
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|10|n1", "2|b|20|n2", "3|c|30|n3"],
        "DO NOTHING on the PK must leave the existing row unchanged"
    );

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (4, 'a', 40, 'dup'), (5, 'e', 50, 'n5') \
             ON CONFLICT (email) DO NOTHING"
        ),
    )
    .await;
    assert_eq!(n, 1, "the UNIQUE-column conflict is skipped, not rejected");
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|10|n1", "2|b|20|n2", "3|c|30|n3", "5|e|50|n5"]
    );

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (2, 'q', 0, 'pk'), (6, 'b', 0, 'uq'), (7, 'g', 70, 'n7') \
             ON CONFLICT DO NOTHING"
        ),
    )
    .await;
    assert_eq!(
        n, 1,
        "with no target, a conflict on ANY unique constraint is skipped"
    );

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (8, 'h', 1, 'first'), (8, 'i', 2, 'second') \
             ON CONFLICT (id) DO NOTHING"
        ),
    )
    .await;
    assert_eq!(
        n, 1,
        "a key repeated within the statement conflicts with its own earlier row"
    );
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec![
            "1|a|10|n1",
            "2|b|20|n2",
            "3|c|30|n3",
            "5|e|50|n5",
            "7|g|70|n7",
            "8|h|1|first"
        ]
    );
}

/// DO UPDATE SET evaluates its assignments against the EXISTING row, with
/// EXCLUDED naming the proposed row; untouched columns keep their values; the
/// optional WHERE gates the update (a false WHERE updates and counts nothing);
/// a non-PK target updates the row holding that unique value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_do_update_evaluates_assignments() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let t = seeded_table(&client, "oc_update").await;

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'zz', 5, 'proposed') \
             ON CONFLICT (id) DO UPDATE SET counter = {t}.counter + EXCLUDED.counter"
        ),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|15|n1", "2|b|20|n2"],
        "only the assigned column changes, computed from existing + EXCLUDED"
    );

    // Read-after-write through the PK point-lookup path too.
    let point = client
        .simple_query(&format!("SELECT counter FROM {t} WHERE id = 1"))
        .await
        .expect("point SELECT");
    let counter = point.iter().find_map(|m| match m {
        SimpleQueryMessage::Row(r) => r.get(0).map(str::to_string),
        _ => None,
    });
    assert_eq!(counter.as_deref(), Some("15"));

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (2, 'b', 1, 'gated') \
             ON CONFLICT (id) DO UPDATE SET note = EXCLUDED.note WHERE {t}.counter > 10"
        ),
    )
    .await;
    assert_eq!(n, 1, "a true WHERE updates the row");
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (2, 'b', 1, 'blocked') \
             ON CONFLICT (id) DO UPDATE SET note = EXCLUDED.note WHERE {t}.counter > 1000"
        ),
    )
    .await;
    assert_eq!(n, 0, "a false WHERE neither updates nor counts the row");
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|15|n1", "2|b|20|gated"]
    );

    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (99, 'a', 1, 'via-email') \
             ON CONFLICT (email) DO UPDATE SET note = EXCLUDED.note, counter = {t}.counter * 2"
        ),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|30|via-email", "2|b|20|gated"],
        "a UNIQUE-column target updates the row holding that value; id 99 is not inserted"
    );

    // Multi-row VALUES mixing updates and an insert.
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'a', 1, 'm1'), (3, 'c', 33, 'n3'), (2, 'b', 2, 'm2') \
             ON CONFLICT (id) DO UPDATE SET counter = {t}.counter + EXCLUDED.counter, note = EXCLUDED.note"
        ),
    )
    .await;
    assert_eq!(n, 3, "two updated rows plus one inserted row");
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|31|m1", "2|b|22|m2", "3|c|33|n3"]
    );
}

/// The post-update row is checked like any write: UNIQUE (23505), CHECK
/// (23514); a statement that would update one row twice is 21000; a target
/// matching no unique constraint is 42P10; a DO UPDATE with no target is
/// 42601; an unknown constraint name is 42704. Every rejection leaves the
/// table unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_rejections_carry_postgres_sqlstates() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let t = seeded_table(&client, "oc_reject").await;
    let before = snapshot(&client, &t, &COLS).await;

    let cases = [
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (2, 'x', 0, '') \
                 ON CONFLICT (id) DO UPDATE SET email = 'a'"
            ),
            "23505",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (2, 'x', 0, '') \
                 ON CONFLICT (id) DO UPDATE SET counter = {t}.counter - 1000"
            ),
            "23514",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, ''), (1, 'y', 2, '') \
                 ON CONFLICT (id) DO UPDATE SET counter = EXCLUDED.counter"
            ),
            "21000",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (5, 'x', 1, ''), (6, 'a', 2, '') \
                 ON CONFLICT (email) DO UPDATE SET email = 'x'"
            ),
            "23505",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, '') \
                 ON CONFLICT (note) DO NOTHING"
            ),
            "42P10",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, '') \
                 ON CONFLICT (id, email) DO UPDATE SET note = 'n'"
            ),
            "42P10",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, '') \
                 ON CONFLICT DO UPDATE SET note = 'n'"
            ),
            "42601",
        ),
        (
            format!(
                "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, '') \
                 ON CONFLICT ON CONSTRAINT no_such_constraint DO NOTHING"
            ),
            "42704",
        ),
    ];
    for (sql, want) in &cases {
        let (code, msg) = expect_db_error(&client, sql).await;
        assert_eq!(&code, want, "`{sql}` → {msg}");
        assert_eq!(
            snapshot(&client, &t, &COLS).await,
            before,
            "a rejected statement must leave the table unchanged: {sql}"
        );
    }

    // The PostgreSQL default constraint names resolve.
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'x', 1, '') \
             ON CONFLICT ON CONSTRAINT {t}_pkey DO NOTHING"
        ),
    )
    .await;
    assert_eq!(n, 0);
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (9, 'b', 1, '') \
             ON CONFLICT ON CONSTRAINT {t}_email_key DO UPDATE SET note = 'by-name'"
        ),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        snapshot(&client, &t, &COLS).await,
        vec!["1|a|10|n1", "2|b|20|by-name"]
    );
}

/// All rows of `SELECT <cols> FROM <table>`, `|`-joined and sorted (for tables
/// without an `id` column).
async fn sorted_rows(client: &Client, table: &str, cols: &[&str]) -> Vec<String> {
    let rows = client
        .simple_query(&format!("SELECT {} FROM {table}", cols.join(", ")))
        .await
        .expect("SELECT");
    let mut out: Vec<String> = rows
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some(
                (0..cols.len())
                    .map(|i| row.get(i).unwrap_or("∅").to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            _ => None,
        })
        .collect();
    out.sort();
    out
}

/// Composite PRIMARY KEY: DO UPDATE of an existing row mixed with an insert,
/// then a plain UPDATE, each rewrite the SAME row (no phantom key, no 42P01).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_composite_pk_and_pk_less_tables() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let cp = format!("oc_cp_{}", suffix());
    client
        .simple_query(&format!(
            "CREATE TABLE {cp} (a VARCHAR NOT NULL, b BIGINT NOT NULL, n BIGINT, PRIMARY KEY (a, b))"
        ))
        .await
        .expect("CREATE composite");
    client
        .simple_query(&format!("INSERT INTO {cp} (a, b, n) VALUES ('x', 1, 1)"))
        .await
        .expect("seed");
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {cp} (a, b, n) VALUES ('x', 1, 5), ('x', 2, 2) \
             ON CONFLICT (a, b) DO UPDATE SET n = {cp}.n + EXCLUDED.n"
        ),
    )
    .await;
    assert_eq!(n, 2);
    assert_eq!(
        sorted_rows(&client, &cp, &["a", "b", "n"]).await,
        vec!["x|1|6", "x|2|2"]
    );
    let n = affected(
        &client,
        &format!("UPDATE {cp} SET n = 9 WHERE a = 'x' AND b = 2"),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        sorted_rows(&client, &cp, &["a", "b", "n"]).await,
        vec!["x|1|6", "x|2|9"]
    );

    // UPDATE may not assign ANY PK column (it used to accept the non-first
    // one, splitting the row's identity from its key columns).
    for sql in [
        format!("UPDATE {cp} SET b = 5 WHERE a = 'x' AND b = 1"),
        format!("UPDATE {cp} SET a = 'y' WHERE a = 'x' AND b = 1"),
    ] {
        let (code, msg) = expect_db_error(&client, &sql).await;
        assert_eq!(code, "0A000", "`{sql}` → {msg}");
        assert_eq!(
            sorted_rows(&client, &cp, &["a", "b", "n"]).await,
            vec!["x|1|6", "x|2|9"],
            "{sql}"
        );
    }

    // No PRIMARY KEY: a UNIQUE column arbitrates and DO UPDATE rewrites the holder.
    let np = format!("oc_np_{}", suffix());
    client
        .simple_query(&format!(
            "CREATE TABLE {np} (email VARCHAR, n BIGINT, UNIQUE (email))"
        ))
        .await
        .expect("CREATE pk-less");
    client
        .simple_query(&format!("INSERT INTO {np} (email, n) VALUES ('a', 1)"))
        .await
        .expect("seed");
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {np} (email, n) VALUES ('a', 5) \
             ON CONFLICT (email) DO UPDATE SET n = {np}.n + EXCLUDED.n"
        ),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        sorted_rows(&client, &np, &["email", "n"]).await,
        vec!["a|6"]
    );
}

/// Expression and binding failures carry PostgreSQL's SQLSTATEs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_expression_errors_carry_postgres_sqlstates() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let t = seeded_table(&client, "oc_expr").await;
    let before = snapshot(&client, &t, &COLS).await;
    let head = format!("INSERT INTO {t} (id, email, counter, note) VALUES (1, 'z', 1, '')");
    for (tail, want) in [
        (
            "ON CONFLICT (id) DO UPDATE SET counter = 'abc'".to_string(),
            "22P02",
        ),
        (
            format!("ON CONFLICT (id) DO UPDATE SET counter = {t}.counter / 0"),
            "22012",
        ),
        (
            "ON CONFLICT (id) DO UPDATE SET nosuch = 1".to_string(),
            "42703",
        ),
        ("ON CONFLICT (nosuch) DO NOTHING".to_string(), "42703"),
        (
            "ON CONFLICT (id) DO UPDATE SET counter = 1, counter = 2".to_string(),
            "42701",
        ),
        (
            "ON CONFLICT (email) DO UPDATE SET id = 7".to_string(),
            "0A000",
        ),
        (
            format!("ON CONFLICT (id) DO UPDATE SET counter = {t}.note - 1"),
            "42883",
        ),
    ] {
        let sql = format!("{head} {tail}");
        let (code, msg) = expect_db_error(&client, &sql).await;
        assert_eq!(code, want, "`{sql}` → {msg}");
        assert_eq!(snapshot(&client, &t, &COLS).await, before, "{sql}");
    }
}

/// Vector tables route to the vector store. A relational table with VECTOR(n)
/// columns validates each vector against its own declared dimension (not the
/// backing collection's), so single- and multi-vector tables accept rows; and
/// UPDATE / DO UPDATE keep every vector column's own value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_vector_tables_insert_and_update_preserve_every_vector() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let sv = format!("oc_sv_{}", suffix());
    client
        .simple_query(&format!(
            "CREATE TABLE {sv} (id BIGINT PRIMARY KEY, note VARCHAR, v VECTOR(3))"
        ))
        .await
        .expect("CREATE single-vector");
    assert_eq!(
        affected(
            &client,
            &format!("INSERT INTO {sv} VALUES (1, 'a', '[1,2,3]')")
        )
        .await,
        1,
        "a VECTOR(3) row is accepted"
    );

    let mv = format!("oc_mv_{}", suffix());
    client
        .simple_query(&format!(
            "CREATE TABLE {mv} (id BIGINT PRIMARY KEY, note VARCHAR, v1 VECTOR(3), v2 VECTOR(3))"
        ))
        .await
        .expect("CREATE multi-vector");
    assert_eq!(
        affected(
            &client,
            &format!("INSERT INTO {mv} VALUES (1, 'a', '[1,1,1]', '[2,2,2]')")
        )
        .await,
        1,
        "a two-vector row is accepted"
    );
    let cols = ["id", "note", "v1", "v2"];
    let n = affected(&client, &format!("UPDATE {mv} SET note = 'u' WHERE id = 1")).await;
    assert_eq!(n, 1);
    assert_eq!(
        sorted_rows(&client, &mv, &cols).await,
        vec!["1|u|[1,1,1]|[2,2,2]"],
        "UPDATE keeps every vector column's own value"
    );
    let n = affected(
        &client,
        &format!(
            "INSERT INTO {mv} (id, note, v1, v2) VALUES (1, 'd', '[9,9,9]', '[8,8,8]') \
             ON CONFLICT (id) DO UPDATE SET note = EXCLUDED.note"
        ),
    )
    .await;
    assert_eq!(n, 1);
    assert_eq!(
        sorted_rows(&client, &mv, &cols).await,
        vec!["1|d|[1,1,1]|[2,2,2]"],
        "DO UPDATE keeps every vector column's own value"
    );
}

/// A DROPPED column's values never come back: not through UPDATE / DO UPDATE
/// carrying them forward, and not when a column of the same name is re-ADDed
/// (it reads NULL for every pre-existing row, as in PostgreSQL).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_dropped_column_values_never_resurrect() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    for with_updates in [true, false] {
        let t = format!("oc_dc_{}_{}", with_updates as u8, suffix());
        client
            .simple_query(&format!(
                "CREATE TABLE {t} (id BIGINT PRIMARY KEY, n BIGINT, secret VARCHAR)"
            ))
            .await
            .expect("CREATE");
        client
            .simple_query(&format!(
                "INSERT INTO {t} VALUES (1, 1, 'old-secret'), (2, 2, 'old2')"
            ))
            .await
            .expect("seed");
        client
            .simple_query(&format!("ALTER TABLE {t} DROP COLUMN secret"))
            .await
            .expect("DROP COLUMN");
        if with_updates {
            affected(&client, &format!("UPDATE {t} SET n = 10 WHERE id = 1")).await;
            affected(
                &client,
                &format!(
                    "INSERT INTO {t} VALUES (2, 5) ON CONFLICT (id) DO UPDATE SET n = {t}.n + EXCLUDED.n"
                ),
            )
            .await;
        }
        client
            .simple_query(&format!("ALTER TABLE {t} ADD COLUMN secret VARCHAR"))
            .await
            .expect("re-ADD COLUMN");
        let rows = client
            .simple_query(&format!("SELECT id FROM {t} WHERE secret IS NOT NULL"))
            .await
            .expect("SELECT");
        let resurrected = rows
            .iter()
            .filter(|m| matches!(m, SimpleQueryMessage::Row(_)))
            .count();
        assert_eq!(
            resurrected, 0,
            "a re-ADDed column must read NULL for pre-existing rows (updates: {with_updates})"
        );
    }
}

/// `CAST(text AS BOOLEAN)` of a non-boolean string is 22P02 (PostgreSQL's
/// invalid_text_representation); assigning a boolean to a BIGINT is 42804.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_on_conflict_boolean_cast_sqlstates() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");
    let t = seeded_table(&client, "oc_cast").await;
    let before = snapshot(&client, &t, &COLS).await;
    for (tail, want) in [
        (format!("SET counter = CAST({t}.note AS BOOLEAN)"), "22P02"),
        ("SET counter = CAST('true' AS BOOLEAN)".to_string(), "42804"),
    ] {
        let sql = format!(
            "INSERT INTO {t} (id, email, counter, note) VALUES (1, 'z', 1, '') ON CONFLICT (id) DO UPDATE {tail}"
        );
        let (code, msg) = expect_db_error(&client, &sql).await;
        assert_eq!(code, want, "`{sql}` → {msg}");
        assert_eq!(snapshot(&client, &t, &COLS).await, before);
    }
}
