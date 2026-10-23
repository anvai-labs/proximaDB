//! TD-RELIDEQ-1: `WHERE id = <value>` (or `record_id`) on a pgwire table with no
//! declared primary key returned no rows, and UPDATE / DELETE with the same
//! predicate reported 0 rows and changed nothing.
//!
//! Cause: with no declared key, `schema_primary_key_column` falls back to a
//! column named `id` or `record_id`, and the DML key fast path looks the value
//! up as the record `oid`. A keyless pgwire row's `oid` is generated, so every
//! lookup missed and the fast path returned the empty result without scanning.

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

/// The first column of every returned row, as text.
async fn column(client: &Client, sql: &str) -> Vec<String> {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)));
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => r.get(0).map(str::to_string),
            _ => None,
        })
        .collect()
}

/// Rows affected by a DML statement.
async fn affected(client: &Client, sql: &str) -> u64 {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {}", pg_server::db_err(&e)));
    messages
        .iter()
        .find_map(|m| match m {
            SimpleQueryMessage::CommandComplete(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0)
}

async fn connect(server: &TestPgServer) -> Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("pgwire connection error: {e}");
        }
    });
    client
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keyless_table_id_equality_reads_and_writes_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    for key in ["id", "record_id"] {
        let table = format!("keyless_{key}");
        exec(
            &client,
            &format!("CREATE TABLE {table} ({key} INT, name TEXT)"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
        )
        .await;

        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 2")
            )
            .await,
            vec!["b".to_string()],
            "{table}: equality must return the row"
        );
        let mut names = column(
            &client,
            &format!("SELECT name FROM {table} WHERE {key} IN (1, 3)"),
        )
        .await;
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "c".to_string()], "{table}: IN");
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 1 AND name = 'a'")
            )
            .await,
            vec!["a".to_string()],
            "{table}: equality under AND"
        );
        assert!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 99")
            )
            .await
            .is_empty(),
            "{table}: a missing key still returns nothing"
        );

        assert_eq!(
            affected(
                &client,
                &format!("UPDATE {table} SET name = 'z' WHERE {key} = 2")
            )
            .await,
            1,
            "{table}: UPDATE must affect the row"
        );
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 2")
            )
            .await,
            vec!["z".to_string()],
            "{table}: the update is visible"
        );
        assert_eq!(
            affected(&client, &format!("DELETE FROM {table} WHERE {key} = 1")).await,
            1,
            "{table}: DELETE must affect the row"
        );
        assert_eq!(
            affected(
                &client,
                &format!("UPDATE {table} SET {key} = 30 WHERE {key} = 3")
            )
            .await,
            1,
            "{table}: a keyless table's `{key}` is an ordinary column and can be updated"
        );
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 30")
            )
            .await,
            vec!["c".to_string()],
            "{table}: the row is found by its new value"
        );
        let mut left = column(
            &client,
            &format!("SELECT name FROM {table} WHERE {key} >= 0"),
        )
        .await;
        left.sort();
        assert_eq!(
            left,
            vec!["c".to_string(), "z".to_string()],
            "{table}: rows after DELETE"
        );
    }
}

/// Control: with a declared primary key the key fast path is correct and must
/// keep serving equality, IN, UPDATE and DELETE.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn declared_primary_key_equality_still_works_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;
    exec(
        &client,
        "CREATE TABLE keyed (id INT PRIMARY KEY, name TEXT)",
    )
    .await;
    exec(&client, "INSERT INTO keyed VALUES (1, 'a'), (2, 'b')").await;
    assert_eq!(
        column(&client, "SELECT name FROM keyed WHERE id = 2").await,
        vec!["b".to_string()]
    );
    assert_eq!(
        affected(&client, "UPDATE keyed SET name = 'z' WHERE id = 2").await,
        1
    );
    assert_eq!(affected(&client, "DELETE FROM keyed WHERE id = 1").await, 1);
    assert_eq!(
        column(&client, "SELECT name FROM keyed WHERE id = 2").await,
        vec!["z".to_string()]
    );
}

/// Whether a statement is rejected (the server's error, if any, for messages).
async fn rejected(client: &Client, sql: &str) -> Option<String> {
    client
        .simple_query(sql)
        .await
        .err()
        .map(|e| pg_server::db_err(&e))
}

/// Assert `sql` fails with SQLSTATE `code` — not merely "some error", so a
/// lock or routing failure cannot pass for a constraint violation.
async fn assert_rejected(client: &Client, sql: &str, code: &str, why: &str) {
    match rejected(client, sql).await {
        Some(error) => assert!(
            error.starts_with(&format!("[{code}]")),
            "{why}: `{sql}` must fail with {code}, got {error}"
        ),
        None => panic!("{why}: `{sql}` must fail with {code}, but it succeeded"),
    }
}

/// TD-RELIDEQ-2: `UNIQUE` on a keyless table's `id`/`record_id` column was
/// vacuous — the unique tuple read the column from the generated `oid`, so
/// duplicate values never conflicted (INSERT, multi-row INSERT, UPDATE).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keyless_unique_on_id_is_enforced_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    for key in ["id", "record_id"] {
        let table = format!("keyless_unique_{key}");
        exec(
            &client,
            &format!("CREATE TABLE {table} ({key} INT, name TEXT, UNIQUE ({key}))"),
        )
        .await;
        exec(&client, &format!("INSERT INTO {table} VALUES (1, 'a')")).await;

        assert_rejected(
            &client,
            &format!("INSERT INTO {table} VALUES (1, 'b')"),
            "23505",
            &format!("{table}: a duplicate {key} must be rejected"),
        )
        .await;
        assert_rejected(
            &client,
            &format!("INSERT INTO {table} VALUES (7, 'x'), (7, 'y')"),
            "23505",
            &format!("{table}: a duplicate {key} within one statement must be rejected"),
        )
        .await;
        exec(&client, &format!("INSERT INTO {table} VALUES (2, 'b')")).await;
        assert_rejected(
            &client,
            &format!("UPDATE {table} SET {key} = 1 WHERE name = 'b'"),
            "23505",
            &format!("{table}: UPDATE to a duplicate {key} must be rejected"),
        )
        .await;
        assert_eq!(
            column(
                &client,
                &format!("SELECT {key} FROM {table} ORDER BY {key}")
            )
            .await,
            vec!["1".to_string(), "2".to_string()],
            "{table}: rejected writes must leave no rows behind"
        );
    }
}

/// TD-RELIDEQ-2: a FOREIGN KEY to a keyless parent probed the child value as
/// the parent's `oid`, so valid children were rejected, and CASCADE matched
/// child values against parent oids and found nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn foreign_key_to_keyless_parent_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE keyless_parent (id INT, name TEXT, UNIQUE (id))",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE keyless_child (cid INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES keyless_parent(id) ON DELETE CASCADE)",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE keyless_child_restrict (cid INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES keyless_parent(id))",
    )
    .await;
    exec(
        &client,
        "INSERT INTO keyless_parent VALUES (1, 'a'), (2, 'b')",
    )
    .await;

    if let Some(error) = rejected(&client, "INSERT INTO keyless_child VALUES (10, 1)").await {
        panic!("a child of an existing keyless parent must be accepted: {error}");
    }
    assert_rejected(
        &client,
        "INSERT INTO keyless_child VALUES (11, 99)",
        "23503",
        "a child of a missing parent must be rejected",
    )
    .await;
    if let Some(error) =
        rejected(&client, "INSERT INTO keyless_child_restrict VALUES (20, 2)").await
    {
        panic!("restrict child insert: {error}");
    }

    // RESTRICT (NO ACTION): the referenced parent cannot be deleted.
    assert_rejected(
        &client,
        "DELETE FROM keyless_parent WHERE id = 2",
        "23503",
        "deleting a referenced keyless parent must be rejected",
    )
    .await;
    // CASCADE: deleting the parent deletes its children.
    assert_eq!(
        affected(&client, "DELETE FROM keyless_parent WHERE id = 1").await,
        1
    );
    assert!(
        column(&client, "SELECT cid FROM keyless_child")
            .await
            .is_empty(),
        "ON DELETE CASCADE must remove the child of a keyless parent"
    );
    assert_eq!(
        column(&client, "SELECT id FROM keyless_parent").await,
        vec!["2".to_string()]
    );
}

/// TD-RELIDEQ-2: on a composite primary key, the first key column was read
/// from the `oid` — the encoded WHOLE tuple — so `UNIQUE` on that column never
/// conflicted. Reads of that column must also see its own value.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn composite_key_first_column_unique_and_reads_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE composite_pk (region TEXT, id INT, name TEXT, \
         PRIMARY KEY (region, id), UNIQUE (region))",
    )
    .await;
    exec(&client, "INSERT INTO composite_pk VALUES ('eu', 1, 'a')").await;

    assert_eq!(
        column(&client, "SELECT region FROM composite_pk").await,
        vec!["eu".to_string()],
        "a composite key's first column reads as its own value"
    );
    assert_eq!(
        column(&client, "SELECT name FROM composite_pk WHERE region = 'eu'").await,
        vec!["a".to_string()],
        "WHERE on a composite key's first column matches"
    );
    assert_rejected(
        &client,
        "INSERT INTO composite_pk VALUES ('eu', 2, 'b')",
        "23505",
        "UNIQUE (region) must reject a second 'eu' row",
    )
    .await;
}

/// TD-RELIDEQ-2: a UNIQUE-referenced parent value can change (a primary key
/// cannot), so UPDATE must not orphan children still using the old value
/// (ON UPDATE NO ACTION). Unreferenced values, and a swap that keeps the
/// referenced value present, stay updatable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_of_referenced_unique_value_is_restricted_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE ref_parent (id INT, name TEXT, UNIQUE (id))",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE ref_child (cid INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES ref_parent(id))",
    )
    .await;
    exec(&client, "INSERT INTO ref_parent VALUES (1, 'a'), (2, 'b')").await;
    exec(&client, "INSERT INTO ref_child VALUES (10, 1)").await;

    assert_rejected(
        &client,
        "UPDATE ref_parent SET id = 5 WHERE id = 1",
        "23503",
        "changing a referenced value must be rejected",
    )
    .await;
    assert_eq!(
        affected(&client, "UPDATE ref_parent SET id = 6 WHERE id = 2").await,
        1,
        "an unreferenced value stays updatable"
    );
    assert_eq!(
        affected(&client, "UPDATE ref_parent SET name = 'z' WHERE id = 1").await,
        1,
        "a non-key column of a referenced row stays updatable"
    );
    assert_eq!(
        column(&client, "SELECT id FROM ref_parent ORDER BY id").await,
        vec!["1".to_string(), "6".to_string()]
    );
    assert_eq!(
        column(&client, "SELECT pid FROM ref_child").await,
        vec!["1".to_string()],
        "the child still references an existing parent"
    );
}

/// TD-RELIDEQ-2 (review M2): every column of a composite key is immutable,
/// and an UPDATE of another column keeps the key's own values (the row
/// builder read the first key column from the oid — the encoded whole tuple).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn composite_key_rows_survive_update_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE composite_upd (region TEXT, id INT, name TEXT, PRIMARY KEY (region, id))",
    )
    .await;
    exec(&client, "INSERT INTO composite_upd VALUES ('eu', 1, 'a')").await;

    assert_eq!(
        affected(
            &client,
            "UPDATE composite_upd SET name = 'b' WHERE region = 'eu'"
        )
        .await,
        1
    );
    assert_eq!(
        column(&client, "SELECT region FROM composite_upd WHERE id = 1").await,
        vec!["eu".to_string()],
        "the key's first column keeps its value after an UPDATE"
    );
    assert_eq!(
        column(
            &client,
            "SELECT name FROM composite_upd WHERE region = 'eu'"
        )
        .await,
        vec!["b".to_string()]
    );
    for column_name in ["region", "id"] {
        let value = if column_name == "id" { "9" } else { "'us'" };
        assert!(
            rejected(
                &client,
                &format!("UPDATE composite_upd SET {column_name} = {value} WHERE name = 'b'")
            )
            .await
            .is_some(),
            "a composite key column ({column_name}) is immutable"
        );
    }
}

/// TD-RELIDEQ-2 (review M1): an UPSERT that replaces a row can change a UNIQUE
/// value children reference — the same ON UPDATE NO ACTION as UPDATE.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsert_of_referenced_unique_value_is_restricted_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE ups_parent (id INT PRIMARY KEY, email TEXT, UNIQUE (email))",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE ups_child (cid INT PRIMARY KEY, email TEXT, \
         FOREIGN KEY (email) REFERENCES ups_parent(email))",
    )
    .await;
    exec(
        &client,
        "INSERT INTO ups_parent VALUES (1, 'a@x'), (2, 'b@x')",
    )
    .await;
    exec(&client, "INSERT INTO ups_child VALUES (10, 'a@x')").await;

    assert_rejected(
        &client,
        "INSERT INTO ups_parent VALUES (1, 'c@x') ON CONFLICT (id) DO UPDATE SET email = excluded.email",
        "23503",
        "an UPSERT must not orphan the child of the replaced value",
    )
    .await;
    assert_eq!(
        affected(
            &client,
            "INSERT INTO ups_parent VALUES (2, 'd@x') ON CONFLICT (id) DO UPDATE SET email = excluded.email"
        )
        .await,
        1,
        "an unreferenced value stays upsertable"
    );
    assert_eq!(
        column(&client, "SELECT email FROM ups_parent ORDER BY email").await,
        vec!["a@x".to_string(), "d@x".to_string()]
    );
}

/// TD-RELIDEQ-2 (review m2): ON DELETE SET NULL updates the child; when the
/// nulled columns are themselves referenced through a UNIQUE set, that
/// UPDATE must not orphan the grandchildren.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn set_null_does_not_orphan_grandchildren_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(&client, "CREATE TABLE sn_root (id INT PRIMARY KEY)").await;
    exec(
        &client,
        "CREATE TABLE sn_mid (mid INT PRIMARY KEY, rid INT, UNIQUE (rid), \
         FOREIGN KEY (rid) REFERENCES sn_root(id) ON DELETE SET NULL)",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE sn_leaf (lid INT PRIMARY KEY, rid INT, \
         FOREIGN KEY (rid) REFERENCES sn_mid(rid))",
    )
    .await;
    exec(&client, "INSERT INTO sn_root VALUES (1)").await;
    exec(&client, "INSERT INTO sn_mid VALUES (100, 1)").await;
    exec(&client, "INSERT INTO sn_leaf VALUES (1000, 1)").await;

    assert_rejected(
        &client,
        "DELETE FROM sn_root WHERE id = 1",
        "23503",
        "SET NULL on sn_mid.rid would orphan sn_leaf",
    )
    .await;
    assert_eq!(
        column(&client, "SELECT rid FROM sn_mid").await,
        vec!["1".to_string()],
        "the rejected DELETE leaves the middle row unchanged"
    );
}

/// TD-RELIDEQ-2 (review round 2, M-A): an UPSERT replaces the WHOLE row, so a
/// column left out of its column list is rewritten too — it must not be able
/// to drop a UNIQUE value children reference. Whatever the outcome, no child
/// may be left without its parent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsert_with_a_column_list_keeps_references_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(
        &client,
        "CREATE TABLE ul_parent (id INT PRIMARY KEY, code INT, name TEXT, UNIQUE (code))",
    )
    .await;
    exec(
        &client,
        "CREATE TABLE ul_child (cid INT PRIMARY KEY, pcode INT, \
         FOREIGN KEY (pcode) REFERENCES ul_parent(code))",
    )
    .await;
    exec(&client, "INSERT INTO ul_parent VALUES (1, 100, 'a')").await;
    exec(&client, "INSERT INTO ul_child VALUES (10, 100)").await;

    let outcome = rejected(
        &client,
        "INSERT INTO ul_parent (id, name) VALUES (1, 'b') ON CONFLICT (id) DO UPDATE SET name = 'b'",
    )
    .await;
    if let Some(error) = &outcome {
        assert!(error.starts_with("[23503]"), "{error}");
    }
    assert_eq!(
        column(&client, "SELECT code FROM ul_parent WHERE code = 100").await,
        vec!["100".to_string()],
        "the referenced value must survive the UPSERT (outcome: {outcome:?})"
    );
}

/// TD-RELIDEQ-2 (review m5): only NO ACTION / RESTRICT are implemented for ON
/// UPDATE; a declared CASCADE must be refused, never enforced as NO ACTION.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn declared_on_update_cascade_is_refused_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    exec(&client, "CREATE TABLE oc_parent (id INT, UNIQUE (id))").await;
    exec(
        &client,
        "CREATE TABLE oc_child (cid INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES oc_parent(id) ON UPDATE CASCADE)",
    )
    .await;
    exec(&client, "INSERT INTO oc_parent VALUES (1)").await;
    exec(&client, "INSERT INTO oc_child VALUES (10, 1)").await;

    let error = rejected(&client, "UPDATE oc_parent SET id = 2 WHERE id = 1")
        .await
        .expect("a declared ON UPDATE CASCADE must be refused");
    assert!(error.contains("not supported"), "{error}");
    assert_eq!(
        column(&client, "SELECT id FROM oc_parent").await,
        vec!["1".to_string()]
    );
}
