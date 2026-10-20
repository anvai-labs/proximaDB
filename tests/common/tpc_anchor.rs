//! Shared TPC accuracy-phase machinery (ADR-040, TD-182 P1) for the TPC-H and
//! TPC-DS pgwire conformance suites.
//!
//! Included per-target via `#[path = "common/tpc_anchor.rs"] mod tpc_anchor;`
//! (same pattern as `common/pg_server.rs`). Everything here is a verbatim move
//! of the helpers that were duplicated across `tests/tpch_pgwire_e2e.rs` and
//! `tests/tpcds_pgwire_e2e.rs`; seed data, expected values and ratchet
//! constants stay suite-local.

/// Render a tokio_postgres error including the real server-side DbError cause.
pub fn explain_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!("[{}] {}", db.code().code(), db.message())
    } else {
        e.to_string()
    }
}

/// Normalize a result cell for value comparison: numeric cells are rounded to 2
/// decimals (collapsing f64 decimal-accumulation error on `DOUBLE PRECISION` price
/// sums, e.g. 69.4799… → "69.48") and rendered minimally (whole numbers without a
/// decimal); SQL NULL → "NULL"; other text passes through.
pub fn norm_cell(s: &str) -> String {
    match s.parse::<f64>() {
        Ok(f) => {
            let r = (f * 100.0).round() / 100.0;
            if r.fract() == 0.0 {
                format!("{}", r as i64)
            } else {
                format!("{r}")
            }
        }
        Err(_) => s.to_string(),
    }
}

/// Run a query and return its rows as normalized text cells, **sorted** for
/// order-independent set comparison (ADR-040 `canon()` style; row order across
/// routes is a separate concern — see TD-185).
pub async fn canon_rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| norm_cell(r.get(i).unwrap_or("NULL")))
                    .collect(),
            ),
            _ => None,
        })
        .collect();
    rows.sort();
    rows
}

/// Phase-boundary re-seed: drop + recreate every schema table in place so the
/// accuracy phase can reload its seed on the SAME server boot as the
/// conformance phase (a second `ProximaDB` boot reuses the process-global WAL
/// manifest pointed at a deleted tempdir — never boot twice).
///
/// The conformance phase's queries + this shared boot leave catalog/WAL state
/// where a best-effort `DROP TABLE IF EXISTS` can ERROR (swallowed,
/// historically) instead of no-op'ing — the subsequent CREATE then hit
/// "already exists" and panicked. Surface the drop error, and if CREATE still
/// collides, retry once via a hard DROP before failing loudly. A true CREATE
/// regression still panics; only the stale-state race is absorbed.
pub async fn reseed_for_accuracy_phase(client: &tokio_postgres::Client, schema: &[(&str, &str)]) {
    for (name, ddl) in schema {
        if let Err(e) = client
            .simple_query(&format!("DROP TABLE IF EXISTS {name}"))
            .await
        {
            eprintln!(
                "DROP IF EXISTS {name} errored (retrying create): {}",
                explain_err(&e)
            );
        }
        if client.simple_query(ddl).await.is_err() {
            let _ = client.simple_query(&format!("DROP TABLE {name}")).await;
            client.simple_query(ddl).await.unwrap_or_else(|e| {
                panic!("CREATE {name} after stale-state retry: {}", explain_err(&e))
            });
        }
    }
}

/// One anchored query in an ADR-040 accuracy phase: the anchor id (a trailing
/// `-KNOWN-BAD` suffix turns the entry into a KNOWN-BAD trip-wire instead of an
/// accuracy assertion), the resolved SQL, and the hand-derived expected rows
/// (cell-by-cell normalized, compared order-independently via `canon_rows`).
pub struct AnchorSpec<'a> {
    pub id: &'a str,
    pub sql: &'a str,
    pub expected: Vec<Vec<&'a str>>,
}

/// Run the KNOWN-BAD-aware anchored accuracy loop and enforce `ratchet`.
///
/// Verbatim machinery shared by the TPC-H/TPC-DS accuracy phases: every
/// non-KNOWN-BAD spec must produce exactly its expected rows (`✓ accurate` /
/// `✗` failure lines, failures summarized), a `-KNOWN-BAD` spec must still
/// produce its pinned WRONG answer (the trip-panic fires the moment the
/// engine's answer CHANGES, forcing re-anchoring), and the accurate count must
/// meet `ratchet`. `suite_label` parameterizes the summary and assert messages
/// ("TPC-H" / "TPC-DS"); `known_bad_hint` / `known_bad_close` carry the
/// suite-specific tail of the KNOWN-BAD trip-panic text. Returns
/// `(accurate, total)`.
pub async fn run_anchored_phase(
    client: &tokio_postgres::Client,
    specs: &[AnchorSpec<'_>],
    suite_label: &str,
    known_bad_hint: &str,
    known_bad_close: &str,
    ratchet: usize,
) -> (usize, usize) {
    let mut accurate = 0;
    let mut failures = Vec::new();
    for spec in specs {
        let id = spec.id;
        let is_known_bad = id.ends_with("-KNOWN-BAD");
        let lookup_id = id.strip_suffix("-KNOWN-BAD").unwrap_or(id);
        let got = canon_rows(client, spec.sql).await;
        let want: Vec<Vec<String>> = spec
            .expected
            .iter()
            .map(|row| row.iter().map(|c| c.to_string()).collect())
            .collect();
        if is_known_bad {
            if got == want {
                // Still returns the pinned wrong answer — known-bad holds.
                eprintln!("  ✓ {id} still KNOWN-BAD (pinned wrong answer)");
                continue;
            }
            panic!(
                "{id}: KNOWN-BAD anchor tripped — the engine's answer CHANGED (got \
                 {got:?}, pinned {want:?}). If the result is now the CORRECT \
                 {known_bad_hint}, move {lookup_id} into the anchored set, and close \
                 {known_bad_close}"
            );
        }
        if got == want {
            accurate += 1;
            eprintln!("  ✓ {id} accurate ({} rows)", got.len());
        } else {
            failures.push(format!("{id}: want {want:?}, got {got:?}"));
        }
    }

    eprintln!(
        "\n=== {suite_label} accuracy: {accurate}/{total} anchored queries accurate (ratchet {ratchet}) ===",
        total = specs.len(),
    );
    for f in &failures {
        eprintln!("  ✗ {f}");
    }
    assert!(
        failures.is_empty(),
        "{suite_label} anchored accuracy failures: {}",
        failures.join("; ")
    );
    assert!(
        accurate >= ratchet,
        "{suite_label} accuracy regressed: {accurate} < ratchet {ratchet}"
    );
    (accurate, specs.len())
}
