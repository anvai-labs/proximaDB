// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Execute DML statements as ONE atomic unit (ADR-095 D3, TD-PGOLTP-1).
//!
//! The single implementation behind both an explicit pgwire `COMMIT` (the
//! session's buffered statements) and, with `[oltp] atomic_commit` on, every
//! autocommit statement (a one-statement transaction — so a cascading DELETE's
//! child and parent rows land in the same WAL group frame):
//!
//! 1. hold the op-locks of every table the statements touch (+ cascade closure);
//! 2. run the statements through the unchanged [`DmlService`], forked onto a
//!    [`StagingTableRecordStore`] — writes are captured, reads see them;
//! 3. make the captured writes durable with one [`TableRecordStore::commit_atomic`]
//!    in its own task (cancellation cannot split the append from the apply);
//! 4. apply the statistics the statements queued, only if step 3 succeeded.
//!
//! Any failure before step 3 leaves nothing durable. Errors are typed so each
//! wire surface maps them to its own status codes.

use std::sync::Arc;
use std::time::Duration;

use crate::services::canonical_wal::{WalAppendIndeterminate, WalAppenderPoisoned};
use crate::services::dml::{DmlResult, DmlService, DmlStatement};
use crate::services::txn_staging::{
    AtomicCommitUnsupported, CommitOutcomeUnknown, StagingTableRecordStore,
    TxnSerializationConflict,
};
use crate::storage::tenant::context::TenantContext;

/// One statement of an atomic unit.
#[derive(Debug, Clone)]
pub struct AtomicStatement {
    pub statement: DmlStatement,
    pub tenant: Option<TenantContext>,
    /// The table the statement writes (resolved in `tenant`'s scope).
    pub table: String,
}

/// Why an atomic unit did not commit. Only [`AtomicTxnError::OutcomeUnknown`]
/// can have made anything durable.
#[derive(Debug)]
pub enum AtomicTxnError {
    /// The `oltp-integrity` key store keeps claims in a second log, so the unit
    /// cannot be atomic yet (TD-PGOLTP-3). Nothing was committed.
    Unsupported(String),
    /// Resolving tables or taking locks failed. Nothing was committed.
    Setup(anyhow::Error),
    /// Statement `index` (0-based) failed while staging. Nothing was committed.
    /// The error is the statement's own (constraint violations keep their type).
    Statement { index: usize, error: anyhow::Error },
    /// Statement `index` exceeded the statement timeout. Nothing was committed.
    Timeout { index: usize },
    /// Current state changed under the unit (40001). Nothing was committed.
    SerializationConflict(anyhow::Error),
    /// The commit may or may not be durable; the store/appender is poisoned.
    /// `written_tables` are the tables the unit would have written (callers
    /// invalidate their caches, since the write may have landed).
    OutcomeUnknown {
        reason: String,
        written_tables: Vec<String>,
    },
    /// Any other commit failure before the durable append. Nothing was committed.
    Commit(anyhow::Error),
}

impl AtomicTxnError {
    /// Whether the unit may have been made durable (callers must not report
    /// "nothing was committed" and should invalidate caches).
    pub fn maybe_durable(&self) -> bool {
        matches!(self, Self::OutcomeUnknown { .. })
    }

    /// Tables whose caches must be invalidated after a maybe-durable failure.
    pub fn maybe_written_tables(&self) -> &[String] {
        match self {
            Self::OutcomeUnknown { written_tables, .. } => written_tables,
            _ => &[],
        }
    }

    /// The PostgreSQL SQLSTATE for this failure. `fallback` is used for a
    /// statement error without a typed constraint violation and for setup /
    /// commit failures — the caller knows whether a transaction block exists
    /// (COMMIT passes 25P02; an autocommit statement must not claim one).
    pub fn sqlstate(&self, fallback: &'static str) -> &'static str {
        let statement_fallback = fallback;
        match self {
            Self::Unsupported(_) => "0A000",
            Self::Setup(_) | Self::Commit(_) => fallback,
            Self::Statement { error, .. } => {
                proximadb_catalog::relational::find_relational_constraint_violation(error)
                    .map(|violation| violation.sqlstate())
                    .unwrap_or(statement_fallback)
            }
            Self::Timeout { .. } => "57014",
            Self::SerializationConflict(_) => "40001",
            Self::OutcomeUnknown { .. } => "08007",
        }
    }

    /// Client-facing message.
    pub fn message(&self) -> String {
        match self {
            Self::Unsupported(reason) => format!("{reason}; nothing was committed"),
            Self::Setup(error) => format!("{error:#}; nothing was committed"),
            Self::Statement { index, error } => {
                format!(
                    "statement {} failed: {error:#}; nothing was committed",
                    index + 1
                )
            }
            Self::Timeout { index } => format!(
                "canceling statement {} due to statement timeout; nothing was committed",
                index + 1
            ),
            Self::SerializationConflict(error) => format!("{error:#}; nothing was committed"),
            Self::OutcomeUnknown { reason, .. } => reason.clone(),
            Self::Commit(error) => {
                format!("transaction commit failed: {error:#}; nothing was committed")
            }
        }
    }
}

fn caused_by<E: std::error::Error + Send + Sync + 'static>(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<E>().is_some())
}

fn classify_commit_error(error: anyhow::Error, written_tables: &[String]) -> AtomicTxnError {
    // A poisoned appender refused BEFORE writing: nothing from this unit landed.
    if caused_by::<WalAppenderPoisoned>(&error) {
        AtomicTxnError::Commit(error)
    } else if caused_by::<CommitOutcomeUnknown>(&error)
        || caused_by::<WalAppendIndeterminate>(&error)
    {
        AtomicTxnError::OutcomeUnknown {
            reason: format!("{error:#}"),
            written_tables: written_tables.to_vec(),
        }
    } else if caused_by::<TxnSerializationConflict>(&error) {
        AtomicTxnError::SerializationConflict(error)
    } else if caused_by::<AtomicCommitUnsupported>(&error) {
        AtomicTxnError::Unsupported(format!("{error:#}"))
    } else {
        AtomicTxnError::Commit(error)
    }
}

/// A committed unit: each statement's [`DmlResult`] in order, plus every table
/// the unit wrote (cascade children included) for cache invalidation.
#[derive(Debug)]
pub struct AtomicOutcome {
    pub results: Vec<DmlResult>,
    pub written_tables: Vec<String>,
}

/// The op-lock set `statements` need: each target, plus the ON DELETE cascade
/// closure for DELETEs only.
pub async fn lock_set_for(
    dml: &DmlService,
    statements: &[AtomicStatement],
) -> anyhow::Result<std::collections::HashSet<crate::catalog::TableIdentifier>> {
    let tables: Vec<(String, Option<TenantContext>, bool)> = statements
        .iter()
        .map(|s| {
            (
                s.table.clone(),
                s.tenant.clone(),
                matches!(s.statement, DmlStatement::Delete { .. }),
            )
        })
        .collect();
    dml.transaction_op_lock_set(&tables).await
}

/// Run `statements` as one atomic unit.
pub async fn execute_atomically(
    dml: &Arc<DmlService>,
    statements: &[AtomicStatement],
    statement_timeout: Option<Duration>,
) -> Result<AtomicOutcome, AtomicTxnError> {
    let lock_set = lock_set_for(dml, statements)
        .await
        .map_err(AtomicTxnError::Setup)?;
    execute_atomically_with_lock_set(dml, statements, statement_timeout, lock_set).await
}

/// [`execute_atomically`] with a lock set the caller already resolved (the
/// autocommit path resolves it to decide whether a unit is needed at all).
pub async fn execute_atomically_with_lock_set(
    dml: &Arc<DmlService>,
    statements: &[AtomicStatement],
    statement_timeout: Option<Duration>,
    lock_set: std::collections::HashSet<crate::catalog::TableIdentifier>,
) -> Result<AtomicOutcome, AtomicTxnError> {
    if dml.has_conditional_key_store() {
        return Err(AtomicTxnError::Unsupported(
            "atomic commit is not supported with the oltp-integrity key store yet (TD-PGOLTP-3)"
                .to_string(),
        ));
    }
    let permits = dml
        .acquire_transaction_op_locks(&lock_set)
        .await
        .map_err(AtomicTxnError::Setup)?;

    let base = dml.record_store();
    let staging = Arc::new(StagingTableRecordStore::new(base.clone()));
    let fork = dml.fork_for_transaction(staging.clone(), lock_set);
    let mut results = Vec::with_capacity(statements.len());
    for (index, statement) in statements.iter().enumerate() {
        let run = fork.execute_scoped(statement.statement.clone(), statement.tenant.as_ref());
        let result = match statement_timeout {
            Some(limit) => match tokio::time::timeout(limit, run).await {
                Ok(result) => result,
                Err(_) => return Err(AtomicTxnError::Timeout { index }),
            },
            None => run.await,
        };
        results.push(result.map_err(|error| AtomicTxnError::Statement { index, error })?);
    }
    let deferred_stats = fork.take_deferred_stats();
    let retained_guards = fork.take_retained_dml_guards();
    drop(fork);
    let staging = Arc::try_unwrap(staging).map_err(|_| {
        AtomicTxnError::Setup(anyhow::anyhow!(
            "transaction staging store still shared at commit"
        ))
    })?;
    let batches = staging.into_batches();
    let mut written_tables: Vec<String> = Vec::new();
    for batch in &batches {
        if !written_tables.contains(&batch.table_schema.name) {
            written_tables.push(batch.table_schema.name.clone());
        }
    }

    // Own task: the caller's cancellation cannot split append from apply.
    // Inside it, in order: durable commit → statistics (still under the
    // op-locks, so concurrent writers' read-modify-write bumps cannot interleave)
    // → release the retained cross-pod DML leases → drop the op-locks.
    let dml_for_task = dml.clone();
    let committed = tokio::spawn(async move {
        let outcome = if batches.is_empty() {
            Ok(Vec::new())
        } else {
            base.commit_atomic(batches).await
        };
        if outcome.is_ok() {
            dml_for_task.apply_deferred_stats(deferred_stats).await;
        }
        for guard in retained_guards {
            guard.release().await;
        }
        drop(permits);
        outcome
    })
    .await
    .map_err(|join| AtomicTxnError::OutcomeUnknown {
        reason: format!("commit task did not complete: {join}"),
        written_tables: written_tables.clone(),
    })?;
    committed.map_err(|error| classify_commit_error(error, &written_tables))?;
    Ok(AtomicOutcome {
        results,
        written_tables,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_errors_are_classified_by_type_not_text() {
        let unknown = classify_commit_error(CommitOutcomeUnknown("x".into()).into(), &[]);
        assert!(unknown.maybe_durable());
        assert_eq!(unknown.sqlstate("XX000"), "08007");
        let wrapped = anyhow::Error::from(WalAppendIndeterminate("io".into())).context("layer");
        assert!(classify_commit_error(wrapped, &[]).maybe_durable());
        assert_eq!(
            classify_commit_error(TxnSerializationConflict("k".into()).into(), &[])
                .sqlstate("XX000"),
            "40001"
        );
        assert_eq!(
            classify_commit_error(AtomicCommitUnsupported("r".into()).into(), &[])
                .sqlstate("XX000"),
            "0A000"
        );
        // Text that merely LOOKS like a typed error is a plain commit failure.
        let lookalike =
            classify_commit_error(anyhow::anyhow!("not supported; spans storage routes"), &[]);
        assert!(!lookalike.maybe_durable());
        assert_eq!(lookalike.sqlstate("25P02"), "25P02");
        let timeout = AtomicTxnError::Timeout { index: 0 };
        assert_eq!(timeout.sqlstate("XX000"), "57014");
        // Review S4a F9: a poisoned appender refused before writing → nothing
        // committed, NOT outcome-unknown.
        let poisoned = classify_commit_error(anyhow::Error::from(WalAppenderPoisoned), &[]);
        assert!(!poisoned.maybe_durable());
        // Review S4a F8: setup/commit failures use the caller's fallback (an
        // autocommit statement has no transaction block to report 25P02 for).
        assert_eq!(poisoned.sqlstate("XX000"), "XX000");
        assert_eq!(poisoned.sqlstate("25P02"), "25P02");
        let unknown = classify_commit_error(
            CommitOutcomeUnknown("y".into()).into(),
            &["parent".to_string(), "child".to_string()],
        );
        assert_eq!(unknown.maybe_written_tables(), ["parent", "child"]);
    }
}
