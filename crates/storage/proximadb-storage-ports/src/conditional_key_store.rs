// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! # `ConditionalKeyStore` — the fenced atomic OLTP uniqueness primitive
//!
//! ADR-072 **D3** (the contract), **D11** (fenced, returns-holder-on-conflict,
//! capability-typed, append-only reclaim) and **D14** (tombstone, not slot-reclaim).
//!
//! This is the storage-plane primitive that enforces composite-natural-key
//! uniqueness. It is deliberately **not** a bare `put_if_absent`: the ledger's
//! proven core returns a version, and a bare object-store `PutMode::Create`
//! drops that token — without it a writer cannot tell "my own idempotent retry"
//! from "a real conflict". So every write is fenced by a [`Generation`] and a
//! conflict **returns the current holder** (Cornus `LogOnce` returns the settled
//! state).
//!
//! ## The key prefix is uniqueness *scoping*, not access control
//!
//! The [`Identity`]'s leading tenant/keyspace prefix (from the F0 codec,
//! `key_codec::encode_identity`, D9) exists so the same PK value in different
//! tenants/tables is a **distinct uniqueness domain** — it *partitions the
//! keyspace*, it does not authorize anyone. The prefix is *derived from* the
//! caller's already-authenticated tenant, so uniqueness is scoped correctly; but
//! the security boundary is the **authenticated subject + ABAC row-level filters
//! applied at runtime** (ADR-072 D15/D6), governed by the catalog metamodel (D12)
//! with subject↔permission bindings in a system namespace. Row-, field-, and
//! attribute-level policies cannot live in a key — they are runtime filters,
//! never this store's concern. `get`/`tombstone` likewise key on the full
//! `Identity`.
//!
//! ## Two implementations, one contract (ADR-072 D5)
//!
//! - generation-fenced **local-WAL** CAS — the hot, correctness-carrying default;
//! - object-store **successor-slot / batch-manifest** — the scale-to-zero escape
//!   hatch.
//!
//! Each declares its [`AtomicScope`] so callers never assume uniform atomicity
//! (ScalarDB's *Atomicity Unit*; D11).

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A typed composite key: the opaque, order-preserving bytes produced by the F0
/// codec (`proximadb_data_model::key_codec::encode_identity`, ADR-072 D9). The
/// bytes embed the `tenant`/`keyspace` prefix, so equality and range order are
/// tenant-scoped (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Identity(pub Vec<u8>);

impl Identity {
    /// Wrap already-encoded key bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// The encoded key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// The object a key resolves to (the row's `oid`). For a composite **natural
/// primary key** `oid` is the key's own encoding (`oid == primary_key_string`);
/// for a **secondary `UNIQUE`** the key is the unique-value encoding and `oid`
/// is the owning row's identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Oid(pub String);

/// A monotonic fence token. Every mutation carries one; the store rejects a write
/// carrying a stale generation, and reads use it to place a version on the MVCC
/// timeline (ADR-072 D11). A generation never decreases for a given key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Generation(pub u64);

/// The logical outcome of [`ConditionalKeyStore::put_if_absent`]. `Err` is
/// reserved for I/O / transport failure — a **conflict is not an error**, it is
/// an expected `Ok` outcome that carries the current holder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PutOutcome {
    /// The key was absent (or a prior version was tombstoned) and is now claimed
    /// for `oid` at `generation`.
    Committed { generation: Generation },
    /// The key is already live. `holder` is the current owner and `generation`
    /// its version — enough for a loser to distinguish an idempotent retry
    /// (`holder == my oid`) from a genuine uniqueness conflict (ADR-072 D11).
    Conflict { holder: Oid, generation: Generation },
}

/// The atomic scope an implementation guarantees for a single call — a
/// *capability* the caller can inspect, never a boolean `supports_cas`
/// (ScalarDB's Atomicity Unit; ADR-072 D11). Callers must not assume atomicity
/// wider than the declared scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicScope {
    /// Atomic per individual key (the generation-fenced local-WAL default).
    PerKey,
    /// Atomic per `(tenant, partition)` — a partition-lease-fenced impl.
    PerPartition,
    /// Atomic per committed batch manifest (the object-store escape hatch:
    /// many keys become visible together on one successor-slot commit).
    PerBatchManifest,
}

/// The fenced atomic uniqueness primitive. One contract, many implementations
/// (ADR-072 D2/D5): each backend implements this behind the same surface.
///
/// Reclaim is **append-only** (ADR-072 D11/D14): [`Self::tombstone`] never reuses
/// a durable slot — it records a delete at a generation, and physical
/// reclamation is a background-plane concern (F6) gated by the oldest-active-reader
/// watermark. A deletion **encodes** a delete; it does **not** enforce uniqueness
/// — that is this store's job via `put_if_absent`.
#[async_trait::async_trait]
pub trait ConditionalKeyStore: Send + Sync {
    /// The atomic scope this implementation guarantees (D11 capability typing).
    fn atomic_scope(&self) -> AtomicScope;

    /// Claim `key` for `oid`, fenced by `fence`, iff no live version exists.
    /// Returns the logical [`PutOutcome`]; `Err` is reserved for I/O/transport
    /// failure, never a conflict. Idempotent under retry: re-issuing the same
    /// `(key, oid)` observes `Conflict { holder: oid, .. }`, which the caller
    /// reads as success.
    async fn put_if_absent(
        &self,
        key: &Identity,
        oid: &Oid,
        fence: Generation,
    ) -> Result<PutOutcome>;

    /// Resolve `key` to its current live holder, or `None` if absent/tombstoned.
    /// Foreign-key existence checks (phase 2) ride this method.
    async fn get(&self, key: &Identity) -> Result<Option<Oid>>;

    /// Append-only tombstone at `fence` — mark the current live version of `key`
    /// deleted **without reusing the slot** (ADR-072 D11/D14). A later
    /// `put_if_absent` of the same `key` is a *new* version, not a slot reuse.
    /// Physical reclamation happens off the request path (F6).
    async fn tombstone(&self, key: &Identity, fence: Generation) -> Result<()>;

    /// Every key that currently has a LIVE holder, with that holder — the
    /// input of a startup reconciliation that releases claims whose row does
    /// not exist (TD-DML-3). `None` ⇒ this backend cannot enumerate (the
    /// caller then relies on lazy, per-key reconciliation only).
    async fn live_claims(&self) -> Result<Option<Vec<(Identity, Oid)>>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A minimal in-memory reference implementation used to pin the contract's
    /// semantics (append-only reclaim, return-holder-on-conflict). Not a
    /// production store.
    #[derive(Default)]
    struct MemStore {
        // key -> (oid, generation, live)
        map: Mutex<HashMap<Identity, (Oid, Generation, bool)>>,
    }

    #[async_trait::async_trait]
    impl ConditionalKeyStore for MemStore {
        fn atomic_scope(&self) -> AtomicScope {
            AtomicScope::PerKey
        }

        async fn put_if_absent(
            &self,
            key: &Identity,
            oid: &Oid,
            fence: Generation,
        ) -> Result<PutOutcome> {
            let mut m = self.map.lock().unwrap();
            match m.get(key) {
                Some((holder, g, true)) => Ok(PutOutcome::Conflict {
                    holder: holder.clone(),
                    generation: *g,
                }),
                // absent or a tombstoned prior version -> claim a new version
                _ => {
                    m.insert(key.clone(), (oid.clone(), fence, true));
                    Ok(PutOutcome::Committed { generation: fence })
                }
            }
        }

        async fn get(&self, key: &Identity) -> Result<Option<Oid>> {
            let m = self.map.lock().unwrap();
            Ok(match m.get(key) {
                Some((oid, _, true)) => Some(oid.clone()),
                _ => None,
            })
        }

        async fn tombstone(&self, key: &Identity, fence: Generation) -> Result<()> {
            let mut m = self.map.lock().unwrap();
            if let Some(entry) = m.get_mut(key) {
                entry.1 = fence;
                entry.2 = false; // append-only: mark not-live, slot retained
            }
            Ok(())
        }
    }

    fn id(s: &str) -> Identity {
        Identity::from_bytes(s.as_bytes().to_vec())
    }

    #[tokio::test]
    async fn lifecycle_commit_conflict_tombstone_reinsert() {
        let store = MemStore::default();
        let k = id("acme/users/\x02pk");
        let a = Oid("row-a".into());
        let b = Oid("row-b".into());

        // First insert claims the key.
        assert_eq!(
            store.put_if_absent(&k, &a, Generation(1)).await.unwrap(),
            PutOutcome::Committed {
                generation: Generation(1)
            }
        );

        // A second, different writer conflicts and learns the holder.
        assert_eq!(
            store.put_if_absent(&k, &b, Generation(2)).await.unwrap(),
            PutOutcome::Conflict {
                holder: a.clone(),
                generation: Generation(1)
            }
        );

        // The original writer's retry sees its own oid as holder -> success.
        match store.put_if_absent(&k, &a, Generation(3)).await.unwrap() {
            PutOutcome::Conflict { holder, .. } => assert_eq!(holder, a),
            other => panic!("expected conflict-with-self, got {other:?}"),
        }

        assert_eq!(store.get(&k).await.unwrap(), Some(a.clone()));

        // Append-only delete, then re-insert is a NEW version (not a slot reuse).
        store.tombstone(&k, Generation(4)).await.unwrap();
        assert_eq!(store.get(&k).await.unwrap(), None);
        assert_eq!(
            store.put_if_absent(&k, &b, Generation(5)).await.unwrap(),
            PutOutcome::Committed {
                generation: Generation(5)
            }
        );
        assert_eq!(store.get(&k).await.unwrap(), Some(b));
    }

    #[test]
    fn scope_is_a_capability() {
        assert_eq!(MemStore::default().atomic_scope(), AtomicScope::PerKey);
    }
}
