// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Writer-epoch port (ADR-095 D2, TD-PGOLTP-2).
//!
//! Exactly one process may write a durable volume at a time. Each process that
//! boots against the volume claims the next epoch; a process that observes a
//! higher epoch than its own must stop acknowledging writes. The epoch store is
//! the authority (an object-store log in production), so it works for every
//! volume class — including network filesystems that offer no hardware fence.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Who is claiming. `nonce` is unique per claim attempt so a claimant can
/// recognise its own write after an ambiguous (lost-response) commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriterClaim {
    pub host: String,
    pub pid: u32,
    pub nonce: String,
    pub claimed_at_ms: i64,
    /// Identity of the volume this claim is for (ADR-095 review F4). Epoch 1
    /// binds the log to a volume; later claimants must present the same id, so
    /// an unmounted mount point or a different deployment's volume can never
    /// take over. `None` only for slots written before the field existed.
    #[serde(default)]
    pub volume_id: Option<String>,
}

/// A won epoch and the claim that won it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochGrant {
    pub epoch: u64,
    pub claim: WriterClaim,
}

#[async_trait]
pub trait EpochStore: Send + Sync {
    /// Durably claim `latest + 1` for `claim`. Never reuses an epoch. Losing a
    /// race to a concurrent claimant is an error: the caller must not serve.
    async fn acquire_next(&self, claim: &WriterClaim) -> anyhow::Result<EpochGrant>;

    /// The highest granted epoch, if any (the watchdog's view of the authority).
    async fn current(&self) -> anyhow::Result<Option<EpochGrant>>;
}
