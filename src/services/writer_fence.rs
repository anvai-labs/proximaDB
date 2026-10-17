// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Process-wide writer fence (ADR-095 D2, TD-PGOLTP-2).
//!
//! When a replacement process wins a newer writer epoch for the same durable
//! volume, the old process must stop acknowledging writes. Two mechanisms, both
//! checked by the canonical WAL appender before every append:
//!
//! * **Lost flag** — set the moment the watchdog OBSERVES a newer epoch. One-way.
//! * **Ownership lease** — when a durable volume is configured, a write is
//!   allowed only while the lease the watchdog last renewed is unexpired. The
//!   watchdog renews it only after a successful check that saw no newer epoch,
//!   and stamps the expiry from the check's START. So a process whose checks
//!   hang (partition), fail, or never run (SIGSTOP, VM pause — the monotonic
//!   clock keeps advancing) stops acknowledging within one lease of its last
//!   good check, without needing to observe anything. A new owner waits at least
//!   one lease after claiming before it opens the WAL, which bounds the overlap.
//!
//! Without a durable volume the lease is not enforced (behaviour unchanged).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Monotonic process clock origin for lease arithmetic.
fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn millis_since_origin(at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(origin()).as_millis()).unwrap_or(u64::MAX)
}

/// Writer fence. `Default` is "not fenced, no lease enforced".
#[derive(Debug, Default)]
pub struct WriterFence {
    lost: AtomicBool,
    lease_enforced: AtomicBool,
    /// Lease expiry, in ms since [`origin`].
    lease_until_ms: AtomicU64,
}

impl WriterFence {
    /// The process-wide fence the server's epoch watchdog drives. Appenders use
    /// it unless a test injects its own.
    pub fn global() -> Arc<WriterFence> {
        static GLOBAL: OnceLock<Arc<WriterFence>> = OnceLock::new();
        GLOBAL
            .get_or_init(|| Arc::new(WriterFence::default()))
            .clone()
    }

    /// Record that a newer writer epoch exists. Irreversible.
    pub fn mark_lost(&self) {
        self.lost.store(true, Ordering::SeqCst);
    }

    /// Whether a newer writer epoch has been observed.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// Start enforcing the ownership lease, valid for `lease` from `checked_at`.
    pub fn enforce_lease(&self, checked_at: Instant, lease: Duration) {
        self.renew_lease(checked_at, lease);
        self.lease_enforced.store(true, Ordering::SeqCst);
    }

    /// Extend the lease to `checked_at + lease` (never shortens it).
    pub fn renew_lease(&self, checked_at: Instant, lease: Duration) {
        self.lease_until_ms
            .fetch_max(millis_since_origin(checked_at + lease), Ordering::SeqCst);
    }

    fn lease_expired_at(&self, now: Instant) -> bool {
        self.lease_enforced.load(Ordering::SeqCst)
            && millis_since_origin(now) > self.lease_until_ms.load(Ordering::SeqCst)
    }

    /// Fail when fenced or when the enforced lease has expired.
    pub fn ensure_owner(&self) -> anyhow::Result<()> {
        if self.is_lost() {
            anyhow::bail!(
                "writer fenced by a newer epoch: another process owns this durable volume; \
                 refusing to make the write durable"
            );
        }
        if self.lease_expired_at(Instant::now()) {
            anyhow::bail!(
                "writer fenced: the durable-volume ownership lease expired (no successful \
                 epoch check within the lease); refusing to make the write durable"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_of(fence: &WriterFence) -> String {
        fence
            .ensure_owner()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn fence_is_one_way() {
        let fence = WriterFence::default();
        assert!(fence.ensure_owner().is_ok());
        fence.mark_lost();
        assert!(fence.is_lost());
        assert!(err_of(&fence).contains("writer fenced"));
    }

    #[test]
    fn unenforced_lease_never_blocks() {
        let fence = WriterFence::default();
        assert!(!fence.lease_expired_at(Instant::now() + Duration::from_secs(3600)));
    }

    // A paused or partitioned owner stops acknowledging once its lease (stamped
    // from the last good check's START) runs out, without observing anything.
    #[test]
    fn enforced_lease_expires_and_renewal_extends_it() {
        let fence = WriterFence::default();
        let start = Instant::now();
        fence.enforce_lease(start, Duration::from_millis(500));
        assert!(!fence.lease_expired_at(start + Duration::from_millis(400)));
        assert!(fence.lease_expired_at(start + Duration::from_millis(600)));
        fence.renew_lease(
            start + Duration::from_millis(450),
            Duration::from_millis(500),
        );
        assert!(!fence.lease_expired_at(start + Duration::from_millis(900)));
        // A stale (older) renewal never shortens the lease.
        fence.renew_lease(start, Duration::from_millis(100));
        assert!(!fence.lease_expired_at(start + Duration::from_millis(900)));
    }
}
