/*
 * Copyright 2025 Vijaykumar Singh
 * (Apache-2.0)
 */

//! The embedded process-lifecycle contract: one state file per data dir.
//!
//! ## Why this exists
//!
//! The boundary between a supervising client (the Python SDK's
//! `EmbeddedProximaDB`, a test harness, an IDE) and a spawned server carried
//! **no state in either direction**, which produced two failure modes that
//! look unrelated but share one cause:
//!
//! * **Startup was indistinguishable from death** (#1667). Listeners bind at
//!   the END of `ProximaDB::start` (recovery first), so during a long WAL
//!   replay there is no socket at all. A supervisor polling HTTP sees exactly
//!   what a crashed process looks like, so it applies a fixed timeout — 30 s
//!   in the SDK — and kills a healthy server that was minutes into a
//!   measured >180 s recovery of a 1.3 GB data dir, then silently fell back
//!   to another backend.
//! * **Ownership was unrecorded** (anvai-labs/victor#911). A server spawned
//!   with `setsid` outlives its parent, keeps writing its data dir, and is
//!   invisible to the next run — so "clear the data dir and re-index"
//!   silently resurrected old state, and orphans accumulated.
//!
//! Both are the same gap: *the lifecycle has no observable state and no
//! ownership record*. Per the co-design mandate, the fix belongs at the
//! boundary and must be a contract, not a heuristic — so this module defines
//! one, and both sides are tested against it.
//!
//! ## The contract
//!
//! While a server owns `<data_dir>`, it maintains `<data_dir>/.proximadb-runtime.json`:
//!
//! ```json
//! {"pid": 123, "phase": "recovering_storage", "started_at_ms": 1, "updated_at_ms": 2,
//!  "heartbeat_interval_ms": 2000, "config_path": "…", "version": "0.3.0"}
//! ```
//!
//! * **Written before any slow work** — a supervisor learns the pid and phase
//!   long before a socket exists.
//! * **Heartbeat advances** (`updated_at_ms`) on a fixed interval for the
//!   process lifetime. A supervisor therefore distinguishes *slow* (phase or
//!   heartbeat advancing) from *dead* (heartbeat stale, or pid gone) without
//!   any fixed assumption about how long recovery "should" take. This is the
//!   property that makes an unbounded operation safely supervisable.
//! * **Removed on clean exit**, so an absent file means "no owner".
//! * **Staleness is decidable**: [`RuntimeState::is_stale`] — pid no longer
//!   alive, or heartbeat older than a generous multiple of its interval.
//!
//! The file is advisory, not a lock: it never blocks the server from
//! starting. It exists so clients can make correct decisions instead of
//! guessing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// File name, relative to the data dir. Dot-prefixed so it never collides
/// with a collection or engine directory.
pub const RUNTIME_STATE_FILE: &str = ".proximadb-runtime.json";

/// Heartbeat cadence. Small enough that a supervisor detects death quickly,
/// large enough to be free relative to recovery work.
pub const HEARTBEAT_INTERVAL_MS: u64 = 2_000;

/// A heartbeat older than this many intervals means the writer is gone or
/// wedged. Generous: a paused process (SIGSTOP, laptop sleep, a stop-the-world
/// pause) must not be declared dead on the first missed beat.
pub const STALE_AFTER_INTERVALS: u64 = 15;

/// Lifecycle phases, coarse by design: a supervisor needs "is it moving",
/// not a trace. Ordered — [`Phase::Serving`] is the only ready state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Process is up; config parsed; nothing expensive started yet.
    Starting,
    /// Storage engine recovery: WAL replay, memtable materialization.
    RecoveringStorage,
    /// Graph engines recovering from WAL/snapshots.
    RecoveringGraphs,
    /// Network surfaces binding.
    Binding,
    /// Fully started; listeners are up and the server answers requests.
    Serving,
    /// Graceful shutdown in progress.
    Stopping,
}

impl Phase {
    /// The one phase in which sockets are guaranteed to exist.
    pub fn is_ready(self) -> bool {
        matches!(self, Phase::Serving)
    }
}

/// On-disk lifecycle record. Field-additive: unknown fields are ignored by
/// readers, so a newer server can add detail without breaking older clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pub pid: u32,
    pub phase: Phase,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub heartbeat_interval_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl RuntimeState {
    /// True when the recorded owner cannot be serving this data dir any more:
    /// the process is gone, or its heartbeat stopped advancing long ago.
    ///
    /// `now_ms` is injected so this is a pure, testable decision.
    pub fn is_stale(&self, now_ms: u64) -> bool {
        if !pid_is_alive(self.pid) {
            return true;
        }
        let interval = self.heartbeat_interval_ms.max(1);
        now_ms.saturating_sub(self.updated_at_ms) > interval * STALE_AFTER_INTERVALS
    }

    /// True when the owner is alive AND its heartbeat is fresh — i.e. another
    /// process legitimately owns this data dir right now.
    pub fn is_live_owner(&self, now_ms: u64) -> bool {
        !self.is_stale(now_ms)
    }
}

/// Path of the state file for a data dir.
pub fn state_path(data_dir: &Path) -> PathBuf {
    data_dir.join(RUNTIME_STATE_FILE)
}

/// Read the current owner record, if any. A malformed file reads as `None`
/// (treated as "no owner") rather than an error: a corrupt record must never
/// prevent a server from starting.
pub fn read_state(data_dir: &Path) -> Option<RuntimeState> {
    let raw = std::fs::read(state_path(data_dir)).ok()?;
    serde_json::from_slice(&raw).ok()
}

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    // Signal 0 probes existence without delivering anything.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    // Conservative on platforms without a cheap probe: assume alive, so
    // staleness falls back to the heartbeat test alone.
    true
}

/// Owns the state file for the process lifetime: writes it, advances the
/// heartbeat on a background task, and removes it on drop.
pub struct RuntimeStateWriter {
    data_dir: PathBuf,
    state: Arc<std::sync::Mutex<RuntimeState>>,
    stop: Arc<AtomicBool>,
}

impl RuntimeStateWriter {
    /// Publish the initial record and start the heartbeat.
    ///
    /// Failure to write is logged, never fatal — the server must start even
    /// on a read-only or unusual filesystem; clients simply lose the
    /// observability the contract offers.
    pub fn start(data_dir: &Path, config_path: Option<String>) -> Self {
        // RuntimeState timestamps are u64 while the shared kernel clock is i64;
        // epoch millis are never negative, so the cast is lossless.
        let now = proximadb_kernel::time::now_unix_ms() as u64;
        let state = RuntimeState {
            pid: std::process::id(),
            phase: Phase::Starting,
            started_at_ms: now,
            updated_at_ms: now,
            heartbeat_interval_ms: HEARTBEAT_INTERVAL_MS,
            config_path,
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        };
        let writer = Self {
            data_dir: data_dir.to_path_buf(),
            state: Arc::new(std::sync::Mutex::new(state)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        writer.persist();

        // Heartbeat: a blocking thread rather than a tokio task, so it keeps
        // advancing even if the async runtime is saturated by recovery work —
        // precisely the situation where a client is deciding whether we are
        // alive.
        let data_dir = writer.data_dir.clone();
        let state = Arc::clone(&writer.state);
        let stop = Arc::clone(&writer.stop);
        std::thread::Builder::new()
            .name("proximadb-runtime-heartbeat".into())
            .spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(HEARTBEAT_INTERVAL_MS));
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(mut guard) = state.lock() {
                        guard.updated_at_ms = proximadb_kernel::time::now_unix_ms() as u64;
                        write_state(&data_dir, &guard);
                    }
                }
            })
            .ok();

        writer
    }

    /// Record a phase transition (also refreshes the heartbeat).
    pub fn set_phase(&self, phase: Phase) {
        if let Ok(mut guard) = self.state.lock() {
            guard.phase = phase;
            guard.updated_at_ms = proximadb_kernel::time::now_unix_ms() as u64;
            write_state(&self.data_dir, &guard);
        }
    }

    fn persist(&self) {
        if let Ok(guard) = self.state.lock() {
            write_state(&self.data_dir, &guard);
        }
    }

    /// Stop the heartbeat and remove the record — the "no owner" signal.
    pub fn finish(&self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::fs::remove_file(state_path(&self.data_dir));
    }
}

impl Drop for RuntimeStateWriter {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Write atomically (temp + rename) so a reader never observes a half-written
/// record — a torn read would look like "no owner" and invite a double-spawn.
fn write_state(data_dir: &Path, state: &RuntimeState) {
    let Ok(payload) = serde_json::to_vec_pretty(state) else {
        return;
    };
    let target = state_path(data_dir);
    let tmp = target.with_extension("json.tmp");
    if std::fs::create_dir_all(data_dir).is_err() {
        return;
    }
    if std::fs::write(&tmp, &payload).is_ok() {
        let _ = std::fs::rename(&tmp, &target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(pid: u32, updated_at_ms: u64) -> RuntimeState {
        RuntimeState {
            pid,
            phase: Phase::RecoveringStorage,
            started_at_ms: 0,
            updated_at_ms,
            heartbeat_interval_ms: HEARTBEAT_INTERVAL_MS,
            config_path: None,
            version: None,
        }
    }

    /// C1: the record appears before any slow work, carrying pid + phase.
    #[test]
    fn writer_publishes_record_immediately() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let writer = RuntimeStateWriter::start(tmp.path(), Some("/cfg.toml".into()));
        let read = read_state(tmp.path()).expect("state file must exist right after start");
        assert_eq!(read.pid, std::process::id());
        assert_eq!(read.phase, Phase::Starting);
        assert_eq!(read.config_path.as_deref(), Some("/cfg.toml"));
        assert!(!read.phase.is_ready(), "Starting must not read as ready");
        drop(writer);
    }

    /// C2: phase transitions are observable, and only Serving is ready.
    #[test]
    fn phase_transitions_are_observable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let writer = RuntimeStateWriter::start(tmp.path(), None);
        for phase in [
            Phase::RecoveringStorage,
            Phase::RecoveringGraphs,
            Phase::Binding,
        ] {
            writer.set_phase(phase);
            let read = read_state(tmp.path()).expect("state");
            assert_eq!(read.phase, phase);
            assert!(!read.phase.is_ready(), "{phase:?} must not read as ready");
        }
        writer.set_phase(Phase::Serving);
        assert!(
            read_state(tmp.path()).expect("state").phase.is_ready(),
            "Serving is the ready phase"
        );
    }

    /// C3: a clean exit removes the record — absence means "no owner".
    #[test]
    fn finish_removes_the_record() {
        let tmp = tempfile::tempdir().expect("tempdir");
        {
            let _writer = RuntimeStateWriter::start(tmp.path(), None);
            assert!(read_state(tmp.path()).is_some());
        } // drop
        assert!(
            read_state(tmp.path()).is_none(),
            "dropping the writer must clear ownership"
        );
    }

    /// C4: staleness is decidable — a dead pid is stale regardless of how
    /// fresh its last heartbeat looks.
    #[test]
    fn dead_pid_is_stale_even_with_fresh_heartbeat() {
        // PID 0 is never a live user process on the platforms we target;
        // kill(0, 0) addresses the caller's process group, so use a pid that
        // is guaranteed absent instead.
        let dead_pid = 0x7FFF_FFFF_u32;
        let now = 1_000_000;
        let state = state_with(dead_pid, now); // heartbeat "just now"
        assert!(
            state.is_stale(now),
            "a vanished process cannot own the data dir"
        );
        assert!(!state.is_live_owner(now));
    }

    /// C4b: a live pid whose heartbeat stopped advancing is stale — this is
    /// the wedged-process case that a pid check alone would miss.
    #[test]
    fn live_pid_with_frozen_heartbeat_is_stale() {
        let now = 10_000_000;
        let ancient = now - HEARTBEAT_INTERVAL_MS * (STALE_AFTER_INTERVALS + 5);
        let state = state_with(std::process::id(), ancient);
        assert!(state.is_stale(now), "frozen heartbeat ⇒ stale");
    }

    /// C4c: the live case — our own pid with a fresh beat is a live owner.
    /// This is what stops a supervisor from double-spawning onto a data dir
    /// another process is actively serving (victor#911).
    #[test]
    fn live_pid_with_fresh_heartbeat_is_a_live_owner() {
        let now = 10_000_000;
        let state = state_with(std::process::id(), now - HEARTBEAT_INTERVAL_MS);
        assert!(state.is_live_owner(now));
        assert!(!state.is_stale(now));
    }

    /// C4d: tolerance is generous — a couple of missed beats (a paused or
    /// heavily loaded process) must NOT read as dead, or supervisors will
    /// kill healthy servers, which is the bug this contract exists to end.
    #[test]
    fn a_few_missed_beats_are_not_stale() {
        let now = 10_000_000;
        let state = state_with(std::process::id(), now - HEARTBEAT_INTERVAL_MS * 3);
        assert!(
            !state.is_stale(now),
            "3 missed beats must be tolerated, not fatal"
        );
    }

    /// C5: the heartbeat actually advances while the process lives — the
    /// property a supervisor waits on for an unbounded recovery.
    #[test]
    fn heartbeat_advances_while_alive() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let writer = RuntimeStateWriter::start(tmp.path(), None);
        let first = read_state(tmp.path()).expect("state").updated_at_ms;
        std::thread::sleep(std::time::Duration::from_millis(HEARTBEAT_INTERVAL_MS * 2));
        let second = read_state(tmp.path()).expect("state").updated_at_ms;
        assert!(
            second > first,
            "heartbeat must advance ({first} -> {second}); without this a \
             supervisor cannot distinguish slow recovery from a hang"
        );
        drop(writer);
    }

    /// C6: a corrupt record reads as "no owner" and never blocks startup.
    #[test]
    fn corrupt_record_reads_as_no_owner() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(state_path(tmp.path()), b"{not json").expect("write");
        assert!(read_state(tmp.path()).is_none());
    }

    /// C7: records are field-additive — an older reader ignores fields a
    /// newer server adds, so the contract can evolve without a flag day.
    #[test]
    fn unknown_fields_are_ignored_by_readers() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = format!(
            r#"{{"pid":{},"phase":"serving","started_at_ms":1,"updated_at_ms":2,
                 "heartbeat_interval_ms":2000,"future_field":{{"nested":true}}}}"#,
            std::process::id()
        );
        std::fs::write(state_path(tmp.path()), json).expect("write");
        let read = read_state(tmp.path()).expect("forward-compatible read");
        assert_eq!(read.phase, Phase::Serving);
    }
}
