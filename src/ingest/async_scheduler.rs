//! Priority-aware ingest scheduler primitives.

#![allow(missing_docs)]

use std::collections::VecDeque;

use tokio::sync::{Mutex, Notify};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IngestPriority {
    P0,
    P1,
    P2,
    P3,
    P4,
}

impl IngestPriority {
    pub const COUNT: usize = 5;

    pub const fn as_index(self) -> usize {
        match self {
            Self::P0 => 0,
            Self::P1 => 1,
            Self::P2 => 2,
            Self::P3 => 3,
            Self::P4 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKind {
    SyncCommit,
    FreshnessSla,
    HotCompaction,
    StatsRefresh,
    ReEmbed,
    Custom(String),
}

impl TaskKind {
    pub const fn default_priority(&self) -> IngestPriority {
        match self {
            Self::SyncCommit => IngestPriority::P0,
            Self::FreshnessSla => IngestPriority::P1,
            Self::HotCompaction => IngestPriority::P2,
            Self::StatsRefresh => IngestPriority::P3,
            Self::ReEmbed | Self::Custom(_) => IngestPriority::P4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestTask {
    pub id: String,
    pub tenant_id: Option<String>,
    pub collection: String,
    pub kind: TaskKind,
    pub priority: IngestPriority,
    pub created_at_ms: u64,
    pub payload_bytes: u64,
}

impl IngestTask {
    pub fn new(id: impl Into<String>, collection: impl Into<String>, kind: TaskKind) -> Self {
        let priority = kind.default_priority();
        Self {
            id: id.into(),
            tenant_id: None,
            collection: collection.into(),
            kind,
            priority,
            // created_at_ms is u64 while the shared kernel clock is i64; the value is
            // epoch millis (never negative), so the widening cast is lossless.
            created_at_ms: proximadb_kernel::time::now_unix_ms() as u64,
            payload_bytes: 0,
        }
    }

    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_priority(mut self, priority: IngestPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_payload_bytes(mut self, payload_bytes: u64) -> Self {
        self.payload_bytes = payload_bytes;
        self
    }
}

/// Backwards-compat alias for [`IngestSchedulerConfig`].
pub type SchedulerConfig = IngestSchedulerConfig;

#[derive(Debug, Clone)]
pub struct IngestSchedulerConfig {
    pub max_depth: usize,
    pub fairness_window: usize,
}

impl Default for IngestSchedulerConfig {
    fn default() -> Self {
        Self {
            max_depth: 65_536,
            fairness_window: 32,
        }
    }
}

/// Backwards-compat alias for [`IngestSchedulerStats`].
pub type SchedulerStats = IngestSchedulerStats;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestSchedulerStats {
    pub queued_by_priority: [usize; IngestPriority::COUNT],
    pub total_queued: usize,
    pub total_enqueued: u64,
    pub total_dequeued: u64,
    pub rejected_full: u64,
}

#[derive(Debug)]
struct QueueState {
    lanes: [VecDeque<IngestTask>; IngestPriority::COUNT],
    total_enqueued: u64,
    total_dequeued: u64,
    rejected_full: u64,
    high_priority_run: usize,
}

impl Default for QueueState {
    fn default() -> Self {
        Self {
            lanes: std::array::from_fn(|_| VecDeque::new()),
            total_enqueued: 0,
            total_dequeued: 0,
            rejected_full: 0,
            high_priority_run: 0,
        }
    }
}

#[derive(Debug)]
pub struct IngestQueue {
    config: IngestSchedulerConfig,
    state: Mutex<QueueState>,
    notify: Notify,
}

impl Default for IngestQueue {
    fn default() -> Self {
        Self::new(IngestSchedulerConfig::default())
    }
}

impl IngestQueue {
    pub fn new(config: IngestSchedulerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(QueueState::default()),
            notify: Notify::new(),
        }
    }

    pub async fn push(&self, task: IngestTask) -> Result<(), IngestTask> {
        let mut state = self.state.lock().await;
        if queued_len(&state) >= self.config.max_depth {
            state.rejected_full = state.rejected_full.saturating_add(1);
            return Err(task);
        }
        state.total_enqueued = state.total_enqueued.saturating_add(1);
        state.lanes[task.priority.as_index()].push_back(task);
        drop(state);
        self.notify.notify_one();
        Ok(())
    }

    pub async fn pop(&self) -> IngestTask {
        loop {
            if let Some(task) = self.try_pop().await {
                return task;
            }
            self.notify.notified().await;
        }
    }

    pub async fn try_pop(&self) -> Option<IngestTask> {
        let mut state = self.state.lock().await;
        let task = choose_next(&mut state, self.config.fairness_window)?;
        state.total_dequeued = state.total_dequeued.saturating_add(1);
        Some(task)
    }

    pub async fn stats(&self) -> IngestSchedulerStats {
        let state = self.state.lock().await;
        let mut queued_by_priority = [0; IngestPriority::COUNT];
        for (idx, lane) in state.lanes.iter().enumerate() {
            queued_by_priority[idx] = lane.len();
        }
        IngestSchedulerStats {
            queued_by_priority,
            total_queued: queued_by_priority.iter().sum(),
            total_enqueued: state.total_enqueued,
            total_dequeued: state.total_dequeued,
            rejected_full: state.rejected_full,
        }
    }
}

fn choose_next(state: &mut QueueState, fairness_window: usize) -> Option<IngestTask> {
    let fairness_window = fairness_window.max(1);
    if state.high_priority_run >= fairness_window {
        for idx in 2..IngestPriority::COUNT {
            if let Some(task) = state.lanes[idx].pop_front() {
                state.high_priority_run = 0;
                return Some(task);
            }
        }
        state.high_priority_run = 0;
    }

    for idx in 0..IngestPriority::COUNT {
        if let Some(task) = state.lanes[idx].pop_front() {
            if idx <= 1 {
                state.high_priority_run = state.high_priority_run.saturating_add(1);
            } else {
                state.high_priority_run = 0;
            }
            return Some(task);
        }
    }
    None
}

fn queued_len(state: &QueueState) -> usize {
    state.lanes.iter().map(VecDeque::len).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn task_kind_maps_to_expected_priority() {
        assert_eq!(TaskKind::SyncCommit.default_priority(), IngestPriority::P0);
        assert_eq!(
            TaskKind::FreshnessSla.default_priority(),
            IngestPriority::P1
        );
        assert_eq!(
            TaskKind::StatsRefresh.default_priority(),
            IngestPriority::P3
        );
        assert_eq!(TaskKind::ReEmbed.default_priority(), IngestPriority::P4);
    }

    #[tokio::test]
    async fn queue_pops_highest_priority_first() {
        let queue = IngestQueue::default();
        queue
            .push(IngestTask::new("p4", "c", TaskKind::ReEmbed))
            .await
            .unwrap();
        queue
            .push(IngestTask::new("p0", "c", TaskKind::SyncCommit))
            .await
            .unwrap();

        assert_eq!(queue.pop().await.id, "p0");
        assert_eq!(queue.pop().await.id, "p4");
    }

    #[tokio::test]
    async fn fairness_window_allows_lower_lane_to_drain() {
        let queue = IngestQueue::new(IngestSchedulerConfig {
            max_depth: 100,
            fairness_window: 2,
        });
        for i in 0..3 {
            queue
                .push(IngestTask::new(
                    format!("p0-{i}"),
                    "c",
                    TaskKind::SyncCommit,
                ))
                .await
                .unwrap();
        }
        queue
            .push(IngestTask::new("p3", "c", TaskKind::StatsRefresh))
            .await
            .unwrap();

        assert_eq!(queue.pop().await.id, "p0-0");
        assert_eq!(queue.pop().await.id, "p0-1");
        assert_eq!(queue.pop().await.id, "p3");
    }

    #[tokio::test]
    async fn bounded_queue_rejects_when_full() {
        let queue = IngestQueue::new(IngestSchedulerConfig {
            max_depth: 1,
            fairness_window: 1,
        });
        queue
            .push(IngestTask::new("ok", "c", TaskKind::SyncCommit))
            .await
            .unwrap();
        assert!(
            queue
                .push(IngestTask::new("reject", "c", TaskKind::SyncCommit))
                .await
                .is_err()
        );
        assert_eq!(queue.stats().await.rejected_full, 1);
    }
}
