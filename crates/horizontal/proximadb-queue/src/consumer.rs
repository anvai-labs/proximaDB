//! Consumer ownership and contiguous progress over the existing queue formats.
//! One non-reused holder per Consumer; clones share it. Loss/cancellation is
//! terminal: construct a new Consumer to reacquire, never reuse old deliveries.
//! Durable ACK is guarded by the admitted lease bytes, not a read-then-write check.

use crate::QueueClient;
use crate::error::{QueueError, Result};
use crate::memory_tier::MemoryEntry;
use crate::message::{Delivery, MessageId};
use crate::topic::PartitionId;
use std::collections::HashSet;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

#[derive(Clone)]
pub struct Consumer {
    pub(crate) inner: Arc<ConsumerInner>,
}

pub(crate) struct ConsumerInner {
    client: Arc<QueueClient>,
    group_id: String,
    holder_id: String,
    usable: Arc<AtomicBool>,
    operation: Mutex<()>,
    subscriptions: std::sync::Mutex<Vec<Arc<Subscription>>>,
    renewers: Mutex<Vec<RenewerHandle>>,
}
struct RenewerHandle {
    shutdown: Option<oneshot::Sender<()>>,
    join: JoinHandle<Result<()>>,
}
struct Subscription {
    topic: String,
    partition: PartitionId,
    tracker: Mutex<InFlight>,
}
struct Pending {
    memo: MemoryEntry,
    acknowledged: bool,
    retry: bool,
}
struct InFlight {
    pending: Vec<Pending>,
    next_read_offset: u64,
}

/// A cancelled async operation can leave blocking publication in progress.
/// Fail this Consumer closed before releasing its logical mutex; later operations
/// must not report a NACK/retry while that uncertain ACK may still publish.
struct CancellationGuard {
    usable: Arc<AtomicBool>,
    armed: bool,
}
impl CancellationGuard {
    fn new(usable: &Arc<AtomicBool>) -> Self {
        Self {
            usable: usable.clone(),
            armed: true,
        }
    }
    fn complete(&mut self) {
        self.armed = false;
    }
}
impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.usable.store(false, Ordering::Release);
        }
    }
}

impl Consumer {
    pub(crate) fn new(client: Arc<QueueClient>, group_id: String) -> Self {
        let holder_id = format!("{}-{}", client.instance_id(), uuid::Uuid::new_v4());
        Self {
            inner: Arc::new(ConsumerInner {
                client,
                group_id,
                holder_id,
                usable: Arc::new(AtomicBool::new(true)),
                operation: Mutex::new(()),
                subscriptions: std::sync::Mutex::new(Vec::new()),
                renewers: Mutex::new(Vec::new()),
            }),
        }
    }
    pub fn group_id(&self) -> &str {
        &self.inner.group_id
    }
    fn check_usable(&self) -> Result<()> {
        if self.inner.usable.load(Ordering::Acquire) && !self.inner.client.is_closed() {
            Ok(())
        } else {
            Err(QueueError::Persistence(
                "consumer ownership lost, closed or cancelled; create a new consumer".into(),
            ))
        }
    }
    fn subscriptions(&self) -> Result<Vec<Arc<Subscription>>> {
        self.inner
            .subscriptions
            .lock()
            .map(|s| s.clone())
            .map_err(|_| QueueError::Persistence("consumer subscriptions poisoned".into()))
    }
    async fn validate_owner(&self, sub: &Subscription) -> Result<()> {
        self.check_usable()?;
        let result = crate::leases::owned_bytes(
            self.inner.client.fs(),
            self.inner.client.root_path(),
            &sub.topic,
            sub.partition,
            &self.inner.group_id,
            &self.inner.holder_id,
        )
        .await;
        if result.is_err() {
            self.inner.usable.store(false, Ordering::Release);
        }
        result.map(|_| ())
    }

    pub async fn subscribe_all(&self, topic: &str) -> Result<()> {
        let _operation = self.inner.operation.lock().await;
        self.check_usable()?;
        crate::offset_store::validate_group(&self.inner.group_id)?;
        self.validate_topic(topic)?;
        let mut cancellation = CancellationGuard::new(&self.inner.usable);
        let result = self.subscribe_all_locked(topic).await;
        if result.is_err() {
            self.inner.usable.store(false, Ordering::Release);
        }
        cancellation.complete();
        result
    }

    fn validate_topic(&self, topic: &str) -> Result<()> {
        if self.subscriptions()?.iter().any(|s| s.topic != topic) {
            return Err(QueueError::Persistence(
                "MessageId has no topic identity; use one topic per consumer".into(),
            ));
        }
        Ok(())
    }

    async fn subscribe_all_locked(&self, topic: &str) -> Result<()> {
        let state = self
            .inner
            .client
            .ensure_topic_async(topic, Default::default())
            .await?;
        let partitions: Vec<_> = (0..state.memory.len() as PartitionId).collect();
        self.subscribe_locked(topic, &partitions).await
    }

    /// Idempotent for an already-owned subscription, never reacquires after loss.
    /// Resume at inclusive committed offset + 1; absence starts at zero.
    pub async fn subscribe(&self, topic: &str, partitions: &[PartitionId]) -> Result<()> {
        let _operation = self.inner.operation.lock().await;
        self.check_usable()?;
        crate::offset_store::validate_group(&self.inner.group_id)?;
        self.validate_topic(topic)?;
        let mut cancellation = CancellationGuard::new(&self.inner.usable);
        let result = self.subscribe_locked(topic, partitions).await;
        if result.is_err() {
            self.inner.usable.store(false, Ordering::Release);
        }
        cancellation.complete();
        result
    }

    async fn subscribe_locked(&self, topic: &str, partitions: &[PartitionId]) -> Result<()> {
        let state = self
            .inner
            .client
            .ensure_topic_async(topic, Default::default())
            .await?;
        let ttl = state.config.lease_duration;
        if ttl < Duration::from_nanos(2) {
            return Err(QueueError::Persistence(
                "consumer lease interval must be positive".into(),
            ));
        }
        for &p in partitions {
            if p as usize >= state.memory.len() {
                return Err(QueueError::PartitionNotFound {
                    topic: topic.into(),
                    partition: p,
                });
            }
        }
        for &p in partitions {
            if let Some(sub) = self
                .subscriptions()?
                .iter()
                .find(|s| s.topic == topic && s.partition == p)
            {
                self.validate_owner(sub).await?;
                continue;
            }
            let fs = self.inner.client.fs();
            let root = self.inner.client.root_path();
            crate::leases::try_acquire(
                fs,
                root,
                topic,
                p,
                &self.inner.group_id,
                &self.inner.holder_id,
                ttl,
            )
            .await?;
            let offset = crate::offset_store::read(fs, root, topic, p, &self.inner.group_id).await;
            let next = match offset.and_then(|o| match o {
                Some(o) => o
                    .checked_add(1)
                    .ok_or_else(|| QueueError::Persistence("consumer offset overflow".into())),
                None => Ok(0),
            }) {
                Ok(next) => next,
                Err(error) => {
                    let _ = crate::leases::release(
                        fs,
                        root,
                        topic,
                        p,
                        &self.inner.group_id,
                        &self.inner.holder_id,
                    )
                    .await;
                    return Err(error);
                }
            };
            let sub = Arc::new(Subscription {
                topic: topic.into(),
                partition: p,
                tracker: Mutex::new(InFlight {
                    pending: Vec::new(),
                    next_read_offset: next,
                }),
            });
            self.inner
                .subscriptions
                .lock()
                .map_err(|_| QueueError::Persistence("consumer subscriptions poisoned".into()))?
                .push(sub.clone());
            let (tx, mut rx) = oneshot::channel();
            let fs = fs.clone();
            let root = root.clone();
            let group = self.inner.group_id.clone();
            let holder = self.inner.holder_id.clone();
            let usable = self.inner.usable.clone();
            let join = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = &mut rx => break,
                        _ = tokio::time::sleep(ttl / 2) => {
                            if !usable.load(Ordering::Acquire) { break; }
                            // ACK and renewal from this consumer share logical serialization.
                            let _tracker = sub.tracker.lock().await;
                            if let Err(error) = crate::leases::renew(&fs, &root, &sub.topic, p, &group, &holder, ttl).await {
                                usable.store(false, Ordering::Release);
                                tracing::warn!(%error, "consumer renewal failed; ownership is terminal");
                                break;
                            }
                        }
                    }
                }
                let _tracker = sub.tracker.lock().await;
                crate::leases::release(&fs, &root, &sub.topic, p, &group, &holder).await
            });
            self.inner.renewers.lock().await.push(RenewerHandle {
                shutdown: Some(tx),
                join,
            });
        }
        Ok(())
    }

    /// Poll is authoritative ownership admission, not fencing of external effects.
    /// One lease read per polled partition/batch; no per-message authority GET.
    pub async fn poll(&self, max_batch: usize, max_wait: Duration) -> Result<Vec<Delivery>> {
        let deadline = tokio::time::Instant::now()
            .checked_add(max_wait)
            .ok_or_else(|| QueueError::Persistence("poll deadline overflow".into()))?;
        loop {
            let out = {
                let _operation = self.inner.operation.lock().await;
                self.check_usable()?;
                let mut cancellation = CancellationGuard::new(&self.inner.usable);
                let result = self.poll_locked(max_batch).await;
                cancellation.complete();
                result?
            };
            if !out.is_empty() || max_batch == 0 || tokio::time::Instant::now() >= deadline {
                return Ok(out);
            }
            let subscriptions = self.subscriptions()?;
            let Some(sub) = subscriptions.first() else {
                return Ok(out);
            };
            let Some(state) = self.inner.client.topic_state(&sub.topic).await else {
                return Ok(out);
            };
            let Some(part) = state.memory.get(sub.partition as usize) else {
                return Ok(out);
            };
            let _ = tokio::time::timeout_at(deadline, part.notify.notified()).await;
        }
    }
    async fn poll_locked(&self, max_batch: usize) -> Result<Vec<Delivery>> {
        let mut out = Vec::with_capacity(max_batch);
        for sub in self.subscriptions()? {
            if out.len() == max_batch {
                break;
            }
            let mut tracker = sub.tracker.lock().await;
            self.validate_owner(&sub).await?;
            for pending in tracker.pending.iter_mut().filter(|p| p.retry) {
                if out.len() == max_batch {
                    break;
                }
                pending.memo.message.attempt_count = pending
                    .memo
                    .message
                    .attempt_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        QueueError::Persistence("delivery attempt count overflow".into())
                    })?;
                pending.retry = false;
                out.push(Delivery {
                    message_id: pending.memo.message_id.clone(),
                    message: pending.memo.message.clone(),
                });
            }
            let Some(state) = self.inner.client.topic_state(&sub.topic).await else {
                continue;
            };
            let Some(part) = state.memory.get(sub.partition as usize) else {
                continue;
            };
            let batch = part
                .read_from(tracker.next_read_offset, max_batch - out.len())
                .await;
            for memo in batch {
                tracker.next_read_offset = memo
                    .offset
                    .checked_add(1)
                    .ok_or_else(|| QueueError::Persistence("delivery offset overflow".into()))?;
                out.push(Delivery {
                    message_id: memo.message_id.clone(),
                    message: memo.message.clone(),
                });
                tracker.pending.push(Pending {
                    memo,
                    acknowledged: false,
                    retry: false,
                });
            }
        }
        Ok(out)
    }

    /// Advance only a contiguous acknowledged prefix. Pending state changes only
    /// after guarded durable publication succeeds; a failed write remains retryable.
    pub async fn ack(&self, message_ids: &[MessageId]) -> Result<()> {
        let _operation = self.inner.operation.lock().await;
        self.check_usable()?;
        let mut cancellation = CancellationGuard::new(&self.inner.usable);
        let result = self.ack_locked(message_ids).await;
        if matches!(
            result,
            Err(QueueError::LeaseConflict { .. } | QueueError::PublicationIndeterminate(_))
        ) {
            self.inner.usable.store(false, Ordering::Release);
        }
        cancellation.complete();
        result
    }
    async fn ack_locked(&self, ids: &[MessageId]) -> Result<()> {
        let targets: HashSet<_> = ids.iter().collect();
        let subscriptions = self.subscriptions()?;
        for sub in subscriptions {
            let mut tracker = sub.tracker.lock().await;
            if !tracker
                .pending
                .iter()
                .any(|p| targets.contains(&p.memo.message_id))
            {
                continue;
            }
            self.validate_owner(&sub).await?;
            let prefix = tracker
                .pending
                .iter()
                .take_while(|p| p.acknowledged || targets.contains(&p.memo.message_id))
                .count();
            if prefix > 0 {
                let offset = tracker.pending[prefix - 1].memo.offset;
                crate::offset_store::commit(
                    self.inner.client.fs(),
                    self.inner.client.root_path(),
                    &sub.topic,
                    sub.partition,
                    &self.inner.group_id,
                    &self.inner.holder_id,
                    offset,
                )
                .await?;
            }
            for pending in &mut tracker.pending {
                if targets.contains(&pending.memo.message_id) {
                    pending.acknowledged = true;
                    pending.retry = false;
                }
            }
            tracker.pending.drain(..prefix);
        }
        Ok(())
    }

    /// Retry pending deliveries without advancing durable progress. No implicit
    /// drop or DLQ: those need a separate durable side-effect protocol.
    pub async fn nack(&self, ids: &[MessageId]) -> Result<()> {
        let _operation = self.inner.operation.lock().await;
        self.check_usable()?;
        let mut cancellation = CancellationGuard::new(&self.inner.usable);
        let result = self.nack_locked(ids).await;
        cancellation.complete();
        result
    }
    async fn nack_locked(&self, ids: &[MessageId]) -> Result<()> {
        let targets: HashSet<_> = ids.iter().collect();
        let subscriptions = self.subscriptions()?;
        for sub in subscriptions {
            let mut tracker = sub.tracker.lock().await;
            if !tracker
                .pending
                .iter()
                .any(|p| targets.contains(&p.memo.message_id))
            {
                continue;
            }
            self.validate_owner(&sub).await?;
            for pending in &mut tracker.pending {
                if targets.contains(&pending.memo.message_id) {
                    pending.acknowledged = false;
                    pending.retry = true;
                }
            }
        }
        Ok(())
    }

    /// Stop renewers and await owner-conditioned release. Clones share closure.
    /// Cancellation is terminal; Drop still signals remaining tasks best effort.
    pub async fn shutdown(&self) -> Result<()> {
        let _operation = self.inner.operation.lock().await;
        self.inner.usable.store(false, Ordering::Release);
        let mut handles = self.inner.renewers.lock().await;
        for handle in handles.iter_mut() {
            if let Some(shutdown) = handle.shutdown.take() {
                let _ = shutdown.send(());
            }
        }
        let mut error = None;
        while let Some(handle) = handles.last_mut() {
            // Keep the JoinHandle stored while awaiting so cancellation of
            // shutdown cannot detach unfinished release from a later shutdown.
            match (&mut handle.join).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error.get_or_insert(e);
                }
                Err(e) => {
                    error.get_or_insert(QueueError::Persistence(format!(
                        "consumer renewal task: {e}"
                    )));
                }
            }
            handles.pop();
        }
        error.map_or(Ok(()), Err)
    }
}
impl Drop for ConsumerInner {
    fn drop(&mut self) {
        self.usable.store(false, Ordering::Release);
        // Dropping the owned senders wakes each renewer; it conditionally releases
        // only this holder. Runtime exit may abort it, so expiry remains necessary.
    }
}
