//! PyO3 bindings for `proximadb-queue`. Builds when `--features python`
//! is set. Provides a sync Python API by bridging Rust's async via a
//! per-process tokio Runtime — Python callers don't see futures.
//!
//! ## Module shape
//!
//! Importing the wheel `proximadb_queue_embedded._native` gives:
//!
//! ```python
//! from proximadb_queue_embedded._native import (
//!     QueueClient, Producer, Consumer, Message, MessageReceipt,
//!     partition_for,
//! )
//!
//! client = QueueClient(root="/var/lib/proximadb/queue",
//!                      topics={"embed-ingest": {"partition_count": 4}})
//! producer = client.producer()
//! producer.send("embed-ingest", "tenant-a", b"...")
//! consumer = client.consumer("g")
//! consumer.subscribe("embed-ingest", [0, 1, 2, 3])
//! batch = consumer.poll(max_batch=32, max_wait_ms=200)
//! consumer.ack([m.message_id for m in batch])
//! ```
//!
//! The Python module name is `_native` — convention used by the
//! existing `proximadb_embedded` package. The user-facing
//! `proximadb_queue_embedded` Python module re-exports these with
//! ergonomic wrappers.

use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::runtime::Runtime;

use crate::message::{Message as RsMessage, MessageId as RsMessageId};
use crate::{
    Consumer as RsConsumer, Producer as RsProducer, QueueClient as RsQueueClient, QueueConfig,
    QueueError, TopicConfig, partition_for as rs_partition_for,
};

/// Single shared tokio runtime for all PyO3-driven async calls. Lazily
/// initialized on first use. Multi-thread so block_on doesn't starve.
static RT: OnceCell<Runtime> = OnceCell::new();

fn rt() -> &'static Runtime {
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("proximadb-queue-pyo3")
            .build()
            .expect("proximadb-queue PyO3 runtime init")
    })
}

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass(name = "QueueClient", module = "proximadb_queue_embedded._native")]
pub struct PyQueueClient {
    inner: Arc<RsQueueClient>,
}

#[pymethods]
impl PyQueueClient {
    /// Open (or recover) a queue at `root`. `topics` is an optional
    /// `{topic_name: {"partition_count": int, ...}}` dict — keys
    /// missing from the dict take TopicConfig defaults.
    #[new]
    #[pyo3(signature = (root, topics=None))]
    fn new(root: String, topics: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut config = QueueConfig {
            root,
            ..QueueConfig::default()
        };
        if let Some(t) = topics {
            for (k, v) in t.iter() {
                let name: String = k.extract()?;
                let dict: &Bound<PyDict> = v.cast()?;
                let mut topic_cfg = TopicConfig::default();
                if let Ok(Some(pc)) = dict.get_item("partition_count") {
                    topic_cfg.partition_count = pc.extract()?;
                }
                if let Ok(Some(cap)) = dict.get_item("memory_capacity") {
                    topic_cfg.memory_capacity = cap.extract()?;
                }
                if let Ok(Some(rot)) = dict.get_item("disk_rotation_size_mb") {
                    topic_cfg.disk_rotation_size_mb = rot.extract()?;
                }
                if let Ok(Some(lease_ms)) = dict.get_item("lease_duration_ms") {
                    let ms: u64 = lease_ms.extract()?;
                    topic_cfg.lease_duration = Duration::from_millis(ms);
                }
                config.topics.insert(name, topic_cfg);
            }
        }
        let inner = rt().block_on(RsQueueClient::open(config)).map_err(err)?;
        Ok(Self { inner })
    }

    fn producer(&self) -> PyProducer {
        PyProducer {
            inner: self.inner.producer(),
        }
    }

    fn consumer(&self, group_id: String) -> PyConsumer {
        PyConsumer {
            inner: self.inner.consumer(group_id),
        }
    }

    fn shutdown(&self) -> PyResult<()> {
        rt().block_on(self.inner.shutdown()).map_err(err)
    }
}

#[pyclass(name = "Producer", module = "proximadb_queue_embedded._native")]
pub struct PyProducer {
    inner: RsProducer,
}

#[pymethods]
impl PyProducer {
    /// Send a single message. Blocks until durable per the topic's
    /// sync_mode (Strict: fsync; Lazy: memory append). Returns the
    /// receipt.
    #[pyo3(signature = (topic, tenant_id, payload))]
    fn send(
        &self,
        py: Python<'_>,
        topic: String,
        tenant_id: String,
        payload: &Bound<'_, PyBytes>,
    ) -> PyResult<PyMessageReceipt> {
        let payload_bytes = payload.as_bytes().to_vec();
        let msg = RsMessage::new(topic, tenant_id, payload_bytes);
        let receipt = py
            .detach(|| rt().block_on(self.inner.send(msg)))
            .map_err(err)?;
        Ok(PyMessageReceipt {
            message_id: receipt.message_id.0,
            partition: receipt.partition,
            offset: receipt.offset,
            fsynced: receipt.fsynced_at.is_some(),
        })
    }
}

#[pyclass(name = "Consumer", module = "proximadb_queue_embedded._native")]
pub struct PyConsumer {
    inner: RsConsumer,
}

#[pymethods]
impl PyConsumer {
    fn subscribe(&self, topic: String, partitions: Vec<u32>) -> PyResult<()> {
        rt().block_on(self.inner.subscribe(&topic, &partitions))
            .map_err(err)
    }

    /// Poll up to `max_batch` messages, waiting up to `max_wait_ms` for
    /// at least one. Returns a list of [`PyMessage`].
    #[pyo3(signature = (max_batch, max_wait_ms))]
    fn poll(&self, py: Python<'_>, max_batch: usize, max_wait_ms: u64) -> PyResult<Vec<PyMessage>> {
        let wait = Duration::from_millis(max_wait_ms);
        let msgs = py
            .detach(|| rt().block_on(self.inner.poll(max_batch, wait)))
            .map_err(err)?;
        Ok(msgs
            .into_iter()
            .map(|delivery| PyMessage {
                message_id: delivery.message_id.0,
                topic: delivery.message.topic,
                tenant_id: delivery.message.tenant_id,
                payload: delivery.message.payload,
                attempt_count: delivery.message.attempt_count,
            })
            .collect())
    }

    fn ack(&self, py: Python<'_>, message_ids: Vec<String>) -> PyResult<()> {
        let ids: Vec<RsMessageId> = message_ids.into_iter().map(RsMessageId).collect();
        py.detach(|| rt().block_on(self.inner.ack(&ids)))
            .map_err(err)
    }

    fn nack(&self, py: Python<'_>, message_ids: Vec<String>) -> PyResult<()> {
        let ids: Vec<RsMessageId> = message_ids.into_iter().map(RsMessageId).collect();
        py.detach(|| rt().block_on(self.inner.nack(&ids)))
            .map_err(err)
    }
}

#[pyclass(
    name = "Message",
    module = "proximadb_queue_embedded._native",
    from_py_object
)]
#[derive(Clone)]
pub struct PyMessage {
    #[pyo3(get)]
    pub message_id: String,
    #[pyo3(get)]
    pub topic: String,
    #[pyo3(get)]
    pub tenant_id: String,
    #[pyo3(get)]
    pub payload: Vec<u8>,
    #[pyo3(get)]
    pub attempt_count: u32,
}

#[pyclass(
    name = "MessageReceipt",
    module = "proximadb_queue_embedded._native",
    from_py_object
)]
#[derive(Clone)]
pub struct PyMessageReceipt {
    #[pyo3(get)]
    pub message_id: String,
    #[pyo3(get)]
    pub partition: u32,
    #[pyo3(get)]
    pub offset: u64,
    /// `True` when the topic's sync_mode is Strict and the producer
    /// waited on the group-commit fsync barrier. `False` for Lazy mode.
    #[pyo3(get)]
    pub fsynced: bool,
}

#[pyfunction]
fn partition_for(tenant_id: &str, partition_count: u32) -> PyResult<u32> {
    if partition_count == 0 {
        return Err(PyValueError::new_err("partition_count must be > 0"));
    }
    Ok(rs_partition_for(tenant_id, partition_count))
}

/// Module init. The maturin pyproject points at this symbol via the
/// `module-name = "proximadb_queue_embedded._native"` setting.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQueueClient>()?;
    m.add_class::<PyProducer>()?;
    m.add_class::<PyConsumer>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyMessageReceipt>()?;
    m.add_function(wrap_pyfunction!(partition_for, m)?)?;
    let _ = QueueError::TopicNotFound("".to_string()); // ensure imported
    Ok(())
}
