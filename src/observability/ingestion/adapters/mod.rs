//! Input adapters for observability ingestion
//!
//! Supports multiple input formats and protocols:
//! - OTLP (OpenTelemetry Protocol) - gRPC and HTTP
//! - Syslog (RFC 3164/5424) - TCP and UDP
//! - Fluent (Fluent Bit/Fluentd forward protocol)
//! - CEF (Common Event Format) - ArcSight
//! - LEEF (Log Event Extended Format) - IBM QRadar
//! - OCSF (Open Cybersecurity Schema Framework)
//! - HTTP (JSON over HTTP)

/// CEF and LEEF format adapter for ArcSight and IBM QRadar
pub mod cef_leef;

/// Fluent Bit/Fluentd forward protocol adapter
pub mod fluent;

/// HTTP JSON ingestion adapter
pub mod http;

/// OCSF (Open Cybersecurity Schema Framework) adapter
pub mod ocsf;

/// OpenTelemetry Protocol HTTP adapter
pub mod otlp;

/// OpenTelemetry Protocol gRPC adapter
pub mod otlp_grpc;

/// Syslog (RFC 3164/5424) TCP/UDP adapter
pub mod syslog;

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::mpsc;

use crate::proto::proximadb_v1::LogEntry;

/// Trait for input adapters
#[async_trait]
pub trait InputAdapter: Send + Sync {
    /// Get the adapter name
    fn name(&self) -> &str;

    /// Start the adapter
    async fn start(&self) -> Result<()>;

    /// Stop the adapter
    async fn stop(&self) -> Result<()>;

    /// Check if the adapter is running
    fn is_running(&self) -> bool;

    /// Get the number of events received
    fn events_received(&self) -> u64;
}

/// Configuration for adapters
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Bind address for network adapters
    pub bind_address: SocketAddr,
    /// Maximum batch size for processing
    pub batch_size: usize,
    /// Buffer size for incoming events
    pub buffer_size: usize,
    /// Channel for sending parsed events
    pub sender: mpsc::Sender<Vec<LogEntry>>,
}

impl AdapterConfig {
    /// Create a new adapter configuration
    pub fn new(bind_address: SocketAddr, sender: mpsc::Sender<Vec<LogEntry>>) -> Self {
        Self {
            bind_address,
            batch_size: 1000,
            buffer_size: 10_000,
            sender,
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

/// Running state owned by a spawned ingestion pump: the run flag polled
/// between accepts/datagrams and the count of parsed events.
///
/// Each pump allocates its own state when spawned; the adapter's `running`
/// / `events_received` fields are the external `start`/`stop` view and stay
/// separate (pre-extraction behaviour: the spawned task never observed
/// `stop()`).
pub(super) struct RunningState {
    /// Whether the pump should keep accepting and parsing.
    running: Arc<AtomicBool>,
    /// Events parsed and batched by this pump.
    events: Arc<AtomicU64>,
}

impl RunningState {
    fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(true)),
            events: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Flush `batch` to `sender` when it reaches `batch_size`, leaving a fresh
/// batch buffer in its place.
async fn flush_batch(
    sender: &mpsc::Sender<Vec<LogEntry>>,
    batch: &mut Vec<LogEntry>,
    batch_size: usize,
) {
    if batch.len() >= batch_size {
        let _ = sender.send(std::mem::take(batch)).await;
        *batch = Vec::with_capacity(batch_size);
    }
}

/// Spawn the shared TCP line pump: accept connections, split each stream
/// into lines, parse every line with `parse`, and batch entries to `sender`
/// (flushing at `batch_size`).
///
/// The per-adapter differences — the bind (and its error context) and the
/// parse function — stay at the call site; everything after a successful
/// bind is this skeleton (syslog and CEF/LEEF are the twins).
pub(super) fn pump_tcp<P>(
    listener: TcpListener,
    sender: mpsc::Sender<Vec<LogEntry>>,
    batch_size: usize,
    parse: P,
) where
    P: Fn(&str) -> Option<LogEntry> + Send + Sync + 'static,
{
    let state = RunningState::new();

    tokio::spawn(async move {
        let mut batch = Vec::with_capacity(batch_size);

        while state.running.load(Ordering::Relaxed) {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let reader = BufReader::new(stream);
                    let mut lines = reader.lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        if let Some(entry) = parse(&line) {
                            batch.push(entry);
                            state.events.fetch_add(1, Ordering::Relaxed);

                            flush_batch(&sender, &mut batch, batch_size).await;
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    });
}

/// Spawn the shared UDP datagram pump: receive datagrams, decode each as
/// UTF-8, parse with `parse`, and batch entries to `sender` (flushing at
/// `batch_size`).
///
/// Companion of [`pump_tcp`]; only the transport and the decode step
/// differ.
pub(super) fn pump_udp<P>(
    socket: UdpSocket,
    sender: mpsc::Sender<Vec<LogEntry>>,
    batch_size: usize,
    parse: P,
) where
    P: Fn(&str) -> Option<LogEntry> + Send + Sync + 'static,
{
    let state = RunningState::new();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65535];
        let mut batch = Vec::with_capacity(batch_size);

        while state.running.load(Ordering::Relaxed) {
            match socket.recv_from(&mut buf).await {
                Ok((len, _addr)) => {
                    if let Ok(msg) = std::str::from_utf8(&buf[..len])
                        && let Some(entry) = parse(msg)
                    {
                        batch.push(entry);
                        state.events.fetch_add(1, Ordering::Relaxed);

                        flush_batch(&sender, &mut batch, batch_size).await;
                    }
                }
                Err(_) => continue,
            }
        }
    });
}

/// Adapter manager for multiple input adapters
pub struct AdapterManager {
    /// Registered adapters
    adapters: Vec<Arc<dyn InputAdapter>>,
}

impl AdapterManager {
    /// Create a new adapter manager
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Register an adapter
    pub fn register(&mut self, adapter: Arc<dyn InputAdapter>) {
        self.adapters.push(adapter);
    }

    /// Start all adapters
    pub async fn start_all(&self) -> Result<()> {
        for adapter in &self.adapters {
            adapter.start().await?;
        }
        Ok(())
    }

    /// Stop all adapters
    pub async fn stop_all(&self) -> Result<()> {
        for adapter in &self.adapters {
            adapter.stop().await?;
        }
        Ok(())
    }

    /// Get adapter by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn InputAdapter>> {
        self.adapters.iter().find(|a| a.name() == name)
    }

    /// Get statistics for all adapters
    pub fn stats(&self) -> Vec<AdapterStats> {
        self.adapters
            .iter()
            .map(|a| AdapterStats {
                name: a.name().to_string(),
                running: a.is_running(),
                events_received: a.events_received(),
            })
            .collect()
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for an adapter
#[derive(Debug, Clone)]
pub struct AdapterStats {
    /// Adapter name
    pub name: String,
    /// Whether the adapter is running
    pub running: bool,
    /// Number of events received
    pub events_received: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_manager_new() {
        let manager = AdapterManager::new();
        assert!(manager.adapters.is_empty());
    }
}
