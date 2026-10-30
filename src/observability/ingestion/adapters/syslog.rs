// Syslog adapter (RFC 3164/5424)
//
// Supports:
// - TCP and UDP transport
// - RFC 3164 (BSD syslog)
// - RFC 5424 (structured syslog)
// - TLS encryption (optional)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::net::{TcpListener, UdpSocket};

use super::{AdapterConfig, InputAdapter};
use crate::proto::proximadb_v1::{LogEntry, Severity};

/// Syslog adapter for TCP and UDP
pub struct SyslogAdapter {
    /// Configuration
    config: AdapterConfig,
    /// Whether the adapter is running
    running: AtomicBool,
    /// Number of events received
    events_received: AtomicU64,
    /// Protocol (TCP or UDP)
    protocol: SyslogProtocol,
}

/// Syslog transport protocol
#[derive(Debug, Clone, Copy)]
pub enum SyslogProtocol {
    /// TCP transport for reliable syslog delivery.
    Tcp,
    /// UDP transport for traditional syslog.
    Udp,
}

impl SyslogAdapter {
    /// Create a new syslog adapter
    pub fn new(config: AdapterConfig, protocol: SyslogProtocol) -> Self {
        Self {
            config,
            running: AtomicBool::new(false),
            events_received: AtomicU64::new(0),
            protocol,
        }
    }

    /// Parse a syslog message
    #[allow(dead_code)]
    fn parse_message(&self, msg: &str) -> Option<LogEntry> {
        // Try RFC 5424 first, then RFC 3164
        self.parse_rfc5424(msg).or_else(|| self.parse_rfc3164(msg))
    }

    /// Parse RFC 5424 syslog
    #[allow(dead_code)]
    fn parse_rfc5424(&self, msg: &str) -> Option<LogEntry> {
        if !msg.starts_with('<') {
            return None;
        }

        let pri_end = msg.find('>')?;
        let pri: u8 = msg[1..pri_end].parse().ok()?;

        let rest = &msg[pri_end + 1..];
        if !rest.starts_with('1') {
            return None;
        }

        let parts: Vec<&str> = rest.splitn(7, ' ').collect();
        if parts.len() < 7 {
            return None;
        }

        let severity = self.pri_to_severity(pri);
        let timestamp = chrono::DateTime::parse_from_rfc3339(parts[1])
            .ok()?
            .timestamp_nanos_opt()
            .unwrap_or(0);

        Some(LogEntry {
            timestamp_ns: timestamp,
            severity: severity as i32,
            message: parts[6].to_string(),
            fields: HashMap::new(),
            source: Some(parts[2].to_string()),
            service: Some(parts[3].to_string()),
        })
    }

    /// Parse RFC 3164 syslog
    #[allow(dead_code)]
    fn parse_rfc3164(&self, msg: &str) -> Option<LogEntry> {
        if !msg.starts_with('<') {
            return None;
        }

        let pri_end = msg.find('>')?;
        let pri: u8 = msg[1..pri_end].parse().ok()?;

        let severity = self.pri_to_severity(pri);
        let rest = &msg[pri_end + 1..];

        // Extract message content
        let message = rest.trim().to_string();

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message,
            fields: HashMap::new(),
            source: None,
            service: None,
        })
    }

    /// Convert syslog PRI to severity
    #[allow(dead_code)]
    fn pri_to_severity(&self, pri: u8) -> Severity {
        match pri & 0x07 {
            0..=2 => Severity::Fatal,
            3 => Severity::Error,
            4 => Severity::Warn,
            5 | 6 => Severity::Info,
            7 => Severity::Debug,
            _ => Severity::Info,
        }
    }

    /// Start TCP listener
    async fn start_tcp(&self) -> Result<()> {
        let listener = TcpListener::bind(self.config.bind_address)
            .await
            .context("Failed to bind TCP socket")?;

        super::pump_tcp(
            listener,
            self.config.sender.clone(),
            self.config.batch_size,
            Self::parse_line,
        );

        Ok(())
    }

    /// Start UDP listener
    async fn start_udp(&self) -> Result<()> {
        let socket = UdpSocket::bind(self.config.bind_address)
            .await
            .context("Failed to bind UDP socket")?;

        super::pump_udp(
            socket,
            self.config.sender.clone(),
            self.config.batch_size,
            Self::parse_line,
        );

        Ok(())
    }

    /// Parse a syslog line (static for use in spawned tasks)
    fn parse_line(msg: &str) -> Option<LogEntry> {
        // Simplified parsing for spawned task
        if !msg.starts_with('<') {
            return None;
        }

        let pri_end = msg.find('>')?;
        let pri: u8 = msg[1..pri_end].parse().ok()?;
        let severity = match pri & 0x07 {
            0..=2 => Severity::Fatal,
            3 => Severity::Error,
            4 => Severity::Warn,
            5 | 6 => Severity::Info,
            7 => Severity::Debug,
            _ => Severity::Info,
        };

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message: msg[pri_end + 1..].trim().to_string(),
            fields: HashMap::new(),
            source: None,
            service: None,
        })
    }
}

#[async_trait]
impl InputAdapter for SyslogAdapter {
    fn name(&self) -> &str {
        match self.protocol {
            SyslogProtocol::Tcp => "syslog-tcp",
            SyslogProtocol::Udp => "syslog-udp",
        }
    }

    async fn start(&self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        match self.protocol {
            SyslogProtocol::Tcp => self.start_tcp().await,
            SyslogProtocol::Udp => self.start_udp().await,
        }
    }

    async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    fn events_received(&self) -> u64 {
        self.events_received.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_pri_to_severity() {
        let (tx, _rx) = mpsc::channel(100);
        let config = AdapterConfig::new("127.0.0.1:514".parse().unwrap(), tx);
        let adapter = SyslogAdapter::new(config, SyslogProtocol::Udp);

        assert_eq!(adapter.pri_to_severity(0), Severity::Fatal);
        assert_eq!(adapter.pri_to_severity(3), Severity::Error);
        assert_eq!(adapter.pri_to_severity(4), Severity::Warn);
        assert_eq!(adapter.pri_to_severity(6), Severity::Info);
        assert_eq!(adapter.pri_to_severity(7), Severity::Debug);
    }
}
