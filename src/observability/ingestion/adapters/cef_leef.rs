// CEF/LEEF adapter (ArcSight and IBM QRadar formats)
//
// Supports:
// - CEF (Common Event Format) - HP ArcSight
// - LEEF (Log Event Extended Format) - IBM QRadar
// - UDP and TCP transport
// - Syslog-wrapped CEF/LEEF

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::net::{TcpListener, UdpSocket};

use super::{AdapterConfig, InputAdapter};
use crate::proto::proximadb_v1::{LogEntry, Severity, SqlValue};

/// CEF/LEEF adapter
pub struct CefLeefAdapter {
    /// Configuration
    config: AdapterConfig,
    /// Whether the adapter is running
    running: AtomicBool,
    /// Number of events received
    events_received: AtomicU64,
    /// Format type
    format: SecurityFormat,
    /// Protocol
    protocol: CefLeefProtocol,
}

/// Security log format
#[derive(Debug, Clone, Copy)]
pub enum SecurityFormat {
    /// Common Event Format (ArcSight)
    Cef,
    /// Log Event Extended Format (IBM QRadar)
    Leef,
    /// Auto-detect from message
    Auto,
}

/// Transport protocol
#[derive(Debug, Clone, Copy)]
pub enum CefLeefProtocol {
    /// TCP transport for reliable delivery.
    Tcp,
    /// UDP transport for low-latency delivery.
    Udp,
}

impl CefLeefAdapter {
    /// Create a new CEF/LEEF adapter
    pub fn new(config: AdapterConfig, format: SecurityFormat, protocol: CefLeefProtocol) -> Self {
        Self {
            config,
            running: AtomicBool::new(false),
            events_received: AtomicU64::new(0),
            format,
            protocol,
        }
    }

    /// Parse a message and detect format
    #[allow(dead_code)]
    fn parse_message(&self, msg: &str) -> Option<LogEntry> {
        // Strip syslog wrapper if present
        let content = self.strip_syslog_header(msg);

        match self.format {
            SecurityFormat::Cef => self.parse_cef(content),
            SecurityFormat::Leef => self.parse_leef(content),
            SecurityFormat::Auto => {
                if content.starts_with("CEF:") {
                    self.parse_cef(content)
                } else if content.starts_with("LEEF:") {
                    self.parse_leef(content)
                } else {
                    None
                }
            }
        }
    }

    /// Strip syslog header from message
    fn strip_syslog_header<'a>(&self, msg: &'a str) -> &'a str {
        // Look for CEF: or LEEF: in the message
        if let Some(pos) = msg.find("CEF:") {
            return &msg[pos..];
        }
        if let Some(pos) = msg.find("LEEF:") {
            return &msg[pos..];
        }
        msg
    }

    /// Parse CEF format
    /// CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
    fn parse_cef(&self, msg: &str) -> Option<LogEntry> {
        if !msg.starts_with("CEF:") {
            return None;
        }

        let content = &msg[4..];
        let parts: Vec<&str> = content.splitn(8, '|').collect();
        if parts.len() < 7 {
            return None;
        }

        let _version = parts[0];
        let vendor = parts[1];
        let product = parts[2];
        let _device_version = parts[3];
        let signature_id = parts[4];
        let name = parts[5];
        let severity_str = parts[6];

        let severity = self.parse_cef_severity(severity_str);
        let source = format!("{} {}", vendor, product);

        let mut attributes: HashMap<String, String> = HashMap::new();
        attributes.insert("signature_id".to_string(), signature_id.to_string());

        // Parse extension if present
        if parts.len() >= 8 {
            let extension = self.parse_cef_extension(parts[7]);
            attributes.extend(extension);
        }

        // Convert attributes to SqlValue map
        let fields: HashMap<String, SqlValue> = attributes
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    SqlValue {
                        value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(v)),
                    },
                )
            })
            .collect();

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message: name.to_string(),
            fields,
            source: Some(source),
            service: Some(product.to_string()),
        })
    }

    /// Parse LEEF format
    /// LEEF:Version|Vendor|Product|Version|EventID|Extension
    fn parse_leef(&self, msg: &str) -> Option<LogEntry> {
        if !msg.starts_with("LEEF:") {
            return None;
        }

        let content = &msg[5..];
        let parts: Vec<&str> = content.splitn(6, '|').collect();
        if parts.len() < 5 {
            return None;
        }

        let version = parts[0];
        let vendor = parts[1];
        let product = parts[2];
        let _product_version = parts[3];
        let event_id = parts[4];

        let source = format!("{} {}", vendor, product);

        let mut attributes: HashMap<String, String> = HashMap::new();
        attributes.insert("event_id".to_string(), event_id.to_string());
        attributes.insert("leef_version".to_string(), version.to_string());

        // Parse extension if present
        let severity = if parts.len() >= 6 {
            let extension = self.parse_leef_extension(parts[5]);
            let sev = extension
                .get("sev")
                .or_else(|| extension.get("severity"))
                .and_then(|s| s.parse::<u8>().ok())
                .map_or(Severity::Info, |n| self.leef_severity_to_severity(n));
            attributes.extend(extension);
            sev
        } else {
            Severity::Info
        };

        // Convert attributes to SqlValue map
        let fields: HashMap<String, SqlValue> = attributes
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    SqlValue {
                        value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(v)),
                    },
                )
            })
            .collect();

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message: event_id.to_string(),
            fields,
            source: Some(source),
            service: Some(product.to_string()),
        })
    }

    /// Parse CEF severity string
    fn parse_cef_severity(&self, sev: &str) -> Severity {
        // CEF severity: 0-3 = Low, 4-6 = Medium, 7-8 = High, 9-10 = Very High
        match sev.trim().parse::<u8>() {
            Ok(n) if n <= 3 => Severity::Info,
            Ok(n) if n <= 6 => Severity::Warn,
            Ok(n) if n <= 8 => Severity::Error,
            Ok(_) => Severity::Fatal,
            Err(_) => match sev.to_lowercase().as_str() {
                "low" | "unknown" => Severity::Info,
                "medium" => Severity::Warn,
                "high" => Severity::Error,
                "very-high" | "critical" => Severity::Fatal,
                _ => Severity::Info,
            },
        }
    }

    /// Convert LEEF severity number to Severity
    fn leef_severity_to_severity(&self, sev: u8) -> Severity {
        match sev {
            0..=3 => Severity::Info,
            4..=6 => Severity::Warn,
            7..=8 => Severity::Error,
            _ => Severity::Fatal,
        }
    }

    /// Parse CEF extension key=value pairs
    fn parse_cef_extension(&self, ext: &str) -> HashMap<String, String> {
        let mut attrs = HashMap::new();
        let mut current_key = String::new();
        let mut current_value = String::new();
        let mut in_value = false;

        for part in ext.split(' ') {
            if let Some(eq_pos) = part.find('=') {
                if in_value && !current_key.is_empty() {
                    attrs.insert(current_key.clone(), current_value.trim().to_string());
                }
                current_key = part[..eq_pos].to_string();
                current_value = part[eq_pos + 1..].to_string();
                in_value = true;
            } else if in_value {
                current_value.push(' ');
                current_value.push_str(part);
            }
        }

        if in_value && !current_key.is_empty() {
            attrs.insert(current_key, current_value.trim().to_string());
        }

        attrs
    }

    /// Parse LEEF extension key=value pairs
    fn parse_leef_extension(&self, ext: &str) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        // LEEF 1.0 uses tab separator, LEEF 2.0 allows custom separator
        let separator = if ext.contains('\t') { '\t' } else { ' ' };

        for pair in ext.split(separator) {
            if let Some(eq_pos) = pair.find('=') {
                let key = pair[..eq_pos].to_string();
                let value = pair[eq_pos + 1..].to_string();
                attrs.insert(key, value);
            }
        }

        attrs
    }

    /// Start TCP listener
    async fn start_tcp(&self) -> Result<()> {
        let listener = TcpListener::bind(self.config.bind_address)
            .await
            .context("Failed to bind CEF/LEEF TCP listener")?;

        let format = self.format;
        super::pump_tcp(
            listener,
            self.config.sender.clone(),
            self.config.batch_size,
            move |line| Self::parse_line_static(line, format),
        );

        Ok(())
    }

    /// Start UDP listener
    async fn start_udp(&self) -> Result<()> {
        let socket = UdpSocket::bind(self.config.bind_address)
            .await
            .context("Failed to bind CEF/LEEF UDP socket")?;

        let format = self.format;
        super::pump_udp(
            socket,
            self.config.sender.clone(),
            self.config.batch_size,
            move |msg| Self::parse_line_static(msg, format),
        );

        Ok(())
    }

    /// Static parsing helper for spawned tasks
    fn parse_line_static(msg: &str, format: SecurityFormat) -> Option<LogEntry> {
        // Strip syslog header
        let content = if let Some(pos) = msg.find("CEF:") {
            &msg[pos..]
        } else if let Some(pos) = msg.find("LEEF:") {
            &msg[pos..]
        } else {
            msg
        };

        match format {
            SecurityFormat::Cef => Self::parse_cef_static(content),
            SecurityFormat::Leef => Self::parse_leef_static(content),
            SecurityFormat::Auto => {
                if content.starts_with("CEF:") {
                    Self::parse_cef_static(content)
                } else if content.starts_with("LEEF:") {
                    Self::parse_leef_static(content)
                } else {
                    None
                }
            }
        }
    }

    /// Static CEF parsing for spawned tasks
    fn parse_cef_static(msg: &str) -> Option<LogEntry> {
        if !msg.starts_with("CEF:") {
            return None;
        }

        let content = &msg[4..];
        let parts: Vec<&str> = content.splitn(8, '|').collect();
        if parts.len() < 7 {
            return None;
        }

        let vendor = parts[1];
        let product = parts[2];
        let name = parts[5];
        let severity_str = parts[6];

        let severity = match severity_str.trim().parse::<u8>() {
            Ok(n) if n <= 3 => Severity::Info,
            Ok(n) if n <= 6 => Severity::Warn,
            Ok(n) if n <= 8 => Severity::Error,
            Ok(_) => Severity::Fatal,
            Err(_) => Severity::Info,
        };

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: severity as i32,
            message: name.to_string(),
            fields: HashMap::new(),
            source: Some(format!("{} {}", vendor, product)),
            service: Some(product.to_string()),
        })
    }

    /// Static LEEF parsing for spawned tasks
    fn parse_leef_static(msg: &str) -> Option<LogEntry> {
        if !msg.starts_with("LEEF:") {
            return None;
        }

        let content = &msg[5..];
        let parts: Vec<&str> = content.splitn(6, '|').collect();
        if parts.len() < 5 {
            return None;
        }

        let vendor = parts[1];
        let product = parts[2];
        let event_id = parts[4];

        Some(LogEntry {
            timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            severity: Severity::Info as i32,
            message: event_id.to_string(),
            fields: HashMap::new(),
            source: Some(format!("{} {}", vendor, product)),
            service: Some(product.to_string()),
        })
    }
}

#[async_trait]
impl InputAdapter for CefLeefAdapter {
    fn name(&self) -> &str {
        match (self.format, self.protocol) {
            (SecurityFormat::Cef, CefLeefProtocol::Tcp) => "cef-tcp",
            (SecurityFormat::Cef, CefLeefProtocol::Udp) => "cef-udp",
            (SecurityFormat::Leef, CefLeefProtocol::Tcp) => "leef-tcp",
            (SecurityFormat::Leef, CefLeefProtocol::Udp) => "leef-udp",
            (SecurityFormat::Auto, CefLeefProtocol::Tcp) => "cef-leef-tcp",
            (SecurityFormat::Auto, CefLeefProtocol::Udp) => "cef-leef-udp",
        }
    }

    async fn start(&self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        match self.protocol {
            CefLeefProtocol::Tcp => self.start_tcp().await,
            CefLeefProtocol::Udp => self.start_udp().await,
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
    fn test_parse_cef() {
        let (tx, _rx) = mpsc::channel(100);
        let config = AdapterConfig::new("127.0.0.1:514".parse().unwrap(), tx);
        let adapter = CefLeefAdapter::new(config, SecurityFormat::Cef, CefLeefProtocol::Udp);

        let msg = "CEF:0|Security|Product|1.0|100|Test Event|5|src=10.0.0.1 dst=10.0.0.2";
        let entry = adapter.parse_cef(msg).unwrap();

        assert_eq!(entry.message, "Test Event");
        assert_eq!(entry.severity, Severity::Warn as i32);
        assert!(
            entry
                .source
                .as_ref()
                .is_some_and(|s| s.contains("Security"))
        );
    }

    #[test]
    fn test_parse_leef() {
        let (tx, _rx) = mpsc::channel(100);
        let config = AdapterConfig::new("127.0.0.1:514".parse().unwrap(), tx);
        let adapter = CefLeefAdapter::new(config, SecurityFormat::Leef, CefLeefProtocol::Udp);

        let msg = "LEEF:1.0|Vendor|Product|1.0|EventID|src=10.0.0.1";
        let entry = adapter.parse_leef(msg).unwrap();

        assert_eq!(entry.message, "EventID");
        assert!(entry.source.as_ref().is_some_and(|s| s.contains("Vendor")));
    }

    #[test]
    fn test_cef_severity() {
        let (tx, _rx) = mpsc::channel(100);
        let config = AdapterConfig::new("127.0.0.1:514".parse().unwrap(), tx);
        let adapter = CefLeefAdapter::new(config, SecurityFormat::Cef, CefLeefProtocol::Udp);

        assert_eq!(adapter.parse_cef_severity("0"), Severity::Info);
        assert_eq!(adapter.parse_cef_severity("5"), Severity::Warn);
        assert_eq!(adapter.parse_cef_severity("8"), Severity::Error);
        assert_eq!(adapter.parse_cef_severity("10"), Severity::Fatal);
        assert_eq!(adapter.parse_cef_severity("High"), Severity::Error);
    }
}
