// Notification channels for alerts
//
// Provides:
// - Webhook notifications
// - Slack integration
// - PagerDuty integration
// - Email notifications (SMTP)

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::ObservabilityAlert as Alert;

/// Notification manager
pub struct NotificationManager {
    /// Registered channels
    channels: RwLock<HashMap<String, NotificationChannel>>,
    /// HTTP client for webhooks
    client: std::result::Result<reqwest::Client, reqwest::Error>,
}

impl NotificationManager {
    /// Create a new notification manager
    pub fn new() -> Self {
        let client = reqwest::Client::builder().no_proxy().build();

        Self {
            channels: RwLock::new(HashMap::new()),
            client,
        }
    }

    fn http_client(&self) -> Result<&reqwest::Client> {
        self.client
            .as_ref()
            .map_err(|err| anyhow::anyhow!("Cannot initialize notification HTTP client: {err}"))
    }

    /// Register a notification channel
    pub async fn register_channel(&self, name: &str, channel: NotificationChannel) -> Result<()> {
        let mut channels = self.channels.write().await;
        channels.insert(name.to_string(), channel);
        info!("Registered notification channel: {}", name);
        Ok(())
    }

    /// Unregister a notification channel
    pub async fn unregister_channel(&self, name: &str) -> Result<()> {
        let mut channels = self.channels.write().await;
        channels.remove(name);
        info!("Unregistered notification channel: {}", name);
        Ok(())
    }

    /// Send notification for an alert
    pub async fn send(&self, alert: &Alert) -> Result<()> {
        let channels = self.channels.read().await;

        for (name, channel) in channels.iter() {
            if let Err(e) = self.send_to_channel(channel, alert).await {
                warn!("Failed to send notification to {}: {}", name, e);
            }
        }

        Ok(())
    }

    /// Send to a specific channel
    async fn send_to_channel(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        match channel {
            NotificationChannel::Webhook(config) => self.send_webhook(config, alert).await,
            NotificationChannel::Slack(config) => self.send_slack(config, alert).await,
            NotificationChannel::PagerDuty(config) => self.send_pagerduty(config, alert).await,
            NotificationChannel::Email(config) => self.send_email(config, alert).await,
        }
    }

    /// Send webhook notification
    async fn send_webhook(&self, config: &WebhookConfig, alert: &Alert) -> Result<()> {
        let payload = WebhookPayload {
            alert_name: alert.name.clone(),
            message: alert.message.clone(),
            severity: alert.severity.to_string(),
            source: alert.source.clone(),
            value: alert.value,
            threshold: alert.threshold,
            labels: alert.labels.clone(),
            annotations: alert.annotations.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let mut request = self.http_client()?.post(&config.url);

        // Add custom headers
        for (key, value) in &config.headers {
            request = request.header(key, value);
        }

        // Add authentication
        if let Some(auth) = &config.auth {
            match auth {
                WebhookAuth::Bearer(token) => {
                    request = request.bearer_auth(token);
                }
                WebhookAuth::Basic { username, password } => {
                    request = request.basic_auth(username, Some(password));
                }
            }
        }

        let response = request
            .json(&payload)
            .send()
            .await
            .context("Failed to send webhook")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Webhook returned status: {}",
                response.status()
            ));
        }

        debug!("Sent webhook notification for alert: {}", alert.name);
        Ok(())
    }

    /// Send Slack notification
    async fn send_slack(&self, config: &SlackConfig, alert: &Alert) -> Result<()> {
        let color = match alert.severity {
            super::AlertSeverity::Low => "#36a64f",
            super::AlertSeverity::Medium => "#daa038",
            super::AlertSeverity::High => "#cc0000",
            super::AlertSeverity::Critical => "#6d0000",
        };

        let payload = serde_json::json!({
            "channel": config.channel,
            "attachments": [{
                "color": color,
                "title": format!("[{}] {}", alert.severity, alert.name),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Source",
                        "value": alert.source,
                        "short": true
                    },
                    {
                        "title": "Value",
                        "value": alert.value.map(|v| format!("{:.2}", v)).unwrap_or_default(),
                        "short": true
                    }
                ],
                "footer": "ProximaDB Alerting",
                "ts": chrono::Utc::now().timestamp()
            }]
        });

        let response = self
            .http_client()?
            .post(&config.webhook_url)
            .json(&payload)
            .send()
            .await
            .context("Failed to send Slack notification")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Slack webhook returned status: {}",
                response.status()
            ));
        }

        debug!("Sent Slack notification for alert: {}", alert.name);
        Ok(())
    }

    /// Send PagerDuty notification
    async fn send_pagerduty(&self, config: &PagerDutyConfig, alert: &Alert) -> Result<()> {
        let severity = match alert.severity {
            super::AlertSeverity::Low => "info",
            super::AlertSeverity::Medium => "warning",
            super::AlertSeverity::High => "error",
            super::AlertSeverity::Critical => "critical",
        };

        let payload = serde_json::json!({
            "routing_key": config.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.key(),
            "payload": {
                "summary": format!("[{}] {}: {}", alert.severity, alert.name, alert.message),
                "source": alert.source,
                "severity": severity,
                "custom_details": {
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "labels": alert.labels,
                }
            }
        });

        let response = self
            .http_client()?
            .post("https://events.pagerduty.com/v2/enqueue")
            .json(&payload)
            .send()
            .await
            .context("Failed to send PagerDuty notification")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "PagerDuty API returned status: {}",
                response.status()
            ));
        }

        debug!("Sent PagerDuty notification for alert: {}", alert.name);
        Ok(())
    }

    /// Send email notification via SMTP
    async fn send_email(&self, config: &EmailConfig, alert: &Alert) -> Result<()> {
        use lettre::{
            Message, SmtpTransport, Transport, message::header::ContentType,
            transport::smtp::authentication::Credentials,
        };

        // Build email message
        let subject = format!("[ProximaDB Alert] {}", alert.name);
        let value_str = alert
            .value
            .map_or("N/A".to_string(), |v| format!("{:.2}", v));
        let threshold_str = alert
            .threshold
            .map_or("N/A".to_string(), |v| format!("{:.2}", v));
        let severity_str = format_severity_from_enum(alert.severity);

        let body = format!(
            "Alert: {}\n\n\
             Severity: {}\n\n\
             Message: {}\n\n\
             Value: {}\n\n\
             Threshold: {}\n\n\
             Source: {}\n\n\
             Timestamp: {}",
            alert.name,
            severity_str,
            alert.message,
            value_str,
            threshold_str,
            alert.source,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );

        // Send to each recipient
        for recipient in &config.recipients {
            let email = Message::builder()
                .from(config.from.parse().context("Invalid from address")?)
                .to(recipient
                    .parse()
                    .with_context(|| format!("Invalid recipient address: {}", recipient))?)
                .subject(&subject)
                .header(ContentType::TEXT_PLAIN)
                .body(body.clone())
                .context("Failed to build email message")?;

            // Create SMTP transport
            let smtp_addr = format!("{}:{}", config.smtp_server, config.smtp_port);
            let creds = Credentials::new(config.username.clone(), config.password.clone());

            let mailer = SmtpTransport::relay(&smtp_addr)
                .map_err(|e| anyhow::anyhow!("Failed to create SMTP transport: {}", e))?
                .credentials(creds)
                .build();

            // Send the email (blocking, but OK for alert notifications)
            tokio::task::spawn_blocking(move || {
                mailer.send(&email).context("Failed to send email via SMTP")
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to join email sending task: {}", e))?
            .context("Failed to send email via SMTP")?;

            debug!(
                "Sent email notification to {} for alert: {}",
                recipient, alert.name
            );
        }

        info!(
            "Sent email notifications to {} recipients for alert: {}",
            config.recipients.len(),
            alert.name
        );
        Ok(())
    }
}

impl Default for NotificationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Notification channel types
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    /// Webhook
    Webhook(WebhookConfig),
    /// Slack
    Slack(SlackConfig),
    /// PagerDuty
    PagerDuty(PagerDutyConfig),
    /// Email
    Email(EmailConfig),
}

/// Webhook configuration
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Webhook URL
    pub url: String,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Authentication
    pub auth: Option<WebhookAuth>,
}

/// Webhook authentication
#[derive(Debug, Clone)]
pub enum WebhookAuth {
    /// Bearer token
    Bearer(String),
    /// Basic auth
    /// Basic authentication with username and password.
    Basic {
        /// HTTP basic auth username.
        username: String,
        /// HTTP basic auth password.
        password: String,
    },
}

/// Slack configuration
#[derive(Debug, Clone)]
pub struct SlackConfig {
    /// Webhook URL
    pub webhook_url: String,
    /// Channel to post to
    pub channel: String,
    /// Username to post as
    pub username: Option<String>,
    /// Icon emoji
    pub icon_emoji: Option<String>,
}

/// PagerDuty configuration
#[derive(Debug, Clone)]
pub struct PagerDutyConfig {
    /// Integration key / routing key
    pub routing_key: String,
}

/// Email configuration
#[derive(Debug, Clone)]
pub struct EmailConfig {
    /// SMTP server
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// From address
    pub from: String,
    /// Recipients
    pub recipients: Vec<String>,
}

/// Webhook payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Alert name
    pub alert_name: String,
    /// Message
    pub message: String,
    /// Severity
    pub severity: String,
    /// Source
    pub source: String,
    /// Value
    pub value: Option<f64>,
    /// Threshold
    pub threshold: Option<f64>,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
    /// Timestamp
    pub timestamp: String,
}

/// Format severity for display
fn format_severity_from_enum(severity: crate::alerting::AlertSeverity) -> String {
    match severity {
        crate::alerting::AlertSeverity::Critical => "Critical".to_string(),
        crate::alerting::AlertSeverity::High => "High".to_string(),
        crate::alerting::AlertSeverity::Medium => "Medium".to_string(),
        crate::alerting::AlertSeverity::Low => "Low".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_channel() {
        let manager = NotificationManager::new();

        let channel = NotificationChannel::Webhook(WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            headers: HashMap::new(),
            auth: None,
        });

        manager.register_channel("test", channel).await.unwrap();

        let channels = manager.channels.read().await;
        assert!(channels.contains_key("test"));
    }

    #[test]
    fn test_webhook_payload() {
        let payload = WebhookPayload {
            alert_name: "HighCPU".to_string(),
            message: "CPU is high".to_string(),
            severity: "high".to_string(),
            source: "server-1".to_string(),
            value: Some(95.0),
            threshold: Some(90.0),
            labels: HashMap::new(),
            annotations: HashMap::new(),
            timestamp: "2023-12-26T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("HighCPU"));
    }

    #[tokio::test]
    #[ignore] // Run manually with: cargo test --lib -- --ignored test_smtp_notification
    async fn test_smtp_notification() {
        use crate::alerting::{AlertSeverity, ObservabilityAlert as Alert};
        use std::collections::HashMap;

        // This test uses a non-routable recipient.
        // Configure your SMTP settings before running this test
        let manager = NotificationManager::new();

        // Create email configuration
        let email_config = EmailConfig {
            smtp_server: std::env::var("SMTP_SERVER")
                .unwrap_or_else(|_| "smtp.gmail.com".to_string()),
            smtp_port: std::env::var("SMTP_PORT")
                .unwrap_or_else(|_| "587".to_string())
                .parse()
                .expect("Invalid SMTP port"),
            username: std::env::var("SMTP_USERNAME").expect("SMTP_USERNAME must be set"),
            password: std::env::var("SMTP_PASSWORD").expect("SMTP_PASSWORD must be set"),
            from: std::env::var("SMTP_FROM")
                .unwrap_or_else(|_| "noreply@proximadb.test".to_string()),
            recipients: vec!["noreply@proximadb.test".to_string()],
        };

        // Register email channel
        let channel = NotificationChannel::Email(email_config.clone());
        manager
            .register_channel("smtp_test", channel)
            .await
            .unwrap();

        // Create a test alert
        let mut labels = HashMap::new();
        labels.insert("host".to_string(), "localhost".to_string());
        labels.insert("test".to_string(), "integration".to_string());

        let mut annotations = HashMap::new();
        annotations.insert(
            "description".to_string(),
            "Test alert from ProximaDB".to_string(),
        );

        let alert = Alert {
            name: "Test SMTP Alert".to_string(),
            message: "This is a test alert from ProximaDB integration test".to_string(),
            severity: AlertSeverity::Medium,
            source: "integration_test".to_string(),
            rule_id: None,
            labels,
            annotations,
            value: Some(42.0),
            threshold: Some(50.0),
        };

        // Send the notification
        let result = manager.send(&alert).await;
        assert!(
            result.is_ok(),
            "Failed to send SMTP notification: {:?}",
            result.err()
        );

        println!("✅ Successfully sent test email");
    }
}
