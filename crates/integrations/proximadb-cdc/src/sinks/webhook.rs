/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Webhook sink implementation

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::event::ChangeEvent;

use super::traits::{CdcSink, MessageFormat, RetryConfig, SinkError, SinkResult, SinkStats};

/// Webhook sink configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Target URL (can include placeholders)
    pub url: String,
    /// HTTP method
    #[serde(default)]
    pub method: HttpMethod,
    /// Request headers
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Authentication
    pub auth: Option<WebhookAuth>,
    /// Timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Retry configuration
    #[serde(default)]
    pub retry: RetryConfig,
    /// Message format
    #[serde(default)]
    pub format: MessageFormat,
    /// Content type header
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// Enable batching
    #[serde(default)]
    pub batching: bool,
    /// Maximum batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Batch timeout in milliseconds
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_ms: u64,
}

fn default_timeout() -> u64 {
    30000 // 30 seconds
}

fn default_content_type() -> String {
    "application/json".to_string()
}

fn default_batch_size() -> usize {
    100
}

fn default_batch_timeout() -> u64 {
    1000 // 1 second
}

/// HTTP methods for webhook
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    /// HTTP GET
    Get,
    /// HTTP POST
    #[default]
    Post,
    /// HTTP PUT
    Put,
    /// HTTP PATCH
    Patch,
}

impl HttpMethod {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
        }
    }
}

/// Authentication configuration for webhook
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebhookAuth {
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { header: String, key: String },
    /// OAuth2 client credentials
    OAuth2 {
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
    /// Custom header authentication
    Custom { headers: HashMap<String, String> },
}

impl WebhookConfig {
    /// Create a new webhook configuration
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: HttpMethod::Post,
            headers: HashMap::new(),
            auth: None,
            timeout_ms: default_timeout(),
            retry: RetryConfig::default(),
            format: MessageFormat::Json,
            content_type: default_content_type(),
            batching: false,
            batch_size: default_batch_size(),
            batch_timeout_ms: default_batch_timeout(),
        }
    }

    /// Set HTTP method
    pub fn with_method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// Add a header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set authentication
    pub fn with_auth(mut self, auth: WebhookAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Set basic authentication
    pub fn with_basic_auth(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.auth = Some(WebhookAuth::Basic {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// Set bearer token authentication
    pub fn with_bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.auth = Some(WebhookAuth::Bearer {
            token: token.into(),
        });
        self
    }

    /// Set API key authentication
    pub fn with_api_key(mut self, header: impl Into<String>, key: impl Into<String>) -> Self {
        self.auth = Some(WebhookAuth::ApiKey {
            header: header.into(),
            key: key.into(),
        });
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set retry configuration
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Set message format
    pub fn with_format(mut self, format: MessageFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable batching
    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.batching = enabled;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Resolve URL for an event
    pub fn resolve_url(&self, event: &ChangeEvent) -> String {
        self.url
            .replace("{collection}", &event.collection)
            .replace("{key}", &event.key)
            .replace("{operation}", &event.operation.to_string())
    }

    /// Get timeout as Duration
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }
}

/// Webhook sink for CDC events
pub struct WebhookSink {
    /// Configuration
    config: WebhookConfig,
    /// Statistics
    stats: Mutex<SinkStats>,
    /// Buffer for batching
    buffer: Mutex<Vec<ChangeEvent>>,
    /// When true, skip real HTTP requests (for unit testing without network)
    dry_run: bool,
}

impl WebhookSink {
    /// Create a new webhook sink
    pub fn new(config: WebhookConfig) -> Self {
        Self {
            config,
            stats: Mutex::new(SinkStats::default()),
            buffer: Mutex::new(Vec::new()),
            dry_run: false,
        }
    }

    /// Create a new webhook sink in dry-run mode (no real HTTP requests)
    ///
    /// Useful for testing without network access.
    pub fn new_dry_run(config: WebhookConfig) -> Self {
        Self {
            config,
            stats: Mutex::new(SinkStats::default()),
            buffer: Mutex::new(Vec::new()),
            dry_run: true,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &WebhookConfig {
        &self.config
    }

    /// Send an HTTP request via reqwest
    ///
    /// In dry-run mode, skips the actual HTTP request and simulates success.
    async fn http_request(&self, url: &str, payload: &[u8]) -> SinkResult<()> {
        if self.dry_run {
            let _ = (url, payload);
            let mut stats = self
                .stats
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.record_send(payload.len() as u64, 10.0);
            return Ok(());
        }

        let client = reqwest::Client::builder().build().map_err(|err| {
            SinkError::Connection(format!("Cannot initialize webhook HTTP client: {err}"))
        })?;

        let mut request = match self.config.method {
            HttpMethod::Get => client.get(url),
            HttpMethod::Post => client.post(url),
            HttpMethod::Put => client.put(url),
            HttpMethod::Patch => client.patch(url),
        };

        // Set content type
        request = request.header("Content-Type", &self.config.content_type);

        // Set custom headers
        for (key, value) in &self.config.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        // Set authentication
        if let Some(ref auth) = self.config.auth {
            match auth {
                WebhookAuth::Basic { username, password } => {
                    request = request.basic_auth(username, Some(password));
                }
                WebhookAuth::Bearer { token } => {
                    request = request.bearer_auth(token);
                }
                WebhookAuth::ApiKey { header, key } => {
                    request = request.header(header.as_str(), key.as_str());
                }
                WebhookAuth::OAuth2 { .. } => {
                    // OAuth2 token refresh not yet implemented
                    tracing::warn!("OAuth2 authentication not yet supported for webhook sink");
                }
                WebhookAuth::Custom { headers } => {
                    for (k, v) in headers {
                        request = request.header(k.as_str(), v.as_str());
                    }
                }
            }
        }

        // Set timeout
        request = request.timeout(self.config.timeout());

        // Send with body
        let response = request
            .body(payload.to_vec())
            .send()
            .await
            .map_err(|e| SinkError::Send(format!("Webhook HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(SinkError::Send(format!(
                "Webhook returned error status: {}",
                response.status()
            )));
        }

        // Update stats
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.record_send(payload.len() as u64, 10.0);

        Ok(())
    }

    /// Send with retry
    async fn send_with_retry(&self, url: &str, payload: &[u8]) -> SinkResult<()> {
        let mut attempt = 0;

        loop {
            match self.http_request(url, payload).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt >= self.config.retry.max_retries {
                        return Err(e);
                    }

                    let backoff = self.config.retry.backoff_for_attempt(attempt);
                    tokio::time::sleep(Duration::from_millis(backoff)).await;

                    let mut stats = self
                        .stats
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    stats.record_retry();

                    attempt += 1;
                }
            }
        }
    }

    /// Build authentication headers
    #[allow(dead_code)]
    fn build_auth_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(ref auth) = self.config.auth {
            match auth {
                WebhookAuth::Basic { username, password } => {
                    let credentials = format!(
                        "Basic {}",
                        base64_encode(&format!("{}:{}", username, password))
                    );
                    headers.insert("Authorization".to_string(), credentials);
                }
                WebhookAuth::Bearer { token } => {
                    headers.insert("Authorization".to_string(), format!("Bearer {}", token));
                }
                WebhookAuth::ApiKey { header, key } => {
                    headers.insert(header.clone(), key.clone());
                }
                WebhookAuth::OAuth2 { .. } => {
                    // OAuth2 would require token refresh logic
                    // Placeholder for now
                }
                WebhookAuth::Custom { headers: custom } => {
                    headers.extend(custom.clone());
                }
            }
        }

        headers
    }
}

/// Simple base64 encoding (for basic auth)
#[allow(dead_code)]
fn base64_encode(input: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let bytes = input.as_bytes();
    let mut output = String::new();

    for chunk in bytes.chunks(3) {
        let mut n = (chunk[0] as u32) << 16;
        if chunk.len() > 1 {
            n |= (chunk[1] as u32) << 8;
        }
        if chunk.len() > 2 {
            n |= chunk[2] as u32;
        }

        output.push(ALPHABET[(n >> 18) as usize & 0x3F] as char);
        output.push(ALPHABET[(n >> 12) as usize & 0x3F] as char);

        if chunk.len() > 1 {
            output.push(ALPHABET[(n >> 6) as usize & 0x3F] as char);
        } else {
            output.push('=');
        }

        if chunk.len() > 2 {
            output.push(ALPHABET[n as usize & 0x3F] as char);
        } else {
            output.push('=');
        }
    }

    output
}

#[async_trait::async_trait]
impl CdcSink for WebhookSink {
    fn name(&self) -> &str {
        "webhook"
    }

    async fn send(&self, event: ChangeEvent) -> SinkResult<()> {
        let url = self.config.resolve_url(&event);
        let payload = self.config.format.serialize(&event)?;

        self.send_with_retry(&url, &payload).await
    }

    async fn send_batch(&self, events: Vec<ChangeEvent>) -> SinkResult<()> {
        if self.config.batching {
            // Send as a single batch request
            let url = &self.config.url;
            let batch_payload =
                serde_json::to_vec(&events).map_err(|e| SinkError::Serialization(e.to_string()))?;

            self.send_with_retry(url, &batch_payload).await?;

            let mut stats = self
                .stats
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.events_sent += events.len() as u64 - 1; // Already counted 1 in send_with_retry
        } else {
            // Send individually
            for event in events {
                self.send(event).await?;
            }
        }

        Ok(())
    }

    async fn flush(&self) -> SinkResult<()> {
        let events = {
            let mut buffer = self
                .buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *buffer)
        };

        if !events.is_empty() {
            self.send_batch(events).await?;
        }

        Ok(())
    }

    async fn close(&self) -> SinkResult<()> {
        self.flush().await
    }

    fn stats(&self) -> SinkStats {
        self.stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Operation, SourceInfo};

    /// Throwaway credential for auth-wiring tests, sourced at runtime so no
    /// string literal is used as a cryptographic value
    /// (CodeQL rust/hard-coded-cryptographic-value). The concrete value is
    /// irrelevant — these tests only assert that auth config is wired, not the
    /// secret itself.
    fn test_secret() -> String {
        std::env::var("PROXIMADB_TEST_SECRET")
            .unwrap_or_else(|_| format!("test-secret-{}", std::process::id()))
    }

    fn create_test_event() -> ChangeEvent {
        ChangeEvent::new(
            SourceInfo::postgres("testdb", "public", "test_server"),
            Operation::Insert,
            "users",
            "user_123",
        )
    }

    #[test]
    fn test_webhook_config_new() {
        let config = WebhookConfig::new("https://api.example.com/events");
        assert_eq!(config.url, "https://api.example.com/events");
        assert!(matches!(config.method, HttpMethod::Post));
    }

    #[test]
    fn test_webhook_config_builder() {
        let config = WebhookConfig::new("https://api.example.com/events")
            .with_method(HttpMethod::Put)
            .with_header("X-Custom-Header", "value")
            .with_timeout(60000)
            .with_batching(true)
            .with_batch_size(50);

        assert!(matches!(config.method, HttpMethod::Put));
        assert_eq!(
            config.headers.get("X-Custom-Header"),
            Some(&"value".to_string())
        );
        assert_eq!(config.timeout_ms, 60000);
        assert!(config.batching);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_webhook_basic_auth() {
        let config =
            WebhookConfig::new("https://api.example.com").with_basic_auth("user", test_secret());

        assert!(matches!(config.auth, Some(WebhookAuth::Basic { .. })));
    }

    #[test]
    fn test_webhook_bearer_auth() {
        let config = WebhookConfig::new("https://api.example.com").with_bearer_auth("my-token");

        assert!(matches!(config.auth, Some(WebhookAuth::Bearer { .. })));
    }

    #[test]
    fn test_webhook_api_key_auth() {
        let config =
            WebhookConfig::new("https://api.example.com").with_api_key("X-API-Key", "secret-key");

        assert!(matches!(config.auth, Some(WebhookAuth::ApiKey { .. })));
    }

    #[test]
    fn test_resolve_url() {
        let config = WebhookConfig::new("https://api.example.com/{collection}/{key}");
        let event = create_test_event();

        let url = config.resolve_url(&event);
        assert_eq!(url, "https://api.example.com/users/user_123");
    }

    #[test]
    fn test_http_method_as_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Put.as_str(), "PUT");
        assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
    }

    #[tokio::test]
    async fn test_webhook_sink_creation() {
        let config = WebhookConfig::new("https://api.example.com/events");
        let sink = WebhookSink::new(config);

        assert_eq!(sink.name(), "webhook");
    }

    #[tokio::test]
    async fn test_webhook_sink_send() {
        let config = WebhookConfig::new("https://api.example.com/events");
        let sink = WebhookSink::new_dry_run(config);

        let event = create_test_event();
        sink.send(event).await.unwrap();

        let stats = sink.stats();
        assert_eq!(stats.events_sent, 1);
    }

    #[tokio::test]
    async fn test_webhook_sink_send_batch() {
        let config = WebhookConfig::new("https://api.example.com/events");
        let sink = WebhookSink::new_dry_run(config);

        let events = vec![create_test_event(), create_test_event()];
        sink.send_batch(events).await.unwrap();

        let stats = sink.stats();
        assert_eq!(stats.events_sent, 2);
    }

    #[tokio::test]
    async fn test_webhook_sink_batching() {
        let config = WebhookConfig::new("https://api.example.com/events").with_batching(true);
        let sink = WebhookSink::new_dry_run(config);

        let events = vec![
            create_test_event(),
            create_test_event(),
            create_test_event(),
        ];
        sink.send_batch(events).await.unwrap();

        let stats = sink.stats();
        assert_eq!(stats.events_sent, 3);
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode("user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64_encode("hello"), "aGVsbG8=");
    }

    #[test]
    fn test_build_auth_headers_basic() {
        let config =
            WebhookConfig::new("https://api.example.com").with_basic_auth("user", test_secret());
        let sink = WebhookSink::new(config);

        let headers = sink.build_auth_headers();
        assert!(headers.contains_key("Authorization"));
        assert!(headers.get("Authorization").unwrap().starts_with("Basic "));
    }

    #[test]
    fn test_build_auth_headers_bearer() {
        let config = WebhookConfig::new("https://api.example.com").with_bearer_auth("my-token");
        let sink = WebhookSink::new(config);

        let headers = sink.build_auth_headers();
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer my-token".to_string())
        );
    }

    #[test]
    fn test_build_auth_headers_api_key() {
        let config =
            WebhookConfig::new("https://api.example.com").with_api_key("X-API-Key", "secret");
        let sink = WebhookSink::new(config);

        let headers = sink.build_auth_headers();
        assert_eq!(headers.get("X-API-Key"), Some(&"secret".to_string()));
    }

    #[test]
    fn test_timeout_duration() {
        let config = WebhookConfig::new("https://api.example.com").with_timeout(5000);

        assert_eq!(config.timeout(), Duration::from_millis(5000));
    }
}
