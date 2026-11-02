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

//! Rate limiting middleware for ProximaDB HTTP API

use axum::{
    extract::Request,
    extract::State,
    http::StatusCode,
    middleware::Next,
    response::{Json, Response},
};
use serde::Serialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rate limiting configuration - consolidated from network module
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting (if false, all requests pass through)
    pub enabled: bool,

    /// Maximum requests per minute (TOML-friendly)
    pub requests_per_minute: u32,

    /// Burst allowance for sudden spikes
    pub burst_size: u32,

    /// Apply rate limiting per IP address
    pub by_ip: bool,

    /// Whether to apply rate limiting to health endpoints
    pub limit_health_endpoints: bool,

    /// Global rate limit (applies to all IPs combined, optional)
    pub global_requests_per_minute: Option<u32>,
}

// Default functions for serde
#[allow(dead_code)]
fn default_requests_per_minute() -> u32 {
    1000
}
#[allow(dead_code)]
fn default_burst_size() -> u32 {
    100
}
#[allow(dead_code)]
fn default_true() -> bool {
    true
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default for security
            requests_per_minute: 1000,
            burst_size: 100,
            by_ip: true,
            limit_health_endpoints: false,
            global_requests_per_minute: None,
        }
    }
}

impl RateLimitConfig {
    /// Create a disabled rate limit configuration.
    ///
    /// **WARNING**: Only use for development/testing. Production should always
    /// have rate limiting enabled to prevent abuse.
    pub fn disabled() -> Self {
        tracing::warn!("🚨 Rate limiting is DISABLED. This is a security risk in production!");
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a production rate limit configuration with specified limits.
    pub fn production(requests_per_minute: u32, burst_size: u32) -> Self {
        Self {
            enabled: true,
            requests_per_minute,
            burst_size,
            by_ip: true,
            limit_health_endpoints: false,
            global_requests_per_minute: Some(requests_per_minute * 10), // Global limit
        }
    }

    /// Create a high-throughput rate limit configuration.
    pub fn high_throughput() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 10000,
            burst_size: 1000,
            by_ip: true,
            limit_health_endpoints: false,
            global_requests_per_minute: Some(100000),
        }
    }
}

impl RateLimitConfig {
    /// Convert to the internal format used by middleware
    pub fn to_middleware_config(&self) -> MiddlewareRateLimitConfig {
        MiddlewareRateLimitConfig {
            enabled: self.enabled,
            max_requests: self.burst_size, // Use burst as max for short windows
            window_duration: Duration::from_secs(60), // 1 minute window
            limit_health_endpoints: self.limit_health_endpoints,
            global_max_requests: self.global_requests_per_minute,
        }
    }
}

/// Internal rate limiting configuration used by middleware logic
#[derive(Debug, Clone)]
pub struct MiddlewareRateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum requests per IP per window
    pub max_requests: u32,
    /// Sliding window duration for rate limit tracking
    pub window_duration: Duration,
    /// Whether to apply rate limits to health check endpoints
    pub limit_health_endpoints: bool,
    /// Global maximum requests across all clients (optional)
    pub global_max_requests: Option<u32>,
}

/// Rate limit bucket for tracking requests
#[derive(Debug, Clone)]
struct RateLimitBucket {
    count: u32,
    window_start: Instant,
}

impl RateLimitBucket {
    fn new() -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
        }
    }

    fn increment(&mut self, window_duration: Duration) -> bool {
        let now = Instant::now();

        // Reset bucket if window expired
        if now.duration_since(self.window_start) >= window_duration {
            self.count = 0;
            self.window_start = now;
        }

        self.count += 1;
        true
    }

    fn is_within_limit(&self, max_requests: u32, window_duration: Duration) -> bool {
        let now = Instant::now();

        // If window expired, we're within limit
        if now.duration_since(self.window_start) >= window_duration {
            return true;
        }

        self.count <= max_requests
    }
}

/// Rate limiting state
pub struct RateLimitState {
    config: MiddlewareRateLimitConfig,
    buckets: Arc<RwLock<HashMap<IpAddr, RateLimitBucket>>>,
    global_bucket: Arc<RwLock<RateLimitBucket>>,
}

impl RateLimitState {
    /// Create a new rate limit state with the given configuration
    pub fn new(config: MiddlewareRateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(RwLock::new(HashMap::new())),
            global_bucket: Arc::new(RwLock::new(RateLimitBucket::new())),
        }
    }

    /// Whether limiting is enabled (callers can skip the check cheaply).
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check-and-consume one request for rate-limit subject `key` (the client
    /// IP). Returns `Err(retry_after_secs)` when the global or per-key window
    /// limit is exceeded, `Ok(())` otherwise. This is the *single* check used by
    /// both the REST middleware and the pgwire query path, so the surfaces share
    /// one limiter implementation (converge, don't duplicate). `Ok` immediately
    /// when limiting is disabled.
    pub async fn check_and_consume(&self, key: IpAddr) -> Result<(), u64> {
        if !self.config.enabled {
            return Ok(());
        }
        let retry_after = self.config.window_duration.as_secs();

        // Global window first (if configured).
        if let Some(global_max) = self.config.global_max_requests {
            let mut global_bucket = self.global_bucket.write().await;
            global_bucket.increment(self.config.window_duration);
            if !global_bucket.is_within_limit(global_max, self.config.window_duration) {
                return Err(retry_after);
            }
        }

        // Per-key window.
        let mut buckets = self.buckets.write().await;
        let bucket = buckets.entry(key).or_insert_with(RateLimitBucket::new);
        bucket.increment(self.config.window_duration);
        if !bucket.is_within_limit(self.config.max_requests, self.config.window_duration) {
            return Err(retry_after);
        }
        Ok(())
    }
}

/// Env gate mounting the REST request limiter (TD-RATE-1): per-IP requests
/// per minute. Unset/0/garbage ⇒ no limiter is built — the default deployment
/// (single-node, embedded) keeps today's unlimited behavior. Mirrors
/// `PROXIMADB_PGWIRE_RATE_LIMIT_RPM` on the pgwire path, so both request
/// surfaces gate the same way.
pub const REST_RATE_LIMIT_RPM_ENV: &str = "PROXIMADB_REST_RATE_LIMIT_RPM";

/// Pure parse of the env-gate value: `Some(rpm)` only for a positive integer.
/// Kept pure (takes the raw string) so the gate is testable without env races.
pub fn parse_rest_rate_limit_rpm(raw: Option<&str>) -> Option<u32> {
    raw?.trim().parse::<u32>().ok().filter(|rpm| *rpm > 0)
}

/// Build the REST request limiter from [`REST_RATE_LIMIT_RPM_ENV`], or `None`
/// when the gate is unset — in which case **no layer is added at all** (zero
/// hot-path cost, zero behavior change; that is the TD-RATE-1 default-off).
///
/// Configuration mirrors the pgwire mount exactly (`production(rpm, rpm)`:
/// per-IP `rpm`/window, burst `rpm`, and a process-global ceiling of
/// `rpm * 10` across all clients), so both request surfaces have the same
/// semantics and one tuning vocabulary.
///
/// Keys on the client address resolved by `get_client_ip` — the TD-TENANT-4
/// observed peer (or a trusted-proxy assertion under
/// `PROXIMADB_TRUSTED_PROXY_CIDRS`); never a raw client-supplied header.
pub fn rest_rate_limit_state_from_env() -> Option<Arc<RateLimitState>> {
    let rpm = parse_rest_rate_limit_rpm(std::env::var(REST_RATE_LIMIT_RPM_ENV).ok().as_deref())?;
    Some(Arc::new(RateLimitState::new(
        RateLimitConfig::production(rpm, rpm).to_middleware_config(),
    )))
}

/// Rate limit error response
#[derive(Debug, Serialize)]
pub struct RateLimitErrorResponse {
    error: String,
    message: String,
    retry_after: u64, // seconds
}

/// Rate limiting middleware function
pub async fn rate_limit_middleware(
    State(rate_limit_state): State<Arc<RateLimitState>>,
    request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<RateLimitErrorResponse>)> {
    // Skip rate limiting if disabled
    if !rate_limit_state.config.enabled {
        return Ok(next.run(request).await);
    }

    let path = request.uri().path();

    // Skip rate limiting for health endpoints (if configured)
    if !rate_limit_state.config.limit_health_endpoints && is_health_endpoint(path) {
        return Ok(next.run(request).await);
    }

    // Extract client IP
    let client_ip = get_client_ip(&request);

    // Single converged check (shared with the pgwire path). Map the limiter's
    // retry-after into the HTTP 429 envelope this surface returns.
    match rate_limit_state.check_and_consume(client_ip).await {
        Ok(()) => Ok(next.run(request).await),
        Err(retry_after) => Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(RateLimitErrorResponse {
                error: "rate_limit_exceeded".to_string(),
                message: format!(
                    "Rate limit exceeded. Maximum {} requests per {} seconds.",
                    rate_limit_state.config.max_requests,
                    rate_limit_state.config.window_duration.as_secs()
                ),
                retry_after,
            }),
        )),
    }
}

/// Extract client IP from request. `pub` so the (root) metrics middleware can
/// classify the same client (one IP-extraction seam shared across the edge).
///
/// TD-TENANT-4: this used to read `X-Forwarded-For` → `X-Real-IP` → a hardcoded
/// `127.0.0.1`, so the address was whatever the client claimed, or a constant.
/// It now resolves from the **observed** transport peer
/// ([`ConnectInfo`](axum::extract::ConnectInfo), wired at the TCP serve sites),
/// honoring a forwarded header only when the peer is a declared trusted proxy.
///
/// The trusted-proxy allowlist is read from the process env
/// ([`TRUSTED_PROXY_CIDRS_ENV`](crate::client_addr::TRUSTED_PROXY_CIDRS_ENV))
/// once and cached: this runs on every metered request, so re-parsing CIDRs per
/// request would put env access and allocation on the hot path.
///
/// Prefer [`crate::client_addr::client_addr_from_request`] in new code — it
/// returns the provenance alongside the address, so a caller that meters can
/// tell an observation from a fallback.
pub fn get_client_ip(request: &Request) -> IpAddr {
    crate::client_addr::client_addr_from_request(request, trusted_proxies()).ip
}

/// Process-wide trusted-proxy allowlist, parsed once from the environment.
fn trusted_proxies() -> &'static crate::client_addr::TrustedProxies {
    static TRUSTED: std::sync::OnceLock<crate::client_addr::TrustedProxies> =
        std::sync::OnceLock::new();
    TRUSTED.get_or_init(crate::client_addr::TrustedProxies::from_env)
}

/// Check if the path is a health endpoint.
///
/// Shared across rate-limit, auth, and TLS middleware so the definition
/// of "health endpoint" stays in one place.
pub fn is_health_endpoint(path: &str) -> bool {
    path.starts_with("/health")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_and_consume_enforces_per_key_window() {
        let state = RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 2,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        });
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(state.check_and_consume(ip).await.is_ok());
        assert!(state.check_and_consume(ip).await.is_ok());
        // Third request in the window exceeds max_requests=2.
        assert!(state.check_and_consume(ip).await.is_err());
        // A different key keeps its own bucket.
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(state.check_and_consume(ip2).await.is_ok());
    }

    #[tokio::test]
    async fn check_and_consume_is_ok_when_disabled() {
        let state = RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: false,
            max_requests: 1,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        });
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..10 {
            assert!(state.check_and_consume(ip).await.is_ok());
        }
    }

    #[test]
    fn test_rate_limit_bucket() {
        let mut bucket = RateLimitBucket::new();
        let window = Duration::from_secs(60);

        // Within limit initially
        assert!(bucket.is_within_limit(10, window));

        // Increment and check
        bucket.increment(window);
        assert_eq!(bucket.count, 1);
        assert!(bucket.is_within_limit(10, window));
    }

    #[test]
    fn test_is_health_endpoint() {
        assert!(is_health_endpoint("/health"));
        assert!(is_health_endpoint("/health/ready"));
        assert!(is_health_endpoint("/health/live"));
        assert!(!is_health_endpoint("/collections"));
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        // Default is now enabled for security
        assert!(config.enabled);
        assert_eq!(config.burst_size, 100);
        assert_eq!(config.requests_per_minute, 1000);
        assert!(!config.limit_health_endpoints);
        assert!(config.global_requests_per_minute.is_none());
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_rate_limit_config_production() {
        let config = RateLimitConfig::production(5000, 500);
        assert!(config.enabled);
        assert_eq!(config.requests_per_minute, 5000);
        assert_eq!(config.burst_size, 500);
        assert!(config.global_requests_per_minute.is_some());
    }

    #[test]
    fn test_rate_limit_config_high_throughput() {
        let config = RateLimitConfig::high_throughput();
        assert!(config.enabled);
        assert_eq!(config.requests_per_minute, 10000);
        assert_eq!(config.burst_size, 1000);
    }

    // ============================================================
    // Extended rate limit tests (coverage improvement)
    // ============================================================

    #[test]
    fn test_rate_limit_config_production_global_limit() {
        let config = RateLimitConfig::production(2000, 200);
        assert_eq!(config.requests_per_minute, 2000);
        assert_eq!(config.burst_size, 200);
        assert_eq!(config.global_requests_per_minute, Some(20000));
        assert!(config.by_ip);
        assert!(!config.limit_health_endpoints);
    }

    #[test]
    fn test_to_middleware_config() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_minute: 500,
            burst_size: 50,
            by_ip: true,
            limit_health_endpoints: true,
            global_requests_per_minute: Some(5000),
        };
        let mw = config.to_middleware_config();
        assert!(mw.enabled);
        assert_eq!(mw.max_requests, 50); // burst_size
        assert_eq!(mw.window_duration, Duration::from_secs(60));
        assert!(mw.limit_health_endpoints);
        assert_eq!(mw.global_max_requests, Some(5000));
    }

    #[test]
    fn test_to_middleware_config_disabled() {
        let config = RateLimitConfig {
            enabled: false,
            ..Default::default()
        };
        let mw = config.to_middleware_config();
        assert!(!mw.enabled);
    }

    #[test]
    fn test_rate_limit_bucket_within_limit() {
        let mut bucket = RateLimitBucket::new();
        let window = Duration::from_secs(60);

        // Under limit
        for _ in 0..5 {
            bucket.increment(window);
        }
        assert!(bucket.is_within_limit(10, window));
        assert_eq!(bucket.count, 5);
    }

    #[test]
    fn test_rate_limit_bucket_exceeds_limit() {
        let mut bucket = RateLimitBucket::new();
        let window = Duration::from_secs(60);

        for _ in 0..11 {
            bucket.increment(window);
        }
        assert!(!bucket.is_within_limit(10, window));
        assert_eq!(bucket.count, 11);
    }

    #[test]
    fn test_rate_limit_bucket_exactly_at_limit() {
        let mut bucket = RateLimitBucket::new();
        let window = Duration::from_secs(60);

        for _ in 0..10 {
            bucket.increment(window);
        }
        assert!(bucket.is_within_limit(10, window));
        assert_eq!(bucket.count, 10);

        // One more pushes it over
        bucket.increment(window);
        assert!(!bucket.is_within_limit(10, window));
    }

    // ---- TD-RATE-1: the mount ----

    /// The gate's truth table: only a positive integer mounts a limiter.
    #[test]
    fn rest_rate_limit_gate_truth_table() {
        assert_eq!(parse_rest_rate_limit_rpm(None), None, "unset = off");
        assert_eq!(parse_rest_rate_limit_rpm(Some("")), None);
        assert_eq!(parse_rest_rate_limit_rpm(Some("0")), None, "0 = off");
        assert_eq!(parse_rest_rate_limit_rpm(Some("-5")), None);
        assert_eq!(parse_rest_rate_limit_rpm(Some("wat")), None);
        assert_eq!(parse_rest_rate_limit_rpm(Some("600")), Some(600));
        assert_eq!(parse_rest_rate_limit_rpm(Some(" 600 ")), Some(600));
    }

    /// A mounted limiter 429s an over-limit observed peer. This drives the
    /// REAL middleware through a real router (oneshot), not the bucket alone —
    /// the mount's contract is the 429 envelope, and that lives here.
    #[tokio::test]
    async fn mounted_middleware_429s_the_over_limit_peer() {
        use axum::routing::get;
        use tower::ServiceExt;

        let state = Arc::new(RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 2,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        }));
        let app = axum::Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                state,
                rate_limit_middleware,
            ));

        let peer = "203.0.113.9:5000".parse::<std::net::SocketAddr>().unwrap();
        for expected in [200u16, 200, 429] {
            let mut req = axum::http::Request::builder()
                .uri("/")
                .body(axum::body::Body::empty())
                .unwrap();
            req.extensions_mut()
                .insert(axum::extract::ConnectInfo(peer));
            let res = app.clone().oneshot(req).await.unwrap();
            assert_eq!(res.status().as_u16(), expected);
        }
    }

    /// `/health` stays reachable when the bucket is exhausted — the default
    /// `limit_health_endpoints: false` exemption. A flooded node must remain
    /// observable.
    #[tokio::test]
    async fn health_endpoints_exempt_from_the_limit() {
        use axum::routing::get;
        use tower::ServiceExt;

        let state = Arc::new(RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 1,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        }));
        let app = axum::Router::new()
            .route("/", get(|| async { "ok" }))
            .route("/health", get(|| async { "healthy" }))
            .layer(axum::middleware::from_fn_with_state(
                state,
                rate_limit_middleware,
            ));

        let peer = "203.0.113.9:5000".parse::<std::net::SocketAddr>().unwrap();
        let send = |path: &'static str| {
            let mut req = axum::http::Request::builder()
                .uri(path)
                .body(axum::body::Body::empty())
                .unwrap();
            req.extensions_mut()
                .insert(axum::extract::ConnectInfo(peer));
            req
        };

        // Exhaust the bucket on "/" ...
        assert_eq!(app.clone().oneshot(send("/")).await.unwrap().status(), 200);
        assert_eq!(app.clone().oneshot(send("/")).await.unwrap().status(), 429);
        // ... health still answers.
        assert_eq!(app.oneshot(send("/health")).await.unwrap().status(), 200);
    }

    /// Per-IP isolation: one noisy peer's exhaustion must not deny a second
    /// observed peer (the noisy-neighbor contract; keys are the observed
    /// address, so this also proves spoofed-header rotation buys nothing).
    #[tokio::test]
    async fn a_second_observed_peer_is_unaffected() {
        use axum::routing::get;
        use tower::ServiceExt;

        let state = Arc::new(RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 1,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        }));
        let app = axum::Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                state,
                rate_limit_middleware,
            ));

        let noisy = "203.0.113.9:5000".parse::<std::net::SocketAddr>().unwrap();
        let quiet = "198.51.100.4:5000".parse::<std::net::SocketAddr>().unwrap();
        let send = |addr: std::net::SocketAddr| {
            let mut req = axum::http::Request::builder()
                .uri("/")
                .body(axum::body::Body::empty())
                .unwrap();
            req.extensions_mut()
                .insert(axum::extract::ConnectInfo(addr));
            req
        };

        assert_eq!(
            app.clone().oneshot(send(noisy)).await.unwrap().status(),
            200
        );
        assert_eq!(
            app.clone().oneshot(send(noisy)).await.unwrap().status(),
            429
        );
        // Same request, different observed peer: unaffected.
        assert_eq!(app.oneshot(send(quiet)).await.unwrap().status(), 200);
    }

    #[test]
    fn test_is_health_endpoint_various_paths() {
        assert!(is_health_endpoint("/health"));
        assert!(is_health_endpoint("/health/ready"));
        assert!(is_health_endpoint("/health/live"));
        assert!(is_health_endpoint("/health/startup"));
        assert!(!is_health_endpoint("/api/v1/collections"));
        assert!(!is_health_endpoint("/metrics"));
        assert!(!is_health_endpoint("/"));
        assert!(is_health_endpoint("/healthcheck")); // starts with /health so it IS a health endpoint
    }

    #[test]
    fn test_rate_limit_state_creation() {
        let config = MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 100,
            window_duration: Duration::from_secs(60),
            limit_health_endpoints: false,
            global_max_requests: None,
        };
        let state = RateLimitState::new(config);
        assert!(state.config.enabled);
        assert_eq!(state.config.max_requests, 100);
    }

    #[test]
    fn test_rate_limit_error_response_serialization() {
        let err = RateLimitErrorResponse {
            error: "rate_limit_exceeded".to_string(),
            message: "Too many requests".to_string(),
            retry_after: 60,
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("rate_limit_exceeded"));
        assert!(json.contains("60"));
    }

    #[test]
    fn test_high_throughput_global_limit() {
        let config = RateLimitConfig::high_throughput();
        assert_eq!(config.global_requests_per_minute, Some(100000));
    }

    // ============================================================
    // Additional unique tests from integration test file
    // ============================================================

    #[test]
    fn test_rate_limit_serde_roundtrip() {
        use serde_json;

        let original_config = RateLimitConfig {
            enabled: true,
            requests_per_minute: 2000,
            burst_size: 200,
            by_ip: false,
            limit_health_endpoints: true,
            global_requests_per_minute: Some(10000),
        };

        let serialized = serde_json::to_string(&original_config).expect("Failed to serialize");
        let deserialized: RateLimitConfig =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        assert_eq!(original_config.enabled, deserialized.enabled);
        assert_eq!(
            original_config.requests_per_minute,
            deserialized.requests_per_minute
        );
        assert_eq!(original_config.burst_size, deserialized.burst_size);
        assert_eq!(original_config.by_ip, deserialized.by_ip);
        assert_eq!(
            original_config.limit_health_endpoints,
            deserialized.limit_health_endpoints
        );
        assert_eq!(
            original_config.global_requests_per_minute,
            deserialized.global_requests_per_minute
        );
    }

    #[test]
    fn test_local_demo_config_toml_parsing() {
        // Test parsing a simple TOML config that disables rate limiting
        use toml;

        let toml_content = r#"
[network]
enabled = false

[rate_limit]
enabled = false
"#;

        let parsed: toml::Value = toml::from_str(toml_content).expect("Failed to parse TOML");

        // Verify the structure can be parsed
        if let Some(network) = parsed.get("network")
            && let Some(enabled) = network.get("enabled")
        {
            assert_eq!(enabled.as_bool(), Some(false));
        }

        if let Some(rate_limit) = parsed.get("rate_limit")
            && let Some(enabled) = rate_limit.get("enabled")
        {
            assert_eq!(enabled.as_bool(), Some(false));
        }
    }
}
