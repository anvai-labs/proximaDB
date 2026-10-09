// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Generic OIDC provider support (TD-SSO-1): ProximaDB as an OIDC
//! **resource server**.
//!
//! Scope is deliberately bearer-only: clients obtain tokens from the IdP
//! (Kanidm, Keycloak, anything OIDC-compliant) themselves — client-credentials
//! for services, PKCE via their own tooling for humans — and present them on
//! the existing bearer surfaces (REST `Authorization`, gRPC metadata). No
//! browser redirect flows and no RP session layer live in the engine.
//!
//! ## The role seam (the reason this module exists beyond verification)
//!
//! PR #1791's adversarial review established the invariant: `UnifiedUserContext.roles`
//! is consumed by `SecurityPredicate::RoleBased` (`security/rls/service.rs`),
//! which grants **Unrestricted** row access on role-string match — so arbitrary
//! role pass-through from any credential source is an escalation. API keys got
//! seam-sanitization in #1791; this module extends it to IdP-asserted claims,
//! and the SAME [`sanitize_idp_roles`] helper retrofits the local-JWT path, so
//! the invariant now holds by construction on every credential path:
//!
//! * `gateway`/`operator` pass only when `allow_delegation_roles` is set
//!   (default FALSE — IdP groups come from an external directory; see
//!   `default_allow_delegation_roles`);
//! * any other string passes only when the operator listed it in
//!   `role_allowlist` (default **empty** — fail-closed);
//! * everything else is dropped with a once-per-role warning.
//!
//! ## Confusion defense
//!
//! Validation accepts ONLY the configured algorithm set (default `RS256`).
//! An `alg: none` token, or an HMAC token presented while the JWKS holds RSA
//! keys (the classic cross-algorithm confusion), is structurally rejected.
//! Routing between the local (HS*) and OIDC (RS*/ES*) verifiers uses the
//! UNAUTHENTICATED JWT header `alg` — safe because it only selects a verifier
//! that either cryptographically validates or rejects; it never authorizes.

use jsonwebtoken::{
    Algorithm, DecodingKey, Validation,
    jwk::{AlgorithmParameters, Jwk, JwkSet},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Bounded once-per-role warning set for the sanitizer (log spam guard; the
/// same shape as the tier-claim warn-once seams).
static DROPPED_ROLE_WARNINGS: std::sync::OnceLock<std::sync::Mutex<HashSet<String>>> =
    std::sync::OnceLock::new();
/// F9: bound the process-global warn set (token-influenced strings must not
/// grow a process-global map without limit).
const DROPPED_ROLE_WARN_CAP: usize = 256;

/// Generic OIDC provider configuration (TD-SSO-1). Hung off
/// `[security.authentication.oidc]`; every optional field carries a safe
/// default so a minimal deployment only sets `issuer_url` + `audience`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OidcProviderConfig {
    /// Enable OIDC bearer validation. Absent ⇒ the whole path is inert.
    #[serde(default)]
    pub enabled: bool,
    /// The expected `iss` claim (e.g. `https://kanidm.example.com/oauth2/openid/proximadb`).
    pub issuer_url: String,
    /// Additional accepted `iss` values (e.g. Google's bare `accounts.google.com`
    /// alongside the URL form). Aliases may be bare hostnames — they are
    /// matched as exact strings against the token's `iss` claim, not fetched.
    #[serde(default)]
    pub issuer_aliases: Vec<String>,
    /// The acceptable `aud` values (any-of semantics). Accepts either a
    /// single string (backward compat) or a list — Azure commonly requires
    /// both `api://{client-id}` and the bare `{client-id}`.
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    pub audience: Vec<String>,
    /// Explicit JWKS URL. When absent, derived from
    /// `{issuer_url}/.well-known/openid-configuration` at first use.
    #[serde(default)]
    pub jwks_url: Option<String>,
    /// Pinned acceptable algorithms. Default `["RS256"]`. HMAC-family entries
    /// are rejected at parse time — an OIDC IdP must not be configured to
    /// accept symmetric tokens.
    #[serde(default = "default_algorithms")]
    pub allowed_algorithms: Vec<String>,
    /// Which token claim carries role/group names. Default `"groups"`
    /// (Kanidm); `"roles"` is the common fallback.
    #[serde(default = "default_roles_claim")]
    pub roles_claim: String,
    /// Optional claim mapped to `tenant_id` (e.g. a custom Kanidm claim).
    /// Absent ⇒ no tenant from the token; the existing header-trust ladder
    /// governs tenant selection.
    #[serde(default)]
    pub tenant_claim: Option<String>,
    /// Operator-approved business roles that may cross the seam (see the
    /// module docs: RLS `RoleBased` grants Unrestricted on role match, so
    /// this list is the deliberate opt-in). Default empty = fail-closed.
    #[serde(default)]
    pub role_allowlist: Vec<String>,
    /// Whether the `gateway`/`operator` delegation markers may cross
    /// (enabling `GatewayOnly` tenant delegation for IdP-authenticated
    /// principals). Default FALSE — see `default_allow_delegation_roles`.
    #[serde(default = "default_allow_delegation_roles")]
    pub allow_delegation_roles: bool,
    /// Clock-skew tolerance for exp/nbf (seconds). Default 60.
    #[serde(default = "default_clock_skew")]
    pub clock_skew_seconds: u64,
    /// JWKS cache lifetime (seconds). Default 3600.
    #[serde(default = "default_jwks_cache")]
    pub jwks_cache_seconds: u64,
    /// Per-request HTTP timeout for discovery/JWKS fetches (seconds). Default 10.
    #[serde(default = "default_http_timeout")]
    pub http_timeout_seconds: u64,
    /// PEM file containing a custom CA certificate for the JWKS/discovery
    /// client (self-hosted IdPs on internal PKI — Kanidm/Keycloak without
    /// public certs). When unset, the system trust store is used.
    #[serde(default)]
    pub ca_cert_path: Option<String>,
}

/// Serde helper: accept `"value"` or `["value1", "value2"]` for the
/// audience field (backward compat with the original single-string config).
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        One(String),
        Many(Vec<String>),
    }
    Ok(match StringOrVec::deserialize(deserializer)? {
        StringOrVec::One(s) => vec![s],
        StringOrVec::Many(v) => v,
    })
}

fn default_clock_skew() -> u64 {
    60
}
fn default_jwks_cache() -> u64 {
    3600
}
fn default_http_timeout() -> u64 {
    10
}

fn default_algorithms() -> Vec<String> {
    vec!["RS256".to_string()]
}
fn default_roles_claim() -> String {
    "groups".to_string()
}
/// F2 (adversarial review): default FALSE for the IdP path. Unlike the
/// #1791 API-key seam (role strings typed by the operator into engine
/// config), IdP groups come from an EXTERNAL directory where `gateway`/
/// `operator` are ordinary pre-existing group names (e.g. a network-gateway
/// team) — importing them by default would silently grant cross-tenant
/// delegation. Operators opt in deliberately.
fn default_allow_delegation_roles() -> bool {
    false
}

impl OidcProviderConfig {
    /// Parse + validate the pinned algorithm set. Rejects HMAC-family
    /// entries and unknown names at CONFIG load (fail-fast on the operator,
    /// never at token time).
    pub fn pinned_algorithms(&self) -> Result<Vec<Algorithm>, String> {
        if self.allowed_algorithms.is_empty() {
            return Err("oidc.allowed_algorithms is empty — pin at least RS256".into());
        }
        let mut out = Vec::with_capacity(self.allowed_algorithms.len());
        for name in &self.allowed_algorithms {
            let alg =
                Algorithm::from_str(name).map_err(|_| format!("unknown algorithm {name:?}"))?;
            if matches!(alg, Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512) {
                return Err(format!(
                    "oidc.allowed_algorithms contains {name}: HMAC-family algorithms are \
                     forbidden on the OIDC path (cross-algorithm confusion)"
                ));
            }
            // ES256/ES384 now supported (provider portability): Azure's
            // recommended configuration and Keycloak EC keys use them.
            if !matches!(
                alg,
                Algorithm::RS256
                    | Algorithm::RS384
                    | Algorithm::RS512
                    | Algorithm::ES256
                    | Algorithm::ES384
            ) {
                return Err(format!(
                    "oidc.allowed_algorithms contains {name}: not supported (RSA and ES only)"
                ));
            }
            out.push(alg);
        }
        Ok(out)
    }
}

/// The seam: filter IdP-asserted role strings down to what may cross into
/// `UnifiedUserContext.roles`. Pure and total — the single helper shared by
/// the OIDC and local-JWT conversion paths (TD-SSO-1; the #1791 invariant,
/// uniform).
pub fn sanitize_idp_roles(raw: &[String], cfg: &OidcProviderConfig) -> Vec<String> {
    let warned = DROPPED_ROLE_WARNINGS.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut out = Vec::new();
    for role in raw {
        let is_delegation =
            role == proximadb_tenant::GATEWAY_ROLE || role == proximadb_tenant::OPERATOR_ROLE;
        let allowed = if is_delegation {
            cfg.allow_delegation_roles
        } else {
            cfg.role_allowlist.iter().any(|a| a == role)
        };
        if allowed {
            if !out.iter().any(|kept: &String| kept == role) {
                out.push(role.clone());
            }
        } else if let Ok(mut seen) = warned.lock()
            && seen.len() < DROPPED_ROLE_WARN_CAP
            && seen.insert(format!("{}:{}", cfg.issuer_url, role))
        {
            tracing::warn!(
                target: "proximadb::tenant_audit",
                issuer = %cfg.issuer_url,
                dropped_role = %role,
                "IdP-asserted role dropped at the seam (not in the allowlist; RLS RoleBased \
                 matches role strings — this drop is the #1791 invariant)"
            );
        }
    }
    out
}

/// Claims accepted from an OIDC token. Only the standard set plus two
/// free-form slots (roles claim + tenant claim) — extracted from the raw
/// payload, never trusted until the signature verifies.
#[derive(Debug, Clone, Deserialize)]
pub struct OidcClaims {
    pub iss: String,
    pub sub: String,
    /// `aud` is string-or-array on the wire.
    pub aud: Aud,
    pub exp: i64,
    #[serde(default)]
    pub iat: i64,
    /// The configured roles claim's value (array of strings; a lone string is
    /// tolerated). Extracted post-verification from the same payload.
    #[serde(default)]
    pub roles_raw: serde_json::Value,
    /// The configured tenant claim's value, when set.
    #[serde(default)]
    pub tenant_raw: serde_json::Value,
}

/// `aud` polymorphism.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Aud {
    One(String),
    Many(Vec<String>),
}

impl Aud {
    fn contains(&self, expected: &str) -> bool {
        match self {
            Aud::One(a) => a == expected,
            Aud::Many(v) => v.iter().any(|a| a == expected),
        }
    }
}

/// A fetched JWKS with its cache stamp.
struct CachedJwks {
    set: Arc<JwkSet>,
    fetched_at: Instant,
}

/// OIDC bearer-token verifier: JWKS fetch/cache/refresh + strict validation.
///
/// Boot-order robustness: constructing the verifier does NOT contact the IdP.
/// Keys are fetched lazily on first use and force-refreshed when a token
/// carries an unknown `kid` (rotation); the cache is additionally refreshed
/// after [`JWKS_TTL`]. An unreachable IdP fails the REQUEST closed with a
/// clear error — never the boot.
pub struct OidcTokenVerifier {
    config: Arc<OidcProviderConfig>,
    algorithms: Vec<Algorithm>,
    client: reqwest::Client,
    jwks: tokio::sync::RwLock<Option<CachedJwks>>,
    /// F1: single-flight for JWKS fetches (concurrent unknown-kid requests
    /// must not each trigger an outbound fetch).
    fetch_gate: tokio::sync::Mutex<()>,
    /// F1: negative-cache throttle — an unknown `kid` forces at most one
    /// refresh per window, so crafted tokens cannot drive a fetch storm
    /// against the IdP (which would rate-limit us into an auth outage).
    last_forced_refresh: std::sync::Mutex<Option<Instant>>,
}

/// F1: minimum interval between unknown-kid-forced refreshes.
/// (JWKS cache TTL and HTTP timeout are now configurable via
/// OidcProviderConfig::jwks_cache_seconds / http_timeout_seconds.)
const FORCED_REFRESH_MIN_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum OidcError {
    #[error("OIDC verifier not configured")]
    NotConfigured,
    #[error("token rejected: {0}")]
    Rejected(String),
    #[error("JWKS unavailable from issuer: {0}")]
    JwksUnavailable(String),
}

impl OidcTokenVerifier {
    /// Construct without network I/O (see the type docs). Returns an error
    /// only for invalid configuration — fail-fast on the operator.
    pub fn new(config: OidcProviderConfig) -> Result<Self, OidcError> {
        let algorithms = config
            .pinned_algorithms()
            .map_err(|e| OidcError::Rejected(format!("invalid oidc config: {e}")))?;
        // F10: pin the fetch targets to https (loopback exempt for local
        // testing) so a misconfigured or hostile discovery document cannot
        // aim the JWKS fetch at cleartext internal endpoints.
        for (what, url) in [("issuer_url", &config.issuer_url)]
            .into_iter()
            .chain(config.jwks_url.as_ref().map(|u| ("jwks_url", u)))
        {
            let Ok(parsed) = reqwest::Url::parse(url) else {
                return Err(OidcError::Rejected(format!("invalid {what}: {url}")));
            };
            let scheme = parsed.scheme();
            let is_loopback = parsed
                .host_str()
                .is_some_and(|h| h.starts_with("127.") || h == "localhost" || h == "[::1]");
            if scheme != "https" && !(scheme == "http" && is_loopback) {
                return Err(OidcError::Rejected(format!(
                    "{what} must be https (http allowed only for loopback): {url}"
                )));
            }
        }
        // N2: no redirect following — a redirect chain (including scheme
        // downgrade) from a pinned-https URL must not re-aim the fetch.
        // E (portability): custom CA bundle for self-hosted IdPs on
        // internal PKI (Kanidm/Keycloak without public certs).
        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.http_timeout_seconds))
            .redirect(reqwest::redirect::Policy::none());
        if let Some(ca_path) = &config.ca_cert_path {
            let pem = std::fs::read_to_string(ca_path).map_err(|e| {
                OidcError::Rejected(format!("cannot read ca_cert_path {ca_path}: {e}"))
            })?;
            let cert = reqwest::Certificate::from_pem(pem.as_bytes())
                .map_err(|e| OidcError::Rejected(format!("bad CA PEM at {ca_path}: {e}")))?;
            builder = builder.add_root_certificate(cert);
        }
        let client = builder
            .build()
            .map_err(|e| OidcError::Rejected(format!("http client: {e}")))?;
        Ok(Self {
            config: Arc::new(config),
            algorithms,
            client,
            jwks: tokio::sync::RwLock::new(None),
            fetch_gate: tokio::sync::Mutex::new(()),
            last_forced_refresh: std::sync::Mutex::new(None),
        })
    }

    pub fn config(&self) -> &OidcProviderConfig {
        &self.config
    }

    /// The effective JWKS URL: explicit, or discovered.
    async fn jwks_url(&self) -> Result<String, OidcError> {
        if let Some(url) = &self.config.jwks_url {
            return Ok(url.clone());
        }
        let discovery = format!(
            "{}/.well-known/openid-configuration",
            self.config.issuer_url.trim_end_matches('/')
        );
        let resp = self
            .client
            .get(&discovery)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| OidcError::JwksUnavailable(format!("discovery: {e}")))?;
        let doc: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| OidcError::JwksUnavailable(format!("discovery body: {e}")))?;
        let uri = doc
            .get("jwks_uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| OidcError::JwksUnavailable("discovery lacks jwks_uri".into()))?;
        // N2: the discovery document is IdP-controlled — apply the SAME https
        // pin (loopback-exempt) to the URI it names, so a hostile or
        // misconfigured discovery response cannot aim the JWKS fetch at
        // cleartext/internal endpoints (SSRF from the DB's network position,
        // or a MITM-substituted key set = forged tokens).
        let parsed = reqwest::Url::parse(uri)
            .map_err(|e| OidcError::JwksUnavailable(format!("bad jwks_uri: {e}")))?;
        let is_loopback = parsed
            .host_str()
            .is_some_and(|h| h.starts_with("127.") || h == "localhost" || h == "[::1]");
        if parsed.scheme() != "https" && !(parsed.scheme() == "http" && is_loopback) {
            return Err(OidcError::JwksUnavailable(format!(
                "discovery jwks_uri must be https (http allowed only for loopback): {uri}"
            )));
        }
        Ok(uri.to_string())
    }

    async fn fetch_jwks(&self) -> Result<Arc<JwkSet>, OidcError> {
        // Single-flight (N1): hold the gate across the fetch AND recheck the
        // cache after acquiring it — the winner performs one outbound
        // request; everyone queued behind the gate serves the fresh result.
        // Without the recheck, a TTL-expiry or failing-fetch burst performs
        // one fetch PER request (serialized), which is the F1 storm via the
        // non-forced path.
        let _gate = self.fetch_gate.lock().await;
        if let Some(cached) = self.jwks.read().await.as_ref()
            && cached.fetched_at.elapsed() < Duration::from_secs(self.config.jwks_cache_seconds)
        {
            return Ok(Arc::clone(&cached.set));
        }
        let url = self.jwks_url().await?;
        let set: JwkSet = self
            .client
            .get(&url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| OidcError::JwksUnavailable(format!("{url}: {e}")))?
            .json()
            .await
            .map_err(|e| OidcError::JwksUnavailable(format!("{url} body: {e}")))?;
        let set = Arc::new(set);
        *self.jwks.write().await = Some(CachedJwks {
            set: Arc::clone(&set),
            fetched_at: Instant::now(),
        });
        Ok(set)
    }

    async fn jwks(&self, force_refresh: bool) -> Result<Arc<JwkSet>, OidcError> {
        if !force_refresh
            && let Some(cached) = self.jwks.read().await.as_ref()
            && cached.fetched_at.elapsed() < Duration::from_secs(self.config.jwks_cache_seconds)
        {
            return Ok(Arc::clone(&cached.set));
        }
        // N1: debounce BOTH paths — a TTL-expiry burst must also collapse to
        // one fetch per window, and a FAILING fetch (IdP 429/5xx under
        // exactly this pressure) must not let every request retry outbound.
        // The std guard is dropped BEFORE the await (a guard held across an
        // await makes the future !Send and infects every middleware).
        if !force_refresh {
            let throttled = match self.last_forced_refresh.lock() {
                Ok(mut last) => {
                    if last.is_some_and(|t| t.elapsed() < FORCED_REFRESH_MIN_INTERVAL) {
                        true
                    } else {
                        *last = Some(Instant::now());
                        false
                    }
                }
                Err(_) => false,
            };
            if throttled && let Some(cached) = self.jwks.read().await.as_ref() {
                // Stale-but-present is fail-closed for rotation (rejects
                // unknown kids); serving it within the 5s window bounds
                // outbound load without accepting anything extra.
                return Ok(Arc::clone(&cached.set));
            }
        }
        if force_refresh {
            // F1 throttle: at most one forced refresh per window. A burst of
            // garbage-kid tokens degrades to "unknown kid" errors instead of
            // an outbound fetch per request. The std guard is dropped BEFORE
            // any await (a guard held across await makes the future !Send
            // and infects every calling middleware).
            let throttled = {
                match self.last_forced_refresh.lock() {
                    Ok(mut last) => {
                        if last.is_some_and(|t| t.elapsed() < FORCED_REFRESH_MIN_INTERVAL) {
                            true
                        } else {
                            *last = Some(Instant::now());
                            false
                        }
                    }
                    Err(_) => false,
                }
            };
            if throttled {
                // Serve the (stale) cache if present, else fail closed.
                if let Some(cached) = self.jwks.read().await.as_ref() {
                    return Ok(Arc::clone(&cached.set));
                }
                return Err(OidcError::JwksUnavailable(
                    "unknown kid and refresh throttled (no cache yet)".into(),
                ));
            }
        }
        self.fetch_jwks().await
    }

    fn key_for<'s>(set: &'s JwkSet, kid: Option<&str>) -> Option<&'s Jwk> {
        match kid {
            Some(kid) => set
                .keys
                .iter()
                .find(|k| k.common.key_id.as_deref() == Some(kid)),
            // No kid: acceptable only when the set holds exactly one key.
            None => {
                if set.keys.len() == 1 {
                    set.keys.first()
                } else {
                    None
                }
            }
        }
    }

    fn decoding_key(jwk: &Jwk) -> Result<DecodingKey, OidcError> {
        match &jwk.algorithm {
            AlgorithmParameters::RSA(_) => DecodingKey::from_jwk(jwk)
                .map_err(|e| OidcError::Rejected(format!("bad RSA JWK: {e}"))),
            AlgorithmParameters::EllipticCurve(_) => DecodingKey::from_jwk(jwk)
                .map_err(|e| OidcError::Rejected(format!("bad EC JWK: {e}"))),
            other => Err(OidcError::Rejected(format!(
                "JWK key type {:?} is not supported (RSA and EC only)",
                other
            ))),
        }
    }

    /// Verify a bearer token strictly: pinned algorithms, issuer, audience,
    /// exp (60s leeway). Returns the verified claims; `roles_raw`/`tenant_raw`
    /// carry the configured claims' raw values for the conversion seam.
    pub async fn verify(&self, token: &str) -> Result<OidcClaims, OidcError> {
        let header = jsonwebtoken::decode_header(token)
            .map_err(|e| OidcError::Rejected(format!("malformed header: {e}")))?;
        if !self.algorithms.contains(&header.alg) {
            return Err(OidcError::Rejected(format!(
                "algorithm {:?} not in the pinned set — confusion defense",
                header.alg
            )));
        }

        let set = self.jwks(false).await?;
        // Both branches resolve to an OWNED DecodingKey so no Jwk outlives its
        // JwkSet borrow.
        let decoding_key = match Self::key_for(&set, header.kid.as_deref()) {
            Some(jwk) => Self::decoding_key(jwk)?,
            // Unknown kid: force one refresh, then retry the lookup.
            None => {
                let refreshed = self.jwks(true).await?;
                let jwk = Self::key_for(&refreshed, header.kid.as_deref()).ok_or_else(|| {
                    OidcError::Rejected(format!(
                        "no JWK matches kid {:?} even after refresh",
                        header.kid
                    ))
                })?;
                Self::decoding_key(jwk)?
            }
        };

        let mut validation = Validation::new(header.alg);
        // C (portability): multiple accepted issuers (Google's bare-hostname
        // `iss` alongside the URL form) and multiple audiences (Azure's
        // `api://{client-id}` + bare `{client-id}`). jsonwebtoken's
        // set_issuer/set_audience already accept any-of semantics.
        // Trailing-slash normalization: discovery URL building already trims
        // the issuer, so an operator-supplied trailing slash would otherwise
        // boot cleanly and then reject every token (fail-closed footgun).
        // IdPs emit canonical no-slash `iss` claims; trim the configured side.
        let mut issuers = vec![self.config.issuer_url.trim_end_matches('/').to_string()];
        issuers.extend(
            self.config
                .issuer_aliases
                .iter()
                .map(|a| a.trim_end_matches('/').to_string()),
        );
        validation.set_issuer(&issuers);
        validation.set_audience(&self.config.audience);
        validation.leeway = self.config.clock_skew_seconds;
        validation.validate_nbf = true;

        let raw: serde_json::Value = jsonwebtoken::decode(token, &decoding_key, &validation)
            .map_err(|e| OidcError::Rejected(format!("validation: {e}")))?
            .claims;

        // Post-verification extraction of the configured claims (they are NOT
        // part of the typed struct on the wire — pull them from the same
        // verified payload).
        let roles_raw = claim_at_path(&raw, &self.config.roles_claim)
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let tenant_raw = self
            .config
            .tenant_claim
            .as_ref()
            .and_then(|claim| claim_at_path(&raw, claim))
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let mut claims: OidcClaims = serde_json::from_value(raw)
            .map_err(|e| OidcError::Rejected(format!("claims shape: {e}")))?;
        if !self
            .config
            .audience
            .iter()
            .any(|expected| claims.aud.contains(expected))
        {
            // Defense in depth: jsonwebtoken already validated aud; keep the
            // explicit check so a library regression cannot open the door.
            return Err(OidcError::Rejected("audience mismatch".into()));
        }
        claims.roles_raw = roles_raw;
        claims.tenant_raw = tenant_raw;
        Ok(claims)
    }
}

/// Walk a dot-separated path through a JSON object (e.g.
/// `realm_access.roles` for Keycloak's nested role claim). Returns `None`
/// if any segment is missing or not an object. A path with no dots is a
/// simple top-level lookup (unchanged behavior).
pub fn claim_at_path<'a>(root: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    // TD-PROTO-2 follow-up: delegate to the shared object-only dot-path
    // walker. Non-objects, including arrays, remain a dead end so query-side
    // traversal changes cannot silently broaden authorization semantics.
    proximadb_search_types::sql_value_filter::json_get_path(root, path.split('.'))
}

/// Flatten the roles claim's raw value to strings (array of strings, or a
/// single string). Anything else reads as no roles — never an error.
pub fn roles_raw_to_strings(raw: &serde_json::Value) -> Vec<String> {
    match raw {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        serde_json::Value::String(s) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Flatten the tenant claim to an optional tenant id.
pub fn tenant_raw_to_option(raw: &serde_json::Value) -> Option<String> {
    raw.as_str().map(str::to_owned)
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    /// Shared fixtures (F5/F6): the production-seam test in
    /// `security::auth_service` needs the same keypair + JWKS + signer.
    pub const TEST_RSA_PEM: &str = include_str!("../../../tests/fixtures/oidc_test_key.pem");
    pub const TEST_JWKS_JSON: &str = include_str!("../../../tests/fixtures/oidc_test_jwks.json");

    pub fn verifier(allowlist: &[&str], allow_delegation: bool) -> OidcProviderConfig {
        OidcProviderConfig {
            enabled: true,
            issuer_url: "https://idp.example.test".into(),
            audience: vec!["proximadb".to_string()],
            jwks_url: None,
            allowed_algorithms: default_algorithms(),
            roles_claim: default_roles_claim(),
            tenant_claim: Some("tenant".into()),
            role_allowlist: allowlist.iter().map(|s| s.to_string()).collect(),
            allow_delegation_roles: allow_delegation,
            issuer_aliases: Vec::new(),
            clock_skew_seconds: 60,
            jwks_cache_seconds: 3600,
            http_timeout_seconds: 10,
            ca_cert_path: None,
        }
    }

    pub fn sign_rs256(claims: &serde_json::Value) -> String {
        use jsonwebtoken::{EncodingKey, Header};
        let key = EncodingKey::from_rsa_pem(TEST_RSA_PEM.as_bytes()).expect("test key");
        jsonwebtoken::encode(&Header::new(jsonwebtoken::Algorithm::RS256), claims, &key)
            .expect("sign")
    }

    pub fn std_claims(now: i64) -> serde_json::Value {
        serde_json::json!({
            "iss": "https://idp.example.test",
            "sub": "user-1",
            "aud": "proximadb",
            "exp": now + 600,
            "iat": now,
            "groups": ["gateway", "analyst"],
            "tenant": "tenant-9"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(allowlist: &[&str]) -> OidcProviderConfig {
        test_fixtures::verifier(allowlist, true)
    }

    #[test]
    fn sanitize_truth_table() {
        let fail_closed = cfg(&[]);
        // cfg() explicitly enables delegation (as an opted-in operator would);
        // business roles not in the allowlist never pass.
        let out = sanitize_idp_roles(
            &["gateway".into(), "analyst".into(), "gateways".into()],
            &fail_closed,
        );
        assert_eq!(out, vec!["gateway".to_string()]);

        // Operator-configured allowlist admits the business role deliberately.
        let allowed = cfg(&["analyst"]);
        let out = sanitize_idp_roles(
            &["gateway".into(), "analyst".into(), "admin".into()],
            &allowed,
        );
        assert_eq!(out, vec!["gateway".to_string(), "analyst".to_string()]);

        // Delegation off ⇒ the marker is dropped too.
        let mut no_delegation = cfg(&[]);
        no_delegation.allow_delegation_roles = false;
        assert!(sanitize_idp_roles(&["gateway".into()], &no_delegation).is_empty());

        // Duplicates collapse; empty input is empty.
        assert_eq!(
            sanitize_idp_roles(&["gateway".into(), "gateway".into()], &fail_closed),
            vec!["gateway".to_string()]
        );
        assert!(sanitize_idp_roles(&[], &fail_closed).is_empty());
    }

    #[test]
    fn hmac_algorithms_are_rejected_at_config_time() {
        let mut c = cfg(&[]);
        c.allowed_algorithms = vec!["HS256".into()];
        assert!(c.pinned_algorithms().is_err());
        let mut c2 = cfg(&[]);
        c2.allowed_algorithms = vec!["RS256".into(), "nonsense".into()];
        assert!(c2.pinned_algorithms().is_err());
        assert!(cfg(&[]).pinned_algorithms().is_ok());
    }

    #[test]
    fn claim_flatten_helpers() {
        let roles = serde_json::json!(["a", "b"]);
        assert_eq!(
            roles_raw_to_strings(&roles),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(
            roles_raw_to_strings(&serde_json::json!("solo")),
            vec!["solo".to_string()]
        );
        assert!(roles_raw_to_strings(&serde_json::json!(42)).is_empty());
        assert_eq!(
            tenant_raw_to_option(&serde_json::json!("tenant-7")),
            Some("tenant-7".to_string())
        );
        assert_eq!(tenant_raw_to_option(&serde_json::Value::Null), None);
    }

    /// D (portability): nested claim paths — Keycloak's `realm_access.roles`.
    #[test]
    fn nested_claim_path_resolves() {
        let root = serde_json::json!({
            "realm_access": {"roles": ["admin", "user"]},
            "groups": ["top-level"],
            "tid": "tenant-42"
        });
        // Dot-path walks nested objects
        let roles = claim_at_path(&root, "realm_access.roles");
        assert_eq!(roles.and_then(|v| v.as_array()).map(|a| a.len()), Some(2));
        // Simple key still works (no dots = top-level; groups is an array here)
        let groups = claim_at_path(&root, "groups").and_then(|v| v.as_array());
        assert_eq!(groups.map(|a| a.len()), Some(1));
        // Tenant claim with dot-path
        assert_eq!(
            claim_at_path(&root, "tid").and_then(|v| v.as_str()),
            Some("tenant-42")
        );
        // Missing segments → None (not an error)
        assert!(claim_at_path(&root, "nonexistent.roles").is_none());
        assert!(claim_at_path(&root, "realm_access.nonexistent").is_none());
        // Authorization paths are object-only: numeric segments do not index
        // arrays, and scalar values cannot be traversed.
        assert!(claim_at_path(&root, "groups.0").is_none());
        assert!(claim_at_path(&root, "tid.child").is_none());
    }

    /// C (portability): audience accepts any-of from the list.
    #[test]
    fn audience_list_backward_compat() {
        let json_one = r#"{"enabled":true,"issuer_url":"https://x","audience":"single","jwks_url":null,"allowed_algorithms":["RS256"],"roles_claim":"groups","tenant_claim":null,"role_allowlist":[],"allow_delegation_roles":false,"issuer_aliases":[],"clock_skew_seconds":60,"jwks_cache_seconds":3600,"http_timeout_seconds":10,"ca_cert_path":null}"#;
        let cfg: OidcProviderConfig = serde_json::from_str(json_one).expect("single string");
        assert_eq!(cfg.audience, vec!["single".to_string()]);

        let json_many = r#"{"enabled":true,"issuer_url":"https://x","audience":["api://my","my"],"jwks_url":null,"allowed_algorithms":["RS256"],"roles_claim":"groups","tenant_claim":null,"role_allowlist":[],"allow_delegation_roles":false,"issuer_aliases":["accounts.google.com"],"clock_skew_seconds":120,"jwks_cache_seconds":1800,"http_timeout_seconds":5,"ca_cert_path":"/etc/ca.pem"}"#;
        let cfg2: OidcProviderConfig = serde_json::from_str(json_many).expect("list");
        assert_eq!(cfg2.audience.len(), 2);
        assert_eq!(cfg2.issuer_aliases, vec!["accounts.google.com".to_string()]);
        assert_eq!(cfg2.clock_skew_seconds, 120);
        assert_eq!(cfg2.jwks_cache_seconds, 1800);
        assert_eq!(cfg2.http_timeout_seconds, 5);
        assert_eq!(cfg2.ca_cert_path.as_deref(), Some("/etc/ca.pem"));
    }

    /// B (portability): ES256 is accepted at config load.
    #[test]
    fn es256_accepted_in_pinned_algorithms() {
        let mut c = test_fixtures::verifier(&[], false);
        c.allowed_algorithms = vec!["ES256".to_string()];
        assert!(c.pinned_algorithms().is_ok(), "ES256 should be accepted");
        let mut c2 = test_fixtures::verifier(&[], false);
        c2.allowed_algorithms = vec!["RS256".to_string(), "ES384".to_string()];
        assert!(c2.pinned_algorithms().is_ok());
        // HMAC still rejected
        let mut c3 = test_fixtures::verifier(&[], false);
        c3.allowed_algorithms = vec!["HS256".to_string()];
        assert!(c3.pinned_algorithms().is_err());
    }

    /// F-3 (third-pass review): REAL end-to-end ES256 — signs with EC P-256
    /// and verifies through the full OidcTokenVerifier pipeline. The prior
    /// placeholder only checked config acceptance; sabotaging the EC arm in
    /// decoding_key or EC verification would have passed.
    #[tokio::test]
    async fn es256_end_to_end_token_verifies() {
        const EC_JWKS_JSON: &str = include_str!("../../../tests/fixtures/ec_test_jwks.json");

        // Serve the EC JWKS
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(EC_JWKS_JSON);
            })
            .await;

        // Configure for ES256
        let cfg = OidcProviderConfig {
            jwks_url: Some(format!("{}/jwks", server.base_url())),
            allowed_algorithms: vec!["ES256".to_string()],
            ..test_fixtures::verifier(&[], false)
        };
        let verifier = OidcTokenVerifier::new(cfg).expect("verifier");

        // Sign a token with EC P-256 (ES256)
        let now = chrono::Utc::now().timestamp();
        let claims = std_claims(now);
        let header_json = serde_json::json!({"alg":"ES256","kid":"test-ec-key-1","typ":"JWT"});
        let signing_input = format!(
            "{}.{}",
            base64_url(serde_json::to_vec(&header_json).unwrap()),
            base64_url(serde_json::to_vec(&claims).unwrap())
        );
        let sig = openssl_es256_sign(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/ec_test_key.pem"
            ),
            signing_input.as_bytes(),
        )
        .expect("EC sign");
        let token = format!("{}.{}", signing_input, sig);

        // Verify through the full pipeline
        let verified = verifier
            .verify(&token)
            .await
            .expect("ES256 token must verify end-to-end");
        assert_eq!(verified.sub, "user-1");
    }

    /// Helper: base64url without padding.
    fn base64_url(data: Vec<u8>) -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
    }

    /// Helper: sign with EC P-256 via openssl CLI, converting the DER
    /// signature to the JWS raw R||S format (64 bytes) that ES256 requires.
    fn openssl_es256_sign(pem_path: &str, input: &[u8]) -> Option<String> {
        use std::io::Write;
        let output = std::process::Command::new("openssl")
            .args(["dgst", "-sha256", "-sign"])
            .arg(pem_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()
            .and_then(|mut c| {
                if let Some(mut stdin) = c.stdin.take() {
                    let _ = stdin.write_all(input);
                }
                c.wait_with_output().ok()
            })?;
        let der = output.stdout;
        let raw = der_to_raw_rs(&der)?;
        Some(base64_url(raw))
    }

    /// Convert a DER-encoded ECDSA signature to the raw R||S format.
    fn der_to_raw_rs(der: &[u8]) -> Option<Vec<u8>> {
        if der.len() < 8 || der[0] != 0x30 {
            return None;
        }
        let mut pos = 2;
        let mut parts: Vec<u8> = Vec::with_capacity(64);
        for _ in 0..2 {
            if der[pos] != 0x02 {
                return None;
            }
            pos += 1;
            let len = der[pos] as usize;
            pos += 1;
            let val = &der[pos..pos + len];
            let stripped = val.iter().position(|&b| b != 0).unwrap_or(len);
            let val = &val[stripped..];
            let mut padded = [0u8; 32];
            let copy_len = val.len().min(32);
            padded[32 - copy_len..].copy_from_slice(&val[val.len() - copy_len..]);
            parts.extend_from_slice(&padded);
            pos += len;
        }
        if parts.len() == 64 { Some(parts) } else { None }
    }

    /// C/I (adversarial review): multi-audience any-of actually works at
    /// the verify() level — not just serde.
    #[tokio::test]
    async fn second_audience_accepted_and_wrong_still_rejected() {
        let (_server, url) = mock_jwks_server().await;
        let mut cfg = oidc_cfg_for(url, &[]);
        cfg.audience = vec!["first".to_string(), "second".to_string()];
        let verifier = OidcTokenVerifier::new(cfg).expect("verifier");
        let now = chrono::Utc::now().timestamp();

        // Token with the SECOND audience → accepted (any-of)
        let mut claims = std_claims(now);
        claims["aud"] = serde_json::json!("second");
        let token = sign_rs256(&claims);
        let result = verifier.verify(&token).await;
        assert!(
            result.is_ok(),
            "second audience must be accepted: {result:?}"
        );

        // Wrong audience → still rejected
        let mut wrong = std_claims(now);
        wrong["aud"] = serde_json::json!("nobody");
        let wrong_token = sign_rs256(&wrong);
        assert!(verifier.verify(&wrong_token).await.is_err());
    }

    /// C/I (adversarial review): issuer alias actually works at verify().
    #[tokio::test]
    async fn issuer_alias_accepted_and_non_alias_rejected() {
        let (_server, url) = mock_jwks_server().await;
        let mut cfg = oidc_cfg_for(url, &[]);
        cfg.issuer_aliases = vec!["accounts.google.com".to_string()];
        let verifier = OidcTokenVerifier::new(cfg).expect("verifier");
        let now = chrono::Utc::now().timestamp();

        // Token with the ALIAS issuer → accepted
        let mut claims = std_claims(now);
        claims["iss"] = serde_json::json!("accounts.google.com");
        let token = sign_rs256(&claims);
        let result = verifier.verify(&token).await;
        assert!(result.is_ok(), "alias issuer must be accepted: {result:?}");

        // Non-issuer non-alias → rejected
        let mut wrong = std_claims(now);
        wrong["iss"] = serde_json::json!("https://evil.example.test");
        let wrong_token = sign_rs256(&wrong);
        assert!(verifier.verify(&wrong_token).await.is_err());
    }

    /// R4 (adversarial review): trailing-slash issuer config must not reject
    /// canonical tokens — discovery trims, so validation must too.
    #[tokio::test]
    async fn trailing_slash_issuer_config_still_validates() {
        let (_server, url) = mock_jwks_server().await;
        let mut cfg = oidc_cfg_for(url, &[]);
        let trimmed = cfg.issuer_url.clone();
        cfg.issuer_url = format!("{}/", trimmed); // operator adds trailing slash
        let verifier = OidcTokenVerifier::new(cfg).expect("verifier");
        let now = chrono::Utc::now().timestamp();

        // Token carries the canonical no-slash issuer → must still pass
        let claims = std_claims(now);
        let token = sign_rs256(&claims);
        let result = verifier.verify(&token).await;
        assert!(
            result.is_ok(),
            "trailing-slash config must accept canonical iss: {result:?}"
        );
    }

    /// D (adversarial review): nested claim path through the full verify().
    #[tokio::test]
    async fn nested_roles_claim_resolves_through_verify() {
        let (_server, url) = mock_jwks_server().await;
        let mut cfg = oidc_cfg_for(url, &["analyst"]);
        cfg.roles_claim = "realm_access.roles".to_string();
        let verifier = OidcTokenVerifier::new(cfg).expect("verifier");
        let now = chrono::Utc::now().timestamp();

        // Token with Keycloak-style nested roles
        let mut claims = std_claims(now);
        claims["realm_access"] = serde_json::json!({"roles": ["analyst", "admin"]});
        let token = sign_rs256(&claims);
        let verified = verifier.verify(&token).await.expect("verified");
        let raw = roles_raw_to_strings(&verified.roles_raw);
        assert_eq!(raw.len(), 2, "nested roles must resolve: {raw:?}");
    }

    #[test]
    fn aud_polymorphism() {
        assert!(Aud::One("x".into()).contains("x"));
        assert!(Aud::Many(vec!["y".into(), "x".into()]).contains("x"));
        assert!(!Aud::Many(vec!["y".into()]).contains("x"));
    }
    // ---- TD-SSO-1 httpmock adversarial suite ----
    // Fixed test RSA keypair (kid "test-key-1"), generated once and embedded:
    // the private PEM signs tokens; the JWKS serves only the PUBLIC JWK.
    const TEST_RSA_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCyGYz1pk+SUIWu
nPviZvX7TcaHZAt81bPRTSKlwfkMLgu37dCAIe1nyIWUEb4OtkflVd6mgMwDJuor
9DDTAwEVv4+iY3fDROsgysCAis/SSAxX4AhQaamn9eI77n8YiFXuiAq5q0XuxPpO
eq4U42su1omKGVdUWMwZJsGbSOGXgZ66461eyjJdt7NtohmHyru5B3rmQ3CyoY7P
Sch0Riq64OLzvDrUU+JvC4E6zG2WYNjGJLptxJ7MCYZ+W5LCen1oGqeJugTvTmQh
u66dJKMOSu8oJzYnJf3s2uV1rpS8ineAbclAZwlEjHK+gUOUAT/9ia6cdV6PaqWO
XS9kazm9AgMBAAECggEAG09as6J8iimpziRJZaa3KoF7I3C+aDaW283xtaxAZdM7
vckMU1Ggh20Suqlb1QKzlKGtwid24Tba6sGHqRvJU03sFxEcoMdVLNKaYTun8Y1V
wzXZ4IbLWrOULO998sOZNboLtvvh/eKbpWQfhJl46pZAQfpvz0HMjkDIqGitGx+m
PBglSi4jUro0fqZwh1rmSB5EF8jBgux62ausGqczNvht+/IbfejLR+T0t5JS34iO
ETdLebZvQsaCnx1IWM6ofNce8dqUYdQ4q+JA9dJ2rzDQXpPkpHNn7YV/vL/pGWfM
R8LfQeSAhHDH6vZZwlRjcHiMyoBheLqxGnHVS8ZLgQKBgQDXMxr+3AsE/9+S3D/V
x7JC3twUgS+i9BIWMLPkZmVrTPDk4Kc9mjCsjTYDLn4W4BJibVvd3PpFba1bV6h6
LlE6wDhPs8O/4izyBtb6oPMRGj2O9sYB+qRxhCv+zcGG9/VZflgVG+5aU6qSVUyC
biyZppsfVSBCjEHPOBIirvTowQKBgQDT3cYs6qtfBMYmVCVw7v6wYoe2R5uffe7w
vOLabfUuY7F7DXz2ZlCTHKJqE2mFUf35lNsIR1NzSG7OKxIgnQ0q+lPiwiSGOoXD
DvXz73iq6fBsUjv+WfnJ1A69bs9Kl26/sINx3YhgqT37KkQViclYYVnZXkEtQWAY
K8DcPfLz/QKBgCSHy1xaFBDMMrKmaruqg4swc6GTcHe0AOH9cHwkGbFGRVpE/H7L
jtmruvB9UvAlJ1nIAKE/4sgoXxYzYikjdayIdsao2GDZTxHisVmoOrq7fpmnMGOj
nYibjDBb0y9LJj4D4YXr0OFKdJkUm8FEXJPUoV6HP8usLXu0o/d5RZ/BAoGBAJt2
zybChEHTJPuXH2pBVU5k3qTY4s8j/6NTVztlGFaT/PYIrbu41EM+7cbcu7+CrNTp
b9ghTpD3g6dxX3njBHiW+9sXDuoYI3NAlSYMgQUZaLxzk2ZO68Y3/yDuINnhSPkM
M0fogVw8lCirmQ4c70wVE3M3gKgOos7ZvElgg9iZAoGARhyfqWM+Qh7zPCpCUgp6
2X0joWW8wirmPekVLjio+rpmxlXIdcjaMvnII01zOiLKB+vPRUMRHBkHfMhkdt2Y
HpMnxMbxQ96sWweZILlro7ShMNp8B/iMQuXhlrkWZqedDUmGDvCznDWoFq0Nfbjt
+zN3jV8qMXThUgiy4dbpG5o=
-----END PRIVATE KEY-----"#;
    const TEST_JWKS_JSON: &str = r#"{"keys": [{"kty": "RSA", "use": "sig", "alg": "RS256", "kid": "test-key-1", "n": "shmM9aZPklCFrpz74mb1-03Gh2QLfNWz0U0ipcH5DC4Lt-3QgCHtZ8iFlBG-DrZH5VXepoDMAybqK_Qw0wMBFb-PomN3w0TrIMrAgIrP0kgMV-AIUGmpp_XiO-5_GIhV7ogKuatF7sT6TnquFONrLtaJihlXVFjMGSbBm0jhl4GeuuOtXsoyXbezbaIZh8q7uQd65kNwsqGOz0nIdEYquuDi87w61FPibwuBOsxtlmDYxiS6bcSezAmGfluSwnp9aBqniboE705kIbuunSSjDkrvKCc2JyX97Nrlda6UvIp3gG3JQGcJRIxyvoFDlAE__YmunHVej2qljl0vZGs5vQ", "e": "AQAB"}]}"#;

    async fn mock_jwks_server() -> (httpmock::MockServer, String) {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/jwks");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(TEST_JWKS_JSON);
            })
            .await;
        let url = format!("{}/jwks", server.base_url());
        (server, url)
    }

    fn oidc_cfg_for(jwks_url: String, allowlist: &[&str]) -> OidcProviderConfig {
        OidcProviderConfig {
            enabled: true,
            issuer_url: "https://idp.example.test".into(),
            audience: vec!["proximadb".to_string()],
            jwks_url: Some(jwks_url),
            allowed_algorithms: default_algorithms(),
            roles_claim: default_roles_claim(),
            tenant_claim: Some("tenant".into()),
            role_allowlist: allowlist.iter().map(|s| s.to_string()).collect(),
            allow_delegation_roles: true,
            issuer_aliases: Vec::new(),
            clock_skew_seconds: 60,
            jwks_cache_seconds: 3600,
            http_timeout_seconds: 10,
            ca_cert_path: None,
        }
    }

    fn sign_rs256(claims: &serde_json::Value) -> String {
        use jsonwebtoken::{EncodingKey, Header};
        let key = EncodingKey::from_rsa_pem(TEST_RSA_PEM.as_bytes()).expect("test key");
        jsonwebtoken::encode(&Header::new(jsonwebtoken::Algorithm::RS256), claims, &key)
            .expect("sign")
    }

    fn std_claims(now: i64) -> serde_json::Value {
        serde_json::json!({
            "iss": "https://idp.example.test",
            "sub": "user-1",
            "aud": "proximadb",
            "exp": now + 600,
            "iat": now,
            "groups": ["gateway", "analyst"],
            "tenant": "tenant-9"
        })
    }

    #[tokio::test]
    async fn happy_path_verifies_sanitizes_and_maps_tenant() {
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &["analyst"])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let verified = verifier
            .verify(&sign_rs256(&std_claims(now)))
            .await
            .expect("verified");
        assert_eq!(verified.sub, "user-1");
        let raw = roles_raw_to_strings(&verified.roles_raw);
        let sanitized = sanitize_idp_roles(&raw, verifier.config());
        assert_eq!(
            sanitized,
            vec!["gateway".to_string(), "analyst".to_string()]
        );
        assert_eq!(
            tenant_raw_to_option(&verified.tenant_raw),
            Some("tenant-9".into())
        );
    }

    #[tokio::test]
    async fn wrong_issuer_and_audience_rejected() {
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let mut bad_iss = std_claims(now);
        bad_iss["iss"] = serde_json::json!("https://evil.example.test");
        assert!(verifier.verify(&sign_rs256(&bad_iss)).await.is_err());
        let mut bad_aud = std_claims(now);
        bad_aud["aud"] = serde_json::json!("somebody-else");
        assert!(verifier.verify(&sign_rs256(&bad_aud)).await.is_err());
    }

    #[tokio::test]
    async fn expired_tokens_rejected() {
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let mut expired = std_claims(now);
        expired["exp"] = serde_json::json!(now - 7200);
        assert!(verifier.verify(&sign_rs256(&expired)).await.is_err());
    }

    /// THE confusion defense: an HMAC token must be rejected by algorithm
    /// pinning, never verified against the RSA key.
    #[tokio::test]
    async fn hmac_confusion_token_rejected() {
        use jsonwebtoken::{EncodingKey, Header};
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        // Attacker signs HS256 using the public JWKS bytes as the secret.
        let key = EncodingKey::from_secret(TEST_JWKS_JSON.as_bytes());
        let token = jsonwebtoken::encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            &std_claims(now),
            &key,
        )
        .expect("attacker token");
        let err = verifier.verify(&token).await.expect_err("must reject");
        assert!(
            err.to_string().contains("not in the pinned set"),
            "got: {err}"
        );
    }

    /// F6 (adversarial review): signature enforcement has its own negative
    /// control — a token whose header (incl. kid) is VALID and claims pass
    /// every other check, but whose SIGNATURE BYTES are tampered, must be
    /// rejected. Every other negative test fails at alg/kid/iss/aud/exp;
    /// this one fails only if signature verification is skipped or weakened.
    #[tokio::test]
    async fn tampered_signature_rejected() {
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let token = sign_rs256(&std_claims(now));
        // Flip a byte in the signature segment (third part).
        let mut parts: Vec<String> = token.split('.').map(str::to_string).collect();
        let sig = parts[2].clone();
        let mut bytes = sig.into_bytes();
        let idx = bytes.len() / 2;
        bytes[idx] = if bytes[idx] == b'A' { b'B' } else { b'A' };
        parts[2] = String::from_utf8(bytes).expect("ascii b64");
        let tampered = parts.join(".");
        let err = verifier
            .verify(&tampered)
            .await
            .expect_err("tampered must fail");
        assert!(
            err.to_string().contains("validation"),
            "must fail at signature validation, got: {err}"
        );
    }

    /// alg:none is rejected structurally.
    #[tokio::test]
    async fn alg_none_rejected() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none","kid":"test-key-1"}"#);
        let payload = URL_SAFE_NO_PAD.encode(std_claims(now).to_string());
        let token = format!("{header}.{payload}.");
        let err = verifier.verify(&token).await.expect_err("must reject");
        // jsonwebtoken 10 rejects `alg: none` AT HEADER DECODE (the Algorithm
        // enum has no `none` variant) — structurally rejected one step BEFORE
        // the pin. Either rejection point satisfies the defense.
        let msg = err.to_string();
        assert!(
            msg.contains("malformed header") || msg.contains("not in the pinned set"),
            "got: {msg}"
        );
    }

    /// Rotation: an unknown kid forces one refresh; still-unknown fails closed.
    #[tokio::test]
    async fn unknown_kid_refreshes_then_fails_closed() {
        use jsonwebtoken::{EncodingKey, Header};
        let (_server, url) = mock_jwks_server().await;
        let verifier = OidcTokenVerifier::new(oidc_cfg_for(url, &[])).expect("verifier");
        let now = chrono::Utc::now().timestamp();
        let key = EncodingKey::from_rsa_pem(TEST_RSA_PEM.as_bytes()).unwrap();
        let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
        header.kid = Some("rotated-away-key".into());
        let token = jsonwebtoken::encode(&header, &std_claims(now), &key).unwrap();
        let err = verifier.verify(&token).await.expect_err("must fail closed");
        assert!(err.to_string().contains("no JWK matches"), "got: {err}");
    }

    /// Unreachable IdP: construction succeeds (boot-order robustness) and
    /// the request fails closed with a clear error.
    #[tokio::test]
    async fn unreachable_idp_fails_the_request_not_construction() {
        let dead = "http://127.0.0.1:1/jwks".to_string();
        let cfg = OidcProviderConfig {
            jwks_url: Some(dead.clone()),
            ..oidc_cfg_for(dead, &[])
        };
        let verifier = OidcTokenVerifier::new(cfg).expect("construction is offline");
        let now = chrono::Utc::now().timestamp();
        let err = verifier
            .verify(&sign_rs256(&std_claims(now)))
            .await
            .expect_err("closed");
        assert!(err.to_string().contains("JWKS unavailable"), "got: {err}");
    }
}
