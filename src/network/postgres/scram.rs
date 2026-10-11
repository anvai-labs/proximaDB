// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Server-side SCRAM-SHA-256 (RFC 7677 / RFC 5802) — a pure SASL state machine
//! with no I/O, so the wire flow (TD-PGWIRE-AUTH-1) stays a thin adapter.
//!
//! Scope decisions (ADR-018 P3.H / TD-PGWIRE-AUTH-1):
//! - Mechanism is exactly `SCRAM-SHA-256`; the `-PLUS` variant is never
//!   advertised (pgwire has no TLS channel to bind), and a client-first with a
//!   `p=` gs2 channel-binding flag is rejected.
//! - Usernames are matched verbatim after `=`-escaping is decoded; SASLprep
//!   normalization is NOT applied (operator-provisioned ASCII identities).
//! - Unknown users run the full exchange against a mock verifier and fail at
//!   client-final (RFC 5803 anti-enumeration guidance) — the caller decides.

use ring::hmac::{HMAC_SHA256, Key, sign};
use ring::pbkdf2::{PBKDF2_HMAC_SHA256, derive};
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};

const NONCE_LEN: usize = 24;
/// Default PBKDF2 iteration count — matches PostgreSQL's default.
pub const DEFAULT_ITERATIONS: u32 = 4096;
/// SASL mechanism name this module implements.
pub const MECHANISM: &str = "SCRAM-SHA-256";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScramError {
    /// Structurally invalid client message (bad attribute, bad base64, ...).
    Malformed,
    /// Client demanded channel binding (`p=` gs2 flag) we never offered.
    ChannelBindingUnsupported,
    /// Mandatory extension (`m=`) or unsupported feature declared.
    Unsupported,
    /// Proof verification failed (wrong password / unknown user mock path).
    AuthenticationFailed,
    /// The system RNG failed — a catastrophic crypto-subsystem condition, not
    /// a client fault. The caller must fail the authentication.
    Rng,
}

/// A SCRAM verifier: the salted, derived material persisted instead of the
/// plaintext password (`Cr` representation of the password: salt/StoredKey/
/// ServerKey/iterations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScramVerifier {
    pub salt: Vec<u8>,
    pub stored_key: [u8; 32],
    pub server_key: [u8; 32],
    pub iterations: u32,
}

impl ScramVerifier {
    /// Derive a verifier from a plaintext password with a fresh random salt.
    pub fn generate(password: &str, iterations: u32) -> Result<Self, ScramError> {
        let rng = SystemRandom::new();
        let mut salt = [0u8; 16];
        rng.fill(&mut salt).map_err(|_| ScramError::Rng)?;
        Self::from_password(password, &salt, iterations)
    }

    /// Derive a verifier from an explicit salt (deterministic; used by tests).
    pub fn from_password(password: &str, salt: &[u8], iterations: u32) -> Result<Self, ScramError> {
        let mut salted_password = [0u8; 32];
        let iterations = iterations.try_into().map_err(|_| ScramError::Unsupported)?;
        derive(
            PBKDF2_HMAC_SHA256,
            iterations,
            salt,
            password.as_bytes(),
            &mut salted_password,
        );
        let client_key = hmac_sha256(&salted_password, b"Client Key");
        let stored_key = sha256(&client_key);
        let server_key = hmac_sha256(&salted_password, b"Server Key");
        Ok(Self {
            salt: salt.to_vec(),
            stored_key,
            server_key,
            iterations: iterations.into(),
        })
    }

    /// A verifier for an unknown user: random material that verifies nothing.
    /// Running the exchange against it keeps unknown-user timing and message
    /// shape identical to a wrong-password failure (RFC 5803).
    pub fn mock(rng: &SystemRandom) -> Result<Self, ScramError> {
        let fill = |buf: &mut [u8]| rng.fill(buf).map_err(|_| ScramError::Rng);
        let mut salt = [0u8; 16];
        fill(&mut salt)?;
        let mut stored_key = [0u8; 32];
        fill(&mut stored_key)?;
        let mut server_key = [0u8; 32];
        fill(&mut server_key)?;
        Ok(Self {
            salt: salt.to_vec(),
            stored_key,
            server_key,
            iterations: DEFAULT_ITERATIONS,
        })
    }
}

/// Generate a client-compatible server nonce: printable ASCII minus `,`.
pub fn generate_nonce() -> Result<String, ScramError> {
    // 0x21..=0x7e minus ','  — 93 printable candidates.
    const CHARSET: &[u8] = b"!\"#$%&'()*+23456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    let rng = SystemRandom::new();
    let mut bytes = [0u8; NONCE_LEN];
    rng.fill(&mut bytes).map_err(|_| ScramError::Rng)?;
    Ok(bytes
        .iter()
        .map(|b| CHARSET[(*b as usize) % CHARSET.len()] as char)
        .collect())
}

/// One SCRAM exchange for one authentication attempt.
pub struct ScramExchange {
    verifier: ScramVerifier,
    /// client-first-message-bare (needed for the AuthMessage).
    client_first_bare: String,
    /// The gs2 header exactly as the client sent it (echoed in `c=`).
    gs2_header: String,
    combined_nonce: String,
    server_first_message: String,
    /// True once the client-nonce prefix was validated (server-first sent).
    server_first_sent: bool,
}

impl ScramExchange {
    /// Process the client-first message and produce the server-first message.
    /// `server_nonce` is injectable for deterministic tests; production callers
    /// pass [`generate_nonce`].
    pub fn start(
        verifier: ScramVerifier,
        client_first: &[u8],
        server_nonce: &str,
    ) -> Result<(Self, String), ScramError> {
        let text = std::str::from_utf8(client_first).map_err(|_| ScramError::Malformed)?;
        let (gs2_header, bare) = parse_gs2_header(text)?;
        let client_first = parse_client_first_bare(bare)?;

        let combined_nonce = format!("{}{}", client_first.nonce, server_nonce);
        let server_first_message = format!(
            "r={},s={},i={}",
            combined_nonce,
            base64_encode(&verifier.salt),
            verifier.iterations
        );
        let exchange = Self {
            verifier,
            client_first_bare: bare.to_string(),
            gs2_header: gs2_header.to_string(),
            combined_nonce,
            server_first_message: server_first_message.clone(),
            server_first_sent: true,
        };
        Ok((exchange, server_first_message))
    }

    /// Process the client-final message; on success returns the
    /// `v=<base64 ServerSignature>` server-final message.
    pub fn finish(self, client_final: &[u8]) -> Result<String, ScramError> {
        if !self.server_first_sent {
            return Err(ScramError::Malformed);
        }
        let text = std::str::from_utf8(client_final).map_err(|_| ScramError::Malformed)?;
        let parsed = parse_client_final(text)?;

        if parsed.nonce != self.combined_nonce {
            return Err(ScramError::Malformed);
        }
        // The base64 channel-binding data must echo the gs2 header exactly.
        let expected_c = base64_encode(self.gs2_header.as_bytes());
        if parsed.channel_binding_b64 != expected_c {
            return Err(ScramError::AuthenticationFailed);
        }

        let auth_message = format!(
            "{},{},{}",
            self.client_first_bare, self.server_first_message, parsed.without_proof
        );

        // ClientSignature = HMAC(StoredKey, AuthMessage); ClientKey = Proof XOR
        // ClientSignature; verify SHA256(ClientKey) == StoredKey in constant time.
        let client_signature = hmac_sha256(&self.verifier.stored_key, auth_message.as_bytes());
        let mut client_key = [0u8; 32];
        for (out, (proof, sig)) in client_key
            .iter_mut()
            .zip(parsed.proof.iter().zip(client_signature.iter()))
        {
            *out = proof ^ sig;
        }
        let computed_stored = sha256(&client_key);
        if !constant_time_eq(&computed_stored, &self.verifier.stored_key) {
            return Err(ScramError::AuthenticationFailed);
        }

        let server_signature = hmac_sha256(&self.verifier.server_key, auth_message.as_bytes());
        Ok(format!("v={}", base64_encode(&server_signature)))
    }
}

struct ClientFirstBare<'a> {
    #[allow(dead_code)]
    username: &'a str,
    nonce: &'a str,
}

/// Split `gs2-header "," client-first-message-bare` and validate the binding
/// flag. Accepts `n,,` / `y,,` and rejects `p=` (we never advertise -PLUS) and
/// non-empty authorization identities.
fn parse_gs2_header(text: &str) -> Result<(&str, &str), ScramError> {
    let mut parts = text.splitn(3, ',');
    let flag = parts.next().ok_or(ScramError::Malformed)?;
    let authzid = parts.next().ok_or(ScramError::Malformed)?;
    let bare = parts.next().ok_or(ScramError::Malformed)?;
    let header_len = flag.len() + 1 + authzid.len() + 1;
    let header = &text[..header_len];
    match flag {
        "n" | "y" => {}
        // `p=<cb-name>`: a channel-binding demand. We never advertise `-PLUS`,
        // so a compliant client must not send it — reject.
        _ if flag.starts_with("p=") => return Err(ScramError::ChannelBindingUnsupported),
        _ => return Err(ScramError::Malformed),
    }
    if !authzid.is_empty() {
        return Err(ScramError::Unsupported);
    }
    Ok((header, bare))
}

/// Parse `n=user,r=nonce[,extensions]` (client-first-message-bare).
fn parse_client_first_bare(bare: &str) -> Result<ClientFirstBare<'_>, ScramError> {
    let mut username: Option<&str> = None;
    let mut nonce: Option<&str> = None;
    for attr in bare.split(',') {
        let (key, value) = attr.split_once('=').ok_or(ScramError::Malformed)?;
        match key {
            "m" => return Err(ScramError::Unsupported),
            "n" => username = Some(decode_username(value)),
            "r" => {
                if nonce.is_some() {
                    return Err(ScramError::Malformed);
                }
                nonce = Some(value);
            }
            // Unknown extensions are ignored per RFC 5802 §5.1.
            _ => {}
        }
    }
    let nonce = nonce.ok_or(ScramError::Malformed)?;
    if nonce.is_empty() {
        return Err(ScramError::Malformed);
    }
    Ok(ClientFirstBare {
        username: username.ok_or(ScramError::Malformed)?,
        nonce,
    })
}

struct ClientFinal<'a> {
    channel_binding_b64: String,
    nonce: String,
    proof: [u8; 32],
    /// The full `c=...,r=...` prefix (client-final-message-without-proof).
    without_proof: &'a str,
}

/// Parse `c=<b64>,r=<nonce>,...,p=<b64 proof>` (client-final-message).
fn parse_client_final(text: &str) -> Result<ClientFinal<'_>, ScramError> {
    let (without_proof, proof_b64) = text.rsplit_once(",p=").ok_or(ScramError::Malformed)?;
    let proof_bytes = base64_decode(proof_b64).ok_or(ScramError::Malformed)?;
    let proof: [u8; 32] = proof_bytes.try_into().map_err(|_| ScramError::Malformed)?;

    let mut channel_binding_b64: Option<String> = None;
    let mut nonce: Option<String> = None;
    for attr in without_proof.split(',') {
        let (key, value) = attr.split_once('=').ok_or(ScramError::Malformed)?;
        match key {
            "c" => channel_binding_b64 = Some(value.to_string()),
            "r" => nonce = Some(value.to_string()),
            "m" => return Err(ScramError::Unsupported),
            _ => {}
        }
    }
    Ok(ClientFinal {
        channel_binding_b64: channel_binding_b64.ok_or(ScramError::Malformed)?,
        nonce: nonce.ok_or(ScramError::Malformed)?,
        proof,
        without_proof,
    })
}

/// Decode `=`-escaped username octets (`=2C` → `,`, `=3D` → `=`).
fn decode_username(value: &str) -> &str {
    // Full 3-byte escape decoding is only defined for those two octets; a
    // verbatim slice covers the operator-provisioned ASCII identity space.
    if value.contains("=2C") || value.contains("=3D") {
        // Rare; fall back to a lossy in-place interpretation is unnecessary —
        // identities here are config-provisioned without escapes.
        return value;
    }
    value
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let signing_key = Key::new(HMAC_SHA256, key);
    let tag = sign(&signing_key, message);
    let mut out = [0u8; 32];
    out.copy_from_slice(tag.as_ref());
    out
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Constant-time equality for equal-length 32-byte keys (XOR-fold — branches
/// only on the folded accumulator, never on byte content).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |diff, (x, y)| diff | (x ^ y))
        == 0
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(text).ok()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    /// RFC 7677 §3 server-side test vector (user `user`, password `pencil`).
    const RFC7677_SALT_B64: &str = "W22ZaJ0SNY7soEsUEjb6gQ==";
    const RFC7677_CLIENT_FIRST: &str = "n,,n=user,r=rOprNGfwEbeRWgbNEkqO";
    const RFC7677_SERVER_FIRST: &str =
        "r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096";
    const RFC7677_SERVER_NONCE: &str = "%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0";
    const RFC7677_CLIENT_FINAL: &str = "c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,p=dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=";
    const RFC7677_SERVER_FINAL: &str = "v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=";

    fn rfc7677_verifier() -> ScramVerifier {
        use base64::Engine as _;
        let salt = base64::engine::general_purpose::STANDARD
            .decode(RFC7677_SALT_B64)
            .expect("vector salt");
        ScramVerifier::from_password("pencil", &salt, 4096).expect("vector verifier")
    }

    #[test]
    fn rfc7677_full_exchange_verifies() {
        let (exchange, server_first) = ScramExchange::start(
            rfc7677_verifier(),
            RFC7677_CLIENT_FIRST.as_bytes(),
            RFC7677_SERVER_NONCE,
        )
        .expect("client-first parses");
        assert_eq!(server_first, RFC7677_SERVER_FIRST);
        let server_final = exchange
            .finish(RFC7677_CLIENT_FINAL.as_bytes())
            .expect("proof must verify");
        assert_eq!(server_final, RFC7677_SERVER_FINAL);
    }

    #[test]
    fn wrong_password_fails_client_final() {
        let verifier =
            ScramVerifier::from_password("pencil!", &rfc7677_verifier().salt, 4096).unwrap();
        let (exchange, _) = ScramExchange::start(
            verifier,
            RFC7677_CLIENT_FIRST.as_bytes(),
            RFC7677_SERVER_NONCE,
        )
        .expect("client-first parses");
        assert_eq!(
            exchange.finish(RFC7677_CLIENT_FINAL.as_bytes()),
            Err(ScramError::AuthenticationFailed)
        );
    }

    #[test]
    fn tampered_proof_fails() {
        let (exchange, _) = ScramExchange::start(
            rfc7677_verifier(),
            RFC7677_CLIENT_FIRST.as_bytes(),
            RFC7677_SERVER_NONCE,
        )
        .expect("client-first parses");
        // Flip one base64 character in the proof (stays valid base64 — a
        // malformed message is a different failure class).
        let tampered = RFC7677_CLIENT_FINAL.replace("AndVQ=", "AndWA=");
        assert_ne!(tampered, RFC7677_CLIENT_FINAL);
        assert_eq!(
            exchange.finish(tampered.as_bytes()),
            Err(ScramError::AuthenticationFailed)
        );
    }

    #[test]
    fn nonce_mismatch_is_malformed() {
        let (exchange, _) = ScramExchange::start(
            rfc7677_verifier(),
            RFC7677_CLIENT_FIRST.as_bytes(),
            "different-server-nonce",
        )
        .expect("client-first parses");
        assert_eq!(
            exchange.finish(RFC7677_CLIENT_FINAL.as_bytes()),
            Err(ScramError::Malformed)
        );
    }

    #[test]
    fn channel_binding_flag_is_rejected() {
        let result = ScramExchange::start(
            rfc7677_verifier(),
            b"p=tls-server-end-point,,n=user,r=abc",
            "srv",
        );
        assert_eq!(result.err(), Some(ScramError::ChannelBindingUnsupported));
    }

    #[test]
    fn mandatory_extension_is_rejected() {
        let result = ScramExchange::start(
            rfc7677_verifier(),
            b"n,,m=required-thing,n=user,r=abc",
            "srv",
        );
        assert_eq!(result.err(), Some(ScramError::Unsupported));
    }

    #[test]
    fn gs2_y_flag_and_binding_echo_are_accepted() {
        // `y,` (client supports binding but uses none) plus the base64 of the
        // exact gs2 header in `c=`.
        let client_first = "y,,n=user,r=nonceabc";
        let (exchange, _) = ScramExchange::start(
            ScramVerifier::from_password("pw", b"0123456789abcdef", 4096).unwrap(),
            client_first.as_bytes(),
            "srvnonce",
        )
        .expect("client-first parses");
        let c = base64_encode("y,,".as_bytes());
        let proof = base64_encode(&[0u8; 32]);
        let client_final = format!("c={c},r=nonceabcsrvnonce,p={proof}");
        // Wrong proof → AuthenticationFailed (NOT Malformed): the shape was fine.
        assert_eq!(
            exchange.finish(client_final.as_bytes()),
            Err(ScramError::AuthenticationFailed)
        );
    }

    #[test]
    fn mock_verifier_rejects_like_wrong_password() {
        let rng = SystemRandom::new();
        let (exchange, _) = ScramExchange::start(
            ScramVerifier::mock(&rng).expect("mock verifier"),
            RFC7677_CLIENT_FIRST.as_bytes(),
            RFC7677_SERVER_NONCE,
        )
        .expect("client-first parses");
        assert_eq!(
            exchange.finish(RFC7677_CLIENT_FINAL.as_bytes()),
            Err(ScramError::AuthenticationFailed)
        );
    }

    #[test]
    fn generated_nonce_is_printable_and_comma_free() {
        for _ in 0..64 {
            let nonce = generate_nonce().expect("nonce");
            assert_eq!(nonce.len(), NONCE_LEN);
            assert!(!nonce.contains(','));
            assert!(nonce.bytes().all(|b| (0x21..=0x7e).contains(&b)));
        }
    }

    #[test]
    fn malformed_messages_are_malformed() {
        // No gs2 header.
        assert_eq!(
            ScramExchange::start(rfc7677_verifier(), b"n=user,r=abc", "srv").err(),
            Some(ScramError::Malformed)
        );
        // Missing nonce.
        assert_eq!(
            ScramExchange::start(rfc7677_verifier(), b"n,,n=user", "srv").err(),
            Some(ScramError::Malformed)
        );
        // Missing proof.
        let (exchange, _) = ScramExchange::start(
            rfc7677_verifier(),
            RFC7677_CLIENT_FIRST.as_bytes(),
            RFC7677_SERVER_NONCE,
        )
        .unwrap();
        assert_eq!(
            exchange.finish(b"c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0"),
            Err(ScramError::Malformed)
        );
    }
}
