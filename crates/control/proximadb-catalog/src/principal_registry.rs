// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-090 L0.1 — catalog-resident principal (user) + API-key registry.
//!
//! The identity half of the ADR-090 model: **an API key authenticates a
//! user; every user belongs to exactly one tenant.** This module is the
//! durable registry that makes that true operationally (today keys live in
//! `config.toml`, plaintext, loaded once at boot — ADR-090 appendix A, gaps
//! 1–6). It is deliberately **modality-agnostic**: principals and keys are
//! catalog objects independent of what a collection stores (vector, graph,
//! document, timeseries, relational) — multi-modal capability comes from
//! grants scoping over the collection abstraction (L1), not from anything
//! modality-specific here.
//!
//! Contract (the tests at the bottom are the specification):
//!
//! * `key → user → tenant` is **required**, not optional. A principal cannot
//!   be created without a non-empty tenant; a key cannot be minted without an
//!   existing, enabled principal.
//! * The full key string is returned **exactly once** at mint. Only a SHA-256
//!   hash of the secret is stored; the on-disk snapshot never contains the
//!   secret. (SHA-256, not a password KDF, is correct here: the secret is 32
//!   bytes of OS randomness, not a human password — brute force is infeasible
//!   and key-derivation slowness would only tax the hot resolve path.)
//! * `resolve_key` is **fail-closed and uniform**: unknown key id, wrong
//!   secret, revoked, expired, principal disabled or missing — all resolve to
//!   `None`, with no distinguishing detail leaked to the caller. The secret
//!   comparison is constant-time.
//! * Persistence is the same OSS mechanism as the ABAC stores
//!   (`proximadb-abac`): JSON snapshot, atomic temp-file + rename,
//!   load-on-open, synchronous persist on every mutation.
//!
//! Wire format of a minted key: `pxk_<key_id>.<secret>` where `key_id` is 16
//! hex chars (8 random bytes, public, indexable) and `secret` is 64 hex chars
//! (32 random bytes, never stored).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fc_metamodel::SubjectId;

/// Public prefix of every ProximaDB API key.
pub const KEY_PREFIX: &str = "pxk_";

const PRINCIPALS_FILE: &str = "principals.json";
const API_KEYS_FILE: &str = "api-keys.json";

/// A user identity. `subject` is the ABAC subject (ADR-090 L0: the subject
/// *is* the user id); `tenant_id` is required — the tenant is derived from the
/// credential, never from a client-asserted header.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrincipalRecord {
    pub subject: SubjectId,
    pub tenant_id: String,
    pub display_name: Option<String>,
    pub created_at_ms: i64,
    pub disabled: bool,
}

/// Stored API-key record. Carries the *hash* of the secret, never the secret.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKeyRecord {
    pub key_id: String,
    pub subject: SubjectId,
    pub tenant_id: String,
    /// Lower-hex SHA-256 of the 32-byte secret.
    pub secret_hash_hex: String,
    pub created_at_ms: i64,
    pub expires_at_ms: Option<i64>,
    pub revoked_at_ms: Option<i64>,
}

/// Returned by [`FileSystemPrincipalRegistry::mint_key`]. `key` is the only
/// copy of the full credential that will ever exist — the registry stores a
/// hash. Callers must hand it to the user and drop it.
#[derive(Debug, Clone)]
pub struct MintedKey {
    pub key_id: String,
    pub key: String,
}

/// The identity a successfully resolved key yields: the user (ABAC subject)
/// and the tenant the credential binds them to.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedApiKey {
    pub key_id: String,
    pub subject: SubjectId,
    pub tenant_id: String,
}

#[derive(Debug)]
pub enum RegistryError {
    EmptyTenant,
    EmptySubject,
    DuplicatePrincipal { tenant_id: String, subject: String },
    UnknownPrincipal { tenant_id: String, subject: String },
    PrincipalDisabled { tenant_id: String, subject: String },
    UnknownKey { key_id: String },
    Io(String),
    Serde(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTenant => write!(f, "principal tenant_id must be non-empty (ADR-090 L0)"),
            Self::EmptySubject => write!(f, "principal subject must be non-empty"),
            Self::DuplicatePrincipal { tenant_id, subject } => {
                write!(
                    f,
                    "principal '{subject}' already exists in tenant '{tenant_id}'"
                )
            }
            Self::UnknownPrincipal { tenant_id, subject } => {
                write!(f, "no principal '{subject}' in tenant '{tenant_id}'")
            }
            Self::PrincipalDisabled { tenant_id, subject } => {
                write!(
                    f,
                    "principal '{subject}' in tenant '{tenant_id}' is disabled"
                )
            }
            Self::UnknownKey { key_id } => write!(f, "no api key with id '{key_id}'"),
            Self::Io(e) => write!(f, "principal registry io error: {e}"),
            Self::Serde(e) => write!(f, "principal registry serde error: {e}"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Durable principal + API-key registry backed by two atomic-rename JSON
/// snapshots under `dir`. Mechanism-only (OSS): provisioning policy — who may
/// mint, tier limits on key counts — belongs to the admin surface, not here.
pub struct FileSystemPrincipalRegistry {
    dir: PathBuf,
    state: RwLock<State>,
}

#[derive(Default)]
struct State {
    /// Keyed `(tenant_id, subject)` — a subject name is unique *within* a
    /// tenant; the same name in two tenants is two distinct principals.
    principals: HashMap<(String, String), PrincipalRecord>,
    /// Keyed by public `key_id`.
    keys: HashMap<String, ApiKeyRecord>,
}

impl FileSystemPrincipalRegistry {
    /// Open (or create) a registry rooted at `dir`, loading any existing
    /// snapshots.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, RegistryError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|e| RegistryError::Io(e.to_string()))?;

        let principals: Vec<PrincipalRecord> = load_json_vec(&dir.join(PRINCIPALS_FILE))?;
        let keys: Vec<ApiKeyRecord> = load_json_vec(&dir.join(API_KEYS_FILE))?;

        let state = State {
            principals: principals
                .into_iter()
                .map(|p| ((p.tenant_id.clone(), p.subject.0.clone()), p))
                .collect(),
            keys: keys.into_iter().map(|k| (k.key_id.clone(), k)).collect(),
        };
        Ok(Self {
            dir,
            state: RwLock::new(state),
        })
    }

    /// Create a principal. Fails on empty tenant/subject (the ADR-090 L0
    /// "user → tenant required" rule) and on duplicates within the tenant.
    pub fn create_principal(
        &self,
        subject: SubjectId,
        tenant_id: &str,
        display_name: Option<String>,
    ) -> Result<PrincipalRecord, RegistryError> {
        if tenant_id.trim().is_empty() {
            return Err(RegistryError::EmptyTenant);
        }
        if subject.0.trim().is_empty() {
            return Err(RegistryError::EmptySubject);
        }
        let record = PrincipalRecord {
            subject: subject.clone(),
            tenant_id: tenant_id.to_string(),
            display_name,
            created_at_ms: proximadb_kernel::time::now_unix_ms(),
            disabled: false,
        };
        {
            let mut state = write_lock(&self.state);
            let key = (tenant_id.to_string(), subject.0.clone());
            if state.principals.contains_key(&key) {
                return Err(RegistryError::DuplicatePrincipal {
                    tenant_id: tenant_id.to_string(),
                    subject: subject.0,
                });
            }
            state.principals.insert(key, record.clone());
            self.persist(&state)?;
        }
        Ok(record)
    }

    /// Disable a principal. Existing keys stop resolving immediately
    /// (fail-closed at resolve time), without needing individual revocation.
    pub fn disable_principal(&self, tenant_id: &str, subject: &str) -> Result<(), RegistryError> {
        let mut state = write_lock(&self.state);
        let key = (tenant_id.to_string(), subject.to_string());
        match state.principals.get_mut(&key) {
            Some(p) => p.disabled = true,
            None => {
                return Err(RegistryError::UnknownPrincipal {
                    tenant_id: tenant_id.to_string(),
                    subject: subject.to_string(),
                });
            }
        }
        self.persist(&state)
    }

    /// Mint a key for an existing, enabled principal. The returned
    /// [`MintedKey::key`] is the only copy of the secret.
    pub fn mint_key(
        &self,
        tenant_id: &str,
        subject: &str,
        expires_at_ms: Option<i64>,
    ) -> Result<MintedKey, RegistryError> {
        let mut state = write_lock(&self.state);
        let pkey = (tenant_id.to_string(), subject.to_string());
        match state.principals.get(&pkey) {
            None => {
                return Err(RegistryError::UnknownPrincipal {
                    tenant_id: tenant_id.to_string(),
                    subject: subject.to_string(),
                });
            }
            Some(p) if p.disabled => {
                return Err(RegistryError::PrincipalDisabled {
                    tenant_id: tenant_id.to_string(),
                    subject: subject.to_string(),
                });
            }
            Some(_) => {}
        }

        let mut id_bytes = [0u8; 8];
        let mut secret_bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut id_bytes);
        rand::rngs::OsRng.fill_bytes(&mut secret_bytes);
        let key_id = to_hex(&id_bytes);
        let secret_hex = to_hex(&secret_bytes);

        let record = ApiKeyRecord {
            key_id: key_id.clone(),
            subject: SubjectId(subject.to_string()),
            tenant_id: tenant_id.to_string(),
            secret_hash_hex: sha256_hex(secret_hex.as_bytes()),
            created_at_ms: proximadb_kernel::time::now_unix_ms(),
            expires_at_ms,
            revoked_at_ms: None,
        };
        state.keys.insert(key_id.clone(), record);
        self.persist(&state)?;

        Ok(MintedKey {
            key: format!("{KEY_PREFIX}{key_id}.{secret_hex}"),
            key_id,
        })
    }

    /// Revoke a key by public id.
    pub fn revoke_key(&self, key_id: &str) -> Result<(), RegistryError> {
        let mut state = write_lock(&self.state);
        match state.keys.get_mut(key_id) {
            Some(k) => k.revoked_at_ms = Some(proximadb_kernel::time::now_unix_ms()),
            None => {
                return Err(RegistryError::UnknownKey {
                    key_id: key_id.to_string(),
                });
            }
        }
        self.persist(&state)
    }

    /// Resolve a presented key string to its principal. **Fail-closed and
    /// uniform**: every failure mode — malformed, unknown id, wrong secret,
    /// revoked, expired, principal disabled or missing — returns `None`.
    pub fn resolve_key(&self, presented: &str) -> Option<ResolvedApiKey> {
        let rest = presented.strip_prefix(KEY_PREFIX)?;
        let (key_id, secret_hex) = rest.split_once('.')?;

        let state = read_lock(&self.state);
        let record = state.keys.get(key_id)?;

        // Constant-time comparison over the fixed-width hash of the presented
        // secret — the lookup above is by public id, so timing reveals only
        // what the id already does.
        let presented_hash = sha256_hex(secret_hex.as_bytes());
        if !ct_eq(presented_hash.as_bytes(), record.secret_hash_hex.as_bytes()) {
            return None;
        }
        if record.revoked_at_ms.is_some() {
            return None;
        }
        if let Some(exp) = record.expires_at_ms
            && proximadb_kernel::time::now_unix_ms() >= exp
        {
            return None;
        }
        let principal = state
            .principals
            .get(&(record.tenant_id.clone(), record.subject.0.clone()))?;
        if principal.disabled {
            return None;
        }
        Some(ResolvedApiKey {
            key_id: record.key_id.clone(),
            subject: record.subject.clone(),
            tenant_id: record.tenant_id.clone(),
        })
    }

    /// Enumerate a tenant's principals — the "many users per tenant" listing
    /// ADR-090 appendix A gap 6 records as impossible today.
    pub fn list_principals(&self, tenant_id: &str) -> Vec<PrincipalRecord> {
        let state = read_lock(&self.state);
        let mut out: Vec<PrincipalRecord> = state
            .principals
            .values()
            .filter(|p| p.tenant_id == tenant_id)
            .cloned()
            .collect();
        out.sort_by(|a, b| a.subject.0.cmp(&b.subject.0));
        out
    }

    /// Enumerate a principal's keys (records carry hashes, never secrets).
    pub fn list_keys(&self, tenant_id: &str, subject: &str) -> Vec<ApiKeyRecord> {
        let state = read_lock(&self.state);
        let mut out: Vec<ApiKeyRecord> = state
            .keys
            .values()
            .filter(|k| k.tenant_id == tenant_id && k.subject.0 == subject)
            .cloned()
            .collect();
        out.sort_by(|a, b| a.key_id.cmp(&b.key_id));
        out
    }

    fn persist(&self, state: &State) -> Result<(), RegistryError> {
        let principals: Vec<&PrincipalRecord> = state.principals.values().collect();
        let keys: Vec<&ApiKeyRecord> = state.keys.values().collect();
        store_json_atomic(&self.dir, PRINCIPALS_FILE, &principals)?;
        store_json_atomic(&self.dir, API_KEYS_FILE, &keys)
    }
}

fn load_json_vec<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, RegistryError> {
    match fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes).map_err(|e| RegistryError::Serde(e.to_string()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(RegistryError::Io(e.to_string())),
    }
}

/// Atomic temp-file + rename snapshot — the same durability mechanism as the
/// ABAC stores (`proximadb-abac::policy_binding_store`).
fn store_json_atomic<T: Serialize>(dir: &Path, file: &str, value: &T) -> Result<(), RegistryError> {
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|e| RegistryError::Serde(e.to_string()))?;
    let tmp = dir.join(format!("{file}.tmp"));
    {
        let mut f = fs::File::create(&tmp).map_err(|e| RegistryError::Io(e.to_string()))?;
        f.write_all(&bytes)
            .map_err(|e| RegistryError::Io(e.to_string()))?;
        f.sync_all().map_err(|e| RegistryError::Io(e.to_string()))?;
    }
    fs::rename(&tmp, dir.join(file)).map_err(|e| RegistryError::Io(e.to_string()))?;
    Ok(())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use fmt::Write as _;
        // Writing to a String cannot fail; ignore the fmt::Result rather than
        // unwrap (panic-policy).
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

/// Constant-time equality over equal-length byte slices; false on length
/// mismatch without early exit on content.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

use crate::lock_util::{read_lock, write_lock};

// ---------------------------------------------------------------------------
// The specification. Written before the implementation (ADR-090 / TD-AUTHZ-1
// L0.1); each test names the contract clause it pins.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> (tempfile::TempDir, FileSystemPrincipalRegistry) {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg = FileSystemPrincipalRegistry::open(dir.path()).expect("open");
        (dir, reg)
    }

    fn subject(s: &str) -> SubjectId {
        SubjectId(s.to_string())
    }

    #[test]
    fn mint_returns_secret_once_and_disk_stores_only_the_hash() {
        let (dir, reg) = registry();
        reg.create_principal(subject("alice"), "tenant-a", None)
            .expect("create");
        let minted = reg.mint_key("tenant-a", "alice", None).expect("mint");

        assert!(minted.key.starts_with(KEY_PREFIX));
        let secret_part = minted.key.split('.').nth(1).expect("secret part");

        // The on-disk snapshot must never contain the secret.
        let on_disk =
            std::fs::read_to_string(dir.path().join(API_KEYS_FILE)).expect("read snapshot");
        assert!(
            !on_disk.contains(secret_part),
            "secret leaked into the persisted snapshot"
        );
        // And the full key resolves to the right principal + tenant.
        let resolved = reg.resolve_key(&minted.key).expect("resolve");
        assert_eq!(resolved.subject.0, "alice");
        assert_eq!(resolved.tenant_id, "tenant-a");
        assert_eq!(resolved.key_id, minted.key_id);
    }

    #[test]
    fn resolve_is_fail_closed_and_uniform() {
        let (_dir, reg) = registry();
        reg.create_principal(subject("alice"), "tenant-a", None)
            .expect("create");
        let minted = reg.mint_key("tenant-a", "alice", None).expect("mint");

        // Malformed / unknown id / wrong secret all -> None.
        assert!(reg.resolve_key("garbage").is_none());
        assert!(reg.resolve_key("pxk_deadbeefdeadbeef.00").is_none());
        let wrong_secret = format!("{KEY_PREFIX}{}.{}", minted.key_id, "0".repeat(64));
        assert!(
            reg.resolve_key(&wrong_secret).is_none(),
            "wrong secret must not resolve"
        );

        // Revoked -> None.
        reg.revoke_key(&minted.key_id).expect("revoke");
        assert!(
            reg.resolve_key(&minted.key).is_none(),
            "revoked key must not resolve"
        );

        // Expired -> None (mint with an expiry already in the past).
        let expired = reg
            .mint_key(
                "tenant-a",
                "alice",
                Some(proximadb_kernel::time::now_unix_ms() - 1_000),
            )
            .expect("mint expired");
        assert!(
            reg.resolve_key(&expired.key).is_none(),
            "expired key must not resolve"
        );

        // Disabled principal -> every remaining key stops resolving.
        let live = reg.mint_key("tenant-a", "alice", None).expect("mint live");
        assert!(reg.resolve_key(&live.key).is_some());
        reg.disable_principal("tenant-a", "alice").expect("disable");
        assert!(
            reg.resolve_key(&live.key).is_none(),
            "disabled principal's key must not resolve"
        );
    }

    #[test]
    fn tenant_is_required_and_keys_require_an_existing_principal() {
        let (_dir, reg) = registry();

        // ADR-090 L0: user -> tenant is REQUIRED.
        assert!(matches!(
            reg.create_principal(subject("alice"), "", None),
            Err(RegistryError::EmptyTenant)
        ));
        assert!(matches!(
            reg.create_principal(subject("  "), "tenant-a", None),
            Err(RegistryError::EmptySubject)
        ));

        // key -> user is REQUIRED: no principal, no key.
        assert!(matches!(
            reg.mint_key("tenant-a", "ghost", None),
            Err(RegistryError::UnknownPrincipal { .. })
        ));
    }

    #[test]
    fn subject_names_are_scoped_per_tenant() {
        let (_dir, reg) = registry();
        reg.create_principal(subject("alice"), "tenant-a", None)
            .expect("a");
        // Same subject name in another tenant is a DIFFERENT principal — fine.
        reg.create_principal(subject("alice"), "tenant-b", None)
            .expect("b");
        // Duplicate within one tenant is rejected.
        assert!(matches!(
            reg.create_principal(subject("alice"), "tenant-a", None),
            Err(RegistryError::DuplicatePrincipal { .. })
        ));

        // Each tenant's key binds to ITS principal.
        let ka = reg.mint_key("tenant-a", "alice", None).expect("mint a");
        let kb = reg.mint_key("tenant-b", "alice", None).expect("mint b");
        assert_eq!(reg.resolve_key(&ka.key).expect("ra").tenant_id, "tenant-a");
        assert_eq!(reg.resolve_key(&kb.key).expect("rb").tenant_id, "tenant-b");
    }

    #[test]
    fn registry_round_trips_through_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let minted = {
            let reg = FileSystemPrincipalRegistry::open(dir.path()).expect("open");
            reg.create_principal(subject("alice"), "tenant-a", Some("Alice".into()))
                .expect("create");
            reg.mint_key("tenant-a", "alice", None).expect("mint")
        };
        // Reopen from disk: principal, key, and resolution all survive.
        let reopened = FileSystemPrincipalRegistry::open(dir.path()).expect("reopen");
        assert_eq!(reopened.list_principals("tenant-a").len(), 1);
        assert_eq!(reopened.list_keys("tenant-a", "alice").len(), 1);
        let resolved = reopened
            .resolve_key(&minted.key)
            .expect("resolve after reopen");
        assert_eq!(resolved.subject.0, "alice");

        // Revocation persists too.
        reopened.revoke_key(&minted.key_id).expect("revoke");
        let reopened_again = FileSystemPrincipalRegistry::open(dir.path()).expect("reopen2");
        assert!(reopened_again.resolve_key(&minted.key).is_none());
    }

    #[test]
    fn rotation_is_mint_plus_revoke_and_other_keys_are_unaffected() {
        let (_dir, reg) = registry();
        reg.create_principal(subject("alice"), "tenant-a", None)
            .expect("create");
        let old = reg.mint_key("tenant-a", "alice", None).expect("old");
        let new = reg.mint_key("tenant-a", "alice", None).expect("new");

        // Both valid until the old one is revoked (graceful rotation window).
        assert!(reg.resolve_key(&old.key).is_some());
        assert!(reg.resolve_key(&new.key).is_some());

        reg.revoke_key(&old.key_id).expect("revoke old");
        assert!(reg.resolve_key(&old.key).is_none());
        assert!(
            reg.resolve_key(&new.key).is_some(),
            "rotation must not break the new key"
        );

        assert_eq!(
            reg.list_keys("tenant-a", "alice").len(),
            2,
            "revoked keys stay listed"
        );
    }

    #[test]
    fn ct_eq_semantics() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(ct_eq(b"", b""));
    }
}
