// Encryption at rest for ProximaDB
//
// Provides AES-256-GCM encryption for all data at rest:
// - WAL files
// - SST files
// - Parquet files
// - Vector storage
//
// Features:
// - AES-256-GCM authenticated encryption
// - Hardware acceleration via AES-NI
// - Per-file key derivation via HKDF
// - Key rotation support
// - <5% performance overhead
//
// Extracted from the root crate's `src/storage/encryption` as a slice of the root-crate
// decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`). The
// transparent `EncryptedFilesystem` decorator wraps the `FileSystem` trait from
// `proximadb-storage-filesystem-types`. The root crate re-exports this crate as
// `crate::storage::encryption` for source compatibility.

pub mod file_encryption;
pub mod filesystem;
pub mod key_manager;
pub mod wal_encryption;

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, AeadCore, KeyInit, OsRng, Payload},
};
use anyhow::Result;
use sha2::{Digest, Sha256};

pub use file_encryption::FileEncryptionLayer;
pub use filesystem::EncryptedFilesystem;
pub use key_manager::{KeyManager, KeyVersion, KeyVersionManager};
pub use wal_encryption::WALEncryptionLayer;

/// Encryption algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (authenticated encryption)
    Aes256Gcm,
}

/// Backwards-compat alias for [`StorageEncryptionConfig`].
pub type EncryptionConfig = StorageEncryptionConfig;

/// Encryption configuration
#[derive(Debug, Clone)]
pub struct StorageEncryptionConfig {
    /// Whether encryption is enabled
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Master key environment variable name
    pub master_key_env_var: String,
    /// Key rotation interval in seconds (default: 30 days)
    pub key_rotation_interval_secs: u64,
    /// Chunk size for encryption (default: 4KB)
    pub chunk_size: usize,
}

impl Default for StorageEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            master_key_env_var: "PROXIMADB_CRYPTO_MASTER_KEY".to_string(),
            key_rotation_interval_secs: 30 * 24 * 3600, // 30 days
            chunk_size: 4096,                           // 4KB
        }
    }
}

/// Encrypt data using AES-256-GCM
pub fn encrypt_data(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.into());
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let mut ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    // Prepend nonce to ciphertext (needed for decryption)
    let mut result = Vec::with_capacity(nonce.len() + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.append(&mut ciphertext);

    Ok(result)
}

/// Decrypt data using AES-256-GCM
pub fn decrypt_data(key: &[u8; 32], ciphertext: &[u8]) -> Result<Vec<u8>> {
    if ciphertext.len() < 12 {
        return Err(anyhow::anyhow!("Ciphertext too short"));
    }

    let cipher = Aes256Gcm::new(key.into());
    let (nonce, encrypted_data) = ciphertext.split_at(12);
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(nonce);
    let nonce = Nonce::from_slice(&nonce_bytes);

    cipher
        .decrypt(nonce, encrypted_data)
        .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))
}

/// AES-GCM encryption with authenticated-but-plaintext associated data.
pub fn encrypt_data_with_aad(key: &[u8; 32], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.into());
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let mut ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;
    let mut result = Vec::with_capacity(nonce.len() + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.append(&mut ciphertext);
    Ok(result)
}

pub fn decrypt_data_with_aad(key: &[u8; 32], ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    if ciphertext.len() < 12 {
        return Err(anyhow::anyhow!("Ciphertext too short"));
    }
    let cipher = Aes256Gcm::new(key.into());
    let (nonce, encrypted_data) = ciphertext.split_at(12);
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: encrypted_data,
                aad,
            },
        )
        .map_err(|e| anyhow::anyhow!("Decryption failed: {e}"))
}

pub fn generate_aes_gcm_nonce() -> [u8; 12] {
    let generated = Aes256Gcm::generate_nonce(&mut OsRng);
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&generated);
    nonce
}

pub fn encrypt_data_with_nonce_and_aad(
    key: &[u8; 32],
    plaintext: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
) -> Result<Vec<u8>> {
    Aes256Gcm::new(key.into())
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))
}

pub fn decrypt_data_with_nonce_and_aad(
    key: &[u8; 32],
    ciphertext: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
) -> Result<Vec<u8>> {
    Aes256Gcm::new(key.into())
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|e| anyhow::anyhow!("Decryption failed: {e}"))
}

/// Derive encryption key from master password using HKDF
pub fn derive_key(master_key: &[u8], context: &[u8]) -> [u8; 32] {
    let hkdf = hkdf::Hkdf::<Sha256>::new(Some(master_key), context);
    let mut key = [0u8; 32];
    if let Err(error) = hkdf.expand(&b"proximadb-encryption"[..], &mut key) {
        tracing::warn!(
            error = %error,
            "HKDF expand failed; falling back to SHA-256(master_key || context)"
        );
        let mut hasher = Sha256::new();
        hasher.update(master_key);
        hasher.update(context);
        let digest = hasher.finalize();
        key.copy_from_slice(&digest[..32]);
    }
    key
}

/// Generate random 256-bit key
pub fn generate_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.fill(&mut key);
    key
}

/// Calculate authentication tag size for AES-256-GCM
pub const fn tag_size() -> usize {
    16 // 128-bit tag
}

/// Calculate nonce size for AES-256-GCM
pub const fn nonce_size() -> usize {
    12 // 96-bit nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt() {
        let key = generate_key();
        let plaintext = b"Hello, ProximaDB!";

        let ciphertext = encrypt_data(&key, plaintext).unwrap();
        assert_ne!(ciphertext, plaintext.to_vec());
        assert!(ciphertext.len() > plaintext.len()); // nonce + tag overhead

        let decrypted = decrypt_data(&key, &ciphertext).unwrap();
        assert_eq!(decrypted, plaintext.to_vec());
    }

    #[test]
    fn test_key_derivation() {
        let master_key = b"my-secret-master-key-32-bytes-long!!";
        let context = b"file-123";

        let key1 = derive_key(master_key, context);
        let key2 = derive_key(master_key, context);

        assert_eq!(key1, key2); // Deterministic
        assert_eq!(key1.len(), 32);
    }

    #[test]
    fn test_generate_key() {
        let key1 = generate_key();
        let key2 = generate_key();

        assert_ne!(key1, key2); // Random
        assert_eq!(key1.len(), 32);
        assert_eq!(key2.len(), 32);
    }

    #[test]
    fn test_decrypt_wrong_key() {
        let key1 = generate_key();
        let key2 = generate_key();
        let plaintext = b"Secret data";

        let ciphertext = encrypt_data(&key1, plaintext).unwrap();
        let result = decrypt_data(&key2, &ciphertext);

        assert!(result.is_err()); // Should fail with wrong key
    }
}
