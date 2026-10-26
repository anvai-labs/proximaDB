// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # TLS Configuration Module
//!
//! Provides TLS configuration and certificate management for ProximaDB servers.
//!
//! Extracted from the root crate's `src/network/tls` as a slice of the root-crate
//! decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`). The
//! root crate re-exports it as `crate::network::tls` for source compatibility.
//!
//! ## Features
//!
//! - TLS server configuration for REST and gRPC
//! - mTLS (mutual TLS) with client certificate verification
//! - Automatic certificate generation for development
//! - Certificate parsing and validation
//! - Rustls configuration builders
//! - TLS acceptor for client certificate extraction (TD-006)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use proximadb::network::tls::{TlsConfig, TlsServerConfig};
//!
//! // Create TLS configuration
//! let config = TlsConfig::new(true)
//!     .with_auto_certificates("/tmp/certs".into())
//!     .with_mtls();
//!
//! // Build rustls server config (async)
//! let rustls_config = config.build_server_config().await?;
//!
//! // Create TLS acceptor with certificate extraction
//! let acceptor = TlsAcceptor::new(rustls_config)?;
//! ```

pub mod acceptor;
pub mod certificate_manager;

pub use acceptor::TlsAcceptor;

#[cfg(feature = "network-rest")]
pub use acceptor::{TlsAcceptorLayer, TlsAcceptorMiddleware};

pub use certificate_manager::{
    CertificateConfig, CertificateManager, CertificateStatus, CertificateSubject,
    GeneratedCertificate, ParsedCertificate, TlsError,
};

use anyhow::Result;
use rustls::server::WebPkiClientVerifier;
use rustls::{RootCertStore, ServerConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// TLS configuration for ProximaDB
#[derive(Debug, Clone, Default)]
pub struct TlsConfig {
    /// Certificate manager
    pub certificate_manager: Option<CertificateManager>,
    /// Enable TLS
    pub enabled: bool,
    /// Require client certificates (mTLS)
    pub require_client_certs: bool,
    /// Path to server certificate file
    pub cert_file: Option<PathBuf>,
    /// Path to server private key file
    pub key_file: Option<PathBuf>,
    /// Path to CA certificate file (for mTLS client verification)
    pub ca_file: Option<PathBuf>,
}

impl TlsConfig {
    /// Create new TLS configuration
    pub fn new(enabled: bool) -> Self {
        Self {
            certificate_manager: None,
            enabled,
            require_client_certs: false,
            cert_file: None,
            key_file: None,
            ca_file: None,
        }
    }

    /// Enable automatic certificate management
    pub fn with_auto_certificates(mut self, cert_dir: PathBuf) -> Self {
        let cert_config = CertificateConfig::default();
        self.certificate_manager = Some(CertificateManager::new(cert_config, cert_dir));
        self
    }

    /// Configure with custom certificate manager
    pub fn with_certificate_manager(mut self, manager: CertificateManager) -> Self {
        self.certificate_manager = Some(manager);
        self
    }

    /// Enable mTLS (mutual TLS)
    pub fn with_mtls(mut self) -> Self {
        self.require_client_certs = true;
        self
    }

    /// Set certificate file path
    pub fn with_cert_file(mut self, path: PathBuf) -> Self {
        self.cert_file = Some(path);
        self
    }

    /// Set private key file path
    pub fn with_key_file(mut self, path: PathBuf) -> Self {
        self.key_file = Some(path);
        self
    }

    /// Set CA certificate file path for mTLS
    pub fn with_ca_file(mut self, path: PathBuf) -> Self {
        self.ca_file = Some(path);
        self
    }

    /// Initialize TLS configuration
    pub async fn initialize(&self) -> Result<()> {
        if let Some(ref cert_manager) = self.certificate_manager {
            cert_manager.initialize().await?;
        }
        Ok(())
    }

    /// Get certificate paths
    pub fn get_certificate_paths(&self) -> Option<(PathBuf, PathBuf)> {
        // First try explicit paths
        if let (Some(cert), Some(key)) = (&self.cert_file, &self.key_file) {
            return Some((cert.clone(), key.clone()));
        }
        // Fall back to certificate manager paths
        self.certificate_manager
            .as_ref()
            .map(|cm| (cm.get_cert_path(), cm.get_key_path()))
    }

    /// Get CA certificate path
    pub fn get_ca_path(&self) -> Option<PathBuf> {
        // First try explicit path
        if let Some(ca) = &self.ca_file {
            return Some(ca.clone());
        }
        // Fall back to certificate manager path
        self.certificate_manager.as_ref().map(|cm| cm.get_ca_path())
    }

    /// Build rustls ServerConfig for the server
    pub async fn build_server_config(&self) -> Result<Arc<ServerConfig>> {
        // rustls 0.23 requires a process-default CryptoProvider before builder(); idempotent.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let (cert_path, key_path) = self
            .get_certificate_paths()
            .ok_or_else(|| anyhow::anyhow!("No certificate paths configured"))?;

        // Load certificates
        let cert_pem = tokio::fs::read(&cert_path).await?;
        let key_pem = tokio::fs::read(&key_path).await?;

        let certs = certificate_manager::utils::load_certs_from_pem(&cert_pem)
            .map_err(|e| anyhow::anyhow!("Failed to load certificates: {}", e))?;
        let key = certificate_manager::utils::load_private_key_from_pem(&key_pem)
            .map_err(|e| anyhow::anyhow!("Failed to load private key: {}", e))?;

        let config = if self.require_client_certs {
            // mTLS configuration
            let ca_path = self
                .get_ca_path()
                .ok_or_else(|| anyhow::anyhow!("CA certificate required for mTLS"))?;

            let ca_pem = tokio::fs::read(&ca_path).await?;
            let ca_certs = certificate_manager::utils::load_certs_from_pem(&ca_pem)
                .map_err(|e| anyhow::anyhow!("Failed to load CA certificates: {}", e))?;

            // Build root cert store
            let mut root_store = RootCertStore::empty();
            for ca_cert in ca_certs {
                root_store.add(ca_cert)?;
            }

            // Client certificate verifier (rustls 0.23 WebPkiClientVerifier).
            let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build client cert verifier: {}", e))?;

            ServerConfig::builder()
                .with_client_cert_verifier(client_verifier)
                .with_single_cert(certs, key)
                .map_err(|e| anyhow::anyhow!("Failed to build TLS config: {}", e))?
        } else {
            // Standard TLS (no client certs required; rustls 0.23 dropped with_safe_defaults).
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)
                .map_err(|e| anyhow::anyhow!("Failed to build TLS config: {}", e))?
        };

        info!(
            "TLS configuration built successfully (mTLS: {})",
            self.require_client_certs
        );

        Ok(Arc::new(config))
    }

    /// Build rustls ServerConfig with ALPN protocols
    pub async fn build_server_config_with_alpn(
        &self,
        alpn_protocols: Vec<Vec<u8>>,
    ) -> Result<Arc<ServerConfig>> {
        let mut config = (*self.build_server_config().await?).clone();
        config.alpn_protocols = alpn_protocols;
        Ok(Arc::new(config))
    }
}

/// TLS server configuration builder for different protocols
pub struct TlsServerConfig {
    tls_config: TlsConfig,
}

impl TlsServerConfig {
    /// Create new TLS server config builder
    pub fn new(tls_config: TlsConfig) -> Self {
        Self { tls_config }
    }

    /// Build configuration for HTTP/1.1 (REST)
    pub async fn for_http1(&self) -> Result<Arc<ServerConfig>> {
        self.tls_config
            .build_server_config_with_alpn(vec![b"http/1.1".to_vec()])
            .await
    }

    /// Build configuration for HTTP/2 (gRPC)
    pub async fn for_http2(&self) -> Result<Arc<ServerConfig>> {
        self.tls_config
            .build_server_config_with_alpn(vec![b"h2".to_vec()])
            .await
    }

    /// Build configuration for both HTTP/1.1 and HTTP/2
    pub async fn for_http1_and_http2(&self) -> Result<Arc<ServerConfig>> {
        self.tls_config
            .build_server_config_with_alpn(vec![b"h2".to_vec(), b"http/1.1".to_vec()])
            .await
    }

    /// Get the underlying TLS config
    pub fn tls_config(&self) -> &TlsConfig {
        &self.tls_config
    }
}

/// Client certificate information extracted from TLS connection
#[derive(Debug, Clone)]
pub struct ClientCertificateInfo {
    /// Client's Common Name from certificate
    pub common_name: Option<String>,
    /// Client's organization from certificate
    pub organization: Option<String>,
    /// Certificate serial number
    pub serial: String,
    /// Certificate fingerprint (SHA-256)
    pub fingerprint: String,
    /// Is the certificate valid
    pub is_valid: bool,
    /// Certificate expiration time
    pub expires_at: std::time::SystemTime,
}

impl ClientCertificateInfo {
    /// Parse client certificate from DER bytes
    pub fn from_der(cert_der: &[u8]) -> Result<Self, TlsError> {
        use sha2::{Digest, Sha256};
        use x509_parser::prelude::*;

        let (_, cert) = X509Certificate::from_der(cert_der)
            .map_err(|e| TlsError::CertificateParse(format!("DER parse error: {:?}", e)))?;

        // Extract CN
        let common_name = cert
            .subject()
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            .map(|s| s.to_string());

        // Extract organization
        let organization = cert
            .subject()
            .iter_organization()
            .next()
            .and_then(|o| o.as_str().ok())
            .map(|s| s.to_string());

        // Calculate fingerprint
        let mut hasher = Sha256::new();
        hasher.update(cert_der);
        let fingerprint = hex::encode(hasher.finalize());

        // Check validity
        let now = std::time::SystemTime::now();
        let not_before = asn1_time_to_system_time(&cert.validity().not_before)?;
        let not_after = asn1_time_to_system_time(&cert.validity().not_after)?;
        let is_valid = now >= not_before && now <= not_after;

        Ok(ClientCertificateInfo {
            common_name,
            organization,
            serial: cert.serial.to_string(),
            fingerprint,
            is_valid,
            expires_at: not_after,
        })
    }
}

/// Convert ASN.1 time to SystemTime
fn asn1_time_to_system_time(
    time: &x509_parser::prelude::ASN1Time,
) -> Result<std::time::SystemTime, TlsError> {
    let datetime = time.to_datetime();
    let timestamp = datetime.unix_timestamp();

    if timestamp >= 0 {
        Ok(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(timestamp as u64))
    } else {
        Err(TlsError::CertificateParse(
            "Invalid certificate time".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tls_config_default() {
        let config = TlsConfig::default();
        assert!(!config.enabled);
        assert!(!config.require_client_certs);
        assert!(config.certificate_manager.is_none());
    }

    #[test]
    fn test_tls_config_builder() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory for TLS config test");
        let config = TlsConfig::new(true)
            .with_auto_certificates(temp_dir.path().to_path_buf())
            .with_mtls();

        assert!(config.enabled);
        assert!(config.require_client_certs);
        assert!(config.certificate_manager.is_some());
    }

    #[test]
    fn test_tls_config_explicit_paths() {
        let config = TlsConfig::new(true)
            .with_cert_file("/path/to/cert.pem".into())
            .with_key_file("/path/to/key.pem".into())
            .with_ca_file("/path/to/ca.pem".into());

        let (cert, key) = config
            .get_certificate_paths()
            .ok_or_else(|| anyhow::anyhow!("Expected certificate paths to be configured"))
            .expect("Failed to get certificate paths");
        assert_eq!(cert, PathBuf::from("/path/to/cert.pem"));
        assert_eq!(key, PathBuf::from("/path/to/key.pem"));
        assert_eq!(config.get_ca_path(), Some(PathBuf::from("/path/to/ca.pem")));
    }

    #[tokio::test]
    async fn test_build_server_config() {
        let temp_dir =
            TempDir::new().expect("Failed to create temp directory for server config test");

        // Create certificate manager and generate certs
        let cert_config = CertificateConfig::default();
        let manager = CertificateManager::new(cert_config, temp_dir.path().to_path_buf());
        manager
            .generate_and_save_certificates()
            .await
            .expect("Failed to generate and save certificates");

        // Build TLS config
        let tls_config = TlsConfig::new(true).with_certificate_manager(manager);
        let server_config = tls_config
            .build_server_config()
            .await
            .expect("Failed to build server config");

        // Verify config was built
        assert!(server_config.alpn_protocols.is_empty()); // No ALPN set
    }

    #[tokio::test]
    async fn test_build_server_config_with_alpn() {
        let temp_dir =
            TempDir::new().expect("Failed to create temp directory for ALPN config test");

        let cert_config = CertificateConfig::default();
        let manager = CertificateManager::new(cert_config, temp_dir.path().to_path_buf());
        manager
            .generate_and_save_certificates()
            .await
            .expect("Failed to generate and save certificates");

        let tls_config = TlsConfig::new(true).with_certificate_manager(manager);
        let server_config = TlsServerConfig::new(tls_config);

        // Test HTTP/2 config
        let h2_config = server_config
            .for_http2()
            .await
            .expect("Failed to build HTTP/2 config");
        assert!(h2_config.alpn_protocols.contains(&b"h2".to_vec()));

        // Test HTTP/1.1 config
        let h1_config = server_config
            .for_http1()
            .await
            .expect("Failed to build HTTP/1.1 config");
        assert!(h1_config.alpn_protocols.contains(&b"http/1.1".to_vec()));
    }

    #[tokio::test]
    async fn test_mtls_server_config() {
        let temp_dir =
            TempDir::new().expect("Failed to create temp directory for mTLS config test");

        // Create certificate manager
        let cert_config = CertificateConfig::default();
        let manager = CertificateManager::new(cert_config, temp_dir.path().to_path_buf());

        // Generate CA and server certificates
        manager
            .generate_and_save_ca()
            .await
            .expect("Failed to generate and save CA certificate");
        manager
            .generate_and_save_certificates()
            .await
            .expect("Failed to generate and save server certificates");

        // Build mTLS config
        let tls_config = TlsConfig::new(true)
            .with_certificate_manager(manager)
            .with_mtls();
        let _server_config = tls_config
            .build_server_config()
            .await
            .expect("Failed to build mTLS server config");

        // Config should be valid - verify by checking it's not empty
        // Note: modern rustls doesn't expose client_auth_mandatory() directly
        assert!(tls_config.require_client_certs);
    }

    #[test]
    fn test_client_certificate_info() {
        let temp_dir =
            TempDir::new().expect("Failed to create temp directory for client cert info test");
        let cert_config = CertificateConfig {
            subject: CertificateSubject {
                common_name: "test-client".to_string(),
                organization: Some("Test Org".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = CertificateManager::new(cert_config, temp_dir.path().to_path_buf());

        let cert = manager
            .generate_self_signed()
            .expect("Failed to generate self-signed certificate");

        // Parse the PEM to get DER
        let pem_data = cert.cert_pem.as_bytes();
        let (_, pem_block) =
            x509_parser::pem::parse_x509_pem(pem_data).expect("Failed to parse PEM certificate");

        let info = ClientCertificateInfo::from_der(&pem_block.contents)
            .expect("Failed to parse client certificate info from DER");

        assert_eq!(info.common_name, Some("test-client".to_string()));
        assert_eq!(info.organization, Some("Test Org".to_string()));
        assert!(info.is_valid);
        assert!(!info.fingerprint.is_empty());
    }
}
