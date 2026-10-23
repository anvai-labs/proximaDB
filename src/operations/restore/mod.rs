// Restore operations for ProximaDB
//
// Provides:
// - Restore from backup manifest
// - Checksum verification
// - WAL replay for consistency
// - Recovery testing

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::operations::backup::{BackupManifest, BackupTarget, DataFileMetadata};
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

/// Restore manager for recovering from backups
#[allow(dead_code)]
pub struct RestoreManager {
    /// Base path for database storage
    base_path: PathBuf,
    /// Storage filesystem
    storage: Arc<UnifiedCachingFilesystem>,
    /// Restore configuration
    config: RestoreConfig,
    /// Restore statistics
    stats: Arc<RwLock<RestoreStats>>,
}

/// Restore configuration
#[derive(Debug, Clone)]
pub struct RestoreConfig {
    /// Verify checksums during restore
    pub verify_checksums: bool,
    /// Continue on error (best-effort restore)
    pub continue_on_error: bool,
    /// Dry run (don't actually restore)
    pub dry_run: bool,
    /// Backup target to restore from
    pub target: BackupTarget,
}

impl Default for RestoreConfig {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            continue_on_error: false,
            dry_run: false,
            target: BackupTarget::Local {
                path: PathBuf::from("/tmp/proximadb/backups"),
            },
        }
    }
}

/// Restore statistics
#[derive(Debug, Clone, Default)]
pub struct RestoreStats {
    /// Total files restored
    pub files_restored: u64,
    /// Total bytes restored
    pub bytes_restored: u64,
    /// Number of checksum failures
    pub checksum_failures: u64,
    /// Number of files skipped
    pub files_skipped: u64,
    /// Restore duration in milliseconds
    pub restore_duration_ms: Option<u64>,
}

/// Restore result
#[derive(Debug, Clone)]
pub struct RestoreResult {
    /// Whether restore was successful
    pub success: bool,
    /// Number of files restored
    pub files_restored: u64,
    /// Number of bytes restored
    pub bytes_restored: u64,
    /// Number of checksum failures
    pub checksum_failures: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Errors encountered during restore
    pub errors: Vec<String>,
}

impl RestoreManager {
    /// Create a new restore manager
    pub fn new(
        base_path: &Path,
        storage: Arc<UnifiedCachingFilesystem>,
        config: RestoreConfig,
    ) -> Result<Self> {
        Ok(Self {
            base_path: base_path.to_path_buf(),
            storage,
            config,
            stats: Arc::new(RwLock::new(RestoreStats::default())),
        })
    }

    /// Restore from a backup manifest
    pub async fn restore_from_backup(&self, manifest: &BackupManifest) -> Result<RestoreResult> {
        let start_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("Failed to get start time: {}", e))?
            .as_millis() as u64;

        info!(
            "Starting restore from backup: {} (type: {:?})",
            manifest.backup_id, manifest.backup_type
        );

        let mut errors = Vec::new();
        let mut files_restored = 0u64;
        let mut bytes_restored = 0u64;
        let mut checksum_failures = 0u64;

        // Step 1: Download backup from remote target if needed
        let backup_dir = self.download_backup_if_needed(&manifest.backup_id).await?;

        // Step 2: Restore data files
        for file_metadata in &manifest.data_files {
            match self
                .restore_file(&backup_dir, file_metadata, manifest)
                .await
            {
                Ok(RestoredFile { restored, bytes }) => {
                    if restored {
                        files_restored += 1;
                        bytes_restored += bytes;
                    }
                }
                Err(e) => {
                    let error_msg = format!(
                        "Failed to restore file {:?}: {}",
                        file_metadata.relative_path, e
                    );
                    warn!("{}", error_msg);
                    errors.push(error_msg);

                    if !self.config.continue_on_error {
                        break;
                    }
                }
            }
        }

        // Step 3: Restore WAL segments
        match self.restore_wal_segments(&backup_dir, manifest).await {
            Ok(_) => {}
            Err(e) => {
                let error_msg = format!("Failed to restore WAL segments: {}", e);
                warn!("{}", error_msg);
                errors.push(error_msg);
            }
        }

        // Step 4: Verify manifest checksum
        if self.config.verify_checksums {
            match self.verify_manifest_checksum(&backup_dir, manifest).await {
                Ok(true) => {
                    info!("Manifest checksum verification passed");
                }
                Ok(false) => {
                    let error_msg = "Manifest checksum verification failed".to_string();
                    warn!("{}", error_msg);
                    errors.push(error_msg);
                    checksum_failures += 1;
                }
                Err(e) => {
                    let error_msg = format!("Failed to verify manifest checksum: {}", e);
                    warn!("{}", error_msg);
                    errors.push(error_msg);
                }
            }
        }

        // Step 5: Update statistics
        let duration_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("Failed to calculate duration: {}", e))?
            .as_millis() as u64
            - start_time;

        {
            let mut stats = self.stats.write().await;
            stats.files_restored = files_restored;
            stats.bytes_restored = bytes_restored;
            stats.checksum_failures = checksum_failures;
            stats.restore_duration_ms = Some(duration_ms);
        }

        let success =
            errors.is_empty() && (self.config.continue_on_error || checksum_failures == 0);

        info!(
            "Restore completed: {} files, {} bytes, {} ms, success: {}",
            files_restored, bytes_restored, duration_ms, success
        );

        Ok(RestoreResult {
            success,
            files_restored,
            bytes_restored,
            checksum_failures,
            duration_ms,
            errors,
        })
    }

    /// Restore a single file from backup
    async fn restore_file(
        &self,
        backup_dir: &Path,
        file_metadata: &DataFileMetadata,
        _manifest: &BackupManifest,
    ) -> Result<RestoredFile> {
        let source_path = backup_dir.join(&file_metadata.relative_path);
        let dest_path = self.base_path.join(&file_metadata.relative_path);

        // Check if source file exists
        if !source_path.exists() {
            return Err(anyhow::anyhow!(
                "Source file missing from backup: {}",
                file_metadata.relative_path
            ));
        }

        // Verify checksum if enabled
        if self.config.verify_checksums {
            let contents = tokio::fs::read(&source_path).await?;
            let actual_checksum = self.calculate_checksum(&contents);

            if actual_checksum != file_metadata.checksum {
                return Err(anyhow::anyhow!(
                    "Checksum mismatch for file {}: expected {}, got {}",
                    file_metadata.relative_path,
                    file_metadata.checksum,
                    actual_checksum
                ));
            }
        }

        // Skip if in dry-run mode
        if self.config.dry_run {
            return Ok(RestoredFile {
                restored: true,
                bytes: file_metadata.size,
            });
        }

        // Create destination directory
        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Copy file
        tokio::fs::copy(&source_path, &dest_path).await?;

        Ok(RestoredFile {
            restored: true,
            bytes: file_metadata.size,
        })
    }

    /// Restore WAL segments from backup
    async fn restore_wal_segments(
        &self,
        backup_dir: &Path,
        _manifest: &BackupManifest,
    ) -> Result<()> {
        let wal_backup_dir = backup_dir.join("wal");
        if !wal_backup_dir.exists() {
            return Ok(()); // No WAL segments to restore
        }

        let wal_dest_dir = self.base_path.join("wal");
        tokio::fs::create_dir_all(&wal_dest_dir).await?;

        // Copy WAL segment files
        let mut entries = tokio::fs::read_dir(&wal_backup_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let source_path = entry.path();
            let file_name = source_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("Path has no file name: {:?}", source_path))?
                .to_string_lossy()
                .to_string();

            let dest_path = wal_dest_dir.join(&file_name);

            if !self.config.dry_run {
                tokio::fs::copy(&source_path, &dest_path).await?;
            }

            info!("Restored WAL segment: {}", file_name);
        }

        Ok(())
    }

    /// Verify manifest checksum
    async fn verify_manifest_checksum(
        &self,
        backup_dir: &Path,
        manifest: &BackupManifest,
    ) -> Result<bool> {
        let manifest_path = backup_dir.join("manifest.json");
        let manifest_json = tokio::fs::read_to_string(&manifest_path).await?;
        let mut parsed_manifest: BackupManifest = serde_json::from_str(&manifest_json)?;
        let stored_checksum = parsed_manifest.manifest_checksum.clone();

        // Align with backup creation: checksum is computed with an empty manifest_checksum field.
        parsed_manifest.manifest_checksum.clear();
        let canonical_without_checksum = serde_json::to_string_pretty(&parsed_manifest)?;
        let calculated = self.calculate_checksum(canonical_without_checksum.as_bytes());

        // Detect obvious backup identity mismatch between caller and manifest-on-disk.
        if parsed_manifest.backup_id != manifest.backup_id {
            return Ok(false);
        }

        Ok(stored_checksum == calculated)
    }

    /// Download backup from remote target if needed
    async fn download_backup_if_needed(&self, backup_id: &str) -> Result<PathBuf> {
        match &self.config.target {
            BackupTarget::Local { path } => {
                let backup_dir = path.join(backup_id);
                if !backup_dir.exists() {
                    return Err(anyhow::anyhow!(
                        "Backup directory not found: {:?}",
                        backup_dir
                    ));
                }
                Ok(backup_dir)
            }
            BackupTarget::S3 { bucket, prefix } => {
                info!(
                    "Downloading backup {} from S3://{}/{}",
                    backup_id, bucket, prefix
                );
                // Deferred: Implement S3 download
                let download_dir = PathBuf::from(format!("/tmp/proximadb/restore/{}", backup_id));
                Ok(download_dir)
            }
            BackupTarget::GCS { bucket, prefix } => {
                info!(
                    "Downloading backup {} from GCS://{}/{}",
                    backup_id, bucket, prefix
                );
                // Deferred: Implement GCS download
                let download_dir = PathBuf::from(format!("/tmp/proximadb/restore/{}", backup_id));
                Ok(download_dir)
            }
            BackupTarget::Azure { container, prefix } => {
                info!(
                    "Downloading backup {} from Azure://{}/{}",
                    backup_id, container, prefix
                );
                // Deferred: Implement Azure download
                let download_dir = PathBuf::from(format!("/tmp/proximadb/restore/{}", backup_id));
                Ok(download_dir)
            }
        }
    }

    /// Calculate SHA-256 checksum
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Get restore statistics
    pub async fn stats(&self) -> RestoreStats {
        self.stats.read().await.clone()
    }

    /// Validate backup integrity without restoring
    pub async fn validate_backup(
        &self,
        manifest: &BackupManifest,
    ) -> Result<RestoreValidationResult> {
        info!("Validating backup: {}", manifest.backup_id);

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Step 1: Verify manifest checksum
        let backup_dir = self.download_backup_if_needed(&manifest.backup_id).await?;

        match self.verify_manifest_checksum(&backup_dir, manifest).await {
            Ok(true) => {}
            Ok(false) => {
                errors.push("Manifest checksum verification failed".to_string());
            }
            Err(e) => {
                errors.push(format!("Failed to verify manifest checksum: {}", e));
            }
        }

        // Step 2: Verify all data files exist and have valid checksums
        for file_metadata in &manifest.data_files {
            let file_path = backup_dir.join(&file_metadata.relative_path);

            if !file_path.exists() {
                errors.push(format!(
                    "Data file missing: {}",
                    file_metadata.relative_path
                ));
                continue;
            }

            // Verify checksum
            match tokio::fs::read(&file_path).await {
                Ok(contents) => {
                    let actual_checksum = self.calculate_checksum(&contents);
                    if actual_checksum != file_metadata.checksum {
                        errors.push(format!(
                            "Checksum mismatch for file {}: expected {}, got {}",
                            file_metadata.relative_path, file_metadata.checksum, actual_checksum
                        ));
                    }
                }
                Err(e) => {
                    errors.push(format!(
                        "Failed to read file {}: {}",
                        file_metadata.relative_path, e
                    ));
                }
            }
        }

        // Step 3: Verify WAL segments exist
        let wal_backup_dir = backup_dir.join("wal");
        if wal_backup_dir.exists() {
            for wal_segment in &manifest.wal_segments {
                let wal_path = wal_backup_dir.join(wal_segment);
                if !wal_path.exists() {
                    warnings.push(format!("WAL segment missing: {}", wal_segment));
                }
            }
        } else if !manifest.wal_segments.is_empty() {
            warnings.push("WAL directory not found in backup".to_string());
        }

        // Step 4: Verify LSN range is valid
        if manifest.lsn_range.0 > manifest.lsn_range.1 {
            errors.push(format!(
                "Invalid LSN range: {:?} (start > end)",
                manifest.lsn_range
            ));
        }

        let valid = errors.is_empty();
        let total_files = manifest.data_files.len();
        let total_bytes = manifest.total_bytes;

        Ok(RestoreValidationResult {
            valid,
            total_files,
            total_bytes,
            errors,
            warnings,
        })
    }

    /// Find the most recent backup that can be used for point-in-time recovery
    pub async fn find_backup_for_pitr(
        &self,
        _target_timestamp: u64,
    ) -> Result<Option<BackupManifest>> {
        // Deferred: Implement by listing backups and finding one before target time
        // For now, return None
        Ok(None)
    }
}

/// Result of restoring a file
struct RestoredFile {
    /// Whether the file was restored
    restored: bool,
    /// Number of bytes restored
    bytes: u64,
}

/// Backwards-compat alias for [`RestoreValidationResult`].
pub type ValidationResult = RestoreValidationResult;

/// Result of validating a backup
#[derive(Debug, Clone)]
pub struct RestoreValidationResult {
    /// Whether backup is valid
    pub valid: bool,
    /// Total number of files in backup
    pub total_files: usize,
    /// Total bytes in backup
    pub total_bytes: u64,
    /// Errors found during validation
    pub errors: Vec<String>,
    /// Warnings found during validation
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::backup::BackupType;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_restore_manager_creation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir for test");
        let base_path = temp_dir.path();
        let storage = UnifiedCachingFilesystem::new_local(base_path)
            .await
            .expect("Failed to create storage for test");
        let config = RestoreConfig::default();

        let restore_manager = RestoreManager::new(base_path, storage, config);
        assert!(restore_manager.is_ok());
    }

    #[tokio::test]
    async fn test_calculate_checksum() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir for test");
        let base_path = temp_dir.path();
        let storage = UnifiedCachingFilesystem::new_local(base_path)
            .await
            .expect("Failed to create storage for test");
        let config = RestoreConfig::default();

        let restore_manager = RestoreManager::new(base_path, storage, config)
            .expect("Failed to create restore manager for test");

        let data = b"Hello, World!";
        let checksum = restore_manager.calculate_checksum(data);

        assert_eq!(checksum.len(), 64); // SHA-256 = 64 hex chars
    }

    #[tokio::test]
    async fn test_restore_dry_run() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir for test");
        let backup_id = "test_backup";
        let backup_dir = temp_dir.path().join(backup_id);
        let restore_dir = temp_dir.path().join("restore");
        tokio::fs::create_dir_all(&backup_dir)
            .await
            .expect("Failed to create backup dir for test");
        tokio::fs::create_dir_all(&restore_dir)
            .await
            .expect("Failed to create restore dir for test");

        // Create a test file in backup
        let test_file = backup_dir.join("test.dat");
        tokio::fs::write(&test_file, b"test data")
            .await
            .expect("Failed to write test file for test");

        let storage = UnifiedCachingFilesystem::new_local(&restore_dir)
            .await
            .expect("Failed to create storage for test");
        let config = RestoreConfig {
            dry_run: true,
            verify_checksums: false,
            target: BackupTarget::Local {
                path: temp_dir.path().to_path_buf(),
            },
            ..Default::default()
        };

        let restore_manager = RestoreManager::new(&restore_dir, storage, config)
            .expect("Failed to create restore manager for test");

        // Create a mock manifest
        let manifest = BackupManifest {
            backup_id: backup_id.to_string(),
            timestamp: 0,
            lsn_range: (0, 100),
            data_files: vec![DataFileMetadata {
                relative_path: "test.dat".to_string(),
                size: 9,
                checksum: restore_manager.calculate_checksum(b"test data"),
                modified_time: 0,
            }],
            total_bytes: 9,
            backup_type: BackupType::Full,
            manifest_checksum: String::new(),
            previous_backup_id: None,
            wal_segments: vec![],
        };

        let result = restore_manager
            .restore_from_backup(&manifest)
            .await
            .expect("Failed to restore from backup in test");

        assert!(result.success);
        assert_eq!(result.files_restored, 1);
        assert_eq!(result.bytes_restored, 9);

        // File should not actually exist in restore dir (dry run)
        assert!(!restore_dir.join("test.dat").exists());
    }

    #[tokio::test]
    async fn test_validate_backup() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir for test");
        let backup_id = "test_backup";
        let backup_dir = temp_dir.path().join(backup_id);
        let restore_dir = temp_dir.path().join("restore");
        tokio::fs::create_dir_all(&backup_dir)
            .await
            .expect("Failed to create backup dir for test");
        tokio::fs::create_dir_all(&restore_dir)
            .await
            .expect("Failed to create restore dir for test");

        // Create test files
        let test_file = backup_dir.join("test.dat");
        tokio::fs::write(&test_file, b"test data")
            .await
            .expect("Failed to write test file for test");

        // Create manifest
        let storage = UnifiedCachingFilesystem::new_local(&restore_dir)
            .await
            .expect("Failed to create storage for test");
        let restore_manager =
            RestoreManager::new(&restore_dir, storage.clone(), RestoreConfig::default())
                .expect("Failed to create restore manager for test");

        let checksum = restore_manager.calculate_checksum(b"test data");
        let mut manifest = BackupManifest {
            backup_id: backup_id.to_string(),
            timestamp: 0,
            lsn_range: (0, 100),
            data_files: vec![DataFileMetadata {
                relative_path: "test.dat".to_string(),
                size: 9,
                checksum: checksum.clone(),
                modified_time: 0,
            }],
            total_bytes: 9,
            backup_type: BackupType::Full,
            manifest_checksum: String::new(),
            previous_backup_id: None,
            wal_segments: vec![],
        };
        let unsigned_manifest_json = serde_json::to_string_pretty(&manifest)
            .expect("Failed to serialize unsigned manifest for test");
        manifest.manifest_checksum =
            restore_manager.calculate_checksum(unsigned_manifest_json.as_bytes());

        // Write manifest to backup dir
        let manifest_path = backup_dir.join("manifest.json");
        let manifest_json =
            serde_json::to_string_pretty(&manifest).expect("Failed to serialize manifest for test");
        tokio::fs::write(&manifest_path, manifest_json)
            .await
            .expect("Failed to write manifest for test");

        let config = RestoreConfig {
            target: BackupTarget::Local {
                path: temp_dir.path().to_path_buf(),
            },
            ..Default::default()
        };
        let restore_manager = RestoreManager::new(&restore_dir, storage, config)
            .expect("Failed to create restore manager for test");

        let validation = restore_manager
            .validate_backup(&manifest)
            .await
            .expect("Failed to validate backup in test");

        assert!(validation.valid);
        assert_eq!(validation.total_files, 1);
        assert!(validation.errors.is_empty());
    }
}
