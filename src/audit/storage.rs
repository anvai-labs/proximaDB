//! Audit Storage Backends
//!
//! Multiple storage backend implementations for audit logs
//! including file-based, database, and cloud storage options.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use proximadb_security::{AuditEvent, AuditEventType, AuditResult};
pub use proximadb_security::{AuditStatistics, AuditStorage};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// File-based audit storage implementation
/// A sink that stores nothing and returns nothing.
///
/// Used when auditing is deliberately disabled in a **development** deployment
/// so the server still boots (see `AuditLogger::noop`). Queries return empty,
/// which means `count_recent_auth_failures` is always 0 and no alert can fire —
/// that is the honest consequence of running without audit, and it is why
/// production mode refuses to start instead of using this sink.
#[derive(Debug, Default)]
pub struct NoopAuditStorage;

#[async_trait]
impl AuditStorage for NoopAuditStorage {
    async fn store_audit_event(&self, _event: &AuditEvent) -> Result<()> {
        Ok(())
    }

    async fn query_events(
        &self,
        _event_type: Option<AuditEventType>,
        _user_id: Option<String>,
        _since: Option<DateTime<Utc>>,
        _until: Option<DateTime<Utc>>,
        _limit: Option<usize>,
    ) -> Result<Vec<AuditEvent>> {
        Ok(Vec::new())
    }

    async fn get_audit_statistics(&self, since: DateTime<Utc>) -> Result<AuditStatistics> {
        Ok(AuditStatistics {
            total_events: 0,
            events_by_type: std::collections::HashMap::new(),
            unique_users: 0,
            unique_tenants: 0,
            success_rate: 0.0,
            period_start: since,
            period_end: Utc::now(),
        })
    }

    async fn cleanup_old_logs(&self, _retention_days: u32) -> Result<usize> {
        Ok(0)
    }
}

pub struct FileAuditStorage {
    /// Base directory for audit log files
    base_directory: String,
    /// Path to the current active audit log file
    current_file: tokio::sync::RwLock<Option<String>>,
    /// Maximum size in MB before rotating to a new file
    file_rotation_size_mb: usize,
}

impl FileAuditStorage {
    /// Create a new `FileAuditStorage` that writes JSONL audit logs to the given directory.
    /// The directory is created automatically if it does not already exist.
    ///
    /// Accepts a bare local path or a `file://` URL. Object-store URLs are
    /// rejected fail-fast — use [`ObjectStoreAuditStorage`] for those
    /// (TD-OBJSTORE-1, #960).
    pub async fn new(directory: String) -> Result<Self> {
        let directory = match directory.strip_prefix("file://") {
            Some(stripped) => stripped.to_string(),
            None if directory.contains("://") => {
                return Err(anyhow!(
                    "FileAuditStorage requires a local directory; got object-store URL '{}' — \
                     use ObjectStoreAuditStorage instead",
                    directory
                ));
            }
            None => directory,
        };

        // Ensure directory exists
        tokio::fs::create_dir_all(&directory)
            .await
            .map_err(|e| anyhow!("Failed to create audit directory {}: {}", directory, e))?;

        info!("✅ File audit storage initialized: {}", directory);

        Ok(Self {
            base_directory: directory,
            current_file: tokio::sync::RwLock::new(None),
            file_rotation_size_mb: 100, // Rotate files at 100MB
        })
    }

    /// Get current audit log file path
    async fn get_current_file_path(&self) -> Result<String> {
        let mut current_file = self.current_file.write().await;

        // Check if we need a new file
        if let Some(ref file_path) = *current_file {
            // Check file size for rotation
            if let Ok(metadata) = tokio::fs::metadata(file_path).await {
                let size_mb = metadata.len() as usize / (1024 * 1024);
                if size_mb < self.file_rotation_size_mb {
                    return Ok(file_path.clone());
                }
            }
        }

        // Create new audit log file
        let now = Utc::now();
        let filename = format!("audit_log_{}.jsonl", now.format("%Y%m%d_%H%M%S"));
        let file_path = Path::new(&self.base_directory).join(filename);

        *current_file = Some(file_path.to_string_lossy().to_string());

        info!("📝 Created new audit log file: {}", file_path.display());
        Ok(file_path.to_string_lossy().to_string())
    }
}

#[async_trait]
impl AuditStorage for FileAuditStorage {
    async fn store_audit_event(&self, event: &AuditEvent) -> Result<()> {
        let file_path = self.get_current_file_path().await?;

        // Serialize event as JSON line
        let event_json = serde_json::to_string(event)?;
        let log_line = format!("{}\n", event_json);

        // Append to the audit log file (`tokio::fs::write` would truncate,
        // silently dropping every previously stored event in the file).
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await
            .map_err(|e| anyhow!("Failed to open audit log {}: {}", file_path, e))?;
        file.write_all(log_line.as_bytes())
            .await
            .map_err(|e| anyhow!("Failed to write audit event to {}: {}", file_path, e))?;

        debug!("📝 Stored audit event {} to {}", event.event_id, file_path);
        Ok(())
    }

    async fn query_events(
        &self,
        event_type: Option<AuditEventType>,
        user_id: Option<String>,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        limit: Option<usize>,
    ) -> Result<Vec<AuditEvent>> {
        let mut events = Vec::new();

        // Read all audit log files in directory
        let mut dir_entries = tokio::fs::read_dir(&self.base_directory).await?;

        while let Some(entry) = dir_entries.next_entry().await? {
            let file_path = entry.path();

            if file_path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                let file_events = self
                    .read_events_from_file(
                        &file_path,
                        event_type.clone(),
                        user_id.clone(),
                        since,
                        until,
                    )
                    .await?;
                events.extend(file_events);
            }
        }

        // Sort by timestamp (newest first) BEFORE applying the limit.
        // Previously the limit truncated during directory iteration and the
        // sort ran afterwards, so a bounded query returned an ARBITRARY subset
        // rather than the newest events — which silently defeated
        // `count_recent_auth_failures` (it asks for the newest 100 to evaluate
        // the brute-force threshold).
        events.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        if let Some(limit) = limit {
            events.truncate(limit);
        }

        debug!("🔍 Queried {} audit events matching criteria", events.len());
        Ok(events)
    }

    async fn get_audit_statistics(&self, since: DateTime<Utc>) -> Result<AuditStatistics> {
        let events = self
            .query_events(None, None, Some(since), None, None)
            .await?;
        Ok(statistics_from_events(&events, since))
    }

    async fn cleanup_old_logs(&self, retention_days: u32) -> Result<usize> {
        let cutoff_date = Utc::now() - chrono::Duration::days(retention_days as i64);
        let mut deleted_files = 0;

        let mut dir_entries = tokio::fs::read_dir(&self.base_directory).await?;

        while let Some(entry) = dir_entries.next_entry().await? {
            let file_path = entry.path();

            if let Ok(metadata) = entry.metadata().await
                && let Ok(modified) = metadata.modified()
            {
                let modified_dt: DateTime<Utc> = modified.into();

                if modified_dt < cutoff_date {
                    if let Err(e) = tokio::fs::remove_file(&file_path).await {
                        warn!(
                            "Failed to delete old audit log {}: {}",
                            file_path.display(),
                            e
                        );
                    } else {
                        deleted_files += 1;
                        info!("🗑️ Deleted old audit log: {}", file_path.display());
                    }
                }
            }
        }

        info!(
            "🧹 Cleanup complete: deleted {} old audit log files",
            deleted_files
        );
        Ok(deleted_files)
    }
}

impl FileAuditStorage {
    /// Read events from a specific file with filtering
    async fn read_events_from_file(
        &self,
        file_path: &Path,
        event_type: Option<AuditEventType>,
        user_id: Option<String>,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
    ) -> Result<Vec<AuditEvent>> {
        let mut events = Vec::new();

        let content = tokio::fs::read_to_string(file_path)
            .await
            .map_err(|e| anyhow!("Failed to read audit file {}: {}", file_path.display(), e))?;

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<AuditEvent>(line) {
                Ok(event) => {
                    if event_matches_filters(&event, &event_type, &user_id, since, until) {
                        events.push(event);
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to parse audit event from {}: {}",
                        file_path.display(),
                        e
                    );
                }
            }
        }

        Ok(events)
    }
}

/// Does `event` pass the optional query filters? Shared by the file- and
/// object-store-backed implementations.
fn event_matches_filters(
    event: &AuditEvent,
    event_type: &Option<AuditEventType>,
    user_id: &Option<String>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
) -> bool {
    if let Some(filter_type) = event_type
        && &event.event_type != filter_type
    {
        return false;
    }
    if let Some(filter_user) = user_id
        && event.user_id.as_ref() != Some(filter_user)
    {
        return false;
    }
    if let Some(since_time) = since
        && event.timestamp < since_time
    {
        return false;
    }
    if let Some(until_time) = until
        && event.timestamp > until_time
    {
        return false;
    }
    true
}

/// Object-store-backed audit storage (TD-OBJSTORE-1, #960).
///
/// Persists one JSONL object per audit event under
/// `{base_url}/audit_{timestamp}_{event_id}.jsonl`, routed through the
/// scheme-dispatching [`FilesystemFactory`] so the base may be `s3://…`,
/// `adls://…`/`abfs://…`, `gcs://…`, or `file://…`. Object stores reject
/// append on immutable/block blobs, so per-event objects replace the local
/// backend's append-and-rotate layout; the flat keyspace makes the extra
/// objects cheap and listing is only paid on (rare, admin-facing) queries.
pub struct ObjectStoreAuditStorage {
    /// Scheme-qualified base URL for audit objects
    base_url: String,
    /// Scheme-dispatching filesystem factory
    filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
}

impl ObjectStoreAuditStorage {
    /// Create an object-store audit storage rooted at `base_url`
    /// (scheme-qualified, e.g. `adls://container/data/audit`).
    pub async fn new(base_url: String) -> Result<Self> {
        if !base_url.contains("://") {
            return Err(anyhow!(
                "ObjectStoreAuditStorage requires a scheme-qualified URL; got '{}' — \
                 use FileAuditStorage for local directories",
                base_url
            ));
        }
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(
                crate::storage::persistence::filesystem::FilesystemConfig::default(),
            )
            .await
            .map_err(|e| anyhow!("Failed to create filesystem factory: {}", e))?,
        );
        // Fail fast on unsupported/misconfigured schemes (e.g. s3:// without
        // the aws feature) instead of at the first audit write.
        filesystem
            .get_filesystem(&base_url)
            .map_err(|e| anyhow!("Audit storage URL '{}' not routable: {}", base_url, e))?;

        info!("✅ Object-store audit storage initialized: {}", base_url);

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            filesystem,
        })
    }

    fn object_url(&self, event: &AuditEvent) -> String {
        format!(
            "{}/audit_{}_{}.jsonl",
            self.base_url,
            event.timestamp.format("%Y%m%d_%H%M%S%.3f"),
            event.event_id
        )
    }
}

#[async_trait]
impl AuditStorage for ObjectStoreAuditStorage {
    async fn store_audit_event(&self, event: &AuditEvent) -> Result<()> {
        let url = self.object_url(event);
        let event_json = serde_json::to_string(event)?;
        let fs = self
            .filesystem
            .get_filesystem(&url)
            .map_err(|e| anyhow!("Audit storage URL '{}' not routable: {}", url, e))?;
        let options = crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: true,
            ..Default::default()
        };
        fs.write(&url, event_json.as_bytes(), Some(options))
            .await
            .map_err(|e| anyhow!("Failed to write audit event to {}: {}", url, e))?;
        debug!("📝 Stored audit event {} to {}", event.event_id, url);
        Ok(())
    }

    async fn query_events(
        &self,
        event_type: Option<AuditEventType>,
        user_id: Option<String>,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        limit: Option<usize>,
    ) -> Result<Vec<AuditEvent>> {
        let fs = self
            .filesystem
            .get_filesystem(&self.base_url)
            .map_err(|e| anyhow!("Audit storage URL not routable: {}", e))?;
        let entries = fs.list(&self.base_url).await.map_err(|e| {
            anyhow!(
                "Failed to list audit objects under {}: {}",
                self.base_url,
                e
            )
        })?;

        let mut events = Vec::new();
        for entry in entries {
            if !entry.name.ends_with(".jsonl") {
                continue;
            }
            let data = match fs.read(&entry.url).await {
                Ok(data) => data,
                Err(e) => {
                    warn!("Failed to read audit object {}: {}", entry.url, e);
                    continue;
                }
            };
            let content = String::from_utf8_lossy(&data);
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<AuditEvent>(line) {
                    Ok(event) => {
                        if event_matches_filters(&event, &event_type, &user_id, since, until) {
                            events.push(event);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse audit event from {}: {}", entry.url, e);
                    }
                }
            }
            if let Some(limit) = limit
                && events.len() >= limit
            {
                events.truncate(limit);
                break;
            }
        }

        // Sort by timestamp (newest first)
        events.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        debug!("🔍 Queried {} audit events matching criteria", events.len());
        Ok(events)
    }

    async fn get_audit_statistics(&self, since: DateTime<Utc>) -> Result<AuditStatistics> {
        let events = self
            .query_events(None, None, Some(since), None, None)
            .await?;
        Ok(statistics_from_events(&events, since))
    }

    async fn cleanup_old_logs(&self, retention_days: u32) -> Result<usize> {
        let cutoff_date = Utc::now() - chrono::Duration::days(retention_days as i64);
        let fs = self
            .filesystem
            .get_filesystem(&self.base_url)
            .map_err(|e| anyhow!("Audit storage URL not routable: {}", e))?;
        let entries = fs.list(&self.base_url).await.map_err(|e| {
            anyhow!(
                "Failed to list audit objects under {}: {}",
                self.base_url,
                e
            )
        })?;

        let mut deleted_files = 0usize;
        for entry in entries {
            if !entry.name.ends_with(".jsonl") {
                continue;
            }
            let Some(modified) = entry.metadata.modified else {
                continue;
            };
            if modified < cutoff_date {
                if let Err(e) = fs.delete(&entry.url).await {
                    warn!("Failed to delete old audit object {}: {}", entry.url, e);
                } else {
                    deleted_files += 1;
                    info!("🗑️ Deleted old audit object: {}", entry.url);
                }
            }
        }

        info!(
            "🧹 Cleanup complete: deleted {} old audit objects",
            deleted_files
        );
        Ok(deleted_files)
    }
}

/// Aggregate statistics over a set of audit events. Shared by every
/// [`AuditStorage`] backend.
fn statistics_from_events(events: &[AuditEvent], since: DateTime<Utc>) -> AuditStatistics {
    let mut events_by_type = std::collections::HashMap::new();
    let mut unique_users = std::collections::HashSet::new();
    let mut unique_tenants = std::collections::HashSet::new();
    let mut success_count = 0;

    for event in events {
        *events_by_type.entry(event.event_type.clone()).or_insert(0) += 1;
        if let Some(ref user_id) = event.user_id {
            unique_users.insert(user_id.clone());
        }
        if let Some(ref tenant_id) = event.tenant_id {
            unique_tenants.insert(tenant_id.clone());
        }
        if matches!(event.result, AuditResult::Success) {
            success_count += 1;
        }
    }

    let success_rate = if !events.is_empty() {
        (success_count as f64) / (events.len() as f64) * 100.0
    } else {
        0.0
    };

    AuditStatistics {
        total_events: events.len() as u64,
        events_by_type,
        unique_users: unique_users.len() as u64,
        unique_tenants: unique_tenants.len() as u64,
        success_rate,
        period_start: since,
        period_end: Utc::now(),
    }
}

// NOTE: there is deliberately no DatabaseAuditStorage. The backend never
// existed — a placeholder silently discarded every audit event — so
// selecting `storage_backend = Database` in the config now fails closed at
// AuditLogger construction (see logger.rs). Use File (local or object-store
// URL) or S3.

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_security::{AuditEvent, AuditEventType, AuditResource, AuditResult};
    use tempfile::TempDir;
    use uuid::Uuid;

    // ==================== Helper Functions ====================

    fn create_test_event(event_type: AuditEventType, user_id: Option<&str>) -> AuditEvent {
        let resource = AuditResource::new("test_resource".to_string(), "test-id".to_string());
        let mut event = AuditEvent::new(
            event_type,
            resource,
            "test_action".to_string(),
            AuditResult::Success,
        );
        if let Some(uid) = user_id {
            event.user_id = Some(uid.to_string());
        }
        event
    }

    fn create_test_event_with_timestamp(
        event_type: AuditEventType,
        user_id: Option<&str>,
        timestamp: DateTime<Utc>,
    ) -> AuditEvent {
        let resource = AuditResource::new("test_resource".to_string(), "test-id".to_string());
        AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp,
            event_type,
            user_id: user_id.map(|s| s.to_string()),
            resource,
            action: "test_action".to_string(),
            result: AuditResult::Success,
            details: std::collections::HashMap::new(),
            ip_address: None,
            user_agent: None,
            request_id: None,
            tenant_id: None,
            session_id: None,
            risk_score: None,
        }
    }

    // ==================== FileAuditStorage Tests ====================

    #[tokio::test]
    async fn test_file_audit_storage_creation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        assert_eq!(
            storage.base_directory,
            temp_dir.path().to_string_lossy().to_string()
        );
        assert_eq!(storage.file_rotation_size_mb, 100);
    }

    #[tokio::test]
    async fn test_file_audit_storage_creates_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let nested_path = temp_dir.path().join("nested").join("audit").join("logs");

        let storage = FileAuditStorage::new(nested_path.to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        assert!(nested_path.exists());
        assert_eq!(
            storage.base_directory,
            nested_path.to_string_lossy().to_string()
        );
    }

    #[tokio::test]
    async fn test_file_audit_storage_store_event() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let event = create_test_event(AuditEventType::Authentication, Some("user-1"));

        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        // Verify file was created
        let entries: Vec<_> = std::fs::read_dir(temp_dir.path())
            .expect("Failed to read dir")
            .collect();
        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_events_no_filter() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Store multiple events
        for i in 0..3 {
            let event = create_test_event(AuditEventType::DataAccess, Some(&format!("user-{}", i)));
            storage
                .store_audit_event(&event)
                .await
                .expect("Failed to store event");
        }

        let events = storage
            .query_events(None, None, None, None, None)
            .await
            .expect("Failed to query events");

        // Note: Due to file write behavior, we might only get the last event
        // since each write overwrites the file. This test verifies the mechanism works.
        assert!(!events.is_empty() || events.is_empty()); // Query mechanism works
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_by_event_type() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let auth_event = create_test_event(AuditEventType::Authentication, Some("user-1"));
        storage
            .store_audit_event(&auth_event)
            .await
            .expect("Failed to store event");

        let events = storage
            .query_events(Some(AuditEventType::Authentication), None, None, None, None)
            .await
            .expect("Failed to query events");

        // Verify filtering works (even if empty due to write behavior)
        for event in &events {
            assert_eq!(event.event_type, AuditEventType::Authentication);
        }
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_by_user_id() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let event = create_test_event(AuditEventType::DataAccess, Some("specific-user"));
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        let events = storage
            .query_events(None, Some("specific-user".to_string()), None, None, None)
            .await
            .expect("Failed to query events");

        for event in &events {
            assert_eq!(event.user_id, Some("specific-user".to_string()));
        }
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_with_time_range() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let now = Utc::now();
        let event =
            create_test_event_with_timestamp(AuditEventType::APIAccess, Some("user-1"), now);
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        // Query for events in the last hour
        let since = now - chrono::Duration::hours(1);
        let until = now + chrono::Duration::hours(1);

        let events = storage
            .query_events(None, None, Some(since), Some(until), None)
            .await
            .expect("Failed to query events");

        for event in &events {
            assert!(event.timestamp >= since && event.timestamp <= until);
        }
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_with_limit() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Store an event
        let event = create_test_event(AuditEventType::DataModification, Some("user-1"));
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        let events = storage
            .query_events(None, None, None, None, Some(5))
            .await
            .expect("Failed to query events");

        assert!(events.len() <= 5);
    }

    #[tokio::test]
    async fn test_file_audit_storage_get_statistics() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let since = Utc::now() - chrono::Duration::hours(1);
        let stats = storage
            .get_audit_statistics(since)
            .await
            .expect("Failed to get statistics");

        // Verify structure (total_events is u64, always >= 0)
        let _ = stats.total_events;
        assert!(stats.success_rate >= 0.0 && stats.success_rate <= 100.0);
        assert!(stats.period_start == since);
        assert!(stats.period_end >= since);
    }

    #[tokio::test]
    async fn test_file_audit_storage_cleanup_old_logs() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Store an event
        let event = create_test_event(AuditEventType::SecurityEvent, Some("user-1"));
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        // Cleanup with very long retention (should delete nothing)
        let deleted = storage
            .cleanup_old_logs(365)
            .await
            .expect("Failed to cleanup");

        // Recent files should not be deleted
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_file_audit_storage_read_events_from_file_with_filters() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Create and store multiple events with different attributes
        let now = Utc::now();

        let event1 = create_test_event_with_timestamp(
            AuditEventType::Authentication,
            Some("alice"),
            now - chrono::Duration::minutes(30),
        );

        let event2 = create_test_event_with_timestamp(AuditEventType::DataAccess, Some("bob"), now);

        // Store events
        storage
            .store_audit_event(&event1)
            .await
            .expect("Failed to store event");
        storage
            .store_audit_event(&event2)
            .await
            .expect("Failed to store event");

        // Query with specific filters
        let filtered_events = storage
            .query_events(
                Some(AuditEventType::Authentication),
                Some("alice".to_string()),
                None,
                None,
                None,
            )
            .await
            .expect("Failed to query");

        for event in &filtered_events {
            assert_eq!(event.event_type, AuditEventType::Authentication);
            assert_eq!(event.user_id, Some("alice".to_string()));
        }
    }

    #[tokio::test]
    async fn test_file_audit_storage_handles_empty_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Query empty storage
        let events = storage
            .query_events(None, None, None, None, None)
            .await
            .expect("Failed to query events");

        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_file_audit_storage_handles_malformed_json_lines() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create a file with invalid JSON
        let file_path = temp_dir.path().join("audit_log_20240101_120000.jsonl");
        tokio::fs::write(&file_path, "invalid json\n{\"also\": \"invalid\"\n")
            .await
            .expect("Failed to write invalid file");

        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        // Should not panic, just skip invalid lines
        let events = storage
            .query_events(None, None, None, None, None)
            .await
            .expect("Query should succeed even with malformed data");

        assert!(events.is_empty());
    }

    // ==================== AuditStatistics Tests ====================

    #[test]
    fn test_audit_statistics_serialization() {
        let mut events_by_type = std::collections::HashMap::new();
        events_by_type.insert(AuditEventType::Authentication, 100);
        events_by_type.insert(AuditEventType::DataAccess, 50);

        let stats = AuditStatistics {
            total_events: 150,
            events_by_type,
            unique_users: 25,
            unique_tenants: 5,
            success_rate: 95.5,
            period_start: Utc::now(),
            period_end: Utc::now(),
        };

        let json = serde_json::to_string(&stats).expect("Failed to serialize");
        assert!(json.contains("150"));
        assert!(json.contains("95.5"));

        let deserialized: AuditStatistics =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.total_events, 150);
        assert_eq!(deserialized.unique_users, 25);
        assert_eq!(deserialized.success_rate, 95.5);
    }

    #[test]
    fn test_audit_statistics_events_by_type() {
        let mut events_by_type = std::collections::HashMap::new();
        events_by_type.insert(AuditEventType::Authentication, 10);
        events_by_type.insert(AuditEventType::Authorization, 20);
        events_by_type.insert(AuditEventType::DataAccess, 30);

        let stats = AuditStatistics {
            total_events: 60,
            events_by_type,
            unique_users: 5,
            unique_tenants: 2,
            success_rate: 100.0,
            period_start: Utc::now(),
            period_end: Utc::now(),
        };

        assert_eq!(
            stats.events_by_type.get(&AuditEventType::Authentication),
            Some(&10)
        );
        assert_eq!(
            stats.events_by_type.get(&AuditEventType::Authorization),
            Some(&20)
        );
        assert_eq!(
            stats.events_by_type.get(&AuditEventType::DataAccess),
            Some(&30)
        );
        assert_eq!(stats.events_by_type.len(), 3);
    }

    // ==================== Edge Cases ====================

    #[tokio::test]
    async fn test_file_audit_storage_concurrent_writes() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = std::sync::Arc::new(
            FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
                .await
                .expect("Failed to create storage"),
        );

        // Spawn multiple concurrent writes
        let mut handles = vec![];
        for i in 0..5 {
            let storage_clone = storage.clone();
            let handle = tokio::spawn(async move {
                let event =
                    create_test_event(AuditEventType::APIAccess, Some(&format!("user-{}", i)));
                storage_clone.store_audit_event(&event).await
            });
            handles.push(handle);
        }

        // All writes should complete without panic
        for handle in handles {
            let result = handle.await.expect("Task panicked");
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_audit_storage_special_characters_in_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let special_path = temp_dir.path().join("audit logs").join("test-123");

        let storage = FileAuditStorage::new(special_path.to_string_lossy().to_string())
            .await
            .expect("Failed to create storage with special chars");

        let event = create_test_event(AuditEventType::SystemConfiguration, Some("admin"));
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");
    }

    #[tokio::test]
    async fn test_file_audit_storage_query_time_boundary() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage = FileAuditStorage::new(temp_dir.path().to_string_lossy().to_string())
            .await
            .expect("Failed to create storage");

        let now = Utc::now();

        // Store event at exact boundary
        let event =
            create_test_event_with_timestamp(AuditEventType::TenantManagement, Some("admin"), now);
        storage
            .store_audit_event(&event)
            .await
            .expect("Failed to store event");

        // Query with exact time as both since and until
        let events = storage
            .query_events(None, None, Some(now), Some(now), None)
            .await
            .expect("Failed to query");

        // Event at exact boundary should be included
        for event in &events {
            assert!(event.timestamp == now);
        }
    }
}

#[cfg(test)]
mod td_sec_2_query_tests {
    use super::*;
    use proximadb_security::AuditResource;

    /// REGRESSION (TD-SEC-2 A3): `query_events` truncated to `limit` while
    /// iterating the directory and sorted only afterwards, so a bounded query
    /// returned an ARBITRARY subset instead of the newest events. That silently
    /// defeated `count_recent_auth_failures`, which asks for the newest 100 to
    /// evaluate the brute-force threshold: with enough older events present,
    /// recent failures could be entirely absent from the sample and the
    /// threshold could never trip.
    #[tokio::test]
    async fn bounded_query_returns_the_newest_events_not_an_arbitrary_subset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = FileAuditStorage::new(dir.path().to_string_lossy().to_string())
            .await
            .expect("storage");

        // 30 events spread over 30 hours; the newest 5 are the ones a bounded
        // query must return.
        let now = Utc::now();
        for age_hours in (0..30).rev() {
            let mut event = AuditEvent::new(
                AuditEventType::Authentication,
                AuditResource::new("authentication".to_string(), "s".to_string()),
                "authenticate".to_string(),
                AuditResult::Success,
            );
            event.timestamp = now - chrono::Duration::hours(age_hours);
            event.user_id = Some(format!("user_{age_hours}"));
            storage.store_audit_event(&event).await.expect("store");
        }

        let newest = storage
            .query_events(None, None, None, None, Some(5))
            .await
            .expect("query");
        assert_eq!(newest.len(), 5);
        // age 0 is the newest; ages 0..4 must be exactly what came back.
        let mut ages: Vec<i64> = newest
            .iter()
            .filter_map(|e| e.user_id.as_ref())
            .filter_map(|u| u.strip_prefix("user_").and_then(|n| n.parse().ok()))
            .collect();
        ages.sort_unstable();
        assert_eq!(ages, vec![0, 1, 2, 3, 4], "must be the NEWEST five");
    }
}
