use crate::core::Config;
use crate::storage::persistence::filesystem::FilesystemFactory;
use std::sync::Arc;
use tracing::{debug, info};

/// Loads and merges configuration from TOML files and environment variables
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration with default merging and unified filesystem support
    ///
    /// Supports all filesystem schemes via unified filesystem API:
    /// - file://path/to/config.toml (local filesystem)
    /// - s3://bucket/path/config.toml (AWS S3)
    /// - adls://account.dfs.core.windows.net/container/path/config.toml (Azure Data Lake)
    /// - gcs://bucket/path/config.toml (Google Cloud Storage)
    /// - /path/to/config.toml (automatically converted to file://)
    pub fn load_with_defaults<P: AsRef<str>>(
        config_path: P,
    ) -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
        let config_url = config_path.as_ref();

        info!("🔧 Loading configuration from: {}", config_url);

        // Use Handle::try_current to check if we're in a runtime
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // We're already in a runtime, use block_in_place
            tokio::task::block_in_place(|| handle.block_on(Self::load_config_async(config_url)))
        } else {
            // Not in a runtime, create one
            use tokio::runtime::Runtime;
            let rt = Runtime::new()?;
            rt.block_on(async { Self::load_config_async(config_url).await })
        }
    }

    /// Async config loading using unified filesystem API
    async fn load_config_async(
        config_url: &str,
    ) -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
        // Create filesystem factory
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);

        // Start with default configuration
        let base_config = Config::default();

        // Check if config file exists using unified filesystem API
        if filesystem_factory.exists(config_url).await? {
            debug!("📄 Reading config file: {}", config_url);

            // Read config file using unified filesystem API
            let config_data = filesystem_factory.read(config_url).await?;
            let config_str = String::from_utf8(config_data)?;

            // Parse as TOML value for selective merging
            let user_toml: toml::Value = toml::from_str(&config_str)?;

            // Merge user configuration with defaults
            let merged_config = Self::merge_config_with_defaults(base_config, user_toml)?;

            info!(
                "✅ Configuration loaded and merged with defaults from: {}",
                config_url
            );
            Ok(merged_config)
        } else {
            info!("ℹ️ Config file not found, using defaults: {}", config_url);
            Ok(base_config)
        }
    }

    /// Get cloud authentication configuration
    pub fn cloud_auth_info() -> String {
        let mut auth_info = Vec::new();

        // AWS authentication
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok()
            && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok()
        {
            auth_info.push("AWS: Access Key + Secret Key");
        } else if std::env::var("AWS_PROFILE").is_ok() {
            auth_info.push("AWS: Profile-based");
        } else {
            auth_info.push("AWS: Instance Role/Default");
        }

        // Azure authentication
        if std::env::var("AZURE_STORAGE_ACCOUNT").is_ok()
            && std::env::var("AZURE_STORAGE_ACCESS_KEY").is_ok()
        {
            auth_info.push("Azure: Storage Account + Access Key");
        } else if std::env::var("AZURE_CLIENT_ID").is_ok() {
            auth_info.push("Azure: Service Principal");
        } else {
            auth_info.push("Azure: Managed Identity");
        }

        // GCP authentication
        if std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok() {
            auth_info.push("GCP: Service Account JSON");
        } else {
            auth_info.push("GCP: Default Application Credentials");
        }

        format!("🔐 Cloud Authentication: {}", auth_info.join(", "))
    }

    /// Merge user configuration with default configuration
    fn merge_config_with_defaults(
        base_config: Config,
        user_toml: toml::Value,
    ) -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
        // Serialize base config to TOML for merging
        let mut base_toml = toml::Value::try_from(&base_config)?;

        // Recursively merge user values into base
        Self::merge_toml_values(&mut base_toml, user_toml);

        // Resolve ${VAR} placeholders before deserializing into typed config.
        Self::expand_env_placeholders(&mut base_toml)?;

        // Deserialize merged TOML back to Config struct
        let mut merged_config: Config = base_toml.try_into()?;

        // Resolve all relative paths to absolute paths
        Self::resolve_config_paths(&mut merged_config)?;
        Self::validate_config(&merged_config)?;

        Ok(merged_config)
    }

    fn validate_config(config: &Config) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(query_config) = &config.query {
            query_config.validate().map_err(|err| {
                let message = format!("Invalid query configuration: {err}");
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    message,
                )) as Box<dyn std::error::Error + Send + Sync>
            })?;
        }
        if let Some(hybrid_config) = &config.hybrid {
            hybrid_config.validate().map_err(|err| {
                let message = format!("Invalid hybrid configuration: {err}");
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    message,
                )) as Box<dyn std::error::Error + Send + Sync>
            })?;
        }
        if let Some(sst_config) = &config.storage.sst_config {
            sst_config.validate().map_err(|err| {
                let message = format!("Invalid SST configuration: {err}");
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    message,
                )) as Box<dyn std::error::Error + Send + Sync>
            })?;
        }

        // TD-IOBUDGET-1: fail closed on invalid per-location I/O budgets — an
        // operator error (unknown disk class already dies at TOML parse; bad
        // bounds here) must abort startup, not silently change GET geometry.
        for location in &config.storage.storage_locations {
            if let Some(budget_cfg) = &location.io_budget {
                crate::core::config::StorageConfig::resolve_location_io_budget(
                    &location.url,
                    budget_cfg,
                )
                .map_err(|err| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Invalid io_budget for storage location {}: {err}",
                            location.url
                        ),
                    )) as Box<dyn std::error::Error + Send + Sync>
                })?;
            }
        }
        // TD-IOBUDGET-1 (review finding): duplicate location URLs with
        // CONFLICTING resolved budgets would silently let the last entry win
        // (the leaf registry is URL-keyed with insert-overwrite).
        crate::core::config::StorageConfig::validate_io_budget_conflicts(&config.storage).map_err(
            |err| {
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, err))
                    as Box<dyn std::error::Error + Send + Sync>
            },
        )?;

        // TD-TIER-1 SC3: reservation pools validate at startup — per-pool
        // shape plus duplicate id/path rejection (two pools sharing an id
        // would collide in the metering namespace; sharing a path would fight
        // over one exclusive ledger).
        crate::services::collection::reservation::validate_pool_configs(
            &config.storage.reservation_pools,
        )
        .map_err(|err| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid reservation pool configuration: {err}"),
            )) as Box<dyn std::error::Error + Send + Sync>
        })?;

        Ok(())
    }

    /// Recursively merge TOML values
    fn merge_toml_values(base: &mut toml::Value, user: toml::Value) {
        match (&mut *base, user) {
            (toml::Value::Table(base_table), toml::Value::Table(user_table)) => {
                // Merge tables recursively
                for (key, user_value) in user_table {
                    if let Some(base_value) = base_table.get_mut(&key) {
                        // Key exists in base, merge recursively
                        Self::merge_toml_values(base_value, user_value);
                    } else {
                        // Key doesn't exist in base, add it
                        base_table.insert(key, user_value);
                    }
                }
            }
            (base_val, user_value) => {
                // For non-table values, user value overwrites base value
                *base_val = user_value;
            }
        }
    }

    fn expand_env_placeholders(
        value: &mut toml::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match value {
            toml::Value::String(s) => {
                *s = Self::expand_env_string(s)?;
            }
            toml::Value::Array(items) => {
                for item in items {
                    Self::expand_env_placeholders(item)?;
                }
            }
            toml::Value::Table(table) => {
                let keys: Vec<String> = table.keys().cloned().collect();
                for key in keys {
                    if let Some(item) = table.get_mut(&key) {
                        Self::expand_env_placeholders(item)?;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn expand_env_string(input: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut output = String::with_capacity(input.len());
        let mut remaining = input;

        while let Some(start) = remaining.find("${") {
            output.push_str(&remaining[..start]);
            let after_start = &remaining[start + 2..];
            let Some(end) = after_start.find('}') else {
                output.push_str(&remaining[start..]);
                return Ok(output);
            };

            let name = &after_start[..end];
            if name.is_empty() {
                output.push_str("${}");
            } else {
                let value = std::env::var(name).map_err(|_| {
                    let message =
                        format!("Environment variable {name} referenced in config but not set");
                    std::io::Error::new(std::io::ErrorKind::NotFound, message)
                })?;
                output.push_str(&value);
            }

            remaining = &after_start[end + 1..];
        }

        output.push_str(remaining);
        Ok(output)
    }

    /// Resolve `path_str` to an absolute path anchored at `base_dir` (rather
    /// than the process cwd). URLs and already-absolute paths pass through
    /// unchanged.
    ///
    /// TD-WAL-4: the WAL write buffer and the SST/VIPER data dirs are rooted
    /// under the instance's `data_dir` with this, so two servers launched from
    /// the same working directory with distinct `data_dir` never silently share
    /// a physical WAL write buffer (which cross-contaminated recovery — each
    /// instance listed and replayed the other's collections).
    pub(crate) fn resolve_path_under(base_dir: &std::path::Path, path_str: &str) -> String {
        use std::path::Path;
        if path_str.contains("://") || Path::new(path_str).is_absolute() {
            return path_str.to_string();
        }
        let mut resolved = base_dir.to_path_buf();
        let mut remaining = path_str;
        while let Some(rest) = remaining.strip_prefix("../") {
            if let Some(parent) = resolved.parent() {
                resolved = parent.to_path_buf();
            }
            remaining = rest;
        }
        let clean = remaining.strip_prefix("./").unwrap_or(remaining);
        if clean.is_empty() || clean == "." {
            return resolved.to_string_lossy().into_owned();
        }
        resolved.join(clean).to_string_lossy().into_owned()
    }

    /// Resolve all relative paths in config to absolute paths
    fn resolve_config_paths(
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use std::path::{Path, PathBuf};

        // Helper function to resolve a path string to absolute
        let resolve_path =
            |path_str: &str| -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                // Skip if it's already a URL or absolute path
                if path_str.contains("://") || Path::new(path_str).is_absolute() {
                    return Ok(path_str.to_string());
                }

                // Get current working directory or fallback
                let base_dir = match std::env::current_dir() {
                    Ok(cwd) => cwd,
                    Err(_) => {
                        // Fallback to PWD or CARGO_MANIFEST_DIR
                        if let Ok(pwd) = std::env::var("PWD") {
                            PathBuf::from(pwd)
                        } else if let Ok(cargo_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                            PathBuf::from(cargo_dir)
                        } else {
                            PathBuf::from("/tmp")
                        }
                    }
                };

                // Resolve relative path
                let mut resolved = base_dir.clone();

                // Handle special relative path patterns
                if path_str.starts_with("../") {
                    let mut current = base_dir.clone();
                    let mut remaining = path_str;

                    while remaining.starts_with("../") {
                        if let Some(parent) = current.parent() {
                            current = parent.to_path_buf();
                            remaining = remaining.strip_prefix("../").unwrap_or(remaining);
                        } else {
                            break;
                        }
                    }
                    resolved = current.join(remaining);
                } else if path_str == ".." {
                    if let Some(parent) = base_dir.parent() {
                        resolved = parent.to_path_buf();
                    }
                } else if path_str.starts_with("./") {
                    let clean_path = path_str.strip_prefix("./").unwrap_or(path_str);
                    resolved = base_dir.join(clean_path);
                } else if path_str == "." {
                    resolved = base_dir;
                } else {
                    resolved = base_dir.join(path_str);
                }

                Ok(resolved.to_string_lossy().into_owned())
            };

        // Helper to convert file path to file:// URL
        let to_file_url =
            |path_str: &str| -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                if path_str.starts_with("file://") || path_str.contains("://") {
                    Ok(path_str.to_string())
                } else {
                    let resolved = resolve_path(path_str)?;
                    Ok(format!("file://{}", resolved))
                }
            };

        // Resolve server data_dir
        let resolved_data_dir = resolve_path(config.server.data_dir.to_string_lossy().as_ref())?;
        config.server.data_dir = PathBuf::from(resolved_data_dir);
        // Resolve storage locations URLs
        for location in &mut config.storage.storage_locations {
            location.url = to_file_url(&location.url)?;
        }

        // Resolve metadata URL
        config.storage.metadata_url = to_file_url(&config.storage.metadata_url)?;

        // TD-WAL-4: anchor the WAL write buffer + SST/VIPER data dirs under the
        // instance's `data_dir` (already resolved to absolute above), NOT the
        // process cwd that `resolve_path` uses. Without this, two servers
        // launched from the same directory with distinct `data_dir` but an unset
        // (cwd-relative default) write_buffer_directory shared one physical WAL,
        // cross-contaminating recovery. A remote/object-store URL or an explicit
        // absolute path passes through unchanged.
        let data_dir = config.server.data_dir.clone();
        config.storage.wal_config.write_buffer_directory =
            Self::resolve_path_under(&data_dir, &config.storage.wal_config.write_buffer_directory);

        // Resolve SST data directory if configured
        if let Some(ref mut sst_config) = config.storage.sst_config {
            sst_config.data_directory =
                Self::resolve_path_under(&data_dir, &sst_config.data_directory);
            if let Some(cache_path) = &sst_config.cache_local_disk_path {
                sst_config.cache_local_disk_path =
                    Some(Self::resolve_path_under(&data_dir, cache_path));
            }
        }

        // Resolve VIPER data directory if configured
        if let Some(ref mut viper_config) = config.storage.viper_config {
            viper_config.data_directory =
                Self::resolve_path_under(&data_dir, &viper_config.data_directory);
        }

        // Resolve TLS certificate paths if configured
        if let Some(ref mut tls_config) = config.tls {
            if let Some(ref cert_file) = tls_config.cert_file {
                tls_config.cert_file = Some(resolve_path(cert_file)?);
            }
            if let Some(ref key_file) = tls_config.key_file {
                tls_config.key_file = Some(resolve_path(key_file)?);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod wal4_tests {
    use super::ConfigLoader;
    use std::path::Path;

    #[test]
    fn data_dirs_anchor_under_data_dir_not_cwd() {
        // TD-WAL-4 repro: two instances with distinct data_dir and the SAME
        // unset (cwd-relative default) write_buffer_directory must resolve to
        // DISTINCT physical paths — otherwise they share one WAL write buffer.
        let a = ConfigLoader::resolve_path_under(Path::new("/srv/pdb"), "./data/write_buffer");
        let b = ConfigLoader::resolve_path_under(Path::new("/srv/pdb2"), "./data/write_buffer");
        assert_eq!(a, "/srv/pdb/data/write_buffer");
        assert_eq!(b, "/srv/pdb2/data/write_buffer");
        assert_ne!(a, b, "distinct data_dir must not share a write buffer");

        // Bare-relative (no leading ./) also anchors under data_dir.
        assert_eq!(
            ConfigLoader::resolve_path_under(Path::new("/srv/pdb"), "sst_data"),
            "/srv/pdb/sst_data"
        );
    }

    #[test]
    fn absolute_and_url_paths_pass_through_unchanged() {
        // An explicit absolute path or object-store URL is an intentional
        // override and is never re-rooted.
        assert_eq!(
            ConfigLoader::resolve_path_under(Path::new("/srv/pdb"), "/mnt/wb"),
            "/mnt/wb"
        );
        assert_eq!(
            ConfigLoader::resolve_path_under(Path::new("/srv/pdb"), "s3://bucket/wb"),
            "s3://bucket/wb"
        );
    }
}
