//! # Format Registry
//!
//! Central registry for all storage formats, providing:
//! - Format registration and discovery
//! - Auto-detection from paths
//! - Format lookup by name or type
//! - Extensibility for custom formats

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::adapters::InternalFormatAdapter;
use super::traits::{
    DefaultFormatDetector, FormatDetector, FormatType, InternalFormat, OpenTableFormat,
    StorageFormat,
};
use crate::storage::traits::UnifiedStorageFormat;

// ============================================================================
// Format Registry
// ============================================================================

/// Central registry for all storage formats
///
/// The registry maintains:
/// - Internal formats (SST, Helix, etc.)
/// - Open table formats (Delta, Iceberg, Hudi, etc.)
/// - Format detectors for auto-detection
///
/// Thread-safe with RwLock for concurrent access.
pub struct FormatRegistry {
    /// Internal format implementations
    internal_formats: RwLock<HashMap<FormatType, Arc<dyn InternalFormat>>>,

    /// Open table format implementations
    open_formats: RwLock<HashMap<FormatType, Arc<dyn OpenTableFormat>>>,

    /// Format detectors (sorted by priority)
    detectors: RwLock<Vec<Box<dyn FormatDetector>>>,

    /// Default internal format
    default_internal: RwLock<Option<FormatType>>,

    /// Default open format
    default_open: RwLock<Option<FormatType>>,
}

impl FormatRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        let registry = Self {
            internal_formats: RwLock::new(HashMap::new()),
            open_formats: RwLock::new(HashMap::new()),
            detectors: RwLock::new(Vec::new()),
            default_internal: RwLock::new(None),
            default_open: RwLock::new(None),
        };

        // Add default detector
        registry.register_detector(Box::new(DefaultFormatDetector));

        registry
    }

    /// Create registry with default formats registered
    pub fn with_defaults() -> Self {
        let registry = Self::new();

        // Set default format types (implementations registered separately)
        *registry.default_internal.write() = Some(FormatType::Sst);
        *registry.default_open.write() = Some(FormatType::DeltaLake);

        registry
    }

    // ========================================================================
    // Registration
    // ========================================================================

    /// Register an internal format
    pub fn register_internal(
        &self,
        format_type: FormatType,
        format: Arc<dyn InternalFormat>,
    ) -> Result<()> {
        let name = format.format_name().to_string();

        // Validate it's actually an internal format
        match format_type {
            FormatType::Sst
            | FormatType::Helix
            | FormatType::Nova
            | FormatType::Swift
            | FormatType::Raptor
            | FormatType::Orion => {}
            _ => {
                return Err(anyhow!(
                    "Format type {:?} is not an internal format",
                    format_type
                ));
            }
        }

        let mut formats = self.internal_formats.write();
        if formats.contains_key(&format_type) {
            warn!(
                "Overwriting existing internal format {:?} with {}",
                format_type, name
            );
        }
        formats.insert(format_type, format);
        info!("Registered internal format: {} ({:?})", name, format_type);

        Ok(())
    }

    /// Register an open table format
    pub fn register_open(
        &self,
        format_type: FormatType,
        format: Arc<dyn OpenTableFormat>,
    ) -> Result<()> {
        let name = format.format_name().to_string();

        // Validate it's actually an open format
        match format_type {
            FormatType::DeltaLake
            | FormatType::Iceberg
            | FormatType::Hudi
            | FormatType::LanceDb
            | FormatType::DuckDb
            | FormatType::Parquet
            | FormatType::Avro => {}
            _ => {
                return Err(anyhow!(
                    "Format type {:?} is not an open table format",
                    format_type
                ));
            }
        }

        let mut formats = self.open_formats.write();
        if formats.contains_key(&format_type) {
            warn!(
                "Overwriting existing open format {:?} with {}",
                format_type, name
            );
        }
        formats.insert(format_type, format);
        info!("Registered open table format: {} ({:?})", name, format_type);

        Ok(())
    }

    /// Register a format detector
    pub fn register_detector(&self, detector: Box<dyn FormatDetector>) {
        let mut detectors = self.detectors.write();
        let priority = detector.priority();
        detectors.push(detector);
        // Sort by priority (higher first)
        detectors.sort_by_key(|d| std::cmp::Reverse(d.priority()));
        debug!("Registered format detector with priority {}", priority);
    }

    // ========================================================================
    // Lookup
    // ========================================================================

    /// Get an internal format by type
    pub fn get_internal(&self, format_type: FormatType) -> Option<Arc<dyn InternalFormat>> {
        self.internal_formats.read().get(&format_type).cloned()
    }

    /// Get an open table format by type
    pub fn get_open(&self, format_type: FormatType) -> Option<Arc<dyn OpenTableFormat>> {
        self.open_formats.read().get(&format_type).cloned()
    }

    /// Get any format as StorageFormat trait object
    pub fn get_format(&self, format_type: FormatType) -> Option<Arc<dyn StorageFormat>> {
        // Try internal first
        if let Some(f) = self.get_internal(format_type) {
            return Some(f as Arc<dyn StorageFormat>);
        }
        // Then open
        if let Some(f) = self.get_open(format_type) {
            return Some(f as Arc<dyn StorageFormat>);
        }
        None
    }

    /// Get default internal format
    pub fn default_internal(&self) -> Option<Arc<dyn InternalFormat>> {
        let default_type = (*self.default_internal.read())?;
        self.get_internal(default_type)
    }

    /// Get default open format
    pub fn default_open(&self) -> Option<Arc<dyn OpenTableFormat>> {
        let default_type = (*self.default_open.read())?;
        self.get_open(default_type)
    }

    /// Set default internal format
    pub fn set_default_internal(&self, format_type: FormatType) {
        *self.default_internal.write() = Some(format_type);
    }

    /// Set default open format
    pub fn set_default_open(&self, format_type: FormatType) {
        *self.default_open.write() = Some(format_type);
    }

    // ========================================================================
    // Detection
    // ========================================================================

    /// Auto-detect format from path
    #[expect(clippy::await_holding_lock)] // FIXME: Requires API redesign to iterate without holding lock
    pub async fn detect_format(&self, path: &str) -> Result<Option<FormatType>> {
        let detectors = self.detectors.read();

        for detector in detectors.iter() {
            if let Some(format_type) = detector.detect(path).await? {
                debug!("Detected format {:?} for path: {}", format_type, path);
                return Ok(Some(format_type));
            }
        }

        debug!("Could not detect format for path: {}", path);
        Ok(None)
    }

    /// Detect and get internal format
    pub async fn detect_internal(&self, path: &str) -> Result<Option<Arc<dyn InternalFormat>>> {
        if let Some(format_type) = self.detect_format(path).await? {
            return Ok(self.get_internal(format_type));
        }
        Ok(None)
    }

    /// Detect and get open format
    pub async fn detect_open(&self, path: &str) -> Result<Option<Arc<dyn OpenTableFormat>>> {
        if let Some(format_type) = self.detect_format(path).await? {
            return Ok(self.get_open(format_type));
        }
        Ok(None)
    }

    // ========================================================================
    // Enumeration
    // ========================================================================

    /// List all registered internal formats
    pub fn list_internal_formats(&self) -> Vec<FormatType> {
        self.internal_formats.read().keys().cloned().collect()
    }

    /// List all registered open formats
    pub fn list_open_formats(&self) -> Vec<FormatType> {
        self.open_formats.read().keys().cloned().collect()
    }

    /// List all registered formats
    pub fn list_all_formats(&self) -> Vec<FormatType> {
        let mut formats = self.list_internal_formats();
        formats.extend(self.list_open_formats());
        formats
    }

    /// Check if a format type is registered
    pub fn is_registered(&self, format_type: FormatType) -> bool {
        self.internal_formats.read().contains_key(&format_type)
            || self.open_formats.read().contains_key(&format_type)
    }

    /// Check if format is internal type
    pub fn is_internal_format(format_type: FormatType) -> bool {
        matches!(
            format_type,
            FormatType::Sst
                | FormatType::Helix
                | FormatType::Nova
                | FormatType::Swift
                | FormatType::Raptor
                | FormatType::Orion
        )
    }

    /// Check if format is open table format
    pub fn is_open_format(format_type: FormatType) -> bool {
        matches!(
            format_type,
            FormatType::DeltaLake
                | FormatType::Iceberg
                | FormatType::Hudi
                | FormatType::LanceDb
                | FormatType::DuckDb
                | FormatType::Parquet
                | FormatType::Avro
        )
    }

    // ========================================================================
    // Adapter Registration Helpers
    // ========================================================================

    /// Register a storage engine as an internal format using the adapter pattern
    ///
    /// This method wraps the given `UnifiedStorageFormat` in an `InternalFormatAdapter`
    /// and registers it with the appropriate format type based on the engine's strategy.
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let sst_engine = Arc::new(SstEngine::new(config)?);
    /// registry.register_engine_as_format(sst_engine)?;
    /// ```
    pub fn register_engine_as_format<E: UnifiedStorageFormat + 'static>(
        &self,
        engine: Arc<E>,
    ) -> Result<()> {
        let adapter = InternalFormatAdapter::new(engine);
        let format_type = adapter.format_type();
        self.register_internal(format_type, Arc::new(adapter))
    }

    /// Register multiple storage engines as internal formats
    ///
    /// Convenience method for registering several engines at once during initialization.
    pub fn register_engines<E: UnifiedStorageFormat + 'static>(
        &self,
        engines: Vec<Arc<E>>,
    ) -> Result<()> {
        for engine in engines {
            self.register_engine_as_format(engine)?;
        }
        Ok(())
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// Global Registry (Singleton)
// ============================================================================

use once_cell::sync::Lazy;

/// Global format registry instance
static GLOBAL_REGISTRY: Lazy<FormatRegistry> = Lazy::new(FormatRegistry::with_defaults);

/// Get the global format registry
pub fn global_registry() -> &'static FormatRegistry {
    &GLOBAL_REGISTRY
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = FormatRegistry::new();
        assert!(registry.list_internal_formats().is_empty());
        assert!(registry.list_open_formats().is_empty());
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = FormatRegistry::with_defaults();
        // Defaults are set but no implementations registered yet
        assert!(registry.default_internal().is_none());
        assert!(registry.default_open().is_none());
    }

    #[test]
    fn test_is_internal_format() {
        assert!(FormatRegistry::is_internal_format(FormatType::Sst));
        assert!(FormatRegistry::is_internal_format(FormatType::Helix));
        assert!(!FormatRegistry::is_internal_format(FormatType::DeltaLake));
        assert!(!FormatRegistry::is_internal_format(FormatType::Iceberg));
    }

    #[test]
    fn test_is_open_format() {
        assert!(FormatRegistry::is_open_format(FormatType::DeltaLake));
        assert!(FormatRegistry::is_open_format(FormatType::Iceberg));
        assert!(FormatRegistry::is_open_format(FormatType::Parquet));
        assert!(!FormatRegistry::is_open_format(FormatType::Sst));
        assert!(!FormatRegistry::is_open_format(FormatType::Helix));
    }

    #[tokio::test]
    async fn test_default_format_detector() {
        let detector = DefaultFormatDetector;

        // Test Parquet detection
        let result = detector.detect("/path/to/file.parquet").await.unwrap();
        assert_eq!(result, Some(FormatType::Parquet));

        // Test Avro detection
        let result = detector.detect("/path/to/file.avro").await.unwrap();
        assert_eq!(result, Some(FormatType::Avro));

        // Test unknown
        let result = detector.detect("/path/to/file.unknown").await.unwrap();
        assert_eq!(result, None);
    }
}
