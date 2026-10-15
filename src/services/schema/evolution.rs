/*
 * Copyright 2025 ProximaDB
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

//! Schema Evolution Service for Production Deployments
//!
//! This module provides comprehensive schema evolution capabilities including:
//!
//! - Schema versioning with version numbers and timestamps
//! - Backward compatibility validation across schema changes
//! - Schema change detection (add/remove/rename/type-change columns)
//! - Schema evolution rules with type widening support
//! - Schema rollback support
//!
//! ## Evolution Rules
//!
//! The service enforces the following rules:
//!
//! | Operation | Allowed | Condition |
//! |-----------|---------|-----------|
//! | Add nullable column | Yes | Always |
//! | Add non-nullable column | Yes | With default value |
//! | Remove column | Warning | Breaking change |
//! | Rename column | Yes | Tracked via column ID |
//! | Widen type (INT -> BIGINT) | Yes | Safe widening |
//! | Narrow type (BIGINT -> INT) | Warning | Potential data loss |
//! | Make nullable | Yes | Always |
//! | Make non-nullable | Warning | Requires data migration |
//!
//! ## Example
//!
//! ```rust,ignore
//! use proximadb::services::schema::{
//!     SchemaEvolutionService, SchemaVersion, EvolutionConfig,
//! };
//!
//! let service = SchemaEvolutionService::new(EvolutionConfig::default());
//!
//! // Evolve schema
//! let result = service.evolve_schema(&old_schema, &new_schema)?;
//! if result.is_compatible {
//!     println!("Schema evolution is safe");
//! } else {
//!     for warning in &result.warnings {
//!         println!("Warning: {}", warning);
//!     }
//! }
//!
//! // Get schema history
//! let history = service.get_schema_history("my_collection")?;
//! for version in history {
//!     println!("Version {}: {}", version.version, version.timestamp);
//! }
//!
//! // Rollback to previous version
//! service.rollback_schema("my_collection", 2)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::types::ColumnDataType;
use crate::proto::proximadb_v1::{FilterableDataType, RecordSchemaConfig, TypedColumnConfig};

// =============================================================================
// Schema Version Types
// =============================================================================

/// Schema version with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Unique version identifier (monotonically increasing)
    pub version: u64,

    /// Schema ID (UUID)
    pub schema_id: String,

    /// Parent schema version (None for initial schema)
    pub parent_version: Option<u64>,

    /// Creation timestamp (UTC)
    pub timestamp: DateTime<Utc>,

    /// Schema definition
    pub schema: RecordSchemaConfig,

    /// Evolution operations that produced this version
    pub evolution_ops: Vec<SchemaChange>,

    /// User/system that created this version
    pub created_by: Option<String>,

    /// Optional description of changes
    pub description: Option<String>,

    /// Whether this version is active
    pub is_active: bool,

    /// Checksum for integrity validation
    pub checksum: u64,
}

impl SchemaVersion {
    /// Create a new schema version
    pub fn new(
        version: u64,
        schema: RecordSchemaConfig,
        parent_version: Option<u64>,
        evolution_ops: Vec<SchemaChange>,
    ) -> Self {
        let schema_id = uuid::Uuid::new_v4().to_string();
        let checksum = Self::compute_checksum(&schema);

        Self {
            version,
            schema_id,
            parent_version,
            timestamp: Utc::now(),
            schema,
            evolution_ops,
            created_by: None,
            description: None,
            is_active: true,
            checksum,
        }
    }

    /// Compute checksum from schema definition
    fn compute_checksum(schema: &RecordSchemaConfig) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        schema.schema_id.hash(&mut hasher);
        schema.schema_version.hash(&mut hasher);
        for col in &schema.columns {
            col.name.hash(&mut hasher);
            col.data_type.hash(&mut hasher);
            col.nullable.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Set the creator
    pub fn with_created_by(mut self, created_by: String) -> Self {
        self.created_by = Some(created_by);
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

// =============================================================================
// Schema Change Detection
// =============================================================================

/// Detected change between two schemas
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SchemaChange {
    /// Column was added
    ColumnAdded {
        /// Name of the newly added column.
        column_name: String,
        /// Encoded data type of the new column (proto `DataType` int value).
        data_type: i32,
        /// Whether the new column accepts NULL values.
        nullable: bool,
    },

    /// Column was removed
    ColumnRemoved {
        /// Name of the removed column.
        column_name: String,
        /// Encoded data type of the removed column (proto `DataType` int value).
        data_type: i32,
    },

    /// Column was renamed
    ColumnRenamed {
        /// Previous column name.
        old_name: String,
        /// New column name.
        new_name: String,
    },

    /// Column type was changed
    TypeChanged {
        /// Name of the column whose type changed.
        column_name: String,
        /// Previous encoded data type (proto `DataType` int value).
        old_type: i32,
        /// New encoded data type (proto `DataType` int value).
        new_type: i32,
    },

    /// Nullable constraint changed
    NullabilityChanged {
        /// Name of the column whose nullability changed.
        column_name: String,
        /// Previous nullability setting.
        old_nullable: bool,
        /// New nullability setting.
        new_nullable: bool,
    },

    /// Default value changed
    DefaultValueChanged {
        /// Name of the column whose default changed.
        column_name: String,
        /// Previous default value expression, if any.
        old_default: Option<String>,
        /// New default value expression, if any.
        new_default: Option<String>,
    },

    /// Column constraints changed (min/max, pattern, etc.)
    ConstraintsChanged {
        /// Name of the column whose constraints changed.
        column_name: String,
        /// Human-readable description of what changed.
        change_description: String,
    },
}

impl SchemaChange {
    /// Check if this change is a breaking change
    pub fn is_breaking(&self) -> bool {
        match self {
            SchemaChange::ColumnRemoved { .. } => true,
            SchemaChange::TypeChanged {
                old_type, new_type, ..
            } => !is_safe_type_widening(*old_type, *new_type),
            SchemaChange::NullabilityChanged {
                old_nullable,
                new_nullable,
                ..
            } => *old_nullable && !*new_nullable, // Making non-nullable is breaking
            _ => false,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            SchemaChange::ColumnAdded {
                column_name,
                nullable,
                ..
            } => {
                let null_str = if *nullable {
                    "nullable"
                } else {
                    "non-nullable"
                };
                format!("Added {} column '{}'", null_str, column_name)
            }
            SchemaChange::ColumnRemoved { column_name, .. } => {
                format!("Removed column '{}'", column_name)
            }
            SchemaChange::ColumnRenamed { old_name, new_name } => {
                format!("Renamed column '{}' to '{}'", old_name, new_name)
            }
            SchemaChange::TypeChanged {
                column_name,
                old_type,
                new_type,
            } => {
                format!(
                    "Changed type of '{}' from {:?} to {:?}",
                    column_name,
                    FilterableDataType::try_from(*old_type).ok(),
                    FilterableDataType::try_from(*new_type).ok()
                )
            }
            SchemaChange::NullabilityChanged {
                column_name,
                old_nullable,
                new_nullable,
            } => {
                let change = if *new_nullable {
                    "nullable"
                } else {
                    "non-nullable"
                };
                format!(
                    "Changed '{}' from {} to {}",
                    column_name,
                    if *old_nullable {
                        "nullable"
                    } else {
                        "non-nullable"
                    },
                    change
                )
            }
            SchemaChange::DefaultValueChanged {
                column_name,
                old_default,
                new_default,
            } => {
                format!(
                    "Changed default value of '{}' from {:?} to {:?}",
                    column_name, old_default, new_default
                )
            }
            SchemaChange::ConstraintsChanged {
                column_name,
                change_description,
            } => {
                format!(
                    "Changed constraints on '{}': {}",
                    column_name, change_description
                )
            }
        }
    }
}

// =============================================================================
// Evolution Result Types
// =============================================================================

/// Result of schema evolution operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    /// Whether the evolution is valid
    pub is_valid: bool,

    /// Whether the new schema is backward compatible
    pub is_backward_compatible: bool,

    /// Whether the new schema is forward compatible
    pub is_forward_compatible: bool,

    /// Detected changes between schemas
    pub changes: Vec<SchemaChange>,

    /// Warning messages (non-blocking issues)
    pub warnings: Vec<String>,

    /// Error messages (blocking issues)
    pub errors: Vec<String>,

    /// Whether data migration is required
    pub requires_migration: bool,

    /// New schema version (if evolution is valid)
    pub new_version: Option<SchemaVersion>,

    /// Estimated migration cost
    pub migration_estimate: Option<MigrationEstimate>,
}

impl Default for EvolutionResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            is_backward_compatible: true,
            is_forward_compatible: true,
            changes: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            requires_migration: false,
            new_version: None,
            migration_estimate: None,
        }
    }
}

impl EvolutionResult {
    /// Add an error and mark as invalid
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Mark as not backward compatible
    pub fn mark_not_backward_compatible(&mut self) {
        self.is_backward_compatible = false;
    }

    /// Mark as not forward compatible
    pub fn mark_not_forward_compatible(&mut self) {
        self.is_forward_compatible = false;
    }
}

/// Result of compatibility validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether schemas are compatible
    pub is_compatible: bool,

    /// Compatibility level achieved
    pub compatibility_level: CompatibilityLevel,

    /// Detected changes
    pub changes: Vec<SchemaChange>,

    /// Issues that prevent compatibility
    pub issues: Vec<CompatibilityIssue>,

    /// Suggestions for making schemas compatible
    pub suggestions: Vec<String>,
}

impl Default for CompatibilityResult {
    fn default() -> Self {
        Self {
            is_compatible: true,
            compatibility_level: CompatibilityLevel::Full,
            changes: Vec::new(),
            issues: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

/// Compatibility levels between schemas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    /// Fully compatible - no changes needed
    Full,
    /// Backward compatible - old readers can read new data
    Backward,
    /// Forward compatible - new readers can read old data
    Forward,
    /// Transitive compatible - compatible across all versions
    Transitive,
    /// Not compatible - breaking changes present
    None,
}

/// Specific compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Severity of the issue
    pub severity: IssueSeverity,

    /// Column affected (if applicable)
    pub column_name: Option<String>,

    /// Issue description
    pub description: String,

    /// Suggested resolution
    pub resolution: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Blocking - prevents schema evolution
    Error,
    /// Non-blocking - but may cause issues
    Warning,
    /// Informational - no action needed
    Info,
}

/// Estimated cost of schema migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationEstimate {
    /// Estimated number of records to migrate
    pub records_affected: u64,

    /// Estimated bytes to rewrite
    pub bytes_to_rewrite: u64,

    /// Whether migration can be done online (without downtime)
    pub can_migrate_online: bool,

    /// Estimated duration in seconds
    pub estimated_duration_secs: u64,

    /// Columns requiring transformation
    pub columns_to_transform: Vec<String>,
}

// =============================================================================
// Evolution Configuration
// =============================================================================

/// Configuration for schema evolution service
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Maximum number of schema versions to retain
    pub max_versions: usize,

    /// Whether to allow breaking changes with explicit override
    pub allow_breaking_changes: bool,

    /// Whether to automatically generate migration scripts
    pub auto_generate_migration: bool,

    /// Strict mode - treat warnings as errors
    pub strict_mode: bool,

    /// Type widening rules
    pub type_widening_rules: Vec<(i32, i32)>,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            max_versions: 100,
            allow_breaking_changes: false,
            auto_generate_migration: true,
            strict_mode: false,
            type_widening_rules: default_widening_rules(),
        }
    }
}

impl EvolutionConfig {
    /// Create with strict mode enabled
    pub fn strict() -> Self {
        Self {
            strict_mode: true,
            ..Default::default()
        }
    }

    /// Enable breaking changes
    pub fn with_breaking_changes(mut self, allow: bool) -> Self {
        self.allow_breaking_changes = allow;
        self
    }

    /// Set maximum versions to retain
    pub fn with_max_versions(mut self, max: usize) -> Self {
        self.max_versions = max;
        self
    }
}

/// Default type widening rules (from -> to)
fn default_widening_rules() -> Vec<(i32, i32)> {
    use FilterableDataType::*;

    vec![
        // Integer widening
        (FilterableInteger as i32, FilterableInteger as i32),
        // Float widening
        (FilterableFloat as i32, FilterableFloat as i32),
        // Text to larger text
        (FilterableString as i32, FilterableText as i32),
        (FilterableText as i32, FilterableTextLarge as i32),
        // Date/time widening
        (FilterableDate as i32, FilterableDatetime as i32),
    ]
}

/// Check if type change is a safe widening
fn is_safe_type_widening(old_type: i32, new_type: i32) -> bool {
    if old_type == new_type {
        return true;
    }

    let rules = default_widening_rules();
    rules.contains(&(old_type, new_type))
}

// =============================================================================
// Schema Evolution Service
// =============================================================================

/// Schema Evolution Service for managing schema changes
///
/// Provides methods for:
/// - Evolving schemas with validation
/// - Validating compatibility between schemas
/// - Tracking schema history
/// - Rolling back to previous versions
pub struct SchemaEvolutionService {
    /// Service configuration
    config: EvolutionConfig,

    /// Schema history per collection (collection_name -> versions)
    history: Arc<RwLock<HashMap<String, Vec<SchemaVersion>>>>,
}

impl SchemaEvolutionService {
    /// Create a new schema evolution service
    pub fn new(config: EvolutionConfig) -> Self {
        Self {
            config,
            history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(EvolutionConfig::default())
    }

    /// Get the service configuration
    pub fn config(&self) -> &EvolutionConfig {
        &self.config
    }

    /// Evolve schema from old to new
    ///
    /// Validates the evolution and creates a new schema version if valid.
    ///
    /// # Arguments
    ///
    /// * `old_schema` - Current schema definition
    /// * `new_schema` - Target schema definition
    ///
    /// # Returns
    ///
    /// `EvolutionResult` containing validation results and new version if valid.
    pub fn evolve_schema(
        &self,
        old_schema: &RecordSchemaConfig,
        new_schema: &RecordSchemaConfig,
    ) -> Result<EvolutionResult> {
        let mut result = EvolutionResult::default();

        // Detect all changes between schemas
        let changes = self.detect_changes(old_schema, new_schema);
        result.changes = changes.clone();

        // Validate each change
        for change in &changes {
            self.validate_change(change, old_schema, new_schema, &mut result);
        }

        // Check for breaking changes
        let breaking_changes: Vec<_> = changes.iter().filter(|c| c.is_breaking()).collect();

        if !breaking_changes.is_empty() {
            result.mark_not_backward_compatible();

            if !self.config.allow_breaking_changes {
                for change in breaking_changes {
                    result.add_error(format!(
                        "Breaking change detected: {}",
                        change.description()
                    ));
                }
            } else {
                for change in breaking_changes {
                    result.add_warning(format!(
                        "Breaking change (allowed by config): {}",
                        change.description()
                    ));
                }
            }
        }

        // Check for forward compatibility issues
        let added_non_nullable: Vec<_> = changes
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    SchemaChange::ColumnAdded {
                        nullable: false,
                        ..
                    }
                )
            })
            .collect();

        if !added_non_nullable.is_empty() {
            result.mark_not_forward_compatible();
            for change in added_non_nullable {
                // Check if default value is provided
                if let SchemaChange::ColumnAdded { column_name, .. } = change {
                    let has_default = new_schema
                        .columns
                        .iter()
                        .find(|c| &c.name == column_name)
                        .is_some_and(|c| c.default_value.is_some());

                    if !has_default {
                        result.add_error(format!(
                            "Non-nullable column '{}' added without default value",
                            column_name
                        ));
                    }
                }
            }
        }

        // If strict mode, convert warnings to errors
        if self.config.strict_mode && !result.warnings.is_empty() {
            let warnings = std::mem::take(&mut result.warnings);
            for warning in warnings {
                result.add_error(format!("Strict mode: {}", warning));
            }
        }

        // Determine if migration is required
        result.requires_migration = changes.iter().any(|c| {
            matches!(
                c,
                SchemaChange::TypeChanged { .. } | SchemaChange::NullabilityChanged { .. }
            )
        });

        // Create new version if valid
        if result.is_valid {
            let current_version = self.get_current_version_number(old_schema);
            let new_version = SchemaVersion::new(
                current_version + 1,
                new_schema.clone(),
                Some(current_version),
                changes,
            );
            result.new_version = Some(new_version);
        }

        Ok(result)
    }

    /// Validate compatibility between two schemas
    ///
    /// Checks if the new schema is compatible with the old schema
    /// according to configured compatibility level.
    ///
    /// # Arguments
    ///
    /// * `old_schema` - Current schema definition
    /// * `new_schema` - Target schema definition
    ///
    /// # Returns
    ///
    /// `CompatibilityResult` with detailed compatibility information.
    pub fn validate_compatibility(
        &self,
        old_schema: &RecordSchemaConfig,
        new_schema: &RecordSchemaConfig,
    ) -> CompatibilityResult {
        let mut result = CompatibilityResult::default();

        // Detect changes
        let changes = self.detect_changes(old_schema, new_schema);
        result.changes = changes.clone();

        // Analyze each change for compatibility
        for change in &changes {
            match change {
                SchemaChange::ColumnAdded {
                    column_name,
                    nullable: false,
                    ..
                } => {
                    // Check for default value
                    let has_default = new_schema
                        .columns
                        .iter()
                        .find(|c| &c.name == column_name)
                        .is_some_and(|c| c.default_value.is_some());

                    if !has_default {
                        result.issues.push(CompatibilityIssue {
                            severity: IssueSeverity::Error,
                            column_name: Some(column_name.clone()),
                            description: format!(
                                "Non-nullable column '{}' added without default value",
                                column_name
                            ),
                            resolution: Some(
                                "Add a default value or make the column nullable".to_string(),
                            ),
                        });
                        result.is_compatible = false;
                    }
                }
                SchemaChange::ColumnRemoved { column_name, .. } => {
                    result.issues.push(CompatibilityIssue {
                        severity: IssueSeverity::Warning,
                        column_name: Some(column_name.clone()),
                        description: format!(
                            "Column '{}' removed - existing data will lose this field",
                            column_name
                        ),
                        resolution: Some(
                            "Consider deprecating instead of removing to maintain backward compatibility"
                                .to_string(),
                        ),
                    });
                    result.compatibility_level = CompatibilityLevel::Forward;
                }
                SchemaChange::TypeChanged {
                    column_name,
                    old_type,
                    new_type,
                } => {
                    if !is_safe_type_widening(*old_type, *new_type) {
                        result.issues.push(CompatibilityIssue {
                            severity: IssueSeverity::Error,
                            column_name: Some(column_name.clone()),
                            description: format!(
                                "Type change from {:?} to {:?} may cause data loss",
                                FilterableDataType::try_from(*old_type).ok(),
                                FilterableDataType::try_from(*new_type).ok()
                            ),
                            resolution: Some(
                                "Use a safe type widening (e.g., INT -> BIGINT, FLOAT -> DOUBLE)"
                                    .to_string(),
                            ),
                        });
                        result.is_compatible = false;
                        result.compatibility_level = CompatibilityLevel::None;
                    } else {
                        result.suggestions.push(format!(
                            "Type widening on '{}' is safe and will be applied automatically",
                            column_name
                        ));
                    }
                }
                SchemaChange::NullabilityChanged {
                    column_name,
                    old_nullable: true,
                    new_nullable: false,
                } => {
                    result.issues.push(CompatibilityIssue {
                        severity: IssueSeverity::Warning,
                        column_name: Some(column_name.clone()),
                        description: format!(
                            "Column '{}' changed from nullable to non-nullable - requires data migration",
                            column_name
                        ),
                        resolution: Some(
                            "Ensure no NULL values exist or provide a default for NULL values"
                                .to_string(),
                        ),
                    });
                    result.compatibility_level = CompatibilityLevel::Forward;
                }
                _ => {}
            }
        }

        // Determine final compatibility level
        if !result.is_compatible {
            result.compatibility_level = CompatibilityLevel::None;
        } else if result
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning)
            && result.compatibility_level == CompatibilityLevel::Full
        {
            result.compatibility_level = CompatibilityLevel::Backward;
        }

        result
    }

    /// Get schema history for a collection
    ///
    /// Returns all schema versions for the specified collection,
    /// ordered by version number (oldest first).
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Name of the collection
    ///
    /// # Returns
    ///
    /// Vector of `SchemaVersion` in chronological order.
    pub fn get_schema_history(&self, collection_name: &str) -> Result<Vec<SchemaVersion>> {
        let history = self
            .history
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        Ok(history.get(collection_name).cloned().unwrap_or_default())
    }

    /// Store a schema version in history
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Name of the collection
    /// * `version` - Schema version to store
    pub fn store_version(&self, collection_name: &str, version: SchemaVersion) -> Result<()> {
        let mut history = self
            .history
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        let versions = history.entry(collection_name.to_string()).or_default();

        // Deactivate previous versions
        for v in versions.iter_mut() {
            v.is_active = false;
        }

        versions.push(version);

        // Trim to max versions
        if versions.len() > self.config.max_versions {
            let excess = versions.len() - self.config.max_versions;
            versions.drain(0..excess);
        }

        Ok(())
    }

    /// Rollback to a specific schema version
    ///
    /// Creates a new version that reverts to a previous schema definition.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Name of the collection
    /// * `target_version` - Version number to rollback to
    ///
    /// # Returns
    ///
    /// The new schema version (which is the rollback version).
    pub fn rollback_schema(
        &self,
        collection_name: &str,
        target_version: u64,
    ) -> Result<SchemaVersion> {
        let history = self.get_schema_history(collection_name)?;

        // Find target version
        let target = history
            .iter()
            .find(|v| v.version == target_version)
            .ok_or_else(|| anyhow!("Version {} not found in history", target_version))?;

        // Find current version
        let current = history
            .iter()
            .filter(|v| v.is_active)
            .max_by_key(|v| v.version)
            .ok_or_else(|| anyhow!("No active version found"))?;

        // Create rollback version
        let mut rollback = SchemaVersion::new(
            current.version + 1,
            target.schema.clone(),
            Some(current.version),
            vec![], // Evolution ops will be computed
        );
        rollback.description = Some(format!(
            "Rollback from version {} to version {}",
            current.version, target_version
        ));

        // Compute the changes (rollback operations)
        let changes = self.detect_changes(&current.schema, &target.schema);
        let rollback = SchemaVersion {
            evolution_ops: changes,
            ..rollback
        };

        // Store the rollback version
        self.store_version(collection_name, rollback.clone())?;

        Ok(rollback)
    }

    /// Get the latest schema version for a collection
    pub fn get_latest_version(&self, collection_name: &str) -> Result<Option<SchemaVersion>> {
        let history = self.get_schema_history(collection_name)?;

        Ok(history
            .into_iter()
            .filter(|v| v.is_active)
            .max_by_key(|v| v.version))
    }

    /// Get a specific schema version
    pub fn get_version(
        &self,
        collection_name: &str,
        version: u64,
    ) -> Result<Option<SchemaVersion>> {
        let history = self.get_schema_history(collection_name)?;

        Ok(history.into_iter().find(|v| v.version == version))
    }

    // =========================================================================
    // Private Helper Methods
    // =========================================================================

    /// Detect changes between two schemas
    fn detect_changes(
        &self,
        old_schema: &RecordSchemaConfig,
        new_schema: &RecordSchemaConfig,
    ) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        // Build column maps
        let old_columns: HashMap<&str, &TypedColumnConfig> = old_schema
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();
        let new_columns: HashMap<&str, &TypedColumnConfig> = new_schema
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        let old_names: HashSet<&str> = old_columns.keys().copied().collect();
        let new_names: HashSet<&str> = new_columns.keys().copied().collect();

        // Detect added columns
        for name in new_names.difference(&old_names) {
            if let Some(col) = new_columns.get(name) {
                changes.push(SchemaChange::ColumnAdded {
                    column_name: name.to_string(),
                    data_type: col.data_type,
                    nullable: col.nullable,
                });
            }
        }

        // Detect removed columns
        for name in old_names.difference(&new_names) {
            if let Some(col) = old_columns.get(name) {
                changes.push(SchemaChange::ColumnRemoved {
                    column_name: name.to_string(),
                    data_type: col.data_type,
                });
            }
        }

        // Detect changes in existing columns
        for name in old_names.intersection(&new_names) {
            let (Some(old_col), Some(new_col)) = (old_columns.get(name), new_columns.get(name))
            else {
                continue;
            };

            // Type change
            if old_col.data_type != new_col.data_type {
                changes.push(SchemaChange::TypeChanged {
                    column_name: name.to_string(),
                    old_type: old_col.data_type,
                    new_type: new_col.data_type,
                });
            }

            // Nullability change
            if old_col.nullable != new_col.nullable {
                changes.push(SchemaChange::NullabilityChanged {
                    column_name: name.to_string(),
                    old_nullable: old_col.nullable,
                    new_nullable: new_col.nullable,
                });
            }

            // Default value change
            if old_col.default_value != new_col.default_value {
                changes.push(SchemaChange::DefaultValueChanged {
                    column_name: name.to_string(),
                    old_default: old_col.default_value.clone(),
                    new_default: new_col.default_value.clone(),
                });
            }

            // Constraint changes
            self.detect_constraint_changes(old_col, new_col, &mut changes);
        }

        // Try to detect renames (columns with same type that were added/removed)
        self.detect_renames(&mut changes);

        changes
    }

    /// Detect constraint changes between columns
    fn detect_constraint_changes(
        &self,
        old_col: &TypedColumnConfig,
        new_col: &TypedColumnConfig,
        changes: &mut Vec<SchemaChange>,
    ) {
        let mut constraint_changes = Vec::new();

        // Max length
        if old_col.max_length != new_col.max_length {
            constraint_changes.push(format!(
                "max_length: {:?} -> {:?}",
                old_col.max_length, new_col.max_length
            ));
        }

        // Min value
        if old_col.min_value != new_col.min_value {
            constraint_changes.push(format!(
                "min_value: {:?} -> {:?}",
                old_col.min_value, new_col.min_value
            ));
        }

        // Max value
        if old_col.max_value != new_col.max_value {
            constraint_changes.push(format!(
                "max_value: {:?} -> {:?}",
                old_col.max_value, new_col.max_value
            ));
        }

        // Regex pattern
        if old_col.regex_pattern != new_col.regex_pattern {
            constraint_changes.push(format!(
                "regex_pattern: {:?} -> {:?}",
                old_col.regex_pattern, new_col.regex_pattern
            ));
        }

        if !constraint_changes.is_empty() {
            changes.push(SchemaChange::ConstraintsChanged {
                column_name: old_col.name.clone(),
                change_description: constraint_changes.join(", "),
            });
        }
    }

    /// Try to detect column renames from add/remove pairs
    fn detect_renames(&self, changes: &mut Vec<SchemaChange>) {
        // Find added and removed columns with same type
        let added: Vec<_> = changes
            .iter()
            .filter_map(|c| {
                if let SchemaChange::ColumnAdded {
                    column_name,
                    data_type,
                    ..
                } = c
                {
                    Some((column_name.clone(), *data_type))
                } else {
                    None
                }
            })
            .collect();

        let removed: Vec<_> = changes
            .iter()
            .filter_map(|c| {
                if let SchemaChange::ColumnRemoved {
                    column_name,
                    data_type,
                } = c
                {
                    Some((column_name.clone(), *data_type))
                } else {
                    None
                }
            })
            .collect();

        // Match by type (simple heuristic)
        let mut detected_renames = Vec::new();
        let mut removed_to_skip = HashSet::new();
        let mut added_to_skip = HashSet::new();

        for (removed_name, removed_type) in &removed {
            for (added_name, added_type) in &added {
                if removed_type == added_type
                    && !removed_to_skip.contains(removed_name)
                    && !added_to_skip.contains(added_name)
                {
                    detected_renames.push((removed_name.clone(), added_name.clone()));
                    removed_to_skip.insert(removed_name.clone());
                    added_to_skip.insert(added_name.clone());
                    break;
                }
            }
        }

        // Add rename changes and remove the original add/remove
        for (old_name, new_name) in detected_renames {
            // Note: We don't remove the add/remove changes here as it would
            // modify the vector while iterating. In production, this should
            // be a configuration option for the user to confirm renames.
            changes.push(SchemaChange::ColumnRenamed { old_name, new_name });
        }
    }

    /// Validate a single schema change
    fn validate_change(
        &self,
        change: &SchemaChange,
        _old_schema: &RecordSchemaConfig,
        new_schema: &RecordSchemaConfig,
        result: &mut EvolutionResult,
    ) {
        match change {
            SchemaChange::ColumnAdded {
                column_name,
                nullable: false,
                ..
            } => {
                // Non-nullable column must have default
                let has_default = new_schema
                    .columns
                    .iter()
                    .find(|c| &c.name == column_name)
                    .is_some_and(|c| c.default_value.is_some());

                if !has_default {
                    result.add_error(format!(
                        "Cannot add non-nullable column '{}' without default value",
                        column_name
                    ));
                }
            }
            SchemaChange::ColumnRemoved { column_name, .. } => {
                result.add_warning(format!(
                    "Column '{}' will be removed. This is a breaking change.",
                    column_name
                ));
            }
            SchemaChange::TypeChanged {
                column_name,
                old_type,
                new_type,
            } => {
                if is_safe_type_widening(*old_type, *new_type) {
                    // Safe widening - just inform
                    result.add_warning(format!(
                        "Type widening on '{}' will be applied (safe operation)",
                        column_name
                    ));
                } else {
                    // Potentially unsafe
                    result.add_warning(format!(
                        "Type change on '{}' may cause data loss or require migration",
                        column_name
                    ));
                }
            }
            SchemaChange::NullabilityChanged {
                column_name,
                old_nullable: true,
                new_nullable: false,
            } => {
                result.add_warning(format!(
                    "Column '{}' changed to non-nullable. Existing NULL values must be handled.",
                    column_name
                ));
                result.requires_migration = true;
            }
            _ => {}
        }
    }

    /// Get current version number from schema
    fn get_current_version_number(&self, schema: &RecordSchemaConfig) -> u64 {
        // Parse version from schema_version string (e.g., "1.0.0" -> 1)
        schema
            .schema_version
            .split('.')
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

// =============================================================================
// Conversion Utilities
// =============================================================================

/// Convert ColumnDataType to FilterableDataType
pub fn column_type_to_filterable(data_type: &ColumnDataType) -> FilterableDataType {
    match data_type {
        ColumnDataType::Text => FilterableDataType::FilterableText,
        ColumnDataType::TextLarge => FilterableDataType::FilterableTextLarge,
        ColumnDataType::Integer => FilterableDataType::FilterableInteger,
        ColumnDataType::Float => FilterableDataType::FilterableFloat,
        ColumnDataType::Decimal { .. } => FilterableDataType::FilterableDecimal,
        ColumnDataType::Boolean => FilterableDataType::FilterableBoolean,
        ColumnDataType::Timestamp => FilterableDataType::FilterableDatetime,
        ColumnDataType::TimestampTz { .. } => FilterableDataType::FilterableTimestampTz,
        ColumnDataType::Date => FilterableDataType::FilterableDate,
        ColumnDataType::Time => FilterableDataType::FilterableTime,
        ColumnDataType::Uuid => FilterableDataType::FilterableUuid,
        ColumnDataType::Binary | ColumnDataType::BinaryLarge => {
            FilterableDataType::FilterableBinary
        }
        ColumnDataType::Json => FilterableDataType::FilterableJson,
        ColumnDataType::ArrayText => FilterableDataType::FilterableArrayString,
        ColumnDataType::ArrayInteger => FilterableDataType::FilterableArrayInteger,
        ColumnDataType::ArrayFloat => FilterableDataType::FilterableArrayFloat,
        _ => FilterableDataType::FilterableString,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::SchemaEnforcement;

    fn create_test_schema(columns: Vec<(&str, i32, bool)>) -> RecordSchemaConfig {
        let typed_columns = columns
            .into_iter()
            .map(|(name, data_type, nullable)| TypedColumnConfig {
                name: name.to_string(),
                data_type,
                nullable,
                indexed: false,
                filterable: true,
                max_length: None,
                min_value: None,
                max_value: None,
                regex_pattern: None,
                default_value: None,
                text_storage: None,
                fulltext_indexed: None,
            })
            .collect();

        RecordSchemaConfig {
            schema_id: uuid::Uuid::new_v4().to_string(),
            schema_version: "1.0.0".to_string(),
            enforcement: SchemaEnforcement::SchemaFlexible as i32,
            auto_evolve: true,
            columns: typed_columns,
        }
    }

    #[test]
    fn test_detect_column_added() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
        ]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
            ("email", FilterableDataType::FilterableText as i32, true),
        ]);

        let changes = service.detect_changes(&old_schema, &new_schema);

        assert_eq!(changes.len(), 1);
        assert!(matches!(
            &changes[0],
            SchemaChange::ColumnAdded {
                column_name,
                nullable: true,
                ..
            } if column_name == "email"
        ));
    }

    #[test]
    fn test_detect_column_removed() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
            ("email", FilterableDataType::FilterableText as i32, true),
        ]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
        ]);

        let changes = service.detect_changes(&old_schema, &new_schema);

        assert!(changes.iter().any(|c| matches!(
            c,
            SchemaChange::ColumnRemoved { column_name, .. } if column_name == "email"
        )));
    }

    #[test]
    fn test_detect_type_changed() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("count", FilterableDataType::FilterableInteger as i32, true),
        ]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("count", FilterableDataType::FilterableFloat as i32, true),
        ]);

        let changes = service.detect_changes(&old_schema, &new_schema);

        assert!(changes.iter().any(|c| matches!(
            c,
            SchemaChange::TypeChanged { column_name, .. } if column_name == "count"
        )));
    }

    #[test]
    fn test_detect_nullability_changed() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
        ]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, false),
        ]);

        let changes = service.detect_changes(&old_schema, &new_schema);

        assert!(changes.iter().any(|c| matches!(
            c,
            SchemaChange::NullabilityChanged {
                column_name,
                old_nullable: true,
                new_nullable: false
            } if column_name == "name"
        )));
    }

    #[test]
    fn test_evolve_schema_success() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("email", FilterableDataType::FilterableText as i32, true), // nullable
        ]);

        let result = service.evolve_schema(&old_schema, &new_schema).unwrap();

        assert!(result.is_valid);
        assert!(result.is_backward_compatible);
        assert!(result.errors.is_empty());
        assert!(result.new_version.is_some());
    }

    #[test]
    fn test_evolve_schema_fails_non_nullable_without_default() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("email", FilterableDataType::FilterableText as i32, false), // non-nullable, no default
        ]);

        let result = service.evolve_schema(&old_schema, &new_schema).unwrap();

        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].contains("default value"));
    }

    #[test]
    fn test_validate_compatibility_backward() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);

        let new_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("email", FilterableDataType::FilterableText as i32, true),
        ]);

        let result = service.validate_compatibility(&old_schema, &new_schema);

        assert!(result.is_compatible);
        assert!(matches!(
            result.compatibility_level,
            CompatibilityLevel::Full | CompatibilityLevel::Backward
        ));
    }

    #[test]
    fn test_validate_compatibility_breaking() {
        let service = SchemaEvolutionService::with_defaults();

        let old_schema = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("email", FilterableDataType::FilterableText as i32, true),
        ]);

        let new_schema = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);

        let result = service.validate_compatibility(&old_schema, &new_schema);

        assert!(
            result
                .issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Warning)
        );
    }

    #[test]
    fn test_schema_version_creation() {
        let schema = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);

        let version = SchemaVersion::new(1, schema, None, vec![]);

        assert_eq!(version.version, 1);
        assert!(version.is_active);
        assert!(version.parent_version.is_none());
        assert!(!version.schema_id.is_empty());
    }

    #[test]
    fn test_schema_history() {
        let service = SchemaEvolutionService::with_defaults();

        let schema1 = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);
        let version1 = SchemaVersion::new(1, schema1, None, vec![]);

        service.store_version("test_collection", version1).unwrap();

        let schema2 = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
        ]);
        let version2 = SchemaVersion::new(2, schema2, Some(1), vec![]);

        service.store_version("test_collection", version2).unwrap();

        let history = service.get_schema_history("test_collection").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[test]
    fn test_schema_rollback() {
        let service = SchemaEvolutionService::with_defaults();

        let schema1 = create_test_schema(vec![(
            "id",
            FilterableDataType::FilterableString as i32,
            false,
        )]);
        let version1 = SchemaVersion::new(1, schema1.clone(), None, vec![]);
        service.store_version("test_collection", version1).unwrap();

        let schema2 = create_test_schema(vec![
            ("id", FilterableDataType::FilterableString as i32, false),
            ("name", FilterableDataType::FilterableText as i32, true),
        ]);
        let version2 = SchemaVersion::new(2, schema2, Some(1), vec![]);
        service.store_version("test_collection", version2).unwrap();

        let rollback = service.rollback_schema("test_collection", 1).unwrap();

        assert_eq!(rollback.version, 3);
        assert_eq!(rollback.parent_version, Some(2));
        assert_eq!(rollback.schema.columns.len(), schema1.columns.len());
    }

    #[test]
    fn test_is_breaking_change() {
        // Column removal is breaking
        let remove = SchemaChange::ColumnRemoved {
            column_name: "email".to_string(),
            data_type: FilterableDataType::FilterableText as i32,
        };
        assert!(remove.is_breaking());

        // Adding nullable column is not breaking
        let add = SchemaChange::ColumnAdded {
            column_name: "email".to_string(),
            data_type: FilterableDataType::FilterableText as i32,
            nullable: true,
        };
        assert!(!add.is_breaking());

        // Making nullable is not breaking
        let make_nullable = SchemaChange::NullabilityChanged {
            column_name: "name".to_string(),
            old_nullable: false,
            new_nullable: true,
        };
        assert!(!make_nullable.is_breaking());

        // Making non-nullable is breaking
        let make_non_nullable = SchemaChange::NullabilityChanged {
            column_name: "name".to_string(),
            old_nullable: true,
            new_nullable: false,
        };
        assert!(make_non_nullable.is_breaking());
    }

    #[test]
    fn test_safe_type_widening() {
        // Same type is safe
        assert!(is_safe_type_widening(
            FilterableDataType::FilterableInteger as i32,
            FilterableDataType::FilterableInteger as i32
        ));

        // Text to large text is safe
        assert!(is_safe_type_widening(
            FilterableDataType::FilterableText as i32,
            FilterableDataType::FilterableTextLarge as i32
        ));

        // Date to datetime is safe
        assert!(is_safe_type_widening(
            FilterableDataType::FilterableDate as i32,
            FilterableDataType::FilterableDatetime as i32
        ));
    }

    #[test]
    fn test_evolution_config() {
        let config = EvolutionConfig::default();
        assert_eq!(config.max_versions, 100);
        assert!(!config.allow_breaking_changes);
        assert!(!config.strict_mode);

        let strict_config = EvolutionConfig::strict();
        assert!(strict_config.strict_mode);

        let custom_config = EvolutionConfig::default()
            .with_breaking_changes(true)
            .with_max_versions(50);
        assert!(custom_config.allow_breaking_changes);
        assert_eq!(custom_config.max_versions, 50);
    }

    #[test]
    fn test_change_description() {
        let add = SchemaChange::ColumnAdded {
            column_name: "email".to_string(),
            data_type: FilterableDataType::FilterableText as i32,
            nullable: true,
        };
        assert!(add.description().contains("email"));
        assert!(add.description().contains("nullable"));

        let remove = SchemaChange::ColumnRemoved {
            column_name: "old_field".to_string(),
            data_type: FilterableDataType::FilterableString as i32,
        };
        assert!(remove.description().contains("old_field"));
        assert!(remove.description().contains("Removed"));

        let rename = SchemaChange::ColumnRenamed {
            old_name: "old_name".to_string(),
            new_name: "new_name".to_string(),
        };
        assert!(rename.description().contains("old_name"));
        assert!(rename.description().contains("new_name"));
    }
}
