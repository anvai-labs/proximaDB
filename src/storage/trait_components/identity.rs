//! Storage Engine Identity Trait
//!
//! Defines the core identity and capability information for storage engines.
//! This trait satisfies the Single Responsibility Principle by focusing solely
//! on engine identification and capability reporting.

use crate::core::types::StorageEngineType;
use crate::storage::traits::StorageFormatStrategy;

/// Core identity trait for storage engines
///
/// Every storage engine must provide basic identification and capability information.
/// This is the foundational trait that all other traits build upon.
///
/// # Example
///
/// ```rust,ignore
/// impl StorageIdentity for SstEngine {
///     fn engine_name(&self) -> &'static str { "SST" }
///     fn engine_version(&self) -> &'static str { "1.0.0" }
///     fn strategy(&self) -> StorageFormatStrategy { StorageFormatStrategy::Sst }
/// }
/// ```
pub trait StorageIdentity: Send + Sync {
    /// Human-readable engine name (e.g., "SST", "NOVA", "HELIX")
    fn engine_name(&self) -> &'static str;

    /// Engine version string (semver format preferred)
    fn engine_version(&self) -> &'static str;

    /// Storage strategy this engine implements
    fn strategy(&self) -> StorageFormatStrategy;

    /// Get the storage engine type for AXIS indexing and event logging
    ///
    /// This method eliminates the need for string matching on engine_name(),
    /// following the Open/Closed Principle. Each engine provides its type
    /// directly, so adding new engines doesn't require modifying dispatch code.
    fn engine_type(&self) -> StorageEngineType {
        // Default implementation maps from strategy for backward compatibility
        match self.strategy() {
            StorageFormatStrategy::Sst => StorageEngineType::SST,
            // VIPER removed (ADR-093); unreachable for live engines (factory
            // rejects it) — SST baseline like the Chrono arm.
            StorageFormatStrategy::Viper => StorageEngineType::SST,
            StorageFormatStrategy::Helix => StorageEngineType::HELIX,
            StorageFormatStrategy::Nova => StorageEngineType::NOVA,
            StorageFormatStrategy::Swift => StorageEngineType::SWIFT,
            StorageFormatStrategy::Raptor => StorageEngineType::RAPTOR,
            StorageFormatStrategy::TimeSeries => StorageEngineType::TST,
            // Future engines should override this method
            _ => StorageEngineType::SST,
        }
    }

    // ── Format-vocabulary aliases (engines → formats convergence) ───────────
    // Canonical `format_*` names delegating to the legacy `engine_*` methods, so
    // new code can use the format vocabulary without a call-site sweep (the
    // legacy names are required methods on a `*Format` trait shared by unrelated
    // types, so a mechanical rename is unsafe). Override only the `engine_*`
    // methods; these aliases follow. See `docs/12-design/NAMING_CONVENTIONS.adoc`.

    /// Canonical alias for [`Self::engine_name`].
    fn format_name(&self) -> &'static str {
        self.engine_name()
    }

    /// Canonical alias for [`Self::engine_version`].
    fn format_version(&self) -> &'static str {
        self.engine_version()
    }

    /// Canonical alias for [`Self::engine_type`].
    fn format_type(&self) -> StorageEngineType {
        self.engine_type()
    }

    /// Check if engine supports collection-level operations
    ///
    /// Engines that return false operate on the entire database.
    fn supports_collection_level_operations(&self) -> bool {
        match self.strategy() {
            StorageFormatStrategy::Sst => false, // SST operates on entire tree
            _ => true,
        }
    }

    /// Check if engine supports atomic operations
    ///
    /// Atomic operations guarantee all-or-nothing semantics.
    fn supports_atomic_operations(&self) -> bool {
        match self.strategy() {
            StorageFormatStrategy::Sst => false,    // Eventual consistency
            StorageFormatStrategy::Raptor => false, // Eventual consistency
            _ => true,
        }
    }

    /// Check if engine supports background operations
    fn supports_background_operations(&self) -> bool {
        true // All engines support background operations by default
    }

    /// Check if engine supports a specific named feature
    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "collection_level_operations" => self.supports_collection_level_operations(),
            "atomic_operations" => self.supports_atomic_operations(),
            "background_operations" => self.supports_background_operations(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod format_alias_tests {
    use super::*;

    struct Mock;
    impl StorageIdentity for Mock {
        fn engine_name(&self) -> &'static str {
            "SST"
        }
        fn engine_version(&self) -> &'static str {
            "1.0.0"
        }
        fn strategy(&self) -> StorageFormatStrategy {
            StorageFormatStrategy::Sst
        }
    }

    #[test]
    fn format_aliases_delegate_to_engine_methods() {
        let m = Mock;
        assert_eq!(m.format_name(), m.engine_name());
        assert_eq!(m.format_version(), m.engine_version());
        assert_eq!(m.format_type(), m.engine_type());
        assert_eq!(m.format_name(), "SST");
    }
}
