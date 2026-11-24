//! Storage engine type identifier shared across storage and index layers.

use serde::{Deserialize, Serialize};

/// Identifies the available storage engines in ProximaDB.
///
/// Shared across storage and index layers to avoid circular dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageEngineType {
    SST,
    NOVA,
    RAPTOR,
    SWIFT,
    HELIX,
    TST,
}

impl StorageEngineType {
    pub fn as_str(self) -> &'static str {
        match self {
            StorageEngineType::SST => "SST",
            StorageEngineType::NOVA => "NOVA",
            StorageEngineType::RAPTOR => "RAPTOR",
            StorageEngineType::SWIFT => "SWIFT",
            StorageEngineType::HELIX => "HELIX",
            StorageEngineType::TST => "TST",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SST" => Some(StorageEngineType::SST),
            "NOVA" => Some(StorageEngineType::NOVA),
            "RAPTOR" => Some(StorageEngineType::RAPTOR),
            "SWIFT" => Some(StorageEngineType::SWIFT),
            "HELIX" => Some(StorageEngineType::HELIX),
            "TST" => Some(StorageEngineType::TST),
            _ => None,
        }
    }
}

impl std::fmt::Display for StorageEngineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for StorageEngineType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| format!("Unknown storage engine type: {}", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn every_engine_round_trips_through_name_display_and_parser() {
        let cases = [
            (StorageEngineType::SST, "SST"),
            (StorageEngineType::NOVA, "NOVA"),
            (StorageEngineType::RAPTOR, "RAPTOR"),
            (StorageEngineType::SWIFT, "SWIFT"),
            (StorageEngineType::HELIX, "HELIX"),
            (StorageEngineType::TST, "TST"),
        ];

        for (engine, name) in cases {
            assert_eq!(engine.as_str(), name);
            assert_eq!(engine.to_string(), name);
            assert_eq!(StorageEngineType::from_name(name), Some(engine));
            assert_eq!(
                StorageEngineType::from_name(&name.to_lowercase()),
                Some(engine)
            );
            assert_eq!(StorageEngineType::from_str(name).unwrap(), engine);
        }
    }

    #[test]
    fn unknown_engine_names_are_rejected_with_context() {
        assert_eq!(StorageEngineType::from_name("unknown"), None);

        let err = StorageEngineType::from_str("unknown").unwrap_err();
        assert!(err.contains("Unknown storage engine type: unknown"));
    }
}
