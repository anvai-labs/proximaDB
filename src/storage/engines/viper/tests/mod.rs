//! VIPER Engine Test Module

// Core test utilities
pub mod test_data_generator;

// Main engine tests
#[cfg(test)]
pub mod engine_tests;

// Compaction tests
#[cfg(test)]
pub mod compaction_tests;

#[cfg(test)]
pub mod debug_compaction_test;

// Storage tests
pub mod parquet_storage_tests;
