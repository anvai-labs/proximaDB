// Engine Migration Implementation
// Handles migration between different storage engines (SST ↔ SWIFT ↔ NOVA)

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::info;

use super::{
    MigrationConfig, MigrationEvent, MigrationEventType, MigrationStatus, MigrationStrategy,
};
use crate::proto::proximadb_v1::StorageEngine as ProtoStorageEngine;
use crate::storage::engines::factory::StorageFormatFactory;
use crate::storage::traits::UnifiedStorageFormat;

/// Engine migrator for moving data between storage engines
pub struct EngineMigrator {
    /// Migration configuration
    config: MigrationConfig,

    /// Source and target engines
    #[allow(dead_code)]
    source_engine: Arc<dyn UnifiedStorageFormat>,
    #[allow(dead_code)]
    target_engine: Arc<dyn UnifiedStorageFormat>,

    /// Migration state
    status: Arc<RwLock<MigrationStatus>>,
    events: Arc<RwLock<Vec<MigrationEvent>>>,

    /// Concurrency control
    semaphore: Arc<Semaphore>,

    /// Progress tracking
    progress: Arc<RwLock<MigrationProgress>>,
}

/// Migration progress tracking
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    pub total_collections: usize,
    pub completed_collections: usize,
    pub current_collection: Option<String>,
    pub total_records: u64,
    pub migrated_records: u64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    pub throughput_records_per_second: f64,
}

/// Backwards-compat alias for [`MigratorMigrationResult`].
pub type MigrationResult = MigratorMigrationResult;

/// Migration result
#[derive(Debug, Clone)]
pub struct MigratorMigrationResult {
    pub success: bool,
    pub migrated_collections: Vec<String>,
    pub total_records_migrated: u64,
    pub total_time_ms: u64,
    pub average_throughput: f64,
    pub performance_metrics: HashMap<String, f64>,
    // Deferred: Restore validation module
    // pub validation_results: Option<super::validation::ValidationReport>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Backwards-compat alias for [`MigratorMigrationPlan`].
pub type MigrationPlan = MigratorMigrationPlan;

/// Migration plan
#[derive(Debug, Clone)]
pub struct MigratorMigrationPlan {
    pub migration_id: String,
    pub source_engine: ProtoStorageEngine,
    pub target_engine: ProtoStorageEngine,
    pub collections: Vec<CollectionMigratorMigrationPlan>,
    pub estimated_duration: chrono::Duration,
    pub resource_requirements: MigratorResourceRequirements,
    pub risk_assessment: RiskAssessment,
}

/// Collection-specific migration plan
#[derive(Debug, Clone)]
pub struct CollectionMigratorMigrationPlan {
    pub collection_id: String,
    pub record_count: u64,
    pub data_size_bytes: u64,
    pub estimated_time: chrono::Duration,
    pub migration_order: usize,
    pub dependencies: Vec<String>,
    pub special_requirements: Vec<String>,
}

/// Backwards-compat alias for [`MigratorResourceRequirements`].
pub type ResourceRequirements = MigratorResourceRequirements;

/// Resource requirements for migration
#[derive(Debug, Clone)]
pub struct MigratorResourceRequirements {
    pub min_memory_gb: f64,
    pub recommended_memory_gb: f64,
    pub min_storage_gb: f64,
    pub temporary_storage_gb: f64,
    pub cpu_cores: usize,
    pub network_bandwidth_mbps: f64,
}

/// Risk assessment for migration
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub data_loss_risk: RiskLevel,
    pub downtime_risk: RiskLevel,
    pub performance_impact_risk: RiskLevel,
    pub rollback_complexity: RiskLevel,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl EngineMigrator {
    /// Create a new migrator
    pub async fn new(config: MigrationConfig) -> Result<Self> {
        // Create source and target engines using async versions for test compatibility
        let source_engine =
            StorageFormatFactory::create_from_proto_async(config.source_engine).await?;
        let target_engine =
            StorageFormatFactory::create_from_proto_async(config.target_engine).await?;

        let semaphore = Arc::new(Semaphore::new(config.performance.parallel_workers));

        Ok(Self {
            config,
            source_engine,
            target_engine,
            status: Arc::new(RwLock::new(MigrationStatus::Planning)),
            events: Arc::new(RwLock::new(Vec::new())),
            semaphore,
            progress: Arc::new(RwLock::new(MigrationProgress::new())),
        })
    }

    /// Create migration plan
    pub async fn create_plan(&self) -> Result<MigratorMigrationPlan> {
        info!(
            "Creating migration plan from {:?} to {:?}",
            self.config.source_engine, self.config.target_engine
        );

        let collections = if self.config.collections.is_empty() {
            // Get all collections from source engine
            self.get_all_collections().await?
        } else {
            self.config.collections.clone()
        };

        let mut collection_plans = Vec::new();
        let mut total_duration = chrono::Duration::zero();
        let mut total_data_size = 0u64;

        for (order, collection_id) in collections.iter().enumerate() {
            let plan = self.create_collection_plan(collection_id, order).await?;
            total_duration += plan.estimated_time;
            total_data_size += plan.data_size_bytes;
            collection_plans.push(plan);
        }

        let resource_requirements = self.calculate_resource_requirements(total_data_size);
        let risk_assessment = self.assess_migration_risks(&collection_plans);

        Ok(MigratorMigrationPlan {
            migration_id: proximadb_kernel::uuid::Uuid::new_v4().to_string(),
            source_engine: self.config.source_engine,
            target_engine: self.config.target_engine,
            // strategy removed -  self.config.strategy.clone(),
            collections: collection_plans,
            estimated_duration: total_duration,
            resource_requirements,
            risk_assessment,
        })
    }

    /// Execute migration
    pub async fn execute(&self, plan: &MigratorMigrationPlan) -> Result<MigratorMigrationResult> {
        info!("Starting migration execution: {}", plan.migration_id);

        // Update status
        *self.status.write().await = MigrationStatus::InProgress {
            progress_percent: 0.0,
            current_collection: "".to_string(),
            estimated_completion: chrono::Utc::now() + plan.estimated_duration,
        };

        let start_time = std::time::Instant::now();

        // Execute based on strategy
        let (migrated_collections, total_records_migrated, errors, warnings) =
            match self.config.strategy {
                MigrationStrategy::CopyThenSwitch => {
                    let result = self.execute_copy_then_switch(&plan.collections).await?;
                    (
                        result.migrated_collections,
                        result.total_records,
                        result.errors,
                        result.warnings,
                    )
                }
                MigrationStrategy::GradualWithDualWrite => {
                    let result = self.execute_gradual_migration(&plan.collections).await?;
                    (
                        result.migrated_collections,
                        result.total_records,
                        result.errors,
                        result.warnings,
                    )
                }
                MigrationStrategy::InPlace => {
                    let result = self.execute_in_place_migration(&plan.collections).await?;
                    (
                        result.migrated_collections,
                        result.total_records,
                        result.errors,
                        result.warnings,
                    )
                }
                MigrationStrategy::BlueGreen => {
                    let result = self.execute_blue_green_migration(&plan.collections).await?;
                    (
                        result.migrated_collections,
                        result.total_records,
                        result.errors,
                        result.warnings,
                    )
                }
            };

        let total_time = start_time.elapsed().as_millis() as u64;
        let average_throughput = if total_time > 0 {
            (total_records_migrated as f64 / total_time as f64) * 1000.0
        } else {
            0.0
        };

        // Update final status
        if errors.is_empty() {
            *self.status.write().await = MigrationStatus::Completed {
                completion_time: chrono::Utc::now(),
                migrated_collections: migrated_collections.clone(),
                performance_metrics: HashMap::new(),
            };
        } else {
            *self.status.write().await = MigrationStatus::Failed {
                error: errors.join("; "),
                failure_time: chrono::Utc::now(),
                rollback_initiated: self.config.rollback.auto_rollback_on_failure,
            };
        }

        Ok(MigratorMigrationResult {
            success: errors.is_empty(),
            migrated_collections,
            total_records_migrated,
            total_time_ms: total_time,
            average_throughput,
            performance_metrics: HashMap::new(),
            errors,
            warnings,
        })
    }

    /// Get current migration status
    pub async fn get_status(&self) -> MigrationStatus {
        self.status.read().await.clone()
    }

    /// Get migration events
    pub async fn get_events(&self) -> Vec<MigrationEvent> {
        self.events.read().await.clone()
    }

    /// Get migration progress
    pub async fn get_progress(&self) -> MigrationProgress {
        self.progress.read().await.clone()
    }

    // Private implementation methods

    async fn get_all_collections(&self) -> Result<Vec<String>> {
        // In production, would query source engine for all collections
        // For now, return empty list
        Ok(Vec::new())
    }

    async fn create_collection_plan(
        &self,
        collection_id: &str,
        order: usize,
    ) -> Result<CollectionMigratorMigrationPlan> {
        // Estimate collection size and complexity
        let record_count = 1000; // Would be queried from source engine
        let data_size_bytes = record_count * 768 * 4; // Estimate based on dimension

        let estimated_time = super::utils::estimate_migration_time(
            self.config.source_engine,
            self.config.target_engine,
            data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0), // Convert to GB
            &self.config.performance,
        );

        Ok(CollectionMigratorMigrationPlan {
            collection_id: collection_id.to_string(),
            record_count,
            data_size_bytes,
            estimated_time,
            migration_order: order,
            dependencies: Vec::new(),
            special_requirements: Vec::new(),
        })
    }

    async fn execute_copy_then_switch(
        &self,
        collections: &[CollectionMigratorMigrationPlan],
    ) -> Result<MigrationExecutionResult> {
        let mut migrated_collections = Vec::new();
        let mut total_records = 0u64;
        let mut errors = Vec::new();
        let warnings = Vec::new();

        for collection_plan in collections {
            self.log_event(
                MigrationEventType::CollectionStarted,
                Some(collection_plan.collection_id.clone()),
                "Starting collection migration".to_string(),
            )
            .await;

            match self
                .migrate_collection_copy_then_switch(collection_plan)
                .await
            {
                Ok(records) => {
                    migrated_collections.push(collection_plan.collection_id.clone());
                    total_records += records;

                    self.log_event(
                        MigrationEventType::CollectionCompleted,
                        Some(collection_plan.collection_id.clone()),
                        format!("Migrated {} records", records),
                    )
                    .await;
                }
                Err(e) => {
                    errors.push(format!(
                        "Collection {}: {}",
                        collection_plan.collection_id, e
                    ));

                    self.log_event(
                        MigrationEventType::Error,
                        Some(collection_plan.collection_id.clone()),
                        e.to_string(),
                    )
                    .await;
                }
            }
        }

        Ok(MigrationExecutionResult {
            migrated_collections,
            total_records,
            errors,
            warnings,
        })
    }

    async fn execute_gradual_migration(
        &self,
        collections: &[CollectionMigratorMigrationPlan],
    ) -> Result<MigrationExecutionResult> {
        // Implementation for gradual migration
        // For brevity, using same implementation as copy-then-switch
        self.execute_copy_then_switch(collections).await
    }

    async fn execute_in_place_migration(
        &self,
        collections: &[CollectionMigratorMigrationPlan],
    ) -> Result<MigrationExecutionResult> {
        // Implementation for in-place migration
        // For brevity, using same implementation as copy-then-switch
        self.execute_copy_then_switch(collections).await
    }

    async fn execute_blue_green_migration(
        &self,
        collections: &[CollectionMigratorMigrationPlan],
    ) -> Result<MigrationExecutionResult> {
        // Implementation for blue-green migration
        // For brevity, using same implementation as copy-then-switch
        self.execute_copy_then_switch(collections).await
    }

    async fn migrate_collection_copy_then_switch(
        &self,
        plan: &CollectionMigratorMigrationPlan,
    ) -> Result<u64> {
        let _permit = self.semaphore.acquire().await?;

        // In production, would:
        // 1. Read all records from source engine
        // 2. Transform records if needed for target engine
        // 3. Write records to target engine in batches
        // 4. Validate migration
        // 5. Switch traffic to target engine

        // For now, return mock result
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Simulate work
        Ok(plan.record_count)
    }

    fn calculate_resource_requirements(
        &self,
        total_data_size: u64,
    ) -> MigratorResourceRequirements {
        let data_size_gb = total_data_size as f64 / (1024.0 * 1024.0 * 1024.0);

        MigratorResourceRequirements {
            min_memory_gb: (data_size_gb * 0.1).max(2.0), // At least 10% of data size, min 2GB
            recommended_memory_gb: (data_size_gb * 0.5).max(8.0), // 50% of data size, min 8GB
            min_storage_gb: data_size_gb * 1.2,           // 20% overhead
            temporary_storage_gb: match self.config.strategy {
                MigrationStrategy::CopyThenSwitch => data_size_gb * 2.0, // Need 2x storage
                MigrationStrategy::BlueGreen => data_size_gb * 2.0,
                MigrationStrategy::GradualWithDualWrite => data_size_gb * 1.5,
                MigrationStrategy::InPlace => data_size_gb * 0.1, // Minimal temp space
            },
            cpu_cores: self.config.performance.parallel_workers,
            network_bandwidth_mbps: 1000.0, // 1Gbps minimum
        }
    }

    fn assess_migration_risks(
        &self,
        collections: &[CollectionMigratorMigrationPlan],
    ) -> RiskAssessment {
        let mut overall_risk = RiskLevel::Low;
        let mut mitigation_strategies = Vec::new();

        // Assess data size risk
        let total_size_gb: f64 = collections
            .iter()
            .map(|c| c.data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            .sum();

        let data_loss_risk = match self.config.strategy {
            MigrationStrategy::InPlace => RiskLevel::High,
            MigrationStrategy::GradualWithDualWrite => RiskLevel::Medium,
            _ => RiskLevel::Low,
        };

        if total_size_gb > 1000.0 {
            overall_risk = RiskLevel::High;
            mitigation_strategies.push("Large dataset - recommend staged migration".to_string());
        }

        if matches!(data_loss_risk, RiskLevel::High) {
            mitigation_strategies.push("High data loss risk - ensure backups".to_string());
        }

        RiskAssessment {
            overall_risk,
            data_loss_risk,
            downtime_risk: RiskLevel::Medium,
            performance_impact_risk: RiskLevel::Medium,
            rollback_complexity: RiskLevel::Medium,
            mitigation_strategies,
        }
    }

    async fn log_event(
        &self,
        event_type: MigrationEventType,
        collection_id: Option<String>,
        message: String,
    ) {
        let event = MigrationEvent {
            timestamp: chrono::Utc::now(),
            event_type,
            collection_id,
            message,
            metadata: HashMap::new(),
        };

        self.events.write().await.push(event);
    }
}

/// Internal migration execution result
struct MigrationExecutionResult {
    migrated_collections: Vec<String>,
    total_records: u64,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl MigrationProgress {
    pub fn new() -> Self {
        Self {
            total_collections: 0,
            completed_collections: 0,
            current_collection: None,
            total_records: 0,
            migrated_records: 0,
            start_time: chrono::Utc::now(),
            estimated_completion: None,
            throughput_records_per_second: 0.0,
        }
    }

    pub fn update_progress(&mut self, migrated_records: u64) {
        self.migrated_records += migrated_records;

        // Update throughput
        let elapsed_seconds = (chrono::Utc::now() - self.start_time).num_seconds() as f64;
        if elapsed_seconds > 0.0 {
            self.throughput_records_per_second = self.migrated_records as f64 / elapsed_seconds;
        }

        // Update estimated completion
        if self.throughput_records_per_second > 0.0 && self.total_records > self.migrated_records {
            let remaining_records = self.total_records - self.migrated_records;
            let remaining_seconds = remaining_records as f64 / self.throughput_records_per_second;
            self.estimated_completion =
                Some(chrono::Utc::now() + chrono::Duration::seconds(remaining_seconds as i64));
        }
    }
}

impl Default for MigrationProgress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_migration_plan_creation() {
        let config = MigrationConfig {
            source_engine: ProtoStorageEngine::Sst,
            target_engine: ProtoStorageEngine::Nova,
            collections: vec!["test_collection".to_string()],
            // strategy removed -  MigrationStrategy::CopyThenSwitch,
            ..Default::default()
        };

        let migrator = EngineMigrator::new(config).await.unwrap();
        let plan = migrator.create_plan().await.unwrap();

        assert_eq!(plan.source_engine, ProtoStorageEngine::Sst);
        assert_eq!(plan.target_engine, ProtoStorageEngine::Nova);
        assert_eq!(plan.collections.len(), 1);
        assert!(plan.estimated_duration.num_seconds() > 0);
    }

    #[tokio::test]
    async fn test_migration_progress_tracking() {
        let mut progress = MigrationProgress::new();

        // Set start time in the past to ensure elapsed time > 0
        progress.start_time = chrono::Utc::now() - chrono::Duration::seconds(1);
        progress.total_records = 1000;
        progress.update_progress(250);

        assert_eq!(progress.migrated_records, 250);
        assert!(progress.throughput_records_per_second > 0.0);
        assert!(progress.estimated_completion.is_some());
    }

    #[tokio::test]
    async fn test_resource_requirements_calculation() {
        let config = MigrationConfig::default();
        let migrator = EngineMigrator::new(config).await.unwrap();

        let requirements = migrator.calculate_resource_requirements(10 * 1024 * 1024 * 1024); // 10GB

        assert!(requirements.min_memory_gb >= 2.0);
        assert!(requirements.recommended_memory_gb >= requirements.min_memory_gb);
        assert!(requirements.min_storage_gb > 10.0);
    }

    #[tokio::test]
    async fn test_risk_assessment() {
        let collections = vec![CollectionMigratorMigrationPlan {
            collection_id: "small_collection".to_string(),
            record_count: 1000,
            data_size_bytes: 1024 * 1024, // 1MB
            estimated_time: chrono::Duration::minutes(5),
            migration_order: 0,
            dependencies: Vec::new(),
            special_requirements: Vec::new(),
        }];

        let config = MigrationConfig {
            strategy: MigrationStrategy::InPlace,
            ..Default::default()
        };

        let migrator = EngineMigrator::new(config).await.unwrap();

        let risk = migrator.assess_migration_risks(&collections);

        // In-place migration should have high data loss risk
        assert!(matches!(risk.data_loss_risk, RiskLevel::High));
        assert!(!risk.mitigation_strategies.is_empty());
    }
}
