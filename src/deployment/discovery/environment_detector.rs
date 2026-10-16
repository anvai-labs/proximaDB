//! Environment Detection and Analysis
//!
//! Automatically detects enterprise deployment environments and generates
//! optimal ProximaDB configurations for one-click deployment.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Enterprise environment detector for automated deployment
pub struct EnvironmentDetector {
    #[allow(dead_code)]
    detection_config: DetectionConfig,
    platform_analyzers: HashMap<PlatformType, Box<dyn PlatformAnalyzer + Send + Sync>>,
}

/// Configuration for environment detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Whether to probe AWS, Azure, and GCP metadata services during detection
    pub enable_cloud_detection: bool,
    /// Whether to test for a reachable Kubernetes API server during detection
    pub enable_kubernetes_detection: bool,
    /// Whether to probe for Docker daemon availability during detection
    pub enable_docker_detection: bool,
    /// Maximum seconds to wait for any individual platform probe before timing out
    pub timeout_seconds: u32,
    /// Whether to perform deep resource and performance analysis in addition to basic detection
    pub detailed_analysis: bool,
}

/// Detected enterprise environment details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEnvironment {
    /// The infrastructure platform that was identified
    pub platform_type: PlatformType,
    /// CPU, memory, storage and GPU resources available on the host
    pub resource_availability: ResourceAvailability,
    /// Network topology and firewall requirements for the deployment
    pub network_configuration: DiscoveryNetworkConfig,
    /// Compliance and security policies that must be enforced
    pub security_constraints: SecurityConstraints,
    /// Performance characteristics and optimal ProximaDB tuning parameters
    pub performance_characteristics: DiscoveryPerformanceProfile,
    /// Recommended deployment strategy and scaling configuration
    pub recommended_deployment: DeploymentRecommendation,
    /// UTC timestamp when this environment snapshot was captured
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Supported deployment platforms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PlatformType {
    /// Container orchestration via Kubernetes (preferred for enterprise)
    Kubernetes,
    /// Single-host Docker Compose deployment
    DockerCompose,
    /// Amazon Web Services (EC2, ECS, EKS, etc.)
    AWS,
    /// Microsoft Azure (AKS, VMs, etc.)
    Azure,
    /// Google Cloud Platform (GKE, Compute Engine, etc.)
    GCP,
    /// VMware vSphere / vCenter virtualisation environment
    VMware,
    /// Physical hardware or unmanaged virtual machines
    BareMetal,
    /// Mixed deployment spanning multiple platform types
    Hybrid,
}

/// Resource availability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    /// Number of logical CPU cores available to ProximaDB
    pub cpu_cores: u32,
    /// Total system memory available in gigabytes
    pub memory_gb: u32,
    /// Available disk storage in gigabytes
    pub storage_gb: u64,
    /// Estimated network throughput in megabits per second
    pub network_bandwidth_mbps: u32,
    /// Whether a CUDA/Metal-compatible GPU was detected
    pub gpu_available: bool,
    /// Whether the storage subsystem supports high IOPS (NVMe/SSD)
    pub high_iops_storage: bool,
    /// ProximaDB capacity estimates derived from the detected resources
    pub estimated_capacity: CapacityEstimate,
}

/// Capacity estimation for ProximaDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEstimate {
    /// Estimated maximum number of collections the instance can handle
    pub max_collections: u32,
    /// Estimated maximum number of vectors storable across all collections
    pub max_vectors_total: u64,
    /// Estimated sustained query throughput in queries per second
    pub estimated_qps: u32,
    /// Storage engine recommended for the detected resource profile
    pub recommended_storage_engine: String,
}

/// Backwards-compat alias for [`DiscoveryNetworkConfig`].
pub type NetworkConfig = DiscoveryNetworkConfig;

/// Network configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryNetworkConfig {
    /// Whether the ProximaDB API must be reachable from outside the cluster
    pub public_access_required: bool,
    /// Whether an external load balancer is available for traffic distribution
    pub load_balancer_available: bool,
    /// Whether TLS termination can be handled at the network boundary
    pub ssl_termination_available: bool,
    /// Whether an internal DNS service is available for service discovery
    pub internal_dns_available: bool,
    /// Firewall rules that must be opened for ProximaDB to operate
    pub firewall_rules_needed: Vec<FirewallRule>,
}

/// Firewall rule requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    /// Network port number that must be allowed
    pub port: u16,
    /// Transport protocol (e.g., "TCP", "UDP")
    pub protocol: String,
    /// Human-readable description of the service using this port
    pub description: String,
    /// Whether the rule is mandatory for ProximaDB to function
    pub required: bool,
}

/// Security constraints for enterprise deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraints {
    /// Whether the environment has no outbound internet connectivity
    pub air_gapped_environment: bool,
    /// Structured compliance frameworks detected or required for this environment
    pub compliance_requirements: Vec<ComplianceFramework>,
    /// Encryption requirements that must be satisfied by the deployment
    pub encryption_requirements: EncryptionRequirements,
    /// Whether full audit logging of all data operations is mandatory
    pub audit_logging_required: bool,
    /// Whether strict network isolation between tenants is required
    pub network_isolation_required: bool,
    /// Whether encryption at rest and in transit is required
    pub encryption_required: bool,
    /// String labels for compliance frameworks (may overlap with `compliance_requirements`)
    pub compliance_frameworks: Vec<String>,
    /// Required access control tier (e.g., "basic", "enterprise", "government")
    pub access_control_level: String,
}

/// Compliance frameworks detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceFramework {
    /// SOC 2 Type II — service organisation controls for security and availability
    SOC2,
    /// General Data Protection Regulation — EU data privacy law
    GDPR,
    /// Health Insurance Portability and Accountability Act — US healthcare data
    HIPAA,
    /// Federal Risk and Authorization Management Program — US government cloud
    FedRAMP,
    /// ISO/IEC 27001 — international information security management standard
    ISO27001,
    /// Payment Card Industry Data Security Standard
    PciDss,
}

/// Encryption requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirements {
    /// Whether all stored data must be encrypted at rest
    pub data_at_rest: bool,
    /// Whether all network communication must be encrypted in transit
    pub data_in_transit: bool,
    /// Whether customer-managed key (CMK) or HSM-backed key management is required
    pub key_management_required: bool,
    /// Specific encryption algorithm to use (e.g., "AES-256"); `None` means any strong algorithm
    pub encryption_algorithm: Option<String>,
}

/// Backwards-compat alias for [`DiscoveryPerformanceProfile`].
pub type PerformanceProfile = DiscoveryPerformanceProfile;

/// Performance profile of target environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryPerformanceProfile {
    /// Estimated maximum query throughput in queries per second
    pub estimated_qps_capacity: u32,
    /// Storage subsystem IOPS capability
    pub storage_iops: u32,
    /// Measured or estimated network round-trip latency in milliseconds
    pub network_latency_ms: f64,
    /// Storage engine recommended based on the performance characteristics
    pub recommended_storage_engine: String,
    /// Computed optimal ProximaDB configuration for this environment
    pub optimal_configuration: OptimalConfig,
}

/// Optimal configuration recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfig {
    /// Recommended memory allocation for the ProximaDB process in megabytes
    pub memory_allocation_mb: u32,
    /// Recommended number of Rayon/Tokio worker threads
    pub worker_threads: u32,
    /// Recommended in-memory cache size in megabytes
    pub cache_size_mb: u32,
    /// Recommended WAL/LSM write buffer size in megabytes
    pub write_buffer_size_mb: u32,
    /// Whether to enable GPU-accelerated distance computation
    pub enable_gpu_acceleration: bool,
    /// Vector quantization strategy to apply (e.g., "PQ8", "SQ8", "none")
    pub quantization_strategy: String,
}

/// Deployment recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecommendation {
    /// Recommended topology strategy (single-node, multi-node, HA, global)
    pub deployment_strategy: DeploymentStrategy,
    /// Replica count and auto-scaling configuration
    pub scaling_configuration: ScalingConfig,
    /// Monitoring and observability setup recommendation
    pub monitoring_setup: DiscoveryMonitoringConfig,
    /// Backup frequency and retention strategy
    pub backup_strategy: BackupStrategy,
    /// Estimated time to complete the full deployment in minutes
    pub estimated_deployment_time_minutes: u32,
}

/// Deployment strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// One ProximaDB instance; suitable for development or small workloads
    SingleNode,
    /// Multiple ProximaDB instances behind a load balancer
    MultiNode,
    /// Redundant multi-node topology with automatic failover
    HighAvailability,
    /// Cross-region distributed deployment for global latency targets
    GlobalDistributed,
}

/// Scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Number of ProximaDB replicas to start with at deployment time
    pub initial_replicas: u32,
    /// Maximum number of replicas the auto-scaler is permitted to create
    pub max_replicas: u32,
    /// Whether the auto-scaler should adjust replica count automatically
    pub auto_scaling_enabled: bool,
    /// Metric thresholds that cause the auto-scaler to act
    pub scaling_triggers: Vec<ScalingTrigger>,
}

/// Auto-scaling triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingTrigger {
    /// Name of the monitored metric (e.g., "cpu_utilization", "memory_utilization")
    pub metric: String,
    /// Value at which the trigger fires (e.g., 80.0 for 80% CPU utilisation)
    pub threshold: f64,
    /// Action to take when the threshold is crossed
    pub action: ScalingAction,
}

/// Scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Add one or more replicas to handle increased load
    ScaleUp,
    /// Remove idle replicas to reduce resource consumption
    ScaleDown,
    /// Emit an alert without changing replica count
    Alert,
}

/// Backwards-compat alias for [`DiscoveryMonitoringConfig`].
pub type MonitoringConfig = DiscoveryMonitoringConfig;

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryMonitoringConfig {
    /// Master switch; when `false` no monitoring subsystems are started
    pub enabled: bool,
    /// Number of days to retain collected metrics time-series data
    pub metrics_retention_days: u32,
    /// Whether alerting rules and notification channels are active
    pub alerting_enabled: bool,
    /// Whether the Grafana/web dashboard is active
    pub dashboard_enabled: bool,
    /// Minimum log severity to capture (e.g., "info", "debug", "warn")
    pub log_level: String,
    /// Whether Prometheus metrics scraping is enabled
    pub enable_metrics: bool,
    /// Whether structured log collection is enabled
    pub enable_logging: bool,
    /// Alias for `alerting_enabled`; kept for API compatibility
    pub enable_alerting: bool,
}

impl DiscoveryMonitoringConfig {
    /// Construct a fully-enabled monitoring configuration suitable for enterprise deployments
    pub fn enterprise_default() -> Self {
        Self {
            enabled: true,
            metrics_retention_days: 30,
            alerting_enabled: true,
            dashboard_enabled: true,
            log_level: "info".to_string(),
            enable_metrics: true,
            enable_logging: true,
            enable_alerting: true,
        }
    }
}

/// Backup strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStrategy {
    /// Whether automated backups are enabled
    pub enabled: bool,
    /// Frequency of backup snapshots in hours (e.g., 24 for daily)
    pub backup_frequency_hours: u32,
    /// Number of days to retain backup snapshots before automatic deletion
    pub retention_days: u32,
    /// Backup type identifier (e.g., "full", "incremental")
    pub backup_type: String,
    /// Storage backend for backup archives (e.g., "s3", "azure-blob", "local")
    pub storage_location: String,
    /// Whether the automated backup scheduler is active (alias for `enabled`)
    pub enable_automated_backup: bool,
}

impl BackupStrategy {
    /// Construct a daily incremental backup strategy suitable for enterprise deployments
    pub fn enterprise_default() -> Self {
        Self {
            enabled: true,
            backup_frequency_hours: 24,
            retention_days: 30,
            backup_type: "incremental".to_string(),
            storage_location: "s3".to_string(),
            enable_automated_backup: true,
        }
    }
}

impl EnvironmentDetector {
    /// Create new environment detector
    pub async fn new() -> Result<Self> {
        let mut platform_analyzers: HashMap<PlatformType, Box<dyn PlatformAnalyzer + Send + Sync>> =
            HashMap::new();

        // Initialize platform-specific analyzers
        platform_analyzers.insert(
            PlatformType::Kubernetes,
            Box::new(KubernetesAnalyzer::new()),
        );
        platform_analyzers.insert(PlatformType::AWS, Box::new(AWSAnalyzer::new()));
        platform_analyzers.insert(PlatformType::Azure, Box::new(AzureAnalyzer::new()));
        platform_analyzers.insert(PlatformType::GCP, Box::new(GCPAnalyzer::new()));
        platform_analyzers.insert(PlatformType::DockerCompose, Box::new(DockerAnalyzer::new()));

        info!(
            "✅ Environment detector initialized with {} platform analyzers",
            platform_analyzers.len()
        );

        Ok(Self {
            detection_config: DetectionConfig::default(),
            platform_analyzers,
        })
    }

    /// Discover and analyze enterprise environment
    pub async fn discover_environment(&self) -> Result<DetectedEnvironment> {
        info!("🔍 Starting enterprise environment discovery and analysis");

        // Step 1: Detect platform type
        let platform_type = self.detect_platform_type().await?;
        info!("✅ Detected platform: {:?}", platform_type);

        // Step 2: Analyze platform-specific capabilities
        let analyzer = self
            .platform_analyzers
            .get(&platform_type)
            .ok_or_else(|| anyhow!("No analyzer available for platform: {:?}", platform_type))?;

        let platform_analysis = analyzer.analyze_platform().await?;

        // Step 3: Analyze resource availability
        let resource_availability = self
            .analyze_resource_availability(&platform_analysis)
            .await?;
        info!(
            "📊 Resources: {} CPU cores, {}GB memory, {}GB storage",
            resource_availability.cpu_cores,
            resource_availability.memory_gb,
            resource_availability.storage_gb
        );

        // Step 4: Detect network configuration
        let network_config = self.analyze_network_configuration().await?;

        // Step 5: Identify security constraints
        let security_constraints = self.analyze_security_constraints().await?;

        // Step 6: Profile performance characteristics
        let performance_profile = self
            .profile_performance_characteristics(&resource_availability)
            .await?;

        // Step 7: Generate deployment recommendation
        let deployment_recommendation = self
            .generate_deployment_recommendation(
                &platform_type,
                &resource_availability,
                &security_constraints,
                &performance_profile,
            )
            .await?;

        let detected_environment = DetectedEnvironment {
            platform_type,
            resource_availability,
            network_configuration: network_config,
            security_constraints,
            performance_characteristics: performance_profile,
            recommended_deployment: deployment_recommendation,
            detected_at: chrono::Utc::now(),
        };

        info!(
            "✅ Environment discovery complete: {:?} deployment recommended",
            detected_environment
                .recommended_deployment
                .deployment_strategy
        );

        Ok(detected_environment)
    }

    /// Detect the deployment platform type
    async fn detect_platform_type(&self) -> Result<PlatformType> {
        debug!("🔍 Detecting deployment platform type...");

        // Check for Kubernetes (highest priority for enterprise)
        if self.is_kubernetes_available().await? {
            info!("📦 Kubernetes detected - enterprise-preferred platform");
            return Ok(PlatformType::Kubernetes);
        }

        // Check for cloud providers
        if self.is_aws_environment().await? {
            info!("☁️ AWS environment detected");
            return Ok(PlatformType::AWS);
        }

        if self.is_azure_environment().await? {
            info!("☁️ Azure environment detected");
            return Ok(PlatformType::Azure);
        }

        if self.is_gcp_environment().await? {
            info!("☁️ GCP environment detected");
            return Ok(PlatformType::GCP);
        }

        // Check for Docker
        if self.is_docker_available().await? {
            info!("🐳 Docker environment detected");
            return Ok(PlatformType::DockerCompose);
        }

        // Default to bare metal/VM
        warn!("⚠️ No containerized platform detected, defaulting to bare metal deployment");
        Ok(PlatformType::BareMetal)
    }

    /// Check if Kubernetes is available and accessible
    async fn is_kubernetes_available(&self) -> Result<bool> {
        debug!("🔍 Checking Kubernetes availability...");

        // Check for kubectl command
        match run_command_with_timeout("kubectl", &["cluster-info"], 5).await {
            Ok(output) => {
                if output.status.success() {
                    debug!("✅ kubectl cluster-info successful");

                    // Additional check: verify we can list nodes
                    match run_command_with_timeout("kubectl", &["get", "nodes"], 5).await {
                        Ok(nodes_output) => {
                            let success = nodes_output.status.success();
                            debug!(
                                "📋 kubectl get nodes: {}",
                                if success { "✅ Success" } else { "❌ Failed" }
                            );
                            Ok(success)
                        }
                        Err(_) => {
                            debug!("⚠️ kubectl get nodes failed");
                            Ok(false)
                        }
                    }
                } else {
                    debug!("❌ kubectl cluster-info failed");
                    Ok(false)
                }
            }
            Err(_) => {
                debug!("❌ kubectl command not found");
                Ok(false)
            }
        }
    }

    /// Check if running in AWS environment
    async fn is_aws_environment(&self) -> Result<bool> {
        debug!("🔍 Checking AWS environment...");

        // Check AWS metadata service
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        Ok(aws_imds_reachable(&client, AWS_IMDS_BASE).await)
    }

    /// Check if running in Azure environment
    async fn is_azure_environment(&self) -> Result<bool> {
        debug!("🔍 Checking Azure environment...");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        match client
            .get("http://169.254.169.254/metadata/instance?api-version=2021-02-01")
            .header("Metadata", "true")
            .send()
            .await
        {
            Ok(response) => {
                let is_azure = response.status().is_success();
                debug!(
                    "☁️ Azure metadata service: {}",
                    if is_azure {
                        "✅ Available"
                    } else {
                        "❌ Not available"
                    }
                );
                Ok(is_azure)
            }
            Err(_) => {
                debug!("❌ Azure metadata service not accessible");
                Ok(false)
            }
        }
    }

    /// Check if running in GCP environment
    async fn is_gcp_environment(&self) -> Result<bool> {
        debug!("🔍 Checking GCP environment...");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        match client
            .get("http://metadata.google.internal/computeMetadata/v1/instance/id")
            .header("Metadata-Flavor", "Google")
            .send()
            .await
        {
            Ok(response) => {
                let is_gcp = response.status().is_success();
                debug!(
                    "☁️ GCP metadata service: {}",
                    if is_gcp {
                        "✅ Available"
                    } else {
                        "❌ Not available"
                    }
                );
                Ok(is_gcp)
            }
            Err(_) => {
                debug!("❌ GCP metadata service not accessible");
                Ok(false)
            }
        }
    }

    /// Check if Docker is available
    async fn is_docker_available(&self) -> Result<bool> {
        debug!("🔍 Checking Docker availability...");

        match run_command_with_timeout("docker", &["version"], 5).await {
            Ok(output) => {
                let available = output.status.success();
                debug!(
                    "🐳 Docker: {}",
                    if available {
                        "✅ Available"
                    } else {
                        "❌ Not available"
                    }
                );
                Ok(available)
            }
            Err(_) => {
                debug!("❌ Docker command not found");
                Ok(false)
            }
        }
    }

    /// Analyze resource availability for ProximaDB deployment
    async fn analyze_resource_availability(
        &self,
        _platform_analysis: &PlatformAnalysis,
    ) -> Result<ResourceAvailability> {
        debug!("📊 Analyzing resource availability...");

        // Get system information
        let cpu_cores = self.detect_cpu_cores().await?;
        let memory_gb = self.detect_memory_gb().await?;
        let storage_gb = self.detect_storage_gb().await?;
        let network_bandwidth = self.detect_network_bandwidth().await?;

        // Check for GPU availability
        let gpu_available = self.detect_gpu_availability().await?;

        // Check for high-IOPS storage
        let high_iops_storage = self.detect_high_iops_storage().await?;

        // Estimate ProximaDB capacity
        let estimated_capacity = self
            .estimate_proximadb_capacity(cpu_cores, memory_gb, storage_gb)
            .await?;

        let resource_availability = ResourceAvailability {
            cpu_cores,
            memory_gb,
            storage_gb,
            network_bandwidth_mbps: network_bandwidth,
            gpu_available,
            high_iops_storage,
            estimated_capacity,
        };

        info!(
            "📊 Resource analysis: {} cores, {}GB RAM, {}GB storage, GPU: {}, High-IOPS: {}",
            cpu_cores, memory_gb, storage_gb, gpu_available, high_iops_storage
        );

        Ok(resource_availability)
    }

    /// Detect number of CPU cores
    async fn detect_cpu_cores(&self) -> Result<u32> {
        // Try multiple methods to detect CPU cores
        if let Ok(output) = run_command_with_timeout("nproc", &[], 2).await
            && let Ok(cores_str) = String::from_utf8(output.stdout)
            && let Ok(cores) = cores_str.trim().parse::<u32>()
        {
            return Ok(cores);
        }

        // Fallback: Use Rust's built-in detection
        Ok(num_cpus::get() as u32)
    }

    /// Detect total memory in GB
    async fn detect_memory_gb(&self) -> Result<u32> {
        // Try reading /proc/meminfo on Linux
        if let Ok(meminfo) = tokio::fs::read_to_string("/proc/meminfo").await {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:")
                    && let Some(kb_str) = line.split_whitespace().nth(1)
                    && let Ok(kb) = kb_str.parse::<u64>()
                {
                    return Ok((kb / 1024 / 1024) as u32); // Convert KB to GB
                }
            }
        }

        // Fallback estimation (conservative)
        warn!("⚠️ Could not detect memory, using conservative 8GB estimate");
        Ok(8)
    }

    /// Detect available storage in GB
    async fn detect_storage_gb(&self) -> Result<u64> {
        // Try using df command to check available disk space
        match run_command_with_timeout("df", &["-BG", "/"], 2).await {
            Ok(output) => {
                if let Ok(df_output) = String::from_utf8(output.stdout) {
                    // Parse df output for available space
                    for line in df_output.lines().skip(1) {
                        let fields: Vec<&str> = line.split_whitespace().collect();
                        if fields.len() >= 4
                            && let Ok(available_gb) = fields[3].trim_end_matches('G').parse::<u64>()
                        {
                            return Ok(available_gb);
                        }
                    }
                }
            }
            Err(_) => {
                warn!("⚠️ df command failed");
            }
        }

        // Fallback estimation
        warn!("⚠️ Could not detect storage, using conservative 100GB estimate");
        Ok(100)
    }

    /// Estimate ProximaDB capacity based on resources
    async fn estimate_proximadb_capacity(
        &self,
        cpu_cores: u32,
        memory_gb: u32,
        storage_gb: u64,
    ) -> Result<CapacityEstimate> {
        // ProximaDB capacity estimation based on hardware
        let max_collections = (memory_gb * 10).min(10000); // ~10 collections per GB RAM, max 10K

        let max_vectors_total = (storage_gb * 1_000_000).min(1_000_000_000); // ~1M vectors per GB, max 1B

        // QPS estimation: ~100 QPS per CPU core with good storage
        let base_qps = cpu_cores * 100;
        let storage_multiplier = if storage_gb > 1000 { 1.5 } else { 1.0 };
        let estimated_qps = (base_qps as f64 * storage_multiplier) as u32;

        // Recommend storage engine based on resources
        let recommended_storage_engine = if memory_gb >= 32 && cpu_cores >= 16 {
            "NOVA".to_string() // High-performance for well-resourced environments
        } else if memory_gb >= 16 {
            "VIPER".to_string() // Balanced performance for medium resources
        } else {
            "SST".to_string() // Memory-efficient for resource-constrained environments
        };

        Ok(CapacityEstimate {
            max_collections,
            max_vectors_total,
            estimated_qps,
            recommended_storage_engine,
        })
    }

    /// Generate deployment recommendation
    async fn generate_deployment_recommendation(
        &self,
        platform_type: &PlatformType,
        resources: &ResourceAvailability,
        security: &SecurityConstraints,
        _performance: &DiscoveryPerformanceProfile,
    ) -> Result<DeploymentRecommendation> {
        // Determine deployment strategy based on resources and requirements
        let deployment_strategy = if resources.cpu_cores >= 16 && resources.memory_gb >= 32 {
            if security.network_isolation_required {
                DeploymentStrategy::HighAvailability
            } else {
                DeploymentStrategy::MultiNode
            }
        } else {
            DeploymentStrategy::SingleNode
        };

        // Configure scaling based on platform capabilities
        let scaling_config = match platform_type {
            PlatformType::Kubernetes => ScalingConfig {
                initial_replicas: if resources.cpu_cores >= 8 { 3 } else { 1 },
                max_replicas: (resources.cpu_cores / 4).clamp(1, 10),
                auto_scaling_enabled: true,
                scaling_triggers: vec![
                    ScalingTrigger {
                        metric: "cpu_utilization".to_string(),
                        threshold: 80.0,
                        action: ScalingAction::ScaleUp,
                    },
                    ScalingTrigger {
                        metric: "memory_utilization".to_string(),
                        threshold: 85.0,
                        action: ScalingAction::ScaleUp,
                    },
                ],
            },
            _ => ScalingConfig {
                initial_replicas: 1,
                max_replicas: 1,
                auto_scaling_enabled: false,
                scaling_triggers: vec![],
            },
        };

        // Estimate deployment time based on complexity
        let estimated_deployment_time = match deployment_strategy {
            DeploymentStrategy::SingleNode => 30,
            DeploymentStrategy::MultiNode => 60,
            DeploymentStrategy::HighAvailability => 90,
            DeploymentStrategy::GlobalDistributed => 120,
        };

        Ok(DeploymentRecommendation {
            deployment_strategy,
            scaling_configuration: scaling_config,
            monitoring_setup: DiscoveryMonitoringConfig::enterprise_default(),
            backup_strategy: BackupStrategy::enterprise_default(),
            estimated_deployment_time_minutes: estimated_deployment_time,
        })
    }

    // Additional detection methods (simplified implementations)
    async fn detect_network_bandwidth(&self) -> Result<u32> {
        Ok(1000)
    } // Default 1Gbps
    async fn detect_gpu_availability(&self) -> Result<bool> {
        Ok(false)
    } // Conservative default
    async fn detect_high_iops_storage(&self) -> Result<bool> {
        Ok(true)
    } // Assume modern storage

    /// Analyze network configuration
    async fn analyze_network_configuration(&self) -> Result<DiscoveryNetworkConfig> {
        debug!("🌐 Analyzing network configuration...");
        Ok(DiscoveryNetworkConfig {
            public_access_required: true,
            load_balancer_available: false,
            ssl_termination_available: false,
            internal_dns_available: true,
            firewall_rules_needed: vec![
                FirewallRule {
                    port: 5678,
                    protocol: "TCP".to_string(),
                    description: "ProximaDB REST API".to_string(),
                    required: true,
                },
                FirewallRule {
                    port: 5679,
                    protocol: "TCP".to_string(),
                    description: "ProximaDB gRPC API".to_string(),
                    required: true,
                },
            ],
        })
    }

    /// Analyze security constraints
    async fn analyze_security_constraints(&self) -> Result<SecurityConstraints> {
        debug!("🔒 Analyzing security constraints...");
        Ok(SecurityConstraints {
            air_gapped_environment: false,
            compliance_requirements: vec![ComplianceFramework::SOC2],
            encryption_requirements: EncryptionRequirements {
                data_at_rest: true,
                data_in_transit: true,
                key_management_required: false,
                encryption_algorithm: Some("AES-256".to_string()),
            },
            audit_logging_required: true,
            network_isolation_required: false,
            encryption_required: true,
            compliance_frameworks: vec!["SOC2".to_string()],
            access_control_level: "basic".to_string(),
        })
    }

    /// Profile performance characteristics
    async fn profile_performance_characteristics(
        &self,
        resource_availability: &ResourceAvailability,
    ) -> Result<DiscoveryPerformanceProfile> {
        debug!("⚡ Profiling performance characteristics...");
        Ok(DiscoveryPerformanceProfile {
            estimated_qps_capacity: resource_availability.cpu_cores * 250,
            storage_iops: if self.detect_high_iops_storage().await? {
                10000
            } else {
                5000
            },
            network_latency_ms: 25.0,
            recommended_storage_engine: "NOVA".to_string(),
            optimal_configuration: OptimalConfig {
                memory_allocation_mb: (resource_availability.memory_gb * 1024 * 3) / 4, // Use 75% of available memory
                worker_threads: resource_availability.cpu_cores.min(16),
                cache_size_mb: resource_availability.memory_gb * 256, // 25% of memory for cache
                write_buffer_size_mb: resource_availability.memory_gb * 128, // 12.5% for write buffer
                enable_gpu_acceleration: self.detect_gpu_availability().await?,
                quantization_strategy: "PQ8".to_string(),
            },
        })
    }
}

/// Trait for platform-specific analysis
#[async_trait]
pub trait PlatformAnalyzer: Send + Sync {
    /// Probe the platform and return a structured analysis of its capabilities and limitations.
    async fn analyze_platform(&self) -> Result<PlatformAnalysis>;
}

/// Platform analysis result
#[derive(Debug, Clone)]
pub struct PlatformAnalysis {
    /// Features and services available on this platform (e.g., "Auto-scaling")
    pub capabilities: Vec<String>,
    /// Known constraints or missing features on this platform
    pub limitations: Vec<String>,
    /// Configuration recommendations specific to this platform
    pub recommendations: Vec<String>,
    /// Arbitrary platform-specific key-value metadata discovered during analysis
    pub resource_details: HashMap<String, String>,
}

/// Kubernetes platform analyzer
#[derive(Debug)]
pub struct KubernetesAnalyzer;

impl KubernetesAnalyzer {
    /// Create a new `KubernetesAnalyzer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for KubernetesAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlatformAnalyzer for KubernetesAnalyzer {
    async fn analyze_platform(&self) -> Result<PlatformAnalysis> {
        debug!("📦 Analyzing Kubernetes platform capabilities...");

        let mut capabilities = vec![
            "Container orchestration".to_string(),
            "Auto-scaling".to_string(),
            "Service discovery".to_string(),
            "Load balancing".to_string(),
        ];

        let mut limitations = vec![];
        let mut recommendations = vec![
            "Use StatefulSets for ProximaDB persistence".to_string(),
            "Configure persistent volumes for data storage".to_string(),
            "Enable horizontal pod autoscaling".to_string(),
        ];

        // Check for additional Kubernetes features
        if let Ok(output) = run_command_with_timeout("kubectl", &["get", "storageclass"], 5).await {
            if output.status.success() {
                capabilities.push("Dynamic volume provisioning".to_string());
            } else {
                limitations.push("No dynamic storage provisioning".to_string());
                recommendations
                    .push("Configure storage classes for dynamic provisioning".to_string());
            }
        }

        Ok(PlatformAnalysis {
            capabilities,
            limitations,
            recommendations,
            resource_details: HashMap::new(),
        })
    }
}

// Similar analyzer implementations for other platforms (AWS, Azure, GCP, Docker)
/// AWS platform analyzer
#[derive(Debug)]
pub struct AWSAnalyzer;

impl AWSAnalyzer {
    /// Create a new `AWSAnalyzer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for AWSAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlatformAnalyzer for AWSAnalyzer {
    async fn analyze_platform(&self) -> Result<PlatformAnalysis> {
        Ok(PlatformAnalysis {
            capabilities: vec!["Managed services".to_string(), "Auto-scaling".to_string()],
            limitations: vec![],
            recommendations: vec!["Use EKS for container deployment".to_string()],
            resource_details: HashMap::new(),
        })
    }
}

// Additional analyzer implementations...
/// Azure platform analyzer
#[derive(Debug)]
pub struct AzureAnalyzer;

impl AzureAnalyzer {
    /// Create a new `AzureAnalyzer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for AzureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlatformAnalyzer for AzureAnalyzer {
    async fn analyze_platform(&self) -> Result<PlatformAnalysis> {
        Ok(PlatformAnalysis {
            capabilities: vec!["Azure services".to_string()],
            limitations: vec![],
            recommendations: vec!["Use AKS for deployment".to_string()],
            resource_details: HashMap::new(),
        })
    }
}

/// GCP platform analyzer
#[derive(Debug)]
pub struct GCPAnalyzer;

impl GCPAnalyzer {
    /// Create a new `GCPAnalyzer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for GCPAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlatformAnalyzer for GCPAnalyzer {
    async fn analyze_platform(&self) -> Result<PlatformAnalysis> {
        Ok(PlatformAnalysis {
            capabilities: vec!["GCP services".to_string()],
            limitations: vec![],
            recommendations: vec!["Use GKE for deployment".to_string()],
            resource_details: HashMap::new(),
        })
    }
}

/// Docker platform analyzer
#[derive(Debug)]
pub struct DockerAnalyzer;

impl DockerAnalyzer {
    /// Create a new `DockerAnalyzer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for DockerAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlatformAnalyzer for DockerAnalyzer {
    async fn analyze_platform(&self) -> Result<PlatformAnalysis> {
        Ok(PlatformAnalysis {
            capabilities: vec!["Container runtime".to_string()],
            limitations: vec!["No orchestration".to_string()],
            recommendations: vec!["Use Docker Compose for multi-container setup".to_string()],
            resource_details: HashMap::new(),
        })
    }
}

// Default implementations and supporting types
impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            enable_cloud_detection: true,
            enable_kubernetes_detection: true,
            enable_docker_detection: true,
            timeout_seconds: 30,
            detailed_analysis: true,
        }
    }
}

// Removed duplicate structs - using original definitions above

// Removed duplicate DiscoveryNetworkConfiguration struct - use DiscoveryNetworkConfig instead

/// Helper function to run a command with a timeout to prevent hanging
async fn run_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout_secs: u64,
) -> Result<std::process::Output> {
    use tokio::time::{Duration, timeout};

    let duration = Duration::from_secs(timeout_secs);
    let output = timeout(
        duration,
        tokio::process::Command::new(program).args(args).output(),
    )
    .await
    .map_err(|_| {
        anyhow!(
            "Command '{}' timed out after {} seconds",
            program,
            timeout_secs
        )
    })??;

    Ok(output)
}

pub use crate::core::utils::aws_imds::AWS_IMDS_BASE;

/// Whether the AWS instance-metadata service at `base` answers an
/// `instance-id` read (environment detection). Always IMDSv2 via the shared
/// [`crate::core::utils::aws_imds::aws_imdsv2_get`] — never a token-less
/// IMDSv1 GET, which is SSRF-reachable and refused (401) on instances with
/// `HttpTokens=required` (TD-SECSCAN-1 §4). `base` is a parameter so the
/// probe is testable against a local IMDS stand-in.
pub async fn aws_imds_reachable(client: &reqwest::Client, base: &str) -> bool {
    match crate::core::utils::aws_imds::aws_imdsv2_get(
        client,
        base,
        "/latest/meta-data/instance-id",
    )
    .await
    {
        Ok(response) => {
            let is_aws = response.status().is_success();
            debug!(
                "☁️ AWS metadata service: {}",
                if is_aws {
                    "✅ Available"
                } else {
                    "❌ Not available"
                }
            );
            is_aws
        }
        Err(e) => {
            debug!("❌ AWS metadata service not accessible: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_environment_detector_creation() {
        let detector = EnvironmentDetector::new().await.unwrap();
        assert!(!detector.platform_analyzers.is_empty());
    }

    #[tokio::test]
    async fn test_platform_detection() {
        let detector = EnvironmentDetector::new().await.unwrap();

        // Test Kubernetes detection
        let k8s_available = detector.is_kubernetes_available().await.unwrap();
        println!("Kubernetes available: {}", k8s_available);

        // Test Docker detection
        let docker_available = detector.is_docker_available().await.unwrap();
        println!("Docker available: {}", docker_available);

        // Smoke test only: the contract is that both detectors run without
        // panicking (asserted via .unwrap() above). We may have neither in CI,
        // so there is nothing meaningful to assert on the bool results.
    }

    #[tokio::test]
    async fn test_resource_detection() {
        let detector = EnvironmentDetector::new().await.unwrap();

        let cpu_cores = detector.detect_cpu_cores().await.unwrap();
        let memory_gb = detector.detect_memory_gb().await.unwrap();

        assert!(cpu_cores > 0);
        assert!(memory_gb > 0);

        println!(
            "Detected resources: {} CPU cores, {}GB memory",
            cpu_cores, memory_gb
        );
    }
}
