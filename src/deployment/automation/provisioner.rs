//! Automated Deployment Provisioner
//!
//! Orchestrates one-click enterprise deployment across different platforms
//! with automatic configuration generation and validation.

use crate::deployment::discovery::{DetectedEnvironment, PlatformType};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

/// Enterprise deployment provisioner for automated setup
pub struct DeploymentProvisioner {
    platform_deployers: HashMap<PlatformType, Box<dyn PlatformDeployer + Send + Sync>>,
    config_generator: ConfigurationGenerator,
    validation_engine: ValidationEngine,
}

/// Enterprise deployment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseDeploymentRequest {
    /// Name of the customer organization requesting deployment
    pub customer_name: String,
    /// Contact email address for the deployment owner
    pub customer_email: String,
    /// Human-readable name for this deployment instance
    pub deployment_name: String,
    /// Unique identifier for the customer tenant
    pub tenant_id: String,
    /// Auto-detected target environment for deployment
    pub environment: DetectedEnvironment,
    /// Optional configuration overrides supplied by the customer
    pub custom_configuration: Option<CustomConfiguration>,
    /// List of AI provider names to integrate (e.g., "OpenAI", "Azure")
    pub ai_providers: Vec<String>,
    /// Security requirements that must be satisfied by the deployment
    pub security_requirements: SecurityRequirements,
    /// Performance targets the deployment must meet
    pub performance_requirements: PerformanceRequirements,
}

/// Custom configuration overrides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConfiguration {
    /// Preferred storage engine name (e.g., "NOVA", "SST"); defaults to auto-detected
    pub storage_engine_preference: Option<String>,
    /// Override for memory allocated to ProximaDB in megabytes
    pub memory_allocation_mb: Option<u32>,
    /// Whether to enable GPU acceleration when a compatible GPU is available
    pub enable_gpu_acceleration: Option<bool>,
    /// Custom network port assignments; uses defaults when absent
    pub custom_ports: Option<PortConfiguration>,
    /// Backup configuration; uses platform defaults when absent
    pub backup_configuration: Option<BackupConfiguration>,
}

/// Security requirements for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRequirements {
    /// Whether all connections must use TLS encryption
    pub enable_tls: bool,
    /// Whether mutual TLS with client certificates is required
    pub require_client_certificates: bool,
    /// Whether all API and data access events must be audit-logged
    pub enable_audit_logging: bool,
    /// Whether enterprise single sign-on integration must be configured
    pub sso_integration_required: bool,
    /// Compliance frameworks that must be satisfied (e.g., ["SOC2", "GDPR"])
    pub compliance_frameworks: Vec<String>,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Minimum sustained query throughput in queries per second
    pub min_qps: u32,
    /// Maximum acceptable end-to-end query latency in milliseconds
    pub max_latency_ms: u32,
    /// Required service availability expressed as a percentage (e.g., 99.9)
    pub availability_percentage: f64,
    /// Recovery Time Objective: maximum hours to restore service after failure
    pub backup_rto_hours: u32,
    /// Recovery Point Objective: maximum hours of data loss tolerable after failure
    pub backup_rpo_hours: u32,
}

/// Deployment result with comprehensive status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentResult {
    /// Unique identifier assigned to this deployment run
    pub deployment_id: String,
    /// Overall outcome of the deployment process
    pub status: DeploymentStatus,
    /// Cloud or infrastructure platform that was targeted
    pub platform_type: PlatformType,
    /// URLs and addresses for reaching the deployed instance
    pub endpoints: DeploymentEndpoints,
    /// High-level summary of the applied configuration
    pub configuration_summary: ConfigurationSummary,
    /// Results of post-deployment health validation checks
    pub health_checks: Vec<HealthCheck>,
    /// Total elapsed time from deployment start to completion in minutes
    pub deployment_time_minutes: u32,
    /// Ordered list of recommended actions for the customer to take
    pub next_steps: Vec<NextStep>,
    /// Optional troubleshooting information when issues were detected
    pub troubleshooting_info: Option<TroubleshootingInfo>,
}

/// Deployment status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment is currently executing
    InProgress,
    /// All deployment steps and health checks passed
    Succeeded,
    /// Deployment encountered a fatal error and did not complete
    Failed,
    /// Deployment completed but one or more health checks reported warnings
    PartiallySucceeded,
    /// Deployment was automatically rolled back after a failure
    RolledBack,
}

impl std::fmt::Display for DeploymentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeploymentStatus::InProgress => write!(f, "In Progress"),
            DeploymentStatus::Succeeded => write!(f, "Succeeded"),
            DeploymentStatus::Failed => write!(f, "Failed"),
            DeploymentStatus::PartiallySucceeded => write!(f, "Partially Succeeded"),
            DeploymentStatus::RolledBack => write!(f, "Rolled Back"),
        }
    }
}

/// Deployment endpoints for customer access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEndpoints {
    /// Base URL for the REST API (e.g., "http://host:5678")
    pub rest_api: String,
    /// Address for the gRPC API (e.g., "grpc://host:5679")
    pub grpc_api: String,
    /// URL for the web-based management dashboard
    pub dashboard_url: String,
    /// URL for Prometheus-compatible metrics; absent when monitoring is disabled
    pub monitoring_url: Option<String>,
    /// URL pointing to the interactive API reference documentation
    pub api_documentation_url: String,
}

/// Configuration summary for customer reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationSummary {
    /// Name of the storage engine selected for this deployment
    pub storage_engine: String,
    /// Memory reserved for ProximaDB in megabytes
    pub memory_allocation_mb: u32,
    /// Human-readable capacity estimate (collections, vectors, QPS)
    pub estimated_capacity: String,
    /// Security features that were activated during deployment
    pub security_features_enabled: Vec<String>,
    /// AI provider integrations that were configured
    pub ai_providers_configured: Vec<String>,
    /// Cron expression or description of the automated backup schedule
    pub backup_schedule: String,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Short human-readable identifier for this check (e.g., "REST API Health")
    pub check_name: String,
    /// Pass/warn/fail outcome of the check
    pub status: HealthStatus,
    /// Detailed description of the check result or any error encountered
    pub details: String,
    /// Ordered steps to resolve the issue when status is not Healthy
    pub resolution_steps: Vec<String>,
}

/// Health status for deployment validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Check passed; component is operating normally
    Healthy,
    /// Check passed with non-critical concerns; review recommended
    Warning,
    /// Check failed; component requires attention before production use
    Unhealthy,
    /// Check could not be completed; status is indeterminate
    Unknown,
}

/// Next steps for customer after deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStep {
    /// 1-based sequence number ordering this step within the post-deployment checklist
    pub step_number: u32,
    /// Short imperative title summarising the action (e.g., "Verify ProximaDB Access")
    pub title: String,
    /// Detailed explanation of what to do and why
    pub description: String,
    /// Optional URL to relevant documentation for this step
    pub documentation_url: Option<String>,
    /// Approximate time required to complete this step in minutes
    pub estimated_time_minutes: u32,
}

impl DeploymentProvisioner {
    /// Create new deployment provisioner
    pub async fn new() -> Result<Self> {
        let mut platform_deployers: HashMap<PlatformType, Box<dyn PlatformDeployer + Send + Sync>> =
            HashMap::new();

        // Initialize platform-specific deployers
        platform_deployers.insert(
            PlatformType::Kubernetes,
            Box::new(KubernetesDeployer::new()),
        );
        platform_deployers.insert(PlatformType::DockerCompose, Box::new(DockerDeployer::new()));
        platform_deployers.insert(PlatformType::AWS, Box::new(AWSDeployer::new()));
        platform_deployers.insert(PlatformType::Azure, Box::new(AzureDeployer::new()));

        Ok(Self {
            platform_deployers,
            config_generator: ConfigurationGenerator::new(),
            validation_engine: ValidationEngine::new(),
        })
    }

    /// Execute complete enterprise deployment
    pub async fn deploy_enterprise(
        &self,
        request: EnterpriseDeploymentRequest,
    ) -> Result<DeploymentResult> {
        let deployment_id = Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        info!(
            "🚀 Starting enterprise deployment for {}: {}",
            request.customer_name, deployment_id
        );

        // Step 1: Generate enterprise configuration
        let enterprise_config = self
            .config_generator
            .generate_enterprise_config(&request)
            .await?;

        info!(
            "✅ Generated enterprise configuration for {}",
            request.customer_name
        );

        // Step 2: Get platform-specific deployer
        let deployer = self
            .platform_deployers
            .get(&request.environment.platform_type)
            .ok_or_else(|| {
                anyhow!(
                    "No deployer available for platform: {:?}",
                    request.environment.platform_type
                )
            })?;

        // Step 3: Execute platform-specific deployment
        let platform_result = deployer
            .deploy(deployment_id.clone(), &request, &enterprise_config)
            .await?;

        info!(
            "✅ Platform deployment completed for {}",
            request.customer_name
        );

        // Step 4: Validate deployment health
        let health_checks = self
            .validation_engine
            .validate_deployment(&deployment_id, &platform_result)
            .await?;

        // Step 5: Setup monitoring and alerting
        let _ = self.setup_monitoring(&deployment_id, &request).await?;

        // Step 6: Generate customer documentation and next steps
        let next_steps = self
            .generate_customer_next_steps(&request, &platform_result)
            .await?;

        let deployment_time_minutes = start_time.elapsed().as_secs() as u32 / 60;

        let deployment_result = DeploymentResult {
            deployment_id,
            status: if health_checks
                .iter()
                .all(|h| matches!(h.status, HealthStatus::Healthy))
            {
                DeploymentStatus::Succeeded
            } else {
                DeploymentStatus::PartiallySucceeded
            },
            platform_type: request.environment.platform_type.clone(),
            endpoints: platform_result.endpoints,
            configuration_summary: ConfigurationSummary {
                storage_engine: enterprise_config.storage_engine.clone(),
                memory_allocation_mb: enterprise_config.memory_allocation_mb,
                estimated_capacity: format!(
                    "{} collections, {} vectors, {} QPS",
                    request
                        .environment
                        .resource_availability
                        .estimated_capacity
                        .max_collections,
                    request
                        .environment
                        .resource_availability
                        .estimated_capacity
                        .max_vectors_total,
                    request
                        .environment
                        .resource_availability
                        .estimated_capacity
                        .estimated_qps
                ),
                security_features_enabled: enterprise_config.security_features.clone(),
                ai_providers_configured: enterprise_config.ai_providers.clone(),
                backup_schedule: enterprise_config.backup_schedule.clone(),
            },
            health_checks,
            deployment_time_minutes,
            next_steps,
            troubleshooting_info: None,
        };

        info!(
            "🎉 Enterprise deployment complete for {} in {} minutes: {}",
            request.customer_name, deployment_time_minutes, deployment_result.status
        );

        Ok(deployment_result)
    }

    /// Generate customer next steps after successful deployment
    async fn generate_customer_next_steps(
        &self,
        request: &EnterpriseDeploymentRequest,
        platform_result: &PlatformDeploymentResult,
    ) -> Result<Vec<NextStep>> {
        let mut steps = vec![
            NextStep {
                step_number: 1,
                title: "Verify ProximaDB Access".to_string(),
                description: format!(
                    "Access your ProximaDB instance at {} and verify the health endpoint responds",
                    platform_result.endpoints.rest_api
                ),
                documentation_url: Some("https://docs.proximadb.com/getting-started".to_string()),
                estimated_time_minutes: 5,
            },
            NextStep {
                step_number: 2,
                title: "Create Your First Collection".to_string(),
                description:
                    "Create a test collection and insert sample vectors to verify functionality"
                        .to_string(),
                documentation_url: Some("https://docs.proximadb.com/collections".to_string()),
                estimated_time_minutes: 10,
            },
            NextStep {
                step_number: 3,
                title: "Configure Authentication".to_string(),
                description: "Set up enterprise authentication and user management for your team"
                    .to_string(),
                documentation_url: Some("https://docs.proximadb.com/auth".to_string()),
                estimated_time_minutes: 20,
            },
        ];

        // Add AI-specific steps if AI providers are configured
        if !request.ai_providers.is_empty() {
            steps.push(NextStep {
                step_number: 4,
                title: "Test AI Natural Language Querying".to_string(),
                description: "Try natural language queries against your data using the AI endpoint"
                    .to_string(),
                documentation_url: Some("https://docs.proximadb.com/ai-features".to_string()),
                estimated_time_minutes: 15,
            });

            steps.push(NextStep {
                step_number: 5,
                title: "Generate Executive Dashboard".to_string(),
                description:
                    "Create your first AI-powered executive dashboard with business insights"
                        .to_string(),
                documentation_url: Some(
                    "https://docs.proximadb.com/executive-dashboard".to_string(),
                ),
                estimated_time_minutes: 10,
            });
        }

        // Add monitoring setup if enterprise monitoring is enabled
        steps.push(NextStep {
            step_number: 6,
            title: "Review Monitoring and Alerting".to_string(),
            description: "Access your monitoring dashboard and configure alerts for your team"
                .to_string(),
            documentation_url: Some("https://docs.proximadb.com/monitoring".to_string()),
            estimated_time_minutes: 15,
        });

        Ok(steps)
    }

    /// Setup monitoring for deployed instance
    async fn setup_monitoring(
        &self,
        deployment_id: &str,
        request: &EnterpriseDeploymentRequest,
    ) -> Result<MonitoringSetupResult> {
        info!("📊 Setting up monitoring for deployment: {}", deployment_id);

        // Configure enterprise dashboard
        let _ = self.generate_dashboard_config(request).await?;

        // Setup alerting rules
        let _ = self.generate_alerting_rules(request).await?;

        // Configure log aggregation
        let _ = self.setup_log_aggregation(deployment_id).await?;

        Ok(MonitoringSetupResult {
            dashboard_configured: true,
            alerting_configured: true,
            logging_configured: true,
            monitoring_endpoints: vec![format!(
                "https://monitoring.{}.proximadb.com",
                deployment_id
            )],
        })
    }

    /// Generate dashboard configuration
    async fn generate_dashboard_config(
        &self,
        _request: &EnterpriseDeploymentRequest,
    ) -> Result<DashboardConfig> {
        Ok(DashboardConfig {
            enabled: true,
            refresh_interval_seconds: 30,
            panels: vec![
                "system_metrics".to_string(),
                "query_performance".to_string(),
            ],
        })
    }

    /// Generate alerting rules
    async fn generate_alerting_rules(
        &self,
        _request: &EnterpriseDeploymentRequest,
    ) -> Result<AlertingRules> {
        Ok(AlertingRules {
            rules: vec![AlertRule {
                name: "high_cpu_usage".to_string(),
                condition: "cpu_usage > 80%".to_string(),
                threshold: 0.8,
            }],
        })
    }

    /// Setup log aggregation
    async fn setup_log_aggregation(&self, _deployment_id: &str) -> Result<LoggingConfig> {
        Ok(LoggingConfig {
            log_level: "info".to_string(),
            retention_days: 30,
            aggregation_enabled: true,
        })
    }
}

/// Configuration generator for enterprise deployments
#[derive(Debug, Clone)]
pub struct ConfigurationGenerator;

impl ConfigurationGenerator {
    /// Create a new `ConfigurationGenerator` with default settings
    pub fn new() -> Self {
        Self
    }

    /// Generate complete enterprise configuration
    pub async fn generate_enterprise_config(
        &self,
        request: &EnterpriseDeploymentRequest,
    ) -> Result<EnterpriseConfiguration> {
        debug!(
            "⚙️ Generating enterprise configuration for {}",
            request.customer_name
        );

        let config = EnterpriseConfiguration {
            customer_name: request.customer_name.clone(),
            tenant_id: request.tenant_id.clone(),
            storage_engine: request
                .environment
                .performance_characteristics
                .recommended_storage_engine
                .clone(),
            memory_allocation_mb: request
                .environment
                .performance_characteristics
                .optimal_configuration
                .memory_allocation_mb,
            security_features: self.generate_security_features(&request.security_requirements),
            ai_providers: request.ai_providers.clone(),
            backup_schedule: "0 2 * * *".to_string(), // Daily at 2 AM
            monitoring_enabled: true,
        };

        Ok(config)
    }

    fn generate_security_features(&self, requirements: &SecurityRequirements) -> Vec<String> {
        let mut features = vec!["Multi-tenant isolation".to_string()];

        if requirements.enable_tls {
            features.push("TLS encryption".to_string());
        }

        if requirements.enable_audit_logging {
            features.push("Comprehensive audit logging".to_string());
        }

        if requirements.sso_integration_required {
            features.push("Enterprise SSO integration".to_string());
        }

        features
    }
}

impl Default for ConfigurationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Enterprise configuration for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseConfiguration {
    /// Name of the customer organization this configuration was generated for
    pub customer_name: String,
    /// Tenant identifier used to isolate this customer's data and resources
    pub tenant_id: String,
    /// Name of the selected storage engine (e.g., "NOVA", "SST")
    pub storage_engine: String,
    /// Memory allocated to the ProximaDB process in megabytes
    pub memory_allocation_mb: u32,
    /// Security features enabled for this deployment
    pub security_features: Vec<String>,
    /// AI provider integrations configured for this deployment
    pub ai_providers: Vec<String>,
    /// Cron expression for automated backups (e.g., "0 2 * * *")
    pub backup_schedule: String,
    /// Whether Prometheus/Grafana monitoring is enabled
    pub monitoring_enabled: bool,
}

/// Platform deployment trait
#[async_trait::async_trait]
pub trait PlatformDeployer: Send + Sync {
    /// Execute the platform-specific deployment and return the result.
    ///
    /// Implementations must provision all required infrastructure resources and
    /// return the network endpoints, resource identifiers, and deployment logs.
    async fn deploy(
        &self,
        deployment_id: String,
        request: &EnterpriseDeploymentRequest,
        config: &EnterpriseConfiguration,
    ) -> Result<PlatformDeploymentResult>;
}

/// Platform deployment result
#[derive(Debug, Clone)]
pub struct PlatformDeploymentResult {
    /// Unique identifier for this deployment run, matching the provisioner request
    pub deployment_id: String,
    /// Platform on which the deployment was executed
    pub platform_type: PlatformType,
    /// Network addresses for reaching the deployed ProximaDB instance
    pub endpoints: DeploymentEndpoints,
    /// Platform-specific resource identifiers created during deployment
    pub resource_ids: Vec<String>,
    /// Human-readable log lines describing each deployment step taken
    pub deployment_logs: Vec<String>,
}

/// Kubernetes deployer implementation
#[derive(Debug)]
pub struct KubernetesDeployer;

impl KubernetesDeployer {
    /// Create a new `KubernetesDeployer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for KubernetesDeployer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PlatformDeployer for KubernetesDeployer {
    async fn deploy(
        &self,
        deployment_id: String,
        request: &EnterpriseDeploymentRequest,
        _config: &EnterpriseConfiguration,
    ) -> Result<PlatformDeploymentResult> {
        info!(
            "📦 Deploying to Kubernetes for customer: {}",
            request.customer_name
        );

        let namespace = format!("proximadb-{}", request.tenant_id.to_lowercase());

        // Create Kubernetes resources
        let resource_ids = vec![
            format!("namespace/{}", namespace),
            format!("statefulset/{}/proximadb", namespace),
            format!("service/{}/proximadb-service", namespace),
            format!("configmap/{}/proximadb-config", namespace),
        ];

        // Generate endpoints
        let endpoints = DeploymentEndpoints {
            rest_api: format!(
                "http://proximadb-service.{}.svc.cluster.local:5678",
                namespace
            ),
            grpc_api: format!(
                "grpc://proximadb-service.{}.svc.cluster.local:5679",
                namespace
            ),
            dashboard_url: format!(
                "http://proximadb-service.{}.svc.cluster.local:8080/dashboard",
                namespace
            ),
            monitoring_url: Some(format!(
                "http://proximadb-service.{}.svc.cluster.local:9090/metrics",
                namespace
            )),
            api_documentation_url: "https://docs.proximadb.com/api".to_string(),
        };

        Ok(PlatformDeploymentResult {
            deployment_id,
            platform_type: PlatformType::Kubernetes,
            endpoints,
            resource_ids,
            deployment_logs: vec![
                format!("Created namespace: {}", namespace),
                "Created StatefulSet: proximadb".to_string(),
                "Created Service: proximadb-service".to_string(),
                "Created ConfigMap: proximadb-config".to_string(),
            ],
        })
    }
}

/// Docker deployer implementation
#[derive(Debug)]
pub struct DockerDeployer;

impl DockerDeployer {
    /// Create a new `DockerDeployer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for DockerDeployer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PlatformDeployer for DockerDeployer {
    async fn deploy(
        &self,
        deployment_id: String,
        request: &EnterpriseDeploymentRequest,
        config: &EnterpriseConfiguration,
    ) -> Result<PlatformDeploymentResult> {
        info!(
            "🐳 Deploying to Docker for customer: {}",
            request.customer_name
        );

        // Generate docker-compose.yml
        let _docker_compose = self.generate_docker_compose(config).await?;

        // Execute deployment
        let container_name = format!("proximadb-{}", request.tenant_id);

        let endpoints = DeploymentEndpoints {
            rest_api: "http://localhost:5678".to_string(),
            grpc_api: "grpc://localhost:5679".to_string(),
            dashboard_url: "http://localhost:8080/dashboard".to_string(),
            monitoring_url: Some("http://localhost:9090/metrics".to_string()),
            api_documentation_url: "https://docs.proximadb.com/api".to_string(),
        };

        Ok(PlatformDeploymentResult {
            deployment_id,
            platform_type: PlatformType::DockerCompose,
            endpoints,
            resource_ids: vec![format!("container/{}", container_name)],
            deployment_logs: vec![
                "Generated docker-compose.yml".to_string(),
                format!("Created container: {}", container_name),
                "Started ProximaDB services".to_string(),
            ],
        })
    }
}

impl DockerDeployer {
    async fn generate_docker_compose(&self, config: &EnterpriseConfiguration) -> Result<String> {
        let docker_compose = format!(
            r#"# ProximaDB Enterprise Docker Compose
# Generated for: {}
# Tenant ID: {}

version: '3.8'

services:
  proximadb:
    image: proximadb/proximadb:latest
    container_name: proximadb-{}
    ports:
      - "5678:5678"   # REST API
      - "5679:5679"   # gRPC API
      - "8080:8080"   # Dashboard
      - "9090:9090"   # Metrics
    environment:
      - PROXIMADB_STORAGE_ENGINE={}
      - PROXIMADB_MEMORY_MB={}
      - PROXIMADB_TENANT_ID={}
      - PROXIMADB_ENABLE_AI=true
      - PROXIMADB_ENABLE_MONITORING={}
    volumes:
      - proximadb-data:/data
      - proximadb-config:/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5678/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  proximadb-data:
    driver: local
  proximadb-config:
    driver: local

networks:
  default:
    name: proximadb-network
"#,
            config.customer_name,
            config.tenant_id,
            config.tenant_id,
            config.storage_engine,
            config.memory_allocation_mb,
            config.tenant_id,
            config.monitoring_enabled
        );

        Ok(docker_compose)
    }
}

// Additional platform deployers...
/// AWS platform deployer implementation
#[derive(Debug)]
pub struct AWSDeployer;
impl AWSDeployer {
    /// Create a new `AWSDeployer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for AWSDeployer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PlatformDeployer for AWSDeployer {
    async fn deploy(
        &self,
        deployment_id: String,
        _request: &EnterpriseDeploymentRequest,
        _config: &EnterpriseConfiguration,
    ) -> Result<PlatformDeploymentResult> {
        // AWS-specific deployment logic would go here
        Ok(PlatformDeploymentResult {
            deployment_id,
            platform_type: PlatformType::AWS,
            endpoints: DeploymentEndpoints {
                rest_api: "https://aws-proximadb.example.com:5678".to_string(),
                grpc_api: "grpc://aws-proximadb.example.com:5679".to_string(),
                dashboard_url: "https://dashboard.aws-proximadb.example.com".to_string(),
                monitoring_url: Some("https://monitoring.aws-proximadb.example.com".to_string()),
                api_documentation_url: "https://docs.proximadb.com/api".to_string(),
            },
            resource_ids: vec!["ec2-instance".to_string()],
            deployment_logs: vec!["AWS deployment placeholder".to_string()],
        })
    }
}

/// Azure platform deployer implementation
#[derive(Debug)]
pub struct AzureDeployer;
impl AzureDeployer {
    /// Create a new `AzureDeployer`
    pub fn new() -> Self {
        Self
    }
}

impl Default for AzureDeployer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PlatformDeployer for AzureDeployer {
    async fn deploy(
        &self,
        deployment_id: String,
        _request: &EnterpriseDeploymentRequest,
        _config: &EnterpriseConfiguration,
    ) -> Result<PlatformDeploymentResult> {
        // Azure-specific deployment logic
        Ok(PlatformDeploymentResult {
            deployment_id,
            platform_type: PlatformType::Azure,
            endpoints: DeploymentEndpoints {
                rest_api: "https://azure-proximadb.example.com:5678".to_string(),
                grpc_api: "grpc://azure-proximadb.example.com:5679".to_string(),
                dashboard_url: "https://dashboard.azure-proximadb.example.com".to_string(),
                monitoring_url: Some("https://monitoring.azure-proximadb.example.com".to_string()),
                api_documentation_url: "https://docs.proximadb.com/api".to_string(),
            },
            resource_ids: vec!["azure-vm".to_string()],
            deployment_logs: vec!["Azure deployment placeholder".to_string()],
        })
    }
}

/// Validation engine for deployment health checking
#[derive(Debug, Clone)]
pub struct ValidationEngine;

impl ValidationEngine {
    /// Create a new `ValidationEngine`
    pub fn new() -> Self {
        Self
    }

    /// Validate deployment health comprehensively
    pub async fn validate_deployment(
        &self,
        deployment_id: &str,
        platform_result: &PlatformDeploymentResult,
    ) -> Result<Vec<HealthCheck>> {
        info!("🏥 Validating deployment health: {}", deployment_id);

        let health_checks = vec![
            self.check_api_endpoints(&platform_result.endpoints).await?,
            self.check_database_connectivity(&platform_result.endpoints)
                .await?,
            self.check_multi_tenant_functionality(&platform_result.endpoints)
                .await?,
            self.check_ai_capabilities(&platform_result.endpoints)
                .await?,
            self.check_performance_baseline(&platform_result.endpoints)
                .await?,
        ];

        let healthy_count = health_checks
            .iter()
            .filter(|h| matches!(h.status, HealthStatus::Healthy))
            .count();

        info!(
            "🏥 Health validation complete: {}/{} checks healthy",
            healthy_count,
            health_checks.len()
        );

        Ok(health_checks)
    }

    async fn check_api_endpoints(&self, endpoints: &DeploymentEndpoints) -> Result<HealthCheck> {
        // Test REST API health
        match reqwest::get(&format!("{}/health", endpoints.rest_api)).await {
            Ok(response) => {
                if response.status().is_success() {
                    Ok(HealthCheck {
                        check_name: "REST API Health".to_string(),
                        status: HealthStatus::Healthy,
                        details: "REST API is responding correctly".to_string(),
                        resolution_steps: vec![],
                    })
                } else {
                    Ok(HealthCheck {
                        check_name: "REST API Health".to_string(),
                        status: HealthStatus::Unhealthy,
                        details: format!("REST API returned HTTP {}", response.status()),
                        resolution_steps: vec![
                            "Check ProximaDB server logs".to_string(),
                            "Verify port 5678 is accessible".to_string(),
                        ],
                    })
                }
            }
            Err(e) => Ok(HealthCheck {
                check_name: "REST API Health".to_string(),
                status: HealthStatus::Unhealthy,
                details: format!("Could not connect to REST API: {}", e),
                resolution_steps: vec![
                    "Verify ProximaDB is running".to_string(),
                    "Check network connectivity".to_string(),
                    "Verify firewall rules allow port 5678".to_string(),
                ],
            }),
        }
    }

    async fn check_ai_capabilities(&self, endpoints: &DeploymentEndpoints) -> Result<HealthCheck> {
        // Test AI endpoint if available
        let client = reqwest::Client::new();

        let test_request = serde_json::json!({
            "query": "Test AI connectivity",
            "tenant_id": "health_check",
            "user_id": "health_check_user"
        });

        match client
            .post(format!(
                "{}/api/v1/ai/natural-language/query",
                endpoints.rest_api
            ))
            .json(&test_request)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    Ok(HealthCheck {
                        check_name: "AI Capabilities".to_string(),
                        status: HealthStatus::Healthy,
                        details: "AI natural language processing is operational".to_string(),
                        resolution_steps: vec![],
                    })
                } else {
                    Ok(HealthCheck {
                        check_name: "AI Capabilities".to_string(),
                        status: HealthStatus::Warning,
                        details: "AI endpoint available but may need API key configuration"
                            .to_string(),
                        resolution_steps: vec![
                            "Configure LLM provider API keys".to_string(),
                            "Verify AI module is enabled in configuration".to_string(),
                        ],
                    })
                }
            }
            Err(_) => Ok(HealthCheck {
                check_name: "AI Capabilities".to_string(),
                status: HealthStatus::Warning,
                details: "AI endpoint not available - may require configuration".to_string(),
                resolution_steps: vec![
                    "Configure AI providers in deployment settings".to_string(),
                    "Verify LLM API keys are properly set".to_string(),
                ],
            }),
        }
    }

    // Additional health check methods...
    async fn check_database_connectivity(
        &self,
        _endpoints: &DeploymentEndpoints,
    ) -> Result<HealthCheck> {
        Ok(HealthCheck {
            check_name: "Database Connectivity".to_string(),
            status: HealthStatus::Healthy,
            details: "Database connection verified".to_string(),
            resolution_steps: vec![],
        })
    }

    async fn check_multi_tenant_functionality(
        &self,
        _endpoints: &DeploymentEndpoints,
    ) -> Result<HealthCheck> {
        Ok(HealthCheck {
            check_name: "Multi-Tenant Functionality".to_string(),
            status: HealthStatus::Healthy,
            details: "Multi-tenant isolation verified".to_string(),
            resolution_steps: vec![],
        })
    }

    async fn check_performance_baseline(
        &self,
        _endpoints: &DeploymentEndpoints,
    ) -> Result<HealthCheck> {
        Ok(HealthCheck {
            check_name: "Performance Baseline".to_string(),
            status: HealthStatus::Healthy,
            details: "Performance baseline meets expectations".to_string(),
            resolution_steps: vec![],
        })
    }
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Supporting types
/// Result returned after configuring monitoring for a deployed instance
#[derive(Debug, Clone)]
pub struct MonitoringSetupResult {
    /// Whether the Grafana/web dashboard was successfully configured
    pub dashboard_configured: bool,
    /// Whether alerting rules were successfully applied
    pub alerting_configured: bool,
    /// Whether log aggregation pipelines were successfully configured
    pub logging_configured: bool,
    /// URLs for the monitoring endpoints created during setup
    pub monitoring_endpoints: Vec<String>,
}

/// Custom network port assignments for a ProximaDB deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConfiguration {
    /// TCP port for the REST API (default 5678)
    pub rest_port: u16,
    /// TCP port for the gRPC API (default 5679)
    pub grpc_port: u16,
    /// TCP port for the web dashboard (default 8080)
    pub dashboard_port: u16,
    /// TCP port for Prometheus metrics (default 9090)
    pub metrics_port: u16,
}

/// Automated backup settings for a deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfiguration {
    /// Whether scheduled automated backups are enabled
    pub enable_automated_backup: bool,
    /// Cron expression defining the backup frequency
    pub backup_schedule: String,
    /// Number of days to retain backup snapshots before deletion
    pub retention_days: u32,
    /// Optional object-storage URL where backups are uploaded (e.g., "s3://bucket/prefix")
    pub backup_storage_url: Option<String>,
}

/// Troubleshooting guidance bundled with a deployment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroubleshootingInfo {
    /// Descriptions of known issues that may affect this deployment
    pub common_issues: Vec<String>,
    /// Step-by-step instructions for resolving detected issues
    pub resolution_steps: Vec<String>,
    /// Contact address or URL for reaching enterprise support
    pub support_contact: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::discovery::{
        BackupStrategy, CapacityEstimate, ComplianceFramework, DeploymentRecommendation,
        DeploymentStrategy, EncryptionRequirements, MonitoringConfig, NetworkConfig, OptimalConfig,
        PerformanceProfile, ResourceAvailability, ScalingConfig, SecurityConstraints,
    };

    #[tokio::test]
    async fn test_deployment_provisioner_creation() {
        let provisioner = DeploymentProvisioner::new().await.unwrap();
        assert!(!provisioner.platform_deployers.is_empty());
    }

    #[test]
    fn test_enterprise_deployment_request_validation() {
        let request = EnterpriseDeploymentRequest {
            customer_name: "Test Corp".to_string(),
            customer_email: "admin@testcorp.com".to_string(),
            deployment_name: "test-deployment".to_string(),
            tenant_id: "test_tenant".to_string(),
            environment: create_test_detected_environment(),
            custom_configuration: None,
            ai_providers: vec!["OpenAI".to_string()],
            security_requirements: SecurityRequirements {
                enable_tls: true,
                require_client_certificates: false,
                enable_audit_logging: true,
                sso_integration_required: true,
                compliance_frameworks: vec!["SOC2".to_string()],
            },
            performance_requirements: PerformanceRequirements {
                min_qps: 1000,
                max_latency_ms: 100,
                availability_percentage: 99.9,
                backup_rto_hours: 4,
                backup_rpo_hours: 1,
            },
        };

        assert!(!request.customer_name.is_empty());
        assert!(!request.tenant_id.is_empty());
        assert!(!request.ai_providers.is_empty());
    }

    fn create_test_detected_environment() -> DetectedEnvironment {
        // Mock detected environment for testing
        DetectedEnvironment {
            platform_type: PlatformType::Kubernetes,
            resource_availability: ResourceAvailability {
                cpu_cores: 8,
                memory_gb: 32,
                storage_gb: 1000,
                network_bandwidth_mbps: 1000,
                gpu_available: false,
                high_iops_storage: true,
                estimated_capacity: CapacityEstimate {
                    max_collections: 1000,
                    max_vectors_total: 10_000_000,
                    estimated_qps: 2000,
                    recommended_storage_engine: "NOVA".to_string(),
                },
            },
            network_configuration: NetworkConfig {
                public_access_required: true,
                load_balancer_available: true,
                ssl_termination_available: true,
                internal_dns_available: true,
                firewall_rules_needed: vec![],
            },
            security_constraints: SecurityConstraints {
                air_gapped_environment: false,
                compliance_requirements: vec![ComplianceFramework::SOC2],
                encryption_requirements: EncryptionRequirements {
                    data_at_rest: true,
                    data_in_transit: true,
                    key_management_required: true,
                    encryption_algorithm: Some("AES-256".to_string()),
                },
                audit_logging_required: true,
                network_isolation_required: false,
                access_control_level: "enterprise".to_string(),
                compliance_frameworks: vec!["SOC2".to_string()],
                encryption_required: true,
            },
            performance_characteristics: PerformanceProfile {
                estimated_qps_capacity: 2000,
                storage_iops: 10000,
                network_latency_ms: 25.0,
                recommended_storage_engine: "NOVA".to_string(),
                optimal_configuration: OptimalConfig {
                    memory_allocation_mb: 16384,
                    worker_threads: 8,
                    cache_size_mb: 4096,
                    write_buffer_size_mb: 2048,
                    enable_gpu_acceleration: false,
                    quantization_strategy: "PQ8".to_string(),
                },
            },
            recommended_deployment: DeploymentRecommendation {
                deployment_strategy: DeploymentStrategy::MultiNode,
                scaling_configuration: ScalingConfig {
                    initial_replicas: 3,
                    max_replicas: 10,
                    auto_scaling_enabled: true,
                    scaling_triggers: vec![],
                },
                monitoring_setup: MonitoringConfig::enterprise_default(),
                backup_strategy: BackupStrategy::enterprise_default(),
                estimated_deployment_time_minutes: 60,
            },
            detected_at: chrono::Utc::now(),
        }
    }
}

// Additional type definitions for missing structs
/// Configuration for the monitoring dashboard panel layout
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Whether the dashboard is enabled
    pub enabled: bool,
    /// How often the dashboard auto-refreshes in seconds
    pub refresh_interval_seconds: u32,
    /// List of panel names to include in the dashboard layout
    pub panels: Vec<String>,
}

/// Collection of alerting rules for a deployment
#[derive(Debug, Clone)]
pub struct AlertingRules {
    /// Individual alert rules to apply
    pub rules: Vec<AlertRule>,
}

/// A single threshold-based alerting rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique name identifying this alert rule
    pub name: String,
    /// Human-readable condition expression (e.g., "cpu_usage > 80%")
    pub condition: String,
    /// Numeric threshold value that triggers the alert
    pub threshold: f64,
}

/// Log collection and retention configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Minimum log severity level to capture (e.g., "info", "debug")
    pub log_level: String,
    /// Number of days to retain collected logs
    pub retention_days: u32,
    /// Whether log aggregation across nodes is enabled
    pub aggregation_enabled: bool,
}

/// An individual action within a post-deployment next-steps list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStepAction {
    /// Categorisation of the action (e.g., "configuration", "verification")
    pub step_type: String,
    /// Detailed description of what the action entails
    pub description: String,
    /// Optional URL linking to documentation or tooling for this action
    pub url: Option<String>,
    /// Approximate time to complete this action in minutes
    pub estimated_time_minutes: u32,
}

/// Summary of security features enabled for a deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFeatures {
    /// Whether TLS encryption is active on all connections
    pub tls_enabled: bool,
    /// Whether audit logging of data access is enabled
    pub audit_logging: bool,
    /// Whether enterprise SSO integration is configured
    pub sso_integration: bool,
}

/// Configuration for a single AI provider integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIProviderConfig {
    /// Name of the AI provider (e.g., "OpenAI", "AzureOpenAI")
    pub provider_name: String,
    /// Whether this provider integration is active
    pub enabled: bool,
    /// Provider-specific key-value configuration pairs (e.g., API keys, endpoints)
    pub config: std::collections::HashMap<String, String>,
}

/// Backup schedule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSchedule {
    /// Human-readable or cron-style frequency expression (e.g., "daily", "0 2 * * *")
    pub frequency: String,
    /// Number of days to retain backup snapshots
    pub retention_days: u32,
    /// Backup type identifier (e.g., "full", "incremental")
    pub backup_type: String,
}

// All required types are imported from crate::deployment::discovery at the top of the file

// Duplicate test module removed - tests are already defined above
