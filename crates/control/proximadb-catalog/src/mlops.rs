//! Typed xCatalog contracts for model registry and embedding execution metadata.
//!
//! The catalog stores small, versioned metadata and content descriptors. Model
//! weights, tokenizer files, datasets, and evaluation payloads remain in object
//! or external artifact storage and are addressed by immutable digests.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Current serialized model-contract schema. New readers default legacy rows to
/// no MLOps facet; future incompatible shapes require a new version.
pub const MODEL_CONTRACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CatalogModelContractError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} must be a sha256 digest in the form sha256:<64 lowercase hex chars>")]
    InvalidDigest { field: &'static str },
    #[error("{field} must contain exactly one {{text}} placeholder")]
    InvalidTemplate { field: &'static str },
    #[error(
        "special token count {special_tokens} must be smaller than context limit {context_limit}"
    )]
    InvalidSpecialTokenBudget {
        special_tokens: u32,
        context_limit: u32,
    },
    #[error("dimension {dimension} exceeds native dimension {native_dimension}")]
    DimensionExceedsNative {
        dimension: u32,
        native_dimension: u32,
    },
    #[error("invalid output dimension policy: {reason}")]
    InvalidDimensionPolicy { reason: String },
    #[error("model version {version} is already registered and immutable")]
    DuplicateVersion { version: u64 },
    #[error("model version {version} is not registered")]
    UnknownVersion { version: u64 },
    #[error("alias '{alias}' is invalid")]
    InvalidAlias { alias: String },
    #[error("alias '{alias}' is not registered")]
    UnknownAlias { alias: String },
    #[error("evidence '{evidence_id}' is already recorded and append-only")]
    DuplicateEvidence { evidence_id: String },
    #[error("evidence '{evidence_id}' is not registered for model version {version}")]
    UnknownEvidence { evidence_id: String, version: u64 },
    #[error("decision '{decision_id}' is already recorded and append-only")]
    DuplicateDecision { decision_id: String },
    #[error("deployment '{deployment}' digest does not match immutable model version {version}")]
    DeploymentDigestMismatch { deployment: String, version: u64 },
    #[error("runtime '{runtime}' is not approved for model version {version}")]
    RuntimeNotApproved { runtime: String, version: u64 },
    #[error("model version {version} contract digest does not match the pinned binding")]
    ContractDigestMismatch { version: u64 },
    #[error("model version {version} does not support output dimension {dimension}")]
    UnsupportedOutputDimension { version: u64, dimension: u32 },
    #[error("model version {version} is not approved for use")]
    ModelNotApproved { version: u64 },
    #[error("model version {version} is rejected and cannot be used")]
    ModelRejected { version: u64 },
    #[error("model version {version} is deprecated and cannot be used")]
    ModelDeprecated { version: u64 },
    #[error("model registry revision conflict: expected {expected}, current {current}")]
    RevisionConflict { expected: u64, current: u64 },
    #[error("contract serialization failed: {message}")]
    Serialization { message: String },
}

fn require_non_empty(value: &str, field: &'static str) -> Result<(), CatalogModelContractError> {
    if value.trim().is_empty() {
        return Err(CatalogModelContractError::Empty { field });
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), CatalogModelContractError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(CatalogModelContractError::InvalidDigest { field });
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CatalogModelContractError::InvalidDigest { field });
    }
    Ok(())
}

fn validate_template(value: &str, field: &'static str) -> Result<(), CatalogModelContractError> {
    if value.match_indices("{text}").count() != 1 {
        return Err(CatalogModelContractError::InvalidTemplate { field });
    }
    Ok(())
}

/// Content-addressed reference to bytes held outside xCatalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogArtifactDescriptor {
    pub uri: String,
    pub digest: String,
    pub size_bytes: u64,
    pub media_type: String,
}

impl CatalogArtifactDescriptor {
    pub fn new(
        uri: impl Into<String>,
        digest: impl Into<String>,
        size_bytes: u64,
        media_type: impl Into<String>,
    ) -> Result<Self, CatalogModelContractError> {
        let descriptor = Self {
            uri: uri.into(),
            digest: digest.into(),
            size_bytes,
            media_type: media_type.into(),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.uri, "artifact uri")?;
        validate_sha256(&self.digest, "artifact digest")?;
        require_non_empty(&self.media_type, "artifact media type")?;
        Ok(())
    }
}

/// Exact rendered-input and tokenizer budget consumed by an embedding runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogEmbeddingInputContract {
    pub model_revision: String,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub tokenizer_fingerprint: String,
    pub declared_context_limit: u32,
    pub effective_context_limit: u32,
    pub special_token_count: u32,
    pub document_template: String,
    pub query_template: String,
    #[serde(default)]
    pub document_parameters: BTreeMap<String, String>,
    #[serde(default)]
    pub query_parameters: BTreeMap<String, String>,
}

impl CatalogEmbeddingInputContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_revision: impl Into<String>,
        tokenizer_id: impl Into<String>,
        tokenizer_revision: impl Into<String>,
        tokenizer_fingerprint: impl Into<String>,
        context_limit: u32,
        special_token_count: u32,
        document_template: impl Into<String>,
        query_template: impl Into<String>,
    ) -> Result<Self, CatalogModelContractError> {
        let contract = Self {
            model_revision: model_revision.into(),
            tokenizer_id: tokenizer_id.into(),
            tokenizer_revision: tokenizer_revision.into(),
            tokenizer_fingerprint: tokenizer_fingerprint.into(),
            declared_context_limit: context_limit,
            effective_context_limit: context_limit,
            special_token_count,
            document_template: document_template.into(),
            query_template: query_template.into(),
            document_parameters: BTreeMap::new(),
            query_parameters: BTreeMap::new(),
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Intersect the model declaration with a tokenizer/runtime limit.
    pub fn with_runtime_context_limit(
        mut self,
        runtime_limit: u32,
    ) -> Result<Self, CatalogModelContractError> {
        self.effective_context_limit = self.declared_context_limit.min(runtime_limit);
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.model_revision, "model revision")?;
        require_non_empty(&self.tokenizer_id, "tokenizer id")?;
        require_non_empty(&self.tokenizer_revision, "tokenizer revision")?;
        validate_sha256(&self.tokenizer_fingerprint, "tokenizer fingerprint")?;
        if self.declared_context_limit == 0 || self.effective_context_limit == 0 {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "context limits must be positive".to_string(),
            });
        }
        if self.effective_context_limit > self.declared_context_limit {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "effective context cannot exceed the declared context".to_string(),
            });
        }
        if self.special_token_count >= self.effective_context_limit {
            return Err(CatalogModelContractError::InvalidSpecialTokenBudget {
                special_tokens: self.special_token_count,
                context_limit: self.effective_context_limit,
            });
        }
        validate_template(&self.document_template, "document template")?;
        validate_template(&self.query_template, "query template")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum CatalogModelAccess {
    Open,
    Gated,
    /// Fail-closed default for a legacy or incomplete registration.
    #[default]
    Unreviewed,
}

/// Supply-chain and runtime policy declared for an immutable model version.
/// Approval decisions remain separate append-only audit records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogModelGovernance {
    #[serde(default = "unknown_license")]
    pub license_id: String,
    #[serde(default)]
    pub access: CatalogModelAccess,
    #[serde(default)]
    pub requires_remote_code: bool,
    #[serde(default)]
    pub approved_runtimes: BTreeSet<String>,
}

fn unknown_license() -> String {
    "unknown".to_string()
}

impl Default for CatalogModelGovernance {
    fn default() -> Self {
        Self {
            license_id: unknown_license(),
            access: CatalogModelAccess::Unreviewed,
            requires_remote_code: false,
            approved_runtimes: BTreeSet::new(),
        }
    }
}

impl CatalogModelGovernance {
    pub fn new(
        license_id: impl Into<String>,
        access: CatalogModelAccess,
        requires_remote_code: bool,
        approved_runtimes: impl IntoIterator<Item = String>,
    ) -> Result<Self, CatalogModelContractError> {
        let policy = Self {
            license_id: license_id.into(),
            access,
            requires_remote_code,
            approved_runtimes: approved_runtimes.into_iter().collect(),
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.license_id, "license id")?;
        if self
            .approved_runtimes
            .iter()
            .any(|item| item.trim().is_empty())
        {
            return Err(CatalogModelContractError::Empty {
                field: "approved runtime",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum CatalogLineageInputKind {
    Dataset,
    FeatureSet,
    Model,
    Artifact,
}

/// Digest-pinned input consumed by the execution that produced a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogLineageInput {
    pub kind: CatalogLineageInputKind,
    pub name: String,
    pub digest: String,
}

impl CatalogLineageInput {
    pub fn new(
        kind: CatalogLineageInputKind,
        name: impl Into<String>,
        digest: impl Into<String>,
    ) -> Result<Self, CatalogModelContractError> {
        let input = Self {
            kind,
            name: name.into(),
            digest: digest.into(),
        };
        require_non_empty(&input.name, "lineage input name")?;
        validate_sha256(&input.digest, "lineage input digest")?;
        Ok(input)
    }
}

/// MLMD/OpenLineage-shaped producer execution and its declared inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogModelLineage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_execution_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_revision: Option<String>,
    #[serde(default)]
    pub inputs: Vec<CatalogLineageInput>,
}

impl CatalogModelLineage {
    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        if self
            .producer_execution_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CatalogModelContractError::Empty {
                field: "producer execution id",
            });
        }
        for input in &self.inputs {
            require_non_empty(&input.name, "lineage input name")?;
            validate_sha256(&input.digest, "lineage input digest")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogDimensionRange {
    pub minimum: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum CatalogDimensionPolicy {
    Fixed,
    Discrete,
    Range(CatalogDimensionRange),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogEmbeddingOutputContract {
    pub native_dimension: u32,
    pub dimension_policy: CatalogDimensionPolicy,
    pub supported_dimensions: Vec<u32>,
    pub normalized: bool,
    pub pooling: String,
}

impl CatalogEmbeddingOutputContract {
    pub fn new(
        native_dimension: u32,
        dimension_policy: CatalogDimensionPolicy,
        mut supported_dimensions: Vec<u32>,
        normalized: bool,
        pooling: impl Into<String>,
    ) -> Result<Self, CatalogModelContractError> {
        supported_dimensions.sort_unstable();
        supported_dimensions.dedup();
        let contract = Self {
            native_dimension,
            dimension_policy,
            supported_dimensions,
            normalized,
            pooling: pooling.into(),
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn supports(&self, dimension: u32) -> bool {
        match self.dimension_policy {
            CatalogDimensionPolicy::Fixed => dimension == self.native_dimension,
            CatalogDimensionPolicy::Discrete => self.supported_dimensions.contains(&dimension),
            CatalogDimensionPolicy::Range(range) => {
                (range.minimum..=self.native_dimension).contains(&dimension)
            }
        }
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        if self.native_dimension == 0 {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "native dimension must be positive".to_string(),
            });
        }
        require_non_empty(&self.pooling, "pooling")?;
        for dimension in &self.supported_dimensions {
            if *dimension == 0 {
                return Err(CatalogModelContractError::InvalidDimensionPolicy {
                    reason: "supported dimensions must be positive".to_string(),
                });
            }
            if *dimension > self.native_dimension {
                return Err(CatalogModelContractError::DimensionExceedsNative {
                    dimension: *dimension,
                    native_dimension: self.native_dimension,
                });
            }
        }
        match self.dimension_policy {
            CatalogDimensionPolicy::Fixed
                if self.supported_dimensions.as_slice() != [self.native_dimension] =>
            {
                Err(CatalogModelContractError::InvalidDimensionPolicy {
                    reason: "fixed policy must contain only the native dimension".to_string(),
                })
            }
            CatalogDimensionPolicy::Discrete
                if self.supported_dimensions.is_empty()
                    || !self.supported_dimensions.contains(&self.native_dimension) =>
            {
                Err(CatalogModelContractError::InvalidDimensionPolicy {
                    reason: "discrete policy must include the native dimension".to_string(),
                })
            }
            CatalogDimensionPolicy::Range(range)
                if range.minimum == 0 || range.minimum > self.native_dimension =>
            {
                Err(CatalogModelContractError::InvalidDimensionPolicy {
                    reason: "range minimum must be between one and native dimension".to_string(),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Immutable executable contract for one registered model version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogEmbeddingModelVersion {
    pub version: u64,
    pub provider_model_id: String,
    pub artifact: CatalogArtifactDescriptor,
    pub input: CatalogEmbeddingInputContract,
    pub output: CatalogEmbeddingOutputContract,
    #[serde(default)]
    pub governance: CatalogModelGovernance,
    #[serde(default)]
    pub lineage: CatalogModelLineage,
    pub created_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_run_id: Option<String>,
}

/// JSON object keys are strings, while the in-memory registry indexes versions
/// numerically. An explicit adapter is required because the registry may be
/// buffered inside an adjacently tagged enum (for example via
/// `serde_json::Value`), where serde_json's streaming map-key coercion is not
/// available and object-field order must not affect deserialization.
mod version_map_wire {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serializer, de::Error as _, ser::SerializeMap as _};

    use super::CatalogEmbeddingModelVersion;

    pub fn serialize<S>(
        versions: &BTreeMap<u64, CatalogEmbeddingModelVersion>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(versions.len()))?;
        for (version, contract) in versions {
            map.serialize_entry(&version.to_string(), contract)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<u64, CatalogEmbeddingModelVersion>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BTreeMap::<String, CatalogEmbeddingModelVersion>::deserialize(deserializer)?;
        let mut versions = BTreeMap::new();
        for (key, contract) in wire {
            let version = key.parse::<u64>().map_err(|_| {
                D::Error::custom(format!(
                    "model version key '{key}' is not an unsigned integer"
                ))
            })?;
            if versions.insert(version, contract).is_some() {
                return Err(D::Error::custom(format!(
                    "model version key '{key}' duplicates numeric version {version}"
                )));
            }
        }
        Ok(versions)
    }
}

impl CatalogEmbeddingModelVersion {
    pub fn new(
        version: u64,
        provider_model_id: impl Into<String>,
        artifact: CatalogArtifactDescriptor,
        input: CatalogEmbeddingInputContract,
        output: CatalogEmbeddingOutputContract,
        created_at_ms: i64,
    ) -> Result<Self, CatalogModelContractError> {
        let contract = Self {
            version,
            provider_model_id: provider_model_id.into(),
            artifact,
            input,
            output,
            governance: CatalogModelGovernance::default(),
            lineage: CatalogModelLineage::default(),
            created_at_ms,
            source_run_id: None,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        if self.version == 0 {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "model versions start at one".to_string(),
            });
        }
        require_non_empty(&self.provider_model_id, "provider model id")?;
        self.artifact.validate()?;
        self.input.validate()?;
        self.output.validate()?;
        self.governance.validate()?;
        self.lineage.validate()?;
        Ok(())
    }

    pub fn with_governance(mut self, governance: CatalogModelGovernance) -> Self {
        self.governance = governance;
        self
    }

    pub fn with_lineage(mut self, lineage: CatalogModelLineage) -> Self {
        self.lineage = lineage;
        self
    }

    /// Hash only executable semantics, excluding timestamps, aliases, evidence,
    /// approval, and deployment state.
    pub fn contract_sha256(&self) -> Result<String, CatalogModelContractError> {
        #[derive(Serialize)]
        struct ExecutableContract<'a> {
            schema_version: u32,
            provider_model_id: &'a str,
            artifact_digest: &'a str,
            input: &'a CatalogEmbeddingInputContract,
            output: &'a CatalogEmbeddingOutputContract,
        }
        let bytes = serde_json::to_vec(&ExecutableContract {
            schema_version: MODEL_CONTRACT_SCHEMA_VERSION,
            provider_model_id: &self.provider_model_id,
            artifact_digest: &self.artifact.digest,
            input: &self.input,
            output: &self.output,
        })
        .map_err(|error| CatalogModelContractError::Serialization {
            message: error.to_string(),
        })?;
        Ok(format!("sha256:{}", hex::encode(Sha256::digest(bytes))))
    }
}

/// Append-only evaluation summary. Large row-level results remain external.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogEvaluationEvidence {
    pub evidence_id: String,
    pub version: u64,
    pub dataset_name: String,
    pub dataset_digest: String,
    pub evaluator: String,
    pub metrics: BTreeMap<String, f64>,
    pub created_at_ms: i64,
}

impl CatalogEvaluationEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: impl Into<String>,
        version: u64,
        dataset_name: impl Into<String>,
        dataset_digest: impl Into<String>,
        evaluator: impl Into<String>,
        metrics: BTreeMap<String, f64>,
        created_at_ms: i64,
    ) -> Result<Self, CatalogModelContractError> {
        let evidence = Self {
            evidence_id: evidence_id.into(),
            version,
            dataset_name: dataset_name.into(),
            dataset_digest: dataset_digest.into(),
            evaluator: evaluator.into(),
            metrics,
            created_at_ms,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.evidence_id, "evidence id")?;
        require_non_empty(&self.dataset_name, "dataset name")?;
        validate_sha256(&self.dataset_digest, "dataset digest")?;
        require_non_empty(&self.evaluator, "evaluator")?;
        if self.metrics.is_empty() || self.metrics.values().any(|value| !value.is_finite()) {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "evaluation metrics must be non-empty and finite".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum CatalogModelDecisionKind {
    Approved,
    Rejected,
    Deprecated,
}

/// Fail-closed policy applied when resolving an immutable model snapshot for
/// a collection, embedding worker, or serving deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogModelUsePolicy {
    /// Runtime implementation that will execute the contract. `None` is valid
    /// for registration-time referential validation before a worker is chosen.
    pub runtime: Option<String>,
    /// Require the latest append-only decision for the version to be Approved.
    pub require_approved: bool,
}

impl CatalogModelUsePolicy {
    pub fn registration_only() -> Self {
        Self {
            runtime: None,
            require_approved: false,
        }
    }

    pub fn approved_for_runtime(runtime: impl Into<String>) -> Self {
        Self {
            runtime: Some(runtime.into()),
            require_approved: true,
        }
    }
}

/// Immutable snapshot returned by the durable catalog resolution seam.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogResolvedEmbeddingModel {
    pub asset_id: u64,
    pub registry_name: String,
    pub registry_revision: u64,
    pub contract_sha256: String,
    pub model: CatalogEmbeddingModelVersion,
}

/// Append-only policy/audit decision, intentionally separate from evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogModelDecision {
    pub decision_id: String,
    pub version: u64,
    pub decision: CatalogModelDecisionKind,
    pub evidence_ids: Vec<String>,
    pub principal: String,
    pub created_at_ms: i64,
}

impl CatalogModelDecision {
    pub fn new(
        decision_id: impl Into<String>,
        version: u64,
        decision: CatalogModelDecisionKind,
        evidence_ids: Vec<String>,
        principal: impl Into<String>,
        created_at_ms: i64,
    ) -> Result<Self, CatalogModelContractError> {
        let record = Self {
            decision_id: decision_id.into(),
            version,
            decision,
            evidence_ids,
            principal: principal.into(),
            created_at_ms,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.decision_id, "decision id")?;
        require_non_empty(&self.principal, "decision principal")?;
        if self.decision == CatalogModelDecisionKind::Approved && self.evidence_ids.is_empty() {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "approval requires evaluation evidence".to_string(),
            });
        }
        if self.evidence_ids.iter().collect::<BTreeSet<_>>().len() != self.evidence_ids.len() {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: "decision evidence ids must be unique".to_string(),
            });
        }
        Ok(())
    }
}

/// Mutable serving intent that always resolves to an immutable version/digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogDeploymentBinding {
    pub name: String,
    pub version: u64,
    pub artifact_digest: String,
    pub runtime: String,
    pub endpoint: String,
    pub updated_at_ms: i64,
}

impl CatalogDeploymentBinding {
    pub fn new(
        name: impl Into<String>,
        version: u64,
        artifact_digest: impl Into<String>,
        runtime: impl Into<String>,
        endpoint: impl Into<String>,
        updated_at_ms: i64,
    ) -> Result<Self, CatalogModelContractError> {
        let binding = Self {
            name: name.into(),
            version,
            artifact_digest: artifact_digest.into(),
            runtime: runtime.into(),
            endpoint: endpoint.into(),
            updated_at_ms,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.name, "deployment name")?;
        validate_sha256(&self.artifact_digest, "deployment artifact digest")?;
        require_non_empty(&self.runtime, "deployment runtime")?;
        require_non_empty(&self.endpoint, "deployment endpoint")?;
        Ok(())
    }
}

/// Registered embedding model: immutable versions plus separately mutable or
/// append-only lifecycle records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogEmbeddingModelRegistry {
    pub schema_version: u32,
    /// Optimistic concurrency token for catalog/API mutations. Old rows default
    /// to zero; every successful command increments it.
    #[serde(default)]
    pub revision: u64,
    pub name: String,
    #[serde(default, with = "version_map_wire")]
    pub versions: BTreeMap<u64, CatalogEmbeddingModelVersion>,
    #[serde(default)]
    pub aliases: BTreeMap<String, u64>,
    #[serde(default)]
    pub evidence: Vec<CatalogEvaluationEvidence>,
    #[serde(default)]
    pub decisions: Vec<CatalogModelDecision>,
    #[serde(default)]
    pub deployments: BTreeMap<String, CatalogDeploymentBinding>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CatalogModelAliasMutation {
    pub alias: String,
    pub version: u64,
}

/// Command-shaped mutations preserve version/evidence/decision immutability at
/// the persistence boundary. API adapters lower into these commands instead
/// of replacing a whole registry document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "operation", content = "payload", rename_all = "snake_case")]
pub enum CatalogModelRegistryMutation {
    #[cfg_attr(feature = "openapi", schema(title = "RegisterModelVersionMutation"))]
    RegisterVersion(Box<CatalogEmbeddingModelVersion>),
    SetAlias(CatalogModelAliasMutation),
    #[cfg_attr(feature = "openapi", schema(title = "AppendModelEvidenceMutation"))]
    AppendEvidence(CatalogEvaluationEvidence),
    #[cfg_attr(feature = "openapi", schema(title = "RecordModelDecisionMutation"))]
    RecordDecision(CatalogModelDecision),
    #[cfg_attr(feature = "openapi", schema(title = "UpsertModelDeploymentMutation"))]
    UpsertDeployment(CatalogDeploymentBinding),
}

impl CatalogModelRegistryMutation {
    pub fn register_version(version: CatalogEmbeddingModelVersion) -> Self {
        Self::RegisterVersion(Box::new(version))
    }

    pub fn set_alias(alias: impl Into<String>, version: u64) -> Self {
        Self::SetAlias(CatalogModelAliasMutation {
            alias: alias.into(),
            version,
        })
    }
}

impl CatalogEmbeddingModelRegistry {
    pub fn new(name: impl Into<String>) -> Result<Self, CatalogModelContractError> {
        let registry = Self {
            schema_version: MODEL_CONTRACT_SCHEMA_VERSION,
            revision: 0,
            name: name.into(),
            versions: BTreeMap::new(),
            aliases: BTreeMap::new(),
            evidence: Vec::new(),
            decisions: Vec::new(),
            deployments: BTreeMap::new(),
            tags: BTreeMap::new(),
        };
        require_non_empty(&registry.name, "registered model name")?;
        Ok(registry)
    }

    pub fn register_version(
        &mut self,
        version: CatalogEmbeddingModelVersion,
    ) -> Result<(), CatalogModelContractError> {
        version.validate()?;
        if self.versions.contains_key(&version.version) {
            return Err(CatalogModelContractError::DuplicateVersion {
                version: version.version,
            });
        }
        self.versions.insert(version.version, version);
        Ok(())
    }

    pub fn version(
        &self,
        version: u64,
    ) -> Result<&CatalogEmbeddingModelVersion, CatalogModelContractError> {
        self.versions
            .get(&version)
            .ok_or(CatalogModelContractError::UnknownVersion { version })
    }

    pub fn set_alias(
        &mut self,
        alias: impl Into<String>,
        version: u64,
    ) -> Result<(), CatalogModelContractError> {
        self.version(version)?;
        let alias = alias.into();
        if !Self::valid_alias(&alias) {
            return Err(CatalogModelContractError::InvalidAlias { alias });
        }
        self.aliases.insert(alias, version);
        Ok(())
    }

    fn valid_alias(alias: &str) -> bool {
        !alias.is_empty()
            && !alias
                .bytes()
                .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b'/' | b'@'))
    }

    pub fn resolve_alias(
        &self,
        alias: &str,
    ) -> Result<&CatalogEmbeddingModelVersion, CatalogModelContractError> {
        let version =
            self.aliases
                .get(alias)
                .ok_or_else(|| CatalogModelContractError::UnknownAlias {
                    alias: alias.to_string(),
                })?;
        self.version(*version)
    }

    /// Resolve and verify every executable field persisted by a collection.
    /// Alias lookup is intentionally absent: callers must persist a concrete
    /// version before entering the data plane.
    pub fn resolve_use(
        &self,
        version: u64,
        contract_sha256: &str,
        dimension: u32,
        policy: &CatalogModelUsePolicy,
    ) -> Result<&CatalogEmbeddingModelVersion, CatalogModelContractError> {
        let model = self.version(version)?;
        if model.contract_sha256()? != contract_sha256 {
            return Err(CatalogModelContractError::ContractDigestMismatch { version });
        }
        if !model.output.supports(dimension) {
            return Err(CatalogModelContractError::UnsupportedOutputDimension {
                version,
                dimension,
            });
        }
        if policy.require_approved {
            match self
                .decisions
                .iter()
                .rev()
                .find(|decision| decision.version == version)
                .map(|decision| decision.decision)
            {
                Some(CatalogModelDecisionKind::Approved) => {}
                Some(CatalogModelDecisionKind::Rejected) => {
                    return Err(CatalogModelContractError::ModelRejected { version });
                }
                Some(CatalogModelDecisionKind::Deprecated) => {
                    return Err(CatalogModelContractError::ModelDeprecated { version });
                }
                None => return Err(CatalogModelContractError::ModelNotApproved { version }),
            }
        }
        if let Some(runtime) = &policy.runtime
            && !model.governance.approved_runtimes.contains(runtime)
        {
            return Err(CatalogModelContractError::RuntimeNotApproved {
                runtime: runtime.clone(),
                version,
            });
        }
        Ok(model)
    }

    pub fn append_evidence(
        &mut self,
        evidence: CatalogEvaluationEvidence,
    ) -> Result<(), CatalogModelContractError> {
        self.version(evidence.version)?;
        if self
            .evidence
            .iter()
            .any(|item| item.evidence_id == evidence.evidence_id)
        {
            return Err(CatalogModelContractError::DuplicateEvidence {
                evidence_id: evidence.evidence_id,
            });
        }
        self.evidence.push(evidence);
        Ok(())
    }

    pub fn record_decision(
        &mut self,
        decision: CatalogModelDecision,
    ) -> Result<(), CatalogModelContractError> {
        self.version(decision.version)?;
        if self
            .decisions
            .iter()
            .any(|item| item.decision_id == decision.decision_id)
        {
            return Err(CatalogModelContractError::DuplicateDecision {
                decision_id: decision.decision_id,
            });
        }
        for evidence_id in &decision.evidence_ids {
            if !self
                .evidence
                .iter()
                .any(|item| item.evidence_id == *evidence_id && item.version == decision.version)
            {
                return Err(CatalogModelContractError::UnknownEvidence {
                    evidence_id: evidence_id.clone(),
                    version: decision.version,
                });
            }
        }
        self.decisions.push(decision);
        Ok(())
    }

    pub fn upsert_deployment(
        &mut self,
        deployment: CatalogDeploymentBinding,
    ) -> Result<(), CatalogModelContractError> {
        let version = self.version(deployment.version)?;
        if version.artifact.digest != deployment.artifact_digest {
            return Err(CatalogModelContractError::DeploymentDigestMismatch {
                deployment: deployment.name,
                version: deployment.version,
            });
        }
        if !version
            .governance
            .approved_runtimes
            .contains(&deployment.runtime)
        {
            return Err(CatalogModelContractError::RuntimeNotApproved {
                runtime: deployment.runtime,
                version: deployment.version,
            });
        }
        self.deployments.insert(deployment.name.clone(), deployment);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        require_non_empty(&self.name, "registered model name")?;
        if self.schema_version != MODEL_CONTRACT_SCHEMA_VERSION {
            return Err(CatalogModelContractError::InvalidDimensionPolicy {
                reason: format!(
                    "unsupported model contract schema version {}",
                    self.schema_version
                ),
            });
        }
        for (number, version) in &self.versions {
            if number != &version.version {
                return Err(CatalogModelContractError::InvalidDimensionPolicy {
                    reason: format!("version map key {number} does not match payload"),
                });
            }
            version.validate()?;
        }
        for (alias, version) in &self.aliases {
            if !Self::valid_alias(alias) || !self.versions.contains_key(version) {
                return Err(CatalogModelContractError::UnknownAlias {
                    alias: alias.clone(),
                });
            }
        }
        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence {
            if !self.versions.contains_key(&evidence.version)
                || !evidence_ids.insert(&evidence.evidence_id)
            {
                return Err(CatalogModelContractError::DuplicateEvidence {
                    evidence_id: evidence.evidence_id.clone(),
                });
            }
            evidence.validate()?;
        }
        let mut decision_ids = BTreeSet::new();
        for decision in &self.decisions {
            decision.validate()?;
            if !self.versions.contains_key(&decision.version)
                || !decision_ids.insert(&decision.decision_id)
            {
                return Err(CatalogModelContractError::DuplicateDecision {
                    decision_id: decision.decision_id.clone(),
                });
            }
            for evidence_id in &decision.evidence_ids {
                if !self.evidence.iter().any(|item| {
                    item.evidence_id == *evidence_id && item.version == decision.version
                }) {
                    return Err(CatalogModelContractError::UnknownEvidence {
                        evidence_id: evidence_id.clone(),
                        version: decision.version,
                    });
                }
            }
        }
        for deployment in self.deployments.values() {
            deployment.validate()?;
            let version = self.version(deployment.version)?;
            if version.artifact.digest != deployment.artifact_digest {
                return Err(CatalogModelContractError::DeploymentDigestMismatch {
                    deployment: deployment.name.clone(),
                    version: deployment.version,
                });
            }
            if !version
                .governance
                .approved_runtimes
                .contains(&deployment.runtime)
            {
                return Err(CatalogModelContractError::RuntimeNotApproved {
                    runtime: deployment.runtime.clone(),
                    version: deployment.version,
                });
            }
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        expected_revision: u64,
        mutation: CatalogModelRegistryMutation,
    ) -> Result<(), CatalogModelContractError> {
        if expected_revision != self.revision {
            return Err(CatalogModelContractError::RevisionConflict {
                expected: expected_revision,
                current: self.revision,
            });
        }
        match mutation {
            CatalogModelRegistryMutation::RegisterVersion(version) => {
                self.register_version(*version)
            }
            CatalogModelRegistryMutation::SetAlias(command) => {
                self.set_alias(command.alias, command.version)
            }
            CatalogModelRegistryMutation::AppendEvidence(evidence) => {
                self.append_evidence(evidence)
            }
            CatalogModelRegistryMutation::RecordDecision(decision) => {
                self.record_decision(decision)
            }
            CatalogModelRegistryMutation::UpsertDeployment(deployment) => {
                self.upsert_deployment(deployment)
            }
        }?;
        self.revision = self.revision.checked_add(1).ok_or_else(|| {
            CatalogModelContractError::InvalidDimensionPolicy {
                reason: "model registry revision exhausted".to_string(),
            }
        })?;
        Ok(())
    }
}

/// Typed MLOps facet on the unified xCatalog object model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "contract", rename_all = "snake_case")]
pub enum CatalogMlopsAsset {
    EmbeddingModel(CatalogEmbeddingModelRegistry),
}

impl CatalogMlopsAsset {
    pub fn validate(&self) -> Result<(), CatalogModelContractError> {
        match self {
            Self::EmbeddingModel(registry) => registry.validate(),
        }
    }

    pub fn apply_model_mutation(
        &mut self,
        expected_revision: u64,
        mutation: CatalogModelRegistryMutation,
    ) -> Result<(), CatalogModelContractError> {
        match self {
            Self::EmbeddingModel(registry) => registry.apply(expected_revision, mutation),
        }
    }
}
