//! Shared xCatalog-to-plan authority context conversion.
//!
//! This module keeps catalog authority semantics in one place for UQL lowering,
//! port-backed EXPLAIN, and future SQL/federated planners.

use std::sync::Arc;

use anyhow::Result;
use proximadb_catalog::{
    CatalogAuthorityMode, CatalogCompressionRejectedCandidate, CatalogCompressionStatsProfile,
    CatalogPhysicalFormat, CatalogProjection, CatalogStorageLayout, CatalogTableSchema,
};

use crate::catalog::{CatalogManager, TableIdentifier};
use crate::query::multimodal::plan::{
    ResolvedAuthorityMode, ResolvedCompressionRejectedCandidateContext,
    ResolvedCompressionStatsProfileContext, ResolvedObjectContext, ResolvedProjectionContext,
    ResolvedStorageLayoutContext,
};

/// Query source identity attached to resolved xCatalog metadata.
#[derive(Debug, Clone)]
pub struct AuthoritySource {
    pub source: String,
    pub alias: Option<String>,
    pub data_model: String,
}

impl AuthoritySource {
    pub fn new(source: impl Into<String>, data_model: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            alias: None,
            data_model: data_model.into(),
        }
    }

    pub fn with_alias(mut self, alias: Option<String>) -> Self {
        self.alias = alias;
        self
    }
}

/// Convert a catalog table schema into planner-native resolved authority context.
pub fn resolved_object_from_catalog_schema(
    source: AuthoritySource,
    table_id: &TableIdentifier,
    schema: &CatalogTableSchema,
) -> ResolvedObjectContext {
    let storage_layouts: Vec<_> = schema
        .storage_layouts
        .iter()
        .map(resolved_layout_from_catalog)
        .collect();
    let projections: Vec<_> = schema
        .projections
        .iter()
        .map(resolved_projection_from_catalog)
        .collect();
    let compression_stats_profiles: Vec<_> = schema
        .compression_stats_profiles
        .iter()
        .map(resolved_compression_profile_from_catalog)
        .collect();
    let authority = storage_layouts
        .iter()
        .find(|layout| layout.name == "primary")
        .map(|layout| layout.authority)
        .or_else(|| storage_layouts.first().map(|layout| layout.authority))
        .unwrap_or(ResolvedAuthorityMode::InternalCanonical);

    let external_policy_boundary = storage_layouts.iter().any(|layout| {
        layout.authority == ResolvedAuthorityMode::ExternalAuthoritative
            && !layout.policy_enforced_in_proxima
    });

    ResolvedObjectContext {
        source: source.source,
        alias: source.alias,
        data_model: source.data_model,
        table_identifier: table_id.to_string(),
        authority,
        storage_layouts,
        projections,
        compression_stats_profiles,
        external_policy_boundary,
        fallback_behavior: fallback_behavior_for_schema(schema),
    }
}

/// Resolve a table/source through xCatalog and convert it into planner authority context.
pub async fn resolve_catalog_authority_context(
    catalog_manager: &Arc<CatalogManager>,
    source: AuthoritySource,
) -> Result<ResolvedObjectContext> {
    // Resolved within the scope in force: a request's tenant, or unscoped for
    // explicit internal execution; no scope is refused.
    let tenant = crate::query::request_scope::resolution_tenant(&source.source)?;
    let (catalog, table_id) = catalog_manager
        .resolve_table_scoped(&source.source, tenant.as_deref())
        .await?;
    let schema = catalog.get_table(&table_id).await?;
    Ok(resolved_object_from_catalog_schema(
        source, &table_id, &schema,
    ))
}

fn resolved_layout_from_catalog(layout: &CatalogStorageLayout) -> ResolvedStorageLayoutContext {
    ResolvedStorageLayoutContext {
        name: layout.name.clone(),
        authority: resolved_authority(layout.authority),
        layout_kind: format!("{:?}", layout.layout_kind),
        physical_format: physical_format_label(&layout.physical_format),
        write_mode: format!("{:?}", layout.write_mode),
        location: layout.location.clone(),
        snapshot_semantics: layout.snapshot_semantics.clone(),
        policy_enforced_in_proxima: layout.policy_enforced_in_proxima,
        lossy_type_mappings: layout.lossy_type_mappings.clone(),
    }
}

fn resolved_projection_from_catalog(projection: &CatalogProjection) -> ResolvedProjectionContext {
    ResolvedProjectionContext {
        name: projection.name.clone(),
        kind: format!("{:?}", projection.kind),
        physical_format: physical_format_label(&projection.physical_format),
        rebuild_source: projection.rebuild_source.clone(),
        freshness: format!("{:?}", projection.freshness),
        max_lag_ms: projection.max_lag_ms,
        rebuildable: projection.rebuildable,
        lossy: projection.lossy,
        support_status: projection.support_status.clone(),
        freshness_state: Some(format!("{:?}", projection.freshness_state)),
        rebuild_rto_seconds_per_10gb: projection
            .rebuild_rto
            .as_ref()
            .map(|rto| rto.rebuild_seconds_per_10gb),
    }
}

fn resolved_compression_profile_from_catalog(
    profile: &CatalogCompressionStatsProfile,
) -> ResolvedCompressionStatsProfileContext {
    ResolvedCompressionStatsProfileContext {
        profile_id: profile.profile_id.clone(),
        layout_name: profile.layout_name.clone(),
        projection_id: profile.projection_id.clone(),
        selected_scheme: profile.selected_scheme.clone(),
        raw_bytes: profile.raw_bytes,
        encoded_bytes: profile.encoded_bytes,
        value_count: profile.value_count,
        measured_ratio: profile.measured_ratio,
        exact_reconstruction: profile.exact_reconstruction,
        encode_cpu_ms_per_block: profile.encode_cpu_ms_per_block,
        decode_ns_per_value: profile.decode_ns_per_value,
        rejected_candidates: profile
            .rejected_candidates
            .iter()
            .map(resolved_rejected_candidate_from_catalog)
            .collect(),
    }
}

fn resolved_rejected_candidate_from_catalog(
    candidate: &CatalogCompressionRejectedCandidate,
) -> ResolvedCompressionRejectedCandidateContext {
    ResolvedCompressionRejectedCandidateContext {
        scheme: candidate.scheme.clone(),
        reason: candidate.reason.clone(),
        expected_ratio: candidate.expected_ratio,
    }
}

fn resolved_authority(authority: CatalogAuthorityMode) -> ResolvedAuthorityMode {
    match authority {
        CatalogAuthorityMode::InternalCanonical | CatalogAuthorityMode::ProximaAuthoritative => {
            ResolvedAuthorityMode::InternalCanonical
        }
        CatalogAuthorityMode::ExternalAuthoritative => ResolvedAuthorityMode::ExternalAuthoritative,
        CatalogAuthorityMode::ImportedSnapshot => ResolvedAuthorityMode::ImportedSnapshot,
        CatalogAuthorityMode::ExportedPublication | CatalogAuthorityMode::ProjectionPublication => {
            ResolvedAuthorityMode::ExportedPublication
        }
        CatalogAuthorityMode::RebuildableProjection => ResolvedAuthorityMode::RebuildableProjection,
        CatalogAuthorityMode::FederatedRead => ResolvedAuthorityMode::ExternalAuthoritative,
    }
}

fn physical_format_label(format: &CatalogPhysicalFormat) -> String {
    match format {
        CatalogPhysicalFormat::External(label) => label.clone(),
        other => format!("{:?}", other),
    }
}

fn fallback_behavior_for_schema(schema: &CatalogTableSchema) -> String {
    if schema
        .storage_layouts
        .iter()
        .any(|layout| layout.authority.is_external_authoritative())
    {
        "apply Proxima policy boundary, then read external authoritative source".to_string()
    } else if schema.projections.is_empty() {
        "read canonical ProximaRecord storage".to_string()
    } else {
        "fall back to canonical ProximaRecord storage and rebuild projections as needed".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::{
        CatalogCompressionStatsProfile, CatalogPhysicalFormat, CatalogStorageLayout,
        CatalogTableSchema,
    };

    #[test]
    fn test_resolved_object_preserves_external_policy_boundary() {
        let table_id = TableIdentifier::new(vec!["lake".to_string()], "docs".to_string());
        let mut schema = CatalogTableSchema::new("docs");
        schema.storage_layouts = vec![CatalogStorageLayout::external_authoritative(
            "iceberg",
            CatalogPhysicalFormat::Iceberg,
            "s3://warehouse/docs",
        )];

        let object = resolved_object_from_catalog_schema(
            AuthoritySource::new("lake.docs", "document"),
            &table_id,
            &schema,
        );

        assert_eq!(
            object.authority,
            ResolvedAuthorityMode::ExternalAuthoritative
        );
        assert!(object.requires_policy_boundary());
        assert_eq!(object.storage_layouts[0].physical_format, "Iceberg");
    }

    #[test]
    fn test_resolved_object_preserves_compression_stats_profiles() {
        let table_id = TableIdentifier::new(vec!["default".to_string()], "vectors".to_string());
        let schema = CatalogTableSchema::new("vectors").with_compression_stats_profile(
            CatalogCompressionStatsProfile::new(
                "bench/vector/base_xor",
                "VectorBaseXorEntropy",
                1024,
                256,
                128,
                true,
            )
            .with_layout_name("pax_vector_spatial")
            .with_projection_id("embedding_exact"),
        );

        let object = resolved_object_from_catalog_schema(
            AuthoritySource::new("vectors", "vector"),
            &table_id,
            &schema,
        );

        assert_eq!(object.compression_stats_profiles.len(), 1);
        assert_eq!(
            object.compression_stats_profiles[0].selected_scheme,
            "VectorBaseXorEntropy"
        );
        assert_eq!(object.compression_stats_profiles[0].measured_ratio, 4.0);
    }
}
