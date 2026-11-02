//! Canonical document record contracts.
//!
//! Documents are a facade over `ProximaRecord`, not a separate durable
//! envelope. This module is the document modality's low-level contract for
//! mapping document API concepts onto the shared record spine described in
//! `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.

use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_kernel::error::ProximaDBError;
use proximadb_records::{
    LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode, RecordKey, RecordStore,
};

/// Stable label attached to canonical records that originated from the
/// document facade.
pub const DOCUMENT_RECORD_LABEL: &str = "document";

/// Reserved property used to retain collection identity in the canonical
/// record while xCatalog collection/table mapping is still being extracted.
pub const DOCUMENT_COLLECTION_PROP: &str = "_document_collection";

/// Reserved property used to retain document type/kind for compatibility with
/// existing document API surfaces.
pub const DOCUMENT_TYPE_PROP: &str = "_document_type";

/// Canonical identity for a document facade record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRecordKey {
    /// User-visible collection name or id.
    pub collection_id: String,
    /// User-visible document id inside the collection.
    pub document_id: String,
}

impl DocumentRecordKey {
    pub fn new(collection_id: impl Into<String>, document_id: impl Into<String>) -> Self {
        Self {
            collection_id: collection_id.into(),
            document_id: document_id.into(),
        }
    }

    /// Deterministic canonical record oid used until collection/table identity
    /// is owned by xCatalog.
    pub fn canonical_oid(&self) -> String {
        format!("document/{}/{}", self.collection_id, self.document_id)
    }
}

/// Cross-cutting canonical metadata for document records.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentRecordMetadata {
    /// Schema variation or explicit schema id from the document facade.
    pub schema_id: Option<String>,
    /// Optional document kind/type from the document facade.
    pub document_type: Option<String>,
    /// Optimistic-concurrency version.
    pub version: u64,
    /// Last update time in nanoseconds since Unix epoch.
    pub updated_at_ns: Option<i64>,
    /// Owning tenant. Empty means single-tenant / unrestricted.
    pub tenant_id: String,
    /// Principals allowed to read this record. Empty means unrestricted.
    pub permitted_principals: Vec<String>,
    /// Engine-level RLS policy id, if one applies.
    pub rls_policy_id: Option<String>,
    /// Source system or connector.
    pub origin: Option<String>,
    /// Principal that authored the record.
    pub actor: Option<String>,
    /// Ingestion method such as "api", "cdc", or "migration".
    pub method: Option<String>,
}

/// Document facade shape before it is written as a canonical record.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalDocument {
    pub key: DocumentRecordKey,
    /// Document body as canonical NF2 properties. Protocol adapters may convert
    /// wire shapes into this rich canonical type system, but the modality
    /// contract itself does not depend on legacy proto document envelopes.
    pub document: ProximaTree,
    pub metadata: DocumentRecordMetadata,
}

impl CanonicalDocument {
    pub fn new(
        collection_id: impl Into<String>,
        document_id: impl Into<String>,
        document: ProximaTree,
    ) -> Self {
        Self {
            key: DocumentRecordKey::new(collection_id, document_id),
            document,
            metadata: DocumentRecordMetadata::default(),
        }
    }

    /// Convert this document facade shape into the durable `ProximaRecord`
    /// envelope.
    pub fn into_proxima_record(self) -> ProximaRecord {
        let mut labels = LabelSet::new();
        labels.insert(DOCUMENT_RECORD_LABEL);

        let mut props = self.document;
        props.insert(
            DOCUMENT_COLLECTION_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.key.collection_id.clone())),
        );

        if let Some(document_type) = &self.metadata.document_type {
            props.insert(
                DOCUMENT_TYPE_PROP.to_string(),
                ProximaTreeNode::Value(ProximaValue::String(document_type.clone())),
            );
        }

        let mut record = ProximaRecord {
            oid: self.key.canonical_oid(),
            local_id: Some(self.key.document_id),
            variation_id: self.metadata.schema_id,
            record_version: self.metadata.version,
            tenant_id: self.metadata.tenant_id,
            permitted_principals: self.metadata.permitted_principals,
            rls_policy_id: self.metadata.rls_policy_id,
            origin: self.metadata.origin,
            actor: self.metadata.actor,
            method: self.metadata.method,
            props,
            labels,
            ..ProximaRecord::default()
        };

        if let Some(updated_at_ns) = self.metadata.updated_at_ns {
            record.updated_at_ns = updated_at_ns;
        }

        record
    }
}

/// Rebuild a document facade shape from a canonical record.
pub fn canonical_document_from_record(record: &ProximaRecord) -> Option<CanonicalDocument> {
    if !record.labels.contains(DOCUMENT_RECORD_LABEL) {
        return None;
    }

    let collection_id = match record.props.get(DOCUMENT_COLLECTION_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(collection_id))) => collection_id.clone(),
        _ => return None,
    };

    let document_id = record
        .local_id
        .clone()
        .unwrap_or_else(|| record.oid.clone());
    let document_type = match record.props.get(DOCUMENT_TYPE_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(document_type))) => {
            Some(document_type.clone())
        }
        _ => None,
    };

    let mut document_tree = record.props.clone();
    document_tree.remove(DOCUMENT_COLLECTION_PROP);
    document_tree.remove(DOCUMENT_TYPE_PROP);

    Some(CanonicalDocument {
        key: DocumentRecordKey::new(collection_id, document_id),
        document: document_tree,
        metadata: DocumentRecordMetadata {
            schema_id: record.variation_id.clone(),
            document_type,
            version: record.record_version,
            updated_at_ns: Some(record.updated_at_ns),
            tenant_id: record.tenant_id.clone(),
            permitted_principals: record.permitted_principals.clone(),
            rls_policy_id: record.rls_policy_id.clone(),
            origin: record.origin.clone(),
            actor: record.actor.clone(),
            method: record.method.clone(),
        },
    })
}

/// Canonical document storage contract.
///
/// Implementations write/read `ProximaRecord` as the durable truth. Document
/// indexes, full-text stores, JSON path indexes, and columnar variation tables
/// are projection consumers of this contract, not separate durable stores.
#[async_trait]
pub trait CanonicalDocumentStore: Send + Sync {
    async fn upsert_document_record(
        &self,
        document: CanonicalDocument,
    ) -> Result<ProximaRecord, ProximaDBError>;

    async fn get_document_record(
        &self,
        key: &DocumentRecordKey,
    ) -> Result<Option<ProximaRecord>, ProximaDBError>;

    async fn delete_document_record(&self, key: &DocumentRecordKey)
    -> Result<bool, ProximaDBError>;
}

#[async_trait]
impl<T> CanonicalDocumentStore for T
where
    T: RecordStore + Send + Sync,
{
    async fn upsert_document_record(
        &self,
        document: CanonicalDocument,
    ) -> Result<ProximaRecord, ProximaDBError> {
        self.upsert_record(document.into_proxima_record())
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn get_document_record(
        &self,
        key: &DocumentRecordKey,
    ) -> Result<Option<ProximaRecord>, ProximaDBError> {
        self.get_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn delete_document_record(
        &self,
        key: &DocumentRecordKey,
    ) -> Result<bool, ProximaDBError> {
        self.delete_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::test_support::MemoryRecordStore;
    use std::collections::HashMap;

    fn value_string(value: &str) -> ProximaTreeNode {
        ProximaTreeNode::Value(ProximaValue::String(value.to_string()))
    }

    fn value_int(value: i64) -> ProximaTreeNode {
        ProximaTreeNode::Value(ProximaValue::Int64(value))
    }

    #[test]
    fn document_key_builds_stable_canonical_oid() {
        let key = DocumentRecordKey::new("profiles", "user-1");
        assert_eq!(key.canonical_oid(), "document/profiles/user-1");
    }

    #[test]
    fn canonical_document_maps_to_proxima_record() {
        let document = HashMap::from([
            ("name".to_string(), value_string("Ada")),
            ("age".to_string(), value_int(37)),
        ]);
        let mut canonical = CanonicalDocument::new("profiles", "user-1", document);
        canonical.metadata.schema_id = Some("profile-v1".to_string());
        canonical.metadata.document_type = Some("profile".to_string());
        canonical.metadata.version = 7;
        canonical.metadata.tenant_id = "tenant-a".to_string();
        canonical.metadata.permitted_principals = vec!["alice".to_string()];
        canonical.metadata.updated_at_ns = Some(123);

        let record = canonical.into_proxima_record();

        assert_eq!(record.oid, "document/profiles/user-1");
        assert_eq!(record.local_id.as_deref(), Some("user-1"));
        assert_eq!(record.variation_id.as_deref(), Some("profile-v1"));
        assert_eq!(record.record_version, 7);
        assert_eq!(record.tenant_id, "tenant-a");
        assert_eq!(record.updated_at_ns, 123);
        assert!(record.labels.contains(DOCUMENT_RECORD_LABEL));
        assert!(record.props.contains_key("name"));
        assert!(record.props.contains_key(DOCUMENT_COLLECTION_PROP));
    }

    #[test]
    fn canonical_record_round_trips_to_document_facade_shape() {
        let nested = HashMap::from([("city".to_string(), value_string("London"))]);
        let document = HashMap::from([
            ("name".to_string(), value_string("Ada")),
            ("address".to_string(), ProximaTreeNode::Object(nested)),
        ]);
        let mut canonical = CanonicalDocument::new("profiles", "user-1", document);
        canonical.metadata.document_type = Some("profile".to_string());

        let record = canonical.into_proxima_record();
        let round_tripped = canonical_document_from_record(&record).expect("document record");

        assert_eq!(round_tripped.key.collection_id, "profiles");
        assert_eq!(round_tripped.key.document_id, "user-1");
        assert_eq!(
            round_tripped.metadata.document_type.as_deref(),
            Some("profile")
        );
        assert!(round_tripped.document.contains_key("name"));
        assert!(round_tripped.document.contains_key("address"));
        assert!(
            !round_tripped
                .document
                .contains_key(DOCUMENT_COLLECTION_PROP)
        );
    }

    #[test]
    fn non_document_record_is_not_rebuilt_as_document() {
        let record = ProximaRecord::default();
        assert!(canonical_document_from_record(&record).is_none());
    }

    #[tokio::test]
    async fn canonical_document_store_adapts_to_record_store() {
        let store = MemoryRecordStore::default();
        let document = HashMap::from([("name".to_string(), value_string("Ada"))]);
        let canonical = CanonicalDocument::new("profiles", "user-1", document);
        let key = canonical.key.clone();

        let written = store
            .upsert_document_record(canonical)
            .await
            .expect("upsert document");
        assert_eq!(written.oid, "document/profiles/user-1");

        let fetched = store
            .get_document_record(&key)
            .await
            .expect("get document")
            .expect("document exists");
        assert!(canonical_document_from_record(&fetched).is_some());

        assert!(
            store
                .delete_document_record(&key)
                .await
                .expect("delete document")
        );
    }
}
