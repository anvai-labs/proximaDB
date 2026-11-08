//! Shared cfg(test)-only test support for the root crate.
//!
//! TD-CAT-10 fallout: the OltpCatalog retirement replaced it in tests with an
//! in-memory double that was pasted into three files; the next Catalog trait
//! change re-broke every copy (five repair commits demonstrated the cost).
//! The root-crate copies now live here once. The catalog crate keeps its own
//! copy (a crate cannot import another crate's cfg(test) items).
#![cfg(test)]

use std::collections::HashMap as StdHashMap;
use tokio::sync::RwLock;

// TD-CAT-10: Minimal in-memory test catalog for precision resolver tests.
//
// Replaces OltpCatalog which is gated behind `oltp-catalog` and unusable
// in both configurations. This catalog stores everything in-memory HashMaps.
pub(crate) struct InMemoryTestCatalog {
    name: String,
    namespaces: RwLock<StdHashMap<Vec<String>, proximadb_catalog::CatalogNamespace>>,
    tables: RwLock<
        StdHashMap<proximadb_catalog::TableIdentifier, proximadb_catalog::CatalogTableSchema>,
    >,
}

impl InMemoryTestCatalog {
    pub(crate) fn new(name: String) -> Self {
        Self {
            name,
            namespaces: RwLock::new(StdHashMap::new()),
            tables: RwLock::new(StdHashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl proximadb_catalog::Catalog for InMemoryTestCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        "test-memory"
    }

    fn identity_authority(&self) -> Option<&dyn proximadb_catalog::CatalogAuthority> {
        None
    }

    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: StdHashMap<String, String>,
    ) -> anyhow::Result<proximadb_catalog::CatalogNamespace> {
        let mut ns = proximadb_catalog::CatalogNamespace::new(namespace.to_vec());
        ns.properties = properties;
        ns.namespace_id = Some(format!("ns_{}", uuid::Uuid::new_v4()));
        let mut namespaces = self.namespaces.write().await;
        namespaces.insert(namespace.to_vec(), ns.clone());
        Ok(ns)
    }

    async fn create_table_inner(
        &self,
        identifier: &proximadb_catalog::TableIdentifier,
        schema: proximadb_catalog::CatalogTableSchema,
    ) -> anyhow::Result<proximadb_catalog::CatalogTableSchema> {
        let mut tables = self.tables.write().await;
        tables.insert(identifier.clone(), schema.clone());
        Ok(schema)
    }

    async fn get_table(
        &self,
        identifier: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<proximadb_catalog::CatalogTableSchema> {
        let tables = self.tables.read().await;
        tables
            .get(identifier)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Table not found: {}", identifier))
    }

    async fn get_namespace(
        &self,
        namespace: &[String],
    ) -> anyhow::Result<proximadb_catalog::CatalogNamespace> {
        let namespaces = self.namespaces.read().await;
        namespaces
            .get(namespace)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Namespace not found: {}", namespace.join(".")))
    }

    async fn list_namespaces(
        &self,
        _parent: Option<&[String]>,
    ) -> anyhow::Result<Vec<proximadb_catalog::CatalogNamespace>> {
        let namespaces = self.namespaces.read().await;
        Ok(namespaces.values().cloned().collect())
    }

    async fn list_tables(
        &self,
        namespace: &[String],
    ) -> anyhow::Result<Vec<proximadb_catalog::TableIdentifier>> {
        let tables = self.tables.read().await;
        Ok(tables
            .keys()
            .filter(|id| id.namespace == *namespace)
            .cloned()
            .collect())
    }

    async fn drop_table(
        &self,
        identifier: &proximadb_catalog::TableIdentifier,
        _purge: bool,
    ) -> anyhow::Result<bool> {
        let mut tables = self.tables.write().await;
        Ok(tables.remove(identifier).is_some())
    }

    async fn get_schema_version(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<i32> {
        Ok(0)
    }

    async fn get_schema_by_version(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
        _version: i32,
    ) -> anyhow::Result<proximadb_catalog::CatalogTableSchema> {
        anyhow::bail!("get_schema_by_version not implemented in test double")
    }

    async fn create_index(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
        _index: proximadb_catalog::CatalogIndex,
    ) -> anyhow::Result<proximadb_catalog::CatalogIndex> {
        anyhow::bail!("create_index not implemented in test double")
    }

    async fn drop_index(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
        _index_name: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn list_indexes(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<Vec<proximadb_catalog::CatalogIndex>> {
        Ok(Vec::new())
    }

    async fn get_statistics(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<proximadb_catalog::CatalogTableStatistics> {
        anyhow::bail!("get_statistics not implemented in test double")
    }

    async fn update_statistics(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
        _stats: proximadb_catalog::CatalogTableStatistics,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    async fn drop_namespace(&self, _namespace: &[String], _cascade: bool) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn namespace_exists(&self, _namespace: &[String]) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn update_namespace_properties(
        &self,
        _namespace: &[String],
        _updates: StdHashMap<String, String>,
        _removals: Vec<String>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    async fn table_exists(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn rename_table(
        &self,
        _from: &proximadb_catalog::TableIdentifier,
        _to: &proximadb_catalog::TableIdentifier,
    ) -> anyhow::Result<()> {
        anyhow::bail!("rename_table not implemented in test double")
    }

    async fn evolve_schema(
        &self,
        _identifier: &proximadb_catalog::TableIdentifier,
        _evolution: proximadb_catalog::CatalogSchemaEvolution,
    ) -> anyhow::Result<proximadb_catalog::CatalogTableSchema> {
        anyhow::bail!("evolve_schema not implemented in test double")
    }
}
