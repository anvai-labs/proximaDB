// B+ tree index for JSON paths
//
// Provides efficient range queries on scalar values extracted from JSON paths.
// Uses an in-memory B+ tree structure with disk persistence via SST engine.

use std::collections::{BTreeMap, HashSet};
use std::sync::RwLock;

use anyhow::Result;
use tracing::warn;

use super::{IndexValue, PathQueryCondition};

/// B+ tree index for a single JSON path
pub struct PathIndex {
    /// JSON path expression (e.g., "$.user.age")
    path: String,
    /// Whether this is a unique index
    unique: bool,
    /// Whether this is a sparse index (skip nulls)
    sparse: bool,
    /// In-memory B-tree: value -> document IDs
    tree: RwLock<BTreeMap<IndexValueKey, HashSet<String>>>,
    /// Reverse lookup: document ID -> value
    reverse: RwLock<BTreeMap<String, IndexValueKey>>,
}

/// Wrapper for IndexValue that implements Ord and Hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct IndexValueKey(IndexValueOrd);

#[derive(Debug, Clone)]
enum IndexValueOrd {
    Null,
    Bool(bool),
    Int(i64),
    Float(OrderedFloat),
    String(String),
    Bytes(Vec<u8>),
}

impl std::hash::Hash for IndexValueOrd {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Null => {}
            Self::Bool(v) => v.hash(state),
            Self::Int(v) => v.hash(state),
            Self::Float(v) => v.hash(state),
            Self::String(v) => v.hash(state),
            Self::Bytes(v) => v.hash(state),
        }
    }
}

/// Ordered float wrapper
use proximadb_records::BitKeyF64 as OrderedFloat;

impl PartialEq for IndexValueOrd {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for IndexValueOrd {}

impl PartialOrd for IndexValueOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexValueOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Null, Self::Null) => Ordering::Equal,
            (Self::Null, _) => Ordering::Less,
            (_, Self::Null) => Ordering::Greater,
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Bool(_), _) => Ordering::Less,
            (_, Self::Bool(_)) => Ordering::Greater,
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Int(a), Self::Float(b)) => {
                // unwrap_or(Equal) handles NaN: partial_cmp returns None when comparing with NaN
                (*a as f64).partial_cmp(&b.0).unwrap_or(Ordering::Equal)
            }
            (Self::Float(a), Self::Int(b)) => {
                // unwrap_or(Equal) handles NaN: partial_cmp returns None when comparing with NaN
                a.0.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (Self::Float(a), Self::Float(b)) => a.cmp(b),
            (Self::Int(_), _) | (Self::Float(_), _) => Ordering::Less,
            (_, Self::Int(_)) | (_, Self::Float(_)) => Ordering::Greater,
            (Self::String(a), Self::String(b)) => a.cmp(b),
            (Self::String(_), _) => Ordering::Less,
            (_, Self::String(_)) => Ordering::Greater,
            (Self::Bytes(a), Self::Bytes(b)) => a.cmp(b),
        }
    }
}

impl From<IndexValue> for IndexValueKey {
    fn from(v: IndexValue) -> Self {
        IndexValueKey(match v {
            IndexValue::Null => IndexValueOrd::Null,
            IndexValue::Bool(b) => IndexValueOrd::Bool(b),
            IndexValue::Int(i) => IndexValueOrd::Int(i),
            IndexValue::Float(f) => IndexValueOrd::Float(OrderedFloat(f)),
            IndexValue::String(s) => IndexValueOrd::String(s),
            IndexValue::Bytes(b) => IndexValueOrd::Bytes(b),
        })
    }
}

impl PathIndex {
    /// Create a new path index
    pub fn new(path: &str, unique: bool, sparse: bool) -> Self {
        Self {
            path: path.to_string(),
            unique,
            sparse,
            tree: RwLock::new(BTreeMap::new()),
            reverse: RwLock::new(BTreeMap::new()),
        }
    }

    /// Insert a document into the index
    pub fn insert(&self, doc_id: &str, value: IndexValue) -> Result<()> {
        // Skip nulls for sparse indexes
        if self.sparse && matches!(value, IndexValue::Null) {
            return Ok(());
        }

        let key = IndexValueKey::from(value);

        // Check uniqueness constraint
        if self.unique {
            let tree = self
                .tree
                .read()
                .map_err(|e| anyhow::anyhow!("RwLock for tree is poisoned: {}", e))?;
            if let Some(existing) = tree.get(&key)
                && !existing.is_empty()
                && !existing.contains(doc_id)
            {
                return Err(anyhow::anyhow!(
                    "Duplicate key violation on unique index for path {}",
                    self.path
                ));
            }
        }

        // Insert into tree
        {
            let mut tree = self
                .tree
                .write()
                .map_err(|e| anyhow::anyhow!("RwLock for tree is poisoned: {}", e))?;
            tree.entry(key.clone())
                .or_default()
                .insert(doc_id.to_string());
        }

        // Update reverse lookup
        {
            let mut reverse = self
                .reverse
                .write()
                .map_err(|e| anyhow::anyhow!("RwLock for reverse is poisoned: {}", e))?;
            reverse.insert(doc_id.to_string(), key);
        }

        Ok(())
    }

    /// Remove a document from the index
    pub fn remove(&self, doc_id: &str) -> Result<()> {
        // Get the value from reverse lookup
        let key = {
            let mut reverse = self
                .reverse
                .write()
                .map_err(|e| anyhow::anyhow!("RwLock for reverse is poisoned: {}", e))?;
            reverse.remove(doc_id)
        };

        // Remove from tree
        if let Some(key) = key {
            let mut tree = self
                .tree
                .write()
                .map_err(|e| anyhow::anyhow!("RwLock for tree is poisoned: {}", e))?;
            if let Some(ids) = tree.get_mut(&key) {
                ids.remove(doc_id);
                if ids.is_empty() {
                    tree.remove(&key);
                }
            }
        }

        Ok(())
    }

    /// Query the index
    pub fn query(&self, condition: &PathQueryCondition) -> Result<Vec<String>> {
        let tree = self
            .tree
            .read()
            .map_err(|e| anyhow::anyhow!("RwLock for tree is poisoned: {}", e))?;
        let mut results = HashSet::new();

        match condition {
            PathQueryCondition::Eq(value) => {
                let key = IndexValueKey::from(value.clone());
                if let Some(ids) = tree.get(&key) {
                    results.extend(ids.iter().cloned());
                }
            }
            PathQueryCondition::Ne(value) => {
                let key = IndexValueKey::from(value.clone());
                for (k, ids) in tree.iter() {
                    if k != &key {
                        results.extend(ids.iter().cloned());
                    }
                }
            }
            PathQueryCondition::Gt(value) => {
                let key = IndexValueKey::from(value.clone());
                for (_k, ids) in
                    tree.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
                {
                    results.extend(ids.iter().cloned());
                }
            }
            PathQueryCondition::Gte(value) => {
                let key = IndexValueKey::from(value.clone());
                for (_k, ids) in
                    tree.range((std::ops::Bound::Included(key), std::ops::Bound::Unbounded))
                {
                    results.extend(ids.iter().cloned());
                }
            }
            PathQueryCondition::Lt(value) => {
                let key = IndexValueKey::from(value.clone());
                for (_k, ids) in
                    tree.range((std::ops::Bound::Unbounded, std::ops::Bound::Excluded(key)))
                {
                    results.extend(ids.iter().cloned());
                }
            }
            PathQueryCondition::Lte(value) => {
                let key = IndexValueKey::from(value.clone());
                for (_k, ids) in
                    tree.range((std::ops::Bound::Unbounded, std::ops::Bound::Included(key)))
                {
                    results.extend(ids.iter().cloned());
                }
            }
            PathQueryCondition::In(values) => {
                for value in values {
                    let key = IndexValueKey::from(value.clone());
                    if let Some(ids) = tree.get(&key) {
                        results.extend(ids.iter().cloned());
                    }
                }
            }
            PathQueryCondition::NotIn(values) => {
                let excluded: HashSet<_> = values
                    .iter()
                    .map(|v| IndexValueKey::from(v.clone()))
                    .collect();
                for (k, ids) in tree.iter() {
                    if !excluded.contains(k) {
                        results.extend(ids.iter().cloned());
                    }
                }
            }
            PathQueryCondition::Exists(exists) => {
                if *exists {
                    // Return all indexed documents
                    for ids in tree.values() {
                        results.extend(ids.iter().cloned());
                    }
                }
                // For exists=false, we'd need to scan all documents
            }
            PathQueryCondition::Regex(_pattern) => {
                // Deferred: Implement regex matching for string values
                let _ = _pattern; // Suppress unused warning
            }
        }

        Ok(results.into_iter().collect())
    }

    /// Get the number of indexed values
    pub fn len(&self) -> usize {
        match self.tree.read() {
            Ok(tree) => tree.len(),
            Err(error) => {
                warn!(error = %error, "PathIndex tree lock poisoned while checking len");
                0
            }
        }
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        match self.tree.read() {
            Ok(tree) => tree.is_empty(),
            Err(error) => {
                warn!(
                    error = %error,
                    "PathIndex tree lock poisoned while checking emptiness; treating as empty"
                );
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_index_insert_query() {
        let index = PathIndex::new("$.age", false, false);

        index.insert("doc1", IndexValue::Int(25)).unwrap();
        index.insert("doc2", IndexValue::Int(30)).unwrap();
        index.insert("doc3", IndexValue::Int(25)).unwrap();

        let results = index
            .query(&PathQueryCondition::Eq(IndexValue::Int(25)))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"doc1".to_string()));
        assert!(results.contains(&"doc3".to_string()));
    }

    #[test]
    fn test_path_index_unique() {
        let index = PathIndex::new("$.email", true, false);

        index
            .insert("doc1", IndexValue::String("a@b.com".to_string()))
            .unwrap();
        let result = index.insert("doc2", IndexValue::String("a@b.com".to_string()));

        assert!(result.is_err());
    }

    #[test]
    fn test_path_index_sparse() {
        let index = PathIndex::new("$.optional", false, true);

        index.insert("doc1", IndexValue::Null).unwrap();
        index.insert("doc2", IndexValue::Int(42)).unwrap();

        // Null should be skipped
        assert_eq!(index.len(), 1);
    }
}
