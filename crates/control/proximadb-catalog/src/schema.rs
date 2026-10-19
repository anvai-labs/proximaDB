// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Schema validation and evolution helpers for xCatalog.
//!
//! All types used here are defined in this crate (`proximadb-catalog`) so this
//! module can be consumed without depending on the root `proximadb` crate.

use std::collections::HashSet;

use anyhow::{Result, anyhow};

use proximadb_data_model::ProximaType;

use crate::{
    CatalogColumn, CatalogIndex, CatalogIndexType, CatalogMlopsAssetExt, CatalogSchemaEvolution,
    CatalogTableSchema, ColumnConstraint, SchemaChange, system_columns,
};

/// Validate a schema for internal consistency.
pub fn validate_schema(schema: &CatalogTableSchema) -> Result<()> {
    if schema.name.is_empty() {
        return Err(anyhow!("Schema name cannot be empty"));
    }

    if schema.columns.is_empty() && schema.mlops_asset.is_none() {
        return Err(anyhow!("Schema must have at least one column"));
    }

    if let Some(asset) = schema.mlops_asset_as_typed()? {
        asset
            .validate()
            .map_err(|error| anyhow!("Invalid MLOps asset: {error}"))?;
    }
    if let Some(binding) = &schema.embedding_config {
        binding.validate_model_binding()?;
    }

    let mut seen = HashSet::new();
    for col in &schema.columns {
        if col.name.is_empty() {
            return Err(anyhow!("Column name cannot be empty"));
        }
        if system_columns::is_reserved_column_name(&col.name) {
            return Err(anyhow!(
                "Column name '{}' is reserved for ProximaDB system metadata",
                col.name
            ));
        }
        if !seen.insert(&col.name) {
            return Err(anyhow!("Duplicate column name: {}", col.name));
        }
    }

    for pk in &schema.primary_key {
        if !schema.columns.iter().any(|c| &c.name == pk) {
            return Err(anyhow!("Primary key column '{}' not found in schema", pk));
        }
    }

    for idx in &schema.indexes {
        for col in &idx.columns {
            if !schema.columns.iter().any(|c| &c.name == col) {
                return Err(anyhow!(
                    "Index '{}' references non-existent column '{}'",
                    idx.name,
                    col
                ));
            }
        }
    }

    // `relational_capabilities` holds a second copy of the identity declarations
    // (TD-CAT-6) and was never validated: its columns went unchecked, and nothing
    // asserted that the two `primary_key` vectors agree. That silence is what let
    // rename/drop leave the copies stale.
    let rc = &schema.relational_capabilities;
    for pk in &rc.primary_key {
        if !schema.columns.iter().any(|c| &c.name == pk) {
            return Err(anyhow!(
                "relational_capabilities primary key column '{}' not found in schema",
                pk
            ));
        }
    }
    // An empty `relational_capabilities.primary_key` is legal — it is only
    // populated by `CREATE TABLE`, so most creation paths leave it unset. But a
    // *populated* one that disagrees means the two readers of "what is the PK"
    // resolve different columns, which is never intentional.
    if !rc.primary_key.is_empty() && !same_column_set(&rc.primary_key, &schema.primary_key) {
        return Err(anyhow!(
            "primary key disagrees between schema.primary_key {:?} and \
             relational_capabilities.primary_key {:?}",
            schema.primary_key,
            rc.primary_key
        ));
    }
    for idx in rc.unique_indexes.iter().chain(rc.secondary_indexes.iter()) {
        for col in &idx.columns {
            if !schema.columns.iter().any(|c| &c.name == col) {
                return Err(anyhow!(
                    "relational_capabilities index '{}' references non-existent column '{}'",
                    idx.name,
                    col
                ));
            }
        }
    }
    for constraint in &rc.constraints {
        let columns = match constraint {
            ColumnConstraint::Unique { columns } => columns,
            ColumnConstraint::ForeignKey { columns, .. } => columns,
            ColumnConstraint::Check { .. } => continue,
        };
        for col in columns {
            if !schema.columns.iter().any(|c| &c.name == col) {
                return Err(anyhow!(
                    "relational_capabilities constraint references non-existent column '{}'",
                    col
                ));
            }
        }
    }

    // ADR-077: `relational_capabilities.constraints` is CANONICAL for uniqueness;
    // `unique_indexes` and `indexes[].is_unique` are projections kept for the
    // pg/JDBC introspection surfaces. A projection holding a UNIQUE the canonical
    // field lacks is the exact shape of the defect this ADR exists to make
    // unrepresentable — enforcement reads canonical, so such a constraint would be
    // cataloged and silently never enforced.
    //
    // Only the dangerous direction is rejected. The reverse (canonical holding
    // something no projection mirrors) is fine: an unnamed UNIQUE has no index
    // form to project into.
    let canonical_unique: Vec<&Vec<String>> = rc
        .constraints
        .iter()
        .filter_map(|c| match c {
            ColumnConstraint::Unique { columns } => Some(columns),
            _ => None,
        })
        .collect();
    let mut projected = rc
        .unique_indexes
        .iter()
        .map(|i| (&i.name, &i.columns))
        .chain(
            schema
                .indexes
                .iter()
                .filter(|i| i.is_unique)
                .map(|i| (&i.name, &i.columns)),
        );
    if let Some((name, columns)) = projected.find(|(_, columns)| {
        !columns.is_empty()
            && !canonical_unique
                .iter()
                .any(|canonical| same_column_set(canonical, columns))
    }) {
        return Err(anyhow!(
            "UNIQUE index '{}' on ({}) is not present in the canonical \
             relational_capabilities.constraints — it would be cataloged but never \
             enforced. Call schema::normalize_identity first (ADR-077)",
            name,
            columns.join(", ")
        ));
    }

    for col in &schema.columns {
        if matches!(
            col.data_type,
            ProximaType::DenseVector { .. } | ProximaType::SparseVector { .. }
        ) && !col.properties.contains_key("dimension")
        {
            return Err(anyhow!(
                "Vector column '{}' must have 'dimension' property",
                col.name
            ));
        }
    }

    validate_storage_contract(schema)?;

    Ok(())
}

fn validate_storage_contract(schema: &CatalogTableSchema) -> Result<()> {
    for layout in &schema.storage_layouts {
        if layout.requires_external_contract() {
            if layout.location.as_deref().unwrap_or_default().is_empty() {
                return Err(anyhow!(
                    "External layout '{}' for table '{}' must declare a location",
                    layout.name,
                    schema.name
                ));
            }
            if layout
                .snapshot_semantics
                .as_deref()
                .unwrap_or_default()
                .is_empty()
            {
                return Err(anyhow!(
                    "External layout '{}' for table '{}' must declare snapshot semantics",
                    layout.name,
                    schema.name
                ));
            }
        }
    }

    for projection in &schema.projections {
        if projection.rebuildable && projection.rebuild_source.is_empty() {
            return Err(anyhow!(
                "Projection '{}' for table '{}' must declare a rebuild source",
                projection.name,
                schema.name
            ));
        }
    }

    Ok(())
}

/// Do two column lists denote the same key? Order-insensitive: `UNIQUE(a,b)` and
/// `UNIQUE(b,a)` fence the same tuple, so they are one key, not two.
pub fn same_column_set(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a: Vec<&str> = a.iter().map(String::as_str).collect();
    let mut b: Vec<&str> = b.iter().map(String::as_str).collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

/// Rewrite `old_name` → `new_name` across **every** place a column name can be
/// recorded as part of an identity or constraint.
///
/// `CatalogTableSchema` records identity in several locations (TD-CAT-6), and
/// evolution used to maintain only the top-level `primary_key` + `indexes`. The
/// `relational_capabilities` copies were left holding the pre-rename name — and
/// since `relational::effective_primary_key` *prefers* `relational_capabilities
/// .primary_key`, renaming a PK column left row validation and oid encoding
/// resolving a column that no longer exists.
fn rename_column_in_identity(schema: &mut CatalogTableSchema, old_name: &str, new_name: &str) {
    let rename = |names: &mut Vec<String>| {
        for n in names.iter_mut() {
            if n == old_name {
                *n = new_name.to_string();
            }
        }
    };

    rename(&mut schema.primary_key);
    rename(&mut schema.relational_capabilities.primary_key);
    for idx in &mut schema.indexes {
        rename(&mut idx.columns);
    }
    for idx in &mut schema.relational_capabilities.unique_indexes {
        rename(&mut idx.columns);
    }
    for idx in &mut schema.relational_capabilities.secondary_indexes {
        rename(&mut idx.columns);
    }
    for constraint in &mut schema.relational_capabilities.constraints {
        match constraint {
            ColumnConstraint::Unique { columns } => rename(columns),
            ColumnConstraint::ForeignKey {
                columns,
                references_columns,
                ..
            } => {
                rename(columns);
                // Only the *local* columns are renamed; `references_columns`
                // name columns in the referenced table, which this evolution
                // does not touch.
                let _ = references_columns;
            }
            ColumnConstraint::Check { .. } => {}
        }
    }
}

/// Purge `name` from every identity/constraint location, dropping any index or
/// constraint left with no columns. Mirrors [`rename_column_in_identity`] so a
/// drop cannot leave a dangling reference in the copies evolution used to skip.
fn drop_column_from_identity(schema: &mut CatalogTableSchema, name: &str) {
    let purge = |names: &mut Vec<String>| names.retain(|c| c != name);

    for idx in &mut schema.indexes {
        purge(&mut idx.columns);
    }
    schema.indexes.retain(|idx| !idx.columns.is_empty());

    for idx in &mut schema.relational_capabilities.unique_indexes {
        purge(&mut idx.columns);
    }
    schema
        .relational_capabilities
        .unique_indexes
        .retain(|idx| !idx.columns.is_empty());

    for idx in &mut schema.relational_capabilities.secondary_indexes {
        purge(&mut idx.columns);
    }
    schema
        .relational_capabilities
        .secondary_indexes
        .retain(|idx| !idx.columns.is_empty());

    for constraint in &mut schema.relational_capabilities.constraints {
        if let ColumnConstraint::Unique { columns } = constraint {
            purge(columns);
        }
    }
    schema
        .relational_capabilities
        .constraints
        .retain(|c| !matches!(c, ColumnConstraint::Unique { columns } if columns.is_empty()));
}

// ===========================================================================
// Canonical identity discovery (TD-CAT-6 slice 1)
// ===========================================================================

/// What a table declares about its identity, resolved across **every** location
/// `CatalogTableSchema` can record it in.
///
/// This is the single source of truth for *discovery*. It deliberately applies
/// no policy: it reports what the schema says, and each consumer keeps its own
/// rules on top (the identity slot excludes a UNIQUE restating the PK because
/// that is not a *secondary*; the write path does not care). Policy was never
/// where the consumers diverged — discovery was.
///
/// Why this lives here: a UNIQUE can arrive by four routes and a primary key by
/// two, so any consumer that walks the fields itself will miss some. Layering
/// forbids `control → root`, so a canonical answer in the root crate is
/// unreachable from below and every lower layer is forced to re-derive it. The
/// crate that owns the type owns the answers about it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TableIdentity {
    /// The declared primary key columns, in declaration order. Empty when the
    /// table declares none — no conventional-name inference is applied here.
    pub primary_key: Vec<String>,
    /// **Every** declared UNIQUE column set, deduplicated order-insensitively,
    /// including one that restates the primary key. Use
    /// [`TableIdentity::secondary_unique_sets`] to exclude that.
    pub unique_sets: Vec<Vec<String>>,
    /// What the schema left undetermined, for consumers that surface gaps.
    pub gaps: Vec<String>,
}

impl TableIdentity {
    /// The UNIQUE sets that are genuinely *secondary* — every declared set
    /// except one restating the primary key, which the PK already fences.
    pub fn secondary_unique_sets(&self) -> Vec<Vec<String>> {
        self.unique_sets
            .iter()
            .filter(|set| !same_column_set(set, &self.primary_key))
            .cloned()
            .collect()
    }
}

/// Resolve a table's identity declarations from all locations.
///
/// Primary key: prefers the top-level `primary_key`, falling back to the
/// `relational_capabilities` copy. `validate_schema` rejects a schema where a
/// populated copy disagrees, so for a valid schema the choice is immaterial.
///
/// UNIQUE sets, in order, deduplicated:
/// 1. `relational_capabilities.unique_indexes` — inline `UNIQUE (...)` at
///    CREATE TABLE (the dominant route).
/// 2. `relational_capabilities.constraints` → `ColumnConstraint::Unique`.
/// 3. `schema.indexes` entries flagged `is_unique` — `CREATE INDEX` and, before
///    the evolution fix, `ALTER … ADD CONSTRAINT`.
/// 4. `properties["constraint:unique:…"]` — the legacy blob. Read so a schema
///    evolved *before* the AddConstraint fix still yields its UNIQUE rather than
///    silently losing it.
///
/// Note entries in `unique_indexes` are unique by virtue of the field they sit
/// in; `CatalogIndex::new` leaves `is_unique` false and DDL never sets it, so
/// that flag must NOT be filtered on for route 1.
pub fn table_identity(schema: &CatalogTableSchema) -> TableIdentity {
    let rc = &schema.relational_capabilities;

    let primary_key = if !schema.primary_key.is_empty() {
        schema.primary_key.clone()
    } else {
        rc.primary_key.clone()
    };

    let mut unique_sets: Vec<Vec<String>> = Vec::new();
    let push = |columns: &[String], unique_sets: &mut Vec<Vec<String>>| {
        if columns.is_empty() {
            return;
        }
        if unique_sets.iter().any(|s| same_column_set(s, columns)) {
            return;
        }
        unique_sets.push(columns.to_vec());
    };

    for idx in &rc.unique_indexes {
        push(&idx.columns, &mut unique_sets);
    }
    for constraint in &rc.constraints {
        if let ColumnConstraint::Unique { columns } = constraint {
            push(columns, &mut unique_sets);
        }
    }
    for idx in &schema.indexes {
        if idx.is_unique {
            push(&idx.columns, &mut unique_sets);
        }
    }
    for (key, value) in &schema.properties {
        if !key.starts_with("constraint:unique:") {
            continue;
        }
        if let Ok(ColumnConstraint::Unique { columns }) =
            serde_json::from_str::<ColumnConstraint>(value)
        {
            push(&columns, &mut unique_sets);
        }
    }

    let mut gaps = Vec::new();
    if primary_key.is_empty() {
        gaps.push("no primary key declared".to_string());
    }

    TableIdentity {
        primary_key,
        unique_sets,
        gaps,
    }
}

/// Flat projection of [`table_identity`] for consumers that only need the UNIQUE
/// column sets (the shape `services::record_store::schema_unique_column_sets`
/// returns today — TD-CAT-6 slice 2 delegates to this).
pub fn unique_column_sets(schema: &CatalogTableSchema) -> Vec<Vec<String>> {
    table_identity(schema).unique_sets
}

/// Flat projection of [`table_identity`] for consumers that only need the PK.
/// No conventional-name (`id`/`record_id`) inference — that is a caller policy.
pub fn primary_key_columns(schema: &CatalogTableSchema) -> Vec<String> {
    table_identity(schema).primary_key
}

/// Fold every identity declaration into its **canonical** location (ADR-077 M1).
///
/// `relational_capabilities.constraints` is canonical for uniqueness. A UNIQUE may
/// still *arrive* by any of the legacy routes — a pre-ADR-077 persisted schema, an
/// external-catalog adapter, `CREATE INDEX` — so this folds whatever
/// [`table_identity`] discovers into the canonical field.
///
/// **Additive and lossless.** The projections are left exactly as they are:
/// `relational_capabilities.unique_indexes` and `schema.indexes` carry index
/// *names* and *types* that the canonical `ColumnConstraint::Unique` form cannot
/// represent, and the pg/JDBC introspection surfaces need them. Normalizing means
/// "canonical contains everything", not "rewrite everything".
///
/// Idempotent: running it twice changes nothing, because the fold deduplicates
/// order-insensitively against what is already canonical.
///
/// This is the *backfill on touch* ADR-077 relies on instead of an offline
/// migration pass — it runs inside [`apply_evolution`], so any schema that is
/// modified converges on the canonical shape.
pub fn normalize_identity(schema: &mut CatalogTableSchema) {
    for columns in table_identity(schema).unique_sets {
        let already_canonical = schema.relational_capabilities.constraints.iter().any(|c| {
            matches!(c, ColumnConstraint::Unique { columns: existing }
                              if same_column_set(existing, &columns))
        });
        if !already_canonical {
            schema
                .relational_capabilities
                .constraints
                .push(ColumnConstraint::Unique { columns });
        }
    }
}

/// Apply schema evolution changes and return a new schema.
pub fn apply_evolution(
    schema: &CatalogTableSchema,
    evolution: &CatalogSchemaEvolution,
) -> Result<CatalogTableSchema> {
    let mut new_schema = schema.clone();
    new_schema.schema_version += 1;
    new_schema.updated_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let mut next_id = new_schema.columns.iter().map(|c| c.id).max().unwrap_or(0) + 1;

    for change in &evolution.changes {
        match change {
            SchemaChange::AddColumn {
                name,
                data_type,
                nullable,
                default_value,
                comment,
                after,
            } => {
                if new_schema.columns.iter().any(|c| &c.name == name) {
                    return Err(anyhow!("Column '{}' already exists", name));
                }
                let mut col = CatalogColumn::new(next_id, name, data_type.clone());
                next_id += 1;
                col.nullable = *nullable;
                col.default_value = default_value.clone();
                col.comment = comment.clone();
                if let Some(after_col) = after {
                    if let Some(pos) = new_schema.columns.iter().position(|c| &c.name == after_col)
                    {
                        new_schema.columns.insert(pos + 1, col);
                    } else {
                        new_schema.columns.push(col);
                    }
                } else {
                    new_schema.columns.push(col);
                }
            }
            SchemaChange::DropColumn { name } => {
                let pos = new_schema
                    .columns
                    .iter()
                    .position(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                // Guard BOTH primary-key vectors: a column that is the PK only in
                // `relational_capabilities` used to pass this check and be dropped.
                if new_schema.primary_key.contains(name)
                    || new_schema
                        .relational_capabilities
                        .primary_key
                        .contains(name)
                {
                    return Err(anyhow!("Cannot drop primary key column '{}'", name));
                }
                new_schema.columns.remove(pos);
                drop_column_from_identity(&mut new_schema, name);
            }
            SchemaChange::RenameColumn { old_name, new_name } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == old_name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", old_name))?;
                col.name = new_name.clone();
                rename_column_in_identity(&mut new_schema, old_name, new_name);
            }
            SchemaChange::ChangeType { name, new_type } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                if !is_compatible_type_change(&col.data_type, new_type) {
                    return Err(anyhow!(
                        "Cannot change column '{}' from {:?} to {:?}",
                        name,
                        col.data_type,
                        new_type
                    ));
                }
                col.data_type = new_type.clone();
            }
            SchemaChange::UpdateComment { name, comment } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                col.comment = Some(comment.clone());
            }
            SchemaChange::MakeNullable { name } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                col.nullable = true;
            }
            SchemaChange::MakeNotNullable { name } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                col.nullable = false;
            }
            SchemaChange::SetDefault {
                name,
                default_value,
            } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                col.default_value = Some(default_value.clone());
            }
            SchemaChange::DropDefault { name } => {
                let col = new_schema
                    .columns
                    .iter_mut()
                    .find(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                col.default_value = None;
            }
            SchemaChange::MoveColumn { name, after } => {
                let pos = new_schema
                    .columns
                    .iter()
                    .position(|c| &c.name == name)
                    .ok_or_else(|| anyhow!("Column '{}' not found", name))?;
                let col = new_schema.columns.remove(pos);
                if let Some(after_col) = after {
                    if let Some(after_pos) =
                        new_schema.columns.iter().position(|c| &c.name == after_col)
                    {
                        new_schema.columns.insert(after_pos + 1, col);
                    } else {
                        return Err(anyhow!(
                            "Column '{}' not found for AFTER positioning",
                            after_col
                        ));
                    }
                } else {
                    new_schema.columns.insert(0, col);
                }
            }
            SchemaChange::AddConstraint {
                constraint_name,
                constraint,
            } => {
                let constraint_key = match &constraint {
                    ColumnConstraint::Unique { columns } => {
                        format!("constraint:unique:{}", columns.join(","))
                    }
                    ColumnConstraint::Check { .. } => {
                        format!(
                            "constraint:check:{}",
                            constraint_name.as_deref().unwrap_or("unnamed")
                        )
                    }
                    ColumnConstraint::ForeignKey {
                        columns,
                        references_table,
                        ..
                    } => {
                        format!("constraint:fk:{}:{}", columns.join(","), references_table)
                    }
                };
                let constraint_value = serde_json::to_string(&constraint)
                    .map_err(|e| anyhow!("Failed to serialize constraint: {}", e))?;
                new_schema
                    .properties
                    .insert(constraint_key, constraint_value);

                if let ColumnConstraint::Unique { columns } = &constraint {
                    for col_name in columns {
                        if !new_schema.columns.iter().any(|c| &c.name == col_name) {
                            return Err(anyhow!(
                                "Column '{}' not found for UNIQUE constraint",
                                col_name
                            ));
                        }
                    }
                    // Record the constraint where UNIQUENESS IS ACTUALLY ENFORCED.
                    //
                    // This used to write only `properties` and (when the constraint
                    // was named) `schema.indexes`. The enforcement path reads
                    // neither — it reads `relational_capabilities.unique_indexes`
                    // and `.constraints` — so an `ALTER TABLE … ADD CONSTRAINT …
                    // UNIQUE` was cataloged and then never enforced, and an
                    // *unnamed* one landed in `properties` alone. Writing the
                    // constraint form covers both, named or not.
                    if !new_schema
                        .relational_capabilities
                        .constraints
                        .iter()
                        .any(|c| {
                            matches!(c, ColumnConstraint::Unique { columns: existing }
                                          if same_column_set(existing, columns))
                        })
                    {
                        new_schema.relational_capabilities.constraints.push(
                            ColumnConstraint::Unique {
                                columns: columns.clone(),
                            },
                        );
                    }
                    if let Some(name) = constraint_name {
                        let unique_index =
                            CatalogIndex::new(name, columns.clone(), CatalogIndexType::BTree)
                                .unique();
                        new_schema.indexes.push(unique_index);
                    }
                }
            }
            SchemaChange::DropConstraint { constraint_name } => {
                // A constraint is identifiable by two different keys, and the
                // original code only understood one of them: the `properties` key
                // is `constraint:unique:<cols>` — built from COLUMNS, never the
                // constraint name — so `k.contains(constraint_name)` never matched
                // a UNIQUE, and its property row leaked on every drop.
                //
                // Resolve the target from whichever source names it, then purge
                // every location together. Doing this by-name-only would leave the
                // enforced `relational_capabilities` copy behind, i.e. a dropped
                // constraint that is still enforced.
                let named_index_columns: Option<Vec<String>> = new_schema
                    .indexes
                    .iter()
                    .find(|idx| &idx.name == constraint_name)
                    .map(|idx| idx.columns.clone());

                let mut keys_to_remove: Vec<String> = new_schema
                    .properties
                    .keys()
                    .filter(|k| {
                        k.starts_with("constraint:") && k.contains(constraint_name.as_str())
                    })
                    .cloned()
                    .collect();

                // Also match the column-derived key for the named index.
                if let Some(columns) = &named_index_columns {
                    let column_key = format!("constraint:unique:{}", columns.join(","));
                    if new_schema.properties.contains_key(&column_key)
                        && !keys_to_remove.contains(&column_key)
                    {
                        keys_to_remove.push(column_key);
                    }
                }

                if keys_to_remove.is_empty() && named_index_columns.is_none() {
                    return Err(anyhow!("Constraint '{}' not found", constraint_name));
                }

                // Columns of every UNIQUE this drop resolves to — from the named
                // index and from the property rows being removed.
                let mut dropped_unique_columns: Vec<Vec<String>> =
                    named_index_columns.into_iter().collect();
                for key in &keys_to_remove {
                    if let Some(value) = new_schema.properties.get(key)
                        && let Ok(ColumnConstraint::Unique { columns }) =
                            serde_json::from_str::<ColumnConstraint>(value)
                    {
                        dropped_unique_columns.push(columns);
                    }
                }

                for key in keys_to_remove {
                    new_schema.properties.remove(&key);
                }
                new_schema
                    .indexes
                    .retain(|idx| &idx.name != constraint_name);
                new_schema
                    .relational_capabilities
                    .unique_indexes
                    .retain(|idx| {
                        &idx.name != constraint_name
                            && !dropped_unique_columns
                                .iter()
                                .any(|dropped| same_column_set(dropped, &idx.columns))
                    });
                new_schema
                    .relational_capabilities
                    .constraints
                    .retain(|c| match c {
                        ColumnConstraint::Unique { columns } => !dropped_unique_columns
                            .iter()
                            .any(|dropped| same_column_set(dropped, columns)),
                        _ => true,
                    });
            }
            SchemaChange::PromotePropsKey {
                key,
                column_type,
                comment,
            } => {
                // Promoted column name: `props__<key>` (double underscore).
                let col_name = format!("props__{}", key);
                if new_schema.columns.iter().any(|c| c.name == col_name) {
                    return Err(anyhow!(
                        "Props key '{}' is already promoted to column '{}'",
                        key,
                        col_name
                    ));
                }
                // Promoted columns start at ID 100 to distinguish them from
                // canonical system columns (ID 1–9) and user columns (ID 10+).
                let promoted_id = new_schema
                    .columns
                    .iter()
                    .filter(|c| c.id >= 100)
                    .map(|c| c.id)
                    .max()
                    .unwrap_or(99)
                    + 1;
                let mut col = CatalogColumn::new(promoted_id, &col_name, column_type.clone());
                col.nullable = true;
                col.comment = comment.clone();
                col.properties
                    .insert("promoted_from_props".to_string(), key.clone());
                new_schema.columns.push(col);

                // Record the promotion so the compaction writer knows which
                // msgpack keys to route into the new typed column.
                new_schema
                    .props_auto_promotion
                    .promoted_keys
                    .insert(key.clone(), col_name);
            }
            SchemaChange::SetTableOption { key, value } => {
                // System-written properties (a collection's record and owner)
                // are never set through schema evolution.
                if crate::table_properties::is_reserved_table_property(key) {
                    anyhow::bail!("table property '{key}' is reserved and cannot be set");
                }
                match key.to_lowercase().as_str() {
                    "props_auto_promotion" => {
                        new_schema.props_auto_promotion.enabled =
                            matches!(value.to_lowercase().as_str(), "enabled" | "true" | "1");
                    }
                    _ => {
                        // Unknown options are stored as table properties so
                        // they round-trip without data loss.
                        new_schema.properties.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    // Backfill on touch (ADR-077 M1): converge the evolved schema on the canonical
    // shape BEFORE validating, so a legacy schema being modified is normalized
    // rather than rejected by the projection-agreement check below.
    normalize_identity(&mut new_schema);

    validate_schema(&new_schema)?;
    Ok(new_schema)
}

/// Returns true when widening `from` → `to` is lossless.
pub fn is_compatible_type_change(from: &ProximaType, to: &ProximaType) -> bool {
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (ProximaType::Int32, ProximaType::Int64)
            | (ProximaType::Int32, ProximaType::Float64)
            | (ProximaType::Int64, ProximaType::Float64)
            | (ProximaType::Float32, ProximaType::Float64)
            | (ProximaType::Int8, ProximaType::Int16)
            | (ProximaType::Int8, ProximaType::Int32)
            | (ProximaType::Int8, ProximaType::Int64)
            | (ProximaType::Int16, ProximaType::Int32)
            | (ProximaType::Int16, ProximaType::Int64)
    )
}

/// Returns the SQL type name for a catalog data type.
pub fn sql_type_name(data_type: &ProximaType) -> &'static str {
    match data_type {
        ProximaType::Boolean => "BOOLEAN",
        ProximaType::Int8 => "TINYINT",
        ProximaType::Int16 => "SMALLINT",
        ProximaType::Int32 => "INTEGER",
        ProximaType::Int64 => "BIGINT",
        ProximaType::Float32 => "REAL",
        ProximaType::Float64 => "DOUBLE PRECISION",
        ProximaType::String => "TEXT",
        ProximaType::Binary => "BYTEA",
        ProximaType::Date => "DATE",
        ProximaType::Timestamp(_) => "TIMESTAMP",
        ProximaType::TimestampTz(_) => "TIMESTAMP WITH TIME ZONE",
        ProximaType::Time(_) => "TIME",
        ProximaType::Uuid => "UUID",
        ProximaType::Json => "JSONB",
        ProximaType::Decimal { .. } => "DECIMAL",
        ProximaType::DenseVector { .. } => "VECTOR",
        ProximaType::SparseVector { .. } => "SPARSE_VECTOR",
        ProximaType::BinaryVector { .. } => "BINARY_VECTOR",
        // Richer ProximaType variants without a dedicated catalog SQL name
        // (unsigned ints, Float16, Symbol, Jsonb, Array, Map, Struct,
        // Interval, Duration, ULID, geo, Null) fall back to TEXT.
        _ => "TEXT",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CatalogColumn, CatalogPhysicalFormat, CatalogProjection, CatalogStorageLayout,
        CatalogTableSchema,
    };
    use proximadb_data_model::{TimeUnit, VectorElement};

    /// Dimensionless dense vector placeholder (real dim lives in column props).
    fn vector_type() -> ProximaType {
        ProximaType::DenseVector {
            element: VectorElement::Float32,
            dim: 0,
        }
    }

    fn base_schema() -> CatalogTableSchema {
        CatalogTableSchema::new("t")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "name", ProximaType::String))
    }

    // ---- TD-CAT-6: identity is recorded in several places; evolution must
    // maintain all of them, and enforcement must be able to see what DDL wrote.

    /// The exact field set the live enforcement path reads
    /// (`services::record_store::schema_unique_column_sets`): unique_indexes ++
    /// `Unique` constraints. Mirrored here so these tests fail if evolution ever
    /// writes a UNIQUE somewhere enforcement cannot see it.
    fn enforceable_unique_sets(schema: &CatalogTableSchema) -> Vec<Vec<String>> {
        schema
            .relational_capabilities
            .unique_indexes
            .iter()
            .map(|i| i.columns.clone())
            .chain(
                schema
                    .relational_capabilities
                    .constraints
                    .iter()
                    .filter_map(|c| match c {
                        ColumnConstraint::Unique { columns } => Some(columns.clone()),
                        _ => None,
                    }),
            )
            .collect()
    }

    fn add_unique(name: Option<&str>, columns: &[&str]) -> CatalogSchemaEvolution {
        CatalogSchemaEvolution {
            changes: vec![SchemaChange::AddConstraint {
                constraint_name: name.map(str::to_string),
                constraint: ColumnConstraint::Unique {
                    columns: columns.iter().map(|c| c.to_string()).collect(),
                },
            }],
        }
    }

    #[test]
    fn alter_add_unique_constraint_is_visible_to_enforcement() {
        // THE BUG: this landed in `properties` + `schema.indexes`, while
        // enforcement reads only `relational_capabilities`. The constraint was
        // cataloged and then never enforced.
        let evolved = apply_evolution(&base_schema(), &add_unique(Some("uq_name"), &["name"]))
            .expect("evolution applies");

        assert_eq!(
            enforceable_unique_sets(&evolved),
            vec![vec!["name".to_string()]],
            "an ALTER-added UNIQUE must reach the fields enforcement reads"
        );
    }

    #[test]
    fn an_unnamed_alter_unique_constraint_is_also_enforced() {
        // Worse case: without a constraint name, nothing was written even to
        // `schema.indexes` — only the `properties` blob, which nothing reads.
        let evolved =
            apply_evolution(&base_schema(), &add_unique(None, &["name"])).expect("applies");
        assert_eq!(
            enforceable_unique_sets(&evolved),
            vec![vec!["name".to_string()]]
        );
    }

    #[test]
    fn adding_the_same_unique_twice_does_not_double_fence_it() {
        let once = apply_evolution(&base_schema(), &add_unique(Some("uq_name"), &["name"]))
            .expect("applies");
        // Re-declared with reversed column order — the same tuple, not a new key.
        let twice =
            apply_evolution(&once, &add_unique(Some("uq_name2"), &["name"])).expect("applies");
        assert_eq!(enforceable_unique_sets(&twice).len(), 1);
    }

    #[test]
    fn dropping_a_unique_constraint_stops_enforcing_it() {
        let added = apply_evolution(&base_schema(), &add_unique(Some("uq_name"), &["name"]))
            .expect("applies");
        assert_eq!(enforceable_unique_sets(&added).len(), 1);

        let dropped = apply_evolution(
            &added,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropConstraint {
                    constraint_name: "uq_name".to_string(),
                }],
            },
        )
        .expect("applies");
        assert!(
            enforceable_unique_sets(&dropped).is_empty(),
            "a dropped UNIQUE must stop being enforced"
        );
    }

    // ---- TD-CAT-6 slice 1: canonical discovery across all four routes ----

    // ---- ADR-077 M1: canonical + projection, kept in sync by construction ----

    #[test]
    fn normalize_folds_a_legacy_unique_into_the_canonical_field() {
        // A pre-ADR-077 schema: the UNIQUE lives only in the projection.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        assert!(schema.relational_capabilities.constraints.is_empty());

        normalize_identity(&mut schema);

        assert_eq!(
            enforceable_unique_sets(&schema),
            vec![vec!["name".to_string()], vec!["name".to_string()]],
            "canonical now carries it (and the projection is preserved)"
        );
        assert!(
            schema
                .relational_capabilities
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::Unique { columns } if columns == &vec!["name".to_string()])),
            "the canonical field must hold the declaration"
        );
        // Lossless: the projection keeps its index name and type.
        assert_eq!(
            schema.relational_capabilities.unique_indexes[0].name,
            "uq_name"
        );
    }

    #[test]
    fn normalize_is_idempotent() {
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        normalize_identity(&mut schema);
        let once = schema.clone();
        normalize_identity(&mut schema);
        assert_eq!(
            schema.relational_capabilities.constraints.len(),
            once.relational_capabilities.constraints.len(),
            "running normalize twice must not duplicate the canonical entry"
        );
    }

    #[test]
    fn evolution_backfills_a_legacy_schema_on_touch() {
        // The migration story: a legacy schema converges on canonical simply by
        // being modified — no offline pass.
        let mut legacy = base_schema().with_primary_key(vec!["id".to_string()]);
        legacy.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];

        let evolved = apply_evolution(
            &legacy,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::AddColumn {
                    name: "extra".to_string(),
                    data_type: ProximaType::String,
                    nullable: true,
                    default_value: None,
                    comment: None,
                    after: None,
                }],
            },
        )
        .expect("an unrelated evolution still normalizes identity");

        assert!(
            evolved
                .relational_capabilities
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::Unique { .. })),
            "touching the schema backfills the canonical field"
        );
    }

    #[test]
    fn validate_rejects_a_unique_the_canonical_field_does_not_carry() {
        // The defect ADR-077 makes unrepresentable: a UNIQUE in the projection
        // only would be cataloged and never enforced.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        let err = validate_schema(&schema).expect_err("a projection-only UNIQUE must be rejected");
        assert!(err.to_string().contains("never enforced"));

        // …and normalizing makes it valid.
        normalize_identity(&mut schema);
        validate_schema(&schema).expect("normalized schema validates");
    }

    #[test]
    fn validate_allows_canonical_without_a_projection() {
        // The reverse direction is legal: an UNNAMED UNIQUE has no index form to
        // project into, and enforcement reads canonical anyway.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.constraints = vec![ColumnConstraint::Unique {
            columns: vec!["name".to_string()],
        }];
        validate_schema(&schema).expect("canonical-only is the target shape");
    }

    #[test]
    fn with_unique_maintains_canonical_and_projection_together() {
        let schema = base_schema()
            .with_primary_key(vec!["id".to_string()])
            .with_unique("uq_name", vec!["name".to_string()]);

        // Canonical carries it (so it is enforced)…
        assert!(
            schema
                .relational_capabilities
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::Unique { .. }))
        );
        // …and the projection carries it (so introspection renders an index)…
        assert_eq!(schema.relational_capabilities.unique_indexes.len(), 1);
        // …and the result is valid by construction.
        validate_schema(&schema).expect("the builder cannot produce a drifted schema");
    }

    #[test]
    fn with_unique_is_idempotent_and_order_insensitive() {
        let schema = base_schema()
            .with_column(CatalogColumn::new(3, "email", ProximaType::String))
            .with_primary_key(vec!["id".to_string()])
            .with_unique("uq_a", vec!["name".to_string(), "email".to_string()])
            .with_unique("uq_b", vec!["email".to_string(), "name".to_string()]);
        assert_eq!(
            schema.relational_capabilities.constraints.len(),
            1,
            "the same tuple in a different order is one key, not two"
        );
        assert_eq!(schema.relational_capabilities.unique_indexes.len(), 1);
    }

    #[test]
    fn adr077_legacy_and_canonical_schemas_are_mutually_readable() {
        // ADR-077's amended migration rests on this: the collapse needs no
        // phased format migration because neither direction can fail to parse.

        // (a) A schema written by a PRE-collapse binary — UNIQUE recorded only in
        // `unique_indexes`, plus a populated `relational_capabilities.primary_key`
        // — deserializes under the current struct.
        let legacy = r#"{
            "name": "t", "columns": [], "primary_key": ["id"], "indexes": [],
            "schema_version": 1, "properties": {}, "location": null,
            "created_at_ms": 0, "updated_at_ms": 0,
            "relational_capabilities": {
                "primary_key": ["id"],
                "unique_indexes": [{"name":"uq","columns":["email"],
                                    "index_type":"BTree","is_unique":false,"properties":{}}],
                "secondary_indexes": [], "constraints": [],
                "materialized_views": [], "transaction_profile": null,
                "schema_evolution_policy": null
            }
        }"#;
        let parsed: CatalogTableSchema =
            serde_json::from_str(legacy).expect("a pre-collapse schema must still deserialize");
        assert_eq!(parsed.relational_capabilities.unique_indexes.len(), 1);

        // …and the canonical accessor recovers the declaration from that legacy
        // shape — which is what normalize-on-load will do at the load boundary.
        assert_eq!(
            crate::schema::table_identity(&parsed).unique_sets,
            vec![vec!["email".to_string()]]
        );

        // (b) A schema written by a POST-collapse binary — the fact recorded only
        // in `constraints`, legacy locations absent entirely — also deserializes.
        let canonical = r#"{
            "name": "t", "columns": [], "primary_key": ["id"], "indexes": [],
            "schema_version": 1, "properties": {}, "location": null,
            "created_at_ms": 0, "updated_at_ms": 0,
            "relational_capabilities": {
                "primary_key": [], "unique_indexes": [], "secondary_indexes": [],
                "constraints": [{"Unique":{"columns":["email"]}}],
                "materialized_views": [], "transaction_profile": null,
                "schema_evolution_policy": null
            }
        }"#;
        let parsed: CatalogTableSchema =
            serde_json::from_str(canonical).expect("a post-collapse schema must deserialize");
        assert_eq!(
            crate::schema::table_identity(&parsed).unique_sets,
            vec![vec!["email".to_string()]]
        );

        // (c) The struct tolerates an entirely absent relational_capabilities —
        // #[serde(default)] — which is how most creation paths leave it.
        let minimal = r#"{
            "name": "t", "columns": [], "primary_key": [], "indexes": [],
            "schema_version": 1, "properties": {}, "location": null,
            "created_at_ms": 0, "updated_at_ms": 0
        }"#;
        let parsed: CatalogTableSchema =
            serde_json::from_str(minimal).expect("absent capabilities must default");
        assert!(parsed.relational_capabilities.unique_indexes.is_empty());
    }

    #[test]
    fn table_identity_finds_a_unique_from_every_route() {
        let mut schema = base_schema()
            .with_column(CatalogColumn::new(3, "email", ProximaType::String))
            .with_column(CatalogColumn::new(4, "slug", ProximaType::String))
            .with_primary_key(vec!["id".to_string()]);

        // route 1: inline UNIQUE(...) at CREATE TABLE
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        // route 2: a Unique constraint
        schema.relational_capabilities.constraints = vec![ColumnConstraint::Unique {
            columns: vec!["email".to_string()],
        }];
        // route 3: a unique index on the top-level vector
        schema = schema.with_index(
            CatalogIndex::new("uq_slug", vec!["slug".to_string()], CatalogIndexType::BTree)
                .unique(),
        );

        let identity = table_identity(&schema);
        let mut found = identity.unique_sets.clone();
        found.sort();
        assert_eq!(
            found,
            vec![
                vec!["email".to_string()],
                vec!["name".to_string()],
                vec!["slug".to_string()],
            ],
            "every route a UNIQUE can arrive by must be discovered"
        );
        assert_eq!(identity.primary_key, vec!["id".to_string()]);
        assert!(identity.gaps.is_empty());
    }

    #[test]
    fn table_identity_recovers_a_unique_from_the_legacy_property_blob() {
        // Route 4: a schema evolved BEFORE the AddConstraint fix recorded the
        // constraint only in `properties`. Reading it keeps those schemas from
        // silently losing their UNIQUE.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.properties.insert(
            "constraint:unique:name".to_string(),
            serde_json::to_string(&ColumnConstraint::Unique {
                columns: vec!["name".to_string()],
            })
            .expect("serializes"),
        );
        assert_eq!(
            table_identity(&schema).unique_sets,
            vec![vec!["name".to_string()]]
        );
    }

    #[test]
    fn table_identity_deduplicates_the_same_key_across_routes() {
        // One DDL `UNIQUE` routinely lands in more than one field; and column
        // order does not make a second key.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        schema.relational_capabilities.constraints = vec![ColumnConstraint::Unique {
            columns: vec!["name".to_string()],
        }];
        schema = schema.with_index(
            CatalogIndex::new(
                "uq_name2",
                vec!["name".to_string()],
                CatalogIndexType::BTree,
            )
            .unique(),
        );
        assert_eq!(table_identity(&schema).unique_sets.len(), 1);
    }

    #[test]
    fn a_unique_index_entry_is_not_filtered_on_is_unique() {
        // Entries in `unique_indexes` are unique by virtue of the field they sit
        // in; DDL leaves `is_unique` false. Filtering on it drops every DDL UNIQUE.
        let idx = CatalogIndex::new("uq_name", vec!["name".to_string()], CatalogIndexType::BTree);
        assert!(!idx.is_unique, "precondition: DDL leaves the flag unset");
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.unique_indexes = vec![idx];
        assert_eq!(table_identity(&schema).unique_sets.len(), 1);
    }

    #[test]
    fn secondary_unique_sets_excludes_one_restating_the_primary_key() {
        // Discovery reports it; the *policy* of excluding it belongs to the
        // caller — the PK already fences that tuple.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.constraints = vec![
            ColumnConstraint::Unique {
                columns: vec!["id".to_string()],
            },
            ColumnConstraint::Unique {
                columns: vec!["name".to_string()],
            },
        ];
        let identity = table_identity(&schema);
        assert_eq!(identity.unique_sets.len(), 2, "discovery reports both");
        assert_eq!(
            identity.secondary_unique_sets(),
            vec![vec!["name".to_string()]],
            "only the genuinely secondary one is a secondary"
        );
    }

    #[test]
    fn table_identity_falls_back_to_the_capabilities_primary_key() {
        let mut schema = base_schema();
        schema.relational_capabilities.primary_key = vec!["id".to_string()];
        assert!(schema.primary_key.is_empty());
        assert_eq!(table_identity(&schema).primary_key, vec!["id".to_string()]);
    }

    #[test]
    fn table_identity_reports_a_missing_primary_key_as_a_gap() {
        let identity = table_identity(&base_schema());
        assert!(identity.primary_key.is_empty());
        assert!(identity.gaps.iter().any(|g| g.contains("no primary key")));
    }

    #[test]
    fn an_alter_added_unique_reaches_the_identity_accessor() {
        // End-to-end across S0 + S1: the evolution fix records it, and the
        // canonical accessor finds it.
        let evolved = apply_evolution(
            &base_schema().with_primary_key(vec!["id".to_string()]),
            &add_unique(Some("uq_name"), &["name"]),
        )
        .expect("applies");
        assert_eq!(
            table_identity(&evolved).secondary_unique_sets(),
            vec![vec!["name".to_string()]]
        );
    }

    #[test]
    fn dropping_a_unique_constraint_does_not_leak_its_property_row() {
        // The `properties` key is `constraint:unique:<cols>` — derived from
        // COLUMNS, never the constraint name — so the by-name filter never
        // matched it and the row leaked on every drop.
        let added = apply_evolution(&base_schema(), &add_unique(Some("uq_name"), &["name"]))
            .expect("applies");
        assert!(
            added
                .properties
                .keys()
                .any(|k| k.starts_with("constraint:unique:")),
            "precondition: AddConstraint records a property row"
        );

        let dropped = apply_evolution(
            &added,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropConstraint {
                    constraint_name: "uq_name".to_string(),
                }],
            },
        )
        .expect("applies");

        assert!(
            !dropped
                .properties
                .keys()
                .any(|k| k.starts_with("constraint:unique:")),
            "the property row must be removed with the constraint"
        );
        assert!(dropped.indexes.is_empty());
    }

    #[test]
    fn dropping_an_unknown_constraint_still_errors() {
        let err = apply_evolution(
            &base_schema(),
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropConstraint {
                    constraint_name: "nope".to_string(),
                }],
            },
        )
        .expect_err("an unknown constraint must be rejected");
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn renaming_a_primary_key_column_updates_every_copy() {
        // `relational::effective_primary_key` PREFERS relational_capabilities, so
        // leaving that copy stale made row validation resolve a column that no
        // longer exists.
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.primary_key = vec!["id".to_string()];
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_id",
            vec!["id".to_string()],
            CatalogIndexType::BTree,
        )];

        let renamed = apply_evolution(
            &schema,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::RenameColumn {
                    old_name: "id".to_string(),
                    new_name: "pk".to_string(),
                }],
            },
        )
        .expect("applies");

        assert_eq!(renamed.primary_key, vec!["pk".to_string()]);
        assert_eq!(
            renamed.relational_capabilities.primary_key,
            vec!["pk".to_string()],
            "the relational_capabilities PK copy went stale"
        );
        assert_eq!(
            renamed.relational_capabilities.unique_indexes[0].columns,
            vec!["pk".to_string()]
        );
        // And the result must still validate — a stale copy now fails the invariant.
        validate_schema(&renamed).expect("renamed schema is self-consistent");
    }

    #[test]
    fn a_primary_key_column_cannot_be_dropped_via_either_copy() {
        let mut schema = base_schema();
        // PK declared ONLY in relational_capabilities — this used to pass the guard.
        schema.relational_capabilities.primary_key = vec!["id".to_string()];

        let err = apply_evolution(
            &schema,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropColumn {
                    name: "id".to_string(),
                }],
            },
        )
        .expect_err("dropping a primary key column must be rejected");
        assert!(err.to_string().contains("primary key"));
    }

    #[test]
    fn dropping_a_column_purges_it_from_the_capabilities_copies() {
        let mut schema = base_schema();
        schema.relational_capabilities.unique_indexes = vec![CatalogIndex::new(
            "uq_name",
            vec!["name".to_string()],
            CatalogIndexType::BTree,
        )];
        schema.relational_capabilities.constraints = vec![ColumnConstraint::Unique {
            columns: vec!["name".to_string()],
        }];

        let dropped = apply_evolution(
            &schema,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropColumn {
                    name: "name".to_string(),
                }],
            },
        )
        .expect("applies");

        assert!(
            enforceable_unique_sets(&dropped).is_empty(),
            "a dropped column must not be left referenced by an enforced constraint"
        );
        validate_schema(&dropped).expect("no dangling column references remain");
    }

    #[test]
    fn validate_rejects_a_disagreeing_primary_key() {
        let mut schema = base_schema().with_primary_key(vec!["id".to_string()]);
        schema.relational_capabilities.primary_key = vec!["name".to_string()];
        let err = validate_schema(&schema).expect_err("disagreeing PK copies must be rejected");
        assert!(err.to_string().contains("primary key disagrees"));
    }

    #[test]
    fn validate_accepts_an_empty_capabilities_primary_key() {
        // Only CREATE TABLE populates it, so most creation paths leave it empty.
        let schema = base_schema().with_primary_key(vec!["id".to_string()]);
        assert!(schema.relational_capabilities.primary_key.is_empty());
        validate_schema(&schema).expect("an unset capabilities PK stays legal");
    }

    #[test]
    fn validate_rejects_a_dangling_capabilities_column() {
        let mut schema = base_schema();
        schema.relational_capabilities.constraints = vec![ColumnConstraint::Unique {
            columns: vec!["ghost".to_string()],
        }];
        let err = validate_schema(&schema).expect_err("dangling column must be rejected");
        assert!(err.to_string().contains("ghost"));
    }

    #[test]
    fn validate_ok() {
        let schema = base_schema();
        assert!(validate_schema(&schema).is_ok());
    }

    #[test]
    fn rejects_reserved_system_column_names() {
        let schema = base_schema().with_column(CatalogColumn::new(
            3,
            "__proxima_deleted",
            ProximaType::Boolean,
        ));
        let err = validate_schema(&schema).expect_err("reserved column should fail");
        assert!(err.to_string().contains("reserved"));
    }

    #[test]
    fn validates_external_layout_contract() {
        let schema = base_schema().with_storage_layout(CatalogStorageLayout::federated_read(
            "raw",
            CatalogPhysicalFormat::Parquet,
            "s3://bucket/raw/",
        ));
        assert!(validate_schema(&schema).is_ok());
    }

    #[test]
    fn validate_empty_name_fails() {
        let mut schema = base_schema();
        schema.name = String::new();
        assert!(validate_schema(&schema).is_err());
    }

    #[test]
    fn add_column_roundtrip() {
        let schema = base_schema();
        let evolution = CatalogSchemaEvolution {
            changes: vec![SchemaChange::AddColumn {
                name: "age".to_string(),
                data_type: ProximaType::Int32,
                nullable: true,
                default_value: None,
                comment: None,
                after: None,
            }],
        };
        let new = apply_evolution(&schema, &evolution).unwrap();
        assert_eq!(new.columns.len(), 3);
        assert_eq!(new.schema_version, schema.schema_version + 1);
    }

    #[test]
    fn type_widening_compatible() {
        assert!(is_compatible_type_change(
            &ProximaType::Int32,
            &ProximaType::Int64
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Float32,
            &ProximaType::Float64
        ));
        assert!(!is_compatible_type_change(
            &ProximaType::String,
            &ProximaType::Int64
        ));
    }

    #[test]
    fn validate_schema_rejects_structural_storage_and_projection_contract_violations() {
        let mut empty_columns = CatalogTableSchema::new("t");
        empty_columns.columns.clear();
        assert!(
            validate_schema(&empty_columns)
                .unwrap_err()
                .to_string()
                .contains("at least one column")
        );

        let empty_column_name =
            CatalogTableSchema::new("t").with_column(CatalogColumn::new(1, "", ProximaType::Int64));
        assert!(
            validate_schema(&empty_column_name)
                .unwrap_err()
                .to_string()
                .contains("Column name cannot be empty")
        );

        let duplicate = base_schema().with_column(CatalogColumn::new(3, "id", ProximaType::Int64));
        assert!(
            validate_schema(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("Duplicate column")
        );

        let missing_pk = base_schema().with_primary_key(vec!["missing".to_string()]);
        assert!(
            validate_schema(&missing_pk)
                .unwrap_err()
                .to_string()
                .contains("Primary key column")
        );

        let missing_index_col = base_schema().with_index(CatalogIndex::new(
            "bad_idx",
            vec!["missing".to_string()],
            CatalogIndexType::BTree,
        ));
        assert!(
            validate_schema(&missing_index_col)
                .unwrap_err()
                .to_string()
                .contains("references non-existent column")
        );

        let vector_missing_dimension =
            base_schema().with_column(CatalogColumn::new(3, "embedding", vector_type()));
        assert!(
            validate_schema(&vector_missing_dimension)
                .unwrap_err()
                .to_string()
                .contains("dimension")
        );

        let mut vector_col = CatalogColumn::new(3, "embedding", vector_type());
        vector_col
            .properties
            .insert("dimension".to_string(), "384".to_string());
        assert!(validate_schema(&base_schema().with_column(vector_col)).is_ok());

        let mut external = CatalogStorageLayout::federated_read(
            "raw",
            CatalogPhysicalFormat::Parquet,
            "s3://bucket/raw",
        );
        external.location = None;
        assert!(
            validate_schema(&base_schema().with_storage_layout(external))
                .unwrap_err()
                .to_string()
                .contains("must declare a location")
        );

        let mut external = CatalogStorageLayout::federated_read(
            "raw",
            CatalogPhysicalFormat::Parquet,
            "s3://bucket/raw",
        );
        external.snapshot_semantics = None;
        assert!(
            validate_schema(&base_schema().with_storage_layout(external))
                .unwrap_err()
                .to_string()
                .contains("snapshot semantics")
        );

        let mut projection = CatalogProjection::rebuildable(
            "ann",
            crate::CatalogProjectionKind::VectorAnn,
            "primary",
        );
        projection.rebuild_source.clear();
        assert!(
            validate_schema(&base_schema().with_projection(projection))
                .unwrap_err()
                .to_string()
                .contains("rebuild source")
        );
    }

    #[test]
    fn apply_evolution_covers_column_mutation_and_error_paths() {
        let schema = base_schema()
            .with_primary_key(vec!["id".to_string()])
            .with_index(CatalogIndex::new(
                "name_idx",
                vec!["name".to_string()],
                CatalogIndexType::BTree,
            ));

        let evolved = apply_evolution(
            &schema,
            &CatalogSchemaEvolution {
                changes: vec![
                    SchemaChange::AddColumn {
                        name: "age".to_string(),
                        data_type: ProximaType::Int32,
                        nullable: true,
                        default_value: Some("0".to_string()),
                        comment: Some("Age".to_string()),
                        after: Some("id".to_string()),
                    },
                    SchemaChange::UpdateComment {
                        name: "age".to_string(),
                        comment: "Years".to_string(),
                    },
                    SchemaChange::MakeNotNullable {
                        name: "age".to_string(),
                    },
                    SchemaChange::SetDefault {
                        name: "age".to_string(),
                        default_value: "18".to_string(),
                    },
                    SchemaChange::MakeNullable {
                        name: "age".to_string(),
                    },
                    SchemaChange::DropDefault {
                        name: "age".to_string(),
                    },
                    SchemaChange::MoveColumn {
                        name: "age".to_string(),
                        after: None,
                    },
                    SchemaChange::RenameColumn {
                        old_name: "name".to_string(),
                        new_name: "display_name".to_string(),
                    },
                    SchemaChange::ChangeType {
                        name: "age".to_string(),
                        new_type: ProximaType::Int64,
                    },
                ],
            },
        )
        .unwrap();

        assert_eq!(evolved.columns[0].name, "age");
        let age = evolved
            .columns
            .iter()
            .find(|col| col.name == "age")
            .unwrap();
        assert_eq!(age.data_type, ProximaType::Int64);
        assert!(age.nullable);
        assert_eq!(age.default_value, None);
        assert_eq!(age.comment.as_deref(), Some("Years"));
        assert_eq!(evolved.indexes[0].columns, vec!["display_name"]);

        let dropped = apply_evolution(
            &evolved,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropColumn {
                    name: "display_name".to_string(),
                }],
            },
        )
        .unwrap();
        assert!(!dropped.columns.iter().any(|col| col.name == "display_name"));
        assert!(dropped.indexes.is_empty());

        for change in [
            SchemaChange::AddColumn {
                name: "id".to_string(),
                data_type: ProximaType::Int64,
                nullable: false,
                default_value: None,
                comment: None,
                after: None,
            },
            SchemaChange::DropColumn {
                name: "id".to_string(),
            },
            SchemaChange::RenameColumn {
                old_name: "missing".to_string(),
                new_name: "x".to_string(),
            },
            SchemaChange::ChangeType {
                name: "name".to_string(),
                new_type: ProximaType::Binary,
            },
            SchemaChange::MoveColumn {
                name: "name".to_string(),
                after: Some("missing".to_string()),
            },
        ] {
            assert!(
                apply_evolution(
                    &schema,
                    &CatalogSchemaEvolution {
                        changes: vec![change]
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn set_table_option_refuses_reserved_keys_in_any_case() {
        let schema = base_schema();
        let set = |key: &str| {
            apply_evolution(
                &schema,
                &CatalogSchemaEvolution {
                    changes: vec![SchemaChange::SetTableOption {
                        key: key.to_string(),
                        value: "x".to_string(),
                    }],
                },
            )
        };
        for key in [
            "owner",
            "OWNER",
            "tags",
            "Tags",
            "collection.id",
            "Asset.Kind",
            "vector.dimension",
        ] {
            assert!(set(key).is_err(), "{key} must be refused");
        }
        let evolved = set("custom_option").expect("ordinary option");
        assert_eq!(
            evolved.properties.get("custom_option").map(String::as_str),
            Some("x")
        );
    }

    #[test]
    fn apply_evolution_covers_constraints_props_promotion_and_table_options() {
        let schema = base_schema();
        let evolved = apply_evolution(
            &schema,
            &CatalogSchemaEvolution {
                changes: vec![
                    SchemaChange::AddConstraint {
                        constraint_name: Some("uniq_name".to_string()),
                        constraint: ColumnConstraint::Unique {
                            columns: vec!["name".to_string()],
                        },
                    },
                    SchemaChange::AddConstraint {
                        constraint_name: Some("check_name".to_string()),
                        constraint: ColumnConstraint::Check {
                            expression: "name <> ''".to_string(),
                        },
                    },
                    SchemaChange::AddConstraint {
                        constraint_name: Some("fk_name".to_string()),
                        constraint: ColumnConstraint::ForeignKey {
                            columns: vec!["name".to_string()],
                            references_table: "other".to_string(),
                            references_columns: vec!["name".to_string()],
                            on_delete: None,
                            on_update: None,
                        },
                    },
                    SchemaChange::PromotePropsKey {
                        key: "status".to_string(),
                        column_type: ProximaType::String,
                        comment: Some("Promoted status".to_string()),
                    },
                    SchemaChange::SetTableOption {
                        key: "props_auto_promotion".to_string(),
                        value: "enabled".to_string(),
                    },
                    SchemaChange::SetTableOption {
                        key: "custom_option".to_string(),
                        value: "custom_value".to_string(),
                    },
                ],
            },
        )
        .unwrap();

        assert!(
            evolved
                .indexes
                .iter()
                .any(|idx| idx.name == "uniq_name" && idx.is_unique)
        );
        assert!(
            evolved
                .properties
                .keys()
                .any(|key| key.starts_with("constraint:check"))
        );
        assert!(
            evolved
                .properties
                .keys()
                .any(|key| key.starts_with("constraint:fk"))
        );
        assert_eq!(
            evolved
                .props_auto_promotion
                .promoted_keys
                .get("status")
                .map(String::as_str),
            Some("props__status")
        );
        assert!(evolved.props_auto_promotion.enabled);
        assert_eq!(
            evolved.properties.get("custom_option").map(String::as_str),
            Some("custom_value")
        );

        let dropped = apply_evolution(
            &evolved,
            &CatalogSchemaEvolution {
                changes: vec![SchemaChange::DropConstraint {
                    constraint_name: "uniq_name".to_string(),
                }],
            },
        )
        .unwrap();
        assert!(!dropped.indexes.iter().any(|idx| idx.name == "uniq_name"));

        for change in [
            SchemaChange::AddConstraint {
                constraint_name: Some("bad_unique".to_string()),
                constraint: ColumnConstraint::Unique {
                    columns: vec!["missing".to_string()],
                },
            },
            SchemaChange::DropConstraint {
                constraint_name: "missing".to_string(),
            },
            SchemaChange::PromotePropsKey {
                key: "status".to_string(),
                column_type: ProximaType::String,
                comment: None,
            },
        ] {
            assert!(
                apply_evolution(
                    &evolved,
                    &CatalogSchemaEvolution {
                        changes: vec![change]
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn sql_type_names_cover_every_catalog_type() {
        let names = [
            (ProximaType::Boolean, "BOOLEAN"),
            (ProximaType::Int8, "TINYINT"),
            (ProximaType::Int16, "SMALLINT"),
            (ProximaType::Int32, "INTEGER"),
            (ProximaType::Int64, "BIGINT"),
            (ProximaType::Float32, "REAL"),
            (ProximaType::Float64, "DOUBLE PRECISION"),
            (ProximaType::String, "TEXT"),
            (ProximaType::Binary, "BYTEA"),
            (ProximaType::Date, "DATE"),
            (ProximaType::Timestamp(TimeUnit::Nanosecond), "TIMESTAMP"),
            (
                ProximaType::TimestampTz(TimeUnit::Nanosecond),
                "TIMESTAMP WITH TIME ZONE",
            ),
            (ProximaType::Time(TimeUnit::Nanosecond), "TIME"),
            (ProximaType::Uuid, "UUID"),
            (ProximaType::Json, "JSONB"),
            (
                ProximaType::Decimal {
                    precision: 38,
                    scale: 10,
                },
                "DECIMAL",
            ),
            (vector_type(), "VECTOR"),
            (
                ProximaType::SparseVector {
                    element: VectorElement::Float32,
                },
                "SPARSE_VECTOR",
            ),
            (ProximaType::BinaryVector { dim: 0 }, "BINARY_VECTOR"),
        ];

        for (data_type, expected) in names {
            assert_eq!(sql_type_name(&data_type), expected);
        }

        assert!(is_compatible_type_change(
            &ProximaType::Int8,
            &ProximaType::Int16
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Int8,
            &ProximaType::Int32
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Int8,
            &ProximaType::Int64
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Int16,
            &ProximaType::Int32
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Int16,
            &ProximaType::Int64
        ));
        assert!(is_compatible_type_change(
            &ProximaType::Int64,
            &ProximaType::Float64
        ));
    }
}
