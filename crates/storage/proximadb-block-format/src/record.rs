// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! ProximaRecord ↔ PAX column projection.
//!
//! Defines the canonical mapping from `ProximaRecord` fields to PAX column
//! stripes (ADR-ICE-001, ADR-010). Every collection uses this fixed column
//! layout as the leading stripe set; user-defined columns follow starting at
//! `FIRST_USER_COLUMN_ID`.
//!
//! The same mapping governs the Iceberg REST `load table` column schema
//! (see `src/catalog/iceberg_rest_service.rs`), ensuring cross-engine
//! consistency: Spark/Trino/DuckDB see the same columns as PAX block readers.

use proximadb_records::ProximaRecord;
use serde::{Deserialize, Serialize};

use crate::reader::PaxBlockReader;
use crate::stripe::ColumnRole;

/// Canonical column IDs for ProximaRecord fields in PAX blocks.
/// These are stable and must not change between format versions.
pub mod col_id {
    pub const OID: i32 = 0;
    pub const TENANT_ID: i32 = 1;
    pub const CREATED_AT: i32 = 2;
    pub const UPDATED_AT: i32 = 3;
    pub const VALID_FROM: i32 = 4;
    pub const VALID_TO: i32 = 5;
    pub const ACTOR: i32 = 6;
    pub const ORIGIN: i32 = 7;
    /// Props column (ID 8) — the canonical **durable-core overflow carrier**
    /// (msgpack-serialized `ProximaTree` bytes).
    ///
    /// Per ADR-045 §Decision (Layer A) and ADR-010 §Props Opacity Limitation: any
    /// value a typed PAX column cannot represent — exotic numerics (NaN/±Inf,
    /// signaling-NaN/subnormal, oversized decimals), non-UTF8 binary, deeply-nested
    /// or extension-shaped payloads — flows through this stripe **losslessly**. It is
    /// part of the durable open core (a columnar stripe, immutable, ships 1:1 to
    /// Iceberg), NOT rebuildable serving metadata. The stripe is a verbatim
    /// length-prefixed byte copy, hence byte-transparent for arbitrary msgpack — see
    /// `props_stripe_round_trips_losslessly_for_exotic_payloads` (the TD-LTAP-1
    /// parity ratchet the durable-core/serving-metadata split must preserve).
    pub const PROPS: i32 = 8;
    /// Labels column (ID 9) — msgpack-serialized `LabelSet` bytes; a secondary
    /// opaque durable-core carrier alongside `PROPS`.
    pub const LABELS: i32 = 9;
    pub const EDGE_SRC: i32 = 10;
    pub const EDGE_TGT: i32 = 11;
    pub const EDGE_TYPE: i32 = 12;
    pub const EDGE_WEIGHT: i32 = 13;
    /// Canonical MVCC sequence. Absent in older PAX blocks and decoded as zero
    /// (the resolver's legacy version-1 sentinel), making this additive stripe
    /// mixed-read-safe.
    pub const RECORD_VERSION: i32 = 14;
    /// Shred directory (ID 15) — the block's OWN record of which prop key each
    /// `USER_BASE`+ stripe carries, as msgpack `Vec<(col_id, prop_key)>`
    /// (TD-USUB-6 slice 2a).
    ///
    /// **Why the block must name its own columns.** `FlatRow::into_record`
    /// otherwise names user columns from the CALLER's `user_column_keys`,
    /// *positionally* — so a caller list that is short, stale or reordered
    /// silently drops values. That is harmless while the msgpack `PROPS` tail is
    /// complete (the tail already holds every value, so the typed stripes are
    /// advisory). It becomes silent data loss the moment the tail is reduced to a
    /// residual, which is the whole point of making declared columns
    /// authoritative — and schema evolution is exactly when a caller's key list
    /// and the block's stripes disagree.
    ///
    /// **This entry carries NO stripe** (`stripe_len == 0`). Its payload lives in
    /// the footer-extras region, addressed by the `ColumnMeta` pointer pair
    /// `bloom_offset`/`bloom_len` reused as a generic footer-payload pointer —
    /// which is what lets the directory be **additive in v2** with no format
    /// version bump, no footer growth, and no new `ColumnRole` byte (it declares
    /// `ColumnRole::Props`, since the payload is opaque msgpack). Both the
    /// 32-byte `BlockFooter` and the 64-byte `ColumnMeta` are fully consumed, so
    /// there was no spare field to grow instead.
    ///
    /// Readers locate columns by `find(|m| m.column_id == X)`, so a reader that
    /// does not know this id simply never asks for it.
    pub const SHRED_DIRECTORY: i32 = 15;
    /// First column ID for embedding stripes (embedding_0, embedding_1, …).
    pub const EMBED_BASE: i32 = 20;
    /// First column ID for user-defined columns from CatalogTableSchema.
    pub const USER_BASE: i32 = 100;
    /// First column ID for co-located rerank stripes (SQ8 rerank_0, rerank_1, …).
    ///
    /// When a collection writes RaBitQ-quantized embedding stripes (the hot
    /// candidate-scan representation), the writer ALSO emits an SQ8-quantized
    /// copy of the same embedding at `RERANK_BASE + i`. The cascade reranks the
    /// RaBitQ candidate pool against this co-located SQ8 column (4× footprint, no
    /// extra GET against an external f32 tier) before taking the final top-k;
    /// f32 rerank is added only if the recall gate can't be met on SQ8. Chosen
    /// well above `USER_BASE` so it never collides with catalog-driven columns.
    pub const RERANK_BASE: i32 = 1000;
    /// First column ID for the OPTIONAL exact-f32 tier (f32_tier_0, …).
    ///
    /// Opt-in only (`pax_f32_tier:on` tag / `PROXIMADB_PAX_F32_TIER` env, default
    /// OFF): when enabled on a RaBitQ collection, the writer ALSO emits the raw
    /// f32 embedding at `F32_TIER_BASE + i`. The column is read LAZILY — normal
    /// id+score queries never touch it (zero scan/egress cost); it is decoded
    /// only for an exact final rerank (→ recall ≈ 1.0) or `include_vectors`. This
    /// is the storage/egress trade for exact-vector fidelity. Well above
    /// `RERANK_BASE` so it never collides.
    pub const F32_TIER_BASE: i32 = 2000;
}

/// Descriptor for a single column stripe in a PAX block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDescriptor {
    pub column_id: i32,
    pub name: String,
    pub role: ColumnRole,
    pub nullable: bool,
}

/// Returns the canonical fixed column descriptors for every ProximaDB collection.
///
/// User-defined columns from `CatalogTableSchema` are appended after these.
/// Embedding columns are generated dynamically based on the registered models.
pub fn canonical_columns() -> Vec<ColumnDescriptor> {
    vec![
        ColumnDescriptor {
            column_id: col_id::OID,
            name: "id".into(),
            role: ColumnRole::Identity,
            nullable: false,
        },
        ColumnDescriptor {
            column_id: col_id::TENANT_ID,
            name: "tenant_id".into(),
            role: ColumnRole::Tenant,
            nullable: false,
        },
        ColumnDescriptor {
            column_id: col_id::CREATED_AT,
            name: "created_at".into(),
            role: ColumnRole::Timestamp,
            nullable: false,
        },
        ColumnDescriptor {
            column_id: col_id::UPDATED_AT,
            name: "updated_at".into(),
            role: ColumnRole::Timestamp,
            nullable: false,
        },
        ColumnDescriptor {
            column_id: col_id::VALID_FROM,
            name: "valid_from".into(),
            role: ColumnRole::Temporal,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::VALID_TO,
            name: "valid_to".into(),
            role: ColumnRole::Temporal,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::ACTOR,
            name: "actor".into(),
            role: ColumnRole::Provenance,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::ORIGIN,
            name: "origin".into(),
            role: ColumnRole::Provenance,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::PROPS,
            name: "props".into(),
            role: ColumnRole::Props,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::LABELS,
            name: "labels".into(),
            role: ColumnRole::Props,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::EDGE_SRC,
            name: "edge_source_id".into(),
            role: ColumnRole::Edge,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::EDGE_TGT,
            name: "edge_target_id".into(),
            role: ColumnRole::Edge,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::EDGE_TYPE,
            name: "edge_type".into(),
            role: ColumnRole::Edge,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::EDGE_WEIGHT,
            name: "edge_weight".into(),
            role: ColumnRole::Edge,
            nullable: true,
        },
        ColumnDescriptor {
            column_id: col_id::RECORD_VERSION,
            name: "record_version".into(),
            role: ColumnRole::Temporal,
            nullable: false,
        },
    ]
}

/// Flat row representation extracted from a `ProximaRecord` for column encoding.
///
/// Each field corresponds directly to one column stripe. Absent optional values
/// contribute a null entry to the null bitmap.
#[derive(Debug, Clone)]
pub struct FlatRow {
    /// Record identifier (oid). Never null.
    pub oid: String,
    /// Tenant identifier. Never null; engine-level RLS.
    pub tenant_id: String,
    pub created_at_ns: i64,
    pub updated_at_ns: i64,
    pub record_version: u64,
    pub valid_from_ns: Option<i64>,
    pub valid_to_ns: Option<i64>,
    pub actor: Option<String>,
    pub origin: Option<String>,
    /// msgpack-serialised props tree bytes.
    pub props_bytes: Option<Vec<u8>>,
    /// msgpack-serialised label list bytes.
    pub labels_bytes: Option<Vec<u8>>,
    pub edge_src: Option<String>,
    pub edge_tgt: Option<String>,
    pub edge_type: Option<String>,
    pub edge_weight: Option<f64>,
    /// One entry per embedding (model order from record).
    pub embeddings: Vec<Vec<f32>>,
    /// User-defined columns projected from props or other sources.
    pub user_columns: Vec<Option<proximadb_data_model::ProximaValue>>,
    /// Shredded props rebuilt from typed stripes and NAMED by the block's own
    /// shred directory (TD-USUB-6). Populated only by [`Self::from_block_reader`]
    /// for blocks that carry a directory; empty otherwise.
    ///
    /// Kept separate from `user_columns` because these are named by the BLOCK,
    /// not positionally by the caller — which is precisely what makes a residual
    /// tail safe to read.
    pub shredded: Vec<(String, proximadb_data_model::ProximaValue)>,
}

impl FlatRow {
    /// Extract a `FlatRow` from a `ProximaRecord`.
    ///
    /// `props` and `labels` are serialised to msgpack bytes so they can be
    /// stored opaquely in the Props column stripe.
    pub fn from_record(record: &ProximaRecord) -> anyhow::Result<Self> {
        Self::from_record_with_user_columns(record, &[])
    }

    /// Extract a `FlatRow` from a `ProximaRecord`, projecting specific keys from `props`
    /// into the `user_columns` list.
    /// Build a row whose msgpack `PROPS` tail OMITS `residual_keys` (TD-USUB-6 2b).
    ///
    /// The caller has already established that each listed key's value is
    /// reproducible EXACTLY from its typed stripe under the declared type, so
    /// dropping it from the tail loses nothing — that removal is the
    /// write-amplification win, since today every shredded value is stored twice.
    ///
    /// Unlike [`Self::from_record_with_user_columns`], this does NOT populate
    /// `user_columns`: the PAX writer buffers shredded values separately, and the
    /// reader rebuilds them from the block's own shred directory rather than from
    /// a positional caller list.
    pub fn from_record_with_residual(
        record: &ProximaRecord,
        residual_keys: &[&str],
    ) -> anyhow::Result<Self> {
        let mut flat = Self::from_record(record)?;
        if residual_keys.is_empty() {
            return Ok(flat);
        }
        let mut props = record.props.clone();
        for key in residual_keys {
            props.remove(*key);
        }
        flat.props_bytes = if props.is_empty() {
            None
        } else {
            Some(rmp_serde::to_vec_named(&props)?)
        };
        Ok(flat)
    }

    pub fn from_record_with_user_columns(
        record: &ProximaRecord,
        user_column_keys: &[String],
    ) -> anyhow::Result<Self> {
        let mut props = record.props.clone();

        // Extract promoted columns from props
        let mut user_columns = Vec::with_capacity(user_column_keys.len());
        for key in user_column_keys {
            let val = match props.remove(key) {
                Some(proximadb_records::ProximaTreeNode::Value(v)) => Some(v),
                _ => None,
            };
            user_columns.push(val);
        }

        let props_bytes = if props.is_empty() {
            None
        } else {
            Some(rmp_serde::to_vec_named(&props)?)
        };

        let labels_vec: Vec<&str> = record.labels.iter().map(|s| s.as_str()).collect();
        let labels_bytes = if labels_vec.is_empty() {
            None
        } else {
            Some(rmp_serde::to_vec_named(&labels_vec)?)
        };

        let (edge_src, edge_tgt, edge_type, edge_weight) = match &record.edge {
            Some(e) => (
                Some(e.source_id.clone()),
                Some(e.target_id.clone()),
                Some(e.edge_type.clone()),
                e.weight,
            ),
            None => (None, None, None, None),
        };

        // INT-2.5b: FlatRow holds fp32 vectors; promote non-Fp32
        // variants on the way in. INT-3's PAX writer will receive
        // typed values directly via a different code path that
        // bypasses FlatRow.
        let embeddings: Vec<Vec<f32>> = record
            .embeddings
            .iter()
            .map(|emb| emb.values.to_fp32_owned())
            .collect();

        Ok(FlatRow {
            shredded: Vec::new(),
            oid: record.oid.clone(),
            tenant_id: record.tenant_id.clone(),
            created_at_ns: record.created_at_ns,
            updated_at_ns: record.updated_at_ns,
            record_version: record.record_version,
            valid_from_ns: record.valid_from_ns,
            valid_to_ns: record.valid_to_ns,
            actor: record.actor.clone(),
            origin: record.origin.clone(),
            props_bytes,
            labels_bytes,
            edge_src,
            edge_tgt,
            edge_type,
            edge_weight,
            embeddings,
            user_columns,
        })
    }

    /// Reconstruct a `ProximaRecord` from a decoded `FlatRow`.
    ///
    /// Embedding model IDs and user column keys are provided externally (from the collection schema).
    /// Reconstruct a [`ProximaRecord`]. `tenant_ctx` is the segment's owning tenant
    /// (resolved from the catalog/path); it is stamped onto `tenant_id` only when the
    /// row carries no stored tenant — i.e. a segment written with the tenant column
    /// dropped (catalog-resolution). Segments that still store the column keep their
    /// own value, so this is mixed-read-safe (old and new segments both reconstruct
    /// the correct tenant). Pass `None` to keep the stored value verbatim.
    /// Decode this row's props msgpack into a `ProximaTree` WITHOUT building a
    /// full `ProximaRecord` — the filter-aware cascade's Stage-F row predicate
    /// (ADR-089 P1) only needs props for `evaluate_filter_proxima`.
    pub fn props_tree(&self) -> anyhow::Result<proximadb_records::ProximaTree> {
        match &self.props_bytes {
            Some(b) => Ok(rmp_serde::from_slice(b)?),
            None => Ok(Default::default()),
        }
    }

    pub fn into_record(
        self,
        embedding_model_ids: &[String],
        user_column_keys: &[String],
        tenant_ctx: Option<&str>,
    ) -> anyhow::Result<ProximaRecord> {
        use proximadb_records::{EdgeShape, EmbeddingCell, LabelSet};
        use std::collections::HashMap;

        let mut props = match self.props_bytes {
            Some(b) => rmp_serde::from_slice(&b)?,
            None => HashMap::new(),
        };

        // Merge user columns back into props
        for (i, val) in self.user_columns.into_iter().enumerate() {
            if let Some(key) = user_column_keys.get(i)
                && let Some(v) = val
            {
                props.insert(key.clone(), proximadb_records::ProximaTreeNode::Value(v));
            }
        }

        // TD-USUB-6: fill props the RESIDUAL tail no longer carries, from the
        // typed stripes, named by the block's own directory.
        //
        // The tail WINS wherever it still has the key. That ordering is the
        // safety property: the writer drops a value from the tail only when the
        // stripe reproduces it exactly, so a key still present in the tail is one
        // the stripe could NOT reproduce — and its (lossy) stripe value must
        // never overwrite the true one.
        for (key, value) in self.shredded {
            props
                .entry(key)
                .or_insert(proximadb_records::ProximaTreeNode::Value(value));
        }

        let labels: LabelSet = match self.labels_bytes {
            Some(b) => {
                let v: Vec<String> = rmp_serde::from_slice(&b)?;
                v.into()
            }
            None => LabelSet::new(),
        };

        let edge = match (self.edge_src, self.edge_tgt, self.edge_type) {
            (Some(src), Some(tgt), Some(etype)) => Some(EdgeShape {
                source_id: src,
                target_id: tgt,
                edge_type: etype,
                weight: self.edge_weight,
            }),
            _ => None,
        };

        let embeddings: Vec<EmbeddingCell> = self
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                let dim = values.len() as u32;
                // INT-2.5b: FlatRow stores fp32, wrap in the typed
                // variant on the way out.
                EmbeddingCell {
                    model_id: embedding_model_ids
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("model_{i}")),
                    modality: "dense".into(),
                    values: proximadb_records::EmbeddingValues::Fp32(values),
                    dim,
                    ..Default::default()
                }
            })
            .collect();

        // Catalog-resolution: a segment with the tenant column dropped yields an
        // empty stored tenant; stamp it from the segment's owning tenant context.
        let tenant_id = if self.tenant_id.is_empty() {
            tenant_ctx.unwrap_or_default().to_string()
        } else {
            self.tenant_id
        };

        Ok(ProximaRecord {
            oid: self.oid,
            tenant_id,
            created_at_ns: self.created_at_ns,
            updated_at_ns: self.updated_at_ns,
            record_version: self.record_version,
            valid_from_ns: self.valid_from_ns,
            valid_to_ns: self.valid_to_ns,
            actor: self.actor,
            origin: self.origin,
            props,
            labels,
            edge,
            embeddings,
            ..Default::default()
        })
    }

    /// Reconstruct every row of a decoded PAX block into [`FlatRow`]s — the
    /// reader-side inverse of the writer's per-row `add_record` + stripe flush.
    ///
    /// Decodes the canonical column stripes (`col_id::OID`/`TENANT_ID`/timestamps/
    /// provenance/`PROPS`/`LABELS`/`EDGE_*`) plus every contiguous embedding
    /// stripe from `EMBED_BASE`. Pair with [`FlatRow::into_record`] (which needs
    /// the embedding model ids + promoted user-column keys from the schema) to
    /// rebuild full `ProximaRecord`s.
    ///
    /// User-promoted columns (`USER_BASE`+) are not reconstructed here; the
    /// canonical PAX writer path (`from_record`) does not promote, so `PROPS`
    /// carries the full prop tree. Promoted-column reconstruction is a follow-up
    /// for the props-auto-promotion read path.
    pub fn from_block_reader(reader: &PaxBlockReader) -> anyhow::Result<Vec<FlatRow>> {
        let n = reader.row_count() as usize;
        // An absent stripe decodes to an empty Vec; every access below is via
        // `.get(i)` so a short/empty column simply yields `None` for each row.
        let oids = reader.decode_str_stripe(col_id::OID).unwrap_or_default();
        let tenants = reader
            .decode_str_stripe(col_id::TENANT_ID)
            .unwrap_or_default();
        let created = reader
            .decode_i64_stripe(col_id::CREATED_AT)
            .unwrap_or_default();
        let updated = reader
            .decode_i64_stripe(col_id::UPDATED_AT)
            .unwrap_or_default();
        let record_versions = reader
            .decode_u64_stripe(col_id::RECORD_VERSION)
            .map_err(|error| anyhow::anyhow!("record_version stripe: {error}"))?
            .unwrap_or_default();
        let valid_from = reader
            .decode_i64_stripe(col_id::VALID_FROM)
            .unwrap_or_default();
        let valid_to = reader
            .decode_i64_stripe(col_id::VALID_TO)
            .unwrap_or_default();
        let actors = reader.decode_str_stripe(col_id::ACTOR).unwrap_or_default();
        let origins = reader.decode_str_stripe(col_id::ORIGIN).unwrap_or_default();
        let props = reader
            .decode_bytes_stripe(col_id::PROPS)
            .unwrap_or_default();
        let labels = reader
            .decode_bytes_stripe(col_id::LABELS)
            .unwrap_or_default();
        let edge_src = reader
            .decode_str_stripe(col_id::EDGE_SRC)
            .unwrap_or_default();
        let edge_tgt = reader
            .decode_str_stripe(col_id::EDGE_TGT)
            .unwrap_or_default();
        let edge_type = reader
            .decode_str_stripe(col_id::EDGE_TYPE)
            .unwrap_or_default();
        let edge_weight = reader
            .decode_f64_stripe(col_id::EDGE_WEIGHT)
            .unwrap_or_default();

        // Embedding stripes are contiguous from EMBED_BASE; probe until absent.
        // When the independently declared exact tier is present, prefer it for
        // materialization. This is the compaction/rebuild authority boundary:
        // EMBED_BASE may be SQ8/RaBitQ and must not silently replace original
        // f32 merely because it is the primary scan stripe.
        let mut embedding_stripes: Vec<Vec<Option<Vec<f32>>>> = Vec::new();
        let mut e = 0;
        loop {
            let base = reader.decode_f32_vec_stripe(col_id::EMBED_BASE + e);
            let exact_column = col_id::F32_TIER_BASE + e;
            let exact = reader
                .vector_params()
                .get(exact_column)
                .filter(|entry| entry.quant_kind == crate::vparam::QUANT_RAW_F32)
                .and_then(|_| reader.decode_f32_vec_stripe(exact_column));
            let materialized = match (base, exact) {
                (None, None) => break,
                (None, Some(exact)) => exact,
                (Some(base), Some(exact)) if exact.len() == base.len() => base
                    .into_iter()
                    .zip(exact)
                    .map(|(approximate, exact)| exact.or(approximate))
                    .collect(),
                (Some(base), _) => base,
            };
            embedding_stripes.push(materialized);
            e += 1;
        }

        let at = |col: &[Option<String>], i: usize| col.get(i).cloned().flatten();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let embeddings: Vec<Vec<f32>> = embedding_stripes
                .iter()
                .filter_map(|stripe| stripe.get(i).cloned().flatten())
                .collect();
            rows.push(FlatRow {
                shredded: Vec::new(),
                oid: at(&oids, i).unwrap_or_default(),
                tenant_id: at(&tenants, i).unwrap_or_default(),
                created_at_ns: created.get(i).copied().flatten().unwrap_or(0),
                updated_at_ns: updated.get(i).copied().flatten().unwrap_or(0),
                record_version: record_versions.get(i).copied().unwrap_or(0),
                valid_from_ns: valid_from.get(i).copied().flatten(),
                valid_to_ns: valid_to.get(i).copied().flatten(),
                actor: at(&actors, i),
                origin: at(&origins, i),
                props_bytes: props.get(i).cloned().flatten(),
                labels_bytes: labels.get(i).cloned().flatten(),
                edge_src: at(&edge_src, i),
                edge_tgt: at(&edge_tgt, i),
                edge_type: at(&edge_type, i),
                edge_weight: edge_weight.get(i).copied().flatten(),
                embeddings,
                user_columns: Vec::new(),
            });
        }

        // TD-USUB-6: rebuild shredded props from the typed stripes, named and
        // typed by the block's OWN directory. Only columns with an EXACT declared
        // type are rebuilt — for an undeclared column the stripe holds one of
        // three physical classes with no way to know which variant to restore,
        // and its tail is complete anyway, so it is deliberately skipped.
        if let Some(dir) = reader.shred_directory() {
            for entry in dir {
                let Some(declared) = entry.declared_type.as_ref() else {
                    continue;
                };
                let Some(class) = crate::writer::declared_shred_class_pub(declared) else {
                    continue;
                };
                let (ints, floats, strs) = match class {
                    crate::writer::ShredClassPub::Int => {
                        (reader.decode_i64_stripe(entry.column_id), None, None)
                    }
                    crate::writer::ShredClassPub::Float => {
                        (None, reader.decode_f64_stripe(entry.column_id), None)
                    }
                    crate::writer::ShredClassPub::Str => {
                        (None, None, reader.decode_str_stripe(entry.column_id))
                    }
                };
                for (i, row) in rows.iter_mut().enumerate() {
                    let value = crate::writer::reconstruct_declared_pub(
                        class,
                        ints.as_ref().and_then(|v| v.get(i).copied().flatten()),
                        floats.as_ref().and_then(|v| v.get(i).copied().flatten()),
                        strs.as_ref().and_then(|v| v.get(i).cloned().flatten()),
                        declared,
                    );
                    if let Some(v) = value {
                        row.shredded.push((entry.key.clone(), v));
                    }
                }
            }
        }
        Ok(rows)
    }
}

/// Encode a string column value as raw bytes for the stripe.
pub fn encode_str_col(values: &[Option<&str>]) -> (Vec<u8>, u32) {
    // Variable-length layout: 4B length prefix per value; null = 0xFFFF_FFFF.
    let mut buf = Vec::new();
    let mut null_count = 0u32;
    for v in values {
        match v {
            Some(s) => {
                let bytes = s.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            None => {
                buf.extend_from_slice(&u32::MAX.to_le_bytes());
                null_count += 1;
            }
        }
    }
    (buf, null_count)
}

/// Encode an i64 column (delta-encoded for temporal columns).
pub fn encode_i64_col(values: &[Option<i64>]) -> (Vec<u8>, u32) {
    let mut buf = Vec::new();
    let mut null_count = 0u32;
    for v in values {
        match v {
            Some(n) => buf.extend_from_slice(&n.to_le_bytes()),
            None => {
                buf.extend_from_slice(&i64::MIN.to_le_bytes()); // sentinel
                null_count += 1;
            }
        }
    }
    (buf, null_count)
}

/// Encode an f32 vector embedding stripe.
pub fn encode_f32_vec_col(values: &[Option<&[f32]>]) -> (Vec<u8>, u32) {
    let mut buf = Vec::new();
    let mut null_count = 0u32;
    for v in values {
        match v {
            Some(floats) => {
                buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
                for &f in *floats {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            None => {
                buf.extend_from_slice(&u32::MAX.to_le_bytes());
                null_count += 1;
            }
        }
    }
    (buf, null_count)
}

/// Update `ColumnMeta.min_val` and `max_val` for an i64 column.
pub fn update_i64_bounds(meta: &mut crate::stripe::ColumnMeta, value: i64) {
    let cur_min = i64::from_le_bytes(meta.min_val[0..8].try_into().unwrap_or([0; 8]));
    let cur_max = i64::from_le_bytes(meta.max_val[0..8].try_into().unwrap_or([0; 8]));

    let new_min = if cur_min == 0 && cur_max == 0 {
        value
    } else {
        cur_min.min(value)
    };
    let new_max = if cur_min == 0 && cur_max == 0 {
        value
    } else {
        cur_max.max(value)
    };

    meta.min_val[0..8].copy_from_slice(&new_min.to_le_bytes());
    meta.max_val[0..8].copy_from_slice(&new_max.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        header::{BlockCompression, BlockMode},
        reader::PaxBlockReader,
        writer::PaxBlockWriter,
    };
    use proximadb_records::{EdgeShape, EmbeddingCell, LabelSet, ProximaRecord, ProximaTreeNode};
    use std::collections::HashMap;

    /// Build a richly populated ProximaRecord with all field types.
    fn rich_record(oid: &str) -> ProximaRecord {
        let mut props = HashMap::new();
        props.insert(
            "score".into(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::Float64(0.95)),
        );
        props.insert(
            "tag".into(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String("hello".into())),
        );

        let mut labels = LabelSet::new();
        labels.insert("ml");
        labels.insert("test");

        ProximaRecord {
            oid: oid.into(),
            tenant_id: "tenant_x".into(),
            created_at_ns: 1_000_000,
            updated_at_ns: 2_000_000,
            valid_from_ns: Some(500_000),
            valid_to_ns: Some(9_000_000),
            actor: Some("agent-1".into()),
            origin: Some("rest-api".into()),
            props,
            labels,
            edge: Some(EdgeShape {
                source_id: "node_a".into(),
                target_id: "node_b".into(),
                edge_type: "knows".into(),
                weight: Some(0.75),
            }),
            embeddings: vec![EmbeddingCell {
                model_id: "text-embed-v1".into(),
                modality: "text".into(),
                values: proximadb_records::EmbeddingValues::Fp32(vec![0.1, 0.2, 0.3, 0.4]),
                dim: 4,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn flat_row_from_record_preserves_all_fields() {
        let rec = rich_record("r1");
        let flat = FlatRow::from_record(&rec).unwrap();

        assert_eq!(flat.oid, "r1");
        assert_eq!(flat.tenant_id, "tenant_x");
        assert_eq!(flat.created_at_ns, 1_000_000);
        assert_eq!(flat.valid_from_ns, Some(500_000));
        assert_eq!(flat.valid_to_ns, Some(9_000_000));
        assert_eq!(flat.actor.as_deref(), Some("agent-1"));
        assert_eq!(flat.origin.as_deref(), Some("rest-api"));
        assert!(flat.props_bytes.is_some(), "props should be serialised");
        assert!(flat.labels_bytes.is_some(), "labels should be serialised");
        assert_eq!(flat.edge_src.as_deref(), Some("node_a"));
        assert_eq!(flat.edge_tgt.as_deref(), Some("node_b"));
        assert_eq!(flat.edge_type.as_deref(), Some("knows"));
        assert_eq!(flat.edge_weight, Some(0.75));
        assert_eq!(flat.embeddings.len(), 1);
        assert_eq!(flat.embeddings[0], vec![0.1f32, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn pax_block_record_round_trip() {
        let records = vec![rich_record("r1"), rich_record("r2")];

        // Write
        let mut writer = PaxBlockWriter::new(
            BlockMode::Pax,
            BlockCompression::None,
            "collection_rt",
            0,
            1,
        );
        for r in &records {
            writer.add_record(r).unwrap();
        }
        let block_bytes = writer.flush().unwrap();

        // Read back field-by-field via stripe decoders
        let reader = PaxBlockReader::open(&block_bytes).unwrap();
        assert_eq!(reader.row_count(), 2);

        let oids = reader.decode_str_stripe(col_id::OID).unwrap();
        assert_eq!(oids[0].as_deref(), Some("r1"));
        assert_eq!(oids[1].as_deref(), Some("r2"));

        let actors = reader.decode_str_stripe(col_id::ACTOR).unwrap();
        assert_eq!(actors[0].as_deref(), Some("agent-1"));

        let valid_from = reader.decode_i64_stripe(col_id::VALID_FROM).unwrap();
        assert_eq!(valid_from[0], Some(500_000));

        let edge_src = reader.decode_str_stripe(col_id::EDGE_SRC).unwrap();
        assert_eq!(edge_src[0].as_deref(), Some("node_a"));

        let embeddings = reader.decode_f32_vec_stripe(col_id::EMBED_BASE).unwrap();
        let emb0 = embeddings[0].as_ref().unwrap();
        assert!((emb0[0] - 0.1f32).abs() < 1e-6);
        assert!((emb0[3] - 0.4f32).abs() < 1e-6);
    }

    #[test]
    fn record_version_stripe_is_exact_and_mixed_read_safe() {
        let versions = [1, u32::MAX as u64 + 17, u64::MAX - 1];
        let mut writer = PaxBlockWriter::new(
            BlockMode::Pax,
            BlockCompression::None,
            "collection_versions",
            0,
            1,
        )
        .with_record_version(true);
        for (row, version) in versions.into_iter().enumerate() {
            let mut record = rich_record(&format!("record-{row}"));
            record.record_version = version;
            writer.add_record(&record).unwrap();
        }
        let block = writer.flush().unwrap();
        let reader = PaxBlockReader::open(&block).unwrap();
        assert_eq!(
            reader.decode_u64_stripe(col_id::RECORD_VERSION).unwrap(),
            Some(versions.to_vec())
        );
        let decoded = FlatRow::from_block_reader(&reader).unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(|row| row.record_version)
                .collect::<Vec<_>>(),
            versions
        );

        let mut legacy_writer = PaxBlockWriter::new(
            BlockMode::Pax,
            BlockCompression::None,
            "collection_legacy",
            0,
            1,
        )
        .with_record_version(false);
        legacy_writer.add_record(&rich_record("legacy")).unwrap();
        let legacy_block = legacy_writer.flush().unwrap();
        let legacy_reader = PaxBlockReader::open(&legacy_block).unwrap();
        assert_eq!(
            legacy_reader
                .decode_u64_stripe(col_id::RECORD_VERSION)
                .unwrap(),
            None
        );
        assert_eq!(
            FlatRow::from_block_reader(&legacy_reader).unwrap()[0].record_version,
            0,
            "an absent stripe must preserve the resolver's legacy sentinel"
        );
    }

    /// Catalog-resolution: with the flag on, the per-row tenant stripe is dropped
    /// and `into_record` stamps tenant from the segment's catalog/path context. The
    /// block-header RLS hash is retained. With the flag off (default) the stripe is
    /// present, used, and a context argument never overrides it (mixed-read safety).
    #[test]
    fn tenant_col_dropped_and_stamped_from_context() {
        let rec = |oid: &str, ts: i64| ProximaRecord {
            oid: oid.into(),
            tenant_id: "tenant-A".into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        };

        // Flag ON: the tenant stripe is omitted entirely.
        let mut w = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0)
            .with_drop_tenant_col(true);
        w.add_record(&rec("a", 1)).unwrap();
        w.add_record(&rec("b", 2)).unwrap();
        let block = w.flush().unwrap();
        let reader = PaxBlockReader::open(&block).unwrap();
        assert!(
            !reader
                .column_metas()
                .iter()
                .any(|m| m.column_id == col_id::TENANT_ID),
            "tenant stripe must be dropped when the flag is on"
        );
        // The block-header RLS skip is still derived from the tenant.
        assert!(
            reader
                .header()
                .tenant_matches(crate::header::fnv1a_hash("tenant-A")),
            "block-header tenant hash must survive dropping the column"
        );
        // Reading back stamps tenant from the catalog/path context.
        for flat in FlatRow::from_block_reader(&reader).unwrap() {
            let record = flat.into_record(&[], &[], Some("tenant-A")).unwrap();
            assert_eq!(record.tenant_id, "tenant-A");
        }

        // Flag OFF (default): the stripe is present and used; a different context
        // never overrides the stored value (old segments stay correct).
        let mut w2 = PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "c", 0, 0);
        w2.add_record(&rec("a", 1)).unwrap();
        let block2 = w2.flush().unwrap();
        let reader2 = PaxBlockReader::open(&block2).unwrap();
        assert!(
            reader2
                .column_metas()
                .iter()
                .any(|m| m.column_id == col_id::TENANT_ID),
            "tenant stripe present by default"
        );
        let flat2 = FlatRow::from_block_reader(&reader2).unwrap().remove(0);
        let record2 = flat2.into_record(&[], &[], Some("tenant-OTHER")).unwrap();
        assert_eq!(
            record2.tenant_id, "tenant-A",
            "a stored tenant must not be overridden by context"
        );
    }

    #[test]
    fn olap_block_no_row_directory() {
        let mut writer =
            PaxBlockWriter::new(BlockMode::Olap, BlockCompression::None, "col_olap", 0, 0);
        writer
            .add_record(&ProximaRecord {
                oid: "x".into(),
                tenant_id: "t".into(),
                created_at_ns: 1,
                updated_at_ns: 1,
                ..Default::default()
            })
            .unwrap();
        let block_bytes = writer.flush().unwrap();
        let reader = PaxBlockReader::open(&block_bytes).unwrap();
        assert!(
            reader.row_directory().unwrap().is_none(),
            "OLAP block must not have row directory"
        );
    }

    /// TD-LTAP-1 / ADR-045 Layer-A parity ratchet: the `props` stripe (column ID 8)
    /// is the durable-open-core carrier for any value a typed PAX column cannot
    /// represent (the Lakebase "structured-overflow field" analog — ADR-045 §Decision
    /// Layer A; ADR-010 §Props Opacity Limitation). The stripe is a verbatim
    /// length-prefixed byte copy, so it MUST survive a full writer→reader round-trip
    /// byte-identical for arbitrary msgpack — including the exotic payloads no typed
    /// column holds (NaN/±Inf, signaling-NaN/subnormal, oversized Decimal, nested
    /// trees, non-UTF8 binary, Json). This is the lossless baseline the
    /// durable-core/serving-metadata format split (TD-LTAP-1) must preserve.
    ///
    /// Follow-up (not v1): raw msgpack *extension-type* bytes cannot be injected
    /// through `add_record` (it rebuilds `props_bytes` from the `ProximaValue` tree,
    /// which has no raw-ext variant); a stripe-level ext-bytes assertion can be added
    /// separately. The `Binary` payload below already proves byte-transparency.
    #[test]
    fn props_stripe_round_trips_losslessly_for_exotic_payloads() {
        use proximadb_data_model::ProximaValue;

        let mut props: HashMap<String, ProximaTreeNode> = HashMap::new();
        // 1. IEEE-754 exotics (compared via to_bits — NaN != NaN under PartialEq).
        props.insert(
            "nan".into(),
            ProximaTreeNode::Value(ProximaValue::Float64(f64::NAN)),
        );
        props.insert(
            "inf".into(),
            ProximaTreeNode::Value(ProximaValue::Float64(f64::INFINITY)),
        );
        props.insert(
            "neg_inf".into(),
            ProximaTreeNode::Value(ProximaValue::Float64(f64::NEG_INFINITY)),
        );
        props.insert(
            "snan".into(),
            ProximaTreeNode::Value(ProximaValue::Float64(f64::from_bits(0x7FF0_0000_0000_0001))),
        );
        props.insert(
            "subnormal".into(),
            ProximaTreeNode::Value(ProximaValue::Float64(f64::from_bits(0x0000_0000_0000_0001))),
        );
        // 2. Oversized decimal beyond any fixed-width numeric column.
        props.insert(
            "big_decimal".into(),
            ProximaTreeNode::Value(ProximaValue::Decimal(
                "99999999999999999999999999.999".into(),
            )),
        );
        // 3. Deeply-nested object tree (~8 levels) — the natural props nesting shape.
        let mut nested = ProximaTreeNode::Value(ProximaValue::String("leaf".into()));
        for _ in 0..8 {
            let mut inner = HashMap::new();
            inner.insert("down".into(), nested);
            nested = ProximaTreeNode::Object(inner);
        }
        props.insert("nested".into(), nested);
        // 4. Non-UTF8 / NUL binary — proves the stripe is byte-transparent.
        props.insert(
            "raw".into(),
            ProximaTreeNode::Value(ProximaValue::Binary(vec![0x00, 0xFF, 0xC0, 0xFE])),
        );
        // 5. Structured Json.
        props.insert(
            "json".into(),
            ProximaTreeNode::Value(ProximaValue::Json(serde_json::json!({
                "k": [1, 2, { "nested": true }]
            }))),
        );

        let rec = ProximaRecord {
            oid: "exotic-1".into(),
            tenant_id: "tenant_x".into(),
            created_at_ns: 1,
            updated_at_ns: 1,
            props: props.clone(),
            ..Default::default()
        };

        // Single canonical serialization — deterministic within the run (the writer
        // serializes a clone of this same HashMap with the same RandomState).
        let expected_props_bytes = rmp_serde::to_vec_named(&rec.props).unwrap();

        // Full writer → reader round-trip.
        let mut writer =
            PaxBlockWriter::new(BlockMode::Pax, BlockCompression::None, "col_props", 0, 0);
        writer.add_record(&rec).unwrap();
        let block = writer.flush().unwrap();

        let reader = PaxBlockReader::open(&block).unwrap();
        assert_eq!(reader.row_count(), 1);

        // PRIMARY: the durable-core props stripe is a verbatim copy → byte-identical
        // to the canonical serialization. This alone proves losslessness for every
        // payload above (NaN bits, oversized decimal, non-UTF8 binary, nesting, Json).
        let stripe = reader
            .decode_bytes_stripe(col_id::PROPS)
            .expect("props stripe must be present");
        assert_eq!(stripe.len(), 1, "one row written ⇒ one props slot");
        assert_eq!(
            stripe[0].as_ref().expect("row 0 props non-null"),
            &expected_props_bytes,
            "props stripe must round-trip byte-identical (lossless durable-core carrier)",
        );

        // SECONDARY: reconstruct the record and confirm the trickiest exotics survive
        // (NaN-aware — PartialEq would say NaN != NaN).
        let rec2 = FlatRow::from_block_reader(&reader)
            .unwrap()
            .remove(0)
            .into_record(&[], &[], None)
            .unwrap();
        assert_eq!(rec2.props.len(), rec.props.len(), "all prop keys survived");
        match rec2.props.get("nan") {
            Some(ProximaTreeNode::Value(ProximaValue::Float64(v))) => {
                assert_eq!(v.to_bits(), f64::NAN.to_bits(), "NaN bit pattern preserved");
            }
            other => panic!("nan did not round-trip as Float64: {other:?}"),
        }
        match rec2.props.get("big_decimal") {
            Some(ProximaTreeNode::Value(ProximaValue::Decimal(s))) => {
                assert_eq!(
                    s, "99999999999999999999999999.999",
                    "oversized decimal preserved verbatim",
                );
            }
            other => panic!("big_decimal did not round-trip: {other:?}"),
        }
        match rec2.props.get("raw") {
            Some(ProximaTreeNode::Value(ProximaValue::Binary(b))) => {
                assert_eq!(
                    b,
                    &vec![0x00u8, 0xFF, 0xC0, 0xFE],
                    "non-UTF8 binary preserved"
                );
            }
            other => panic!("raw did not round-trip: {other:?}"),
        }
    }
}
