//! TD-SPINE-4 (ADR-098 D4.3): manifests are planning data.
//!
//! Two pieces, both pure functions of an **immutable** Iceberg metadata
//! version:
//!
//! * [`ManifestPlanCache`] — the parsed snapshot (data files, decoded deletion
//!   vectors, per-file manifest bounds) cached in a byte-weighted,
//!   tenant-fair [`TenantCache`], keyed by the store base plus the LISTed
//!   revision of the metadata object (path, e-tag, last-modified, size). The
//!   latest-version LIST still runs on every open, so freshness and
//!   read-after-write are unchanged (mandate #16); a hit saves the
//!   `metadata.json`, manifest-list, manifest and deletion-vector reads. A
//!   dropped and re-created table restarts at v0 under a NEW object revision,
//!   so it can never hit the old table's entry.
//! * [`FileManifestStats`] — the manifest's per-file `lower_bounds` /
//!   `upper_bounds` / `null_value_counts`, decoded for the conservative subset
//!   of Iceberg types, and [`FileManifestStats::prunes`], which proves that a
//!   pushed-down filter matches NO row of the file. Conservative by
//!   construction: a missing bound keeps the file; a string upper bound of 16
//!   or more characters may be a `truncate(16)` artefact and is never used
//!   (this writer drops truncated upper bounds, a foreign one may not); a
//!   literal whose type is not the column's exact family keeps the file (no
//!   cross-type comparison, no casts); float/double/timestamp/decimal/boolean
//!   columns are never pruned here (NaN ordering and unit ambiguity).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use datafusion::common::ScalarValue as DfScalarValue;
use datafusion::logical_expr::{Expr, Operator};
use proximadb_cache::{CacheBudget, CacheKey, CacheKind, TenantCache};
use proximadb_iceberg_engine::deletion_vector::PositionDeletes;
use proximadb_iceberg_engine::iceberg::{ColumnFileStats, IcebergField};
use proximadb_storage_common::format_splits::{ColumnBounds, ScalarPredicate, ScalarValue};

/// Kill-switch for the manifest cache (unset = cache on).
pub(crate) const MANIFEST_CACHE_DISABLE_ENV: &str = "PROXIMADB_OLAP_MANIFEST_CACHE_DISABLE";
/// Kill-switch for manifest-bound file pruning (unset = pruning on).
pub(crate) const MANIFEST_PRUNE_DISABLE_ENV: &str = "PROXIMADB_OLAP_MANIFEST_PRUNE_DISABLE";

/// Pool size of the process-wide manifest cache.
const MANIFEST_CACHE_BYTES: u64 = 64 * 1024 * 1024;
/// Per-tenant hard ceiling inside the pool (fairness).
const MANIFEST_CACHE_TENANT_CEILING_BYTES: u64 = 16 * 1024 * 1024;

/// Iceberg's `truncate(16)` metrics length: a string bound this long may be
/// truncated.
const TRUNCATE_LEN: usize = 16;

fn disabled(env: &str) -> bool {
    proximadb_iceberg_engine::retention::disable_flag(std::env::var(env).ok().as_deref())
}

pub(crate) fn manifest_cache_enabled() -> bool {
    !disabled(MANIFEST_CACHE_DISABLE_ENV)
}

pub(crate) fn manifest_prune_enabled() -> bool {
    !disabled(MANIFEST_PRUNE_DISABLE_ENV)
}

/// The bound family a column prunes in; anything else is not decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Iceberg `int` / `long`.
    Integer,
    /// Iceberg `date` (days since epoch).
    Date,
    /// Iceberg `string`.
    Text,
}

impl Family {
    fn of(type_name: &str) -> Option<Self> {
        match type_name {
            "int" | "long" => Some(Self::Integer),
            "date" => Some(Self::Date),
            "string" => Some(Self::Text),
            _ => None,
        }
    }

    /// The literal as a comparable value, only when its DataFusion type is
    /// exactly this family (no cross-type comparison).
    fn literal(self, value: &DfScalarValue) -> Option<ScalarValue> {
        match (self, value) {
            (Self::Integer, DfScalarValue::Int8(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::Int16(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::Int32(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::Int64(Some(v))) => Some(ScalarValue::Int64(*v)),
            (Self::Integer, DfScalarValue::UInt8(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::UInt16(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::UInt32(Some(v))) => Some(ScalarValue::Int64(*v as i64)),
            (Self::Integer, DfScalarValue::UInt64(Some(v))) => {
                i64::try_from(*v).ok().map(ScalarValue::Int64)
            }
            (Self::Date, DfScalarValue::Date32(Some(days))) => {
                Some(ScalarValue::Int64(*days as i64))
            }
            (
                Self::Text,
                DfScalarValue::Utf8(Some(s))
                | DfScalarValue::Utf8View(Some(s))
                | DfScalarValue::LargeUtf8(Some(s)),
            ) => Some(ScalarValue::String(s.clone())),
            _ => None,
        }
    }

    /// Decode an Iceberg single-value binary bound (Appendix D).
    fn decode(self, type_name: &str, bytes: &[u8], is_upper: bool) -> Option<serde_json::Value> {
        match (self, type_name) {
            (Self::Integer, "int") | (Self::Date, "date") => {
                let raw: [u8; 4] = bytes.try_into().ok()?;
                Some(serde_json::json!(i32::from_le_bytes(raw) as i64))
            }
            (Self::Integer, "long") => {
                let raw: [u8; 8] = bytes.try_into().ok()?;
                Some(serde_json::json!(i64::from_le_bytes(raw)))
            }
            (Self::Text, _) => {
                let s = std::str::from_utf8(bytes).ok()?;
                // A truncated lower bound is a prefix (still <= every value);
                // a possibly-truncated upper bound is never trusted.
                if is_upper && s.chars().count() >= TRUNCATE_LEN {
                    return None;
                }
                Some(serde_json::json!(s))
            }
            _ => None,
        }
    }
}

/// One column's decoded manifest metrics for one data file.
#[derive(Debug, Clone)]
struct ColumnManifestBounds {
    family: Family,
    bounds: ColumnBounds,
    /// `null_value_counts`, when the manifest carries it.
    null_count: Option<i64>,
}

/// A data file's manifest metrics, by column name.
#[derive(Debug, Clone, Default)]
pub(crate) struct FileManifestStats {
    columns: HashMap<String, ColumnManifestBounds>,
}

impl FileManifestStats {
    /// Decode one file's manifest metrics against the schema the metrics'
    /// field ids refer to (the same metadata version's current schema).
    pub(crate) fn from_manifest(fields: &[IcebergField], stats: &[ColumnFileStats]) -> Self {
        let by_id: HashMap<i32, &IcebergField> = fields.iter().map(|f| (f.id, f)).collect();
        let mut columns = HashMap::new();
        for stat in stats {
            let Some(field) = by_id.get(&stat.field_id) else {
                continue;
            };
            let Some(family) = Family::of(&field.type_name) else {
                continue;
            };
            let decode = |b: &Option<Vec<u8>>, upper| {
                b.as_deref()
                    .and_then(|b| family.decode(&field.type_name, b, upper))
            };
            columns.insert(
                field.name.clone(),
                ColumnManifestBounds {
                    family,
                    bounds: ColumnBounds {
                        min: decode(&stat.lower_bound, false),
                        max: decode(&stat.upper_bound, true),
                        // Never read by the comparisons below; IS NULL uses
                        // `null_count` (which may be unknown).
                        null_count: u64::MAX,
                        distinct_count: None,
                    },
                    null_count: stat.null_value_count,
                },
            );
        }
        Self { columns }
    }

    /// Approximate heap bytes, for the cache weight.
    fn weight(&self) -> usize {
        self.columns
            .iter()
            .map(|(name, c)| {
                let bound = |v: &Option<serde_json::Value>| {
                    v.as_ref().and_then(|v| v.as_str()).map_or(8, str::len)
                };
                name.len() + 64 + bound(&c.bounds.min) + bound(&c.bounds.max)
            })
            .sum()
    }

    /// True when the AND of `filters` provably matches no row of this file.
    pub(crate) fn prunes(&self, filters: &[Expr]) -> bool {
        filters.iter().any(|f| self.prunes_expr(f))
    }

    fn prunes_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(binary) => match binary.op {
                Operator::And => self.prunes_expr(&binary.left) || self.prunes_expr(&binary.right),
                Operator::Or => self.prunes_expr(&binary.left) && self.prunes_expr(&binary.right),
                op => self.prunes_comparison(&binary.left, op, &binary.right),
            },
            Expr::IsNull(inner) => match inner.as_ref() {
                Expr::Column(c) => self
                    .columns
                    .get(&c.name)
                    .is_some_and(|col| col.null_count == Some(0)),
                _ => false,
            },
            Expr::InList(list) if !list.negated => {
                let Expr::Column(c) = list.expr.as_ref() else {
                    return false;
                };
                let Some(col) = self.columns.get(&c.name) else {
                    return false;
                };
                let mut values = Vec::with_capacity(list.list.len());
                for item in &list.list {
                    let Expr::Literal(v, _) = item else {
                        return false;
                    };
                    let Some(v) = col.family.literal(v) else {
                        return false;
                    };
                    values.push(v);
                }
                !values.is_empty() && col.bounds.can_prune(&ScalarPredicate::In(values))
            }
            _ => false,
        }
    }

    /// `column op literal` (either side), bare column and bare literal only.
    fn prunes_comparison(&self, left: &Expr, op: Operator, right: &Expr) -> bool {
        let (column, literal, op) = match (left, right) {
            (Expr::Column(c), Expr::Literal(v, _)) => (c, v, op),
            (Expr::Literal(v, _), Expr::Column(c)) => match op.swap() {
                Some(swapped) => (c, v, swapped),
                None => return false,
            },
            _ => return false,
        };
        let Some(col) = self.columns.get(&column.name) else {
            return false;
        };
        let Some(value) = col.family.literal(literal) else {
            return false;
        };
        let predicate = match op {
            Operator::Eq => ScalarPredicate::Equal(value),
            Operator::Lt => ScalarPredicate::LessThan(value),
            Operator::LtEq => ScalarPredicate::LessThanOrEqual(value),
            Operator::Gt => ScalarPredicate::GreaterThan(value),
            Operator::GtEq => ScalarPredicate::GreaterThanOrEqual(value),
            _ => return false,
        };
        col.bounds.can_prune(&predicate)
    }
}

/// The parsed, decoded snapshot of one immutable metadata version.
#[derive(Debug, Clone, Default)]
pub(crate) struct CachedSnapshot {
    pub(crate) version: u64,
    /// Bridge-relative parquet data files, sorted.
    pub(crate) paths: Vec<object_store::path::Path>,
    /// Decoded deletion vectors, keyed by FULL data-file path.
    pub(crate) deletions: HashMap<String, Arc<PositionDeletes>>,
    /// Manifest metrics, keyed by FULL data-file path.
    pub(crate) file_stats: HashMap<String, Arc<FileManifestStats>>,
}

impl CachedSnapshot {
    /// Approximate heap bytes (paths, 8 bytes per deleted position, bounds).
    fn weight(&self) -> u32 {
        let paths: usize = self.paths.iter().map(|p| p.as_ref().len() + 32).sum();
        let deletions: usize = self
            .deletions
            .iter()
            .map(|(k, d)| k.len() + 8 * d.cardinality() as usize + 64)
            .sum();
        let stats: usize = self
            .file_stats
            .iter()
            .map(|(k, s)| k.len() + s.weight())
            .sum();
        u32::try_from(256 + paths + deletions + stats).unwrap_or(u32::MAX)
    }
}

/// The process-wide manifest cache (lazily built).
pub(crate) struct ManifestPlanCache {
    inner: TenantCache<Arc<CachedSnapshot>>,
}

impl ManifestPlanCache {
    pub(crate) fn global() -> &'static ManifestPlanCache {
        static CACHE: OnceLock<ManifestPlanCache> = OnceLock::new();
        CACHE.get_or_init(|| ManifestPlanCache {
            inner: TenantCache::new(CacheBudget::new(
                MANIFEST_CACHE_BYTES,
                MANIFEST_CACHE_TENANT_CEILING_BYTES,
            )),
        })
    }

    /// The key of one immutable metadata object: the store base plus its
    /// LISTed revision. Scoped per tenant for fairness; isolation is also
    /// structural (the DrPath base is in the key).
    pub(crate) fn key(tenant: Option<&str>, base: &str, revision: &str) -> CacheKey {
        let key = format!("iceberg-manifest|{base}|{revision}");
        match tenant {
            Some(t) => CacheKey::new(t, CacheKind::Other, key),
            None => CacheKey::shared(CacheKind::Other, key),
        }
    }

    pub(crate) async fn get(&self, key: &CacheKey) -> Option<Arc<CachedSnapshot>> {
        self.inner.get(key).await
    }

    pub(crate) async fn insert(&self, key: CacheKey, snapshot: Arc<CachedSnapshot>) {
        let weight = snapshot.weight();
        self.inner.insert(key, weight, snapshot).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};

    fn fields() -> Vec<IcebergField> {
        let f = |id, name: &str, ty: &str| IcebergField {
            id,
            name: name.into(),
            type_name: ty.into(),
            required: false,
        };
        vec![
            f(1, "k", "long"),
            f(2, "n", "int"),
            f(3, "d", "date"),
            f(4, "s", "string"),
            f(5, "x", "double"),
        ]
    }

    fn stat(
        id: i32,
        lo: Option<Vec<u8>>,
        hi: Option<Vec<u8>>,
        nulls: Option<i64>,
    ) -> ColumnFileStats {
        ColumnFileStats {
            field_id: id,
            null_value_count: nulls,
            lower_bound: lo,
            upper_bound: hi,
            ..Default::default()
        }
    }

    /// A file holding k in [10, 20], n in [1, 5], d in [100, 200] days,
    /// s in ["apple", "melon"], x in [0.5, 1.5]; no nulls in k, unknown
    /// null count elsewhere.
    fn file() -> FileManifestStats {
        FileManifestStats::from_manifest(
            &fields(),
            &[
                stat(
                    1,
                    Some(10i64.to_le_bytes().to_vec()),
                    Some(20i64.to_le_bytes().to_vec()),
                    Some(0),
                ),
                stat(
                    2,
                    Some(1i32.to_le_bytes().to_vec()),
                    Some(5i32.to_le_bytes().to_vec()),
                    None,
                ),
                stat(
                    3,
                    Some(100i32.to_le_bytes().to_vec()),
                    Some(200i32.to_le_bytes().to_vec()),
                    Some(3),
                ),
                stat(4, Some(b"apple".to_vec()), Some(b"melon".to_vec()), None),
                stat(
                    5,
                    Some(0.5f64.to_le_bytes().to_vec()),
                    Some(1.5f64.to_le_bytes().to_vec()),
                    Some(0),
                ),
            ],
        )
    }

    fn date(days: i32) -> Expr {
        lit(DfScalarValue::Date32(Some(days)))
    }

    #[test]
    fn boundary_values_never_prune_a_matching_file() {
        let f = file();
        // Every predicate here is satisfied by a row AT a bound.
        for keep in [
            col("k").eq(lit(10i64)),
            col("k").eq(lit(20i64)),
            col("k").lt_eq(lit(10i64)),
            col("k").gt_eq(lit(20i64)),
            col("k").lt(lit(11i64)),
            col("k").gt(lit(19i64)),
            lit(20i64).gt_eq(col("k")),
            col("n").eq(lit(5i32)),
            col("d").eq(date(100)),
            col("d").gt_eq(date(200)),
            col("s").eq(lit("apple")),
            col("s").eq(lit("melon")),
            col("s").gt_eq(lit("melon")),
        ] {
            assert!(
                !f.prunes(std::slice::from_ref(&keep)),
                "{keep} must keep the file"
            );
        }
        // Just outside the bounds: provably empty.
        for prune in [
            col("k").eq(lit(9i64)),
            col("k").eq(lit(21i64)),
            col("k").lt(lit(10i64)),
            col("k").gt(lit(20i64)),
            lit(9i64).gt_eq(col("k")),
            col("d").lt(date(100)),
            col("s").gt(lit("melon")),
            col("s").lt(lit("apple")),
            col("k").in_list(vec![lit(1i64), lit(30i64)], false),
            col("k").is_null(),
        ] {
            assert!(f.prunes(std::slice::from_ref(&prune)), "{prune} must prune");
        }
    }

    #[test]
    fn and_or_compose_conservatively() {
        let f = file();
        assert!(f.prunes(&[col("k").eq(lit(15i64)).and(col("k").gt(lit(30i64)))]));
        assert!(!f.prunes(&[col("k").gt(lit(30i64)).or(col("k").eq(lit(15i64)))]));
        assert!(f.prunes(&[col("k").gt(lit(30i64)).or(col("k").lt(lit(0i64)))]));
        // Separate pushed filters are ANDed.
        assert!(f.prunes(&[col("k").eq(lit(15i64)), col("k").gt(lit(30i64))]));
        assert!(!f.prunes(&[col("k").eq(lit(15i64)), col("n").eq(lit(3i32))]));
    }

    #[test]
    fn nulls_and_missing_bounds_keep_the_file() {
        let f = file();
        // Unknown null count ⇒ IS NULL keeps; a known nonzero one keeps too.
        assert!(!f.prunes(&[col("n").is_null()]));
        assert!(!f.prunes(&[col("d").is_null()]));
        // A NULL literal never prunes; NOT IN / != never prune.
        assert!(!f.prunes(&[col("k").eq(lit(DfScalarValue::Int64(None)))]));
        assert!(!f.prunes(&[col("k").not_eq(lit(99i64))]));
        assert!(!f.prunes(&[col("k").in_list(vec![lit(99i64)], true)]));
        // A column with no metrics at all keeps.
        assert!(!f.prunes(&[col("missing").eq(lit(1i64))]));
        // An all-null file has no bounds: comparisons keep it.
        let all_null = FileManifestStats::from_manifest(&fields(), &[stat(1, None, None, Some(7))]);
        assert!(!all_null.prunes(&[col("k").eq(lit(1i64))]));
        assert!(!all_null.prunes(&[col("k").is_null()]));
    }

    #[test]
    fn type_mismatch_and_unsupported_families_keep_the_file() {
        let f = file();
        // Float literal against an integer column, string against date, int
        // against string: never compared.
        assert!(!f.prunes(&[col("k").eq(lit(99.0f64))]));
        assert!(!f.prunes(&[col("d").eq(lit("2020-01-01"))]));
        assert!(!f.prunes(&[col("s").eq(lit(5i64))]));
        // A timestamp literal against a date column: unit-ambiguous, kept.
        assert!(!f.prunes(&[col("d").lt(lit(DfScalarValue::TimestampMicrosecond(Some(0), None)))]));
        // Doubles are never pruned (NaN ordering).
        assert!(!f.prunes(&[col("x").gt(lit(99.0f64))]));
        // A cast column is not the column's raw bounds.
        assert!(
            !f.prunes(&[Expr::Cast(datafusion::logical_expr::Cast::new(
                Box::new(col("k")),
                datafusion::arrow::datatypes::DataType::Utf8
            ))
            .eq(lit("zzz"))])
        );
        // A bound whose width does not match the type is dropped.
        let bad = FileManifestStats::from_manifest(
            &fields(),
            &[stat(1, Some(vec![1, 2, 3]), Some(vec![9; 3]), None)],
        );
        assert!(!bad.prunes(&[col("k").eq(lit(1_000_000i64))]));
    }

    #[test]
    fn truncated_string_bounds_never_drop_a_matching_file() {
        // The writer's truncate(16): a 20-char min keeps a 16-char prefix
        // lower bound; the upper bound is dropped. A foreign writer may keep a
        // (truncated) 16-char upper bound — it must not be trusted either.
        let lower = b"abcdefghijklmnop".to_vec(); // prefix of "abcdefghijklmnopqrst"
        let foreign_upper = b"abcdefghijklmnoq".to_vec();
        let f = FileManifestStats::from_manifest(
            &fields(),
            &[stat(4, Some(lower), Some(foreign_upper), None)],
        );
        // The true values are "abcdefghijklmnopqrst" .. "abcdefghijklmnoqzzzz".
        assert!(!f.prunes(&[col("s").eq(lit("abcdefghijklmnopqrst"))]));
        assert!(
            !f.prunes(&[col("s").gt(lit("abcdefghijklmnoqa"))]),
            "upper bound untrusted"
        );
        assert!(!f.prunes(&[col("s").gt_eq(lit("abcdefghijklmnoqzzzz"))]));
        // The truncated lower bound still proves `< prefix` empty.
        assert!(f.prunes(&[col("s").lt(lit("abcdefghijklmnop"))]));
        // Writer-shaped stats (no upper bound) keep every value >= lower.
        let writer = FileManifestStats::from_manifest(
            &fields(),
            &[stat(4, Some(b"abcdefghijklmnop".to_vec()), None, None)],
        );
        assert!(!writer.prunes(&[col("s").eq(lit("zzzz"))]));
        assert!(!writer.prunes(&[col("s").gt(lit("zzzz"))]));
    }
}
