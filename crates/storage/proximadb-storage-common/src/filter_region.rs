//! TD-FPRUNE-1 PR-E — PAX **Region E (filter columns)**.
//!
//! Layout `[prefix][A0?][A][B][C?][OPR?][D][E?][footer]`: one contiguous
//! chunk per filterable column, rows in the segment's on-disk ordinal order,
//! written in the same pass (and the same atomic object) as Region D. A
//! filtered search answers its predicate from the referenced chunks — one
//! ranged GET each — instead of decoding every Region-D row.
//!
//! **Exactness by construction.** A chunk stores, per row, the literal the
//! filter evaluator resolves the field to
//! ([`proximadb_search_types::sql_value_filter::resolve_filter_literal`],
//! `None` = absent) — not the shred stripe, whose null conflates absent,
//! null and wrong-typed values. The reader evaluates the same tree evaluator
//! over these literals, so the allow-set equals a full-row Stage F for every
//! operator and value type.
//!
//! Chunk encoding (`v1`, little-endian):
//! `[ver u8][rows u64][dict_n u32]`, `dict_n ×` `[tag u8 0=absent 1=value][len u32][msgpack]`,
//! `[counts u32 × dict_n][bits u8][codes: rows × bits, LSB-first]`.
//! Entries are MessagePack-encoded `serde_json::Value`s — bit-exact for f64
//! (JSON text is not: without serde_json's `float_roundtrip` feature a
//! re-parsed float can land one ULP away, which flips `<`/`>`), and the writer
//! decodes every new entry back and omits the column if it does not compare
//! equal. A column whose dictionary would exceed [`FILTER_REGION_MAX_DICT`]
//! entries is omitted too; filters on an omitted column use the full-row path.

use std::collections::HashMap;

use anyhow::{Result, bail};
use proximadb_records::ProximaTree;
use proximadb_search_types::sql_value_filter::resolve_filter_literal;

/// Largest dictionary a Region-E column may carry (12-bit codes). A
/// higher-cardinality column (e.g. a unique id) is omitted from Region E.
pub const FILTER_REGION_MAX_DICT: usize = 4096;

const CHUNK_V1: u8 = 1;
const TAG_ABSENT: u8 = 0;
const TAG_VALUE: u8 = 1;

/// Region E as written: the bytes (placed at `base_off` in the segment) and,
/// per kept column, `(name, absolute offset, length)`.
#[derive(Debug, Default)]
pub struct BuiltFilterRegion {
    pub bytes: Vec<u8>,
    pub columns: Vec<(String, u64, u64)>,
}

/// Accumulates Region E row by row, in on-disk ordinal order.
#[derive(Debug)]
pub struct FilterRegionBuilder {
    columns: Vec<ColumnBuilder>,
    rows: u64,
}

#[derive(Debug)]
struct ColumnBuilder {
    name: String,
    /// Serialized literal (`None` = absent) → code. Keyed by bytes, so two
    /// equal values can at worst get two codes — never one code for two
    /// different values.
    index: HashMap<Option<Vec<u8>>, u16>,
    entries: Vec<Option<Vec<u8>>>,
    counts: Vec<u32>,
    /// Dictionary codes (< `FILTER_REGION_MAX_DICT`, so `u16` suffices).
    codes: Vec<u16>,
    /// Over the dictionary cap, or a literal that does not round-trip
    /// exactly: omitted at finish.
    dropped: bool,
}

impl FilterRegionBuilder {
    pub fn new(names: &[String]) -> Self {
        Self {
            columns: names
                .iter()
                .map(|name| ColumnBuilder {
                    name: name.clone(),
                    index: HashMap::new(),
                    entries: Vec::new(),
                    counts: Vec::new(),
                    codes: Vec::new(),
                    dropped: false,
                })
                .collect(),
            rows: 0,
        }
    }

    /// Record one row's props: each column's resolved filter literal.
    pub fn push_row(&mut self, props: &ProximaTree) {
        self.rows += 1;
        for col in &mut self.columns {
            if col.dropped {
                continue;
            }
            let (key, value) = match resolve_filter_literal(props, &col.name) {
                None => (None, None),
                Some(value) => match rmp_serde::to_vec(&value) {
                    Ok(bytes) => (Some(bytes), Some(value)),
                    Err(_) => {
                        col.drop_all();
                        continue;
                    }
                },
            };
            let code = match col.index.get(&key) {
                Some(&code) => code,
                None => {
                    // A new entry must decode back to exactly this literal
                    // (and fit the cap), or the column cannot answer exactly.
                    let exact = match (&key, &value) {
                        (Some(bytes), Some(value)) => {
                            rmp_serde::from_slice::<serde_json::Value>(bytes)
                                .is_ok_and(|back| &back == value)
                        }
                        _ => true,
                    };
                    if !exact || col.entries.len() >= FILTER_REGION_MAX_DICT {
                        col.drop_all();
                        continue;
                    }
                    let code = col.entries.len() as u16;
                    col.index.insert(key.clone(), code);
                    col.entries.push(key);
                    col.counts.push(0);
                    code
                }
            };
            col.counts[code as usize] += 1;
            col.codes.push(code);
        }
    }

    /// Encode the kept columns, contiguous, as if placed at `base_off`.
    pub fn finish(self, base_off: u64) -> BuiltFilterRegion {
        let mut out = BuiltFilterRegion::default();
        if self.rows == 0 {
            return out;
        }
        for col in self.columns {
            if col.dropped {
                continue;
            }
            let start = out.bytes.len();
            encode_column(&col, self.rows, &mut out.bytes);
            out.columns.push((
                col.name,
                base_off + start as u64,
                (out.bytes.len() - start) as u64,
            ));
        }
        out
    }
}

impl ColumnBuilder {
    fn drop_all(&mut self) {
        self.dropped = true;
        self.index = HashMap::new();
        self.entries = Vec::new();
        self.counts = Vec::new();
        self.codes = Vec::new();
    }
}

fn code_bits(dict_len: usize) -> u8 {
    if dict_len <= 1 {
        0
    } else {
        (usize::BITS - (dict_len - 1).leading_zeros()) as u8
    }
}

fn packed_len(rows: u64, bits: u8) -> Option<u64> {
    rows.checked_mul(u64::from(bits))?
        .checked_add(7)
        .map(|b| b / 8)
}

fn encode_column(col: &ColumnBuilder, rows: u64, out: &mut Vec<u8>) {
    out.push(CHUNK_V1);
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&(col.entries.len() as u32).to_le_bytes());
    for entry in &col.entries {
        match entry {
            None => {
                out.push(TAG_ABSENT);
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(json) => {
                out.push(TAG_VALUE);
                out.extend_from_slice(&(json.len() as u32).to_le_bytes());
                out.extend_from_slice(json);
            }
        }
    }
    for count in &col.counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    let bits = code_bits(col.entries.len());
    out.push(bits);
    let start = out.len();
    let len = packed_len(rows, bits).unwrap_or(0) as usize;
    out.resize(start + len, 0);
    if bits > 0 {
        for (row, &code) in col.codes.iter().enumerate() {
            let pos = row * bits as usize;
            for bit in 0..bits as usize {
                if code >> bit & 1 == 1 {
                    let p = pos + bit;
                    out[start + p / 8] |= 1 << (p % 8);
                }
            }
        }
    }
}

/// The footer's view of Region E (section 8): the region extent and, per
/// column, `(name, absolute offset, length)`. Kept tiny — every cascade
/// footer read carries it; the chunks themselves live in Region E.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterRegionDirectory {
    pub e_off: u64,
    pub e_len: u64,
    pub columns: Vec<(String, u64, u64)>,
}

impl FilterRegionDirectory {
    /// The `(offset, length)` of column `name`, if Region E carries it.
    pub fn column(&self, name: &str) -> Option<(u64, u64)> {
        self.columns
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, off, len)| (*off, *len))
    }

    /// Section payload: `[e_off u64][e_len u64][n u16]` then per column
    /// `[name_len u16][name][off u64][len u64]`.
    pub fn to_section_payload(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.e_off.to_le_bytes());
        out.extend_from_slice(&self.e_len.to_le_bytes());
        out.extend_from_slice(&u16::try_from(self.columns.len())?.to_le_bytes());
        for (name, off, len) in &self.columns {
            out.extend_from_slice(&u16::try_from(name.len())?.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }
        Ok(out)
    }

    /// Parse a section payload. Every column must lie inside the region and
    /// the payload must be consumed exactly; anything else is an error.
    pub fn parse_section(bytes: &[u8]) -> Result<Self> {
        let mut c = Cursor { bytes, pos: 0 };
        let e_off = c.u64()?;
        let e_len = c.u64()?;
        let e_end = e_off
            .checked_add(e_len)
            .ok_or_else(|| anyhow::anyhow!("filter region extent overflows"))?;
        let mut n = [0u8; 2];
        n.copy_from_slice(c.take(2)?);
        let n = u16::from_le_bytes(n) as usize;
        let mut columns = Vec::with_capacity(n.min(bytes.len()));
        for _ in 0..n {
            let mut len = [0u8; 2];
            len.copy_from_slice(c.take(2)?);
            let name = std::str::from_utf8(c.take(u16::from_le_bytes(len) as usize)?)?.to_string();
            let off = c.u64()?;
            let len = c.u64()?;
            let inside = off >= e_off && off.checked_add(len).is_some_and(|end| end <= e_end);
            if !inside || len == 0 {
                bail!("filter region column {name} lies outside the region");
            }
            columns.push((name, off, len));
        }
        if c.pos != bytes.len() {
            bail!("filter region section has trailing bytes");
        }
        Ok(Self {
            e_off,
            e_len,
            columns,
        })
    }
}

/// A parsed, fully validated Region-E column.
#[derive(Debug, Clone)]
pub struct FilterColumn {
    rows: usize,
    entries: Vec<Option<serde_json::Value>>,
    counts: Vec<u32>,
    bits: u8,
    codes: Vec<u8>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n);
        match end.and_then(|end| self.bytes.get(self.pos..end)) {
            Some(slice) => {
                self.pos += n;
                Ok(slice)
            }
            None => bail!("filter column truncated"),
        }
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }
}

impl FilterColumn {
    /// Parse and validate one chunk. Any inconsistency (truncation, bad
    /// version or tag, a code outside the dictionary, counts that disagree
    /// with the codes, trailing bytes) is an error — the caller then uses
    /// the full-row path, never a partial answer.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut c = Cursor { bytes, pos: 0 };
        if c.u8()? != CHUNK_V1 {
            bail!("filter column: unknown chunk version");
        }
        let rows = usize::try_from(c.u64()?)?;
        let dict_n = c.u32()? as usize;
        if dict_n > FILTER_REGION_MAX_DICT || (dict_n == 0) != (rows == 0) {
            bail!("filter column: dictionary size {dict_n} invalid for {rows} rows");
        }
        let mut entries = Vec::with_capacity(dict_n);
        for _ in 0..dict_n {
            let tag = c.u8()?;
            let len = c.u32()? as usize;
            let body = c.take(len)?;
            entries.push(match tag {
                TAG_ABSENT if len == 0 => None,
                TAG_VALUE => Some(rmp_serde::from_slice(body)?),
                _ => bail!("filter column: bad dictionary entry tag {tag}"),
            });
        }
        let mut counts = Vec::with_capacity(dict_n);
        for _ in 0..dict_n {
            counts.push(c.u32()?);
        }
        let bits = c.u8()?;
        if bits != code_bits(dict_n) {
            bail!("filter column: {bits}-bit codes for {dict_n} entries");
        }
        let len = packed_len(rows as u64, bits)
            .and_then(|l| usize::try_from(l).ok())
            .ok_or_else(|| anyhow::anyhow!("filter column: code length overflow"))?;
        let codes = c.take(len)?.to_vec();
        if c.pos != bytes.len() {
            bail!("filter column: trailing bytes");
        }
        let col = Self {
            rows,
            entries,
            counts,
            bits,
            codes,
        };
        // Every code must name a dictionary entry and the counts must match
        // the codes. With 0-bit codes (one entry) there is nothing to walk —
        // and a corrupt `rows` must not make this loop for 2^40 rows.
        let counts_match = if bits == 0 {
            col.counts.iter().map(|&c| u64::from(c)).sum::<u64>() == rows as u64
        } else {
            let mut seen = vec![0u64; dict_n];
            for row in 0..rows {
                let code = col.raw_code(row);
                match seen.get_mut(code) {
                    Some(n) => *n += 1,
                    None => bail!("filter column: row {row} code {code} outside dictionary"),
                }
            }
            seen.iter()
                .zip(&col.counts)
                .all(|(&s, &c)| s == u64::from(c))
        };
        if !counts_match {
            bail!("filter column: counts disagree with codes");
        }
        Ok(col)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dict_len(&self) -> usize {
        self.entries.len()
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Rows carrying dictionary entry `code`.
    pub fn count(&self, code: usize) -> u32 {
        self.counts.get(code).copied().unwrap_or(0)
    }

    /// Dictionary entry `code` (`None` = the field is absent).
    pub fn entry(&self, code: usize) -> Option<&serde_json::Value> {
        self.entries.get(code).and_then(|e| e.as_ref())
    }

    /// Row `row`'s dictionary code.
    pub fn code(&self, row: usize) -> Result<usize> {
        if row >= self.rows {
            bail!("filter column: row {row} out of {}", self.rows);
        }
        Ok(self.raw_code(row))
    }

    /// Row `row`'s resolved literal (`None` = absent).
    pub fn literal(&self, row: usize) -> Result<Option<&serde_json::Value>> {
        Ok(self.entry(self.code(row)?))
    }

    /// Code of an in-range row (`row < rows`; validated at parse).
    fn raw_code(&self, row: usize) -> usize {
        let bits = self.bits as usize;
        if bits == 0 {
            return 0;
        }
        let pos = row * bits;
        let mut window = 0u32;
        for k in 0..3 {
            if let Some(&b) = self.codes.get(pos / 8 + k) {
                window |= u32::from(b) << (8 * k);
            }
        }
        ((window >> (pos % 8)) & ((1 << bits) - 1)) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{ProximaTree, ProximaTreeNode};
    use serde_json::json;

    fn row(key: &str, v: Option<ProximaValue>) -> ProximaTree {
        let mut t = ProximaTree::new();
        if let Some(v) = v {
            t.insert(key.to_string(), ProximaTreeNode::Value(v));
        }
        t
    }

    #[test]
    fn column_round_trips_every_resolved_literal() {
        use ProximaValue as V;
        let values = [
            None,
            Some(V::Null),
            Some(V::Int64(3)),
            Some(V::UInt64(u64::MAX)),
            Some(V::Float64(3.0)),
            Some(V::Float32(2.5)),
            Some(ProximaValue::Float64(f64::NAN)),
            Some(ProximaValue::Float64(f64::NEG_INFINITY)),
            Some(ProximaValue::Float64(0.1 + 0.2)),
            // JSON text does not round-trip these bit-exactly (one ULP off).
            Some(ProximaValue::Float64(1.0 / 11.0)),
            Some(ProximaValue::Float32(2.59)),
            Some(ProximaValue::Boolean(false)),
            Some(ProximaValue::String("3".into())),
            Some(ProximaValue::Json(json!({"x": [1, 2]}))),
            Some(ProximaValue::Int64(3)),
        ];
        let rows: Vec<ProximaTree> = values.iter().cloned().map(|v| row("f", v)).collect();
        let mut builder = FilterRegionBuilder::new(&["f".to_string()]);
        for r in &rows {
            builder.push_row(r);
        }
        let built = builder.finish(1000);
        assert_eq!(built.columns.len(), 1);
        let (name, off, len) = &built.columns[0];
        assert_eq!(name, "f");
        let start = (*off - 1000) as usize;
        let col = FilterColumn::parse(&built.bytes[start..start + *len as usize]).unwrap();
        assert_eq!(col.rows(), rows.len());
        for (i, r) in rows.iter().enumerate() {
            let expected = proximadb_search_types::sql_value_filter::resolve_filter_literal(r, "f");
            assert_eq!(col.literal(i).unwrap(), expected.as_ref(), "row {i}");
        }
        // Equal literals share one code; counts are exact per entry.
        assert_eq!(col.code(2).unwrap(), col.code(rows.len() - 1).unwrap());
        let total: u64 = (0..col.dict_len()).map(|c| u64::from(col.count(c))).sum();
        assert_eq!(total, rows.len() as u64);
    }

    #[test]
    fn code_width_follows_dictionary_size() {
        for (distinct, bits) in [(1usize, 0u8), (2, 1), (16, 4), (4096, 12)] {
            let mut builder = FilterRegionBuilder::new(&["k".to_string()]);
            for i in 0..distinct * 2 {
                builder.push_row(&row("k", Some(ProximaValue::Int64((i % distinct) as i64))));
            }
            let built = builder.finish(0);
            let (_, off, len) = &built.columns[0];
            let col =
                FilterColumn::parse(&built.bytes[*off as usize..(*off + *len) as usize]).unwrap();
            assert_eq!(col.bits(), bits, "{distinct} distinct");
            assert_eq!(col.dict_len(), distinct);
            for i in 0..distinct * 2 {
                assert_eq!(col.literal(i).unwrap(), Some(&json!((i % distinct) as i64)));
            }
        }
    }

    #[test]
    fn over_cap_column_is_omitted_and_others_kept() {
        let names = ["hi".to_string(), "lo".to_string()];
        let mut builder = FilterRegionBuilder::new(&names);
        for i in 0..(FILTER_REGION_MAX_DICT + 1) {
            let mut t = row("hi", Some(ProximaValue::Int64(i as i64)));
            t.insert(
                "lo".into(),
                ProximaTreeNode::Value(ProximaValue::Int64((i % 3) as i64)),
            );
            builder.push_row(&t);
        }
        let built = builder.finish(0);
        let kept: Vec<&str> = built.columns.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(kept, vec!["lo"]);
    }

    #[test]
    fn empty_builder_or_no_columns_writes_nothing() {
        let built = FilterRegionBuilder::new(&[]).finish(0);
        assert!(built.bytes.is_empty() && built.columns.is_empty());
        let mut b = FilterRegionBuilder::new(&["x".to_string()]);
        let built = {
            b.push_row(&ProximaTree::new());
            b.finish(0)
        };
        // One absent-only row still records the column (absence is a value).
        assert_eq!(built.columns.len(), 1);
    }

    #[test]
    fn corrupt_chunks_fail_closed() {
        let mut builder = FilterRegionBuilder::new(&["k".to_string()]);
        for i in 0..40 {
            builder.push_row(&row("k", Some(ProximaValue::Int64(i % 5))));
        }
        let built = builder.finish(0);
        let good = built.bytes.clone();
        assert!(FilterColumn::parse(&good).is_ok());
        for cut in 0..good.len() {
            assert!(
                FilterColumn::parse(&good[..cut]).is_err(),
                "truncated at {cut}"
            );
        }
        let mut bad_version = good.clone();
        bad_version[0] = 9;
        assert!(FilterColumn::parse(&bad_version).is_err());
        let mut huge_dict = good.clone();
        huge_dict[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(FilterColumn::parse(&huge_dict).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(FilterColumn::parse(&trailing).is_err());
        // A code outside the dictionary is rejected at parse time.
        let mut bad_code = good;
        let last = bad_code.len() - 1;
        bad_code[last] = 0xFF;
        assert!(FilterColumn::parse(&bad_code).is_err());
    }
}
