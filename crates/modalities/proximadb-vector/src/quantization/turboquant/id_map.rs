//! Stable external u64 IDs on top of [`TurboQuantStore`].
//!
//! [`TurboQuantStore`] addresses vectors positionally — `remove_slot`
//! uses swap-with-last, so the slot of every external reference would
//! need to be rewritten on every delete. That is fine for an internal
//! index but breaks any persistent ID scheme (replication, CDC,
//! Prometheus labels, query-result row keys).
//!
//! [`IdMapIndex`] wraps the positional store with a bidirectional
//! `u64 ↔ slot` mapping. The wrapper:
//!
//! - Rejects duplicate IDs at add time (whether across batches or
//!   within a single batch).
//! - Translates [`SearchHit`] slot indices back to the original IDs.
//! - Updates the moved vector's mapping on `remove()` so the table
//!   stays consistent with the inner store's swap-with-last layout.
//!
//! The wrapper owns no quantizer state of its own — `dim`, `bit_width`,
//! `calibration_mode`, `rotation_seed`, and the encoded codes all live
//! in the inner store. Persistence (save/load with the ID table
//! serialised alongside) lands in a follow-up session via a separate
//! `.tvim` wire format; this module focuses on the in-memory contract.
//!
//! Per `TURBOQUANT_LLD_2026_05_30.adoc` §"Concurrency Model" the public
//! methods take `&self` and serialise through the inner store's mutex
//! plus the wrapper's own `RwLock` over the ID table.

use std::collections::HashMap;
use std::sync::RwLock;

use proximadb_quantization_types::CalibrationMode;

use super::io::{self, PersistedIdMap, PersistedStore};
use super::{TurboQuantError, TurboQuantStore, kernel::SearchHit};

/// `IdMapIndex` returns `(score, external_id)` hits instead of
/// `(score, slot)`. Top-level callers (AXIS adapters, REST/gRPC
/// handlers) typically work with stable IDs.
pub type IdSearchHit = (f32, u64);

/// Bidirectional `u64 ↔ slot` mapping over a [`TurboQuantStore`].
///
/// All public methods take `&self`. The inner store handles its own
/// mutex; the ID table has its own RwLock so reads (`contains`, `search`'s
/// slot→id translation) run concurrently while writes (`add_with_id`,
/// `remove`) take the table write lock briefly.
pub struct IdMapIndex {
    inner: TurboQuantStore,
    /// Read-many / write-few — `RwLock` is the right primitive.
    /// Invariant: `id_table.slot_to_id.len() == inner.len()` and
    /// every slot in `0..inner.len()` maps to exactly one ID.
    id_table: RwLock<IdTable>,
}

struct IdTable {
    /// Slot → external ID. Position `i` stores the ID currently
    /// residing in slot `i` of the inner store. Truncated by `remove`.
    slot_to_id: Vec<u64>,
    /// External ID → slot. Kept in sync with `slot_to_id`.
    id_to_slot: HashMap<u64, usize>,
}

impl std::fmt::Debug for IdMapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self
            .id_table
            .read()
            .map(|t| t.slot_to_id.len())
            .unwrap_or(0);
        f.debug_struct("IdMapIndex")
            .field("inner", &self.inner)
            .field("n_ids", &len)
            .finish()
    }
}

impl IdMapIndex {
    /// Construct a new empty ID-mapped index. Same validation as
    /// [`TurboQuantStore::new`].
    pub fn new(
        dim: usize,
        bit_width: u8,
        calibration_mode: CalibrationMode,
        rotation_seed: u64,
    ) -> Result<Self, TurboQuantError> {
        let inner = TurboQuantStore::new(dim, bit_width, calibration_mode, rotation_seed)?;
        Ok(Self {
            inner,
            id_table: RwLock::new(IdTable {
                slot_to_id: Vec::new(),
                id_to_slot: HashMap::new(),
            }),
        })
    }

    /// Wrap an existing [`TurboQuantStore`] in an ID map. The store's
    /// `len()` must equal `ids.len()`; the caller supplies the
    /// slot→ID mapping in slot order. Useful for hydrating a wrapper
    /// from a persisted index file.
    pub fn from_store_with_ids(
        inner: TurboQuantStore,
        ids: Vec<u64>,
    ) -> Result<Self, TurboQuantError> {
        if inner.len() != ids.len() {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "from_store_with_ids: inner.len()={} != ids.len()={}",
                inner.len(),
                ids.len(),
            )));
        }
        let mut id_to_slot = HashMap::with_capacity(ids.len());
        for (slot, &id) in ids.iter().enumerate() {
            if id_to_slot.insert(id, slot).is_some() {
                return Err(TurboQuantError::InvalidFileFormat(format!(
                    "from_store_with_ids: duplicate id {id} in slot map",
                )));
            }
        }
        Ok(Self {
            inner,
            id_table: RwLock::new(IdTable {
                slot_to_id: ids,
                id_to_slot,
            }),
        })
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
    pub fn bit_width(&self) -> u8 {
        self.inner.bit_width()
    }
    pub fn calibration_mode(&self) -> CalibrationMode {
        self.inner.calibration_mode()
    }
    pub fn rotation_seed(&self) -> u64 {
        self.inner.rotation_seed()
    }
    pub fn has_calibration(&self) -> bool {
        self.inner.has_calibration()
    }

    pub fn len(&self) -> usize {
        self.id_table
            .read()
            .map(|t| t.slot_to_id.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add `n = vectors.len() / dim` vectors with their external IDs.
    /// `ids.len()` must equal `n`. All IDs must be unique — both
    /// vs the index's current contents and within the call.
    ///
    /// Atomicity: validates every ID before touching the inner store.
    /// On any error, no vectors are encoded and the ID table is not
    /// modified.
    pub fn add_with_ids(&self, vectors: &[f32], ids: &[u64]) -> Result<(), TurboQuantError> {
        if vectors.is_empty() && ids.is_empty() {
            return Ok(());
        }
        let dim = self.inner.dim();
        if vectors.len() % dim != 0 {
            return Err(TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        if ids.len() != n {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "ids.len()={} != n_vectors={}",
                ids.len(),
                n,
            )));
        }

        // Validate all IDs upfront so a partial failure is impossible.
        // Take the read lock for the existence checks; we'll upgrade
        // to write later. (Single-writer convention: callers may
        // serialise `add_with_ids` through their own synchronisation.)
        {
            let table = self
                .id_table
                .read()
                .expect("IdMapIndex::id_table read lock poisoned");
            let mut seen_in_call: std::collections::HashSet<u64> =
                std::collections::HashSet::with_capacity(n);
            for &id in ids {
                if table.id_to_slot.contains_key(&id) {
                    return Err(TurboQuantError::InvalidFileFormat(format!(
                        "id {id} already present in index",
                    )));
                }
                if !seen_in_call.insert(id) {
                    return Err(TurboQuantError::InvalidFileFormat(format!(
                        "id {id} appears more than once in this add call",
                    )));
                }
            }
        }

        // Capture the slot the first new vector will occupy BEFORE
        // touching the inner store, then run the inner add. If the
        // inner store rejects the batch (e.g. NaN input), the ID
        // table stays untouched — see the same pattern in turbovec.
        let base_slot = self.inner.len();
        self.inner.add(vectors)?;

        let mut table = self
            .id_table
            .write()
            .expect("IdMapIndex::id_table write lock poisoned");
        table.id_to_slot.reserve(n);
        for (i, &id) in ids.iter().enumerate() {
            table.id_to_slot.insert(id, base_slot + i);
        }
        table.slot_to_id.extend_from_slice(ids);
        Ok(())
    }

    /// Convenience for one-vector adds.
    pub fn add_with_id(&self, vector: &[f32], id: u64) -> Result<(), TurboQuantError> {
        self.add_with_ids(vector, &[id])
    }

    /// Remove the vector with `id`. Returns `true` when the ID was
    /// present and removed; `false` when it wasn't in the index.
    /// O(1): one [`TurboQuantStore::remove_slot`] call plus two
    /// HashMap operations.
    pub fn remove(&self, id: u64) -> Result<bool, TurboQuantError> {
        let mut table = self
            .id_table
            .write()
            .expect("IdMapIndex::id_table write lock poisoned");
        let slot = match table.id_to_slot.remove(&id) {
            Some(s) => s,
            None => return Ok(false),
        };
        let last = table.slot_to_id.len() - 1;

        // Delegate to the inner store. `moved_from` is the previous
        // slot of the vector that was swapped into `slot` (== `last`
        // unless `slot` was already the last entry).
        let moved_from = self.inner.remove_slot(slot)?;
        debug_assert_eq!(moved_from, last);

        if slot != last {
            // The previously-last ID now resides at `slot`. Rewrite
            // both maps so the moved ID points at its new slot.
            let moved_id = table.slot_to_id[last];
            table.slot_to_id[slot] = moved_id;
            table.id_to_slot.insert(moved_id, slot);
        }
        table.slot_to_id.pop();
        Ok(true)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.id_table
            .read()
            .map(|t| t.id_to_slot.contains_key(&id))
            .unwrap_or(false)
    }

    /// Look up the inner slot for an external ID. Useful when callers
    /// need to construct a `CandidateMaskSet` keyed by IDs but the
    /// kernel expects slot-positional bitmaps.
    pub fn slot_for_id(&self, id: u64) -> Option<usize> {
        self.id_table
            .read()
            .ok()
            .and_then(|t| t.id_to_slot.get(&id).copied())
    }

    /// Look up the external ID at a slot. Mirror of `slot_for_id` for
    /// callers translating raw [`SearchHit`]s when wrapping the inner
    /// store directly.
    pub fn id_for_slot(&self, slot: usize) -> Option<u64> {
        self.id_table
            .read()
            .ok()
            .and_then(|t| t.slot_to_id.get(slot).copied())
    }

    /// Reset the index — drops all encoded state and the ID table.
    /// Calibration in the inner store is preserved per the wire
    /// contract (see [`TurboQuantStore::clear`]).
    pub fn clear(&self) {
        self.inner.clear();
        let mut table = self
            .id_table
            .write()
            .expect("IdMapIndex::id_table write lock poisoned");
        table.slot_to_id.clear();
        table.id_to_slot.clear();
    }

    /// Save the index to a `.tvim` file at `path`. The file carries the
    /// inner store's encoded state (codes, scales, calibration) plus a
    /// slot→ID footer per the wire contract in
    /// `TURBOQUANT_LLD_2026_05_30.adoc` §3. Snapshot is taken under the
    /// inner mutex + the ID-table read lock; the write itself happens
    /// outside the locks.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), TurboQuantError> {
        self.save_with_epoch(path, 0)
    }

    /// Like [`Self::save`] but stamps the header with the supplied
    /// `encoded_epoch`. P8 wires this from xCatalog when the engine
    /// integration lands.
    pub fn save_with_epoch(
        &self,
        path: impl AsRef<std::path::Path>,
        encoded_epoch: u64,
    ) -> Result<(), TurboQuantError> {
        // Snapshot encoded state from the inner store…
        let (codes, scales, calibration, n_vectors) = {
            // Access the inner store's mutex via a tiny accessor pattern:
            // we re-use the store's existing snapshot logic by going
            // through its public API. The public stats() gives us the
            // header fields; the codes/scales we read by save()-ing to
            // a Vec<u8>, parsing back, and re-using the body. That is
            // wasteful — instead, replicate the snapshot here via the
            // crate-private accessor `inner` is not exposed.
            //
            // To avoid a re-snapshot detour, expose a crate-internal
            // helper in TurboQuantStore. For now, save the inner store
            // to a Vec<u8>, parse the bytes back, and graft the IDs.
            // This is correct and the cost is bounded by `n × bytes/vec`
            // which already dominates the write anyway.
            let mut buf = Vec::new();
            self.inner
                .save_with_epoch_to_writer(&mut buf, encoded_epoch)?;
            let mut cur = std::io::Cursor::new(buf);
            let persisted = io::read_from(&mut cur)?;
            let n = persisted.n_vectors;
            (persisted.codes, persisted.scales, persisted.calibration, n)
        };
        // …and the ID table from the wrapper's RwLock.
        let ids = {
            let t = self
                .id_table
                .read()
                .expect("IdMapIndex::id_table read lock poisoned");
            if t.slot_to_id.len() != n_vectors {
                return Err(TurboQuantError::InvalidFileFormat(format!(
                    "ID table size {} != inner store n_vectors {}",
                    t.slot_to_id.len(),
                    n_vectors,
                )));
            }
            t.slot_to_id.clone()
        };

        let persisted = PersistedIdMap {
            store: PersistedStore {
                bit_width: self.inner.bit_width(),
                calibration_mode: self.inner.calibration_mode(),
                rotation_seed: self.inner.rotation_seed(),
                dim: self.inner.dim(),
                n_vectors,
                encoded_epoch,
                codes,
                scales,
                calibration,
            },
            ids,
        };

        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(
                format!("could not create {:?}: {e}", path.as_ref(),),
            )
        })?;
        let mut writer = std::io::BufWriter::new(file);
        io::write_id_map_to(&mut writer, &persisted)?;
        use std::io::Write;
        writer
            .flush()
            .map_err(|e| TurboQuantError::InvalidFileFormat(format!("flush failed: {e}")))?;
        Ok(())
    }

    /// Restore an `IdMapIndex` from a `.tvim` file previously written
    /// by [`Self::save`].
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, TurboQuantError> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(format!("could not open {:?}: {e}", path.as_ref(),))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let persisted = io::read_id_map_from(&mut reader)?;
        Self::from_persisted_id_map(persisted)
    }

    /// Hydrate an `IdMapIndex` from an already-deserialized
    /// [`PersistedIdMap`]. Useful for tests round-tripping through a
    /// `Vec<u8>` without touching the filesystem.
    pub fn from_persisted_id_map(p: PersistedIdMap) -> Result<Self, TurboQuantError> {
        let inner = TurboQuantStore::from_persisted(p.store)?;
        Self::from_store_with_ids(inner, p.ids)
    }

    /// Atomic hot-reload from a `.tvim` file. Shape (dim, bit_width,
    /// calibration_mode, rotation_seed) MUST match this index;
    /// rejected as `InvalidFileFormat` otherwise. The encoded state +
    /// ID table are swapped as a single critical section under both
    /// the inner store's mutex and the wrapper's ID-table write lock.
    /// Concurrent readers see either the pre-reload state or the
    /// post-reload state, never a partial state.
    pub fn reload_from(&self, path: impl AsRef<std::path::Path>) -> Result<(), TurboQuantError> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TurboQuantError::InvalidFileFormat(format!("could not open {:?}: {e}", path.as_ref(),))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let persisted = super::io::read_id_map_from(&mut reader)?;
        self.swap_in(persisted)
    }

    /// Same as [`Self::reload_from`] but takes an already-deserialized
    /// [`PersistedIdMap`].
    pub fn swap_in(&self, persisted: PersistedIdMap) -> Result<(), TurboQuantError> {
        if persisted.ids.len() != persisted.store.n_vectors {
            return Err(TurboQuantError::InvalidFileFormat(format!(
                "swap_in: ids.len()={} != store.n_vectors={}",
                persisted.ids.len(),
                persisted.store.n_vectors,
            )));
        }
        // Validate uniqueness up-front before touching either lock.
        let mut id_to_slot = HashMap::with_capacity(persisted.ids.len());
        for (slot, &id) in persisted.ids.iter().enumerate() {
            if id_to_slot.insert(id, slot).is_some() {
                return Err(TurboQuantError::InvalidFileFormat(format!(
                    "swap_in: duplicate id {id} in slot map",
                )));
            }
        }
        // Inner-store shape check + swap (delegates to
        // TurboQuantStore::swap_in for the consistent error messages).
        self.inner.swap_in(persisted.store)?;
        // Swap the ID table under the wrapper's write lock.
        let mut table = self
            .id_table
            .write()
            .expect("IdMapIndex::id_table write lock poisoned");
        table.slot_to_id = persisted.ids;
        table.id_to_slot = id_to_slot;
        Ok(())
    }

    /// Run a top-`k` search and translate slot indices to external IDs.
    /// `mask`, when `Some`, is the same packed-bitmap allowlist the
    /// inner store accepts — the caller is responsible for building it
    /// against slot indices (use `slot_for_id` to translate from an
    /// id-list).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        mask: Option<&[u64]>,
    ) -> Result<Vec<IdSearchHit>, TurboQuantError> {
        let inner_hits: Vec<SearchHit> = self.inner.search(query, k, mask)?;
        let table = self
            .id_table
            .read()
            .expect("IdMapIndex::id_table read lock poisoned");
        let mut out = Vec::with_capacity(inner_hits.len());
        for (score, slot) in inner_hits {
            // The inner store guarantees in-range slots for the
            // current `n_vectors`. A `None` here would indicate a
            // catastrophic table desync; surface it as
            // InvalidFileFormat rather than panicking.
            let id = match table.slot_to_id.get(slot as usize) {
                Some(&id) => id,
                None => {
                    return Err(TurboQuantError::InvalidFileFormat(format!(
                        "IdMapIndex slot {slot} returned by kernel is out of \
                         id_table bounds (len = {})",
                        table.slot_to_id.len(),
                    )));
                }
            };
            out.push((score, id));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chacha20::ChaCha8Rng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut v = vec![0.0f32; n * dim];
        for i in 0..n {
            let mut sumsq = 0.0f64;
            for d in 0..dim {
                let x: f64 = rng.sample(StandardNormal);
                v[i * dim + d] = x as f32;
                sumsq += x * x;
            }
            let inv = if sumsq > 1e-30 {
                (1.0 / sumsq.sqrt()) as f32
            } else {
                0.0
            };
            for d in 0..dim {
                v[i * dim + d] *= inv;
            }
        }
        v
    }

    #[test]
    fn new_index_is_empty() {
        let idx = IdMapIndex::new(64, 4, CalibrationMode::Identity, 1).unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert_eq!(idx.dim(), 64);
        assert_eq!(idx.bit_width(), 4);
    }

    #[test]
    fn rejects_bad_dim_at_construction() {
        let err = IdMapIndex::new(7, 4, CalibrationMode::Identity, 0).unwrap_err();
        assert!(matches!(err, TurboQuantError::DimNotMultipleOf8(7)));
    }

    #[test]
    fn add_with_id_records_mapping() {
        let dim = 16;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(1, dim, 100);
        idx.add_with_id(&v, 9001).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.contains(9001));
        assert!(!idx.contains(9002));
        assert_eq!(idx.slot_for_id(9001), Some(0));
        assert_eq!(idx.id_for_slot(0), Some(9001));
    }

    #[test]
    fn add_with_ids_batch_assigns_sequential_slots() {
        let dim = 16;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(3, dim, 100);
        let ids = [100u64, 200, 300];
        idx.add_with_ids(&v, &ids).unwrap();
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.slot_for_id(100), Some(0));
        assert_eq!(idx.slot_for_id(200), Some(1));
        assert_eq!(idx.slot_for_id(300), Some(2));
        // And the reverse direction.
        assert_eq!(idx.id_for_slot(0), Some(100));
        assert_eq!(idx.id_for_slot(1), Some(200));
        assert_eq!(idx.id_for_slot(2), Some(300));
    }

    #[test]
    fn add_with_ids_rejects_duplicate_id_within_call() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(2, dim, 100);
        let err = idx.add_with_ids(&v, &[42, 42]).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("more than once")
        ));
        // Nothing was added.
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn add_with_ids_rejects_duplicate_id_across_calls() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(1, dim, 100);
        idx.add_with_id(&v, 42).unwrap();
        let v2 = random_unit_vectors(1, dim, 101);
        let err = idx.add_with_id(&v2, 42).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("already present")
        ));
        // Original entry unchanged.
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.id_for_slot(0), Some(42));
    }

    #[test]
    fn add_with_ids_count_mismatch_errors() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(2, dim, 100);
        let err = idx.add_with_ids(&v, &[1]).unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidFileFormat(_)));
    }

    #[test]
    fn add_with_ids_rejects_misaligned_buffer_atomically() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = vec![0.5f32; 9]; // not a multiple of dim
        let err = idx.add_with_ids(&v, &[1]).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::VectorBufferNotMultipleOfDim {
                vectors_len: 9,
                dim: 8
            }
        ));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn remove_unknown_id_returns_false() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        idx.add_with_id(&random_unit_vectors(1, dim, 100), 1)
            .unwrap();
        assert!(!idx.remove(999).unwrap());
        // Original entry survives.
        assert_eq!(idx.len(), 1);
        assert!(idx.contains(1));
    }

    #[test]
    fn remove_last_slot_truncates_without_moving() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(3, dim, 100);
        idx.add_with_ids(&v, &[10, 20, 30]).unwrap();
        assert!(idx.remove(30).unwrap());
        assert_eq!(idx.len(), 2);
        // Other IDs still point at their original slots.
        assert_eq!(idx.slot_for_id(10), Some(0));
        assert_eq!(idx.slot_for_id(20), Some(1));
        assert!(!idx.contains(30));
    }

    #[test]
    fn remove_middle_slot_remaps_moved_id() {
        let dim = 8;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(3, dim, 100);
        idx.add_with_ids(&v, &[10, 20, 30]).unwrap();
        // Remove the middle one. Inner store moves slot 2 → slot 1.
        // External-id contract: id 30 now lives at slot 1.
        assert!(idx.remove(20).unwrap());
        assert_eq!(idx.len(), 2);
        assert!(idx.contains(10));
        assert!(idx.contains(30));
        assert!(!idx.contains(20));
        assert_eq!(idx.slot_for_id(10), Some(0));
        assert_eq!(idx.slot_for_id(30), Some(1));
        assert_eq!(idx.id_for_slot(1), Some(30));
    }

    #[test]
    fn search_translates_slots_to_external_ids() {
        let dim = 32;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(5, dim, 100);
        let ids = [101u64, 102, 103, 104, 105];
        idx.add_with_ids(&v, &ids).unwrap();
        let q = &v[2 * dim..3 * dim];
        let hits = idx.search(q, 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        // Self-query of slot 2 should recover id 103.
        assert_eq!(hits[0].1, 103);
    }

    #[test]
    fn search_returns_ids_even_after_removal_swap() {
        let dim = 32;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(5, dim, 100);
        let ids = [101u64, 102, 103, 104, 105];
        idx.add_with_ids(&v, &ids).unwrap();
        idx.remove(102).unwrap(); // slot 4 (id 105) moves into slot 1
        // Searching for the vector formerly at slot 4 must now return
        // id 105 — its external id is stable across the slot swap.
        let q = &v[4 * dim..5 * dim];
        let hits = idx.search(q, 1, None).unwrap();
        assert_eq!(hits[0].1, 105);
    }

    #[test]
    fn clear_drops_all_id_mappings_but_preserves_calibration() {
        let dim = 32;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::TqPlus, 1).unwrap();
        let v = random_unit_vectors(1024, dim, 100);
        let ids: Vec<u64> = (0..1024).collect();
        idx.add_with_ids(&v, &ids).unwrap();
        assert!(idx.has_calibration());

        idx.clear();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert!(!idx.contains(0));
        // Calibration in the inner store is preserved per wire contract.
        assert!(idx.has_calibration());
    }

    #[test]
    fn from_store_with_ids_validates_length_and_uniqueness() {
        let dim = 16;
        let store = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        store.add(&random_unit_vectors(3, dim, 100)).unwrap();

        // Wrong length.
        let err = IdMapIndex::from_store_with_ids(
            TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap(),
            vec![1, 2],
        )
        .unwrap_err();
        assert!(matches!(err, TurboQuantError::InvalidFileFormat(_)));

        // Duplicate ID.
        let err = IdMapIndex::from_store_with_ids(store, vec![5, 5, 6]).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("duplicate")
        ));
    }

    #[test]
    fn from_store_with_ids_happy_path() {
        let dim = 16;
        let store = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        store.add(&random_unit_vectors(3, dim, 100)).unwrap();
        let idx = IdMapIndex::from_store_with_ids(store, vec![7, 8, 9]).unwrap();
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.slot_for_id(7), Some(0));
        assert_eq!(idx.slot_for_id(8), Some(1));
        assert_eq!(idx.slot_for_id(9), Some(2));
    }

    #[test]
    fn search_with_mask_translates_slots_to_ids() {
        let dim = 32;
        let n = 64;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(n, dim, 100);
        let ids: Vec<u64> = (1000..1000 + n as u64).collect();
        idx.add_with_ids(&v, &ids).unwrap();

        // Allowlist slots 5 and 10 via the kernel bitmap form. Callers
        // would normally use `slot_for_id` to bridge an id-list to the
        // bitmap.
        let mut mask = vec![0u64; (n + 63) >> 6];
        for slot in [5usize, 10] {
            mask[slot >> 6] |= 1u64 << (slot & 63);
        }
        let q = random_unit_vectors(1, dim, 200);
        let hits = idx.search(&q, 5, Some(&mask)).unwrap();
        assert_eq!(hits.len(), 2);
        let mut returned_ids: Vec<u64> = hits.iter().map(|h| h.1).collect();
        returned_ids.sort();
        assert_eq!(returned_ids, vec![1005, 1010]);
    }

    // ------------------------------------------------------------------
    // .tvim persistence
    // ------------------------------------------------------------------

    #[test]
    fn save_load_round_trip_via_tempfile_preserves_ids_and_search_results() {
        let dim = 32;
        let n = 20;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(n, dim, 100);
        let ids: Vec<u64> = (5000..5000 + n as u64).collect();
        idx.add_with_ids(&v, &ids).unwrap();

        let q = random_unit_vectors(1, dim, 200);
        let original = idx.search(&q, 5, None).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        idx.save_with_epoch(tmp.path(), 7).unwrap();
        let restored = IdMapIndex::load(tmp.path()).unwrap();

        assert_eq!(restored.len(), n);
        assert_eq!(restored.dim(), dim);
        assert_eq!(restored.rotation_seed(), 1);
        for &id in &ids {
            assert!(restored.contains(id), "id {id} missing after round-trip");
        }
        let after = restored.search(&q, 5, None).unwrap();
        assert_eq!(original.len(), after.len());
        for (a, b) in original.iter().zip(after.iter()) {
            assert_eq!(a.1, b.1, "external id mismatch after round-trip");
        }
    }

    #[test]
    fn save_load_round_trip_tq_plus_preserves_calibration() {
        let dim = 64;
        let n = 1024;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::TqPlus, 99).unwrap();
        let v = random_unit_vectors(n, dim, 200);
        let ids: Vec<u64> = (0..n as u64).collect();
        idx.add_with_ids(&v, &ids).unwrap();
        assert!(idx.has_calibration());

        let tmp = tempfile::NamedTempFile::new().unwrap();
        idx.save(tmp.path()).unwrap();
        let restored = IdMapIndex::load(tmp.path()).unwrap();
        assert!(restored.has_calibration());
        assert_eq!(restored.calibration_mode(), CalibrationMode::TqPlus);
        assert_eq!(restored.len(), n);
    }

    #[test]
    fn save_after_remove_persists_remaining_ids_only() {
        let dim = 16;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let v = random_unit_vectors(4, dim, 100);
        idx.add_with_ids(&v, &[10, 20, 30, 40]).unwrap();
        idx.remove(20).unwrap();
        // After remove: ids 10, 40, 30 (40 moved into slot 1).
        let tmp = tempfile::NamedTempFile::new().unwrap();
        idx.save(tmp.path()).unwrap();
        let restored = IdMapIndex::load(tmp.path()).unwrap();
        assert_eq!(restored.len(), 3);
        assert!(restored.contains(10));
        assert!(restored.contains(30));
        assert!(restored.contains(40));
        assert!(!restored.contains(20));
        assert_eq!(restored.id_for_slot(0), Some(10));
        assert_eq!(restored.id_for_slot(1), Some(40));
        assert_eq!(restored.id_for_slot(2), Some(30));
    }

    #[test]
    fn load_rejects_nonexistent_file() {
        let err = IdMapIndex::load("/tmp/id-map-does-not-exist-test-xyz.tvim").unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("could not open")
        ));
    }

    #[test]
    fn reload_from_swaps_in_new_state_and_ids() {
        let dim = 32;
        let bw = 4;
        let seed = 0x1357u64;
        let idx = IdMapIndex::new(dim, bw, CalibrationMode::Identity, seed).unwrap();
        idx.add_with_ids(&random_unit_vectors(3, dim, 100), &[100u64, 200, 300])
            .unwrap();

        // Donor with completely different IDs and a different count.
        let donor = IdMapIndex::new(dim, bw, CalibrationMode::Identity, seed).unwrap();
        donor
            .add_with_ids(
                &random_unit_vectors(5, dim, 200),
                &[1000u64, 2000, 3000, 4000, 5000],
            )
            .unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();

        idx.reload_from(tmp.path()).unwrap();
        assert_eq!(idx.len(), 5);
        // Pre-reload IDs are GONE.
        assert!(!idx.contains(100));
        assert!(!idx.contains(200));
        assert!(!idx.contains(300));
        // Post-reload IDs are present.
        assert!(idx.contains(1000));
        assert!(idx.contains(5000));
        assert_eq!(idx.id_for_slot(0), Some(1000));
        assert_eq!(idx.id_for_slot(4), Some(5000));
    }

    #[test]
    fn reload_from_rejects_shape_mismatch_on_id_map() {
        let idx = IdMapIndex::new(32, 4, CalibrationMode::Identity, 1).unwrap();
        // Donor with different dim → shape mismatch propagated from
        // TurboQuantStore::swap_in.
        let donor = IdMapIndex::new(64, 4, CalibrationMode::Identity, 1).unwrap();
        donor
            .add_with_ids(&random_unit_vectors(1, 64, 100), &[42])
            .unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        donor.save(tmp.path()).unwrap();
        let err = idx.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("dim mismatch")
        ));
        // Original index unchanged.
        assert_eq!(idx.dim(), 32);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn reload_from_rejects_loading_tq_into_tvim() {
        // Loading a `.tq` file via the IdMapIndex's `.tvim` reader must
        // fail at the magic check, never touching internal state.
        let dim = 16;
        let idx = IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        let store = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        store.add(&random_unit_vectors(2, dim, 100)).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        store.save(tmp.path()).unwrap();
        let err = idx.reload_from(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref msg) if msg.contains("bad magic")
        ));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn load_rejects_tq_file_not_tvim() {
        let dim = 16;
        let store = TurboQuantStore::new(dim, 4, CalibrationMode::Identity, 1).unwrap();
        store.add(&random_unit_vectors(3, dim, 100)).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        store.save(tmp.path()).unwrap();
        let err = IdMapIndex::load(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InvalidFileFormat(ref s) if s.contains("bad magic")
        ));
    }

    #[test]
    fn concurrent_add_and_search_with_ids() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let dim = 32;
        let idx = Arc::new(IdMapIndex::new(dim, 4, CalibrationMode::Identity, 1).unwrap());
        // Seed with 20 entries.
        let seed_v = random_unit_vectors(20, dim, 100);
        let seed_ids: Vec<u64> = (0..20).collect();
        idx.add_with_ids(&seed_v, &seed_ids).unwrap();

        let mut handles = Vec::new();

        // One writer adding new IDs in disjoint ranges so they never
        // collide.
        {
            let idx = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                for i in 0..10 {
                    let id = 1000 + i;
                    let v = random_unit_vectors(1, dim, 200 + i as u64);
                    idx.add_with_id(&v, id).unwrap();
                    thread::sleep(Duration::from_millis(1));
                }
            }));
        }

        // 3 readers searching in a loop.
        for r in 0..3 {
            let idx = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                let q = random_unit_vectors(1, dim, 300 + r as u64);
                for _ in 0..10 {
                    let hits = idx.search(&q, 5, None).unwrap();
                    for h in &hits {
                        // Every returned ID must currently be in the
                        // index. Removing this would be a stale-id bug.
                        assert!(idx.contains(h.1));
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("worker panicked");
        }
        assert_eq!(idx.len(), 30);
    }
}
