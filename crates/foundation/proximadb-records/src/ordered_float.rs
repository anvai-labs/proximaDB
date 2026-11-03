//! Shared ordered-float wrappers.
//!
//! Two contracts coexist in the codebase and are preserved exactly:
//!
//! * [`TotalOrderedF32`]/[`TotalOrderedF64`] — *numeric* equality
//!   (`NaN != NaN`, `0.0 == -0.0`) with a NaN-greatest total order. For
//!   `BinaryHeap` / sort keys. (Consolidates the former `hnsw_index` and
//!   `swift` local `OrderedFloat` wrappers.)
//! * [`BitKeyF32`]/[`BitKeyF64`] — *bitwise* identity (`to_bits` equality and
//!   hash, so `0.0 != -0.0` and a NaN equals only the same NaN payload) with
//!   a NaN-greatest total order. For `HashMap`/`HashSet` keys,
//!   `BTreeMap` keys and identity-compared values. (Consolidates the former
//!   `result_set`, `path_index`, and `filterable_metadata` local wrappers.)
//!
//! Known, intentional quirk kept from the consolidated sites: `BitKey*`'s
//! `Eq` (bitwise) is finer-grained than its `Ord` (NaN-greatest, where all
//! NaNs compare `Equal` and `0.0 == -0.0`). No consolidated site depends on
//! the two agreeing for signed zeros; `result_set`'s `ResultKey` orders via
//! `Ord` only, which is unaffected.

use std::cmp::Ordering;

macro_rules! total_ordered_float {
    ($name:ident, $t:ty) => {
        /// Float wrapper with numeric equality and a NaN-greatest total
        /// order. `NaN` compares greater than every other value; two `NaN`s
        /// compare `Equal` via `Ord` while remaining `!=` via `PartialEq` —
        /// the standard `BinaryHeap` pattern.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name(pub $t);

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                match (self.0.is_nan(), other.0.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal),
                }
            }
        }
    };
}

macro_rules! bit_key_float {
    ($name:ident, $t:ty) => {
        /// Float wrapper with bitwise identity (equality/hash by `to_bits`):
        /// `0.0` and `-0.0` are distinct; a `NaN` equals only a NaN with the
        /// same payload. `Ord` is the NaN-greatest total order. Derives
        /// `Serialize`/`Deserialize` so metadata structs can embed it.
        #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
        pub struct $name(pub $t);

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.to_bits().hash(state);
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                match (self.0.is_nan(), other.0.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal),
                }
            }
        }
    };
}

total_ordered_float!(TotalOrderedF32, f32);
total_ordered_float!(TotalOrderedF64, f64);
bit_key_float!(BitKeyF32, f32);
bit_key_float!(BitKeyF64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn total_orders_numbers_and_places_nan_greatest() {
        assert!(TotalOrderedF64(1.0) < TotalOrderedF64(2.0));
        assert_eq!(
            TotalOrderedF64(1.0).cmp(&TotalOrderedF64(f64::NAN)),
            Ordering::Less
        );
        assert_eq!(
            TotalOrderedF64(f64::NAN).cmp(&TotalOrderedF64(f64::NEG_INFINITY)),
            Ordering::Greater
        );
        assert_eq!(
            TotalOrderedF32(f32::NAN).cmp(&TotalOrderedF32(f32::NAN)),
            Ordering::Equal
        );
        // numeric equality: signed zeros are equal, NaN != NaN
        assert_eq!(TotalOrderedF64(0.0), TotalOrderedF64(-0.0));
        assert_ne!(TotalOrderedF64(f64::NAN), TotalOrderedF64(f64::NAN));
    }

    #[test]
    fn total_works_as_binary_heap_key() {
        let mut heap = BinaryHeap::new();
        heap.push(TotalOrderedF32(3.0));
        heap.push(TotalOrderedF32(f32::NAN));
        heap.push(TotalOrderedF32(1.0));
        heap.push(TotalOrderedF32(f32::NAN));
        // NaNs pop first (greatest); numbers pop in descending order.
        assert!(heap.pop().unwrap().0.is_nan());
        assert!(heap.pop().unwrap().0.is_nan());
        assert_eq!(heap.pop().unwrap().0, 3.0);
        assert_eq!(heap.pop().unwrap().0, 1.0);
    }

    #[test]
    fn bit_key_identity_is_bitwise() {
        assert_eq!(BitKeyF64(1.0), BitKeyF64(1.0));
        assert_ne!(BitKeyF64(0.0), BitKeyF64(-0.0));
        assert_eq!(BitKeyF64(f64::NAN), BitKeyF64(f64::NAN));
        assert_ne!(BitKeyF64(f64::NAN), BitKeyF64(-f64::NAN));
    }

    #[test]
    fn bit_key_hash_is_consistent_with_eq() {
        // equal values must hash identically (mutation check for eq/hash drift)
        let a = BitKeyF64(1.5);
        let b = BitKeyF64(1.5);
        assert_eq!(hash_of(&a), hash_of(&b));
        let nan1 = BitKeyF32(f32::NAN);
        let nan2 = BitKeyF32(f32::NAN);
        assert_eq!(nan1, nan2);
        assert_eq!(hash_of(&nan1), hash_of(&nan2));
    }

    #[test]
    fn bit_key_orders_nan_greatest() {
        assert_eq!(BitKeyF64(1.0).cmp(&BitKeyF64(f64::NAN)), Ordering::Less);
        assert_eq!(
            BitKeyF64(f64::NAN).cmp(&BitKeyF64(f64::NAN)),
            Ordering::Equal
        );
        assert!(BitKeyF32(1.0) < BitKeyF32(2.0));
    }

    #[test]
    fn bit_key_serde_round_trips_signed_zero() {
        let v = BitKeyF64(-0.0);
        let json = serde_json::to_string(&v).expect("serialize");
        let back: BitKeyF64 = serde_json::from_str(&json).expect("deserialize");
        // bitwise identity must survive the round trip
        assert_eq!(v, back);
        assert_ne!(back, BitKeyF64(0.0));
    }
}
