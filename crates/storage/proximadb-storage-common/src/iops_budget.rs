//! Per-backend I/O round-trip budget (ADR-062 D6 / TD-RDSTRAT-6).
//!
//! On object storage the dominant vector-route cost term is **GET round-trips**
//! (`per_get=20 ≫ per_mib=5`, `route_cost_model`). The coalesced-RaBitQ layout
//! drives survivor-fetch block geometry off a per-backend **IOPS budget**: the
//! byte size a single ranged GET wants to move so that a survivor rerank touches
//! *few, large, coalesced* ranges rather than many tiny ones.
//!
//! `IopsBudget` is deliberately a **per-backend property** of the storage config —
//! it is NOT the [`crate::storage_profile::StorageProfile`] `{AppendBulk, Churn}`
//! *workload* enum (ADR-061 D1). The two are orthogonal: the workload enum selects
//! the read-projection *strategy*; this budget sizes the *physical* ranged GET.
//! Resolving a budget never changes a durability contract.

use std::collections::BTreeMap;

use parking_lot::RwLock;

/// Fallback rows-per-block when a row's survivor byte cost is unknown (0), so the
/// geometry derivation stays total (no divide-by-zero). Matches the legacy
/// `SST_DEFAULT_VECTORS_PER_BLOCK=128` until the per-row cost is measured.
pub const FALLBACK_ROWS_PER_BLOCK: u64 = 128;

/// The URL-scheme taxonomy every scheme-dependent decision in this module
/// resolves through (TD-IOBUDGET-1: one home for the scheme lists — three
/// independent match arms is the TD-IVF-4 defect class; a scheme alias added
/// to one arm but not the others would make `DiskClass::Cloud` resolve a
/// different profile than `for_scheme_and_env` for the same URL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    /// Azure Blob / ABFS / ADLS family.
    Azure,
    /// AWS S3 and HTTP(S) (no hard range limit).
    S3,
    /// Google Cloud Storage.
    Gcs,
    /// Local disk / MinIO / scheme-less bare path.
    Local,
    /// Anything else (generic default profile).
    Unknown,
}

impl Scheme {
    /// Object stores resolve through the filesystem seam's single scheme list
    /// (`ObjectStoreScheme`), which the ranged-read wave width shares.
    fn of(path: &str) -> Self {
        use proximadb_storage_filesystem_types::object_store_scheme::ObjectStoreScheme;
        match ObjectStoreScheme::of(path) {
            Some(ObjectStoreScheme::Azure) => Self::Azure,
            Some(ObjectStoreScheme::S3) => Self::S3,
            Some(ObjectStoreScheme::Gcs) => Self::Gcs,
            None => match path
                .split_once("://")
                .map(|(s, _)| s.to_ascii_lowercase())
                .as_deref()
            {
                Some("file" | "minio") | None => Self::Local,
                Some(_) => Self::Unknown,
            },
        }
    }
}

/// Per-backend ranged-GET byte budget for the coalesced-RaBitQ survivor fetch
/// (ADR-062 D6). `target` is the preferred block byte size; `min`/`max` bound the
/// self-derived rows-per-block.
///
/// * **Cloud object stores** (S3 / Azure Blob / ABFS / ADLS / GCS) lean to a
///   ~4 MiB target — large enough to amortise a 20-cost GET round-trip.
/// * **Local disk / MinIO** (no round-trip cost) lean to ~512 KiB–1 MiB.
/// * The generic [`IopsBudget::DEFAULT`] (~512 KiB / 2 MiB / 8 MiB) is used when
///   the backend cannot be inferred from the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IopsBudget {
    /// Smallest useful ranged GET (bytes). Blocks are never sized below this.
    pub min: u64,
    /// Preferred block byte size — the survivor-fetch ranged-GET target.
    pub target: u64,
    /// Largest single coalesced range (bytes). Blocks are never sized above this.
    pub max: u64,
}

impl IopsBudget {
    /// Cloud object store budget (S3 / Azure Blob / ABFS / ADLS / GCS):
    /// ~512 KiB min · 4 MiB target · 8 MiB max.
    pub const CLOUD: Self = Self {
        min: 512 * 1024,
        target: 4 * 1024 * 1024,
        max: 8 * 1024 * 1024,
    };

    /// Azure Blob Storage budget: 4 MiB target. This is the conservative
    /// planner policy being evaluated by TD-SEARCH-3, not a Blob billing
    /// quantum or a proven SDK range limit. Wire-level Azurite/Azure evidence
    /// must precede any larger default.
    pub const AZURE: Self = Self {
        min: 512 * 1024,
        target: 4 * 1024 * 1024,
        max: 4 * 1024 * 1024,
    };

    /// AWS S3 budget: 8 MiB target (no hard range limit; sweet spot for
    /// throughput vs per-GET cost). Compression → ~4 MiB on-disk.
    pub const S3: Self = Self {
        min: 512 * 1024,
        target: 8 * 1024 * 1024,
        max: 16 * 1024 * 1024,
    };

    /// Google Cloud Storage budget: same as S3 (no hard range limit).
    pub const GCS: Self = Self {
        min: 512 * 1024,
        target: 8 * 1024 * 1024,
        max: 16 * 1024 * 1024,
    };

    /// Local disk / MinIO budget (no round-trip cost):
    /// ~256 KiB min · 1 MiB target · 8 MiB max.
    pub const LOCAL: Self = Self {
        min: 256 * 1024,
        target: 1024 * 1024,
        max: 8 * 1024 * 1024,
    };

    /// Generic default when the backend is unknown:
    /// ~512 KiB min · 2 MiB target · 8 MiB max.
    pub const DEFAULT: Self = Self {
        min: 512 * 1024,
        target: 2 * 1024 * 1024,
        max: 8 * 1024 * 1024,
    };

    /// Resolve the budget from a storage path's URL scheme (ADR-036 / ADR-062 D6).
    ///
    /// Resolution order (TD-IOBUDGET-1): an operator-configured per-location
    /// budget (registered at boot from `[[storage.storage_locations]]
    /// io_budget`) wins by longest matching URL prefix; otherwise the scheme
    /// consts and the `PROXIMADB_DISK_CLASS` env hint apply unchanged.
    ///
    /// Per-provider defaults: Azure schemes → [`IopsBudget::AZURE`]
    /// (conservative 4 MiB evaluation policy); S3 / HTTP → [`IopsBudget::S3`]
    /// (8 MiB — no hard limit); GCS → [`IopsBudget::GCS`] (8 MiB); local /
    /// `file` / `minio` / bare paths → [`IopsBudget::LOCAL`]; unknown →
    /// [`IopsBudget::DEFAULT`].
    pub fn for_path(path: &str) -> Self {
        if let Some(registered) = location_budget_for(path) {
            return registered;
        }
        Self::for_scheme_and_env(path)
    }

    /// The registry-free scheme + env resolution (`for_path` minus any
    /// registered per-location budget). Exposed so the root crate can seed the
    /// registry itself: an operator's partial `io_budget` (e.g. only
    /// `target_bytes`) resolves its unspecified fields from the SAME scheme
    /// default a path would get unregistered, never from a stale entry.
    pub fn for_scheme_and_env(path: &str) -> Self {
        match Scheme::of(path) {
            // Azure: conservative 4 MiB policy pending TD-SEARCH-3 wire sweep.
            Scheme::Azure => Self::AZURE,
            // S3 / HTTP: no hard range limit; 8 MiB sweet spot.
            Scheme::S3 => Self::S3,
            // GCS: same as S3.
            Scheme::Gcs => Self::GCS,
            // Local / MinIO / bare path: no round-trip cost. ADR-073: an
            // HDD-backed location (500 IOPS, seek-bound) must coalesce like a
            // cloud store — the measured sweep puts the LOCAL profile at ~191
            // reads/query (≈2.6 QPS on HDD) vs ~41 with the CLOUD profile
            // (≈12 QPS). `PROXIMADB_DISK_CLASS=hdd` is the deploy-time hint
            // (per-location tags don't reach this leaf crate).
            Scheme::Local => {
                if std::env::var("PROXIMADB_DISK_CLASS")
                    .map(|v| v.trim().eq_ignore_ascii_case("hdd"))
                    .unwrap_or(false)
                {
                    Self::CLOUD
                } else {
                    Self::LOCAL
                }
            }
            Scheme::Unknown => Self::DEFAULT,
        }
    }

    /// The cloud-default profile for a path's scheme (TD-IOBUDGET-1).
    ///
    /// `DiskClass::Cloud` maps here — "coalesce like a remote store for MY
    /// scheme": a `file://` location gets [`IopsBudget::CLOUD`] (4 MiB, the
    /// ADR-073 remote-style coalescing), while an `s3://` location keeps its
    /// own [`IopsBudget::S3`] profile (8 MiB).
    pub fn scheme_cloud_budget(path: &str) -> Self {
        match Scheme::of(path) {
            Scheme::Azure => Self::AZURE,
            Scheme::S3 => Self::S3,
            Scheme::Gcs => Self::GCS,
            Scheme::Local | Scheme::Unknown => Self::CLOUD,
        }
    }

    /// The survivor-fetch block byte target — the ranged-GET size the writer aims
    /// for when cutting data blocks (replaces the static
    /// `SST_DEFAULT_VECTORS_PER_BLOCK` / `PROXIMADB_PAX_BLOCK_SIZE` default).
    pub fn target_block_bytes(self) -> u64 {
        self.target
    }

    /// Self-derived rows-per-block from a row's survivor-fetch byte cost
    /// (ADR-062 D6):
    ///
    /// ```text
    /// rows = clamp( target / per_row_bytes,
    ///               ceil(min / per_row_bytes),   // never below `min` bytes
    ///               floor(max / per_row_bytes) ) // never above `max` bytes
    /// ```
    ///
    /// `per_row_bytes = sq8(if present)·dim + fp32(if present)·4·dim + fixed_metadata`
    /// (RaBitQ excluded — it lives in the coalesced header region). A `0` cost
    /// (unknown composition) returns [`FALLBACK_ROWS_PER_BLOCK`]. Always ≥ 1.
    pub fn rows_per_block(self, per_row_bytes: u64) -> u64 {
        if per_row_bytes == 0 {
            return FALLBACK_ROWS_PER_BLOCK;
        }
        let want = self.target / per_row_bytes;
        let floor = self.min.div_ceil(per_row_bytes);
        let ceil = self.max / per_row_bytes;
        let lower = floor.max(1);
        let upper = ceil.max(lower);
        want.clamp(lower, upper)
    }
}

impl Default for IopsBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Boot-seeded registry of operator-configured per-storage-location budgets
/// (TD-IOBUDGET-1). The root crate resolves each
/// `[[storage.storage_locations]] io_budget` into an [`IopsBudget`] and
/// registers it here under the location URL before engines start; `for_path`
/// then prefers the longest matching registered prefix. Insert-overwrite (not
/// first-wins) so re-seeding a restart or an embedded re-boot is idempotent
/// (the TD-COMPACT-9 lesson: a first-wins seed silently ignores the TOML).
///
/// This is the leaf-side half of the seam ADR-073 asked for: `tags` on
/// `StorageLocation` cannot reach this crate (foundation may not depend on
/// storage), so the root converts the config and pushes resolved values down.
static LOCATION_BUDGETS: RwLock<BTreeMap<String, IopsBudget>> = RwLock::new(BTreeMap::new());

/// Register (or overwrite) the budget for a storage location URL prefix.
/// Called at boot from the root crate after config validation.
pub fn register_location_budget(url_prefix: &str, budget: IopsBudget) {
    LOCATION_BUDGETS
        .write()
        .insert(normalize_prefix(url_prefix), budget);
}

/// Drop every registered location budget. Seeding is authoritative-REPLACE:
/// the root crate drains before re-registering the config's full set, so a
/// location DROPPED from a later config cannot keep serving its stale budget
/// to an in-process re-boot (embedded re-init, test suites).
pub fn clear_location_budgets() {
    LOCATION_BUDGETS.write().clear();
}

/// The registered budget for a path, if any — the longest registered URL
/// prefix that matches wins. A prefix matches when the path equals it or
/// continues it at a `/` boundary (so `s3://bucket` never claims
/// `s3://bucket-other/…`).
pub fn location_budget_for(path: &str) -> Option<IopsBudget> {
    LOCATION_BUDGETS
        .read()
        .iter()
        .filter(|(prefix, _)| prefix_matches(path, prefix))
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|(_, budget)| *budget)
}

fn normalize_prefix(url_prefix: &str) -> String {
    url_prefix.trim_end_matches('/').to_string()
}

fn prefix_matches(path: &str, prefix: &str) -> bool {
    path.starts_with(prefix)
        && (path.len() == prefix.len() || path.as_bytes()[prefix.len()] == b'/')
}

/// The IOP target a **writer** should size blocks/cells against.
///
/// Read-side planning resolves the backend from the object URL, but the write
/// path stages segments to a LOCAL file and publishes afterwards, so at
/// clustering time it does not know its destination backend. Passing the staging
/// path here would resolve `LOCAL` and be actively wrong.
///
/// This exists so both sides share **one** definition of "the IOP target"
/// (TD-IVF-4: two independent constants is the defect, a single helper consulted
/// by both is the fix), and so the unknown-destination case is explicit in the
/// signature instead of hidden inside a hard-coded constant.
///
/// TD-IVF-4 is closed (TD-IOBUDGET-2 batch): the compaction/flush callers pass
/// their real destination (`task.output_file` / the collection storage URL), so
/// writer geometry resolves the same per-location budget the reader uses.
/// `None` remains the explicit fallback for callers that genuinely lack a
/// destination (the compactor_impl silent local variant, unit tests).
pub fn write_target_block_bytes(destination_url: Option<&str>) -> u64 {
    match destination_url {
        Some(url) => IopsBudget::for_path(url).target_block_bytes(),
        None => IopsBudget::CLOUD.target_block_bytes(),
    }
}

/// Candidate merged-range caps for the adaptive **read** planner.
///
/// This is intentionally separate from [`IopsBudget::max`]. That field bounds
/// writer block geometry; raising it to improve reads would silently create
/// larger blocks at flush/compaction time. Read candidates instead begin at the
/// backend's conservative target and grow through a small deterministic set up
/// to a provider-bounded ceiling. The exact range planner scores the resulting
/// GET/byte plans for the current selected cells.
///
/// Azure's 24 MiB ceiling covers the measured 1M and 768-d knees without
/// admitting the 32 MiB point that exceeded the declared RSS guard. S3/GCS keep
/// their existing 16 MiB maximum; local storage remains bounded at 8 MiB.
/// Those provider ceilings bound UNREGISTERED paths; a registered location
/// budget (TD-IOBUDGET-1) tightens the enumeration to its own `max` — and an
/// explicit registered target ABOVE a scheme ceiling is honored operator
/// intent (the TD-SEARCH-3-style opt-in), never clamped back to the default.
pub fn read_range_cap_candidates(path: &str) -> Vec<u64> {
    let budget = IopsBudget::for_path(path);
    let scheme_ceiling = match Scheme::of(path) {
        Scheme::Azure => 24 * 1024 * 1024,
        Scheme::S3 | Scheme::Gcs => 16 * 1024 * 1024,
        Scheme::Local | Scheme::Unknown => 8 * 1024 * 1024,
    };
    // TD-IOBUDGET-1: an operator-registered location budget tightens the read
    // enumeration to its own `max` (explicit intent — e.g. an HDD location
    // with a 4 MiB max should never enumerate 8 MiB GETs). Scheme presets
    // deliberately do NOT tighten via `IopsBudget::max`: that field bounds
    // writer block geometry (Azure's 4 MiB writer max must not collapse the
    // measured 24 MiB read enumeration). The floor at `budget.target` keeps
    // the candidate list non-empty under any registered value.
    let ceiling = match location_budget_for(path) {
        Some(registered) => scheme_ceiling.min(registered.max).max(budget.target),
        None => scheme_ceiling,
    };

    let mut caps = Vec::with_capacity(6);
    for multiplier in [1_u64, 2, 3, 4, 6] {
        let cap = budget.target.saturating_mul(multiplier);
        if cap > 0 && cap <= ceiling && caps.last().copied() != Some(cap) {
            caps.push(cap);
        }
    }
    if caps.last().copied() != Some(ceiling) {
        caps.push(ceiling);
    }
    caps
}

#[cfg(test)]
mod tests {
    /// ADR-073: the HDD disk-class hint swaps LOCAL for the CLOUD profile on
    /// file:// paths; unset/other values keep LOCAL. (nextest = process-per-
    /// test, so the env mutation cannot leak.)
    #[test]
    fn disk_class_hdd_selects_cloud_profile_for_local_paths() {
        unsafe { std::env::set_var("PROXIMADB_DISK_CLASS", "hdd") };
        assert_eq!(
            super::IopsBudget::for_path("file:///tmp/x.pax").target,
            super::IopsBudget::CLOUD.target
        );
        unsafe { std::env::set_var("PROXIMADB_DISK_CLASS", "ssd") };
        assert_eq!(
            super::IopsBudget::for_path("file:///tmp/x.pax").target,
            super::IopsBudget::LOCAL.target
        );
        // Cloud schemes unaffected by the hint.
        assert_eq!(
            super::IopsBudget::for_path("s3://b/k").target,
            super::IopsBudget::S3.target
        );
    }

    use super::*;

    #[test]
    fn cloud_vs_local_budgets() {
        assert_eq!(IopsBudget::CLOUD.target, 4 * 1024 * 1024);
        assert_eq!(IopsBudget::LOCAL.target, 1024 * 1024);
        // min ≤ target ≤ max invariant for every preset.
        for b in [IopsBudget::CLOUD, IopsBudget::LOCAL, IopsBudget::DEFAULT] {
            assert!(b.min <= b.target && b.target <= b.max, "{b:?}");
        }
    }

    #[test]
    fn for_path_resolves_by_scheme() {
        assert_eq!(IopsBudget::for_path("s3://bucket/key"), IopsBudget::S3);
        assert_eq!(
            IopsBudget::for_path("abfs://container/path"),
            IopsBudget::AZURE
        );
        assert_eq!(IopsBudget::for_path("adls://acct/fs"), IopsBudget::AZURE);
        assert_eq!(IopsBudget::for_path("gs://b/o"), IopsBudget::GCS);
        assert_eq!(IopsBudget::for_path("/var/data/seg.pax"), IopsBudget::LOCAL);
        assert_eq!(
            IopsBudget::for_path("file:///var/data/seg.pax"),
            IopsBudget::LOCAL
        );
        assert_eq!(
            IopsBudget::for_path("minio://bucket/key"),
            IopsBudget::LOCAL
        );
        // Unknown scheme → generic default.
        assert_eq!(IopsBudget::for_path("redis://x"), IopsBudget::DEFAULT);
    }

    #[test]
    fn rows_per_block_derives_from_per_row_cost() {
        let cloud = IopsBudget::CLOUD;
        // SQ8-only survivor: dim=128 → 128 B/row.
        let sq8_only = 128u64;
        let rows = cloud.rows_per_block(sq8_only);
        // 4 MiB / 128 B = 32768 rows — but capped by `max` (8 MiB / 128 = 65536),
        // so the target wins → 32768.
        assert_eq!(rows, (4 * 1024 * 1024) / 128);
        assert!(rows >= 1);
    }

    #[test]
    fn rows_per_block_floors_at_min_and_caps_at_max() {
        let b = IopsBudget::DEFAULT; // min 512KB, target 2MB, max 8MB
        // Huge per-row cost (1 MiB/row): target/per = 2, min ceil = 1, max floor = 8 → 2.
        assert_eq!(b.rows_per_block(1024 * 1024), 2);
        // Tiny per-row cost (1 B/row): target/per = 2M, but capped at max/per = 8M → 2M.
        assert_eq!(b.rows_per_block(1), 2 * 1024 * 1024);
    }

    #[test]
    fn rows_per_block_zero_cost_falls_back() {
        assert_eq!(
            IopsBudget::DEFAULT.rows_per_block(0),
            FALLBACK_ROWS_PER_BLOCK
        );
    }

    #[test]
    fn target_block_bytes_is_the_target() {
        assert_eq!(IopsBudget::CLOUD.target_block_bytes(), 4 * 1024 * 1024);
    }

    #[test]
    fn adaptive_read_caps_cover_the_measured_azure_knees() {
        let mib = 1024 * 1024;
        let expected = vec![4 * mib, 8 * mib, 12 * mib, 16 * mib, 24 * mib];
        for path in ["az://c/k", "azure://c/k", "adls://c/k", "abfs://c/k"] {
            assert_eq!(read_range_cap_candidates(path), expected, "{path}");
        }
    }

    #[test]
    fn adaptive_read_caps_are_backend_bounded_and_monotone() {
        let mib = 1024 * 1024;
        assert_eq!(
            read_range_cap_candidates("s3://b/k"),
            vec![8 * mib, 16 * mib]
        );
        assert_eq!(
            read_range_cap_candidates("gs://b/k"),
            vec![8 * mib, 16 * mib]
        );
        assert_eq!(
            read_range_cap_candidates("gcs://b/k"),
            vec![8 * mib, 16 * mib]
        );
        assert_eq!(
            read_range_cap_candidates("file:///data/k"),
            vec![mib, 2 * mib, 3 * mib, 4 * mib, 6 * mib, 8 * mib]
        );
        for path in ["az://c/k", "s3://b/k", "file:///data/k", "redis://x/k"] {
            let caps = read_range_cap_candidates(path);
            assert!(!caps.is_empty(), "{path}");
            assert!(caps.windows(2).all(|pair| pair[0] < pair[1]), "{path}");
        }
    }

    // ---- TD-IOBUDGET-1: per-location registry ----
    // Registered prefixes use iobudget-test/* namespaces so tests never
    // contend for the same registry entries (nextest isolates processes, but
    // keep each test's footprint local regardless).

    #[test]
    fn registered_location_budget_wins_over_scheme_default() {
        let registered = IopsBudget {
            min: 512 * 1024,
            target: 16 * 1024 * 1024,
            max: 16 * 1024 * 1024,
        };
        super::register_location_budget("file:///iobudget-test-a", registered);
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-a/coll/seg.pax"),
            registered
        );
        // An unregistered sibling location keeps the scheme default.
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-a-sibling/seg.pax"),
            IopsBudget::LOCAL
        );
    }

    #[test]
    fn longest_registered_prefix_wins() {
        let wide = IopsBudget {
            min: 512 * 1024,
            target: 8 * 1024 * 1024,
            max: 16 * 1024 * 1024,
        };
        let narrow = IopsBudget {
            min: 512 * 1024,
            target: 2 * 1024 * 1024,
            max: 4 * 1024 * 1024,
        };
        super::register_location_budget("file:///iobudget-test-b", wide);
        super::register_location_budget("file:///iobudget-test-b/nvme", narrow);
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-b/nvme/seg.pax"),
            narrow
        );
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-b/sas/seg.pax"),
            wide
        );
    }

    #[test]
    fn registered_prefix_respects_path_boundary() {
        let registered = IopsBudget {
            min: 512 * 1024,
            target: 4 * 1024 * 1024,
            max: 4 * 1024 * 1024,
        };
        super::register_location_budget("s3://iobudget-test-bucket", registered);
        // Inside the bucket → registered; a bucket whose name merely extends
        // the prefix → scheme default.
        assert_eq!(
            IopsBudget::for_path("s3://iobudget-test-bucket/coll/seg.pax"),
            registered
        );
        assert_eq!(
            IopsBudget::for_path("s3://iobudget-test-bucket-two/seg.pax"),
            IopsBudget::S3
        );
    }

    #[test]
    fn registration_is_overwrite_not_first_wins() {
        let first = IopsBudget {
            min: 512 * 1024,
            target: 1 * 1024 * 1024,
            max: 2 * 1024 * 1024,
        };
        let second = IopsBudget {
            min: 512 * 1024,
            target: 8 * 1024 * 1024,
            max: 16 * 1024 * 1024,
        };
        super::register_location_budget("file:///iobudget-test-c", first);
        super::register_location_budget("file:///iobudget-test-c", second);
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-c/seg.pax"),
            second
        );
    }

    #[test]
    fn registered_budget_tightens_read_cap_candidates() {
        let mib = 1024 * 1024;
        let registered = IopsBudget {
            min: 512 * 1024,
            target: mib,
            max: 2 * mib,
        };
        super::register_location_budget("file:///iobudget-test-d", registered);
        // Enumeration now ends at the registered max (2 MiB), not the local
        // 8 MiB scheme ceiling — and stays monotone/non-empty.
        assert_eq!(
            read_range_cap_candidates("file:///iobudget-test-d/seg.pax"),
            vec![mib, 2 * mib]
        );

        // An explicit registered target ABOVE the scheme ceiling is operator
        // intent (the TD-SEARCH-3-style opt-in): the enumeration honors it
        // rather than clamping back to the 8 MiB local default.
        let big = IopsBudget {
            min: 512 * 1024,
            target: 16 * mib,
            max: 16 * mib,
        };
        super::register_location_budget("file:///iobudget-test-e", big);
        assert_eq!(
            read_range_cap_candidates("file:///iobudget-test-e/seg.pax"),
            vec![16 * mib]
        );
    }

    #[test]
    fn scheme_cloud_budget_maps_per_scheme() {
        assert_eq!(
            IopsBudget::scheme_cloud_budget("file:///x"),
            IopsBudget::CLOUD
        );
        assert_eq!(
            IopsBudget::scheme_cloud_budget("minio://b/k"),
            IopsBudget::CLOUD
        );
        assert_eq!(IopsBudget::scheme_cloud_budget("s3://b/k"), IopsBudget::S3);
        assert_eq!(
            IopsBudget::scheme_cloud_budget("az://c/k"),
            IopsBudget::AZURE
        );
        assert_eq!(IopsBudget::scheme_cloud_budget("gs://b/k"), IopsBudget::GCS);
    }

    /// TD-IOBUDGET-1 review finding N8: seeding is authoritative-REPLACE.
    /// (Relies on nextest process isolation — like the env-var test above —
    /// because the drain clears the process-global registry.)
    #[test]
    fn clear_location_budgets_drains_the_registry() {
        let registered = IopsBudget {
            min: 512 * 1024,
            target: 4 * 1024 * 1024,
            max: 8 * 1024 * 1024,
        };
        super::register_location_budget("file:///iobudget-test-clear", registered);
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-clear/seg.pax"),
            registered
        );
        super::clear_location_budgets();
        assert_eq!(
            IopsBudget::for_path("file:///iobudget-test-clear/seg.pax"),
            IopsBudget::LOCAL
        );
    }
}
