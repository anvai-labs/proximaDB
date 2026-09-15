//! # proximadb-object-store — decoupled object-storage plumbing (F1)
//!
//! The bottom layer of the ProximaDB warehouse base tier (course-correction
//! `DATA_WAREHOUSE_AND_ENGINEERING_COURSE_CORRECTION_2026_06_04.adoc` §6 P3): a thin,
//! write-capable wrapper over the `object_store` crate.
//!
//! - [`store_for_url`] maps a URL (`file://`, `memory://`; `s3://`/`gs://`/`az://` behind
//!   the `aws`/`gcp`/`azure` features) to an `Arc<dyn object_store::ObjectStore>` + the base
//!   `Path`, reusing `object_store::parse_url` (the canonical scheme dispatch) — it does NOT
//!   fork a fourth storage abstraction.
//! - [`ProximaObjectStore`] bundles a store + base prefix with `put`/`get`/`get_range`/
//!   `list`/`delete` helpers. Cloud writes are the gap this fills: the existing `FileSystem`
//!   cloud backends are read-only.
//!
//! Plumbing ONLY. Parquet/Iceberg encoding and the concrete `ObjectStoreBridge`
//! implementation live in `proximadb-iceberg-engine` (F2), built on top of this. The
//! `object_store` version is pinned via the workspace dep so the `Arc<dyn ObjectStore>` here
//! is the exact type `proximadb-storage-common`'s `ObjectStoreBridge::inner_store()` expects.

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use object_store::path::Path;
use object_store::{
    Attribute, Attributes, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutResult,
};
use proximadb_kernel::error::StorageError;
use proximadb_storage_filesystem_types::ObjectAccessTier;
use url::Url;

/// Which cloud backend an [`ObjectStore`] handle is, detected from its `Display`
/// type name (stable `object_store` store names: `MicrosoftAzure`, `AmazonS3`,
/// `GoogleCloudStorage`, `LocalFileSystem`, `InMemory`). Substring-matched so it
/// survives middleware wrappers (prefix/limit/throttle). Drives the
/// canonical-tier → provider-native-class mapping in [`ProximaObjectStore::put_with_tier`],
/// because `object_store`'s `Attribute::StorageClass` forwards the value verbatim
/// into the provider header (`x-ms-access-tier` / `x-amz-storage-class`) and each
/// cloud expects its own spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectBackendKind {
    Azure,
    S3,
    Gcs,
    /// Local file / in-memory / unrecognized — no per-object access tier applies.
    Untiered,
}

impl ObjectBackendKind {
    fn detect(store: &dyn ObjectStore) -> Self {
        let name = store.to_string();
        if name.contains("MicrosoftAzure") {
            Self::Azure
        } else if name.contains("AmazonS3") {
            Self::S3
        } else if name.contains("GoogleCloudStorage") {
            Self::Gcs
        } else {
            Self::Untiered
        }
    }

    /// The provider-native storage-class string for `tier`, or `None` for an
    /// untiered backend (local/memory) where the access tier is meaningless.
    fn native_tier(self, tier: ObjectAccessTier) -> Option<&'static str> {
        match self {
            Self::Azure => Some(tier.as_azure_access_tier()),
            Self::S3 => Some(tier.as_s3_storage_class()),
            Self::Gcs => Some(tier.as_gcs_storage_class()),
            Self::Untiered => None,
        }
    }
}

/// Map an `object_store`/url failure to the canonical `StorageError` (the same error the
/// `ObjectStoreBridge` seam returns), preserving NotFound/AlreadyExists where possible.
fn os_err(context: &str, e: object_store::Error) -> StorageError {
    match &e {
        object_store::Error::NotFound { .. } => StorageError::NotFound(format!("{context}: {e}")),
        object_store::Error::AlreadyExists { .. } => {
            StorageError::AlreadyExists(format!("{context}: {e}"))
        }
        _ => StorageError::DiskIO(std::io::Error::other(format!(
            "object_store {context}: {e}"
        ))),
    }
}

fn url_err(url: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::DiskIO(std::io::Error::other(format!(
        "object-store url `{url}`: {e}"
    )))
}

/// Build an object store backend + base `Path` from a URL.
///
/// `file://` and `memory://` always work. `s3://`/`gs://`/`az://` (incl. `adls://`/`abfs://`)
/// require the matching crate feature (`aws`/`gcp`/`azure`); without it the underlying
/// `object_store` builder returns an error (no panic).
///
/// Credentials come from the standard `object_store` environment variables. We forward the
/// process environment (lower-cased so the upper-case `AZURE_*` / `AWS_*` / `GOOGLE_*` names
/// the deployment sets match `object_store`'s lower-case config keys) into `parse_url_opts`;
/// `object_store` applies the keys it recognises for the URL's scheme and silently ignores the
/// rest. This is what makes **secret-less cloud auth** work here: e.g. the AKS workload-identity
/// trio `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_FEDERATED_TOKEN_FILE`, or AWS web-identity
/// (`AWS_ROLE_ARN` / `AWS_WEB_IDENTITY_TOKEN_FILE`), authenticate ADLS/S3 with no static key —
/// matching the `FileSystem` Azure backend's posture. `file://`/`memory://` ignore the options.
pub fn store_for_url(url: &str) -> Result<(Arc<dyn ObjectStore>, Path), StorageError> {
    // ProximaDB scheme aliases (ADR-036: every Azure spelling resolves to the
    // SAME flat Blob backend; `gcs://` is a documented GCS alias). The upstream
    // `object_store` URL parser only recognises `az`/`adl`/`azure`/`abfs[s]`
    // and `gs` — an unnormalized `adls://`/`gcs://` URL errors as an
    // unsupported scheme, which silently demoted catalog durability on Azure
    // deployments (TD-OBJSTORE-1, #960).
    let url = if let Some(rest) = url.strip_prefix("adls://") {
        format!("az://{rest}")
    } else if let Some(rest) = url.strip_prefix("gcs://") {
        format!("gs://{rest}")
    } else {
        url.to_string()
    };
    let parsed = Url::parse(&url).map_err(|e| url_err(&url, e))?;
    let env_opts = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
    let (store, path) = object_store::parse_url_opts(&parsed, env_opts)
        .map_err(|e| os_err(&format!("parse_url({url})"), e))?;
    Ok((Arc::from(store), path))
}

/// A store handle bundled with a base prefix; all helper paths are taken relative to it.
#[derive(Clone)]
pub struct ProximaObjectStore {
    store: Arc<dyn ObjectStore>,
    base: Path,
    /// Backend kind, captured once at construction so [`Self::put_with_tier`] can
    /// map a canonical tier to the provider-native class without re-inspecting the
    /// store per write.
    backend: ObjectBackendKind,
}

/// Process-global observer for reads issued through [`ProximaObjectStore`].
///
/// **Why this exists.** ProximaDB has two object-storage stacks. The
/// `FileSystem` stack records every ranged read into the per-query `IoTrace` at
/// its leaf backends. This one — `ProximaObjectStore` over upstream
/// `object_store` — records **nothing**, so graph cold payloads, graph cold
/// segments, the Iceberg bridge and `RangedSegmentReader` are all invisible to
/// the trace. Every per-query I/O claim the system makes today therefore covers
/// only the vector and relational paths.
///
/// **Why a process-global rather than constructor injection.** `from_url` is
/// called from a dozen scattered sites (graph stores, cluster leases, tests).
/// Threading a recorder through each one would leave any missed site silently
/// untraced — which is the same failure shape as the gap being closed. A single
/// install point cannot be partially applied. This mirrors
/// `counting::global_counters()` and `init_coarse_probe_settings`.
///
/// **Layering.** [`IoRecorder`] lives in `proximadb-storage-filesystem-types`,
/// a leaf this crate already depends on, so the root can inject an impl that
/// writes into its own `io_trace` without this storage-layer crate ever
/// depending up into `crates/modalities`.
static IO_RECORDER: std::sync::OnceLock<
    std::sync::Arc<dyn proximadb_storage_filesystem_types::counting::IoRecorder>,
> = std::sync::OnceLock::new();

/// Install the process-wide read observer. First call wins; later calls are
/// ignored so a test cannot silently displace the server's recorder.
///
/// Unset (the default) means no recording at all — byte-identical to the
/// behaviour before this hook existed.
pub fn install_io_recorder(
    recorder: std::sync::Arc<dyn proximadb_storage_filesystem_types::counting::IoRecorder>,
) {
    let _ = IO_RECORDER.set(recorder);
}

fn io_recorder()
-> Option<&'static std::sync::Arc<dyn proximadb_storage_filesystem_types::counting::IoRecorder>> {
    IO_RECORDER.get()
}

impl ProximaObjectStore {
    /// Open a store from a URL (see [`store_for_url`]).
    pub fn from_url(url: &str) -> Result<Self, StorageError> {
        let (store, base) = store_for_url(url)?;
        let backend = ObjectBackendKind::detect(store.as_ref());
        Ok(Self {
            store,
            base,
            backend,
        })
    }

    /// Wrap an existing `object_store` handle (base = root). Useful when the store is built
    /// elsewhere (e.g. a shared `Arc<dyn ObjectStore>` from the bridge).
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        let backend = ObjectBackendKind::detect(store.as_ref());
        Self {
            store,
            base: Path::default(),
            backend,
        }
    }

    /// The detected cloud backend kind (drives [`Self::put_with_tier`]).
    pub fn backend(&self) -> ObjectBackendKind {
        self.backend
    }

    /// The underlying object store (for `ObjectStoreBridge::inner_store()` and direct use).
    pub fn store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    /// The base prefix this handle resolves relative paths against.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Join the base prefix with a caller-relative path.
    pub fn full_path(&self, path: &Path) -> Path {
        let base = self.base.as_ref();
        if base.is_empty() {
            path.clone()
        } else {
            Path::from(format!("{base}/{path}"))
        }
    }

    /// Write `bytes` to `path` (atomic for stores that support it). Overwrites.
    pub async fn put(&self, path: &Path, bytes: Bytes) -> Result<(), StorageError> {
        self.put_opts(path, bytes, PutOptions::default())
            .await
            .map(|_| ())
            .map_err(|e| os_err("put", e))
    }

    /// Write through the existing backend with its native preconditions and
    /// return its opaque revision. Paths remain relative to this handle's base.
    ///
    /// Unlike the compatibility helpers returning [`StorageError`], this method
    /// preserves upstream `AlreadyExists`, `Precondition`, `NotImplemented` and
    /// transport errors. It adds no retries and never substitutes an unconditional
    /// write. The backend's transport may retry internally: even a returned
    /// `Precondition`/`AlreadyExists` can follow an earlier committed attempt whose
    /// response was lost. Callers must reconcile their durable operation identity
    /// before treating an error as proof that nothing changed.
    ///
    /// `Update` must carry a nonempty validator. Preserve BOTH ETag and version
    /// from a previous PUT or [`Self::get_with_meta`]; which fields are used is
    /// backend-specific. An ETag is not an ownership generation or permission.
    /// Content-derived ETags can repeat: authority records must carry a non-reused
    /// revision/incarnation, and must not be deleted/recreated as a CAS shortcut.
    /// Local files currently reject Update; no rename-based emulation is used.
    pub async fn put_opts(
        &self,
        path: &Path,
        bytes: Bytes,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        if let PutMode::Update(expected) = &options.mode
            && ((expected.e_tag.is_none() && expected.version.is_none())
                || expected.e_tag.as_deref() == Some("")
                || expected.version.as_deref() == Some(""))
        {
            return Err(object_store::Error::Generic {
                store: "ProximaObjectStore",
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "conditional update requires nonempty version/ETag validators",
                )),
            });
        }
        self.store
            .put_opts(&self.full_path(path), bytes.into(), options)
            .await
    }

    /// Write `bytes` to `path` at a per-object **access tier** — the object-storage
    /// cost lever (ADR-035/036, TD-173): a colder tier trades retrieval latency/cost
    /// for a far lower at-rest GB-month price. The canonical [`ObjectAccessTier`] is
    /// mapped to the backend's native class (Azure `x-ms-access-tier` / S3
    /// `x-amz-storage-class` / GCS storage class) and set via `object_store`'s
    /// `Attribute::StorageClass` on the PUT. On an untiered backend (local/memory)
    /// the tier is meaningless, so this degrades to a plain overwrite [`Self::put`].
    ///
    /// This closes the TD-173 `ProximaObjectStore` tier gap (the FileSystem Azure/S3
    /// backends already do this); it is the prerequisite for tiering cold
    /// graph/warehouse payloads written through the object-store path. It is a
    /// **capability only** — callers opt in per write, so no data is tiered until a
    /// caller asks for it.
    pub async fn put_with_tier(
        &self,
        path: &Path,
        bytes: Bytes,
        tier: ObjectAccessTier,
    ) -> Result<(), StorageError> {
        let native = match self.backend.native_tier(tier) {
            Some(native) => native,
            // Untiered backend: the access tier has no meaning — write normally.
            None => return self.put(path, bytes).await,
        };
        let mut attributes = Attributes::new();
        attributes.insert(Attribute::StorageClass, native.into());
        let opts = PutOptions {
            attributes,
            ..Default::default()
        };
        self.put_opts(path, bytes, opts)
            .await
            .map(|_| ())
            .map_err(|e| os_err("put_with_tier", e))
    }

    /// Atomically write `bytes` to `path` ONLY if no object already exists there
    /// (`PutMode::Create`). Returns [`StorageError::AlreadyExists`] if the object
    /// is already present, and never overwrites it.
    ///
    /// This is the optimistic-concurrency primitive for Iceberg-style manifest
    /// commits (the warehouse base tier): a committer writes a new
    /// manifest/metadata object under a fresh name with create-only semantics,
    /// so two concurrent committers cannot clobber each other's commit — the
    /// a collision returns `AlreadyExists`. Backend retries can also return
    /// that error after this caller's earlier attempt committed; reconcile the
    /// operation before retrying (see [`Self::put_opts`]).
    /// (Supported by the `memory` and local-file backends; cloud backends need
    /// conditional-put support.)
    pub async fn put_if_absent(&self, path: &Path, bytes: Bytes) -> Result<(), StorageError> {
        self.put_opts(path, bytes, PutOptions::from(PutMode::Create))
            .await
            .map(|_| ())
            .map_err(|e| os_err("put_if_absent", e))
    }

    /// Atomically create `path` at the requested access tier. This combines the
    /// collision safety of [`Self::put_if_absent`] with the provider-native
    /// storage-class mapping of [`Self::put_with_tier`]. Untiered stores simply
    /// use create-only mode.
    pub async fn put_if_absent_with_tier(
        &self,
        path: &Path,
        bytes: Bytes,
        tier: ObjectAccessTier,
    ) -> Result<(), StorageError> {
        let mut opts = PutOptions::from(PutMode::Create);
        if let Some(native) = self.backend.native_tier(tier) {
            opts.attributes
                .insert(Attribute::StorageClass, native.into());
        }
        self.put_opts(path, bytes, opts)
            .await
            .map(|_| ())
            .map_err(|e| os_err("put_if_absent_with_tier", e))
    }

    /// Read the whole object at `path`.
    pub async fn get(&self, path: &Path) -> Result<Bytes, StorageError> {
        self.get_with_meta(path)
            .await
            .map(|(bytes, _)| bytes)
            .map_err(|e| os_err("get", e))
    }

    /// Read body and metadata from the SAME backend GET, retaining both opaque
    /// revision fields for conditional publication. A separate HEAD then GET is
    /// not a consistent basis for a read/modify/conditional-write transition.
    ///
    /// Preserves upstream read errors. Successful bytes are opaque: schema and
    /// integrity validation belong to the caller. Like [`Self::get`], records
    /// one full read only after the body is successfully consumed. Metadata locations retain
    /// upstream semantics; use the original caller-relative path for updates.
    pub async fn get_with_meta(&self, path: &Path) -> object_store::Result<(Bytes, ObjectMeta)> {
        let result = self.store.get(&self.full_path(path)).await?;
        let meta = result.meta.clone();
        let bytes = result.bytes().await?;
        if let Some(recorder) = io_recorder() {
            recorder.record_full_read(bytes.len() as u64);
        }
        Ok((bytes, meta))
    }

    /// Read a byte range of the object at `path` (the warehouse footer/row-group read path).
    pub async fn get_range(&self, path: &Path, range: Range<u64>) -> Result<Bytes, StorageError> {
        let bytes = self
            .store
            .get_range(&self.full_path(path), range)
            .await
            .map_err(|e| os_err("get_range", e))?;
        if let Some(recorder) = io_recorder() {
            recorder.record_range_read(bytes.len() as u64);
        }
        Ok(bytes)
    }

    /// Read **multiple** byte ranges of the object at `path` in one batched call.
    /// `object_store::get_ranges` coalesces adjacent ranges and issues them
    /// concurrently, so K block-body reads cost ~one round-trip instead of K
    /// serial GETs — the depth-collapse primitive for TD-167 / ADR-034 P1.
    pub async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>, StorageError> {
        let buffers = self
            .store
            .get_ranges(&self.full_path(path), ranges)
            .await
            .map_err(|e| os_err("get_ranges", e))?;
        if let Some(recorder) = io_recorder() {
            // Upstream `object_store::get_ranges` coalesces internally at a
            // hard-coded 1 MiB gap and does not report how many HTTP requests it
            // actually issued, so the physical count is NOT observable here.
            // Reporting `ranges.len()` is the honest upper bound: it can only
            // over-state requests, never hide them. Recorded as such rather than
            // silently presented as measured truth.
            let bytes: u64 = buffers.iter().map(|b| b.len() as u64).sum();
            recorder.record_batched_ranges(ranges.len() as u64, bytes);
        }
        Ok(buffers)
    }

    /// Fetch object metadata (size, last-modified, e-tag) for `path` WITHOUT
    /// reading the body.
    ///
    /// This is the prerequisite for the footer/row-group range-read path
    /// ([`get_range`]): a Parquet reader must know the object length to compute
    /// the trailing footer's byte range before it can range-read it. `head` is a
    /// metadata-only request (cheap on cloud stores — an HTTP HEAD), so callers
    /// avoid a full-object GET just to learn the size.
    pub async fn head(&self, path: &Path) -> Result<ObjectMeta, StorageError> {
        self.store
            .head(&self.full_path(path))
            .await
            .map_err(|e| os_err("head", e))
    }

    /// The byte length of the object at `path` (metadata-only; see [`head`]).
    /// Returns the value that bounds a `get_range` over the whole object.
    pub async fn object_size(&self, path: &Path) -> Result<u64, StorageError> {
        Ok(self.head(path).await?.size)
    }

    /// Read the last `n` bytes of the object at `path` — the Parquet-footer read
    /// pattern (a reader fetches the trailing bytes to locate the footer length,
    /// then the footer itself).
    ///
    /// Implemented as a metadata [`head`] (to learn the size) followed by a
    /// bounded [`get_range`]. `n` is clamped to the object size, so requesting
    /// more bytes than the object holds returns the whole object; `n == 0`
    /// returns empty without any request. (A future optimization could use the
    /// object store's native suffix range to save the `head` round-trip.)
    pub async fn get_suffix(&self, path: &Path, n: u64) -> Result<Bytes, StorageError> {
        if n == 0 {
            return Ok(Bytes::new());
        }
        let size = self.object_size(path).await?;
        if size == 0 {
            return Ok(Bytes::new());
        }
        let start = size.saturating_sub(n);
        self.get_range(path, start..size).await
    }

    /// List objects under an optional caller-relative prefix. A `None` prefix lists under
    /// the base (NOT the whole store/filesystem root — `file://` parses to root `/`).
    pub async fn list(&self, prefix: Option<&Path>) -> Result<Vec<ObjectMeta>, StorageError> {
        let resolved = match prefix {
            Some(p) => self.full_path(p),
            None => self.base.clone(),
        };
        let mut stream = self.store.list(Some(&resolved));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.map_err(|e| os_err("list", e))?);
        }
        Ok(out)
    }

    /// Delete the object at `path`.
    pub async fn delete(&self, path: &Path) -> Result<(), StorageError> {
        self.store
            .delete(&self.full_path(path))
            .await
            .map_err(|e| os_err("delete", e))
    }
}

#[cfg(test)]
mod tests {
    /// TD-OBJSTORE-1 (#960): the documented `adls://`/`gcs://` aliases must
    /// normalize to schemes the upstream `object_store` URL parser knows
    /// (`az://`, `gs://`). Under the matching cloud feature they open a store;
    /// without it they fail with the parser/builder error for the NORMALIZED
    /// scheme — never `Unknown url scheme "adls"`.
    #[test]
    fn adls_and_gcs_aliases_normalize_before_parse() {
        for (alias, canonical) in [
            ("adls://container/prefix/x", "az"),
            ("gcs://bucket/prefix/x", "gs"),
        ] {
            match store_for_url(alias) {
                // Cloud feature compiled in: the alias opens a store.
                Ok(_) => {}
                // Feature off: the error must be about the canonical scheme's
                // backend, not an unrecognized alias.
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        !msg.contains("adls") && !msg.contains("gcs"),
                        "alias {alias} leaked through unnormalized (canonical {canonical}): {msg}"
                    );
                }
            }
        }
    }

    use super::*;

    #[tokio::test]
    async fn memory_store_roundtrip() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let os = ProximaObjectStore::new(store);
        let p = Path::from("a/b.bin");

        os.put(&p, Bytes::from_static(b"hello world"))
            .await
            .unwrap();
        assert_eq!(&os.get(&p).await.unwrap()[..], b"hello world");
        assert_eq!(&os.get_range(&p, 0..5).await.unwrap()[..], b"hello");
        assert_eq!(os.list(None).await.unwrap().len(), 1);

        os.delete(&p).await.unwrap();
        assert!(os.get(&p).await.is_err());
    }

    #[tokio::test]
    async fn file_url_roundtrip_under_base_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("file://{}", dir.path().display());
        let os = ProximaObjectStore::from_url(&url).unwrap();
        let p = Path::from("sub/x.parquet");

        os.put(&p, Bytes::from_static(b"parquet-bytes"))
            .await
            .unwrap();
        assert_eq!(&os.get(&p).await.unwrap()[..], b"parquet-bytes");
        assert_eq!(&os.get_range(&p, 0..7).await.unwrap()[..], b"parquet");
        assert_eq!(os.list(None).await.unwrap().len(), 1);
        // The write landed under the URL's directory (base prefix honored).
        assert!(dir.path().join("sub/x.parquet").exists());
    }

    /// `head`/`object_size` report the object length without a body GET, and
    /// that length is exactly what bounds a trailing-footer `get_range` — the
    /// warehouse Parquet-footer read pattern. A missing object errors (no panic).
    #[tokio::test]
    async fn head_reports_size_and_enables_footer_range_read() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        let p = Path::from("t/data.parquet");
        os.put(&p, Bytes::from_static(b"0123456789")).await.unwrap();

        let meta = os.head(&p).await.unwrap();
        assert_eq!(meta.size, 10, "head reports the exact byte length");
        assert_eq!(os.object_size(&p).await.unwrap(), 10);

        // The size drives a "last 4 bytes" footer-style range read.
        let n = os.object_size(&p).await.unwrap();
        let footer = os.get_range(&p, (n - 4)..n).await.unwrap();
        assert_eq!(&footer[..], b"6789");

        // Metadata on a missing object surfaces an error, not a panic.
        assert!(os.head(&Path::from("missing.parquet")).await.is_err());
    }

    /// `get_suffix` reads the trailing N bytes (the Parquet-footer pattern),
    /// clamps N to the object size, treats N==0 as empty, and errors on a
    /// missing object.
    #[tokio::test]
    async fn get_suffix_reads_trailing_bytes() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        let p = Path::from("t/data.parquet");
        os.put(&p, Bytes::from_static(b"0123456789")).await.unwrap(); // 10 bytes

        assert_eq!(&os.get_suffix(&p, 4).await.unwrap()[..], b"6789"); // last 4
        assert_eq!(
            &os.get_suffix(&p, 100).await.unwrap()[..],
            b"0123456789",
            "n >= size returns the whole object"
        );
        assert!(
            os.get_suffix(&p, 0).await.unwrap().is_empty(),
            "n == 0 returns empty"
        );
        assert!(
            os.get_suffix(&Path::from("missing.parquet"), 4)
                .await
                .is_err(),
            "missing object errors"
        );
    }

    /// `put_if_absent` creates a new object but rejects (and does not overwrite)
    /// an existing one — the Iceberg-style commit-atomicity primitive.
    #[tokio::test]
    async fn put_if_absent_is_create_only() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        let p = Path::from("manifest/v1.json");

        // First create succeeds.
        os.put_if_absent(&p, Bytes::from_static(b"first"))
            .await
            .unwrap();
        assert_eq!(&os.get(&p).await.unwrap()[..], b"first");

        // Second create is rejected and leaves the existing object untouched.
        let err = os.put_if_absent(&p, Bytes::from_static(b"second")).await;
        assert!(
            matches!(err, Err(StorageError::AlreadyExists(_))),
            "create-only must reject an existing key, got {err:?}"
        );
        assert_eq!(
            &os.get(&p).await.unwrap()[..],
            b"first",
            "rejected create must not overwrite"
        );
    }

    #[tokio::test]
    async fn put_if_absent_with_tier_is_create_only_on_untiered_store() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        let p = Path::from("trace/segment.jsonl.zst");

        os.put_if_absent_with_tier(&p, Bytes::from_static(b"first"), ObjectAccessTier::Cold)
            .await
            .unwrap();
        let err = os
            .put_if_absent_with_tier(&p, Bytes::from_static(b"second"), ObjectAccessTier::Cold)
            .await;
        assert!(matches!(err, Err(StorageError::AlreadyExists(_))));
        assert_eq!(&os.get(&p).await.unwrap()[..], b"first");
    }

    /// The canonical tier maps to each cloud's native storage-class spelling; an
    /// untiered backend yields `None` (the tier is meaningless there).
    #[test]
    fn tier_native_mapping_per_backend() {
        use ObjectBackendKind::*;
        assert_eq!(Azure.native_tier(ObjectAccessTier::Cool), Some("Cool"));
        assert_eq!(Azure.native_tier(ObjectAccessTier::Cold), Some("Cold"));
        assert_eq!(S3.native_tier(ObjectAccessTier::Cool), Some("STANDARD_IA"));
        assert_eq!(Gcs.native_tier(ObjectAccessTier::Cool), Some("NEARLINE"));
        assert_eq!(Untiered.native_tier(ObjectAccessTier::Cool), None);
    }

    /// A memory/local store is detected as untiered.
    #[test]
    fn memory_store_is_untiered() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        assert_eq!(os.backend(), ObjectBackendKind::Untiered);
    }

    /// `put_with_tier` on an untiered backend degrades to a plain write (the tier is a
    /// no-op there) — so callers can request a tier unconditionally and it only takes
    /// effect on a cloud backend. Asserting the bytes land verifies the degrade path.
    #[tokio::test]
    async fn put_with_tier_degrades_to_put_on_untiered_backend() {
        let os = ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()));
        let p = Path::from("cold/seg.pax");
        os.put_with_tier(&p, Bytes::from_static(b"payload"), ObjectAccessTier::Cool)
            .await
            .unwrap();
        assert_eq!(&os.get(&p).await.unwrap()[..], b"payload");
    }

    #[test]
    fn memory_scheme_dispatches() {
        assert!(store_for_url("memory:///").is_ok());
    }

    #[test]
    fn cloud_scheme_without_feature_errors_not_panics() {
        // s3:// needs the `aws` feature; the default build must error gracefully.
        let result = store_for_url("s3://bucket/key");
        #[cfg(not(feature = "aws"))]
        assert!(result.is_err());
        let _ = result;
    }

    // ── Cloud-emulator integration: put_with_tier against real Azure/S3/GCS APIs ──
    //
    // The memory-degrade unit test cannot catch a native-class mapping regression
    // (an invalid storage-class string a real cloud API would 4xx). These run the
    // tier path against emulators — Azurite (Azure), MinIO (S3), fake-gcs (GCP) —
    // so the `x-ms-access-tier` / `x-amz-storage-class` / `x-goog-storage-class`
    // header is exercised end-to-end. Azure + S3 go through the PRODUCTION
    // `store_for_url` + forwarded-env path (highest fidelity); GCS uses the builder
    // (object_store has no clean emulator env key for GCS) and is best-effort.
    //
    // `object_store` 0.13 does not surface the tier on read (GET/HEAD attributes
    // omit it; see `client::get::get_attributes`), so these assert *acceptance* +
    // round-trip, not the resident tier value. The CI job (qa-gate) adds an
    // out-of-band Azurite tier read-back (`az ... --query blobTier`) for the strong
    // proof; a strong in-test read-back needs the Azure SDK (deferred, TD-168).
    //
    // Gated `#[ignore]` + per-cloud env presence, so they compile under
    // `--features aws,azure,gcp`/`cloud-full` (CI-drift-safe) but run only when the
    // matching emulator env is set. See `.github/workflows/qa-gate.yml` (the
    // develop→qa gate) and `make cloud-emulator-test` (local) /
    // `docs/12-design/runtime-evidence/TD168_COOL_TIER_AZURITE_VALIDATION_2026_06_28.md`.

    /// Azure (Azurite) via the production `from_url` + env path. Azurite accepts AND
    /// persists `x-ms-access-tier: Cool` (the CI job verifies the resident tier
    /// out-of-band). Set by CI: `AZURE_STORAGE_USE_EMULATOR=true`, `AZURE_ALLOW_HTTP=true`.
    #[cfg(feature = "azure")]
    #[tokio::test]
    #[ignore = "needs Azurite — set AZURE_STORAGE_USE_EMULATOR=true with Azurite running"]
    async fn put_with_tier_accepted_by_azurite() {
        if std::env::var("AZURE_STORAGE_USE_EMULATOR").is_err() {
            eprintln!("skip: set AZURE_STORAGE_USE_EMULATOR=true with Azurite running");
            return;
        }
        let os = ProximaObjectStore::from_url("az://proximadb-test/cold/probe-azure.bin")
            .or_else(|_| ProximaObjectStore::from_url("az://proximadb-test"))
            .expect("open Azurite store via from_url");
        assert_eq!(os.backend(), ObjectBackendKind::Azure);
        let p = Path::from("cold/probe-azure.bin");
        os.put_with_tier(
            &p,
            Bytes::from_static(b"cool-azure"),
            ObjectAccessTier::Cool,
        )
        .await
        .expect("Azurite must accept x-ms-access-tier: Cool");
        assert_eq!(&os.get(&p).await.expect("get")[..], b"cool-azure");
        // NOTE: intentionally NOT deleted — the qa-gate job reads this blob's
        // resident tier back out-of-band (`az ... --query blobTier`) for the strong
        // Cool-tier proof object_store cannot surface. The emulator is ephemeral.
    }

    /// AWS S3 (MinIO) via the production `from_url` + env path. Default MinIO
    /// *rejects* `x-amz-storage-class: STANDARD_IA` (InvalidStorageClass) unless
    /// object tiering/ILM is configured — impractical for a CI emulator — so this
    /// best-effort-skips on that rejection (real AWS S3 accepts it; the header
    /// mapping is unit-tested via `native_tier`). It still hard-fails on a
    /// non-storage-class error. Set by CI: `AWS_ENDPOINT`, `AWS_ALLOW_HTTP=true`,
    /// `AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false`, `AWS_ACCESS_KEY_ID/SECRET/REGION`.
    #[cfg(feature = "aws")]
    #[tokio::test]
    #[ignore = "needs MinIO/S3 — set AWS_ENDPOINT with the emulator running"]
    async fn put_with_tier_accepted_by_minio() {
        if std::env::var("AWS_ENDPOINT").is_err() && std::env::var("AWS_ENDPOINT_URL").is_err() {
            eprintln!("skip: set AWS_ENDPOINT to the MinIO/S3 emulator");
            return;
        }
        let os = ProximaObjectStore::from_url("s3://proximadb-test/cold/probe-s3.bin")
            .or_else(|_| ProximaObjectStore::from_url("s3://proximadb-test"))
            .expect("open MinIO/S3 store via from_url");
        assert_eq!(os.backend(), ObjectBackendKind::S3);
        let p = Path::from("cold/probe-s3.bin");
        // Best-effort: skip on the emulator's STANDARD_IA rejection (mirrors the
        // GCS pattern); fail on other errors. The header mapping itself is
        // unit-tested (`native_tier(ObjectAccessTier::Cool) == "STANDARD_IA"`).
        match os
            .put_with_tier(&p, Bytes::from_static(b"cool-s3"), ObjectAccessTier::Cool)
            .await
        {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{e:?}");
                if msg.contains("InvalidStorageClass") {
                    eprintln!(
                        "skip (best-effort): MinIO emulator rejected STANDARD_IA \
                         (InvalidStorageClass) — real AWS S3 accepts it; the \
                         put_with_tier mapping is unit-tested separately. err: {msg}"
                    );
                    return;
                }
                panic!("MinIO/S3 put_with_tier failed (non-storage-class error): {msg}");
            }
        }
        assert_eq!(&os.get(&p).await.expect("get")[..], b"cool-s3");
        let _ = os.delete(&p).await;
    }

    /// GCP (fake-gcs-server) via the builder (`object_store` has no emulator env key
    /// for GCS). Best-effort ("to extent feasible"): fake-gcs may not honor the
    /// `x-goog-storage-class` header, and object_store's GCS builder may reject an
    /// anonymous/emulator build — so a connect/build failure SKIPS rather than fails
    /// (this is the known-fragile backend; tracked in TD-168). Set by CI:
    /// `PROXIMADB_GCS_TEST_ENDPOINT=http://localhost:4443`.
    #[cfg(feature = "gcp")]
    #[tokio::test]
    #[ignore = "needs fake-gcs — set PROXIMADB_GCS_TEST_ENDPOINT with the emulator running"]
    async fn put_with_tier_against_fake_gcs() {
        let endpoint = match std::env::var("PROXIMADB_GCS_TEST_ENDPOINT") {
            Ok(e) => e,
            Err(_) => {
                eprintln!("skip: set PROXIMADB_GCS_TEST_ENDPOINT to the fake-gcs emulator");
                return;
            }
        };
        let built = object_store::gcp::GoogleCloudStorageBuilder::new()
            .with_base_url(&endpoint)
            .with_bucket_name("proximadb-test")
            .build();
        let store = match built {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip (best-effort): fake-gcs build unsupported by object_store: {e}");
                return;
            }
        };
        let os = ProximaObjectStore::new(Arc::new(store));
        assert_eq!(os.backend(), ObjectBackendKind::Gcs);
        let p = Path::from("cold/probe-gcs.bin");
        if let Err(e) = os
            .put_with_tier(&p, Bytes::from_static(b"cool-gcs"), ObjectAccessTier::Cool)
            .await
        {
            eprintln!("skip (best-effort): fake-gcs rejected the tiered PUT: {e}");
            return;
        }
        assert_eq!(&os.get(&p).await.expect("get")[..], b"cool-gcs");
        let _ = os.delete(&p).await;
    }
}

#[cfg(test)]
mod io_recorder_tests {
    //! Stack B was invisible: `ProximaObjectStore` goes straight to upstream
    //! `object_store` and recorded nothing, so graph cold storage, the Iceberg
    //! bridge and `RangedSegmentReader` contributed no bytes and no ranged GETs
    //! to any per-query trace.
    use super::{ProximaObjectStore, install_io_recorder};
    use proximadb_storage_filesystem_types::counting::IoRecorder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, Default)]
    struct SpyRecorder {
        full_reads: AtomicU64,
        range_reads: AtomicU64,
        batched_calls: AtomicU64,
        reported_gets: AtomicU64,
        bytes: AtomicU64,
    }

    impl IoRecorder for SpyRecorder {
        fn record_full_read(&self, bytes: u64) {
            self.full_reads.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
        fn record_range_read(&self, bytes: u64) {
            self.range_reads.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
        fn record_batched_ranges(&self, physical_gets: u64, bytes: u64) {
            self.batched_calls.fetch_add(1, Ordering::Relaxed);
            self.reported_gets
                .fetch_add(physical_gets, Ordering::Relaxed);
            self.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Reads through this stack must reach the recorder at all — the whole point
    /// of the hook. Covers the three read shapes graph/Iceberg actually use:
    /// whole-object, single range, and batched ranges.
    #[tokio::test]
    async fn reads_through_the_object_store_reach_the_recorder() {
        let spy = Arc::new(SpyRecorder::default());
        // First install wins process-wide; if another test installed first this
        // assertion tells us plainly rather than failing somewhere confusing.
        install_io_recorder(spy.clone());
        if !Arc::ptr_eq(
            &(super::io_recorder()
                .expect("a recorder is installed")
                .clone()),
            &(spy.clone() as Arc<dyn IoRecorder>),
        ) {
            eprintln!("another recorder already owns the process slot; skipping");
            return;
        }

        let store = ProximaObjectStore::from_url("memory://").expect("memory store");
        let path = object_store::path::Path::from("probe");
        store
            .put(&path, bytes::Bytes::from_static(&[7u8; 512]))
            .await
            .expect("put");

        store.get(&path).await.expect("whole-object read");
        assert_eq!(spy.full_reads.load(Ordering::Relaxed), 1);

        let (body, meta) = store
            .get_with_meta(&path)
            .await
            .expect("body plus revision");
        assert_eq!(body.len() as u64, meta.size);
        assert_eq!(
            spy.full_reads.load(Ordering::Relaxed),
            2,
            "one read, not HEAD plus GET or double accounting"
        );
        assert_eq!(spy.bytes.load(Ordering::Relaxed), 1024);

        assert!(
            store
                .get_with_meta(&object_store::path::Path::from("absent"))
                .await
                .is_err()
        );
        assert_eq!(
            spy.full_reads.load(Ordering::Relaxed),
            2,
            "failed reads do not report successful body reads"
        );

        store.get_range(&path, 0..64).await.expect("ranged read");
        assert_eq!(spy.range_reads.load(Ordering::Relaxed), 1);

        store
            .get_ranges(&path, &[0..16, 32..48, 400..416])
            .await
            .expect("batched read");
        assert_eq!(spy.batched_calls.load(Ordering::Relaxed), 1);

        // The reported request count is an UPPER BOUND, not a measurement:
        // upstream coalesces internally at a hard-coded 1 MiB gap and never says
        // how many HTTP requests it issued. Three logical ranges 400 bytes apart
        // certainly became ONE upstream request, yet we report 3 — over-stating
        // cost, never hiding it. Pinned so nobody later mistakes it for measured
        // truth or "fixes" it into silence.
        assert_eq!(
            spy.reported_gets.load(Ordering::Relaxed),
            3,
            "batched ranges are reported as an upper bound of one per logical range"
        );
        assert!(
            spy.bytes.load(Ordering::Relaxed) > 0,
            "bytes must be recorded"
        );
    }
}
