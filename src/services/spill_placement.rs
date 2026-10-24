//! ADR-097 P2 / TD-SEAM-3: where the relational spill writes its segments.
//!
//! By default segments sit on local disk beside the canonical WAL
//! (`{data_dir}/pgwire/record-spill`). With [`SPILL_URL_ENV`] set they go to
//! an object store instead, at `{url}/{tenant}/{collection}` — a prefix derived
//! from exactly what WAL replay carries, so recovery needs no catalog lookup.
//!
//! Once the WAL below a segment is truncated, the segment is the only copy of
//! its rows. Serving from any other set of segments — an old placement, a
//! wiped or swapped bucket, a prefix another server writes to — would serve a
//! silent subset, so boot refuses each of these:
//!
//! * **Placement marker** (`{data_dir}/pgwire/SPILL_PLACEMENT`): where this
//!   directory's segments live, its identity (`instance`) and its claim
//!   `epoch`. A placement change refuses while the old placement holds rows
//!   the truncated WAL does not, and whenever the new one already holds
//!   segments (they would be re-adopted as stale rows).
//! * **Owner marker** (`{url}/_proximadb_spill_owner.json`): the prefix's
//!   owner instance and epoch. Every boot with an object placement advances
//!   the epoch on both markers. Boot refuses a prefix another instance owns;
//!   one nested inside or around another server's prefix; one that lost its
//!   owner marker, or whose epoch went backwards, after truncation (wiped,
//!   rolled back, or the URL now reaches a different store); and one whose
//!   epoch moved past this directory's (a copy of this data directory booted
//!   against it since).
//!
//! A migration of a truncated server moves its segments to the new placement
//! and boots once with [`SPILL_ADOPT_ENV`] set.
//!
//! Limits, all checked at boot only:
//! * a copy of a data directory booted while the original runs shares the
//!   prefix until the original restarts (and is fenced); either one's DROP can
//!   delete segments the other serves;
//! * a rollback of the bucket to a point inside the current run (same epoch)
//!   goes unnoticed;
//! * the owner marker is overwritten without compare-and-swap, so two copies
//!   booting in the same instant can both pass, and two servers claiming
//!   nested prefixes at the same instant can both pass;
//! * nesting is checked between instances: copies of one data directory share
//!   an instance, so one can take a prefix nested in the other's (their
//!   segment paths stay disjoint unless a tenant is named like the sub-prefix).
//!
//! The WAL itself stays local: an object-store placement offloads disk and
//! makes segments reachable to readers, but durability still rests on the
//! local WAL until the WAL moves too.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::services::record_spill::is_absent;
use crate::storage::persistence::filesystem::{
    FileOptions, FileSystem, FilesystemError, FilesystemFactory,
};

/// Object-store URL (`s3://…`, `az://…`, `gs://…`, `file:///…`) the relational
/// spill writes its segments under. Unset or empty: local, beside the WAL.
pub const SPILL_URL_ENV: &str = "PROXIMADB_RELATIONAL_SPILL_URL";

/// `1` lets one boot's placement change adopt the segments already at the new
/// placement — for a migration where the operator moved them there. Unset the
/// next boot.
pub const SPILL_ADOPT_ENV: &str = "PROXIMADB_RELATIONAL_SPILL_ADOPT";

/// Placement marker, beside the canonical WAL.
pub const PLACEMENT_MARKER: &str = "SPILL_PLACEMENT";

/// Owner marker, at the root of an object-store placement. Not a segment, so
/// discovery (which reads `spill-*.parquet` below partition prefixes) never
/// sees it.
pub const OWNER_MARKER: &str = "_proximadb_spill_owner.json";

/// The placement value recorded for local segments.
const LOCAL: &str = "local";

/// Attempts at the object-store claim before boot gives up on a store it
/// cannot reach. Refusals are decisive and never retried.
const CLAIM_ATTEMPTS: u32 = 3;

/// Where a server's spill segments go.
#[derive(Clone)]
pub enum SpillTarget {
    /// A local directory, beside the canonical WAL.
    Local(PathBuf),
    /// An object-store prefix (canonical, see [`canonical_spill_url`]) and the
    /// filesystem that serves it.
    Object {
        url: String,
        filesystem: Arc<dyn FileSystem>,
    },
}

/// What a target holds from earlier runs.
#[derive(Default)]
struct Scan {
    /// Any file but this target's own owner marker: a segment or a purge
    /// obligation. Empty directories are not state.
    has_state: bool,
    /// Owner markers below this target other than its own (URLs).
    nested_owners: Vec<String>,
}

impl SpillTarget {
    /// How the placement marker names this target.
    fn placement(&self) -> &str {
        match self {
            Self::Local(_) => LOCAL,
            Self::Object { url, .. } => url,
        }
    }

    /// The directory or URL, for messages.
    pub fn location(&self) -> String {
        match self {
            Self::Local(root) => root.display().to_string(),
            Self::Object { url, .. } => url.clone(),
        }
    }

    /// Whether this target holds any file from an earlier run (a segment or a
    /// purge obligation). The owner marker is not state. An error means the
    /// answer is unknown: callers must not treat it as "empty".
    pub async fn has_state(&self) -> Result<bool> {
        Ok(self.scan().await?.has_state)
    }

    async fn scan(&self) -> Result<Scan> {
        match self {
            Self::Local(root) => Ok(Scan {
                has_state: crate::services::storage_format::spill_root_has_state(root),
                nested_owners: Vec::new(),
            }),
            Self::Object { url, filesystem } => scan_object(url, filesystem.as_ref()).await,
        }
    }
}

/// [`Scan`] of an object-store prefix. Object-store listings are recursive and
/// hold only objects; `file://` lists one level, so it is walked like a local
/// root.
async fn scan_object(url: &str, filesystem: &dyn FileSystem) -> Result<Scan> {
    if let Some(path) = url.strip_prefix("file://") {
        let root = PathBuf::from(path);
        let mut scan = Scan::default();
        scan_local(&root, &root.join(OWNER_MARKER), &mut scan);
        return Ok(scan);
    }
    let entries = match filesystem.list(&format!("{url}/")).await {
        Ok(entries) => entries,
        Err(error) if is_absent(&error) => return Ok(Scan::default()),
        Err(error) => return Err(fs_error(&format!("listing spill prefix {url}"), error)),
    };
    let own = owner_url(url);
    let mut scan = Scan::default();
    for entry in entries.iter().filter(|e| !e.metadata.is_directory) {
        if after_scheme(&entry.url) == after_scheme(&own) {
            continue;
        }
        if entry.name == OWNER_MARKER {
            scan.nested_owners.push(entry.url.clone());
        } else {
            scan.has_state = true;
        }
    }
    Ok(scan)
}

/// Recursive [`Scan`] of a local directory. An unreadable entry counts as
/// state.
fn scan_local(dir: &Path, own_marker: &Path, scan: &mut Scan) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            scan.has_state |= error.kind() != std::io::ErrorKind::NotFound;
            return;
        }
    };
    for entry in entries {
        let Ok(entry) = entry else {
            scan.has_state = true;
            continue;
        };
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => scan_local(&path, own_marker, scan),
            Ok(_) if path == own_marker => {}
            Ok(_) if entry.file_name() == OWNER_MARKER => {
                scan.nested_owners
                    .push(format!("file://{}", path.display()));
            }
            // A marker install cut off before its rename.
            Ok(_) if is_marker_temp(&entry.file_name().to_string_lossy()) => {}
            _ => scan.has_state = true,
        }
    }
}

fn owner_url(url: &str) -> String {
    format!("{url}/{OWNER_MARKER}")
}

fn is_marker_temp(name: &str) -> bool {
    name.starts_with(OWNER_MARKER) && name.ends_with(".tmp")
}

/// A filesystem error as boot reports it: a refused credential is decisive
/// (retrying cannot fix it) and names what the server needs.
fn fs_error(context: &str, error: FilesystemError) -> anyhow::Error {
    match error {
        FilesystemError::PermissionDenied(_) | FilesystemError::Auth(_) => Refused(format!(
            "{context}: {error}; the server's credentials need GET, HEAD, PUT and LIST on the \
             spill prefix, and GET on `{OWNER_MARKER}` at each ancestor prefix below the bucket"
        ))
        .into(),
        other => anyhow!("{context}: {other}"),
    }
}

/// The owner marker at `path`: `None` when absent, the bytes otherwise (empty
/// for a write that never completed). `file://` (tests) goes through `std::fs`,
/// like [`write_owner`].
async fn read_owner(path: &str, filesystem: &dyn FileSystem) -> Result<Option<Vec<u8>>> {
    let read = match path.strip_prefix("file://") {
        Some(local) => std::fs::read(local).map_err(FilesystemError::Io),
        None => filesystem.read(path).await,
    };
    match read {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if is_absent(&error) => Ok(None),
        Err(error) => Err(fs_error(&format!("reading {path}"), error)),
    }
}

fn parse_owner(path: &str, bytes: &[u8]) -> Result<Option<OwnerMarker>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    serde_json::from_slice(bytes)
        .map(Some)
        .map_err(|e| Refused(format!("spill owner marker {path} is unreadable: {e}")).into())
}

/// Write the owner marker: create-only for a fresh prefix (two servers racing
/// for it cannot both win), else an overwrite. `file://` installs atomically
/// through `std::fs`, since the local backend writes in place (a crash would
/// leave a torn marker) and may encrypt.
async fn write_owner(
    path: &str,
    filesystem: &dyn FileSystem,
    bytes: &[u8],
    create_only: bool,
) -> Result<()> {
    let created = match path.strip_prefix("file://").map(Path::new) {
        Some(local) => {
            let dir = local.parent().unwrap_or(Path::new("/"));
            if create_only {
                crate::services::storage_format::install_marker_if_absent(dir, OWNER_MARKER, bytes)?
            } else {
                crate::services::storage_format::write_marker(dir, OWNER_MARKER, bytes)?;
                true
            }
        }
        None if create_only => match filesystem
            .write_if_absent(
                path,
                bytes,
                Some(FileOptions {
                    create_dirs: true,
                    ..Default::default()
                }),
            )
            .await
        {
            Ok(()) => true,
            Err(FilesystemError::AlreadyExists(_)) => false,
            Err(error) => return Err(fs_error(&format!("writing {path}"), error)),
        },
        None => {
            let overwrite = Some(FileOptions {
                overwrite: true,
                ..Default::default()
            });
            filesystem
                .write(path, bytes, overwrite)
                .await
                .map_err(|e| fs_error(&format!("writing {path}"), e))?;
            true
        }
    };
    if !created {
        bail!("{path} was created concurrently; re-reading");
    }
    Ok(())
}

/// `bucket/key` of a URL: listings may spell the scheme differently.
fn after_scheme(url: &str) -> &str {
    url.split_once("://").map_or(url, |(_, rest)| rest)
}

/// The one spelling of a spill URL, so the markers compare equal for every
/// alias of one prefix: lower-case scheme; `azure`/`adls`/`abfs` → `az` and
/// `gcs` → `gs` (one backend each); no empty, trailing or repeated `/`.
///
/// Refuses a bucket (or filesystem) root — a prefix shared with everything
/// else in the bucket — and `.`/`..` segments.
pub fn canonical_spill_url(raw: &str) -> Result<String> {
    let raw = raw.trim();
    let (scheme, rest) = raw.split_once("://").ok_or_else(|| {
        anyhow!(
            "{SPILL_URL_ENV}={raw}: not a URL; expected s3://bucket/prefix, az://container/prefix, \
             gs://bucket/prefix or file:///path"
        )
    })?;
    let scheme = scheme.to_ascii_lowercase();
    let authority = rest.split('/').next().unwrap_or_default();
    let scheme = match scheme.as_str() {
        "az" | "azure" | "adls" | "abfs" => "az".to_string(),
        "gs" | "gcs" => "gs".to_string(),
        _ => scheme,
    };
    let parts: Vec<&str> = rest.split('/').filter(|part| !part.is_empty()).collect();
    if parts.iter().any(|part| matches!(*part, "." | "..")) {
        bail!("{SPILL_URL_ENV}={raw}: `.` and `..` segments are not allowed");
    }
    let is_file = scheme == "file";
    if is_file && !rest.starts_with('/') {
        bail!("{SPILL_URL_ENV}={raw}: a file:// URL needs an absolute path (file:///…)");
    }
    // A bucket needs a key below it; a local path needs a directory below `/`.
    if parts.len() < if is_file { 1 } else { 2 } {
        bail!(
            "{SPILL_URL_ENV}={raw}: names a bucket root, which other data shares; give a key \
             prefix (e.g. {scheme}://{authority}/proximadb-spill)"
        );
    }
    let lead = if is_file { "/" } else { "" };
    Ok(format!("{scheme}://{lead}{}", parts.join("/")))
}

/// [`SPILL_URL_ENV`], canonical; `None` when unset or empty.
pub fn configured_spill_url() -> Result<Option<String>> {
    match std::env::var(SPILL_URL_ENV) {
        Ok(raw) if !raw.trim().is_empty() => canonical_spill_url(&raw).map(Some),
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PlacementMarker {
    /// `"local"` or the canonical object-store URL.
    placement: String,
    /// This data directory's identity, the value an owner marker must carry.
    instance: String,
    /// The epoch of this directory's last completed claim (0: none yet).
    #[serde(default)]
    epoch: u64,
    /// Nonce of a claim begun but not recorded here: a crash after the owner
    /// marker was written leaves it, so the next boot recognises the marker as
    /// its own.
    #[serde(default)]
    pending: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OwnerMarker {
    instance: String,
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    nonce: Option<String>,
    written_by: String,
}

/// A decisive refusal: retrying cannot change it.
#[derive(Debug)]
struct Refused(String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

fn refuse<T>(message: String) -> Result<T> {
    Err(Refused(message).into())
}

fn read_placement(dir: &Path) -> Result<Option<PlacementMarker>> {
    let path = dir.join(PLACEMENT_MARKER);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).with_context(|| {
            format!(
                "spill placement marker {} is unreadable; refusing to guess where the \
                 segments are",
                path.display()
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn write_placement(dir: &Path, marker: &PlacementMarker) -> Result<()> {
    crate::services::storage_format::write_marker(
        dir,
        PLACEMENT_MARKER,
        &serde_json::to_vec_pretty(marker)?,
    )
}

fn target_for(
    placement: &str,
    local_root: &Path,
    factory: &FilesystemFactory,
) -> Result<SpillTarget> {
    if placement == LOCAL {
        return Ok(SpillTarget::Local(local_root.to_path_buf()));
    }
    let filesystem = factory.get_filesystem(placement).map_err(|e| {
        anyhow!(
            "spill placement {placement}: no filesystem serves this URL ({e}); build with the \
             matching cloud feature (aws / azure / gcp)"
        )
    })?;
    Ok(SpillTarget::Object {
        url: placement.to_string(),
        filesystem,
    })
}

/// Resolve where this server's spill segments go and record it.
///
/// `dir` holds the canonical WAL (and the placement marker); `wal_truncated`
/// is whether the WAL no longer starts at its first entry. With the spill off
/// nothing is claimed or recorded: the target is the recorded placement, for
/// the caller's check that no rows live only there.
pub async fn resolve_spill_target(
    dir: &Path,
    factory: &FilesystemFactory,
    wal_truncated: bool,
) -> Result<SpillTarget> {
    let url = configured_spill_url()?;
    let settings = Settings {
        url: url.as_deref(),
        spill_on: crate::services::record_spill::configured_flush_threshold().is_some(),
        adopt: std::env::var(SPILL_ADOPT_ENV).is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        }),
    };
    resolve_spill_target_for(dir, factory, wal_truncated, settings).await
}

/// What boot asked for.
#[derive(Clone, Copy, Default)]
struct Settings<'a> {
    /// Canonical [`SPILL_URL_ENV`]; `None` is local.
    url: Option<&'a str>,
    /// Whether the spill is on.
    spill_on: bool,
    /// [`SPILL_ADOPT_ENV`]: a placement change may adopt the segments already
    /// at the new placement (the operator moved them there).
    adopt: bool,
}

async fn resolve_spill_target_for(
    dir: &Path,
    factory: &FilesystemFactory,
    wal_truncated: bool,
    settings: Settings<'_>,
) -> Result<SpillTarget> {
    let local_root = dir.join("record-spill");
    let recorded = read_placement(dir)?;
    if !settings.spill_on {
        // With the WAL intact replay rebuilds every row, so where earlier
        // segments sit does not matter.
        return match (&recorded, wal_truncated) {
            (Some(marker), true) => target_for(&marker.placement, &local_root, factory),
            _ => Ok(SpillTarget::Local(local_root)),
        };
    }
    let configured = settings.url.unwrap_or(LOCAL);
    // Persist the identity before anything can be claimed with it. A
    // directory from before the marker spilled locally, if at all.
    let mut state = match recorded {
        Some(marker) => marker,
        None if crate::services::storage_format::ensure_data_format(dir)?
            >= crate::services::storage_format::OBJECT_SPILL_FORMAT =>
        {
            return refuse(format!(
                "{} is missing but this data directory once placed spill segments on an object \
                 store (data format 3); without the marker their location is unknown. Restore it",
                dir.join(PLACEMENT_MARKER).display()
            ));
        }
        None => {
            let marker = PlacementMarker {
                placement: LOCAL.to_string(),
                instance: uuid::Uuid::new_v4().to_string(),
                epoch: 0,
                pending: None,
            };
            write_placement(dir, &marker)?;
            marker
        }
    };
    let before = state.clone();
    let target = target_for(configured, &local_root, factory)?;
    if settings.adopt && state.placement == configured {
        tracing::warn!(
            "{SPILL_ADOPT_ENV} is set but the spill placement is unchanged; unset it (it only \
             matters for the boot that changes the placement)"
        );
    }
    let step = Step {
        dir,
        local_root: &local_root,
        factory,
        target: &target,
        changing: state.placement != configured,
        wal_truncated,
        adopt: settings.adopt,
    };
    let mut attempt = 1;
    loop {
        match step.place(&mut state).await {
            Ok(()) => break,
            Err(error) if error.is::<Refused>() || attempt == CLAIM_ATTEMPTS => return Err(error),
            Err(error) => {
                tracing::warn!(
                    target = %target.location(),
                    attempt,
                    "resolving the spill placement failed; retrying: {error:#}"
                );
                tokio::time::sleep(Duration::from_secs(1 << (attempt - 1))).await;
                attempt += 1;
            }
        }
    }
    if matches!(target, SpillTarget::Object { .. }) {
        // Once the checks passed and before any segment can land there: a
        // binary that only knows local segments must refuse this directory,
        // not serve it without them.
        crate::services::storage_format::raise_data_format(
            dir,
            crate::services::storage_format::OBJECT_SPILL_FORMAT,
        )?;
    }
    state.placement = configured.to_string();
    if state != before {
        write_placement(dir, &state)?;
    }
    if step.changing {
        tracing::info!(
            dir = %dir.display(),
            from = %before.placement,
            to = %target.placement(),
            adopted = settings.adopt,
            "spill placement recorded"
        );
    }
    Ok(target)
}

/// One attempt at resolving a placement: every check and the claim, so a
/// transient error anywhere is retried as a whole.
struct Step<'a> {
    dir: &'a Path,
    local_root: &'a Path,
    factory: &'a FilesystemFactory,
    target: &'a SpillTarget,
    changing: bool,
    wal_truncated: bool,
    adopt: bool,
}

impl Step<'_> {
    async fn place(&self, state: &mut PlacementMarker) -> Result<()> {
        if self.changing {
            let configured = self.target.placement();
            let old = target_for(&state.placement, self.local_root, self.factory)?;
            let old_has_state = old.scan().await?.has_state;
            if old_has_state && self.adopt {
                return refuse(format!(
                    "{SPILL_ADOPT_ENV} is set but {} still holds spill segments: adopting {configured}'s \
                     would leave two copies of rows; move the segments, do not copy them",
                    old.location()
                ));
            }
            if old_has_state && self.wal_truncated {
                let migrate = match self.target {
                    SpillTarget::Object { .. } => format!(
                        ", or move (not copy) the segments into {configured} — a prefix this \
                         directory never used — and boot once with {SPILL_ADOPT_ENV}=1"
                    ),
                    SpillTarget::Local(_) => String::new(),
                };
                return refuse(format!(
                    "spill segments live at {} but {SPILL_URL_ENV} now selects {configured}, and \
                     the WAL below those segments was truncated, so their rows are in no \
                     retained WAL entry; restore the previous setting{migrate}",
                    old.location()
                ));
            }
        }
        match self.target {
            SpillTarget::Object { url, filesystem } => {
                self.claim(state, url, filesystem.as_ref()).await
            }
            SpillTarget::Local(root) => {
                if self.changing && self.target.scan().await?.has_state {
                    // A local root holds no owner marker, so whether it is a
                    // root this directory left (with older segments) cannot
                    // be told: adoption is for object prefixes only.
                    return refuse(stale_segments(&root.display().to_string(), false));
                }
                Ok(())
            }
        }
    }

    /// Claim `url` for this directory under the next epoch. On success `state`
    /// carries that epoch; the caller records it.
    async fn claim(
        &self,
        state: &mut PlacementMarker,
        url: &str,
        filesystem: &dyn FileSystem,
    ) -> Result<()> {
        let (changing, wal_truncated) = (self.changing, self.wal_truncated);
        // A full listing only when this directory first takes the prefix: a
        // server nesting a prefix in ours later finds our owner marker among
        // its ancestors, and one wrapping ours later lists ours.
        let scan = if changing {
            scan_object(url, filesystem).await?
        } else {
            Scan::default()
        };
        // Nested prefixes, in either direction. This directory's own markers
        // (a placement it left) are not another server's.
        let key = after_scheme(url);
        let scheme = &url[..url.len() - key.len()];
        // Ancestors below the bucket (or filesystem) root: `parts[0]` is the
        // bucket, or the empty segment before a `file://` path's leading `/`.
        let parts: Vec<&str> = key.split('/').collect();
        let ancestors =
            (2..parts.len()).map(|end| owner_url(&format!("{scheme}{}", parts[..end].join("/"))));
        for marker in scan.nested_owners.iter().cloned().chain(ancestors) {
            let Some(bytes) = read_owner(&marker, filesystem).await? else {
                continue;
            };
            let other = parse_owner(&marker, &bytes)?;
            if other.is_none_or(|other| other.instance != state.instance) {
                return refuse(format!(
                    "spill prefix {url} and another server's prefix (owner marker {marker}) \
                     nest; prefixes must not nest"
                ));
            }
        }

        let path = owner_url(url);
        // `present`: a file is there to overwrite, even an empty one.
        let (owner, present) = match read_owner(&path, filesystem).await? {
            Some(bytes) => (parse_owner(&path, &bytes)?, true),
            None => (None, false),
        };
        // Ownership first: another server's segments are never "stale".
        if let Some(owner) = &owner
            && owner.instance != state.instance
        {
            return refuse(format!(
                "spill prefix {url} belongs to another data directory (instance {}, this one is \
                 {}); give each server its own {SPILL_URL_ENV}",
                owner.instance, state.instance
            ));
        }
        if changing && scan.has_state {
            if !self.adopt {
                return refuse(stale_segments(url, true));
            }
            // Our own marker: this directory used the prefix before, so its
            // older segments would mix with the moved ones — unless it is
            // this very claim, cut off (a retry, or the boot after a crash).
            let resuming = owner
                .as_ref()
                .is_some_and(|o| o.nonce.is_some() && o.nonce == state.pending);
            if present && !resuming {
                return refuse(format!(
                    "{SPILL_ADOPT_ENV}: {url} is a prefix this data directory used before, so \
                     its earlier segments would mix with the moved ones; move the segments into \
                     an empty prefix this directory has never used"
                ));
            }
        }
        match &owner {
            None if !changing && state.epoch > 0 && wal_truncated => {
                return refuse(format!(
                    "spill prefix {url} lost its owner marker (the bucket was emptied, or the URL \
                     now reaches a different store), and the WAL was truncated, so rows that \
                     only its segments held may be gone; restore the prefix"
                ));
            }
            None => {}
            Some(owner) => {
                if owner.nonce.is_some() && owner.nonce == state.pending {
                    // Our own claim, cut off before it was recorded here:
                    // record it, then claim afresh under a new nonce (a copy
                    // taken meanwhile carries the old one).
                    state.epoch = owner.epoch;
                    state.pending = None;
                    state.placement = url.to_string();
                    write_placement(self.dir, state)?;
                }
                let current = owner.epoch == state.epoch
                    // Used before, since left for another placement.
                    || (changing && owner.epoch <= state.epoch);
                if !current && owner.epoch < state.epoch && wal_truncated {
                    return refuse(format!(
                        "spill prefix {url} is at claim epoch {} but this data directory reached \
                         {}: the prefix was rolled back, and the WAL was truncated, so newer \
                         segments may be gone; restore the prefix",
                        owner.epoch, state.epoch
                    ));
                }
                if !current && owner.epoch > state.epoch {
                    return refuse(format!(
                        "spill prefix {url} was claimed at epoch {} by a copy of this data \
                         directory (this copy is at {}); this copy's view is stale and must not \
                         serve or purge it",
                        owner.epoch, state.epoch
                    ));
                }
            }
        }

        let epoch = state.epoch.max(owner.as_ref().map_or(0, |o| o.epoch)) + 1;
        // The nonce is durable here before the owner marker can carry it, so
        // any crash or retry after this point recognises its own claim.
        let nonce = uuid::Uuid::new_v4().to_string();
        let mut pending = state.clone();
        pending.pending = Some(nonce.clone());
        write_placement(self.dir, &pending)?;
        *state = pending;
        let bytes = serde_json::to_vec_pretty(&OwnerMarker {
            instance: state.instance.clone(),
            epoch,
            nonce: Some(nonce.clone()),
            written_by: crate::version::PROXIMADB_VERSION.to_string(),
        })?;
        write_owner(&path, filesystem, &bytes, !present).await?;
        let check = read_owner(&path, filesystem)
            .await?
            .ok_or_else(|| anyhow!("{path} vanished after it was written"))?;
        let check = parse_owner(&path, &check)?
            .ok_or_else(|| anyhow!("{path} is empty after it was written"))?;
        if check.nonce.as_deref() != Some(nonce.as_str()) {
            return refuse(format!(
                "spill prefix {url} was claimed by another server (instance {}) while this one \
                 claimed it",
                check.instance
            ));
        }
        state.epoch = epoch;
        state.pending = None;
        Ok(())
    }
}

fn stale_segments(location: &str, adoptable: bool) -> String {
    let adopt = if adoptable {
        format!(
            ", or — if they are this directory's own, moved into a prefix it never used — boot \
             once with {SPILL_ADOPT_ENV}=1"
        )
    } else {
        String::new()
    };
    format!(
        "{location} already holds spill segments, from an earlier placement or a copy; adopting \
         it would serve them as current rows. Remove them or choose an empty placement{adopt}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn factory() -> FilesystemFactory {
        FilesystemFactory::create_default().await.expect("factory")
    }

    /// `file://` exercises the object-store arm without an emulator: the same
    /// URL-addressed paths, owner claim and fencing as `s3://` / `az://`.
    fn file_url(path: &Path) -> String {
        format!("file://{}", path.display())
    }

    async fn resolve(
        dir: &Path,
        factory: &FilesystemFactory,
        wal_truncated: bool,
        url: Option<&str>,
    ) -> Result<SpillTarget> {
        let settings = Settings {
            url,
            spill_on: true,
            adopt: false,
        };
        resolve_spill_target_for(dir, factory, wal_truncated, settings).await
    }

    async fn refusal(
        dir: &Path,
        factory: &FilesystemFactory,
        wal_truncated: bool,
        url: Option<&str>,
    ) -> String {
        match resolve(dir, factory, wal_truncated, url).await {
            Err(error) => {
                assert!(error.is::<Refused>(), "decisive: {error:#}");
                format!("{error:#}")
            }
            Ok(_) => panic!("expected a refusal"),
        }
    }

    fn owner(url: &str) -> OwnerMarker {
        let path = owner_url(url);
        let bytes = std::fs::read(path.strip_prefix("file://").expect("file url")).expect("owner");
        serde_json::from_slice(&bytes).expect("owner json")
    }

    fn segment(root: &Path, name: &str) {
        let dir = root.join("t").join("c");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join(name), b"x").expect("segment");
    }

    fn state(dir: &Path) -> PlacementMarker {
        read_placement(dir).expect("read").expect("recorded")
    }

    #[test]
    fn spill_urls_have_one_spelling_and_never_name_a_bucket_root() {
        for (raw, want) in [
            ("s3://b/spill", "s3://b/spill"),
            (" S3://b//spill/ ", "s3://b/spill"),
            ("azure://c/p/q", "az://c/p/q"),
            ("adls://c/p", "az://c/p"),
            ("abfs://c/p", "az://c/p"),
            ("gcs://b/p", "gs://b/p"),
            ("file:///var//spill/", "file:///var/spill"),
        ] {
            assert_eq!(canonical_spill_url(raw).expect(raw), want, "{raw}");
        }
        for raw in [
            "s3://b",
            "s3://b/",
            "az://c//",
            "file:///",
            "file://relative/path",
            "s3://b/../other",
            "s3://b/./p",
            "/no/scheme",
        ] {
            assert!(canonical_spill_url(raw).is_err(), "{raw} must be refused");
        }
    }

    #[tokio::test]
    async fn placement_is_recorded_claimed_and_advanced_each_boot() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;

        // Fresh directory, no URL: local, and the identity is recorded.
        let target = resolve(&dir, &factory, false, None).await.expect("local");
        assert!(
            matches!(target, SpillTarget::Local(ref root) if root == &dir.join("record-spill"))
        );
        let local = state(&dir);
        assert_eq!((local.placement.as_str(), local.epoch), (LOCAL, 0));

        // To the URL with the WAL intact, over local segments: allowed (replay
        // rebuilds every row); the prefix is claimed at epoch 1.
        segment(&dir.join("record-spill"), "spill-0.parquet");
        let target = resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("object");
        assert_eq!(target.location(), url);
        assert!(
            !target.has_state().await.expect("scan"),
            "the owner marker is not state"
        );
        let claimed = state(&dir);
        assert_eq!(claimed.instance, local.instance, "identity is kept");
        assert_eq!((claimed.epoch, claimed.pending.as_deref()), (1, None));
        assert_eq!(owner(&url).epoch, 1);
        assert_eq!(
            crate::services::storage_format::ensure_data_format(&dir).expect("format"),
            crate::services::storage_format::OBJECT_SPILL_FORMAT,
            "an object placement raises the data format, so older binaries refuse it"
        );

        // Every boot advances both markers.
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("same placement, truncated WAL");
        assert_eq!((state(&dir).epoch, owner(&url).epoch), (2, 2));

        // Back to local after truncation while the URL holds segments: refused,
        // and nothing is recorded.
        segment(&bucket.path().join("spill"), "spill-1.parquet");
        let err = refusal(&dir, &factory, true, None).await;
        assert!(err.contains("no retained WAL entry"), "{err}");
        assert_eq!(state(&dir).placement, url, "a refusal records nothing");

        // A lost placement marker is not a fresh directory once segments went
        // to an object store: their location would be unknown.
        std::fs::remove_file(dir.join(PLACEMENT_MARKER)).expect("lose the marker");
        let err = refusal(&dir, &factory, true, None).await;
        assert!(err.contains("is missing"), "{err}");
    }

    #[tokio::test]
    async fn a_prefix_with_stale_segments_is_never_adopted() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let a = file_url(&bucket.path().join("a"));
        let b = file_url(&bucket.path().join("b"));
        let factory = factory().await;
        resolve(&dir, &factory, false, Some(&a)).await.expect("a");
        // B holds segments (an earlier use, or another server's): switching to
        // it would serve them as current rows, even with the WAL intact.
        segment(&bucket.path().join("b"), "spill-9.parquet");
        let err = refusal(&dir, &factory, false, Some(&b)).await;
        assert!(err.contains("already holds spill segments"), "{err}");
        // Local, likewise.
        segment(&dir.join("record-spill"), "spill-0.parquet");
        let err = refusal(&dir, &factory, false, None).await;
        assert!(err.contains("already holds spill segments"), "{err}");
        assert_eq!(state(&dir).placement, a);

        // Empty prefixes are adopted, and one used before (its owner epoch is
        // behind this directory's) is re-adopted.
        std::fs::remove_dir_all(bucket.path().join("b")).expect("clear b");
        resolve(&dir, &factory, false, Some(&b)).await.expect("b");
        resolve(&dir, &factory, false, Some(&a))
            .await
            .expect("back to a");
        assert_eq!(state(&dir).placement, a);
    }

    #[tokio::test]
    async fn moved_segments_are_adopted_only_on_request() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        resolve(&dir, &factory, false, None).await.expect("local");
        segment(&dir.join("record-spill"), "spill-0.parquet");
        let adopting = Settings {
            url: Some(&url),
            spill_on: true,
            adopt: true,
        };

        // Truncated with local segments: moving is refused, with the way out.
        let err = refusal(&dir, &factory, true, Some(&url)).await;
        assert!(err.contains(SPILL_ADOPT_ENV), "{err}");
        // Copied rather than moved: adopting would leave two copies.
        let err = resolve_spill_target_for(&dir, &factory, true, adopting)
            .await
            .err()
            .expect("the old placement still holds segments");
        assert!(format!("{err:#}").contains("still holds"), "{err:#}");

        // Moved: refused as stale without the opt-in, adopted with it.
        std::fs::create_dir_all(bucket.path().join("spill")).expect("mkdir");
        std::fs::rename(
            dir.join("record-spill").join("t"),
            bucket.path().join("spill").join("t"),
        )
        .expect("move");
        let err = refusal(&dir, &factory, true, Some(&url)).await;
        assert!(err.contains("already holds spill segments"), "{err}");
        resolve_spill_target_for(&dir, &factory, true, adopting)
            .await
            .expect("adopted");
        assert_eq!((state(&dir).placement, state(&dir).epoch), (url.clone(), 1));
        // Later boots need no opt-in.
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("same placement");

        // Never into a prefix this directory used before: its older segments
        // would mix with the moved ones.
        let other = file_url(&bucket.path().join("other"));
        resolve(&dir, &factory, false, Some(&other))
            .await
            .expect("to an empty prefix, WAL intact");
        segment(&bucket.path().join("other"), "spill-5.parquet");
        let back = Settings {
            url: Some(&url),
            spill_on: true,
            adopt: true,
        };
        std::fs::remove_dir_all(bucket.path().join("other").join("t")).expect("moved away");
        let err = resolve_spill_target_for(&dir, &factory, true, back)
            .await
            .err()
            .expect("a used prefix");
        assert!(format!("{err:#}").contains("used before"), "{err:#}");
    }

    #[tokio::test]
    async fn other_directories_copies_and_nested_prefixes_are_fenced() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("claim");

        // Another data directory may not take this prefix — told it is owned,
        // not that its segments are stale (they may be the only copy).
        segment(&bucket.path().join("spill"), "spill-1.parquet");
        let other = tempfile::tempdir().expect("tempdir");
        let err = refusal(&other.path().join("pgwire"), &factory, false, Some(&url)).await;
        assert!(err.contains("belongs to another data directory"), "{err}");

        // A copy of this data directory (same instance) is fenced once the
        // original boots again.
        let copy = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(copy.path().join("pgwire")).expect("mkdir");
        std::fs::copy(
            dir.join(PLACEMENT_MARKER),
            copy.path().join("pgwire").join(PLACEMENT_MARKER),
        )
        .expect("copy marker");
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("original");
        let err = refusal(&copy.path().join("pgwire"), &factory, true, Some(&url)).await;
        assert!(err.contains("copy of this data directory"), "{err}");
        // Whichever boots first wins: a copy taken now and booted first fences
        // the original.
        std::fs::copy(
            dir.join(PLACEMENT_MARKER),
            copy.path().join("pgwire").join(PLACEMENT_MARKER),
        )
        .expect("copy marker");
        resolve(&copy.path().join("pgwire"), &factory, true, Some(&url))
            .await
            .expect("the copy boots first");
        let err = refusal(&dir, &factory, true, Some(&url)).await;
        assert!(err.contains("copy of this data directory"), "{err}");

        // Prefixes may not nest, in either direction.
        let inner = format!("{url}/inner");
        let err = refusal(&other.path().join("pgwire"), &factory, false, Some(&inner)).await;
        assert!(err.contains("must not nest"), "{err}");
        let outer = file_url(bucket.path());
        let err = refusal(&other.path().join("pgwire"), &factory, false, Some(&outer)).await;
        assert!(err.contains("must not nest"), "{err}");

        // This directory's own markers are not another server's: moving into
        // a sub-prefix of the one it used (WAL intact) is allowed.
        resolve(&dir, &factory, false, Some(&format!("{url}/v2")))
            .await
            .expect("own ancestor marker");
    }

    #[tokio::test]
    async fn a_lost_owner_marker_refuses_only_after_truncation() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("claim");

        // The prefix was emptied (or the URL now reaches another store).
        std::fs::remove_dir_all(bucket.path().join("spill")).expect("wipe");
        let err = refusal(&dir, &factory, true, Some(&url)).await;
        assert!(err.contains("lost its owner marker"), "{err}");
        // With the WAL intact, replay rebuilds every row: re-claimed.
        resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("re-claim over an intact WAL");
        assert_eq!(owner(&url).epoch, 2);

        // Rolled back to an older epoch: refused after truncation.
        let path = PathBuf::from(owner_url(&url).strip_prefix("file://").expect("file"));
        let mut old = owner(&url);
        old.epoch = 1;
        old.nonce = None;
        std::fs::write(&path, serde_json::to_vec(&old).expect("json")).expect("roll back");
        let err = refusal(&dir, &factory, true, Some(&url)).await;
        assert!(err.contains("rolled back"), "{err}");
    }

    #[tokio::test]
    async fn a_crash_inside_the_claim_recovers_on_the_next_boot() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("claim");
        let path = PathBuf::from(owner_url(&url).strip_prefix("file://").expect("file"));

        // Crash after the owner write, before the local commit: the owner is
        // one epoch ahead, carrying the nonce the local marker left pending.
        let mut local = state(&dir);
        local.pending = Some("n-1".to_string());
        write_placement(&dir, &local).expect("pending");
        let mut ahead = owner(&url);
        ahead.epoch = local.epoch + 1;
        ahead.nonce = Some("n-1".to_string());
        std::fs::write(&path, serde_json::to_vec(&ahead).expect("json")).expect("ahead");
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("our own interrupted claim");
        let recovered = state(&dir);
        assert_eq!((recovered.epoch, recovered.pending), (3, None));

        // Crash before the owner write: pending set, owner unchanged.
        let mut local = state(&dir);
        local.pending = Some("n-2".to_string());
        write_placement(&dir, &local).expect("pending");
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("claim never reached the store");

        // A first claim cut off mid-write left an empty owner marker.
        let fresh = tempfile::tempdir().expect("tempdir");
        let fresh_url = file_url(&fresh.path().join("spill"));
        std::fs::create_dir_all(fresh.path().join("spill")).expect("mkdir");
        std::fs::write(fresh.path().join("spill").join(OWNER_MARKER), b"").expect("empty");
        let other = tempfile::tempdir().expect("tempdir");
        resolve(
            &other.path().join("pgwire"),
            &factory,
            false,
            Some(&fresh_url),
        )
        .await
        .expect("an empty owner marker is no owner");
    }

    #[tokio::test]
    async fn with_the_spill_off_nothing_is_claimed_or_recorded() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        let off = |url| Settings {
            url,
            spill_on: false,
            adopt: false,
        };
        let target = resolve_spill_target_for(&dir, &factory, true, off(Some(&url)))
            .await
            .expect("off");
        assert!(matches!(target, SpillTarget::Local(_)));
        assert!(read_placement(&dir).expect("read").is_none());
        assert!(!bucket.path().join("spill").exists());

        // After truncation the recorded placement is what the caller checks.
        resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("claim");
        let target = resolve_spill_target_for(&dir, &factory, true, off(None))
            .await
            .expect("off");
        assert_eq!(target.location(), url);
    }

    /// The claim against a real object store (run by the `--restart` scope of
    /// `scripts/run_cloud_emulator_tests.sh`, Azurite).
    #[tokio::test]
    #[ignore = "requires an object-store emulator (PROXIMADB_OBJECT_STORE_URL)"]
    async fn placement_is_claimed_and_fenced_on_object_store() {
        let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")
            .expect("set PROXIMADB_OBJECT_STORE_URL to an emulator prefix");
        let run = uuid::Uuid::new_v4().simple().to_string();
        let url = canonical_spill_url(&format!("{base}/spill-placement/{run}")).expect("url");
        let factory = factory().await;
        let data = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let target = resolve(&dir, &factory, false, Some(&url))
            .await
            .expect("claim");
        assert!(
            !target.has_state().await.expect("scan"),
            "the owner marker is not state"
        );
        resolve(&dir, &factory, true, Some(&url))
            .await
            .expect("again");
        assert_eq!(state(&dir).epoch, 2);
        let SpillTarget::Object { filesystem, .. } = &target else {
            panic!("expected an object target");
        };
        filesystem
            .write(
                &format!("{url}/t/c/spill-1.parquet"),
                b"x",
                Some(FileOptions {
                    create_dirs: true,
                    ..Default::default()
                }),
            )
            .await
            .expect("segment");
        assert!(target.has_state().await.expect("scan"));
        let other = tempfile::tempdir().expect("tempdir");
        let err = refusal(&other.path().join("pgwire"), &factory, false, Some(&url)).await;
        assert!(err.contains("belongs to another data directory"), "{err}");
        let err = refusal(
            &other.path().join("pgwire"),
            &factory,
            false,
            Some(&format!("{url}/t")),
        )
        .await;
        assert!(err.contains("must not nest"), "{err}");
    }
}
