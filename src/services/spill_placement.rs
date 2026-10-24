//! ADR-097 P2 / TD-SEAM-3: where the relational spill writes its segments.
//!
//! By default segments sit on local disk beside the canonical WAL
//! (`{data_dir}/pgwire/record-spill`). With [`SPILL_URL_ENV`] set they go to
//! an object store instead, at `{url}/{tenant}/{collection}` — a prefix derived
//! from exactly what WAL replay carries, so recovery needs no catalog lookup.
//!
//! Two guards make the derived prefix safe:
//!
//! * **Placement marker** (`{data_dir}/pgwire/SPILL_PLACEMENT`): records where
//!   this directory's segments live. Once the WAL below them was truncated,
//!   their rows are in no retained WAL entry, so moving the placement would
//!   serve a silent subset — boot refuses instead.
//! * **Owner marker** (`{url}/_proximadb_spill_owner.json`): claims the prefix
//!   for this data directory, conditionally (create-only). Two servers given
//!   one URL would otherwise read — and purge — each other's segments.
//!
//! The WAL itself stays local: an object-store placement offloads disk and
//! makes segments reachable to readers, but durability still rests on the
//! local WAL until the WAL moves too.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::storage::persistence::filesystem::{
    FileOptions, FileSystem, FilesystemError, FilesystemFactory,
};

/// Object-store URL (`s3://…`, `az://…`, `gs://…`) the relational spill writes
/// its segments under. Unset or empty: local, beside the WAL.
pub const SPILL_URL_ENV: &str = "PROXIMADB_RELATIONAL_SPILL_URL";

/// Placement marker, beside the canonical WAL.
pub const PLACEMENT_MARKER: &str = "SPILL_PLACEMENT";

/// Owner marker, at the root of an object-store placement. Not a segment, so
/// discovery (which lists partition prefixes below it) never sees it.
pub const OWNER_MARKER: &str = "_proximadb_spill_owner.json";

/// The placement value recorded for local segments.
const LOCAL: &str = "local";

/// Where a server's spill segments go.
pub enum SpillTarget {
    /// A local directory, beside the canonical WAL.
    Local(PathBuf),
    /// An object-store prefix and the filesystem that serves it.
    Object {
        url: String,
        filesystem: Arc<dyn FileSystem>,
    },
}

impl SpillTarget {
    /// How the placement marker names this target.
    fn placement(&self) -> &str {
        match self {
            Self::Local(_) => LOCAL,
            Self::Object { url, .. } => url,
        }
    }

    /// Whether this target holds any file from an earlier run (a segment or a
    /// purge obligation). The owner marker is not state. Unsure counts as
    /// state: when unsure, keep the data reachable.
    pub async fn has_state(&self) -> bool {
        match self {
            Self::Local(root) => crate::services::storage_format::spill_root_has_state(root),
            Self::Object { url, filesystem } => match filesystem.list(&format!("{url}/")).await {
                Ok(entries) => entries.iter().any(|entry| {
                    let name = entry.name.rsplit('/').next().unwrap_or(&entry.name);
                    name != OWNER_MARKER
                }),
                Err(FilesystemError::NotFound(_)) => false,
                Err(_) => true,
            },
        }
    }
}

/// [`SPILL_URL_ENV`], trimmed of whitespace and trailing `/`; `None` when unset
/// or empty.
pub fn configured_spill_url() -> Option<String> {
    let raw = std::env::var(SPILL_URL_ENV).ok()?;
    let url = raw.trim().trim_end_matches('/');
    (!url.is_empty()).then(|| url.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PlacementMarker {
    /// `"local"` or the object-store URL.
    placement: String,
    /// This data directory's identity, the value an owner marker must carry.
    instance: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OwnerMarker {
    instance: String,
    written_by: String,
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
            "{SPILL_URL_ENV}={placement}: no filesystem serves this URL ({e}); build with \
             the matching cloud feature (aws / azure / gcp)"
        )
    })?;
    Ok(SpillTarget::Object {
        url: placement.to_string(),
        filesystem,
    })
}

/// Claim `url` for `instance`: create the owner marker if absent (create-only,
/// so two servers racing for a fresh prefix cannot both win), then require it
/// to name this instance.
async fn claim_owner(url: &str, filesystem: &dyn FileSystem, instance: &str) -> Result<()> {
    let path = format!("{url}/{OWNER_MARKER}");
    let bytes = serde_json::to_vec_pretty(&OwnerMarker {
        instance: instance.to_string(),
        written_by: crate::version::PROXIMADB_VERSION.to_string(),
    })?;
    match filesystem
        .write_if_absent(
            &path,
            &bytes,
            Some(FileOptions {
                create_dirs: true,
                ..Default::default()
            }),
        )
        .await
    {
        Ok(()) => return Ok(()),
        Err(FilesystemError::AlreadyExists(_)) => {}
        Err(error) => bail!("claiming spill prefix {url}: writing {path} failed: {error}"),
    }
    let existing = filesystem
        .read(&path)
        .await
        .map_err(|e| anyhow!("claiming spill prefix {url}: reading {path} failed: {e}"))?;
    let owner: OwnerMarker = serde_json::from_slice(&existing)
        .with_context(|| format!("spill owner marker {path} is unreadable"))?;
    if owner.instance != instance {
        bail!(
            "spill prefix {url} belongs to another data directory (instance {}, this one is \
             {instance}); give each server its own {SPILL_URL_ENV}",
            owner.instance
        );
    }
    Ok(())
}

/// Resolve where this server's spill segments go and record it.
///
/// `dir` holds the canonical WAL (and the markers); `wal_truncated` is whether
/// the WAL no longer starts at its first entry. A placement change refuses
/// when the WAL was truncated and the previous placement holds segments: their
/// rows are in no retained WAL entry. With the WAL intact, replay rebuilds
/// every row and the change is safe (the old segments are left in place).
pub async fn resolve_spill_target(
    dir: &Path,
    factory: &FilesystemFactory,
    wal_truncated: bool,
) -> Result<SpillTarget> {
    let local_root = dir.join("record-spill");
    let configured = configured_spill_url().unwrap_or_else(|| LOCAL.to_string());
    let marker = read_placement(dir)?;
    // A directory from before the marker spilled locally, if at all.
    let previous = marker
        .as_ref()
        .map_or(LOCAL, |marker| marker.placement.as_str());
    if previous != configured
        && wal_truncated
        && target_for(previous, &local_root, factory)?
            .has_state()
            .await
    {
        bail!(
            "spill segments live at {previous} but {SPILL_URL_ENV} now selects {configured}, and \
             the WAL below those segments was truncated, so their rows are in no retained WAL \
             entry; restore the previous setting (or move the segments there)"
        );
    }
    let instance = marker
        .as_ref()
        .map_or_else(|| uuid::Uuid::new_v4().to_string(), |m| m.instance.clone());
    let target = target_for(&configured, &local_root, factory)?;
    if let SpillTarget::Object { url, filesystem } = &target {
        claim_owner(url, filesystem.as_ref(), &instance).await?;
    }
    let recorded = PlacementMarker {
        placement: configured,
        instance,
    };
    if marker.as_ref() != Some(&recorded) {
        crate::services::storage_format::write_marker(
            dir,
            PLACEMENT_MARKER,
            &serde_json::to_vec_pretty(&recorded)?,
        )?;
        tracing::info!(
            dir = %dir.display(),
            from = %previous,
            to = %target.placement(),
            "spill placement recorded"
        );
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn factory() -> FilesystemFactory {
        FilesystemFactory::create_default().await.expect("factory")
    }

    /// `file://` exercises the object-store arm without an emulator: same
    /// URL-addressed paths, owner claim and listing as `s3://` / `az://`.
    fn file_url(path: &Path) -> String {
        format!("file://{}", path.display())
    }

    // One test, so the process-wide env var is never shared between tests.
    #[tokio::test]
    async fn placement_is_recorded_owned_and_guarded() {
        let data = tempfile::tempdir().expect("tempdir");
        let bucket = tempfile::tempdir().expect("tempdir");
        let dir = data.path().join("pgwire");
        let url = file_url(&bucket.path().join("spill"));
        let factory = factory().await;
        let set_url = |value: Option<&str>| match value {
            // SAFETY: only this test reads the variable.
            Some(value) => unsafe { std::env::set_var(SPILL_URL_ENV, value) },
            None => unsafe { std::env::remove_var(SPILL_URL_ENV) },
        };

        // Fresh directory, no URL: local, recorded.
        set_url(None);
        let target = resolve_spill_target(&dir, &factory, false)
            .await
            .expect("local");
        assert!(
            matches!(target, SpillTarget::Local(ref root) if root == &dir.join("record-spill"))
        );
        let local = read_placement(&dir).expect("read").expect("recorded");
        assert_eq!(local.placement, LOCAL);

        // Switching to the URL with the WAL intact is allowed even with local
        // segments (replay rebuilds every row); the prefix is claimed.
        std::fs::create_dir_all(dir.join("record-spill/t/c")).expect("mkdir");
        std::fs::write(dir.join("record-spill/t/c/spill-0.parquet"), b"x").expect("write");
        set_url(Some(&format!("{url}/")));
        let target = resolve_spill_target(&dir, &factory, false)
            .await
            .expect("object");
        let SpillTarget::Object {
            url: got,
            filesystem,
        } = &target
        else {
            panic!("expected an object target");
        };
        assert_eq!(got, &url, "trailing '/' trimmed");
        let owner: OwnerMarker = serde_json::from_slice(
            &filesystem
                .read(&format!("{url}/{OWNER_MARKER}"))
                .await
                .expect("owner marker"),
        )
        .expect("owner json");
        assert_eq!(
            owner.instance, local.instance,
            "the directory keeps its identity"
        );
        assert!(!target.has_state().await, "the owner marker is not state");
        let recorded = read_placement(&dir).expect("read").expect("recorded");
        assert_eq!(recorded.placement, url);

        // Re-resolving is idempotent.
        resolve_spill_target(&dir, &factory, true)
            .await
            .expect("same placement, truncated WAL");

        // Back to local after truncation, while the URL holds segments: refused.
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
        set_url(None);
        let err = match resolve_spill_target(&dir, &factory, true).await {
            Err(err) => err,
            Ok(_) => panic!("a placement change over a truncated WAL must refuse"),
        };
        assert!(
            format!("{err:#}").contains("no retained WAL entry"),
            "{err:#}"
        );
        assert_eq!(
            read_placement(&dir).expect("read").expect("kept").placement,
            url,
            "a refusal records nothing"
        );

        // Another data directory may not take this prefix.
        let other = tempfile::tempdir().expect("tempdir");
        set_url(Some(&url));
        let err = match resolve_spill_target(&other.path().join("pgwire"), &factory, false).await {
            Err(err) => err,
            Ok(_) => panic!("a second data directory must not share the prefix"),
        };
        assert!(
            format!("{err:#}").contains("belongs to another data directory"),
            "{err:#}"
        );
        set_url(None);
    }
}
