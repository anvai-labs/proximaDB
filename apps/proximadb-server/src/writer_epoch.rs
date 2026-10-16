//! Writer-epoch fencing for a reattachable durable volume (ADR-095 D1/D2,
//! TD-PGOLTP-2).
//!
//! At boot, before any storage is opened, a server configured with
//! `[oltp.durable_volume]`:
//!
//! 1. refuses to start unless the volume root already exists (it is never
//!    created: an unmounted mount point must not become an empty database) and
//!    every durable local root lies on it;
//! 2. binds the volume's identity to the epoch log: epoch 1 records the id in
//!    `{volume}/_control/volume.id`; every later claimant must present the same
//!    id, so a wrong or unmounted volume can never take over;
//! 3. wins the next writer epoch from the object-store epoch log (the
//!    authority; works for every volume class, including NFS);
//! 4. publishes it in `{volume}/_control/epoch.fence` (refusing if the volume
//!    already names an equal or newer owner);
//! 5. holds an ownership LEASE (see `WriterFence`): the WAL refuses appends once
//!    the lease, renewed only by a fully successful check that saw no newer
//!    epoch, runs out — so a partitioned or paused old owner stops acknowledging
//!    without observing anything;
//! 6. after a takeover, waits one full lease (+ margin) before opening the WAL,
//!    which bounds any overlap with a still-running previous owner;
//! 7. runs a watchdog: on seeing a newer epoch — or failing three checks in a
//!    row — it trips the fence and exits with code 75, WITHOUT the graceful
//!    shutdown (a fenced writer must not flush or publish anything).
//!
//! The same-host `flock` ownership lease stays in force; this module adds the
//! cross-host fence it cannot provide.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use proximadb::cluster::writer_epoch::{
    EpochStore, ObjectStoreEpochStore, claim_for_this_process, new_volume_id,
    set_writer_epoch_gauge,
};
use proximadb::core::Config;
use proximadb::core::config::{DurableVolumeConfig, OltpConfig};
use proximadb::services::writer_fence::WriterFence;
use proximadb_runtime_common::FileLockManager;

use crate::durable_locations;
use crate::storage_ownership::{RootRole, root_plan};

/// Exit code of a process fenced by a newer writer (EX_TEMPFAIL): the
/// supervisor must not restart it against the same volume blindly.
pub(crate) const FENCED_EXIT_CODE: i32 = 75;
const CONTROL_DIR: &str = "_control";
const FENCE_FILE: &str = "epoch.fence";
const VOLUME_ID_FILE: &str = "volume.id";
const MAX_FAILED_CHECKS: u32 = 3;

/// Proof this process owns the volume. Holding it keeps the watchdog running.
pub(crate) struct WriterEpochGuard {
    pub(crate) epoch: u64,
    _watchdog: tokio::task::JoinHandle<()>,
}

/// Lease timing derived from the configured check interval.
#[derive(Debug, Clone, Copy)]
struct Timing {
    interval: Duration,
    /// Upper bound on one authority check; a hung call counts as a failure.
    check_timeout: Duration,
    /// Validity of one successful check, stamped from the check's START.
    lease: Duration,
}

impl Timing {
    fn from_interval_ms(ms: u64) -> Self {
        let interval = Duration::from_millis(ms.max(50));
        Self {
            interval,
            check_timeout: interval,
            lease: interval * 2,
        }
    }

    /// How long a new owner waits after claiming before it opens the WAL. A
    /// previous owner's last successful check started before our claim, so its
    /// lease — and with it any acknowledgement — ends at most one lease later.
    fn takeover_grace(self) -> Duration {
        self.lease + Duration::from_millis(500)
    }
}

/// Run the boot sequence when a durable volume is configured; `Ok(None)`
/// otherwise (no fencing, unchanged behaviour).
pub(crate) async fn acquire(config: &Config) -> anyhow::Result<Option<WriterEpochGuard>> {
    let Some(volume) = OltpConfig::resolve_durable_volume(config.oltp.as_ref()) else {
        set_writer_epoch_gauge(0);
        return Ok(None);
    };
    let deployment_id = volume
        .deployment_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .context(
            "[oltp.durable_volume] deployment_id (or PROXIMADB_OLTP_DEPLOYMENT_ID) is required: \
             it names the epoch log every host that may own this volume shares",
        )?;
    let root = existing_root(&volume.root)?;
    check_durable_roots_on_volume(config, &root)?;
    let control_url = validated_control_url(&volume, &root)?;
    let control_dir = root.join(CONTROL_DIR);
    std::fs::create_dir_all(&control_dir)
        .with_context(|| format!("preparing {}", control_dir.display()))?;

    let timing = Timing::from_interval_ms(volume.fence_check_interval_ms);
    let store = Arc::new(ObjectStoreEpochStore::from_url(
        &control_url,
        &deployment_id,
    )?);
    let volume_id = bind_volume_identity(store.as_ref(), &control_dir, timing).await?;

    let mut claim = claim_for_this_process();
    claim.volume_id = Some(volume_id.clone());
    let claim_started = Instant::now();
    let grant = store
        .acquire_next(&claim)
        .await
        .context("winning the writer epoch for the durable volume")?;
    let epoch = grant.epoch;
    let fence_path = control_dir.join(FENCE_FILE);
    publish_fence(&fence_path, epoch)?;
    let fence = WriterFence::global();
    fence.enforce_lease(claim_started, timing.lease);
    set_writer_epoch_gauge(epoch);
    tracing::info!(
        epoch,
        volume = %root.display(),
        volume_id = %volume_id,
        control_url = %control_url,
        deployment_id = %deployment_id,
        lease_ms = timing.lease.as_millis() as u64,
        "won the writer epoch for the durable volume"
    );

    // The watchdog renews our own lease, so it must run during the grace wait.
    let watchdog = tokio::spawn(watchdog(store, fence_path, epoch, timing, fence));
    if epoch > 1 {
        let grace = timing.takeover_grace();
        tracing::info!(?grace, "takeover: waiting out the previous owner's lease");
        tokio::time::sleep(grace).await;
    }
    Ok(Some(WriterEpochGuard {
        epoch,
        _watchdog: watchdog,
    }))
}

/// The volume root must already exist: creating it would turn an unmounted
/// mount point into an empty database on the root filesystem.
fn existing_root(root: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(root);
    if !path.is_absolute() {
        bail!("[oltp.durable_volume] root '{root}' must be an absolute path");
    }
    if !path.is_dir() {
        bail!(
            "refusing to start: the durable volume root {root} does not exist or is not a \
             directory (is the volume attached and mounted?)"
        );
    }
    path.canonicalize()
        .with_context(|| format!("resolving the durable volume root {root}"))
}

/// Resolve `path` against the cwd, canonicalising its nearest existing
/// ancestor (roots may not exist yet), so symlinks cannot fake containment.
fn resolve(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute =
        std::path::absolute(path).with_context(|| format!("resolving {}", path.display()))?;
    let mut existing = absolute.as_path();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if existing.exists() {
            let mut resolved = existing.canonicalize()?;
            for part in rest.iter().rev() {
                resolved.push(part);
            }
            return Ok(resolved);
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                existing = parent;
            }
            _ => return Ok(absolute),
        }
    }
}

/// Every durable local root (all roles except the rebuildable SST cache, plus
/// env/config-chosen durable locations) must resolve under the volume.
fn check_durable_roots_on_volume(config: &Config, root: &Path) -> anyhow::Result<()> {
    let mut offenders = Vec::new();
    for (role, path) in root_plan(config).local {
        if role == RootRole::SstCache {
            continue;
        }
        if !resolve(&path)?.starts_with(root) {
            offenders.push(format!("{role:?} ({})", path.display()));
        }
    }
    for location in
        durable_locations::effective_durable_locations(config, |key| std::env::var(key).ok())
    {
        if let Some(path) = durable_locations::local_path(&location.location)
            && !resolve(&path)?.starts_with(root)
        {
            offenders.push(format!("{} ({})", location.source, path.display()));
        }
    }
    if !offenders.is_empty() {
        bail!(
            "refusing to start: these durable locations are not on the durable volume {} and would \
             not survive the compute being replaced: {}",
            root.display(),
            offenders.join(", ")
        );
    }
    Ok(())
}

/// The epoch authority must be shared by every host that may own the volume,
/// so it can be neither implicit, relative, nor on the volume itself. A
/// `file://` URL is accepted for single-host and test setups, but it fences
/// only if it is a SHARED mount (NFS/EFS/Azure Files) — a per-host local disk
/// gives each host its own epoch log.
fn validated_control_url(volume: &DurableVolumeConfig, root: &Path) -> anyhow::Result<String> {
    let Some(url) = volume.control_url.clone().filter(|u| !u.trim().is_empty()) else {
        bail!(
            "[oltp.durable_volume] control_url (or PROXIMADB_OLTP_CONTROL_URL) is required: the \
             writer epoch must live in storage every host that may own the volume can reach"
        );
    };
    if let Some(local) = url.strip_prefix("file://") {
        let path = Path::new(local);
        if !path.is_absolute() {
            bail!("control_url '{url}' must be an absolute file:// path");
        }
        if resolve(path)?.starts_with(root) {
            bail!(
                "control_url '{url}' is on the durable volume; the epoch authority must outlive \
                 and be independent of the volume"
            );
        }
        tracing::warn!(
            control_url = %url,
            "file:// epoch authority: fencing across hosts requires this path to be a shared mount"
        );
    } else if !url.contains("://") {
        bail!("control_url '{url}' must be an object-store URL (s3://, az://, gs://, file:///…)");
    }
    Ok(url)
}

fn read_trimmed(path: &Path) -> anyhow::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text.trim().to_string()).filter(|t| !t.is_empty())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).with_context(|| format!("reading {}", path.display())),
    }
}

/// Bind this volume to the epoch log (review F4). The log's newest grant names
/// the volume that owns it; a claimant must present that volume. A volume is
/// only ever given an identity when the log has none to compare against.
async fn bind_volume_identity(
    store: &ObjectStoreEpochStore,
    control_dir: &Path,
    timing: Timing,
) -> anyhow::Result<String> {
    let id_path = control_dir.join(VOLUME_ID_FILE);
    let on_volume = read_trimmed(&id_path)?;
    let latest = tokio::time::timeout(timing.check_timeout * 5, store.current())
        .await
        .context("timed out reading the writer-epoch log")??;
    let expected = latest
        .as_ref()
        .and_then(|grant| grant.claim.volume_id.clone());
    match (expected, on_volume) {
        (Some(expected), Some(found)) if expected == found => Ok(found),
        (Some(expected), Some(found)) => bail!(
            "refusing to start: this volume's identity {found} does not match the volume that \
             owns the epoch log ({expected}); is the right volume attached?"
        ),
        (Some(expected), None) => bail!(
            "refusing to start: the epoch log is bound to volume {expected}, but {} has no \
             identity marker (unmounted or wrong volume?)",
            control_dir.display()
        ),
        (None, Some(found)) => Ok(found),
        (None, None) => {
            let id = new_volume_id();
            let published = FileLockManager::compare_exchange_file(
                &id_path,
                None,
                format!("{id}\n").as_bytes(),
                &[],
            )
            .with_context(|| format!("writing {}", id_path.display()))?;
            if !published {
                bail!(
                    "could not create the volume identity at {} (contention)",
                    id_path.display()
                );
            }
            Ok(id)
        }
    }
}

fn read_fence(path: &Path) -> anyhow::Result<Option<(Vec<u8>, u64)>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let epoch = std::str::from_utf8(&bytes)
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .with_context(|| format!("unreadable fence file {}", path.display()))?;
            Ok(Some((bytes, epoch)))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).with_context(|| format!("reading {}", path.display())),
    }
}

/// Write `epoch` to the volume's fence file with a byte-CAS (fsync'd file and
/// directory). An equal or newer epoch already on the volume is fatal.
fn publish_fence(path: &Path, epoch: u64) -> anyhow::Result<()> {
    let replacement = format!("{epoch}\n");
    for _ in 0..20 {
        let existing = read_fence(path)?;
        if let Some((_, current)) = &existing
            && *current >= epoch
        {
            bail!(
                "refusing to start: the durable volume is fenced at epoch {current} \
                 (this process won {epoch}); another writer owns it"
            );
        }
        let expected = existing.as_ref().map(|(bytes, _)| bytes.as_slice());
        if FileLockManager::compare_exchange_file(path, expected, replacement.as_bytes(), &[])
            .with_context(|| format!("publishing {}", path.display()))?
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    bail!(
        "could not publish the epoch fence at {} (persistent contention)",
        path.display()
    )
}

/// What the watchdog decided on one check.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Both sources were read and neither names a newer owner: renew the lease.
    Renew,
    /// A source could not be read: do NOT renew (the lease runs down), keep
    /// counting failures.
    Hold,
    Fenced(String),
}

fn judge(
    mine: u64,
    fence: anyhow::Result<Option<u64>>,
    authority: anyhow::Result<Option<u64>>,
    failures: &mut u32,
) -> Verdict {
    if let Ok(Some(volume_epoch)) = fence
        && volume_epoch > mine
    {
        return Verdict::Fenced(format!("the volume is fenced at epoch {volume_epoch}"));
    }
    if let Ok(Some(latest)) = authority
        && latest > mine
    {
        return Verdict::Fenced(format!("the epoch authority granted epoch {latest}"));
    }
    match (fence, authority) {
        (Ok(_), Ok(_)) => {
            *failures = 0;
            Verdict::Renew
        }
        (fence, authority) => {
            *failures += 1;
            if *failures >= MAX_FAILED_CHECKS {
                let fence_err = fence.err().map(|e| format!("{e:#}"));
                let authority_err = authority.err().map(|e| format!("{e:#}"));
                Verdict::Fenced(format!(
                    "{MAX_FAILED_CHECKS} ownership checks in a row failed (fence: {fence_err:?}, \
                     authority: {authority_err:?}); self-fencing rather than risk two writers"
                ))
            } else {
                Verdict::Hold
            }
        }
    }
}

async fn watchdog(
    store: Arc<ObjectStoreEpochStore>,
    fence_path: PathBuf,
    mine: u64,
    timing: Timing,
    fence: Arc<WriterFence>,
) {
    let mut failures = 0u32;
    let mut tick = tokio::time::interval(timing.interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let started = Instant::now();
        let volume = read_fence(&fence_path).map(|f| f.map(|(_, epoch)| epoch));
        let authority = match tokio::time::timeout(timing.check_timeout, store.current()).await {
            Ok(result) => result.map(|grant| grant.map(|g| g.epoch)),
            Err(_) => Err(anyhow::anyhow!(
                "epoch authority did not answer within {:?}",
                timing.check_timeout
            )),
        };
        match judge(mine, volume, authority, &mut failures) {
            Verdict::Renew => fence.renew_lease(started, timing.lease),
            Verdict::Hold => {
                tracing::warn!(
                    epoch = mine,
                    failures,
                    "writer-epoch check failed; the ownership lease is not renewed"
                );
            }
            Verdict::Fenced(reason) => {
                fence.mark_lost();
                tracing::error!(
                    epoch = mine,
                    %reason,
                    "writer fenced: another process owns the durable volume; exiting without flushing"
                );
                eprintln!("proximadb: writer epoch {mine} fenced ({reason}); exiting");
                std::process::exit(FENCED_EXIT_CODE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb::cluster::writer_epoch::WriterClaim;

    fn volume(root: &Path, control_url: Option<String>) -> DurableVolumeConfig {
        DurableVolumeConfig {
            root: root.display().to_string(),
            deployment_id: Some("default".to_string()),
            control_url,
            fence_check_interval_ms: 1000,
        }
    }

    #[test]
    fn control_url_must_be_explicit_absolute_and_off_volume() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("vol");
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        assert!(validated_control_url(&volume(&root, None), &root).is_err());
        assert!(
            validated_control_url(&volume(&root, Some("file://rel/ctl".into())), &root).is_err()
        );
        let on_volume = format!("file://{}/ctl", root.display());
        assert!(validated_control_url(&volume(&root, Some(on_volume)), &root).is_err());
        assert!(validated_control_url(&volume(&root, Some("bare/path".into())), &root).is_err());
        let off_volume = format!("file://{}/ctl", dir.path().display());
        assert!(validated_control_url(&volume(&root, Some(off_volume)), &root).is_ok());
        assert!(
            validated_control_url(&volume(&root, Some("s3://bucket/ctl".into())), &root).is_ok()
        );
        Ok(())
    }

    #[test]
    fn a_missing_root_is_never_created() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let missing = dir.path().join("not-mounted");
        assert!(existing_root(&missing.display().to_string()).is_err());
        assert!(!missing.exists(), "the root must not be created");
        Ok(())
    }

    #[test]
    fn fence_refuses_an_equal_or_newer_owner() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let fence = dir.path().join(CONTROL_DIR).join(FENCE_FILE);
        std::fs::create_dir_all(dir.path().join(CONTROL_DIR))?;
        publish_fence(&fence, 3)?;
        assert_eq!(read_fence(&fence)?.map(|(_, e)| e), Some(3));
        assert!(
            publish_fence(&fence, 3).is_err(),
            "equal epoch must be refused"
        );
        assert!(
            publish_fence(&fence, 2).is_err(),
            "older epoch must be refused"
        );
        publish_fence(&fence, 4)?;
        assert_eq!(read_fence(&fence)?.map(|(_, e)| e), Some(4));
        Ok(())
    }

    #[test]
    fn watchdog_renews_only_on_a_fully_successful_check() {
        let mut failures = 0;
        assert_eq!(
            judge(5, Ok(Some(5)), Ok(Some(5)), &mut failures),
            Verdict::Renew
        );
        assert!(matches!(
            judge(5, Ok(Some(6)), Ok(Some(5)), &mut failures),
            Verdict::Fenced(_)
        ));
        assert!(matches!(
            judge(5, Ok(Some(5)), Ok(Some(7)), &mut failures),
            Verdict::Fenced(_)
        ));
        let lost = || Err(anyhow::anyhow!("unreachable"));
        assert_eq!(judge(5, Ok(Some(5)), lost(), &mut failures), Verdict::Hold);
        assert_eq!(judge(5, lost(), Ok(Some(5)), &mut failures), Verdict::Hold);
        assert!(matches!(
            judge(5, Ok(Some(5)), lost(), &mut failures),
            Verdict::Fenced(_)
        ));
        // A successful check resets the streak.
        let mut failures = 2;
        assert_eq!(
            judge(5, Ok(None), Ok(Some(5)), &mut failures),
            Verdict::Renew
        );
        assert_eq!(failures, 0);
    }

    #[test]
    fn takeover_grace_covers_the_previous_owners_lease() {
        let timing = Timing::from_interval_ms(200);
        assert!(timing.takeover_grace() > timing.lease);
        assert!(timing.check_timeout <= timing.interval);
    }

    // Review F4: an unmounted or foreign volume can never take over the log.
    #[tokio::test]
    async fn volume_identity_binds_the_epoch_log() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let ctl = format!("file://{}/ctl", dir.path().display());
        std::fs::create_dir_all(dir.path().join("ctl"))?;
        let store = ObjectStoreEpochStore::from_url(&ctl, "dep")?;
        let timing = Timing::from_interval_ms(200);
        let vol_a = dir.path().join("a/_control");
        let vol_b = dir.path().join("b/_control");
        std::fs::create_dir_all(&vol_a)?;
        std::fs::create_dir_all(&vol_b)?;

        // First boot: the volume is given an identity and claims epoch 1 with it.
        let id_a = bind_volume_identity(&store, &vol_a, timing).await?;
        let mut claim: WriterClaim = claim_for_this_process();
        claim.volume_id = Some(id_a.clone());
        store.acquire_next(&claim).await?;

        // The same volume rebinds to the same identity.
        assert_eq!(bind_volume_identity(&store, &vol_a, timing).await?, id_a);
        // An empty (e.g. unmounted) volume is refused and NOT given an identity.
        assert!(bind_volume_identity(&store, &vol_b, timing).await.is_err());
        assert!(!vol_b.join(VOLUME_ID_FILE).exists());
        // A different volume's identity is refused.
        std::fs::write(vol_b.join(VOLUME_ID_FILE), "someone-else\n")?;
        assert!(bind_volume_identity(&store, &vol_b, timing).await.is_err());
        Ok(())
    }

    #[test]
    fn resolve_canonicalises_the_existing_prefix() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let real = dir.path().join("real");
        std::fs::create_dir_all(&real)?;
        let resolved = resolve(&real.join("not-yet/created"))?;
        assert!(resolved.starts_with(real.canonicalize()?));
        Ok(())
    }
}
