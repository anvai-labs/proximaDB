//! Handle-relative local reads for registered external tables.

use std::fmt;
use std::io;
use std::sync::Arc;

use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use object_store::path::Path;
use object_store::{
    CopyOptions, Error, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};

#[derive(Debug)]
pub(super) struct ConfinedFileStore {
    root: Arc<Dir>,
    expose_delta_log: bool,
}

impl ConfinedFileStore {
    pub(super) async fn open(root: std::path::PathBuf, table: Path) -> io::Result<Self> {
        tokio::task::spawn_blocking(move || {
            let root = Dir::open_ambient_dir(root, cap_std::ambient_authority())?;
            let dir = table
                .parts()
                .try_fold(root, |dir, part| dir.open_dir_nofollow(part.as_ref()))?;
            Ok(Self {
                root: Arc::new(dir),
                expose_delta_log: false,
            })
        })
        .await?
    }
    pub(super) fn with_delta_log(mut self) -> Self {
        self.expose_delta_log = true;
        self
    }
}

impl fmt::Display for ConfinedFileStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("confined local external file store")
    }
}

fn unsupported() -> Error {
    Error::NotSupported {
        source: "only local external file object reads are supported".into(),
    }
}

fn store_error(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Error {
    Error::Generic {
        store: "confined local external file store",
        source: source.into(),
    }
}

#[async_trait::async_trait]
impl ObjectStore for ConfinedFileStore {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        if options.version.is_some() {
            return Err(unsupported());
        }
        let root = self.root.clone();
        let location = location.clone();
        tokio::task::spawn_blocking(move || {
            let result = (|| {
                let mut parts = location.parts().peekable();
                let mut dir = None;
                let file = loop {
                    let part = parts.next().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "expected a file path")
                    })?;
                    let parent = dir.as_ref().unwrap_or(root.as_ref());
                    if parts.peek().is_none() {
                        // Open first and check the resulting handle. O_NONBLOCK
                        // prevents a substituted FIFO from blocking the reader.
                        let mut opts = OpenOptions::new();
                        opts.read(true)
                            .follow(FollowSymlinks::No)
                            .custom_flags(libc::O_NONBLOCK);
                        break parent.open_with(part.as_ref(), &opts)?.into_std();
                    }
                    dir = Some(parent.open_dir_nofollow(part.as_ref())?);
                };
                let metadata = file.metadata()?;
                if !metadata.is_file() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "expected a regular file",
                    ));
                }
                Ok((file, metadata))
            })();
            let (file, metadata) = result.map_err(|source| {
                if source.kind() == io::ErrorKind::NotFound {
                    Error::NotFound {
                        path: location.to_string(),
                        source: source.into(),
                    }
                } else {
                    store_error(source)
                }
            })?;
            let size = metadata.len();
            let meta = ObjectMeta {
                location: location.clone(),
                last_modified: metadata.modified().map_err(store_error)?.into(),
                size,
                e_tag: None,
                version: None,
            };
            options.check_preconditions(&meta)?;
            let (payload, range) = if options.head {
                (GetResultPayload::Stream(stream::empty().boxed()), 0..0)
            } else {
                let range = options
                    .range
                    .as_ref()
                    .map_or(Ok(0..size), |r| r.as_range(size))
                    .map_err(store_error)?;
                (
                    GetResultPayload::File(file, location.as_ref().into()),
                    range,
                )
            };
            Ok(GetResult {
                payload,
                range,
                meta,
                attributes: Default::default(),
                extensions: Default::default(),
            })
        })
        .await
        .map_err(store_error)?
    }

    async fn put_opts(&self, _: &Path, _: PutPayload, _: PutOptions) -> Result<PutResult> {
        Err(unsupported())
    }

    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Err(unsupported())
    }

    fn delete_stream(
        &self,
        _: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        stream::iter([Err(unsupported())]).boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        use futures::TryStreamExt;

        // Kernel's main log discovery lists direct children. Checkpoint sidecars
        // are opened by their recorded paths and do not need directory listing.
        if !self.expose_delta_log || !prefix.is_some_and(|p| p.as_ref() == "_delta_log") {
            return stream::iter([Err(unsupported())]).boxed();
        }
        let root = self.root.clone();
        stream::once(async move {
            let entries = tokio::task::spawn_blocking(move || delta_log_entries(&root))
                .await
                .map_err(store_error)?
                .map_err(store_error)?;
            Ok::<_, Error>(stream::iter(entries.into_iter().map(Ok)))
        })
        .try_flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, _: Option<&Path>) -> Result<ListResult> {
        Err(unsupported())
    }

    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> Result<()> {
        Err(unsupported())
    }
}

// Bound directory work independently of the table size. This includes ignored
// entries so unrelated files cannot turn a log lookup into an unbounded walk.
fn delta_log_entries(root: &Dir) -> io::Result<Vec<ObjectMeta>> {
    let prefix = Path::from("_delta_log");
    let dir = root.open_dir_nofollow("_delta_log")?;
    let mut result = Vec::new();
    for (index, entry) in dir.entries()?.enumerate() {
        if index >= 16_384 {
            return Err(io::Error::other(
                "Delta log directory exceeds 16384 entries",
            ));
        }
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.is_ascii() || name.len() > 255 {
            continue;
        }
        let metadata = dir.symlink_metadata(name)?;
        if metadata.is_symlink() {
            return Err(io::Error::other("symlink in confined Delta log"));
        }
        if metadata.is_file() {
            result.push(ObjectMeta {
                location: prefix.clone().join(name),
                last_modified: metadata.modified()?.into_std().into(),
                size: metadata.len(),
                e_tag: None,
                version: None,
            });
        }
    }
    result.sort_unstable_by(|left, right| left.location.cmp(&right.location));
    Ok(result)
}
