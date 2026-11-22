//! Complete immutable PAX objects held only for the lifetime of one search.
//! Each top-level search binds one scope to its engine, collection, storage
//! assignment and discovery snapshot. Compaction publishes new immutable keys.
//! A pathname here can reuse bytes only during that search and after normal
//! membership/retirement checks; it is never a cross-query membership proof.
//! OnceCell joins concurrent fills of the same key. Invalid or short reads
//! never become visible to later stages. Dropping the search returns its
//! reservations on success, cancellation or error.
//!
//! Per-query and shared limits bound the retained bodies (decoder buffers and
//! result copies have separate lifetimes). Admission never waits for memory:
//! a busy query falls back to its existing range plan and quality contract.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::{OnceCell, Semaphore, SemaphorePermit};

const QUERY_BYTES: u32 = 8 << 20;
static PROCESS_BYTES: Semaphore = Semaphore::const_new(4 * QUERY_BYTES as usize);

#[derive(Default)]
struct Bodies {
    bytes: u32,
    objects: HashMap<String, Arc<Slot>>,
}

struct Slot {
    _permit: SemaphorePermit<'static>,
    size: u32,
    body: OnceCell<Arc<Vec<u8>>>,
}

tokio::task_local! {
    static BODIES: QueryBodies;
}

#[derive(Clone, Default)]
pub(super) struct QueryBodies(Arc<OnceLock<Mutex<Bodies>>>);

impl QueryBodies {
    #[cfg(test)]
    pub(super) fn current() -> Self {
        BODIES.try_with(Clone::clone).unwrap_or_default()
    }

    pub(super) async fn scope<F: Future>(self, future: F) -> F::Output {
        BODIES.scope(self, future).await
    }
}

pub(super) fn get(path: &str) -> Option<Arc<Vec<u8>>> {
    BODIES
        .try_with(|memo| {
            let memo = memo
                .0
                .get()?
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            memo.objects.get(path)?.body.get().cloned()
        })
        .ok()
        .flatten()
}

pub(super) async fn load<F>(path: &str, size: u64, fetch: F) -> Option<anyhow::Result<Arc<Vec<u8>>>>
where
    F: Future<Output = anyhow::Result<Vec<u8>>>,
{
    let size = u32::try_from(size).ok()?;
    if size == 0
        || size > QUERY_BYTES
        || proximadb_storage_filesystem_types::object_store_scheme::ObjectStoreScheme::of(path)
            .is_none()
    {
        return None;
    }
    let slot = BODIES
        .try_with(|memo| {
            let mut memo = memo
                .0
                .get_or_init(Mutex::default)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(slot) = memo.objects.get(path) {
                return (slot.size == size).then(|| slot.clone());
            }
            if size > QUERY_BYTES - memo.bytes {
                return None;
            }
            let permit = PROCESS_BYTES.try_acquire_many(size).ok()?;
            let slot = Arc::new(Slot {
                _permit: permit,
                size,
                body: OnceCell::new(),
            });
            memo.bytes += size;
            memo.objects.insert(path.to_owned(), slot.clone());
            Some(slot)
        })
        .ok()
        .flatten()?;
    Some(
        slot.body
            .get_or_try_init(|| async {
                let bytes = fetch.await?;
                anyhow::ensure!(
                    bytes.len() == size as usize,
                    "incomplete PAX body for {path}"
                );
                Ok(Arc::new(bytes))
            })
            .await
            .cloned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn query_body_retries_failures_deduplicates_and_expires() {
        const PATH: &str = "s3://bucket/a.pax";
        QueryBodies::default()
            .scope(async {
                assert!(
                    load(PATH, 4, async { Ok(vec![0; 3]) })
                        .await
                        .unwrap()
                        .is_err()
                );
                assert!(get(PATH).is_none());
                let calls = std::sync::atomic::AtomicUsize::new(0);
                let read = || {
                    load(PATH, 4, async {
                        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tokio::task::yield_now().await;
                        Ok(vec![1, 2, 3, 4])
                    })
                };
                let (a, b) = tokio::join!(read(), read());
                assert!(Arc::ptr_eq(&a.unwrap().unwrap(), &b.unwrap().unwrap()));
                assert_eq!(calls.into_inner(), 1);
                assert!(get("s3://other/a.pax").is_none());
                assert!(
                    load(PATH, 5, async { panic!("size changed") })
                        .await
                        .is_none()
                );
                assert!(
                    load("s3://bucket/too-big", QUERY_BYTES as u64, async {
                        panic!("query budget exhausted")
                    })
                    .await
                    .is_none()
                );
            })
            .await;
        assert!(get(PATH).is_none());
    }
}
