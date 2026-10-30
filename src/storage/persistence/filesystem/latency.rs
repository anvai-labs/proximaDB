//! TD-PAXDIR-1 test support: a fixed per-request latency injected **inside**
//! the physical request (design §4.1, "a test-only latency wrapper").
//!
//! With `PROXIMADB_TEST_FS_LATENCY_MS` set, every request the local backend
//! times for the per-query trace — read, ranged read, LIST, HEAD and
//! existence check — first sleeps that long. The sleep sits inside the
//! interval the backend records (`record_physical_*_read`,
//! `traced_request`), so the production wave count
//! (`io_trace::critical_path_waves`, TD-PAXDIR-1 D7) sees requests a task
//! issues together overlap and requests it issues in sequence not, exactly as
//! round trips to an object store would. Tests read waves and requests from
//! the query's `IoTraceSnapshot`, i.e. from the production metric itself.
//!
//! Compiled only for tests; a no-op unless the env gate is set.

use std::sync::OnceLock;
use std::time::Duration;

fn delay() -> Option<Duration> {
    static DELAY: OnceLock<Option<Duration>> = OnceLock::new();
    *DELAY.get_or_init(|| {
        std::env::var("PROXIMADB_TEST_FS_LATENCY_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .map(Duration::from_millis)
    })
}

/// Sleep for the injected per-request latency, if armed.
pub(crate) async fn inject() {
    if let Some(delay) = delay() {
        tokio::time::sleep(delay).await;
    }
}
