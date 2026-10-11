//! Process-owned shutdown sequencing. The database keeps its retryable API;
//! the executable retains its lifecycle record until its shutdown work is done.

use crate::runtime_state::{Phase, RuntimeStateWriter};

/// The attempt returns its result and whether the queue remains attached.
/// Database shutdown retains that queue only on an incomplete drainer stop;
/// later shutdown errors occur after queue removal and must not be retried.
pub(crate) async fn finish(
    runtime_state: RuntimeStateWriter,
    mut attempt: impl AsyncFnMut() -> (anyhow::Result<()>, bool),
) -> anyhow::Result<()> {
    runtime_state.set_phase(Phase::Stopping);
    loop {
        let (result, queue_attached) = attempt().await;
        match (result, queue_attached) {
            (Err(error), true) => {
                // The database's bounded attempt already waited for its grace
                // period. Keep the same task/storage alive, without restarting
                // a consumer or clearing the process ownership record.
                tracing::warn!(%error, "Shutdown incomplete; awaiting retained drainer");
            }
            (result, _) => {
                if let Err(error) = &result {
                    tracing::error!(%error, "Error during shutdown");
                }
                runtime_state.finish();
                return result;
            }
        }
    }
}
