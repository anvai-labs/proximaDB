//! The process-wide ONNX Runtime environment (TD-ORT-1).
//!
//! `ort` 2.0.0-rc.15 makes the environment explicit: `Session::builder`
//! takes an `&Environment`, and only ONE environment may exist per process —
//! `EnvironmentBuilder::build` fails with "only one environment is allowed
//! per process" once the process-global cell is set, and the cell is a
//! `OnceLock<Weak<_>>` that stays set for the life of the process even after
//! every strong handle is dropped. `Environment::current()` is `pub(crate)`,
//! so a crate cannot borrow one another crate created.
//!
//! ProximaDB builds ONNX sessions from two crates that do not depend on each
//! other (`proximadb-embedding`'s BGE pool and `proximadb-rank-onnx`'s
//! scorers), both enabled together by the root `onnx` feature. A per-crate
//! environment would compile and then fail at runtime in any process that
//! loads both — the failure the tests below pin.
//!
//! A live environment is also required beyond session construction: copying
//! tensors goes through it (`ort`'s `value/impl_tensor/copy.rs`), so the
//! shared handle is held for the life of the process (leaked on first use)
//! rather than only while building sessions.

use ort::environment::Environment;
use std::sync::Mutex;

static SHARED: Mutex<Option<&'static Environment>> = Mutex::new(None);

/// The one ONNX Runtime environment for this process, built on first use.
///
/// The first call initializes the runtime (an arbitrary thread/platform
/// configuration is NOT needed: execution providers still register
/// per-session); later calls return the same handle. A failed build is not
/// cached — the next call retries — so a transient startup failure does not
/// permanently disable ONNX for the process.
pub fn shared_environment() -> ort::Result<&'static Environment> {
    // The critical section is panic-free (? returns without unwinding), so
    // the lock cannot be poisoned; recover the inner value regardless
    // (mandate #4: no .expect() in production code) — the Option it guards
    // is always None or a valid leaked reference, never half-written.
    let mut guard = SHARED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(environment) = *guard {
        return Ok(environment);
    }
    let environment: &'static Environment = Box::leak(Box::new(ort::init().build()?));
    *guard = Some(environment);
    Ok(environment)
}

#[cfg(test)]
mod tests {
    use super::shared_environment;
    use ort::session::Session;

    /// TD-ORT-1's regression gate: two sessions built in one process through
    /// the shared environment. Under a per-crate (or per-call) environment
    /// the second `ort::init().build()` fails for the life of the process —
    /// exactly the production failure this crate exists to prevent. The
    /// fixture is a 65-byte Identity model, so the test needs no weights.
    #[test]
    fn two_sessions_share_one_environment() {
        let model = include_bytes!("../testdata/identity.onnx");
        for _ in 0..2 {
            let mut session = Session::builder(shared_environment().expect("shared environment"))
                .and_then(|mut builder| builder.commit_from_memory(model))
                .expect("session from shared environment");
            let input = ort::value::Tensor::from_array((vec![1usize], vec![42.0f32]))
                .expect("input tensor");
            let output = session
                .run(ort::inputs!["x" => input])
                .expect("run identity");
            let (_, data) = output["y"].try_extract_tensor::<f32>().expect("y");
            assert_eq!(data, &[42.0]);
        }
    }
}
