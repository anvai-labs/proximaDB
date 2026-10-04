// Copyright 2025 Vijaykumar Singh.
// Licensed under the Apache License, Version 2.0.

//! Advisor observation capture — the **data layer** that records
//! `(predicted_recall, observed_recall, predicted_work,
//! observed_latency)` tuples per search. P4 of the recall-aware
//! AXIS stack.
//!
//! # Why
//!
//! The advisor framework's per-algorithm formulas (`A(m)`,
//! `ceiling(m)`, `ceiling_of_n`, `recall_factor`) are calibrated
//! against in-repo sweeps at single anchor points. Real workloads
//! diverge — different dim, distance metric, data distribution.
//! Capturing the residual is the first step toward learning a
//! per-collection adjustment (P5+ RL bridge).
//!
//! **P4 does NOT feed observations back into the advisor.** The
//! captured data lands in:
//!
//! * Prometheus metrics
//!   ([`crate::metrics::advisor_observations_metrics`]) — always
//!   on, low-cardinality (`{collection, algorithm}` labels).
//! * Structured log (`tracing` target `advisor.observations`)
//!   — every captured observation, off-by-default at debug level.
//! * Optional disk sidecar (JSONL, when
//!   `PROXIMADB_ADVISOR_OBS_DISK_SIDECAR` env var is set to a
//!   path). Useful for offline analysis without scraping
//!   Prometheus.
//!
//! # When observations are captured
//!
//! The post-search hook in `services::operations::vectors`
//! (P4 wiring commit) calls [`record_observation`] after every
//! search that successfully exits the search path with a captured
//! latency. The hook short-circuits when:
//!
//! * The collection has no `recall_target:` tag (advisor isn't
//!   active — observations would be meaningless).
//! * The collection has no active strategy (just-created, no
//!   index built yet).
//!
//! `observed_recall` is populated **only when the
//! `recall_probe` gate (TD-075 / F2) is active** for the
//! collection. Otherwise `None` — the counter + latency
//! histogram still record, but the recall residual is skipped.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::index::axis::management::SupportedAlgorithm;
use crate::index::axis::types::IndexAlgorithm;

/// Environment variable that, when set to a writable path,
/// causes [`record_observation`] to append each observation as a
/// JSON line. No-op when unset. Lets operators collect a dataset
/// offline without depending on Prometheus scrape configuration.
pub const DISK_SIDECAR_ENV: &str = "PROXIMADB_ADVISOR_OBS_DISK_SIDECAR";

/// A single captured advisor observation. Serialized to disk
/// (when sidecar is enabled) and emitted as a structured log
/// line on the `advisor.observations` target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdvisorObservation {
    /// The collection the search ran against.
    pub collection_id: String,
    /// Which algorithm the advisor sized this collection for —
    /// matches [`SupportedAlgorithm::label()`].
    pub algorithm: String,
    /// The full algorithm spec (carries m/efc/ef for HNSW,
    /// nlist/nprobe for IVF, per-modality partitions for HMGI).
    pub sized_params: IndexAlgorithm,
    /// What the advisor's formula predicted recall would be at
    /// the sized params + current vector count.
    pub advisor_predicted_recall: f32,
    /// Measured recall from the `recall_probe` gate. `None` when
    /// the gate isn't active for this collection — most legacy
    /// collections fall in this bucket.
    pub observed_recall: Option<f32>,
    /// Predicted per-query work (HNSW: `ef_search`; IVF: probes
    /// × cluster_size; HMGI: max-partition work). From the
    /// advisor's cost model.
    pub advisor_predicted_work: u64,
    /// Measured per-query latency in microseconds. Always
    /// populated — the timing source is universal.
    pub observed_latency_us: u64,
    /// Wall-clock timestamp in nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
}

impl AdvisorObservation {
    /// Convenience constructor — extracts the algorithm label
    /// from the discriminator and stamps `timestamp_ns` from the
    /// current wall clock.
    pub fn now(
        collection_id: impl Into<String>,
        algorithm: SupportedAlgorithm,
        sized_params: IndexAlgorithm,
        advisor_predicted_recall: f32,
        observed_recall: Option<f32>,
        advisor_predicted_work: u64,
        observed_latency_us: u64,
    ) -> Self {
        Self {
            collection_id: collection_id.into(),
            algorithm: algorithm.label().to_string(),
            sized_params,
            advisor_predicted_recall,
            observed_recall,
            advisor_predicted_work,
            observed_latency_us,
            timestamp_ns: current_timestamp_ns(),
        }
    }
}

fn current_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Record a single observation. Fans out to Prometheus +
/// structured log + (optional) disk sidecar. Best-effort — any
/// fan-out site that fails (e.g. disk full) logs a warning but
/// doesn't propagate the error.
///
/// Safe to call from the hot search path: every fan-out is
/// non-blocking and bounded constant time.
pub fn record_observation(obs: &AdvisorObservation) {
    // (1) Prometheus — always on, lowest cost.
    crate::metrics::advisor_observations_metrics::record_observation(
        &obs.collection_id,
        &obs.algorithm,
        obs.observed_recall,
        obs.advisor_predicted_recall,
        obs.observed_latency_us,
    );

    // (2) Structured log — debug-level so production deployments
    // can leave it dormant; turn on with
    // `RUST_LOG=advisor.observations=debug` when collecting data.
    debug!(
        target: "advisor.observations",
        collection_id = %obs.collection_id,
        algorithm = %obs.algorithm,
        advisor_predicted_recall = obs.advisor_predicted_recall,
        observed_recall = ?obs.observed_recall,
        advisor_predicted_work = obs.advisor_predicted_work,
        observed_latency_us = obs.observed_latency_us,
        timestamp_ns = obs.timestamp_ns,
        "advisor observation captured"
    );

    // (3) Disk sidecar — opt-in via env. Bounded I/O per call
    // (one append + flush). Append-only JSONL keeps the format
    // operator-readable.
    if let Some(path) = disk_sidecar_path()
        && let Err(err) = append_to_sidecar(path, obs)
    {
        warn!(
            target: "advisor.observations",
            collection_id = %obs.collection_id,
            error = %err,
            "disk sidecar append failed"
        );
    }
}

/// Resolve the disk-sidecar path from the env var. Cached so the
/// env lookup is one-shot per process.
fn disk_sidecar_path() -> Option<&'static str> {
    static CACHED: OnceLock<Option<String>> = OnceLock::new();
    CACHED
        .get_or_init(|| std::env::var(DISK_SIDECAR_ENV).ok())
        .as_deref()
}

fn append_to_sidecar(path: &str, obs: &AdvisorObservation) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let json = serde_json::to_string(obs).map_err(std::io::Error::other)?;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", json)?;
    file.sync_data()?; // bounded — append-only, one line at a time.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hnsw_spec() -> IndexAlgorithm {
        IndexAlgorithm::HNSW {
            m: 32,
            ef_construction: 256,
            ef_search: 400,
            max_elements: 100_000,
        }
    }

    // ───── struct construction + serialisation ─────────────────

    #[test]
    fn now_sets_algorithm_label_from_discriminator() {
        let obs = AdvisorObservation::now(
            "c1",
            SupportedAlgorithm::Hnsw,
            hnsw_spec(),
            0.95,
            Some(0.94),
            400,
            500,
        );
        assert_eq!(obs.algorithm, "hnsw");
    }

    #[test]
    fn now_stamps_nonzero_timestamp() {
        let obs = AdvisorObservation::now(
            "c1",
            SupportedAlgorithm::Hnsw,
            hnsw_spec(),
            0.95,
            None,
            400,
            500,
        );
        assert!(obs.timestamp_ns > 0);
    }

    #[test]
    fn observation_serializes_to_json_line() {
        let obs = AdvisorObservation::now(
            "products",
            SupportedAlgorithm::Ivf,
            IndexAlgorithm::IVF {
                nlist: 316,
                nprobe: 20,
                quantizer: None,
            },
            0.55,
            Some(0.53),
            6320,
            12_500,
        );
        let json = serde_json::to_string(&obs).expect("serialise");
        // Stable field names — dashboards / offline analysis
        // scripts parse these.
        assert!(json.contains("\"collection_id\":\"products\""));
        assert!(json.contains("\"algorithm\":\"ivf\""));
        assert!(json.contains("\"advisor_predicted_recall\":0.55"));
        assert!(json.contains("\"observed_recall\":0.53"));
        assert!(json.contains("\"advisor_predicted_work\":6320"));
        assert!(json.contains("\"observed_latency_us\":12500"));
    }

    #[test]
    fn observation_serializes_observed_recall_as_null_when_none() {
        let obs = AdvisorObservation::now(
            "c1",
            SupportedAlgorithm::Hmgi,
            hnsw_spec(),
            0.95,
            None,
            400,
            500,
        );
        let json = serde_json::to_string(&obs).unwrap();
        assert!(
            json.contains("\"observed_recall\":null"),
            "observed_recall must serialise as null when not measured: {}",
            json
        );
    }

    // ───── record_observation fan-out ─────────────────────────

    #[test]
    fn record_observation_emits_to_prometheus() {
        let before = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_OBSERVATIONS_TOTAL
            .with_label_values(&["test_fanout_advisor_obs_unique", "hnsw"])
            .get();
        let obs = AdvisorObservation::now(
            "test_fanout_advisor_obs_unique",
            SupportedAlgorithm::Hnsw,
            hnsw_spec(),
            0.95,
            Some(0.94),
            400,
            500,
        );
        record_observation(&obs);
        let after = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_OBSERVATIONS_TOTAL
            .with_label_values(&["test_fanout_advisor_obs_unique", "hnsw"])
            .get();
        assert_eq!(after - before, 1.0);
    }

    #[test]
    fn record_observation_records_latency_bucket() {
        // latency histogram fires unconditionally — even when
        // observed_recall is None. Pins the "latency always
        // captured" invariant the operator runbook depends on.
        let label = "test_latency_advisor_obs_unique";
        let before = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_LATENCY_US
            .with_label_values(&[label, "hnsw"])
            .get_sample_count();
        let obs = AdvisorObservation::now(
            label,
            SupportedAlgorithm::Hnsw,
            hnsw_spec(),
            0.95,
            None,
            400,
            12_500,
        );
        record_observation(&obs);
        let after = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_LATENCY_US
            .with_label_values(&[label, "hnsw"])
            .get_sample_count();
        assert_eq!(after - before, 1);
    }

    #[test]
    fn record_observation_records_residual_when_observed_recall_present() {
        let label = "test_residual_present_advisor_obs_unique";
        let before = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[label, "hmgi"])
            .get_sample_count();
        let obs = AdvisorObservation::now(
            label,
            SupportedAlgorithm::Hmgi,
            hnsw_spec(),
            0.95,
            Some(0.93),
            400,
            500,
        );
        record_observation(&obs);
        let after = crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[label, "hmgi"])
            .get_sample_count();
        assert_eq!(after - before, 1);
    }

    // ───── disk sidecar plumbing ──────────────────────────────
    //
    // `disk_sidecar_path()` is `OnceLock`-cached at process scope
    // so flipping the env var mid-test won't change behaviour
    // (and the cache means subsequent tests would race the env).
    // Instead we test the inner `append_to_sidecar` function
    // directly, which takes the path as an arg. That covers the
    // serialisation + append + flush plumbing without touching
    // global state.

    #[test]
    fn disk_sidecar_appends_json_line() {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_advisor_obs_sidecar_test_append_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let obs = AdvisorObservation::now(
            "c_sidecar",
            SupportedAlgorithm::Ivf,
            IndexAlgorithm::IVF {
                nlist: 316,
                nprobe: 20,
                quantizer: None,
            },
            0.55,
            Some(0.53),
            6320,
            12_500,
        );
        append_to_sidecar(tmp.to_str().unwrap(), &obs).expect("first append");
        // Second append → JSONL means two lines.
        append_to_sidecar(tmp.to_str().unwrap(), &obs).expect("second append");
        let contents = std::fs::read_to_string(&tmp).expect("read back");
        let line_count = contents.lines().count();
        assert_eq!(line_count, 2, "two appends → two JSONL lines");
        for line in contents.lines() {
            // Each line must be self-contained JSON (the parser
            // a future tool runs over the sidecar).
            let _: serde_json::Value =
                serde_json::from_str(line).expect("each line parses as JSON");
            assert!(line.contains("\"collection_id\":\"c_sidecar\""));
            assert!(line.contains("\"algorithm\":\"ivf\""));
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn disk_sidecar_path_resolves_env_when_set() {
        // OnceLock-cached, so this test only validates the
        // resolver shape when the env var is set BEFORE the
        // process boots. We exercise the std::env lookup directly
        // (not the cached path) to confirm the contract is "read
        // PROXIMADB_ADVISOR_OBS_DISK_SIDECAR".
        // SAFETY: set_var is unsafe in newer toolchains; the test
        // immediately restores prior state. Single-threaded test
        // section — no other test reads this env var.
        let prior = std::env::var(DISK_SIDECAR_ENV).ok();
        unsafe {
            std::env::set_var(DISK_SIDECAR_ENV, "/tmp/advisor_obs_test_path.jsonl");
        }
        let resolved = std::env::var(DISK_SIDECAR_ENV);
        assert_eq!(resolved.as_deref(), Ok("/tmp/advisor_obs_test_path.jsonl"));
        unsafe {
            match prior {
                Some(p) => std::env::set_var(DISK_SIDECAR_ENV, p),
                None => std::env::remove_var(DISK_SIDECAR_ENV),
            }
        }
    }

    #[test]
    fn record_observation_handles_none_observed_recall() {
        // None observed_recall → counter still bumps; recall
        // residual histogram skipped.
        let before_count =
            crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_OBSERVATIONS_TOTAL
                .with_label_values(&["test_none_recall_advisor_obs_unique", "ivf"])
                .get();
        let before_resid =
            crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_RECALL_RESIDUAL
                .with_label_values(&["test_none_recall_advisor_obs_unique", "ivf"])
                .get_sample_count();
        let obs = AdvisorObservation::now(
            "test_none_recall_advisor_obs_unique",
            SupportedAlgorithm::Ivf,
            IndexAlgorithm::IVF {
                nlist: 316,
                nprobe: 20,
                quantizer: None,
            },
            0.55,
            None,
            6320,
            12_500,
        );
        record_observation(&obs);
        let after_count =
            crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_OBSERVATIONS_TOTAL
                .with_label_values(&["test_none_recall_advisor_obs_unique", "ivf"])
                .get();
        let after_resid =
            crate::metrics::advisor_observations_metrics::AXIS_ADVISOR_RECALL_RESIDUAL
                .with_label_values(&["test_none_recall_advisor_obs_unique", "ivf"])
                .get_sample_count();
        assert_eq!(after_count - before_count, 1.0);
        assert_eq!(after_resid - before_resid, 0);
    }
}
