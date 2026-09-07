//! MLflow-compatible tracking store port (TD-MLOPS-1 slice 1).
//!
//! Experiments, runs, params, metrics and tags are tenant-scoped substrate
//! records — NOT catalog assets and NOT a private metadata database. This
//! module defines the port (types + trait); the first implementation persists
//! through the platform's storage seams and lives beside its wiring. The
//! MLflow wire adapter (slice 2) is a codec over this port only.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// MLflow experiment lifecycle marker. Deletes are soft — deleted experiments
/// are hidden from search and default listing but restorable and directly
/// gettable, matching MLflow client expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStage {
    Active,
    Deleted,
}

/// MLflow run lifecycle stage — orthogonal to [`RunStatus`] exactly as
/// MLflow models it (`lifecycle_stage` x `status`): deleting a finished run
/// and restoring it must yield a finished run again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunLifecycle {
    Active,
    Deleted,
}

/// MLflow run status. `Running -> Finished` is one-way; finished runs
/// reject further param/metric writes (params are immutable even while
/// running).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Finished,
    Failed,
    Killed,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, RunStatus::Running)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub experiment_id: u64,
    pub name: String,
    pub artifact_location: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub stage: ExperimentStage,
    pub creation_time_ms: i64,
    pub last_update_time_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub experiment_id: u64,
    pub run_name: Option<String>,
    pub user_id: Option<String>,
    pub lifecycle: RunLifecycle,
    pub status: RunStatus,
    pub start_time_ms: i64,
    pub end_time_ms: Option<i64>,
    /// Immutable after first write for a key (MLflow param semantics).
    pub params: BTreeMap<String, String>,
    /// Latest value per metric key; full history is append-only.
    pub latest_metrics: BTreeMap<String, MetricPoint>,
    pub tags: BTreeMap<String, String>,
}

/// One append-only metric sample. History preserves insertion (timestamp,
/// step) order per key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub key: String,
    /// Proto3-JSON convention both ways: serde_json writes non-finite f64
    /// as `null` (unreadable on the way back), so non-finite values
    /// serialize as the strings "NaN" / "Infinity" / "-Infinity" and the
    /// deserializer accepts numbers or those strings. Without this, one
    /// diverged-loss metric corrupts the whole run document.
    #[serde(with = "metric_value_lossy")]
    pub value: f64,
    pub timestamp_ms: i64,
    pub step: i64,
}

mod metric_value_lossy {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_nan() {
            "NaN".serialize(s)
        } else if *v == f64::INFINITY {
            "Infinity".serialize(s)
        } else if *v == f64::NEG_INFINITY {
            "-Infinity".serialize(s)
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        match raw {
            serde_json::Value::Number(n) => n
                .as_f64()
                .ok_or_else(|| serde::de::Error::custom("metric value out of f64 range")),
            serde_json::Value::String(s) => match s.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                other => other.parse::<f64>().map_err(|_| {
                    serde::de::Error::custom(format!("invalid metric value '{other}'"))
                }),
            },
            other => Err(serde::de::Error::custom(format!(
                "metric value must be a number or numeric string, got {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunDatasetInput {
    /// MLflow dataset name — an alias, never a resolution authority.
    pub dataset_name: String,
    /// Content digest pinning the exact input (reproducibility contract).
    pub digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RunStoreError {
    #[error("experiment {experiment_id} not found")]
    UnknownExperiment { experiment_id: u64 },
    #[error("experiment name '{name}' already exists")]
    ExperimentNameConflict { name: String },
    #[error("run '{run_id}' not found")]
    UnknownRun { run_id: String },
    #[error("run '{run_id}' already exists")]
    RunIdConflict { run_id: String },
    #[error("param '{key}' is immutable once logged on run '{run_id}'")]
    ParamImmutable { key: String, run_id: String },
    #[error("run '{run_id}' is finished; metric/param writes are rejected")]
    RunFinished { run_id: String },
    #[error("finish_run requires a terminal status, got Running")]
    NotTerminal,
    #[error("experiment {experiment_id} is deleted; run creation is rejected")]
    ExperimentDeleted { experiment_id: u64 },
    #[error("names and ids must be non-empty")]
    Empty { field: &'static str },
    #[error("tracking store internal error: {message}")]
    Internal { message: String },
}

/// Result of a metric append — the caller decides whether the latest-value
/// projection advanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricAppend {
    pub history_len: u64,
}

/// Tenant-scoped experiment/run tracking port. Implementations MUST enforce
/// tenant isolation structurally (the caller threads tenant identity into the
/// store's construction or per-call context — never inside record payloads).
#[async_trait::async_trait]
pub trait RunStore: Send + Sync {
    /// Create an experiment with a caller-chosen unique name.
    async fn create_experiment(
        &self,
        name: &str,
        artifact_location: Option<&str>,
        tags: BTreeMap<String, String>,
    ) -> Result<ExperimentRecord, RunStoreError>;

    async fn get_experiment(&self, experiment_id: u64) -> Result<ExperimentRecord, RunStoreError>;

    /// Deterministic listing (by id) of active experiments unless
    /// `include_deleted` is set.
    async fn list_experiments(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<ExperimentRecord>, RunStoreError>;

    /// Soft-delete; a deleted experiment can be restored.
    async fn delete_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError>;

    async fn restore_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError>;

    /// Create a run under an experiment with a server-unique caller-supplied
    /// id (MLflow allows client-chosen run ids; conflicts fail loudly).
    async fn create_run(
        &self,
        experiment_id: u64,
        run_id: &str,
        run_name: Option<&str>,
        user_id: Option<&str>,
        tags: BTreeMap<String, String>,
        start_time_ms: i64,
    ) -> Result<RunRecord, RunStoreError>;

    async fn get_run(&self, run_id: &str) -> Result<RunRecord, RunStoreError>;

    /// Runs of one experiment (active only unless `include_deleted`),
    /// ordered by creation.
    async fn list_runs(
        &self,
        experiment_id: u64,
        include_deleted: bool,
    ) -> Result<Vec<RunRecord>, RunStoreError>;

    /// Terminal transition with the MLflow status (Finished/Failed/Killed);
    /// any terminal status freezes param/metric writes.
    async fn finish_run(
        &self,
        run_id: &str,
        status: RunStatus,
        end_time_ms: i64,
    ) -> Result<(), RunStoreError>;

    /// Reopen a terminal run (MLflow UpdateRun -> RUNNING).
    async fn reopen_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    /// Soft-delete a run (hides from listing; direct get still works).
    async fn delete_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    async fn restore_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    /// Log one param. Second write with a DIFFERENT value is an error; the
    /// same value is idempotent (MLflow log-batch retries must succeed).
    async fn log_param(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError>;

    /// Append a metric sample to the per-key history and advance the
    /// latest-value projection. Rejected on finished runs.
    async fn log_metric(
        &self,
        run_id: &str,
        point: MetricPoint,
    ) -> Result<MetricAppend, RunStoreError>;

    /// Full per-key history in append order.
    async fn metric_history(
        &self,
        run_id: &str,
        key: &str,
    ) -> Result<Vec<MetricPoint>, RunStoreError>;

    /// Set (or overwrite) / delete a tag.
    async fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError>;

    async fn delete_tag(&self, run_id: &str, key: &str) -> Result<(), RunStoreError>;

    /// Record a typed dataset input (name + digest lineage).
    async fn log_dataset_input(
        &self,
        run_id: &str,
        input: RunDatasetInput,
    ) -> Result<(), RunStoreError>;

    async fn dataset_inputs(&self, run_id: &str) -> Result<Vec<RunDatasetInput>, RunStoreError>;
}

/// Executable port semantics shared by every implementation.
pub mod conformance_tests {
    use super::*;

    /// Reference in-memory implementation: executable semantics for the port.
    /// The substrate-backed store (TD-MLOPS-1 slice 1) must pass the same
    /// conformance battery; the MLflow wire adapter's unit tests use this
    /// double without a substrate.
    pub struct InMemoryRunStore {
        experiments: std::sync::Mutex<Vec<ExperimentRecord>>,
        runs: std::sync::Mutex<Vec<RunRecord>>,
        history: std::sync::Mutex<Vec<(String, MetricPoint)>>,
        datasets: std::sync::Mutex<Vec<(String, RunDatasetInput)>>,
    }

    impl Default for InMemoryRunStore {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InMemoryRunStore {
        pub fn new() -> Self {
            Self {
                experiments: std::sync::Mutex::new(Vec::new()),
                runs: std::sync::Mutex::new(Vec::new()),
                history: std::sync::Mutex::new(Vec::new()),
                datasets: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl RunStore for InMemoryRunStore {
        async fn create_experiment(
            &self,
            name: &str,
            artifact_location: Option<&str>,
            tags: BTreeMap<String, String>,
        ) -> Result<ExperimentRecord, RunStoreError> {
            if name.is_empty() {
                return Err(RunStoreError::Empty { field: "name" });
            }
            let mut guard = self.experiments.lock().unwrap();
            if guard.iter().any(|e| e.name == name) {
                return Err(RunStoreError::ExperimentNameConflict {
                    name: name.to_string(),
                });
            }
            let id = guard.len() as u64;
            let record = ExperimentRecord {
                experiment_id: id,
                name: name.to_string(),
                artifact_location: artifact_location.map(str::to_string),
                tags,
                stage: ExperimentStage::Active,
                creation_time_ms: 1_000,
                last_update_time_ms: 1_000,
            };
            guard.push(record.clone());
            Ok(record)
        }

        async fn get_experiment(
            &self,
            experiment_id: u64,
        ) -> Result<ExperimentRecord, RunStoreError> {
            self.experiments
                .lock()
                .unwrap()
                .iter()
                .find(|e| e.experiment_id == experiment_id)
                .cloned()
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })
        }

        async fn list_experiments(
            &self,
            include_deleted: bool,
        ) -> Result<Vec<ExperimentRecord>, RunStoreError> {
            Ok(self
                .experiments
                .lock()
                .unwrap()
                .iter()
                .filter(|e| include_deleted || e.stage == ExperimentStage::Active)
                .cloned()
                .collect())
        }

        async fn delete_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
            let mut guard = self.experiments.lock().unwrap();
            let record = guard
                .iter_mut()
                .find(|e| e.experiment_id == experiment_id)
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })?;
            record.stage = ExperimentStage::Deleted;
            Ok(())
        }

        async fn restore_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
            let mut guard = self.experiments.lock().unwrap();
            let record = guard
                .iter_mut()
                .find(|e| e.experiment_id == experiment_id)
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })?;
            record.stage = ExperimentStage::Active;
            Ok(())
        }

        async fn create_run(
            &self,
            experiment_id: u64,
            run_id: &str,
            run_name: Option<&str>,
            user_id: Option<&str>,
            tags: BTreeMap<String, String>,
            start_time_ms: i64,
        ) -> Result<RunRecord, RunStoreError> {
            if run_id.is_empty() {
                return Err(RunStoreError::Empty { field: "run_id" });
            }
            self.get_experiment(experiment_id).await?;
            {
                let runs = self.runs.lock().unwrap();
                if runs.iter().any(|r| r.run_id == run_id) {
                    return Err(RunStoreError::RunIdConflict {
                        run_id: run_id.to_string(),
                    });
                }
            }
            {
                let experiments = self.experiments.lock().unwrap();
                let stage = experiments
                    .iter()
                    .find(|e| e.experiment_id == experiment_id)
                    .map(|e| e.stage);
                if stage == Some(ExperimentStage::Deleted) {
                    return Err(RunStoreError::ExperimentDeleted { experiment_id });
                }
            }
            let record = RunRecord {
                run_id: run_id.to_string(),
                experiment_id,
                run_name: run_name.map(str::to_string),
                user_id: user_id.map(str::to_string),
                lifecycle: RunLifecycle::Active,
                status: RunStatus::Running,
                start_time_ms,
                end_time_ms: None,
                params: BTreeMap::new(),
                latest_metrics: BTreeMap::new(),
                tags,
            };
            self.runs.lock().unwrap().push(record.clone());
            Ok(record)
        }

        async fn get_run(&self, run_id: &str) -> Result<RunRecord, RunStoreError> {
            self.runs
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.run_id == run_id)
                .cloned()
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })
        }

        async fn list_runs(
            &self,
            experiment_id: u64,
            include_deleted: bool,
        ) -> Result<Vec<RunRecord>, RunStoreError> {
            self.get_experiment(experiment_id).await?;
            Ok(self
                .runs
                .lock()
                .unwrap()
                .iter()
                .filter(|r| {
                    r.experiment_id == experiment_id
                        && (include_deleted || r.lifecycle != RunLifecycle::Deleted)
                })
                .cloned()
                .collect())
        }

        async fn finish_run(
            &self,
            run_id: &str,
            status: RunStatus,
            end_time_ms: i64,
        ) -> Result<(), RunStoreError> {
            if !status.is_terminal() {
                return Err(RunStoreError::NotTerminal);
            }
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.status = status;
            run.end_time_ms = Some(end_time_ms);
            Ok(())
        }

        async fn reopen_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.status = RunStatus::Running;
            run.end_time_ms = None;
            Ok(())
        }

        async fn delete_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.lifecycle = RunLifecycle::Deleted;
            Ok(())
        }

        async fn restore_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.lifecycle = RunLifecycle::Active;
            Ok(())
        }

        async fn log_param(
            &self,
            run_id: &str,
            key: &str,
            value: &str,
        ) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            {
                let run = runs.iter().find(|r| r.run_id == run_id).ok_or_else(|| {
                    RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    }
                })?;
                if run.status.is_terminal() {
                    return Err(RunStoreError::RunFinished {
                        run_id: run_id.to_string(),
                    });
                }
            }
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            match run.params.get(key) {
                Some(existing) if existing == value => Ok(()),
                Some(_) => Err(RunStoreError::ParamImmutable {
                    key: key.to_string(),
                    run_id: run_id.to_string(),
                }),
                None => {
                    run.params.insert(key.to_string(), value.to_string());
                    Ok(())
                }
            }
        }

        async fn log_metric(
            &self,
            run_id: &str,
            point: MetricPoint,
        ) -> Result<MetricAppend, RunStoreError> {
            {
                let runs = self.runs.lock().unwrap();
                let run = runs.iter().find(|r| r.run_id == run_id).ok_or_else(|| {
                    RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    }
                })?;
                if run.status.is_terminal() {
                    return Err(RunStoreError::RunFinished {
                        run_id: run_id.to_string(),
                    });
                }
            }
            {
                let mut history = self.history.lock().unwrap();
                history.push((run_id.to_string(), point.clone()));
            }
            let history_len;
            {
                let mut runs = self.runs.lock().unwrap();
                let run = runs
                    .iter_mut()
                    .find(|r| r.run_id == run_id)
                    .ok_or_else(|| RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    })?;
                run.latest_metrics.insert(point.key.clone(), point.clone());
                history_len = self
                    .history
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(id, p)| id == run_id && p.key == point.key)
                    .count() as u64;
            }
            Ok(MetricAppend { history_len })
        }

        async fn metric_history(
            &self,
            run_id: &str,
            key: &str,
        ) -> Result<Vec<MetricPoint>, RunStoreError> {
            self.get_run(run_id).await?;
            Ok(self
                .history
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, p)| id == run_id && p.key == key)
                .map(|(_, p)| p.clone())
                .collect())
        }

        async fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.tags.insert(key.to_string(), value.to_string());
            Ok(())
        }

        async fn delete_tag(&self, run_id: &str, key: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.tags.remove(key);
            Ok(())
        }

        async fn log_dataset_input(
            &self,
            run_id: &str,
            input: RunDatasetInput,
        ) -> Result<(), RunStoreError> {
            self.get_run(run_id).await?;
            self.datasets
                .lock()
                .unwrap()
                .push((run_id.to_string(), input));
            Ok(())
        }

        async fn dataset_inputs(
            &self,
            run_id: &str,
        ) -> Result<Vec<RunDatasetInput>, RunStoreError> {
            self.get_run(run_id).await?;
            Ok(self
                .datasets
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == run_id)
                .map(|(_, input)| input.clone())
                .collect())
        }
    }

    /// Port conformance battery — substrate implementations re-run this
    /// against themselves (same semantics, different durability).
    pub async fn port_conformance<S: RunStore>(store: &S) {
        // Experiment lifecycle: create conflicts on duplicate names, soft
        // delete hides from listing, restore revives, direct get always works.
        let exp = store
            .create_experiment("classify-iris", None, BTreeMap::new())
            .await
            .expect("create experiment");
        assert_eq!(exp.name, "classify-iris");
        assert_eq!(exp.stage, ExperimentStage::Active);

        let dup = store
            .create_experiment("classify-iris", None, BTreeMap::new())
            .await
            .unwrap_err();
        assert_eq!(
            dup,
            RunStoreError::ExperimentNameConflict {
                name: "classify-iris".to_string()
            }
        );

        assert_eq!(store.list_experiments(false).await.unwrap().len(), 1);
        store.delete_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(store.list_experiments(false).await.unwrap().len(), 0);
        assert_eq!(store.list_experiments(true).await.unwrap().len(), 1);
        assert_eq!(
            store.get_experiment(exp.experiment_id).await.unwrap().stage,
            ExperimentStage::Deleted
        );
        store.restore_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(store.list_experiments(false).await.unwrap().len(), 1);

        // Run lifecycle.
        let run = store
            .create_run(
                exp.experiment_id,
                "run-0001",
                Some("baseline"),
                Some("tester"),
                BTreeMap::new(),
                1_000,
            )
            .await
            .expect("create run");
        assert_eq!(run.status, RunStatus::Running);

        let unknown_exp = store
            .create_run(999, "run-x", None, None, BTreeMap::new(), 0)
            .await
            .unwrap_err();
        assert_eq!(
            unknown_exp,
            RunStoreError::UnknownExperiment { experiment_id: 999 }
        );

        let dup_run = store
            .create_run(
                exp.experiment_id,
                "run-0001",
                None,
                None,
                BTreeMap::new(),
                0,
            )
            .await
            .unwrap_err();
        assert_eq!(
            dup_run,
            RunStoreError::RunIdConflict {
                run_id: "run-0001".to_string()
            }
        );

        // Params: immutable after first write, idempotent on same value.
        store.log_param("run-0001", "lr", "0.01").await.unwrap();
        store.log_param("run-0001", "lr", "0.01").await.unwrap();
        assert_eq!(
            store.log_param("run-0001", "lr", "0.02").await.unwrap_err(),
            RunStoreError::ParamImmutable {
                key: "lr".to_string(),
                run_id: "run-0001".to_string()
            }
        );

        // Metrics: append-only history in order, latest projection advances.
        let p1 = MetricPoint {
            key: "rmse".to_string(),
            value: 0.9,
            timestamp_ms: 1_000,
            step: 0,
        };
        let p2 = MetricPoint {
            key: "rmse".to_string(),
            value: 0.7,
            timestamp_ms: 2_000,
            step: 1,
        };
        store.log_metric("run-0001", p1.clone()).await.unwrap();
        let append = store.log_metric("run-0001", p2.clone()).await.unwrap();
        assert_eq!(append.history_len, 2);
        assert_eq!(
            store.metric_history("run-0001", "rmse").await.unwrap(),
            vec![p1.clone(), p2.clone()]
        );
        assert_eq!(
            store.get_run("run-0001").await.unwrap().latest_metrics["rmse"],
            p2
        );

        // Tags are mutable.
        store.set_tag("run-0001", "team", "search").await.unwrap();
        store.set_tag("run-0001", "team", "vector").await.unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().tags["team"],
            "vector"
        );
        store.delete_tag("run-0001", "team").await.unwrap();
        assert!(
            !store
                .get_run("run-0001")
                .await
                .unwrap()
                .tags
                .contains_key("team")
        );

        // Mutations on an ABSENT run are UnknownRun — never an internal
        // error (the wire adapter maps this to RESOURCE_DOES_NOT_EXIST).
        assert_eq!(
            store.set_tag("no-such-run", "k", "v").await.unwrap_err(),
            RunStoreError::UnknownRun {
                run_id: "no-such-run".to_string()
            }
        );
        assert_eq!(
            store
                .finish_run("no-such-run", RunStatus::Finished, 1)
                .await
                .unwrap_err(),
            RunStoreError::UnknownRun {
                run_id: "no-such-run".to_string()
            }
        );

        // Counter preservation across unrelated rewrites: tag writes must
        // not disturb the metric seq (a lost counter would make the NEXT
        // append overwrite an existing history doc).
        store.set_tag("run-0001", "unrelated", "x").await.unwrap();
        store.delete_tag("run-0001", "unrelated").await.unwrap();
        store
            .log_metric(
                "run-0001",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.8,
                    timestamp_ms: 3_000,
                    step: 3,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .metric_history("run-0001", "rmse")
                .await
                .unwrap()
                .len(),
            3,
            "seq must survive tag rewrites — history docs must never be overwritten"
        );

        // Dataset inputs are name+digest lineage records.
        let ds = RunDatasetInput {
            dataset_name: "sift1m".to_string(),
            digest: "sha256:abc".to_string(),
        };
        store
            .log_dataset_input("run-0001", ds.clone())
            .await
            .unwrap();
        assert_eq!(store.dataset_inputs("run-0001").await.unwrap(), vec![ds]);

        // Finish is one-way and freezes param writes too (port contract).
        store
            .finish_run("run-0001", RunStatus::Finished, 9_000)
            .await
            .unwrap();
        assert_eq!(
            store.log_param("run-0001", "late", "1").await.unwrap_err(),
            RunStoreError::RunFinished {
                run_id: "run-0001".to_string()
            }
        );

        // Delete -> restore must PRESERVE the finished status (MLflow
        // separates lifecycle_stage from status).
        store.delete_run("run-0001").await.unwrap();
        store.restore_run("run-0001").await.unwrap();
        let restored = store.get_run("run-0001").await.unwrap();
        assert_eq!(restored.lifecycle, RunLifecycle::Active);
        assert_eq!(restored.status, RunStatus::Finished);

        // Reopen (MLflow UpdateRun -> RUNNING): writes accepted again, end
        // time cleared; re-finish with a FAILED terminal status.
        store.reopen_run("run-0001").await.unwrap();
        let reopened = store.get_run("run-0001").await.unwrap();
        assert_eq!(reopened.status, RunStatus::Running);
        assert_eq!(reopened.end_time_ms, None);
        store
            .log_metric(
                "run-0001",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.05,
                    timestamp_ms: 9_600,
                    step: 4,
                },
            )
            .await
            .unwrap();
        store
            .finish_run("run-0001", RunStatus::Failed, 9_700)
            .await
            .unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().status,
            RunStatus::Failed
        );
        assert_eq!(
            store
                .finish_run("run-0001", RunStatus::Running, 9_800)
                .await
                .unwrap_err(),
            RunStoreError::NotTerminal
        );
        store.reopen_run("run-0001").await.unwrap();
        store
            .finish_run("run-0001", RunStatus::Finished, 9_000)
            .await
            .unwrap();
        assert_eq!(
            store
                .log_metric(
                    "run-0001",
                    MetricPoint {
                        key: "rmse".to_string(),
                        value: 0.1,
                        timestamp_ms: 9_500,
                        step: 2,
                    }
                )
                .await
                .unwrap_err(),
            RunStoreError::RunFinished {
                run_id: "run-0001".to_string()
            }
        );

        // Cross-run metric isolation: a second run logging the same key has
        // its own history; the first run's is untouched.
        store
            .create_run(
                exp.experiment_id,
                "run-0002",
                None,
                None,
                BTreeMap::new(),
                2_000,
            )
            .await
            .unwrap();
        store
            .log_metric(
                "run-0002",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.5,
                    timestamp_ms: 2_500,
                    step: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .metric_history("run-0001", "rmse")
                .await
                .unwrap()
                .len(),
            4,
            "run-0001 history: two early points + post-tag-rewrite + post-reopen appends"
        );
        assert_eq!(
            store
                .metric_history("run-0002", "rmse")
                .await
                .unwrap()
                .len(),
            1
        );

        // Run creation in a deleted experiment is rejected.
        store.delete_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(
            store
                .create_run(
                    exp.experiment_id,
                    "run-0003",
                    None,
                    None,
                    BTreeMap::new(),
                    0
                )
                .await
                .unwrap_err(),
            RunStoreError::ExperimentDeleted {
                experiment_id: exp.experiment_id
            }
        );
        store.restore_experiment(exp.experiment_id).await.unwrap();
        store.delete_run("run-0001").await.unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().status,
            RunStatus::Finished
        );
        assert_eq!(
            store.get_run("run-0001").await.unwrap().end_time_ms,
            Some(9_000)
        );

        // Listing after the delete above: run-0001 hidden (lifecycle=deleted),
        // run-0002 active; include_deleted shows both.
        assert_eq!(
            store
                .list_runs(exp.experiment_id, false)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .list_runs(exp.experiment_id, true)
                .await
                .unwrap()
                .len(),
            2
        );
    }
}

#[cfg(test)]
mod tests {
    use super::conformance_tests::{InMemoryRunStore, port_conformance};

    #[tokio::test]
    async fn in_memory_reference_passes_port_conformance() {
        let store = InMemoryRunStore::new();
        port_conformance(&store).await;
    }
}
