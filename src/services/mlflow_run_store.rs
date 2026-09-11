//! Substrate-backed RunStore (TD-MLOPS-1 slice 1).
//!
//! Experiments, runs, params, metrics and tags persist as documents in ONE
//! tenant-scoped collection in the document substrate — the single storage
//! spine (mandate 18a), never a private metadata database. Structural tenant
//! isolation comes from the scoped collection key (constructed once per
//! store); record payloads never carry tenant identity.
//!
//! Document layout (one collection per tenant):
//! * `meta`              — `{next_experiment_id}`
//! * `exp-{id}`          — experiment record (serde JSON in `payload` +
//!   indexed `name` / `stage` fields)
//! * `run-{run_id}`      — run record (payload + indexed `experiment_id` /
//!   `stage` / per-run append counters)
//! * `mtr-{run}-{seq}`   — one append-only metric sample (indexed `run_id`,
//!   `key`, `seq`) — history is a filtered query
//! * `ds-{run}-{n}`      — dataset lineage input (indexed `run_id`)
//!
//! Mutations serialize under one process mutex: tracking is low-frequency
//! control-plane traffic, and the lock substitutes for per-document
//! optimistic concurrency in this slice. The seq counters live on the run
//! document so ordering survives process restarts.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result};
use proximadb_catalog::run_store::{
    ExperimentRecord, ExperimentStage, MetricAppend, MetricPoint, RunDatasetInput, RunLifecycle,
    RunRecord, RunStatus, RunStore, RunStoreError,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaTree, ProximaTreeNode};
use tokio::sync::Mutex;

use crate::storage::document::service::scoped_document_collection;
use crate::storage::document::{DocumentRecord, DocumentService};

const COLLECTION: &str = "mlflow_tracking_system";

fn lock_unpoisoned<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn payload_json(record: &DocumentRecord) -> Result<&str> {
    match record.props.get("payload") {
        Some(ProximaTreeNode::Value(ProximaValue::String(json))) => Ok(json),
        other => Err(anyhow::anyhow!(
            "tracking document '{}' has no string payload ({other:?})",
            record.id
        )),
    }
}

/// Process-global mutation locks keyed by collection: two `for_tenant`
/// instances over the same (service, tenant) must serialize their RMW
/// cycles against EACH OTHER, not just themselves.
fn mutation_lock_for(collection: &str) -> Arc<Mutex<()>> {
    static LOCKS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Arc<Mutex<()>>>>,
    > = std::sync::OnceLock::new();
    let registry = LOCKS.get_or_init(Default::default);
    let mut guard = lock_unpoisoned(registry);
    guard
        .entry(collection.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Local-filesystem [`ArtifactRepository`] (tenant-scoped at construction):
/// files under `<root>/...` with traversal-checked, verbatim path segments.
pub struct LocalFsArtifacts {
    root: std::path::PathBuf,
}

impl LocalFsArtifacts {
    pub fn new(root: std::path::PathBuf) -> Self {
        Self { root }
    }

    fn resolve(&self, path: &str) -> Result<std::path::PathBuf, String> {
        let mut target = self.root.clone();
        for segment in path.split('/') {
            if segment.is_empty() || segment == "." {
                continue;
            }
            if segment == ".." || segment.contains('\\') || segment.contains('\0') {
                return Err(format!("invalid artifact path segment '{segment}'"));
            }
            target = target.join(segment);
        }
        Ok(target)
    }
}

#[async_trait::async_trait]
impl proximadb_catalog::run_store::ArtifactRepository for LocalFsArtifacts {
    async fn put(&self, path: &str, bytes: &[u8]) -> Result<(), String> {
        let target = self.resolve(path)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create artifact dir: {e}"))?;
        }
        std::fs::write(&target, bytes).map_err(|e| format!("write artifact: {e}"))
    }

    async fn get(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        let target = self.resolve(path)?;
        if !target.is_file() {
            return Ok(None);
        }
        std::fs::read(&target)
            .map(Some)
            .map_err(|e| format!("read artifact: {e}"))
    }

    async fn list(
        &self,
        path: &str,
    ) -> Result<Vec<proximadb_catalog::run_store::ArtifactEntry>, String> {
        let target = self.resolve(path)?;
        if !tokio::fs::metadata(&target)
            .await
            .map(|m| m.is_dir())
            .unwrap_or(false)
        {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        let mut read = tokio::fs::read_dir(&target)
            .await
            .map_err(|e| format!("read dir: {e}"))?;
        while let Some(entry) = read
            .next_entry()
            .await
            .map_err(|e| format!("iterate: {e}"))?
        {
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let size = if is_dir {
                0
            } else {
                entry.metadata().await.map(|m| m.len()).unwrap_or(0)
            };
            entries.push(proximadb_catalog::run_store::ArtifactEntry {
                path: entry.file_name().to_string_lossy().to_string(),
                is_dir,
                size_bytes: size,
            });
        }
        Ok(entries)
    }

    async fn delete(&self, path: &str) -> Result<(), String> {
        let target = self.resolve(path)?;
        if tokio::fs::metadata(&target)
            .await
            .map(|m| m.is_dir())
            .unwrap_or(false)
        {
            tokio::fs::remove_dir_all(&target)
                .await
                .map_err(|e| format!("delete artifacts: {e}"))
        } else if target.exists() {
            tokio::fs::remove_file(&target)
                .await
                .map_err(|e| format!("delete artifact: {e}"))
        } else {
            Ok(())
        }
    }
}

/// Production [`RunStoreFactory`](proximadb_catalog::run_store::RunStoreFactory):
/// builds a tenant-scoped store over the shared document substrate.
pub struct SubstrateRunStoreFactory {
    document: Arc<DocumentService>,
}

impl SubstrateRunStoreFactory {
    pub fn new(document: Arc<DocumentService>) -> Self {
        Self { document }
    }
}

impl proximadb_catalog::run_store::RunStoreFactory for SubstrateRunStoreFactory {
    fn store_for(&self, tenant_id: &str) -> Result<std::sync::Arc<dyn RunStore>, RunStoreError> {
        let store =
            SubstrateRunStore::for_tenant(self.document.clone(), tenant_id).map_err(|e| {
                RunStoreError::Internal {
                    message: format!("tenant '{tenant_id}' cannot be scoped: {e:#}"),
                }
            })?;
        Ok(std::sync::Arc::new(store))
    }
}

pub struct SubstrateRunStore {
    document: Arc<DocumentService>,
    collection: String,
    mutation_lock: Arc<Mutex<()>>,
}

impl SubstrateRunStore {
    /// Tenant-scoped construction: the collection key embeds the tenant once,
    /// structurally — mirrors `GraphOperationsService::for_tenant`.
    pub fn for_tenant(document: Arc<DocumentService>, tenant: &str) -> Result<Self> {
        let collection = scoped_document_collection(tenant, COLLECTION)
            .context("scope mlflow tracking collection to tenant")?;
        let mutation_lock = mutation_lock_for(&collection);
        Ok(Self {
            collection,
            document,
            mutation_lock,
        })
    }

    async fn ensure(&self) -> Result<()> {
        self.document
            .ensure_or_create_collection(&self.collection)
            .await?;
        Ok(())
    }

    fn err(e: anyhow::Error) -> RunStoreError {
        RunStoreError::Internal {
            message: e.to_string(),
        }
    }

    async fn put_payload(
        &self,
        id: &str,
        index_fields: &[(&str, ProximaValue)],
        payload: &impl serde::Serialize,
    ) -> Result<()> {
        self.put_payload_fields(
            id,
            index_fields
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            payload,
        )
        .await
    }

    async fn put_payload_fields(
        &self,
        id: &str,
        fields: Vec<(String, ProximaValue)>,
        payload: &impl serde::Serialize,
    ) -> Result<()> {
        let mut tree: ProximaTree = HashMap::new();
        for (key, value) in fields {
            tree.insert(key, ProximaTreeNode::Value(value));
        }
        tree.insert(
            "payload".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(serde_json::to_string(payload)?)),
        );
        let record = DocumentRecord::from_tree(
            id.to_string(),
            tree,
            self.collection.clone(),
            None,
            Some("mlflow_tracking_system".to_string()),
        );
        self.document
            .insert_document_record(&self.collection, record)
            .await?;
        Ok(())
    }

    async fn get_payload<T: serde::de::DeserializeOwned>(&self, id: &str) -> Result<Option<T>> {
        // Reads by a tenant whose collection was never written must look
        // EMPTY — without materializing anything (a read must not create
        // catalog/billing state; MLflow garbage-probes must not provision).
        if self
            .document
            .get_collection(&self.collection)
            .await
            .map_err(Self::err)?
            .is_none()
        {
            return Ok(None);
        }
        let Some(record) = self
            .document
            .get_document(&self.collection, id, None)
            .await?
        else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(payload_json(&record)?)?))
    }

    async fn next_experiment_id(&self) -> Result<u64> {
        let current: Option<u64> = self.get_payload("meta").await?;
        let next = current.map(|n| n + 1).unwrap_or(0);
        self.put_payload("meta", &[], &next).await?;
        Ok(next)
    }

    async fn experiments(&self) -> Result<Vec<ExperimentRecord>> {
        // Slice-1 scale (per-tenant experiment counts are small): full scan
        // of the kind via list-by-prefix is not exposed by the seam, so
        // enumerate by walking the indexed query below.
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter_of(&[eq_cond("kind", "experiment")])),
            ..Default::default()
        };
        let result = self
            .document
            .query_documents(&self.collection, params)
            .await?;
        let mut records = Vec::new();
        for doc in result.documents {
            let record: ExperimentRecord = serde_json::from_str(payload_json(&doc)?)
                .with_context(|| format!("corrupt experiment doc '{}'", doc.id))?;
            records.push(record);
        }
        records.sort_by_key(|r| r.experiment_id);
        Ok(records)
    }

    async fn runs_of(&self, experiment_id: u64) -> Result<Vec<RunRecord>> {
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter_of(&[
                eq_cond("kind", "run"),
                crate::proto::proximadb_v1::DocFilterCondition {
                    path: "experiment_id".to_string(),
                    operator: crate::proto::proximadb_v1::DocFilterOperator::Eq as i32,
                    value: Some(crate::proto::proximadb_v1::SqlValue {
                        value: Some(crate::proto::proximadb_v1::sql_value::Value::Int64Value(
                            experiment_id as i64,
                        )),
                    }),
                    values: vec![],
                },
            ])),
            ..Default::default()
        };
        let result = self
            .document
            .query_documents(&self.collection, params)
            .await?;
        let mut records = Vec::new();
        for doc in result.documents {
            let record: RunRecord = serde_json::from_str(payload_json(&doc)?)
                .with_context(|| format!("corrupt run doc '{}'", doc.id))?;
            records.push(record);
        }
        records.sort_by_key(|r| r.start_time_ms);
        Ok(records)
    }
}

// --- indexed-field filters (JSONPath filter DSL over the indexed fields) ---

fn eq_cond(path: &str, value: &str) -> crate::proto::proximadb_v1::DocFilterCondition {
    crate::proto::proximadb_v1::DocFilterCondition {
        path: path.to_string(),
        operator: crate::proto::proximadb_v1::DocFilterOperator::Eq as i32,
        value: Some(crate::proto::proximadb_v1::SqlValue {
            value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(
                value.to_string(),
            )),
        }),
        values: vec![],
    }
}

fn filter_of(
    conditions: &[crate::proto::proximadb_v1::DocFilterCondition],
) -> crate::proto::proximadb_v1::DocumentFilter {
    crate::proto::proximadb_v1::DocumentFilter {
        conditions: conditions.to_vec(),
        ..Default::default()
    }
}

#[async_trait::async_trait]
impl RunStore for SubstrateRunStore {
    async fn create_experiment(
        &self,
        name: &str,
        artifact_location: Option<&str>,
        tags: BTreeMap<String, String>,
    ) -> Result<ExperimentRecord, RunStoreError> {
        if name.is_empty() {
            return Err(RunStoreError::Empty { field: "name" });
        }
        let _guard = self.mutation_lock.lock().await;
        self.ensure()
            .await
            .map_err(|e| Self::err(e.context("ensure collection")))?;
        if self
            .experiments()
            .await
            .map_err(Self::err)?
            .iter()
            .any(|e| e.name == name)
        {
            return Err(RunStoreError::ExperimentNameConflict {
                name: name.to_string(),
            });
        }
        let id = self.next_experiment_id().await.map_err(Self::err)?;
        let now = chrono::Utc::now().timestamp_millis();
        let record = ExperimentRecord {
            experiment_id: id,
            name: name.to_string(),
            artifact_location: artifact_location.map(str::to_string),
            tags,
            stage: ExperimentStage::Active,
            creation_time_ms: now,
            last_update_time_ms: now,
        };
        self.put_payload(
            &format!("exp-{id}"),
            &[
                ("kind", ProximaValue::String("experiment".to_string())),
                ("name", ProximaValue::String(record.name.clone())),
                (
                    "stage",
                    ProximaValue::String(serde_json::to_string(&record.stage).unwrap_or_default()),
                ),
            ],
            &record,
        )
        .await
        .map_err(Self::err)?;
        Ok(record)
    }

    async fn get_experiment(&self, experiment_id: u64) -> Result<ExperimentRecord, RunStoreError> {
        self.get_payload(&format!("exp-{experiment_id}"))
            .await
            .map_err(Self::err)?
            .ok_or(RunStoreError::UnknownExperiment { experiment_id })
    }

    async fn list_experiments(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<ExperimentRecord>, RunStoreError> {
        if self
            .document
            .get_collection(&self.collection)
            .await
            .map_err(Self::err)?
            .is_none()
        {
            return Ok(Vec::new());
        }
        Ok(self
            .experiments()
            .await
            .map_err(Self::err)?
            .into_iter()
            .filter(|e| include_deleted || e.stage == ExperimentStage::Active)
            .collect())
    }

    async fn delete_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut record = self.get_experiment(experiment_id).await?;
        record.stage = ExperimentStage::Deleted;
        record.last_update_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_payload(
            &format!("exp-{experiment_id}"),
            &[
                ("kind", ProximaValue::String("experiment".to_string())),
                ("name", ProximaValue::String(record.name.clone())),
                (
                    "stage",
                    ProximaValue::String(serde_json::to_string(&record.stage).unwrap_or_default()),
                ),
            ],
            &record,
        )
        .await
        .map_err(Self::err)
    }

    async fn restore_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut record = self.get_experiment(experiment_id).await?;
        record.stage = ExperimentStage::Active;
        record.last_update_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_payload(
            &format!("exp-{experiment_id}"),
            &[
                ("kind", ProximaValue::String("experiment".to_string())),
                ("name", ProximaValue::String(record.name.clone())),
                (
                    "stage",
                    ProximaValue::String(serde_json::to_string(&record.stage).unwrap_or_default()),
                ),
            ],
            &record,
        )
        .await
        .map_err(Self::err)
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
        let _guard = self.mutation_lock.lock().await;
        if self
            .document
            .get_collection(&self.collection)
            .await
            .map_err(Self::err)?
            .is_none()
        {
            return Err(RunStoreError::UnknownExperiment { experiment_id });
        }
        let experiment = self.get_experiment(experiment_id).await?;
        if experiment.stage == ExperimentStage::Deleted {
            return Err(RunStoreError::ExperimentDeleted { experiment_id });
        }
        if self
            .get_payload::<RunRecord>(&format!("run-{run_id}"))
            .await
            .map_err(Self::err)?
            .is_some()
        {
            return Err(RunStoreError::RunIdConflict {
                run_id: run_id.to_string(),
            });
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
        self.put_run(&record, &BTreeMap::new())
            .await
            .map_err(Self::err)?;
        Ok(record)
    }

    async fn get_run(&self, run_id: &str) -> Result<RunRecord, RunStoreError> {
        self.get_payload(&format!("run-{run_id}"))
            .await
            .map_err(Self::err)?
            .ok_or_else(|| RunStoreError::UnknownRun {
                run_id: run_id.to_string(),
            })
    }

    async fn list_runs(
        &self,
        experiment_id: u64,
        include_deleted: bool,
    ) -> Result<Vec<RunRecord>, RunStoreError> {
        self.ensure()
            .await
            .map_err(|e| Self::err(e.context("ensure collection")))?;
        self.get_experiment(experiment_id).await?;
        Ok(self
            .runs_of(experiment_id)
            .await
            .map_err(Self::err)?
            .into_iter()
            .filter(|r| include_deleted || r.lifecycle != RunLifecycle::Deleted)
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
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.status = status;
        run.end_time_ms = Some(end_time_ms);
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn reopen_run(&self, run_id: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.status = RunStatus::Running;
        run.end_time_ms = None;
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn delete_run(&self, run_id: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.lifecycle = RunLifecycle::Deleted;
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn restore_run(&self, run_id: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.lifecycle = RunLifecycle::Active;
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn log_param(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        if run.status.is_terminal() {
            return Err(RunStoreError::RunFinished {
                run_id: run_id.to_string(),
            });
        }
        match run.params.get(key) {
            Some(existing) if existing == value => Ok(()),
            Some(_) => Err(RunStoreError::ParamImmutable {
                key: key.to_string(),
                run_id: run_id.to_string(),
            }),
            None => {
                run.params.insert(key.to_string(), value.to_string());
                self.put_run(&run, &counters).await.map_err(Self::err)
            }
        }
    }

    async fn log_metric(
        &self,
        run_id: &str,
        point: MetricPoint,
    ) -> Result<MetricAppend, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, mut counters) = self.run_with_counters(run_id).await?;
        if run.status.is_terminal() {
            return Err(RunStoreError::RunFinished {
                run_id: run_id.to_string(),
            });
        }
        // seq lives in the run DOC's ctr_* fields (never the tags map); the
        // history point is its own append-only doc.
        let counter_key = counter_key(&point.key);
        let next_seq = counters.get(&counter_key).copied().unwrap_or(0) + 1;
        counters.insert(counter_key, next_seq);
        let advances_projection = run.latest_metrics.get(&point.key).is_none_or(|current| {
            point.timestamp_ms > current.timestamp_ms
                || (point.timestamp_ms == current.timestamp_ms && point.value > current.value)
        });
        if advances_projection {
            run.latest_metrics.insert(point.key.clone(), point.clone());
        }
        self.put_run(&run, &counters).await.map_err(Self::err)?;
        self.put_payload(
            &format!("mtr-{run_id}-{next_seq}"),
            &[
                ("kind", ProximaValue::String("metric".to_string())),
                ("run_id", ProximaValue::String(run_id.to_string())),
                ("key", ProximaValue::String(point.key.clone())),
                ("seq", ProximaValue::Int64(next_seq as i64)),
            ],
            &point,
        )
        .await
        .map_err(Self::err)?;
        Ok(MetricAppend {
            history_len: next_seq,
        })
    }

    async fn metric_history(
        &self,
        run_id: &str,
        key: &str,
    ) -> Result<Vec<MetricPoint>, RunStoreError> {
        self.get_run(run_id).await?;
        let filter = filter_of(&[
            eq_cond("kind", "metric"),
            eq_cond("run_id", run_id),
            eq_cond("key", key),
        ]);
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter),
            ..Default::default()
        };
        let result = self
            .document
            .query_documents(&self.collection, params)
            .await
            .map_err(Self::err)?;
        let mut points: Vec<(u64, MetricPoint)> = Vec::new();
        for doc in result.documents {
            let seq = match doc.props.get("seq") {
                Some(ProximaTreeNode::Value(ProximaValue::Int64(s))) => *s as u64,
                other => {
                    return Err(Self::err(anyhow::anyhow!(
                        "metric doc '{}' missing its seq field ({other:?}) — cannot order history",
                        doc.id
                    )));
                }
            };
            let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = doc.props.get("payload")
            else {
                return Err(Self::err(anyhow::anyhow!(
                    "metric doc '{}' has no payload",
                    doc.id
                )));
            };
            let point: MetricPoint = serde_json::from_str(json)
                .with_context(|| format!("corrupt metric doc '{}'", doc.id))
                .map_err(Self::err)?;
            points.push((seq, point));
        }
        points.sort_by_key(|(seq, _)| *seq);
        Ok(points.into_iter().map(|(_, p)| p).collect())
    }

    async fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.tags.insert(key.to_string(), value.to_string());
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn delete_tag(&self, run_id: &str, key: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.tags.remove(key);
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn log_dataset_input(
        &self,
        run_id: &str,
        input: RunDatasetInput,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (run, mut counters) = self.run_with_counters(run_id).await?;
        let n = counters.get("ds").copied().unwrap_or(0) + 1;
        counters.insert("ds".to_string(), n);
        self.put_run(&run, &counters).await.map_err(Self::err)?;
        self.put_payload(
            &format!("ds-{run_id}-{n}"),
            &[
                ("kind", ProximaValue::String("dataset".to_string())),
                ("run_id", ProximaValue::String(run_id.to_string())),
            ],
            &input,
        )
        .await
        .map_err(Self::err)
    }

    async fn dataset_inputs(&self, run_id: &str) -> Result<Vec<RunDatasetInput>, RunStoreError> {
        self.get_run(run_id).await?;
        let filter = filter_of(&[eq_cond("kind", "dataset"), eq_cond("run_id", run_id)]);
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter),
            ..Default::default()
        };
        let result = self
            .document
            .query_documents(&self.collection, params)
            .await
            .map_err(Self::err)?;
        let mut inputs = Vec::new();
        for doc in result.documents {
            let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = doc.props.get("payload")
            else {
                return Err(Self::err(anyhow::anyhow!(
                    "dataset doc '{}' has no payload",
                    doc.id
                )));
            };
            let input: RunDatasetInput = serde_json::from_str(json)
                .with_context(|| format!("corrupt dataset doc '{}'", doc.id))
                .map_err(Self::err)?;
            inputs.push(input);
        }
        Ok(inputs)
    }
}

impl SubstrateRunStore {
    /// Persist a run plus its append counters as SEPARATE document fields
    /// (`ctr_<name>`). Counters must never ride the user-writable tags map:
    /// a client `delete_tag`/`set_tag` on a counter-looking key must not be
    /// able to rewind a counter and silently overwrite history points
    /// (review round 1, MAJOR).
    async fn put_run(&self, run: &RunRecord, counters: &BTreeMap<String, u64>) -> Result<()> {
        let mut fields: Vec<(String, ProximaValue)> = vec![
            ("kind".to_string(), ProximaValue::String("run".to_string())),
            (
                "experiment_id".to_string(),
                ProximaValue::Int64(run.experiment_id as i64),
            ),
            (
                "lifecycle".to_string(),
                ProximaValue::String(serde_json::to_string(&run.lifecycle).unwrap_or_default()),
            ),
        ];
        fields.extend(
            counters
                .iter()
                .map(|(name, value)| (format!("ctr_{name}"), ProximaValue::Int64(*value as i64))),
        );
        self.put_payload_fields(&format!("run-{}", run.run_id), fields, run)
            .await
    }

    /// Read the run record TOGETHER with its durable counters.
    /// Absent runs surface as the typed [`RunStoreError::UnknownRun`] —
    /// mutations on a nonexistent id must NOT collapse into Internal (the
    /// MLflow adapter maps UnknownRun to RESOURCE_DOES_NOT_EXIST).
    async fn run_with_counters(
        &self,
        run_id: &str,
    ) -> Result<(RunRecord, BTreeMap<String, u64>), RunStoreError> {
        let record = self
            .document
            .get_document(&self.collection, &format!("run-{run_id}"), None)
            .await
            .map_err(Self::err)?
            .ok_or_else(|| RunStoreError::UnknownRun {
                run_id: run_id.to_string(),
            })?;
        let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = record.props.get("payload")
        else {
            return Err(Self::err(anyhow::anyhow!(
                "run document '{run_id}' has no payload"
            )));
        };
        let run: RunRecord = serde_json::from_str(json)
            .with_context(|| format!("corrupt run doc '{}'", record.id))
            .map_err(Self::err)?;
        let mut counters = BTreeMap::new();
        for (key, node) in &record.props {
            if let (Some(name), ProximaTreeNode::Value(ProximaValue::Int64(v))) =
                (key.strip_prefix("ctr_"), node)
            {
                counters.insert(name.to_string(), *v as u64);
            }
        }
        Ok((run, counters))
    }
}

/// Lossless counter-key encoding: every byte outside `[A-Za-z0-9]` becomes
/// `%xx`, so distinct metric keys (e.g. `a/b` vs `a-b`) can never share a
/// counter. Metric counters are namespaced `m_<key>` so a metric literally
/// named `ds` cannot collide with the dataset counter.
fn counter_key(key: &str) -> String {
    let mut out = String::from("m_");
    for b in key.as_bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(*b as char);
        } else {
            out.push_str(&format!("%{b:02x}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::lock_unpoisoned;
    use std::sync::{Arc, Mutex};

    #[test]
    fn poisoned_registry_lock_is_recovered_without_panicking() {
        let registry = Arc::new(Mutex::new(Vec::<u8>::new()));
        let poison_target = Arc::clone(&registry);
        let _ = std::thread::spawn(move || {
            let _guard = poison_target
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            panic!("poison registry for recovery test");
        })
        .join();

        lock_unpoisoned(&registry).push(7);
        assert_eq!(*lock_unpoisoned(&registry), vec![7]);
    }
}
