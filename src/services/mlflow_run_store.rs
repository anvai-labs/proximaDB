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
//! * `mtr-{run}-{key}-{seq}` — one append-only metric sample (indexed
//!   `run_id`, `key`, `seq`) — history is a filtered query
//! * `ds-{run}-{n}`      — dataset lineage input (indexed `run_id`)
//! * `trc-{trace_id}`    — MLflow 3.x trace record with embedded
//!   assessments (indexed `experiment_id` / `state`) — TD-MLOPS-2
//! * `lm-{model_id}`     — logged model (indexed `experiment_id` /
//!   `lifecycle` + per-model append counters `ctr_m_<key>`)
//! * `mtrm-{model}-{key}-{seq}` — one append-only MODEL metric sample
//!   (indexed `model_id`, `key`, `seq`). The `mtr-` (run) and `mtrm-` (model) id
//!   spaces are disjoint at byte 3 (`-` vs `m`) for ALL id values —
//!   collision-free by construction, independent of id formats.
//!
//! Mutations serialize under one process mutex: tracking is low-frequency
//! control-plane traffic, and the lock substitutes for per-document
//! optimistic concurrency in this slice. The seq counters live on the owner
//! document (run / logged model) so ordering survives process restarts.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result};
use proximadb_catalog::run_store::{
    AssessmentRecord, ExperimentRecord, ExperimentStage, LoggedModelRecord, LoggedModelStatus,
    MetricAppend, MetricPoint, ModelMetricPoint, ModelOutputRef, RunDatasetInput, RunLifecycle,
    RunRecord, RunStatus, RunStore, RunStoreError, TraceRecord, valid_trace_id,
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

    async fn traces_of(&self, experiment_id: u64) -> Result<Vec<TraceRecord>> {
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter_of(&[
                eq_cond("kind", "trace"),
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
            let record: TraceRecord = serde_json::from_str(payload_json(&doc)?)
                .with_context(|| format!("corrupt trace doc '{}'", doc.id))?;
            records.push(record);
        }
        records.sort_by(|a, b| {
            a.request_time_ms
                .cmp(&b.request_time_ms)
                .then_with(|| a.trace_id.cmp(&b.trace_id))
        });
        Ok(records)
    }

    async fn logged_models_of(&self, experiment_id: u64) -> Result<Vec<LoggedModelRecord>> {
        let params = crate::storage::document::DocumentQueryParams {
            filter: Some(filter_of(&[
                eq_cond("kind", "logged_model"),
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
            let record: LoggedModelRecord = serde_json::from_str(payload_json(&doc)?)
                .with_context(|| format!("corrupt logged-model doc '{}'", doc.id))?;
            records.push(record);
        }
        records.sort_by(|a, b| {
            a.creation_time_ms
                .cmp(&b.creation_time_ms)
                .then_with(|| a.model_id.cmp(&b.model_id))
        });
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
            model_inputs: Vec::new(),
            model_outputs: Vec::new(),
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
        // history point is its own append-only doc. The doc id embeds the
        // PER-KEY counter: seq alone is per-key (two keys both start at 1)
        // and the id-only-keyed upsert would silently overwrite the other
        // key's sample (review MAJOR-2). The encoded key contains no '-',
        // so {run}-{counter_key}-{seq} parses unambiguously from the right.
        let counter_key = counter_key(&point.key);
        let next_seq = counters.get(&counter_key).copied().unwrap_or(0) + 1;
        counters.insert(counter_key.clone(), next_seq);
        let advances_projection = run.latest_metrics.get(&point.key).is_none_or(|current| {
            point.timestamp_ms > current.timestamp_ms
                || (point.timestamp_ms == current.timestamp_ms && point.value > current.value)
        });
        if advances_projection {
            run.latest_metrics.insert(point.key.clone(), point.clone());
        }
        self.put_run(&run, &counters).await.map_err(Self::err)?;
        self.put_payload(
            &format!("mtr-{run_id}-{counter_key}-{next_seq}"),
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

    // --- Traces (TD-MLOPS-2) ---

    async fn start_trace(&self, trace: TraceRecord) -> Result<TraceRecord, RunStoreError> {
        if trace.trace_id.is_empty() {
            return Err(RunStoreError::Empty { field: "trace_id" });
        }
        if !valid_trace_id(&trace.trace_id) {
            return Err(RunStoreError::InvalidTraceId {
                trace_id: trace.trace_id,
            });
        }
        let _guard = self.mutation_lock.lock().await;
        if self
            .document
            .get_collection(&self.collection)
            .await
            .map_err(Self::err)?
            .is_none()
        {
            return Err(RunStoreError::UnknownExperiment {
                experiment_id: trace.experiment_id,
            });
        }
        let experiment = self.get_experiment(trace.experiment_id).await?;
        if experiment.stage == ExperimentStage::Deleted {
            return Err(RunStoreError::ExperimentDeleted {
                experiment_id: trace.experiment_id,
            });
        }
        if self
            .get_payload::<TraceRecord>(&format!("trc-{}", trace.trace_id))
            .await
            .map_err(Self::err)?
            .is_some()
        {
            return Err(RunStoreError::TraceIdConflict {
                trace_id: trace.trace_id,
            });
        }
        self.put_trace(&trace).await.map_err(Self::err)?;
        Ok(trace)
    }

    async fn get_trace(&self, trace_id: &str) -> Result<TraceRecord, RunStoreError> {
        self.get_payload(&format!("trc-{trace_id}"))
            .await
            .map_err(Self::err)?
            .ok_or_else(|| RunStoreError::UnknownTrace {
                trace_id: trace_id.to_string(),
            })
    }

    async fn list_traces(&self, experiment_id: u64) -> Result<Vec<TraceRecord>, RunStoreError> {
        self.ensure()
            .await
            .map_err(|e| Self::err(e.context("ensure collection")))?;
        self.get_experiment(experiment_id).await?;
        self.traces_of(experiment_id).await.map_err(Self::err)
    }

    async fn delete_traces(
        &self,
        experiment_id: u64,
        max_timestamp_ms: Option<i64>,
        max_traces: Option<u64>,
        request_ids: &[String],
    ) -> Result<u64, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        self.get_experiment(experiment_id).await?;
        let traces = self.traces_of(experiment_id).await.map_err(Self::err)?;
        let selected: Vec<String> = if !request_ids.is_empty() {
            traces
                .iter()
                .filter(|t| request_ids.contains(&t.trace_id))
                .map(|t| t.trace_id.clone())
                .collect()
        } else {
            let mut candidates: Vec<String> = traces
                .iter()
                .filter(|t| max_timestamp_ms.is_none_or(|ts| t.request_time_ms <= ts))
                .map(|t| t.trace_id.clone())
                .collect();
            if let Some(max) = max_traces {
                candidates.truncate(max as usize);
            }
            candidates
        };
        for trace_id in &selected {
            self.document
                .delete_document(&self.collection, &format!("trc-{trace_id}"))
                .await
                .map_err(Self::err)?;
        }
        Ok(selected.len() as u64)
    }

    async fn set_trace_tag(
        &self,
        trace_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut trace = self.get_trace(trace_id).await?;
        trace.tags.insert(key.to_string(), value.to_string());
        self.put_trace(&trace).await.map_err(Self::err)
    }

    async fn delete_trace_tag(&self, trace_id: &str, key: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut trace = self.get_trace(trace_id).await?;
        trace.tags.remove(key);
        self.put_trace(&trace).await.map_err(Self::err)
    }

    async fn upsert_assessment(
        &self,
        trace_id: &str,
        assessment: AssessmentRecord,
    ) -> Result<AssessmentRecord, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut trace = self.get_trace(trace_id).await?;
        match trace
            .assessments
            .iter_mut()
            .find(|a| a.assessment_id == assessment.assessment_id)
        {
            Some(existing) => *existing = assessment.clone(),
            None => trace.assessments.push(assessment.clone()),
        }
        self.put_trace(&trace).await.map_err(Self::err)?;
        Ok(assessment)
    }

    async fn get_assessment(
        &self,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<AssessmentRecord, RunStoreError> {
        let trace = self.get_trace(trace_id).await?;
        trace
            .assessments
            .into_iter()
            .find(|a| a.assessment_id == assessment_id)
            .ok_or_else(|| RunStoreError::UnknownAssessment {
                trace_id: trace_id.to_string(),
                assessment_id: assessment_id.to_string(),
            })
    }

    async fn delete_assessment(
        &self,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let mut trace = self.get_trace(trace_id).await?;
        let before = trace.assessments.len();
        trace
            .assessments
            .retain(|a| a.assessment_id != assessment_id);
        if trace.assessments.len() == before {
            return Err(RunStoreError::UnknownAssessment {
                trace_id: trace_id.to_string(),
                assessment_id: assessment_id.to_string(),
            });
        }
        self.put_trace(&trace).await.map_err(Self::err)
    }

    // --- Logged models (TD-MLOPS-2) ---

    async fn create_logged_model(
        &self,
        model: LoggedModelRecord,
    ) -> Result<LoggedModelRecord, RunStoreError> {
        if model.name.is_empty() {
            return Err(RunStoreError::Empty { field: "name" });
        }
        if model.model_id.is_empty() {
            return Err(RunStoreError::Empty { field: "model_id" });
        }
        let _guard = self.mutation_lock.lock().await;
        if self
            .document
            .get_collection(&self.collection)
            .await
            .map_err(Self::err)?
            .is_none()
        {
            return Err(RunStoreError::UnknownExperiment {
                experiment_id: model.experiment_id,
            });
        }
        let experiment = self.get_experiment(model.experiment_id).await?;
        if experiment.stage == ExperimentStage::Deleted {
            return Err(RunStoreError::ExperimentDeleted {
                experiment_id: model.experiment_id,
            });
        }
        if self
            .get_payload::<LoggedModelRecord>(&format!("lm-{}", model.model_id))
            .await
            .map_err(Self::err)?
            .is_some()
        {
            return Err(RunStoreError::LoggedModelIdConflict {
                model_id: model.model_id,
            });
        }
        self.put_logged_model(&model, &BTreeMap::new())
            .await
            .map_err(Self::err)?;
        Ok(model)
    }

    async fn get_logged_model(
        &self,
        model_id: &str,
        allow_deleted: bool,
    ) -> Result<LoggedModelRecord, RunStoreError> {
        let model: LoggedModelRecord = self
            .get_payload(&format!("lm-{model_id}"))
            .await
            .map_err(Self::err)?
            .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                model_id: model_id.to_string(),
            })?;
        if !allow_deleted && model.lifecycle == RunLifecycle::Deleted {
            return Err(RunStoreError::UnknownLoggedModel {
                model_id: model_id.to_string(),
            });
        }
        Ok(model)
    }

    async fn list_logged_models(
        &self,
        experiment_id: u64,
        include_deleted: bool,
    ) -> Result<Vec<LoggedModelRecord>, RunStoreError> {
        self.ensure()
            .await
            .map_err(|e| Self::err(e.context("ensure collection")))?;
        self.get_experiment(experiment_id).await?;
        Ok(self
            .logged_models_of(experiment_id)
            .await
            .map_err(Self::err)?
            .into_iter()
            .filter(|m| include_deleted || m.lifecycle != RunLifecycle::Deleted)
            .collect())
    }

    async fn finalize_logged_model(
        &self,
        model_id: &str,
        status: LoggedModelStatus,
        now_ms: i64,
    ) -> Result<LoggedModelRecord, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut model, counters) = self.logged_model_with_counters(model_id).await?;
        model.status = status;
        model.last_updated_time_ms = now_ms;
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)?;
        Ok(model)
    }

    async fn delete_logged_model(&self, model_id: &str) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut model, counters) = self.logged_model_with_counters(model_id).await?;
        model.lifecycle = RunLifecycle::Deleted;
        model.last_updated_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)
    }

    async fn log_logged_model_params(
        &self,
        model_id: &str,
        params: &BTreeMap<String, String>,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut model, counters) = self.logged_model_with_counters(model_id).await?;
        for (key, value) in params {
            match model.params.get(key) {
                Some(existing) if existing == value => {}
                Some(_) => {
                    return Err(RunStoreError::ParamImmutable {
                        key: key.clone(),
                        run_id: model_id.to_string(),
                    });
                }
                None => {
                    model.params.insert(key.clone(), value.clone());
                }
            }
        }
        model.last_updated_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)
    }

    async fn set_logged_model_tags(
        &self,
        model_id: &str,
        tags: &BTreeMap<String, String>,
    ) -> Result<LoggedModelRecord, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut model, counters) = self.logged_model_with_counters(model_id).await?;
        for (key, value) in tags {
            model.tags.insert(key.clone(), value.clone());
        }
        model.last_updated_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)?;
        Ok(model)
    }

    async fn delete_logged_model_tag(
        &self,
        model_id: &str,
        key: &str,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut model, counters) = self.logged_model_with_counters(model_id).await?;
        if model.tags.remove(key).is_none() {
            return Err(RunStoreError::UnknownLoggedModelTag {
                model_id: model_id.to_string(),
                key: key.to_string(),
            });
        }
        model.last_updated_time_ms = chrono::Utc::now().timestamp_millis();
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)
    }

    async fn log_model_metric(
        &self,
        model_id: &str,
        sample: ModelMetricPoint,
    ) -> Result<MetricAppend, RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (model, mut counters) = self.logged_model_with_counters(model_id).await?;
        if model.lifecycle == RunLifecycle::Deleted {
            return Err(RunStoreError::UnknownLoggedModel {
                model_id: model_id.to_string(),
            });
        }
        // Doc id embeds the per-key counter (same collision fix as the run
        // side — review MAJOR-2; mtr- vs mtrm- stays disjoint at byte 3).
        let counter_key = counter_key(&sample.point.key);
        let next_seq = counters.get(&counter_key).copied().unwrap_or(0) + 1;
        counters.insert(counter_key.clone(), next_seq);
        self.put_logged_model(&model, &counters)
            .await
            .map_err(Self::err)?;
        self.put_payload(
            &format!("mtrm-{model_id}-{counter_key}-{next_seq}"),
            &[
                ("kind", ProximaValue::String("model_metric".to_string())),
                ("model_id", ProximaValue::String(model_id.to_string())),
                ("key", ProximaValue::String(sample.point.key.clone())),
                ("seq", ProximaValue::Int64(next_seq as i64)),
            ],
            &sample,
        )
        .await
        .map_err(Self::err)?;
        Ok(MetricAppend {
            history_len: next_seq,
        })
    }

    async fn model_metric_history(
        &self,
        model_id: &str,
        key: &str,
    ) -> Result<Vec<ModelMetricPoint>, RunStoreError> {
        self.get_logged_model(model_id, true).await?;
        let filter = filter_of(&[
            eq_cond("kind", "model_metric"),
            eq_cond("model_id", model_id),
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
        let mut points: Vec<(u64, ModelMetricPoint)> = Vec::new();
        for doc in result.documents {
            let seq = match doc.props.get("seq") {
                Some(ProximaTreeNode::Value(ProximaValue::Int64(s))) => *s as u64,
                other => {
                    return Err(Self::err(anyhow::anyhow!(
                        "model metric doc '{}' missing its seq field ({other:?})",
                        doc.id
                    )));
                }
            };
            let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = doc.props.get("payload")
            else {
                return Err(Self::err(anyhow::anyhow!(
                    "model metric doc '{}' has no payload",
                    doc.id
                )));
            };
            let sample: ModelMetricPoint = serde_json::from_str(json)
                .with_context(|| format!("corrupt model metric doc '{}'", doc.id))
                .map_err(Self::err)?;
            points.push((seq, sample));
        }
        points.sort_by_key(|(seq, _)| *seq);
        Ok(points.into_iter().map(|(_, s)| s).collect())
    }

    async fn model_metrics(&self, model_id: &str) -> Result<Vec<ModelMetricPoint>, RunStoreError> {
        self.get_logged_model(model_id, true).await?;
        let filter = filter_of(&[
            eq_cond("kind", "model_metric"),
            eq_cond("model_id", model_id),
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
        let mut points: Vec<(u64, ModelMetricPoint)> = Vec::new();
        for doc in result.documents {
            let seq = match doc.props.get("seq") {
                Some(ProximaTreeNode::Value(ProximaValue::Int64(s))) => *s as u64,
                other => {
                    return Err(Self::err(anyhow::anyhow!(
                        "model metric doc '{}' missing its seq field ({other:?})",
                        doc.id
                    )));
                }
            };
            let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = doc.props.get("payload")
            else {
                return Err(Self::err(anyhow::anyhow!(
                    "model metric doc '{}' has no payload",
                    doc.id
                )));
            };
            let sample: ModelMetricPoint = serde_json::from_str(json)
                .with_context(|| format!("corrupt model metric doc '{}'", doc.id))
                .map_err(Self::err)?;
            points.push((seq, sample));
        }
        points.sort_by_key(|(seq, _)| *seq);
        Ok(points.into_iter().map(|(_, s)| s).collect())
    }

    // --- Run <-> model links (TD-MLOPS-2) ---

    async fn log_run_outputs(
        &self,
        run_id: &str,
        outputs: Vec<ModelOutputRef>,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.model_outputs.extend(outputs);
        self.put_run(&run, &counters).await.map_err(Self::err)
    }

    async fn log_run_model_inputs(
        &self,
        run_id: &str,
        model_ids: Vec<String>,
    ) -> Result<(), RunStoreError> {
        let _guard = self.mutation_lock.lock().await;
        let (mut run, counters) = self.run_with_counters(run_id).await?;
        run.model_inputs.extend(model_ids);
        self.put_run(&run, &counters).await.map_err(Self::err)
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

    /// Persist a trace record (assessments embedded in the payload).
    async fn put_trace(&self, trace: &TraceRecord) -> Result<()> {
        self.put_payload(
            &format!("trc-{}", trace.trace_id),
            &[
                ("kind", ProximaValue::String("trace".to_string())),
                (
                    "experiment_id",
                    ProximaValue::Int64(trace.experiment_id as i64),
                ),
                (
                    "state",
                    ProximaValue::String(
                        serde_json::to_string(&trace.state)
                            .map(|s| s.trim_matches('"').to_uppercase())
                            .unwrap_or_default(),
                    ),
                ),
            ],
            trace,
        )
        .await
    }

    /// Persist a logged model plus its append counters (`ctr_m_<key>`), the
    /// same durable-counter discipline as [`Self::put_run`].
    async fn put_logged_model(
        &self,
        model: &LoggedModelRecord,
        counters: &BTreeMap<String, u64>,
    ) -> Result<()> {
        let mut fields: Vec<(String, ProximaValue)> = vec![
            (
                "kind".to_string(),
                ProximaValue::String("logged_model".to_string()),
            ),
            (
                "experiment_id".to_string(),
                ProximaValue::Int64(model.experiment_id as i64),
            ),
            (
                "lifecycle".to_string(),
                ProximaValue::String(serde_json::to_string(&model.lifecycle).unwrap_or_default()),
            ),
        ];
        fields.extend(
            counters
                .iter()
                .map(|(name, value)| (format!("ctr_{name}"), ProximaValue::Int64(*value as i64))),
        );
        self.put_payload_fields(&format!("lm-{}", model.model_id), fields, model)
            .await
    }

    /// Read a logged model TOGETHER with its durable counters; absent ids
    /// surface as typed [`RunStoreError::UnknownLoggedModel`].
    async fn logged_model_with_counters(
        &self,
        model_id: &str,
    ) -> Result<(LoggedModelRecord, BTreeMap<String, u64>), RunStoreError> {
        let record = self
            .document
            .get_document(&self.collection, &format!("lm-{model_id}"), None)
            .await
            .map_err(Self::err)?
            .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                model_id: model_id.to_string(),
            })?;
        let Some(ProximaTreeNode::Value(ProximaValue::String(json))) = record.props.get("payload")
        else {
            return Err(Self::err(anyhow::anyhow!(
                "logged-model document '{model_id}' has no payload"
            )));
        };
        let model: LoggedModelRecord = serde_json::from_str(json)
            .with_context(|| format!("corrupt logged-model doc '{}'", record.id))
            .map_err(Self::err)?;
        let mut counters = BTreeMap::new();
        for (key, node) in &record.props {
            if let (Some(name), ProximaTreeNode::Value(ProximaValue::Int64(v))) =
                (key.strip_prefix("ctr_"), node)
            {
                counters.insert(name.to_string(), *v as u64);
            }
        }
        Ok((model, counters))
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
