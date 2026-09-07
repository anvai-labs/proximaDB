"""TD-MLOPS-1 slice 2: real-client MLflow conformance (the ratchet).

Drives the CANONICAL mlflow workflow against a live server with the
compatibility gate on, counting passing workflow steps. The checked-in
ratchet (clients/python/tests/mlflow_conformance_steps.txt) records the
high-water count; CI fails if fewer steps pass (counts only go up).

Usage: python tests/mlflow_conformance.py <tracking_uri>
"""

import sys

PASSED = []


def step(name):
    PASSED.append(name)
    print(f"PASS {name}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mlflow_conformance.py <tracking_uri>", file=sys.stderr)
        return 2
    tracking_uri = sys.argv[1]

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Experiment lifecycle.
    experiment = client.create_experiment("conformance-e2e")
    fetched = client.get_experiment(experiment)
    assert fetched.name == "conformance-e2e", fetched.name
    step("create+get_experiment")

    by_name = client.get_experiment_by_name("conformance-e2e")
    assert by_name is not None and by_name.experiment_id == experiment
    step("get_experiment_by_name")

    # 2. Run lifecycle with params/metrics/tags via the fluent-ish path.
    run = client.create_run(experiment, run_name="wf")
    run_id = run.info.run_id
    assert run.info.status == "RUNNING"
    step("create_run")

    client.log_param(run_id, "lr", "0.01")
    client.log_metric(run_id, "rmse", 0.9, step=0)
    client.log_metric(run_id, "rmse", 0.7, step=1)
    client.set_tag(run_id, "phase", "tune")
    step("log_param_metric_tag")

    run = client.get_run(run_id)
    assert run.data.params == {"lr": "0.01"}, run.data.params
    assert run.data.metrics == {"rmse": 0.7}, run.data.metrics
    assert run.data.tags.get("phase") == "tune"
    assert run.data.tags.get("mlflow.runName") == "wf"
    step("get_run_shape")

    history = client.get_metric_history(run_id, "rmse")
    assert [m.step for m in history] == [0, 1], history
    step("metric_history_order")

    client.log_batch(
        run_id,
        metrics=[],
        params=[],
        tags=[mlflow.entities.RunTag("batched", "yes")],
    )
    run = client.get_run(run_id)
    assert run.data.tags.get("batched") == "yes"
    step("log_batch")

    # 3. Search: match, non-match (negative control), view types.
    client.set_terminated(run_id, status="FINISHED")
    runs = client.search_runs([experiment], filter_string="params.lr = '0.01'")
    assert len(runs) == 1, len(runs)
    step("search_runs_match")

    runs = client.search_runs([experiment], filter_string="params.lr = '9.9'")
    assert len(runs) == 0, len(runs)
    step("search_runs_non_match_empty")

    # 4. Terminal + terminate semantics.
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED", run.info.status
    assert run.info.end_time is not None
    step("set_terminated_finished")

    # 5. Experiment deletion + restore.
    client.delete_experiment(experiment)
    fetched = client.get_experiment(experiment)
    assert fetched.lifecycle_stage == "deleted", fetched.lifecycle_stage
    step("delete_experiment_soft")

    client.restore_experiment(experiment)
    fetched = client.get_experiment(experiment)
    assert fetched.lifecycle_stage == "active"
    step("restore_experiment")

    total = len(PASSED)
    print(f"CONFORMANCE_STEPS={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
