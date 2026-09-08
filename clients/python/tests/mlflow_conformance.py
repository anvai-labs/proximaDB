"""TD-MLOPS-1 slice 2: real-client MLflow conformance (the ratchet).

Drives the CANONICAL mlflow workflow against a live server with the
compatibility gate on, counting passing workflow steps. The checked-in
ratchet (clients/python/tests/mlflow_conformance_steps.txt) records the
high-water count; CI fails if fewer steps pass (counts only go up).

Usage: python tests/mlflow_conformance.py <tracking_uri>
"""

import sys
import traceback

PASSED = []


def step(name, fn):
    """Run one workflow step; a failure records a miss and CONTINUES so the
    ratchet observes the passing count (fewer passing steps = regression).
    Any step failing still exits non-zero at the end via the ratchet."""
    try:
        fn()
        PASSED.append(name)
        print(f"PASS {name}")
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mlflow_conformance.py <tracking_uri>", file=sys.stderr)
        return 2
    tracking_uri = sys.argv[1]

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    state = {}

    def do_create():
        experiment = client.create_experiment("conformance-e2e")
        fetched = client.get_experiment(experiment)
        assert fetched.name == "conformance-e2e", fetched.name
        state["experiment"] = experiment

    step("create+get_experiment", do_create)

    def do_by_name():
        by_name = client.get_experiment_by_name("conformance-e2e")
        assert by_name is not None
        assert by_name.experiment_id == state["experiment"]

    step("get_experiment_by_name", do_by_name)

    def do_create_run():
        run = client.create_run(state["experiment"], run_name="wf")
        assert run.info.status == "RUNNING"
        state["run_id"] = run.info.run_id

    step("create_run", do_create_run)

    def do_log():
        run_id = state["run_id"]
        client.log_param(run_id, "lr", "0.01")
        client.log_metric(run_id, "rmse", 0.9, step=0)
        client.log_metric(run_id, "rmse", 0.7, step=1)
        client.set_tag(run_id, "phase", "tune")

    step("log_param_metric_tag", do_log)

    def do_get_shape():
        run = client.get_run(state["run_id"])
        assert run.data.params == {"lr": "0.01"}, run.data.params
        assert run.data.metrics == {"rmse": 0.7}, run.data.metrics
        assert run.data.tags.get("phase") == "tune"
        assert run.data.tags.get("mlflow.runName") == "wf"

    step("get_run_shape", do_get_shape)

    def do_history():
        history = client.get_metric_history(state["run_id"], "rmse")
        assert [m.step for m in history] == [0, 1], history

    step("metric_history_order", do_history)

    def do_batch():
        client.log_batch(
            state["run_id"],
            metrics=[],
            params=[],
            tags=[mlflow.entities.RunTag("batched", "yes")],
        )
        run = client.get_run(state["run_id"])
        assert run.data.tags.get("batched") == "yes"

    step("log_batch", do_batch)

    def do_search_match():
        client.set_terminated(state["run_id"], status="FINISHED")
        runs = client.search_runs([state["experiment"]], filter_string="params.lr = '0.01'")
        assert len(runs) == 1, len(runs)

    step("search_runs_match", do_search_match)

    def do_search_non_match():
        runs = client.search_runs([state["experiment"]], filter_string="params.lr = '9.9'")
        assert len(runs) == 0, len(runs)

    step("search_runs_non_match_empty", do_search_non_match)

    def do_terminated():
        run = client.get_run(state["run_id"])
        assert run.info.status == "FINISHED", run.info.status
        assert run.info.end_time is not None

    step("set_terminated_finished", do_terminated)

    def do_delete():
        client.delete_experiment(state["experiment"])
        fetched = client.get_experiment(state["experiment"])
        assert fetched.lifecycle_stage == "deleted", fetched.lifecycle_stage

    step("delete_experiment_soft", do_delete)

    def do_restore():
        client.restore_experiment(state["experiment"])
        fetched = client.get_experiment(state["experiment"])
        assert fetched.lifecycle_stage == "active"

    step("restore_experiment", do_restore)

    total = len(PASSED)
    print(f"CONFORMANCE_STEPS={total}")
    # Fail the workflow outright if ANY step missed (the ratchet in CI adds
    # the historical high-water comparison).
    return 0 if total == 12 else 1


if __name__ == "__main__":
    sys.exit(main())
