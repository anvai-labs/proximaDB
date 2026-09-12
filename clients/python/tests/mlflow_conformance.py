"""Real-client MLflow conformance (the ratchet).

Drives the CANONICAL mlflow workflow against a live server with the
compatibility gate on, counting passing workflow steps. Ratchets are
PER GENERATION (clients/python/tests/mlflow_conformance_steps_2x.txt /
mlflow_conformance_steps_3x.txt): steps requiring an MLflow 3.x client
(generation-gated below) never run under the 2.x client and vice versa,
so each generation's high-water count is independent and only goes up.

Usage: python tests/mlflow_conformance.py <tracking_uri> [tag]
"""

import sys
import traceback

PASSED = []
FAILED = []

# The active client generation (2 or 3); main() sets it from the imported
# mlflow version. Steps may declare a MINIMUM generation via `gen=`; the
# 2-arg call contract (pinned by test_mlflow_conformance_harness) stays.
_ACTIVE_GEN = 2


def step(name, fn, gen=2):
    """Run one workflow step; a failure records a miss and CONTINUES so the
    ratchet observes the passing count (fewer passing steps = regression).
    Any step failing still exits non-zero at the end via the ratchet."""
    if gen > _ACTIVE_GEN:
        print(f"SKIP {name} (requires an MLflow 3.x client)")
        return
    try:
        fn()
        PASSED.append(name)
        print(f"PASS {name}")
    except Exception:
        FAILED.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()


def result_code() -> int:
    """Fail if any attempted step failed, independent of the ratchet count."""
    return 0 if not FAILED else 1


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: mlflow_conformance.py <tracking_uri> [tag]", file=sys.stderr)
        return 2
    tracking_uri = sys.argv[1]
    # Resource names are tagged per invocation so successive runs (e.g. the
    # 2.x and 3.x clients in one lane) never collide on create.
    tag = sys.argv[2] if len(sys.argv) == 3 else "default"

    import mlflow
    from mlflow.tracking import MlflowClient

    global _ACTIVE_GEN
    _ACTIVE_GEN = int(mlflow.__version__.split(".")[0])
    print(f"client generation: mlflow {mlflow.__version__} (gen {_ACTIVE_GEN})")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    state = {}

    def do_create():
        experiment = client.create_experiment(f"conformance-e2e-{tag}")
        fetched = client.get_experiment(experiment)
        assert fetched.name == f"conformance-e2e-{tag}", fetched.name
        state["experiment"] = experiment

    step("create+get_experiment", do_create)

    def do_by_name():
        by_name = client.get_experiment_by_name(f"conformance-e2e-{tag}")
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
        runs = client.search_runs(
            [state["experiment"]], filter_string="params.lr = '0.01'"
        )
        assert len(runs) == 1, len(runs)

    step("search_runs_match", do_search_match)

    def do_search_non_match():
        runs = client.search_runs(
            [state["experiment"]], filter_string="params.lr = '9.9'"
        )
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

    # 6. Registry through the REAL client (TD-MLOPS-1 slice 3): registered
    # models lower to xCatalog registries; version creation and stage
    # transitions are the documented honest rejections.
    def do_model_create():
        client.create_registered_model(f"conformance-model-{tag}")
        model = client.get_registered_model(f"conformance-model-{tag}")
        assert model.name == f"conformance-model-{tag}"

    step("create+get_registered_model", do_model_create)

    def do_model_search():
        models = client.search_registered_models(
            filter_string=f"name LIKE '%conformance-model-{tag}%'"
        )
        assert any(m.name == f"conformance-model-{tag}" for m in models), models

    step("search_registered_models", do_model_search)

    def do_version_create_rejected():
        import mlflow

        try:
            client.create_model_version(f"conformance-model-{tag}", "s3://bucket/model")
        except mlflow.exceptions.MlflowException as exc:
            assert "lifecycle API" in str(exc), str(exc)
        else:
            raise AssertionError("create_model_version must be rejected")

    step("model_version_create_rejected", do_version_create_rejected)

    def do_stage_rejected():
        import mlflow

        try:
            client.transition_model_version_stage(
                f"conformance-model-{tag}", "1", "Production"
            )
        except mlflow.exceptions.MlflowException as exc:
            assert "alias" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("transition-stage must be rejected")

    step("transition_stage_rejected_with_alias_pointer", do_stage_rejected)

    def do_model_versions_empty():
        versions = client.search_model_versions(f"name='conformance-model-{tag}'")
        assert list(versions) == [], list(versions)

    step("model_versions_search_empty", do_model_versions_empty)

    # 7. Artifacts through the proxy family (TD-MLOPS-1 slice 4): the run's
    # artifact_uri resolves against the tracking host's
    # /api/2.0/mlflow-artifacts proxy.
    import os
    import tempfile

    def do_artifact_roundtrip():
        run = client.get_run(state["run_id"])
        assert run.info.artifact_uri.startswith(
            "mlflow-artifacts:/"
        ), run.info.artifact_uri
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "model.txt")
            with open(local, "w") as fh:
                fh.write("artifact-bytes")
            client.log_artifact(state["run_id"], local)
            artifacts = client.list_artifacts(state["run_id"])
            assert any(
                a.path == "model.txt" and not a.is_dir for a in artifacts
            ), artifacts
            dest = os.path.join(tmp, "dl")
            os.makedirs(dest)
            downloaded = client.download_artifacts(state["run_id"], "model.txt", dest)
            with open(downloaded, "r") as fh:
                assert fh.read() == "artifact-bytes"

    step("artifact_roundtrip", do_artifact_roundtrip)

    # 8. Logged models (TD-MLOPS-2) — MLflow 3.x clients only.
    def do_lm_create():
        model = client.create_logged_model(
            state["experiment"], name=f"conformance-lm-{tag}"
        )
        assert model.model_id.startswith("m-"), model.model_id
        state["model_id"] = model.model_id
        fetched = client.get_logged_model(model.model_id)
        assert fetched.name == f"conformance-lm-{tag}", fetched.name
        assert fetched.artifact_location.startswith(
            "mlflow-artifacts:/"
        ), fetched.artifact_location

    step("logged_model_create+get", do_lm_create, gen=3)

    def do_lm_params_tags():
        client.log_model_params(state["model_id"], {"lr": "0.1"})
        client.set_logged_model_tags(state["model_id"], {"stage": "dev"})
        model = client.get_logged_model(state["model_id"])
        assert model.params["lr"] == "0.1", model.params
        assert model.tags["stage"] == "dev", model.tags

    step("logged_model_params+tags", do_lm_params_tags, gen=3)

    def do_lm_metric():
        client.log_metric(
            state["run_id"],
            "lm_accuracy",
            0.91,
            step=0,
            model_id=state["model_id"],
            dataset_name="holdout",
            dataset_digest="sha256:aa",
        )
        model = client.get_logged_model(state["model_id"])
        values = [m.value for m in model.metrics if m.key == "lm_accuracy"]
        assert values == [0.91], model.metrics
        # The model-owned metric must NOT leak into the run projection.
        run = client.get_run(state["run_id"])
        assert "lm_accuracy" not in run.data.metrics, run.data.metrics

    step("logged_model_metric_via_runs_log_metric", do_lm_metric, gen=3)

    def do_lm_search():
        models = client.search_logged_models(
            [state["experiment"]],
            filter_string=f"name LIKE '%conformance-lm-{tag}%'",
        )
        assert any(m.model_id == state["model_id"] for m in models), models

    step("logged_model_search_filter", do_lm_search, gen=3)

    def do_lm_finalize():
        client.finalize_logged_model(state["model_id"], "READY")
        model = client.get_logged_model(state["model_id"])
        assert model.status == "READY", model.status

    step("finalize_logged_model", do_lm_finalize, gen=3)

    def do_run_outputs():
        from mlflow.entities import LoggedModelOutput

        client.log_outputs(
            state["run_id"], [LoggedModelOutput(model_id=state["model_id"], step=0)]
        )
        run = client.get_run(state["run_id"])
        outputs = [o.model_id for o in run.outputs.model_outputs]
        assert state["model_id"] in outputs, outputs

    step("run_outputs_embedded_in_get_run", do_run_outputs, gen=3)

    # 9. Traces v3 (TD-MLOPS-2) — MLflow 3.x clients only. Spans ride the
    # artifact repo (this server deliberately serves no /version), so
    # get_trace PROVES the client could upload AND download trace data.
    # The client exports traces on a background processor (optionally an
    # async queue), and its IN-MEMORY trace info never carries the
    # server-minted artifactLocation tag — so reads must AWAIT the export
    # landing (exactly like a real consumer polling a fresh trace).
    import time as _time

    def _await_trace(trace_id, timeout=20.0):
        deadline = _time.monotonic() + timeout
        last = None
        while _time.monotonic() < deadline:
            try:
                trace = client.get_trace(trace_id)
                if trace.data.spans:
                    return trace
                last = AssertionError("trace readable but spans not yet uploaded")
            except Exception as exc:  # not exported yet / in-memory window
                last = exc
            _time.sleep(0.5)
        raise last if last else AssertionError("trace never became readable")

    def do_start_trace():
        # Pin the destination experiment — without it the client defaults
        # to experiment "0", not the conformance experiment.
        span = client.start_trace(
            name=f"conformance-trace-{tag}", experiment_id=state["experiment"]
        )
        state["trace_id"] = span.trace_id
        span.end()
        assert span.trace_id.startswith("tr-"), span.trace_id

    step("start_trace_v3", do_start_trace, gen=3)

    def do_get_trace():
        trace = _await_trace(state["trace_id"])
        assert trace.info.request_id == state["trace_id"], trace.info
        assert trace.data.spans, "span data must round-trip via the artifact repo"

    step("get_trace_v3_spans_via_artifacts", do_get_trace, gen=3)

    def do_search_traces():
        traces = client.search_traces(experiment_ids=[state["experiment"]])
        assert any(t.info.request_id == state["trace_id"] for t in traces), [
            t.info.request_id for t in traces
        ]

    step("search_traces_v3", do_search_traces, gen=3)

    def do_trace_tags():
        client.set_trace_tag(state["trace_id"], "reviewed", "yes")
        trace = client.get_trace(state["trace_id"])
        assert trace.info.tags.get("reviewed") == "yes", trace.info.tags

    step("trace_tags_set", do_trace_tags, gen=3)

    def do_assessment():
        import mlflow as mlflow_mod
        from mlflow.entities.assessment import AssessmentSource, Feedback

        feedback = Feedback(
            name="quality",
            value=5,
            source=AssessmentSource(source_type="HUMAN", source_id="conformance"),
        )
        mlflow_mod.log_assessment(state["trace_id"], feedback)
        trace = client.get_trace(state["trace_id"])
        # v3 assessments surface as v2 Feedback entities (`.name`).
        assessments = list(trace.info.assessments or [])
        assert any(a.name == "quality" for a in assessments), assessments
        state["assessment_id"] = next(
            a.assessment_id for a in assessments if a.name == "quality"
        )
        fetched = mlflow_mod.get_assessment(state["trace_id"], state["assessment_id"])
        assert fetched.name == "quality"

    step("assessments_log_get", do_assessment, gen=3)

    def do_delete_traces():
        client.delete_traces(state["experiment"], trace_ids=[state["trace_id"]])
        import mlflow as mlflow_mod

        try:
            client.get_trace(state["trace_id"])
        except mlflow_mod.exceptions.MlflowException:
            pass
        else:
            raise AssertionError("deleted trace must not be retrievable")

    step("delete_traces_v3", do_delete_traces, gen=3)

    total = len(PASSED)
    print(f"CONFORMANCE_STEPS={total}")
    # Fail the workflow outright if ANY attempted step missed. Keep this
    # independent of the count so adding a passing step can raise the ratchet.
    return result_code()


if __name__ == "__main__":
    sys.exit(main())
