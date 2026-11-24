"""MLOps and AutoML catalog contracts for ProximaDB.

These helpers keep ML metadata SQL-queryable through pgwire while leaving
training orchestration, artifacts, and long-running jobs to SDK/REST/gRPC
surfaces. Optional integrations such as MLflow, AutoGluon, and PyCaret are
detected lazily so the base SDK remains dependency-light.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from typing import Any

from ..chunking_strategies.contracts import ResolvedInputContract

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Content-addressed bytes referenced by an xCatalog asset.

    A URI locates bytes; the digest establishes their immutable identity.
    Model repositories should use a deterministic package/manifest digest when
    they contain more than one file.
    """

    uri: str
    digest: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("artifact uri must not be empty")
        if not _SHA256.fullmatch(self.digest):
            raise ValueError("artifact digest must be sha256:<64 lowercase hex chars>")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes cannot be negative")
        if not self.media_type.strip():
            raise ValueError("artifact media_type must not be empty")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class EmbeddingModelRegistration:
    """SDK adapter from a runtime-resolved contract to native xCatalog JSON.

    This object only builds the generated-API payload. It deliberately performs
    no hand-written HTTP so the eventual REST surface remains OpenAPI-generated.
    """

    registered_name: str
    version: int
    contract: ResolvedInputContract
    artifact: ArtifactDescriptor
    created_at_ms: int
    declared_context_limit: int
    normalized: bool = True
    pooling: str = "model-defined"
    license_id: str = "unknown"
    access: str = "unreviewed"
    requires_remote_code: bool = False
    approved_runtimes: tuple[str, ...] = ()
    source_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.registered_name.strip():
            raise ValueError("registered_name must not be empty")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if self.declared_context_limit < self.contract.effective_context_limit:
            raise ValueError(
                "declared_context_limit cannot be below the runtime effective limit"
            )
        if self.access not in {"open", "gated", "unreviewed"}:
            raise ValueError("access must be open, gated, or unreviewed")
        if not self.license_id.strip():
            raise ValueError("license_id must not be empty")
        if not self.pooling.strip():
            raise ValueError("pooling must not be empty")
        if any(not runtime.strip() for runtime in self.approved_runtimes):
            raise ValueError("approved runtime names must not be empty")

    @classmethod
    def from_resolved_contract(
        cls,
        registered_name: str,
        version: int,
        contract: ResolvedInputContract,
        artifact: ArtifactDescriptor,
        created_at_ms: int,
        *,
        declared_context_limit: int | None = None,
        normalized: bool = True,
        pooling: str = "model-defined",
        license_id: str = "unknown",
        access: str = "unreviewed",
        requires_remote_code: bool = False,
        approved_runtimes: tuple[str, ...] = (),
        source_run_id: str | None = None,
    ) -> EmbeddingModelRegistration:
        return cls(
            registered_name=registered_name,
            version=version,
            contract=contract,
            artifact=artifact,
            created_at_ms=created_at_ms,
            declared_context_limit=(
                declared_context_limit or contract.effective_context_limit
            ),
            normalized=normalized,
            pooling=pooling,
            license_id=license_id,
            access=access,
            requires_remote_code=requires_remote_code,
            approved_runtimes=approved_runtimes,
            source_run_id=source_run_id,
        )

    @staticmethod
    def _digest(value: str) -> str:
        return value if value.startswith("sha256:") else f"sha256:{value}"

    def _dimension_contract(self) -> dict[str, Any]:
        native = self.contract.native_dimension
        if native is None:
            raise ValueError("native_dimension is required for model registration")
        if self.contract.supported_output_dimensions:
            dimensions = sorted(set(self.contract.supported_output_dimensions))
            if native not in dimensions:
                dimensions.append(native)
                dimensions.sort()
            policy: str | dict[str, dict[str, int]] = "discrete"
        elif self.contract.minimum_output_dimension is not None:
            dimensions = []
            policy = {"range": {"minimum": self.contract.minimum_output_dimension}}
        else:
            dimensions = [native]
            policy = "fixed"
        return {
            "native_dimension": native,
            "dimension_policy": policy,
            "supported_dimensions": dimensions,
            "normalized": self.normalized,
            "pooling": self.pooling,
        }

    def model_version(self) -> dict[str, Any]:
        tokenizer_revision = (
            getattr(self.contract.counter, "resolved_revision", None)
            or self.contract.model_revision
        )
        payload: dict[str, Any] = {
            "version": self.version,
            "provider_model_id": self.contract.model_id,
            "artifact": self.artifact.to_manifest(),
            "input": {
                "model_revision": self.contract.model_revision,
                "tokenizer_id": self.contract.counter.name,
                "tokenizer_revision": str(tokenizer_revision),
                "tokenizer_fingerprint": self._digest(
                    self.contract.counter.fingerprint
                ),
                "declared_context_limit": self.declared_context_limit,
                "effective_context_limit": self.contract.effective_context_limit,
                "special_token_count": self.contract.counter.count(""),
                "document_template": self.contract.renderer.document_template,
                "query_template": self.contract.renderer.query_template,
                "document_parameters": dict(self.contract.document_encode_parameters),
                "query_parameters": dict(self.contract.query_encode_parameters),
            },
            "output": self._dimension_contract(),
            "governance": {
                "license_id": self.license_id,
                "access": self.access,
                "requires_remote_code": self.requires_remote_code,
                "approved_runtimes": sorted(set(self.approved_runtimes)),
            },
            "lineage": {
                "producer_execution_id": self.source_run_id,
                "code_revision": self.contract.model_revision,
                "inputs": [],
            },
            "created_at_ms": self.created_at_ms,
        }
        if self.source_run_id is not None:
            payload["source_run_id"] = self.source_run_id
        return payload

    def xcatalog_asset(self) -> dict[str, Any]:
        """Return the typed facet stored on a unified xCatalog object."""
        return {
            "kind": "embedding_model",
            "contract": {
                "schema_version": 1,
                "revision": 0,
                "name": self.registered_name,
                "versions": {str(self.version): self.model_version()},
                "aliases": {},
                "evidence": [],
                "decisions": [],
                "deployments": {},
                "tags": {},
            },
        }


@dataclass(frozen=True)
class FeatureTableDDL:
    name: str
    entity_key: str = "entity_id"
    event_time: str = "event_time"
    features_column: str = "features"
    embedding_dimension: int | None = None
    storage_engine: str = "SST"
    physical_layout: str = "columnar"
    catalog_namespace: str | None = None

    def create_table_sql(self) -> str:
        columns = [
            f"{_q(self.entity_key)} TEXT NOT NULL",
            f"{_q(self.event_time)} TIMESTAMP NOT NULL",
            f"{_q(self.features_column)} JSONB NOT NULL DEFAULT '{{}}'::jsonb",
        ]
        if self.embedding_dimension is not None:
            columns.append(f"{_q('embedding')} VECTOR({self.embedding_dimension})")
        columns.append(f"PRIMARY KEY ({_q(self.entity_key)}, {_q(self.event_time)})")
        return f"CREATE TABLE IF NOT EXISTS {_q(_ident(self.name))} ({', '.join(columns)});"

    def xcatalog_sql(self) -> list[str]:
        namespace = self.catalog_namespace or f"features.{_ident(self.name)}"
        return [
            f"COMMENT ON TABLE {_q(_ident(self.name))} IS "
            f"'xcatalog.namespace={namespace};"
            f"engine={self.storage_engine};layout={self.physical_layout};"
            "kind=feature_table';"
        ]

    def statements(self, *, include_xcatalog: bool = True) -> list[str]:
        statements = [self.create_table_sql()]
        if include_xcatalog:
            statements.extend(self.xcatalog_sql())
        return statements


@dataclass(frozen=True)
class ExperimentTrackerDDL:
    prefix: str = "mlops"
    storage_engine: str = "SST"
    physical_layout: str = "hybrid"

    def statements(self) -> list[str]:
        runs = _ident(f"{self.prefix}_runs")
        metrics = _ident(f"{self.prefix}_metrics")
        params = _ident(f"{self.prefix}_params")
        registry = _ident(f"{self.prefix}_models")
        predictions = _ident(f"{self.prefix}_predictions")
        return [
            (
                f"CREATE TABLE IF NOT EXISTS {_q(runs)} ("
                '"run_id" TEXT NOT NULL, "experiment_name" TEXT NOT NULL, '
                '"status" TEXT NOT NULL, "framework" TEXT, "started_at_ms" BIGINT, '
                '"ended_at_ms" BIGINT, "metadata" JSONB NOT NULL DEFAULT \'{}\'::jsonb, '
                'PRIMARY KEY ("run_id"));'
            ),
            (
                f"CREATE TABLE IF NOT EXISTS {_q(metrics)} ("
                '"run_id" TEXT NOT NULL, "metric_name" TEXT NOT NULL, '
                '"step" BIGINT NOT NULL, "metric_value" DOUBLE PRECISION NOT NULL, '
                '"recorded_at_ms" BIGINT, PRIMARY KEY ("run_id", "metric_name", "step"));'
            ),
            (
                f"CREATE TABLE IF NOT EXISTS {_q(params)} ("
                '"run_id" TEXT NOT NULL, "param_name" TEXT NOT NULL, '
                '"param_value" TEXT NOT NULL, PRIMARY KEY ("run_id", "param_name"));'
            ),
            (
                f"CREATE TABLE IF NOT EXISTS {_q(registry)} ("
                '"model_name" TEXT NOT NULL, "version" TEXT NOT NULL, "stage" TEXT, '
                '"run_id" TEXT, "artifact_uri" TEXT, "signature" JSONB NOT NULL DEFAULT \'{}\'::jsonb, '
                "\"metrics\" JSONB NOT NULL DEFAULT '{}'::jsonb, "
                'PRIMARY KEY ("model_name", "version"));'
            ),
            (
                f"CREATE TABLE IF NOT EXISTS {_q(predictions)} ("
                '"prediction_id" TEXT NOT NULL, "model_name" TEXT NOT NULL, '
                '"model_version" TEXT, "entity_id" TEXT, "event_time" TIMESTAMP, '
                "\"inputs\" JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "\"prediction\" JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "\"labels\" JSONB NOT NULL DEFAULT '{}'::jsonb, "
                'PRIMARY KEY ("prediction_id"));'
            ),
            (
                f"COMMENT ON TABLE {_q(runs)} IS "
                f"'xcatalog.namespace=mlops.{_ident(self.prefix)};"
                f"engine={self.storage_engine};layout={self.physical_layout};kind=experiment_runs';"
            ),
        ]


@dataclass(frozen=True)
class AutoMLRunSpec:
    framework: str
    target: str
    feature_table: str
    label_column: str
    problem_type: str | None = None
    eval_metric: str | None = None
    presets: str | None = None
    time_limit_seconds: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "target": self.target,
            "feature_table": self.feature_table,
            "label_column": self.label_column,
            "problem_type": self.problem_type,
            "eval_metric": self.eval_metric,
            "presets": self.presets,
            "time_limit_seconds": self.time_limit_seconds,
            "extra": dict(self.extra),
        }


def optional_integrations() -> dict[str, bool]:
    """Return installed optional ML integration availability."""
    return {
        "mlflow": importlib.util.find_spec("mlflow") is not None,
        "autogluon": importlib.util.find_spec("autogluon") is not None,
        "pycaret": importlib.util.find_spec("pycaret") is not None,
    }


def _ident(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.lower())
    return cleaned.strip("_") or "mlops"


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
