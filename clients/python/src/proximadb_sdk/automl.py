"""
AutoML Module for ProximaDB Python SDK

Provides automated engine selection, workload prediction,
and hyperparameter optimization.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkloadType(str, Enum):
    """Types of database workloads"""

    READ_HEAVY = "read_heavy"
    WRITE_HEAVY = "write_heavy"
    MIXED = "mixed"
    ANALYTICS = "analytics"
    STREAMING = "streaming"
    BATCH = "batch"


class OptimizationGoal(str, Enum):
    """Optimization goals for engine selection"""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COST = "cost"
    RECALL = "recall"
    BALANCED = "balanced"


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a workload"""

    # Query patterns
    read_ratio: float = 0.5  # 0-1, ratio of reads to total ops
    write_ratio: float = 0.5  # 0-1, ratio of writes to total ops
    query_complexity: float = 0.5  # 0-1, simple to complex

    # Data characteristics
    vector_count: int = 0
    vector_dimension: int = 0
    metadata_cardinality: int = 0

    # Access patterns
    temporal_locality: float = 0.5  # 0-1, random to sequential
    spatial_locality: float = 0.5  # 0-1, scattered to clustered
    hot_data_ratio: float = 0.2  # ratio of frequently accessed data

    # Performance requirements
    target_latency_ms: float | None = None
    target_throughput: int | None = None
    memory_budget_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_ratio": self.read_ratio,
            "write_ratio": self.write_ratio,
            "query_complexity": self.query_complexity,
            "vector_count": self.vector_count,
            "vector_dimension": self.vector_dimension,
            "metadata_cardinality": self.metadata_cardinality,
            "temporal_locality": self.temporal_locality,
            "spatial_locality": self.spatial_locality,
            "hot_data_ratio": self.hot_data_ratio,
            "target_latency_ms": self.target_latency_ms,
            "target_throughput": self.target_throughput,
            "memory_budget_mb": self.memory_budget_mb,
        }


@dataclass
class EngineRecommendation:
    """Recommendation for a storage engine"""

    engine: str
    confidence: float
    reasoning: str
    estimated_latency_ms: float
    estimated_throughput: int
    estimated_memory_mb: int
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_throughput": self.estimated_throughput,
            "estimated_memory_mb": self.estimated_memory_mb,
            "config_overrides": self.config_overrides,
        }


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for optimization"""

    name: str
    current_value: Any
    min_value: Any | None = None
    max_value: Any | None = None
    step: Any | None = None
    allowed_values: list[Any] | None = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""

    best_config: dict[str, Any]
    best_score: float
    iterations: int
    search_history: list[dict[str, Any]] = field(default_factory=list)
    improvement_ratio: float = 0


class WorkloadPredictor:
    """
    Predicts workload characteristics from observed operations.

    Example:
        >>> predictor = WorkloadPredictor()
        >>> predictor.observe_operation("search", latency_ms=5.2, vector_count=100)
        >>> predictor.observe_operation("insert", latency_ms=2.1, vector_count=10)
        >>> characteristics = predictor.predict()
    """

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._operations: list[dict[str, Any]] = []
        self._operation_counts: dict[str, int] = {}

    def observe_operation(
        self,
        operation: str,
        latency_ms: float,
        vector_count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Observe an operation for workload prediction.

        Args:
            operation: Operation type (search, insert, update, delete)
            latency_ms: Operation latency in milliseconds
            vector_count: Number of vectors involved
            metadata: Additional operation metadata
        """
        self._operations.append(
            {
                "operation": operation,
                "latency_ms": latency_ms,
                "vector_count": vector_count,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

        # Keep window size
        if len(self._operations) > self._window_size:
            self._operations.pop(0)

        # Update counts
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1

    def predict(self) -> WorkloadCharacteristics:
        """
        Predict workload characteristics from observed operations.

        Returns:
            WorkloadCharacteristics based on observations
        """
        if not self._operations:
            return WorkloadCharacteristics()

        total_ops = sum(self._operation_counts.values())
        read_ops = self._operation_counts.get("search", 0) + self._operation_counts.get(
            "get", 0
        )
        write_ops = self._operation_counts.get(
            "insert", 0
        ) + self._operation_counts.get("update", 0)

        # Calculate ratios
        read_ratio = read_ops / total_ops if total_ops > 0 else 0.5
        write_ratio = write_ops / total_ops if total_ops > 0 else 0.5

        # Calculate query complexity from latency distribution
        latencies = [op["latency_ms"] for op in self._operations]
        avg_latency = sum(latencies) / len(latencies)
        query_complexity = min(1.0, avg_latency / 100)  # Normalize to 0-1

        # Estimate temporal locality from timestamps
        if len(self._operations) > 1:
            timestamps = [op["timestamp"] for op in self._operations]
            intervals = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals) if intervals else 1
            temporal_locality = max(0, 1 - min(1, avg_interval / 10))
        else:
            temporal_locality = 0.5

        # Estimate vector counts
        vector_counts = [op["vector_count"] for op in self._operations]
        total_vectors = sum(vector_counts)

        return WorkloadCharacteristics(
            read_ratio=read_ratio,
            write_ratio=write_ratio,
            query_complexity=query_complexity,
            vector_count=total_vectors,
            temporal_locality=temporal_locality,
        )

    def get_workload_type(self) -> WorkloadType:
        """
        Determine the workload type from observations.

        Returns:
            WorkloadType enum value
        """
        char = self.predict()

        if char.write_ratio > 0.7:
            return WorkloadType.WRITE_HEAVY
        elif char.read_ratio > 0.7:
            return WorkloadType.READ_HEAVY
        else:
            return WorkloadType.MIXED


class EngineSelector:
    """
    Automatically selects the optimal storage engine.

    Example:
        >>> selector = EngineSelector()
        >>> recommendation = selector.recommend(
        ...     characteristics=WorkloadCharacteristics(
        ...         read_ratio=0.8,
        ...         vector_count=100000,
        ...         target_latency_ms=10
        ...     ),
        ...     goal=OptimizationGoal.LATENCY
        ... )
        >>> print(f"Recommended: {recommendation.engine}")
    """

    # Engine profiles with performance characteristics
    ENGINE_PROFILES = {
        "sst": {
            "latency_base": 5,
            "latency_scale": 0.001,  # per 1000 vectors
            "throughput_base": 100000,
            "memory_base": 45,  # MB per 10K vectors
            "read_score": 0.8,
            "write_score": 0.95,
            "recall": 0.95,
            "best_for": ["write_heavy", "streaming", "real_time"],
        },
        "helix": {
            "latency_base": 13,
            "latency_scale": 0.002,
            "throughput_base": 50000,
            "memory_base": 52,
            "read_score": 0.9,
            "write_score": 0.6,
            "recall": 0.98,
            "best_for": ["semantic_search", "clustering", "spatial"],
        },
        "viper": {
            "latency_base": 90,
            "latency_scale": 0.005,
            "throughput_base": 30000,
            "memory_base": 28,
            "read_score": 0.7,
            "write_score": 0.4,
            "recall": 0.99,
            "best_for": ["analytics", "batch", "columnar"],
        },
        "swift": {
            "latency_base": 5,
            "latency_scale": 0.01,  # scales poorly with size
            "throughput_base": 150000,
            "memory_base": 120,
            "read_score": 0.95,
            "write_score": 0.7,
            "recall": 1.0,
            "best_for": ["small_datasets", "exact_search", "prototyping"],
            "max_vectors": 10000,
        },
        "nova": {
            "latency_base": 30,
            "latency_scale": 0.003,
            "throughput_base": 40000,
            "memory_base": 48,
            "read_score": 0.75,
            "write_score": 0.75,
            "recall": 0.97,
            "best_for": ["mixed", "progressive", "columnar"],
        },
        "raptor": {
            "latency_base": 10,
            "latency_scale": 0.002,
            "throughput_base": 80000,
            "memory_base": 55,
            "read_score": 0.85,
            "write_score": 0.85,
            "recall": 0.96,
            "best_for": ["adaptive", "dynamic", "evolving"],
        },
    }

    def __init__(self, client=None):
        """
        Initialize engine selector.

        Args:
            client: Optional ProximaDBClient for server-side recommendations
        """
        self._client = client

    def recommend(
        self,
        characteristics: WorkloadCharacteristics,
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ) -> EngineRecommendation:
        """
        Recommend the optimal storage engine.

        Args:
            characteristics: Workload characteristics
            goal: Optimization goal

        Returns:
            EngineRecommendation with engine choice and configuration
        """
        scores: list[tuple[str, float, dict[str, Any]]] = []

        for engine, profile in self.ENGINE_PROFILES.items():
            # TD-SDKENG-1: never RECOMMEND an engine the server rejects or
            # gates — VIPER is deprecated (ADR-093), SWIFT/RAPTOR are
            # experimental. Profiles stay for scoring history.
            if engine in ("viper", "swift", "raptor"):
                continue

            score = self._calculate_score(engine, profile, characteristics, goal)
            config = self._generate_config(engine, profile, characteristics)
            scores.append((engine, score, config))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        best_engine, best_score, best_config = scores[0]
        profile = self.ENGINE_PROFILES[best_engine]

        # Calculate estimates
        vector_count_k = characteristics.vector_count / 1000
        estimated_latency = (
            profile["latency_base"] + profile["latency_scale"] * vector_count_k
        )
        estimated_throughput = int(
            profile["throughput_base"] * (1 - 0.1 * vector_count_k / 100)
        )
        estimated_memory = int(
            profile["memory_base"] * (characteristics.vector_count / 10000)
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(best_engine, characteristics, goal)

        return EngineRecommendation(
            engine=best_engine,
            confidence=best_score,
            reasoning=reasoning,
            estimated_latency_ms=estimated_latency,
            estimated_throughput=max(1000, estimated_throughput),
            estimated_memory_mb=max(10, estimated_memory),
            config_overrides=best_config,
        )

    def _calculate_score(
        self,
        engine: str,
        profile: dict[str, Any],
        characteristics: WorkloadCharacteristics,
        goal: OptimizationGoal,
    ) -> float:
        """Calculate a score for an engine given characteristics and goal"""
        score = 0.0

        # Base score from read/write affinity
        score += characteristics.read_ratio * profile["read_score"]
        score += characteristics.write_ratio * profile["write_score"]

        # Goal-specific adjustments
        if goal == OptimizationGoal.LATENCY:
            # Prefer lower latency engines
            latency_factor = 1.0 / (1 + profile["latency_base"] / 50)
            score += latency_factor * 0.5
        elif goal == OptimizationGoal.THROUGHPUT:
            throughput_factor = profile["throughput_base"] / 100000
            score += throughput_factor * 0.5
        elif goal == OptimizationGoal.MEMORY:
            memory_factor = 1.0 / (1 + profile["memory_base"] / 100)
            score += memory_factor * 0.5
        elif goal == OptimizationGoal.RECALL:
            score += profile["recall"] * 0.5

        # Check target constraints
        if characteristics.target_latency_ms:
            if profile["latency_base"] > characteristics.target_latency_ms:
                score *= 0.5  # Penalize if can't meet latency

        if characteristics.target_throughput:
            if profile["throughput_base"] < characteristics.target_throughput:
                score *= 0.5

        # Boost for matching workload type
        workload_type = self._infer_workload_type(characteristics)
        if workload_type in profile.get("best_for", []):
            score *= 1.2

        return min(1.0, score)

    def _infer_workload_type(self, characteristics: WorkloadCharacteristics) -> str:
        """Infer workload type from characteristics"""
        if characteristics.write_ratio > 0.7:
            return "write_heavy"
        elif characteristics.read_ratio > 0.7:
            return "read_heavy"
        elif characteristics.query_complexity > 0.7:
            return "analytics"
        elif characteristics.temporal_locality > 0.8:
            return "streaming"
        else:
            return "mixed"

    def _generate_config(
        self,
        engine: str,
        profile: dict[str, Any],
        characteristics: WorkloadCharacteristics,
    ) -> dict[str, Any]:
        """Generate optimized configuration for an engine.

        TD-SDKENG-1 (review): only CollectionConfig-field overrides — the
        documented pattern `client.create_collection(..., engine=rec.engine,
        **rec.config_overrides)` now REJECTS unknown keywords (rightly), so
        the old tuning knobs that were silently dropped (flush_threshold_mb,
        pca_dimensions, ...) must not be emitted. The engines' real tuning
        surfaces (storage_config) are a follow-up.
        """
        config: dict[str, Any] = {}

        if engine == "sst":
            if characteristics.write_ratio > 0.7:
                config["compression"] = {"algorithm": "lz4"}
            else:
                config["compression"] = {"algorithm": "zstd"}

        return config

    def _generate_reasoning(
        self,
        engine: str,
        characteristics: WorkloadCharacteristics,
        goal: OptimizationGoal,
    ) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasons = []

        if engine == "sst":
            if characteristics.write_ratio > 0.6:
                reasons.append("High write ratio matches SST's write-optimized design")
            reasons.append("SST provides low latency for real-time applications")

        elif engine == "helix":
            reasons.append("HELIX excels at semantic similarity search")
            if characteristics.spatial_locality > 0.5:
                reasons.append("Spatial locality benefits from Hilbert curve encoding")

        elif engine == "viper":
            reasons.append("VIPER's columnar format is optimal for analytics")
            if characteristics.query_complexity > 0.5:
                reasons.append("Complex queries benefit from columnar storage")

        elif engine == "swift":
            reasons.append("SWIFT provides exact search for small datasets")
            reasons.append("In-memory storage enables ultra-low latency")

        elif engine == "nova":
            reasons.append("NOVA balances read and write performance")
            reasons.append("Progressive search adapts to query patterns")

        elif engine == "raptor":
            reasons.append("RAPTOR adapts to evolving workload patterns")
            if characteristics.hot_data_ratio > 0.3:
                reasons.append("Hot block caching improves repeated queries")

        if goal != OptimizationGoal.BALANCED:
            reasons.append(f"Optimized for {goal.value}")

        return "; ".join(reasons)

    def compare_engines(
        self,
        characteristics: WorkloadCharacteristics,
    ) -> list[EngineRecommendation]:
        """
        Compare all engines for the given characteristics.

        Args:
            characteristics: Workload characteristics

        Returns:
            List of recommendations for all engines, sorted by score
        """
        recommendations = []

        for engine in self.ENGINE_PROFILES:
            rec = self._get_engine_recommendation(engine, characteristics)
            if rec:
                recommendations.append(rec)

        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations

    def _engine_selectable(self, engine: str) -> bool:
        # TD-SDKENG-1: never hand back an engine the server would reject —
        # VIPER is deprecated (ADR-093), SWIFT/RAPTOR experimental, and
        # MMAP/HYBRID map to VIPER on the gRPC path and are absent from the
        # v2 REST engine options (round-3 review).
        return engine in ("sst", "nova", "helix")

    def _get_engine_recommendation(
        self,
        engine: str,
        characteristics: WorkloadCharacteristics,
    ) -> EngineRecommendation | None:
        """Get recommendation for a specific engine"""
        profile = self.ENGINE_PROFILES.get(engine)
        if not profile:
            return None
        # TD-SDKENG-1: no recommendation for engines the server rejects or
        # gates (VIPER deprecated, SWIFT/RAPTOR experimental).
        if not self._engine_selectable(engine):
            return None

        score = self._calculate_score(
            engine, profile, characteristics, OptimizationGoal.BALANCED
        )
        config = self._generate_config(engine, profile, characteristics)

        vector_count_k = characteristics.vector_count / 1000
        estimated_latency = (
            profile["latency_base"] + profile["latency_scale"] * vector_count_k
        )
        estimated_throughput = int(
            profile["throughput_base"] * (1 - 0.1 * vector_count_k / 100)
        )
        estimated_memory = int(
            profile["memory_base"] * (characteristics.vector_count / 10000)
        )

        return EngineRecommendation(
            engine=engine,
            confidence=score,
            reasoning=self._generate_reasoning(
                engine, characteristics, OptimizationGoal.BALANCED
            ),
            estimated_latency_ms=estimated_latency,
            estimated_throughput=max(1000, estimated_throughput),
            estimated_memory_mb=max(10, estimated_memory),
            config_overrides=config,
        )


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for ProximaDB collections.

    Example:
        >>> optimizer = HyperparameterOptimizer(client)
        >>> result = optimizer.optimize(
        ...     collection="my_collection",
        ...     goal=OptimizationGoal.LATENCY,
        ...     max_iterations=20
        ... )
        >>> print(f"Best config: {result.best_config}")
    """

    def __init__(self, client=None):
        """
        Initialize optimizer.

        Args:
            client: ProximaDBClient for evaluating configurations
        """
        self._client = client

    def optimize(
        self,
        collection: str,
        goal: OptimizationGoal = OptimizationGoal.LATENCY,
        max_iterations: int = 20,
        test_queries: list[list[float]] | None = None,
    ) -> OptimizationResult:
        """
        Optimize hyperparameters for a collection.

        Args:
            collection: Collection to optimize
            goal: Optimization goal
            max_iterations: Maximum optimization iterations
            test_queries: Optional test queries for evaluation

        Returns:
            OptimizationResult with best configuration
        """
        # Define searchable hyperparameters
        params = self._get_searchable_params()

        best_config = {p.name: p.current_value for p in params}
        best_score = float("-inf")
        history = []

        for i in range(max_iterations):
            # Generate candidate config
            candidate = self._generate_candidate(params, i, max_iterations)

            # Evaluate candidate
            score = self._evaluate_config(collection, candidate, goal, test_queries)

            history.append(
                {
                    "iteration": i,
                    "config": candidate.copy(),
                    "score": score,
                }
            )

            if score > best_score:
                best_score = score
                best_config = candidate.copy()

        initial_score = history[0]["score"] if history else 0
        improvement = (
            (best_score - initial_score) / abs(initial_score)
            if initial_score != 0
            else 0
        )

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            iterations=max_iterations,
            search_history=history,
            improvement_ratio=improvement,
        )

    def _get_searchable_params(self) -> list[HyperparameterConfig]:
        """Get list of searchable hyperparameters"""
        return [
            HyperparameterConfig(
                name="ef_search",
                current_value=64,
                min_value=16,
                max_value=512,
                step=16,
            ),
            HyperparameterConfig(
                name="ef_construction",
                current_value=128,
                min_value=64,
                max_value=512,
                step=32,
            ),
            HyperparameterConfig(
                name="m",
                current_value=16,
                min_value=4,
                max_value=64,
                step=4,
            ),
            HyperparameterConfig(
                name="bloom_filter_fpp",
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
            ),
            HyperparameterConfig(
                name="block_size",
                current_value=65536,
                allowed_values=[16384, 32768, 65536, 131072],
            ),
        ]

    def _generate_candidate(
        self,
        params: list[HyperparameterConfig],
        iteration: int,
        max_iterations: int,
    ) -> dict[str, Any]:
        """Generate a candidate configuration"""
        import random

        config = {}
        exploration_rate = 1 - (iteration / max_iterations)

        for param in params:
            if random.random() < exploration_rate:
                # Explore: random value
                if param.allowed_values:
                    config[param.name] = random.choice(param.allowed_values)
                elif param.min_value is not None and param.max_value is not None:
                    if param.step:
                        steps = int((param.max_value - param.min_value) / param.step)
                        config[param.name] = (
                            param.min_value + random.randint(0, steps) * param.step
                        )
                    else:
                        config[param.name] = random.uniform(
                            param.min_value, param.max_value
                        )
                else:
                    config[param.name] = param.current_value
            else:
                # Exploit: use current best
                config[param.name] = param.current_value

        return config

    def _evaluate_config(
        self,
        collection: str,
        config: dict[str, Any],
        goal: OptimizationGoal,
        test_queries: list[list[float]] | None,
    ) -> float:
        """Evaluate a configuration"""
        # Without a client, use heuristic scoring
        if not self._client or not test_queries:
            return self._heuristic_score(config, goal)

        # Run test queries and measure performance
        latencies = []
        for query in test_queries[:10]:  # Limit test queries
            start = time.time()
            try:
                self._client.search(collection, query, top_k=10)
                latencies.append((time.time() - start) * 1000)
            except Exception:
                latencies.append(1000)  # Penalty for errors

        avg_latency = sum(latencies) / len(latencies) if latencies else 1000

        if goal == OptimizationGoal.LATENCY:
            return 1000 / (avg_latency + 1)
        elif goal == OptimizationGoal.THROUGHPUT:
            return 1000 / (avg_latency + 1) * 10
        else:
            return 1000 / (avg_latency + 1)

    def _heuristic_score(self, config: dict[str, Any], goal: OptimizationGoal) -> float:
        """Heuristic scoring without actual evaluation"""
        score = 0.5

        # Prefer middle-ground values
        if "ef_search" in config:
            # Higher ef_search improves recall but increases latency
            if goal == OptimizationGoal.RECALL:
                score += config["ef_search"] / 512 * 0.2
            elif goal == OptimizationGoal.LATENCY:
                score += (1 - config["ef_search"] / 512) * 0.2

        if "m" in config:
            # Higher m improves recall but uses more memory
            if goal == OptimizationGoal.MEMORY:
                score += (1 - config["m"] / 64) * 0.2
            elif goal == OptimizationGoal.RECALL:
                score += config["m"] / 64 * 0.2

        return score


class AutoML:
    """
    Unified AutoML interface for ProximaDB.

    Combines workload prediction, engine selection, and optimization.

    Example:
        >>> from proximadb_sdk import ProximaDBClient
        >>> from proximadb_sdk.automl import AutoML
        >>>
        >>> client = ProximaDBClient("http://localhost:5678")
        >>> automl = AutoML(client)
        >>>
        >>> # Get engine recommendation
        >>> rec = automl.recommend_engine(
        ...     vector_count=100000,
        ...     vector_dimension=768,
        ...     read_ratio=0.8,
        ...     goal=OptimizationGoal.LATENCY
        ... )
        >>> print(f"Recommended: {rec.engine} ({rec.confidence:.2%} confidence)")
        >>>
        >>> # Create collection with recommended settings
        >>> client.create_collection(
        ...     "optimized_collection",
        ...     dimension=768,
        ...     engine=rec.engine,
        ...     **rec.config_overrides
        ... )
    """

    def __init__(self, client=None):
        """
        Initialize AutoML.

        Args:
            client: Optional ProximaDBClient
        """
        self._client = client
        self.predictor = WorkloadPredictor()
        self.selector = EngineSelector(client)
        self.optimizer = HyperparameterOptimizer(client)

    def recommend_engine(
        self,
        vector_count: int = 0,
        vector_dimension: int = 0,
        read_ratio: float = 0.5,
        write_ratio: float | None = None,
        target_latency_ms: float | None = None,
        target_throughput: int | None = None,
        memory_budget_mb: int | None = None,
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ) -> EngineRecommendation:
        """
        Get engine recommendation for given requirements.

        Args:
            vector_count: Expected number of vectors
            vector_dimension: Vector dimensionality
            read_ratio: Ratio of read operations (0-1)
            write_ratio: Ratio of write operations (defaults to 1-read_ratio)
            target_latency_ms: Target latency in milliseconds
            target_throughput: Target operations per second
            memory_budget_mb: Memory budget in MB
            goal: Optimization goal

        Returns:
            EngineRecommendation with engine choice and configuration
        """
        if write_ratio is None:
            write_ratio = 1 - read_ratio

        characteristics = WorkloadCharacteristics(
            read_ratio=read_ratio,
            write_ratio=write_ratio,
            vector_count=vector_count,
            vector_dimension=vector_dimension,
            target_latency_ms=target_latency_ms,
            target_throughput=target_throughput,
            memory_budget_mb=memory_budget_mb,
        )

        return self.selector.recommend(characteristics, goal)

    def auto_configure(
        self,
        collection: str,
        goal: OptimizationGoal = OptimizationGoal.LATENCY,
        max_iterations: int = 20,
    ) -> dict[str, Any]:
        """
        Automatically configure a collection for optimal performance.

        Args:
            collection: Collection to configure
            goal: Optimization goal
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized configuration dictionary
        """
        result = self.optimizer.optimize(
            collection=collection,
            goal=goal,
            max_iterations=max_iterations,
        )
        return result.best_config

    def analyze_workload(self) -> WorkloadCharacteristics:
        """
        Analyze current workload from observations.

        Returns:
            WorkloadCharacteristics from observed operations
        """
        return self.predictor.predict()

    def observe(
        self,
        operation: str,
        latency_ms: float,
        vector_count: int = 1,
    ) -> None:
        """
        Observe an operation for workload analysis.

        Args:
            operation: Operation type
            latency_ms: Operation latency
            vector_count: Number of vectors
        """
        self.predictor.observe_operation(operation, latency_ms, vector_count)


__all__ = [
    # Main classes
    "AutoML",
    "WorkloadPredictor",
    "EngineSelector",
    "HyperparameterOptimizer",
    # Data classes
    "WorkloadCharacteristics",
    "EngineRecommendation",
    "HyperparameterConfig",
    "OptimizationResult",
    # Enums
    "WorkloadType",
    "OptimizationGoal",
]
