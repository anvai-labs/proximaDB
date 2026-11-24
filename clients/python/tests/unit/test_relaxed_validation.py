#!/usr/bin/env python3
"""
Test relaxed validation behavior in Python SDK.
Verifies that the SDK now allows server-unsupported configurations
and provides appropriate warnings about fallbacks.
"""

import pytest
from pydantic import ValidationError

from proximadb_sdk.models import (
    CollectionConfig,
    DistanceMetric,
    ServerCapabilities,
    StorageEngine,
)


class TestRelaxedValidation:
    """Test relaxed validation and server capabilities"""

    def test_distance_metric_capabilities(self):
        """Test distance metric capabilities detection"""
        caps = ServerCapabilities()

        # Test supported metrics (no fallback)
        assert caps.is_supported("distance_metric", "cosine")
        assert caps.is_supported("distance_metric", "euclidean")
        assert caps.is_supported("distance_metric", "dot_product")

        # Test metrics that are now supported (server has been enhanced)
        assert caps.is_supported("distance_metric", "manhattan")
        assert caps.is_supported("distance_metric", "hamming")
        assert caps.is_supported("distance_metric", "jaccard")

        # Test extended metrics (also now supported)
        assert caps.is_supported("distance_metric", "chebyshev")

        # Note: No distance metrics require fallback anymore (all supported natively)
        # Unknown metrics return None (no fallback defined)
        assert caps.get_fallback_for("distance_metric", "unknown_metric") is None

    def test_collection_builder_defaults_to_sst(self):
        """ADR-093 S3: the bare builder must default to SST (was VIPER — a live
        S2-era bug where every default collection creation was rejected)."""
        from proximadb_sdk.builders.collection import CollectionBuilder

        config = CollectionBuilder("defaulted", 8).build()
        assert config.storage_engine.value == "sst"

    def test_storage_engine_capabilities(self):
        """Test storage engine capabilities detection"""
        caps = ServerCapabilities()

        # Test supported engines (no fallback) — VIPER removed (ADR-093 S3)
        assert caps.is_supported("storage_engine", "sst")
        assert caps.is_supported("storage_engine", "nova")
        assert caps.is_supported("storage_engine", "helix")
        assert not caps.is_supported("storage_engine", "viper")

        # Test engines that fallback
        assert not caps.is_supported("storage_engine", "mmap")
        assert not caps.is_supported("storage_engine", "hybrid")

        # Test fallback values
        assert caps.get_fallback_for("storage_engine", "mmap") == "sst"
        assert caps.get_fallback_for("storage_engine", "hybrid") == "sst"

    def test_indexing_algorithm_capabilities(self):
        """Test indexing algorithm capabilities detection"""
        caps = ServerCapabilities()

        # Test supported algorithms (no fallback)
        assert caps.is_supported("indexing_algorithm", "hnsw")
        assert caps.is_supported("indexing_algorithm", "ivf")
        assert caps.is_supported("indexing_algorithm", "flat")
        assert caps.is_supported("indexing_algorithm", "annoy")
        assert caps.is_supported("indexing_algorithm", "pq")

        # Test algorithm that is now supported
        assert caps.is_supported("indexing_algorithm", "lsh")

        # Note: No indexing algorithms require fallback anymore (all supported natively)
        # Unknown algorithms return None (no fallback defined)
        assert caps.get_fallback_for("indexing_algorithm", "unknown_algo") is None

    def test_quantization_capabilities(self):
        """Test quantization type capabilities"""
        caps = ServerCapabilities()

        # All quantization types should be supported in VIPER
        assert caps.is_supported("quantization_type", "none")
        assert caps.is_supported("quantization_type", "uniform")
        assert caps.is_supported("quantization_type", "pq")
        assert caps.is_supported("quantization_type", "scalar")
        assert caps.is_supported("quantization_type", "binary")
        assert caps.is_supported("quantization_type", "custom")

    def test_extended_distance_metrics_enum(self):
        """Test that extended distance metrics are available in enum"""
        # Previously unsupported metrics should now be available
        assert DistanceMetric.MANHATTAN == "manhattan"
        assert DistanceMetric.HAMMING == "hamming"
        assert DistanceMetric.JACCARD == "jaccard"

        # New extended metrics should be available
        assert DistanceMetric.CHEBYSHEV == "chebyshev"
        assert DistanceMetric.CANBERRA == "canberra"
        assert DistanceMetric.MINKOWSKI == "minkowski"
        assert DistanceMetric.ANGULAR == "angular"
        assert DistanceMetric.BRAY_CURTIS == "bray_curtis"
        assert DistanceMetric.HELLINGER == "hellinger"

    def test_relaxed_name_validation(self):
        """Collection names are no longer required to be 8+ characters.

        The former 8-char minimum was a vector-collection-era constraint the
        server relaxed for relational-DDL tables (part / orders / region); the
        SDK must not be stricter than the server (#1113, TD-SDK-1). Only
        empty / whitespace-only names are rejected.
        """
        # 8+ character names still work
        assert (
            CollectionConfig(name="valid_collection_name", dimension=128).name
            == "valid_collection_name"
        )
        assert CollectionConfig(name="exactly8", dimension=128).name == "exactly8"

        # Short names are now ACCEPTED (relational tables)
        assert CollectionConfig(name="short", dimension=128).name == "short"
        assert CollectionConfig(name="a", dimension=128).name == "a"

        # Surrounding whitespace is stripped; the short result is still accepted
        assert CollectionConfig(name="  short  ", dimension=128).name == "short"

        # Empty names still fail (Pydantic min_length=1)
        with pytest.raises(ValidationError):
            CollectionConfig(name="", dimension=128)

        # Whitespace-only names still fail (validator rejects empty-after-strip)
        with pytest.raises(ValidationError):
            CollectionConfig(name="   ", dimension=128)

    def test_relaxed_dimension_validation(self):
        """Test that dimension validation is relaxed"""
        # Large dimensions should be allowed (server handles limits)
        config = CollectionConfig(
            name="test_large_dimension_collection",
            dimension=50000,  # Previously limited to 10000
        )
        assert config.dimension == 50000

        # Very large dimensions up to server limit (65536) should be allowed
        config = CollectionConfig(
            name="test_very_large_dimension_coll", dimension=65536
        )
        assert config.dimension == 65536

        # Zero dimensions should still fail
        with pytest.raises(ValueError):
            CollectionConfig(name="test_zero_dimension_coll", dimension=0)

    def test_collection_config_with_extended_options(self):
        """Test creating collection configs with extended options"""
        # Test collection with extended distance metric (8+ char name)
        config = CollectionConfig(
            name="test_extended_metrics_collection",
            dimension=512,
            distance_metric=DistanceMetric.CHEBYSHEV,
            storage_engine=StorageEngine.MMAP,
        )

        assert config.distance_metric == DistanceMetric.CHEBYSHEV
        assert config.storage_engine == StorageEngine.MMAP
        # Note: primary_indexing_algorithm is deprecated, now use index_configs

    def test_server_capabilities_notes(self):
        """Test server capabilities notes"""
        caps = ServerCapabilities()

        assert "fallback_policy" in caps.notes
        assert "dimension_limit" in caps.notes
        assert "name_validation" in caps.notes
        assert "quantization_engine" in caps.notes

        assert "intelligent fallbacks" in caps.notes["fallback_policy"]
        assert "65536" in caps.notes["dimension_limit"]
        assert "pax_vector_quant" in caps.notes["quantization_engine"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
