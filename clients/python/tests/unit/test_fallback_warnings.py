#!/usr/bin/env python3
"""
Test that fallback warnings are properly shown during collection creation.
"""

import warnings
from unittest.mock import Mock, patch

import pytest

from proximadb_sdk.models import (
    CollectionConfig,
    DistanceMetric,
    IndexingAlgorithm,
    StorageEngine,
)
from proximadb_sdk.protocols.rest_sync import ProximaDBClient


class TestFallbackWarnings:
    """Test fallback warning system"""

    @patch("proximadb_sdk.protocols.rest_sync.ProximaDBClient._make_request")
    def test_distance_metric_no_fallback_warning(self, mock_request):
        """Test no warning for now-supported distance metric (all 13 metrics supported as of 2025-08)"""
        # Mock the server response
        mock_response = Mock()
        mock_response.json.return_value = {
            "collection": {
                "id": "test_collection",
                "config": {
                    "name": "test_collection",
                    "dimension": 128,
                    "distance_metric": "manhattan",  # Now supported natively
                    "storage_engine": "sst",
                    "primary_indexing_algorithm": "hnsw",
                },
                "created_at": 1643723400,
                "updated_at": 1643723400,
            }
        }
        mock_request.return_value = mock_response

        client = ProximaDBClient(url="http://localhost:5678")

        # Create collection with previously unsupported distance metric (now supported)
        config = CollectionConfig(
            name="test_no_fallback_warning",
            dimension=128,
            distance_metric=DistanceMetric.MANHATTAN,  # Now fully supported
        )

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collection = client.create_collection("test_no_fallback_warning", config)

            # Check that no distance metric fallback warning was issued
            distance_warnings = [
                warning
                for warning in w
                if "manhattan" in str(warning.message).lower()
                and "fallback" in str(warning.message).lower()
            ]
            assert len(distance_warnings) == 0

    @patch("proximadb_sdk.protocols.rest_sync.ProximaDBClient._make_request")
    def test_storage_engine_fallback_warning(self, mock_request):
        """Test warning for unsupported storage engine"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "collection": {
                "id": "test_collection",
                "config": {
                    "name": "test_collection",
                    "dimension": 256,
                    "distance_metric": "cosine",
                    "storage_engine": "sst",  # Server fell back to sst
                    "primary_indexing_algorithm": "hnsw",
                },
                "created_at": 1643723400,
                "updated_at": 1643723400,
            }
        }
        mock_request.return_value = mock_response

        client = ProximaDBClient(url="http://localhost:5678")

        config = CollectionConfig(
            name="test_storage_fallback",
            dimension=256,
            storage_engine=StorageEngine.MMAP,  # Should fallback to VIPER
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collection = client.create_collection("test_storage_fallback", config)

            assert len(w) == 1
            warning_message = str(w[0].message).lower()
            assert "mmap" in warning_message
            assert "fallback to 'sst'" in warning_message
            assert "server decision" in warning_message

    @patch("proximadb_sdk.protocols.rest_sync.ProximaDBClient._make_request")
    def test_indexing_algorithm_no_fallback_warning(self, mock_request):
        """Test no warning for now-supported indexing algorithm (LSH supported as of 2025-08)"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "collection": {
                "id": "test_collection",
                "config": {
                    "name": "test_collection",
                    "dimension": 384,
                    "distance_metric": "cosine",
                    "storage_engine": "sst",
                    "primary_indexing_algorithm": "lsh",  # Now supported natively
                },
                "created_at": 1643723400,
                "updated_at": 1643723400,
            }
        }
        mock_request.return_value = mock_response

        client = ProximaDBClient(url="http://localhost:5678")

        config = CollectionConfig(
            name="test_index_no_fallback",
            dimension=384,
            primary_indexing_algorithm=IndexingAlgorithm.LSH,  # Now fully supported
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collection = client.create_collection("test_index_no_fallback", config)

            # Check that no indexing algorithm fallback warning was issued
            indexing_warnings = [
                warning
                for warning in w
                if "lsh" in str(warning.message).lower()
                and "fallback" in str(warning.message).lower()
            ]
            assert len(indexing_warnings) == 0

    @patch("proximadb_sdk.protocols.rest_sync.ProximaDBClient._make_request")
    def test_storage_engine_only_fallback_warnings(self, mock_request):
        """Test warnings for storage engine fallbacks only (distance metrics and indexing algorithms now supported)"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "collection": {
                "id": "test_collection",
                "config": {
                    "name": "test_collection",
                    "dimension": 512,
                    "distance_metric": "hamming",  # Now supported natively (no fallback)
                    "storage_engine": "sst",  # Fell back from hybrid
                    "primary_indexing_algorithm": "lsh",  # Now supported natively (no fallback)
                },
                "created_at": 1643723400,
                "updated_at": 1643723400,
            }
        }
        mock_request.return_value = mock_response

        client = ProximaDBClient(url="http://localhost:5678")

        config = CollectionConfig(
            name="test_storage_only_fallbacks",
            dimension=512,
            distance_metric=DistanceMetric.HAMMING,  # Now fully supported (no fallback)
            storage_engine=StorageEngine.HYBRID,  # Fallback to VIPER
            primary_indexing_algorithm=IndexingAlgorithm.LSH,  # Now fully supported (no fallback)
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collection = client.create_collection("test_storage_only_fallbacks", config)

            # Should have 1 warning for storage engine only
            assert len(w) == 1

            warning_message = str(w[0].message).lower()
            # Distance metric should NOT be in fallback warning anymore
            assert "hamming" not in warning_message or "fallback" not in warning_message
            # Indexing algorithm should NOT be in fallback warning anymore
            assert "lsh" not in warning_message or "fallback" not in warning_message
            # Only storage engine fallback should be present
            assert "hybrid" in warning_message and "sst" in warning_message
            assert "server decision" in warning_message

    @patch("proximadb_sdk.protocols.rest_sync.ProximaDBClient._make_request")
    def test_no_warnings_for_supported_config(self, mock_request):
        """Test no warnings are issued for fully supported configurations"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "collection": {
                "id": "test_collection",
                "config": {
                    "name": "test_collection",
                    "dimension": 128,
                    "distance_metric": "manhattan",  # Now supported natively
                    "storage_engine": "sst",
                    "primary_indexing_algorithm": "lsh",  # Now supported natively
                },
                "created_at": 1643723400,
                "updated_at": 1643723400,
            }
        }
        mock_request.return_value = mock_response

        client = ProximaDBClient(url="http://localhost:5678")

        config = CollectionConfig(
            name="test_supported_config",
            dimension=128,
            distance_metric=DistanceMetric.MANHATTAN,  # Now fully supported (previously fallback)
            storage_engine=StorageEngine.NOVA,  # Supported
            primary_indexing_algorithm=IndexingAlgorithm.LSH,  # Now fully supported (previously fallback)
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collection = client.create_collection("test_supported_config", config)

            # Should have no fallback warnings (maybe filterable metadata warning)
            fallback_warnings = [
                warning for warning in w if "fallback" in str(warning.message).lower()
            ]
            assert len(fallback_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
