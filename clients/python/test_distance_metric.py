#!/usr/bin/env python3
"""Test distance metric issue with gRPC"""

import os
import sys
import time

sys.path.insert(0, "src")
os.environ["PROXIMADB_URL"] = "http://localhost:5678"
os.environ["PROXIMADB_GRPC_URL"] = "http://localhost:5679"

from proximadb import ProximaDBClient
from proximadb.models import CollectionConfig, DistanceMetric, StorageEngine


def test_distance_metrics():
    """Test all distance metrics via gRPC"""

    print("=" * 80)
    print("Distance Metric Test - gRPC")
    print("=" * 80)

    grpc_client = ProximaDBClient(protocol="grpc")
    rest_client = ProximaDBClient(protocol="rest")

    # Test all distance metrics
    test_metrics = [
        (DistanceMetric.COSINE, "cosine", 1),
        (DistanceMetric.EUCLIDEAN, "euclidean", 2),
        (DistanceMetric.DOT_PRODUCT, "dot_product", 3),
        (DistanceMetric.MANHATTAN, "manhattan", 5),
        (DistanceMetric.HAMMING, "hamming", 4),
    ]

    for metric_enum, metric_name, expected_value in test_metrics:
        collection_name = f"test_metric_{metric_name}_{int(time.time())}"

        print(f"\n📊 Testing {metric_name} (enum value: {expected_value})...")

        try:
            # Create collection with gRPC
            config = CollectionConfig(
                name=collection_name,
                dimension=128,
                distance_metric=metric_enum,
                storage_engine=StorageEngine.NOVA,
            )

            print(
                f"   Creating with distance_metric={metric_enum.name} (value={metric_enum.value})..."
            )
            collection = grpc_client.create_collection(collection_name, config)

            # Check what was created
            print(f"   ✅ Created collection: {collection.id}")
            print(
                f"   📋 Config returned: distance_metric={collection.config.distance_metric.name} (value={collection.config.distance_metric.value})"
            )

            # Verify with get_collection
            retrieved = grpc_client.get_collection(collection_name)
            print(
                f"   🔍 Retrieved config: distance_metric={retrieved.config.distance_metric.name} (value={retrieved.config.distance_metric.value})"
            )

            if retrieved.config.distance_metric != metric_enum:
                print(
                    f"   ❌ MISMATCH: Expected {metric_enum.name}, got {retrieved.config.distance_metric.name}"
                )
            else:
                print(f"   ✅ SUCCESS: Distance metric preserved correctly")

            # Cleanup
            grpc_client.delete_collection(collection_name)

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_distance_metrics()
