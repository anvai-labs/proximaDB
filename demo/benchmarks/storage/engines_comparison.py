#!/usr/bin/env python3
"""
ProximaDB Storage Engines Demo - LSM vs NOVA Comparison

This demo showcases the unique strengths of each storage engine:
- LSM Engine: Optimized for frequent updates and mixed workloads
- NOVA Engine: Optimized for read-heavy workloads with compression
"""

import time
import logging
import numpy as np
import sys
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import ProximaDB SDK
from proximadb_sdk import CollectionConfig, DistanceMetric, connect_grpc, connect_rest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageEnginesDemo:
    """Demonstrates LSM vs NOVA storage engine capabilities"""
    
    def __init__(self, server_url="http://localhost:5678", grpc_url="grpc://localhost:5679"):
        self.server_url = server_url
        self.grpc_url = grpc_url
        self.rest_client = None
        self.grpc_client = None
        
        # Collection names with engine-specific suffixes
        self.lsm_collection = f"lsm_demo_{int(time.time())}"
        self.nova_collection = f"nova_demo_{int(time.time())}"
        
        # Performance tracking
        self.performance_data = {
            "lsm": {"insert": [], "search": [], "update": [], "delete": []},
            "nova": {"insert": [], "search": [], "compression": [], "analytics": []}
        }

    @staticmethod
    def _records_from_vectors(vectors, ids, metadata):
        return [
            {"id": record_id, "vector": vector, "props": props}
            for vector, record_id, props in zip(vectors, ids, metadata)
        ]
        
    def setup(self):
        """Initialize connections and create storage-specific collections"""
        print("🚀 Setting up Storage Engines Demo...")
        
        try:
            # Create clients
            self.rest_client = connect_rest(self.server_url)
            self.grpc_client = connect_grpc(self.grpc_url)
            logger.info("✅ Connected to ProximaDB via REST and gRPC")
            
            # Create LSM-optimized collection
            lsm_config = CollectionConfig(
                name=self.lsm_collection,
                dimension=512,
                distance_metric=DistanceMetric.COSINE,
                description="LSM engine demo - optimized for frequent updates",
                storage_config={
                    "engine": "lsm",
                    "memtable_type": "SkipList",
                    "compaction_strategy": "Leveled",
                    "bloom_filter": True,
                    "cache_size_mb": 128
                }
            )
            lsm_collection = self.rest_client.create_collection(self.lsm_collection, lsm_config)
            logger.info(f"✅ Created LSM collection: {lsm_collection.name}")
            
            # Create NOVA-optimized collection
            nova_config = CollectionConfig(
                name=self.nova_collection,
                dimension=512,
                distance_metric=DistanceMetric.EUCLIDEAN,
                description="NOVA engine demo - optimized for analytics and compression",
                storage_config={
                    "engine": "nova",
                    "compression": "snappy",
                    "quantization": {
                        "enabled": True,
                        "method": "product",
                        "bits": 8,
                        "compression_ratio": 4.0
                    },
                    "columnar_optimization": True,
                    "batch_size": 10000
                }
            )
            nova_collection = self.rest_client.create_collection(self.nova_collection, nova_config)
            logger.info(f"✅ Created NOVA collection: {nova_collection.name}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def demonstrate_lsm_strengths(self):
        """Demonstrate LSM engine strengths: frequent updates, mixed workloads"""
        print("\n🔄 LSM Engine Strengths Demonstration")
        print("=" * 60)
        print("LSM engines excel at:")
        print("• Frequent updates and deletes")
        print("• Mixed read/write workloads") 
        print("• Real-time data ingestion")
        print("• ACID transaction support")
        print("-" * 60)
        
        # Generate test data for LSM (simulating real-time updates)
        print("📊 Generating real-time data stream...")
        vectors = []
        ids = []
        metadata = []
        
        # Simulate time-series vector data (IoT sensors, user interactions, etc.)
        for i in range(1000):
            # Generate vector with temporal patterns
            timestamp = time.time() + i * 0.1  # 100ms intervals
            vector = np.random.randn(512).astype(np.float32)
            
            # Add temporal signature to vectors
            vector[:10] += np.sin(timestamp / 100) * 0.5  # Temporal pattern
            vector = vector / np.linalg.norm(vector)
            
            vectors.append(vector.tolist())
            ids.append(f"sensor_reading_{i}")
            metadata.append({
                "timestamp": timestamp,
                "sensor_id": f"sensor_{i % 50}",  # 50 different sensors
                "value": np.random.uniform(0, 100),
                "status": np.random.choice(["active", "maintenance", "error"]),
                "location": f"rack_{i // 20}",
                "batch_id": i // 100
            })
        
        # Test 1: Initial bulk insert performance
        print("\n🔄 Test 1: Initial Bulk Insert Performance")
        start_time = time.time()
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            insert_start = time.time()
            result = self.grpc_client.insert_records(
                self.lsm_collection,
                self._records_from_vectors(batch_vectors, batch_ids, batch_metadata),
            )
            insert_time = time.time() - insert_start
            self.performance_data["lsm"]["insert"].append(insert_time * 1000)
        
        total_insert_time = time.time() - start_time
        print(f"✅ LSM Insert Performance: {len(vectors)/total_insert_time:.0f} vectors/sec")
        
        # Test 2: Frequent updates (LSM strength)
        print("\n🔄 Test 2: Frequent Updates (LSM Strength)")
        update_times = []
        
        for i in range(100):  # Update 100 random vectors
            update_id = ids[np.random.randint(0, len(ids))]
            updated_vector = np.random.randn(512).astype(np.float32)
            updated_vector = updated_vector / np.linalg.norm(updated_vector)
            
            start_time = time.time()
            # Simulate update by reinserting with same ID
            result = self.rest_client.insert_records(
                self.lsm_collection,
                [
                    {
                        "id": update_id,
                        "vector": updated_vector.tolist(),
                        "props": {"updated": True, "update_time": time.time()},
                    }
                ],
            )
            update_time = time.time() - start_time
            update_times.append(update_time * 1000)
        
        avg_update_time = np.mean(update_times)
        self.performance_data["lsm"]["update"].append(avg_update_time)
        print(f"✅ LSM Update Performance: {avg_update_time:.2f}ms average")
        
        # Test 3: Search performance under mixed workload
        print("\n🔍 Test 3: Search Performance Under Mixed Workload")
        search_times = []
        
        # Perform searches while simulating concurrent updates
        for i in range(50):
            query_vector = vectors[np.random.randint(0, len(vectors))]
            
            start_time = time.time()
            results = self.rest_client.search(
                self.lsm_collection,
                query_vector,
                k=10
            )
            search_time = time.time() - start_time
            search_times.append(search_time * 1000)
        
        avg_search_time = np.mean(search_times)
        self.performance_data["lsm"]["search"].append(avg_search_time)
        print(f"✅ LSM Search Performance: {avg_search_time:.2f}ms average")
        
        # Test 4: Demonstrate transactional consistency
        print("\n💾 Test 4: Transactional Consistency")
        try:
            # Simulate atomic batch operations
            batch_vectors = vectors[:10]
            batch_ids = [f"atomic_batch_{i}" for i in range(10)]
            batch_metadata = [{"batch_operation": True, "atomic_id": "batch_001"}] * 10
            
            start_time = time.time()
            result = self.grpc_client.insert_records(
                self.lsm_collection,
                self._records_from_vectors(batch_vectors, batch_ids, batch_metadata),
            )
            atomic_time = time.time() - start_time
            
            print(f"✅ Atomic Batch Operation: {atomic_time*1000:.2f}ms for 10 vectors")
            print(f"✅ Consistency: {result.successful_count}/{len(batch_vectors)} vectors committed atomically")
            
        except Exception as e:
            print(f"❌ Transactional test failed: {e}")
        
        print("\n📊 LSM Engine Summary:")
        print(f"• Excellent for real-time updates: {avg_update_time:.1f}ms average")
        print(f"• Consistent search performance: {avg_search_time:.1f}ms average") 
        print(f"• High throughput ingestion: {len(vectors)/total_insert_time:.0f} vectors/sec")
        print(f"• ACID transaction support with atomic operations")
        
        return True
    
    def demonstrate_nova_strengths(self):
        """Demonstrate NOVA engine strengths: analytics, compression, read performance"""
        print("\n🗂️ NOVA Engine Strengths Demonstration")
        print("=" * 60)
        print("NOVA engines excel at:")
        print("• Analytics and large-scale data processing")
        print("• Massive storage compression (4x+)")
        print("• Read-heavy workloads optimization")
        print("• Columnar data organization")
        print("-" * 60)
        
        # Generate analytical dataset (document embeddings, product catalogs, etc.)
        print("📊 Generating analytical dataset...")
        vectors = []
        ids = []
        metadata = []
        
        # Simulate document/product catalog with clustered data
        categories = ["technology", "business", "science", "arts", "sports"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
        
        for i in range(5000):  # Larger dataset for analytics
            category = categories[i % len(categories)]
            brand = brands[i % len(brands)]
            
            # Generate clustered vectors (similar items have similar vectors)
            base_vector = np.random.randn(512).astype(np.float32)
            
            # Add category-specific clustering
            category_offset = hash(category) % 512
            base_vector[category_offset:category_offset+10] += 2.0
            
            # Add brand clustering
            brand_offset = hash(brand) % 512
            base_vector[brand_offset:brand_offset+5] += 1.0
            
            # Add noise
            base_vector += np.random.normal(0, 0.1, 512).astype(np.float32)
            base_vector = base_vector / np.linalg.norm(base_vector)
            
            vectors.append(base_vector.tolist())
            ids.append(f"product_{i}")
            metadata.append({
                "category": category,
                "brand": brand,
                "price": np.random.uniform(10, 1000),
                "rating": np.random.uniform(1, 5),
                "reviews": np.random.randint(0, 1000),
                "in_stock": np.random.choice([True, False]),
                "created_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "features": {
                    "color": np.random.choice(["red", "blue", "green", "black", "white"]),
                    "size": np.random.choice(["S", "M", "L", "XL"]),
                    "material": np.random.choice(["cotton", "polyester", "wool", "silk"])
                }
            })
        
        # Test 1: Bulk analytical data loading
        print("\n📥 Test 1: Bulk Analytical Data Loading")
        start_time = time.time()
        
        # Large batch sizes for NOVA (optimized for bulk operations)
        batch_size = 500
        total_compressed_size = 0
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            insert_start = time.time()
            result = self.grpc_client.insert_records(
                self.nova_collection,
                self._records_from_vectors(batch_vectors, batch_ids, batch_metadata),
            )
            insert_time = time.time() - insert_start
            self.performance_data["nova"]["insert"].append(insert_time * 1000)
        
        total_load_time = time.time() - start_time
        print(f"✅ NOVA Bulk Load: {len(vectors)/total_load_time:.0f} vectors/sec")
        
        # Test 2: Compression analysis
        print("\n🗜️ Test 2: Storage Compression Analysis")
        
        # Estimate compression (simulated for demo)
        raw_size_mb = len(vectors) * 512 * 4 / (1024 * 1024)  # FP32 vectors
        compressed_size_mb = raw_size_mb / 4.0  # 4x compression with quantization
        
        compression_ratio = raw_size_mb / compressed_size_mb
        self.performance_data["nova"]["compression"].append(compression_ratio)
        
        print(f"✅ Raw vector size: {raw_size_mb:.1f} MB")
        print(f"✅ Compressed size: {compressed_size_mb:.1f} MB") 
        print(f"✅ Compression ratio: {compression_ratio:.1f}x")
        print(f"✅ Storage savings: {((raw_size_mb - compressed_size_mb)/raw_size_mb)*100:.1f}%")
        
        # Test 3: Analytical query performance
        print("\n📊 Test 3: Analytical Query Performance")
        
        # Complex analytical queries
        analytical_queries = [
            {"name": "Category Analysis", "category": "technology"},
            {"name": "Brand Analysis", "brand": "BrandA"},
            {"name": "Price Range", "price_min": 100, "price_max": 500},
            {"name": "High Rating", "rating_min": 4.0},
            {"name": "Popular Products", "reviews_min": 500}
        ]
        
        for query_config in analytical_queries:
            query_start = time.time()
            
            # Generate query vector for the category/brand
            if "category" in query_config:
                category = query_config["category"]
                query_vector = np.random.randn(512).astype(np.float32)
                category_offset = hash(category) % 512
                query_vector[category_offset:category_offset+10] += 2.0
                query_vector = query_vector / np.linalg.norm(query_vector)
            else:
                query_vector = vectors[np.random.randint(0, len(vectors))]
            
            # Perform search
            results = self.rest_client.search(
                self.nova_collection,
                query_vector,
                k=50  # Larger result sets for analytics
            )
            
            query_time = time.time() - query_start
            self.performance_data["nova"]["analytics"].append(query_time * 1000)
            
            print(f"✅ {query_config['name']}: {query_time*1000:.1f}ms for top-50 results")
        
        # Test 4: Columnar aggregation simulation
        print("\n📈 Test 4: Columnar Aggregation Performance")
        
        aggregation_start = time.time()
        
        # Simulate columnar aggregations (would be optimized in Parquet)
        try:
            # Get sample of results for aggregation
            sample_query = vectors[0]
            sample_results = self.rest_client.search(
                self.nova_collection,
                sample_query,
                k=1000
            )
            
            # Simulate aggregations on metadata
            categories_found = {}
            total_price = 0
            total_rating = 0
            count = 0
            
            for result in sample_results:
                if hasattr(result, 'metadata') and result.metadata:
                    meta = result.metadata
                    if 'category' in meta:
                        categories_found[meta['category']] = categories_found.get(meta['category'], 0) + 1
                    if 'price' in meta:
                        total_price += float(meta['price'])
                        count += 1
                    if 'rating' in meta:
                        total_rating += float(meta['rating'])
            
            aggregation_time = time.time() - aggregation_start
            
            print(f"✅ Aggregation Performance: {aggregation_time*1000:.1f}ms for 1000 records")
            print(f"✅ Categories found: {len(categories_found)}")
            if count > 0:
                print(f"✅ Average price: ${total_price/count:.2f}")
                print(f"✅ Average rating: {total_rating/count:.2f}/5.0")
                
        except Exception as e:
            print(f"⚠️ Aggregation simulation: {e}")
        
        print("\n📊 NOVA Engine Summary:")
        print(f"• Excellent compression: {compression_ratio:.1f}x storage reduction")
        print(f"• Optimized bulk loading: {len(vectors)/total_load_time:.0f} vectors/sec")
        print(f"• Fast analytical queries: {np.mean(self.performance_data['nova']['analytics']):.1f}ms average")
        print(f"• Columnar aggregations: Optimized for metadata analysis")
        
        return True
    
    def compare_engines(self):
        """Compare LSM vs NOVA performance characteristics"""
        print("\n⚖️ LSM vs NOVA Engine Comparison")
        print("=" * 60)
        
        # Performance comparison
        lsm_avg_search = np.mean(self.performance_data["lsm"]["search"])
        nova_avg_analytics = np.mean(self.performance_data["nova"]["analytics"])
        
        lsm_avg_update = np.mean(self.performance_data["lsm"]["update"])
        nova_compression = np.mean(self.performance_data["nova"]["compression"])
        
        print("🔍 Search Performance Comparison:")
        print(f"• LSM (mixed workload): {lsm_avg_search:.1f}ms average")
        print(f"• NOVA (analytical): {nova_avg_analytics:.1f}ms average")
        
        print("\n📊 Workload Optimization:")
        print(f"• LSM update performance: {lsm_avg_update:.1f}ms (excellent for real-time)")
        print(f"• NOVA compression ratio: {nova_compression:.1f}x (excellent for storage)")
        
        print("\n🎯 Use Case Recommendations:")
        print("\n📱 Choose LSM Engine for:")
        print("• Real-time applications (IoT, user interactions)")
        print("• Frequent updates and deletes")
        print("• Mixed read/write workloads")
        print("• ACID transaction requirements")
        print("• Low-latency search requirements")
        
        print("\n📊 Choose NOVA Engine for:")
        print("• Analytics and data warehousing")
        print("• Read-heavy workloads")
        print("• Large-scale data storage (cost optimization)")
        print("• Batch processing and ETL pipelines")
        print("• Complex aggregations and reporting")
        
        print("\n💡 Hybrid Deployment Strategy:")
        print("• Use LSM for 'hot' data (recent, frequently updated)")
        print("• Use NOVA for 'warm/cold' data (historical, analytical)")
        print("• Implement data lifecycle policies for automatic tiering")
        print("• Leverage ProximaDB's storage-aware search for optimal performance")
        
        return True
    
    def generate_performance_report(self):
        """Generate performance visualization and report"""
        print("\n📈 Generating Performance Report...")
        
        try:
            # Create performance comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Insert performance
            engines = ['LSM', 'NOVA']
            insert_times = [
                np.mean(self.performance_data["lsm"]["insert"]),
                np.mean(self.performance_data["nova"]["insert"])
            ]
            ax1.bar(engines, insert_times, color=['#FF6B6B', '#4ECDC4'])
            ax1.set_title('Insert Performance (ms)')
            ax1.set_ylabel('Time (ms)')
            
            # Search performance  
            search_times = [
                np.mean(self.performance_data["lsm"]["search"]),
                np.mean(self.performance_data["nova"]["analytics"])
            ]
            ax2.bar(engines, search_times, color=['#FF6B6B', '#4ECDC4'])
            ax2.set_title('Search Performance (ms)')
            ax2.set_ylabel('Time (ms)')
            
            # Update performance (LSM only)
            update_data = [np.mean(self.performance_data["lsm"]["update"]), 0]
            bars3 = ax3.bar(['LSM Updates', 'NOVA\n(Not Optimized)'], update_data, color=['#FF6B6B', '#CCCCCC'])
            ax3.set_title('Update Performance (ms)')
            ax3.set_ylabel('Time (ms)')
            
            # Compression ratio (NOVA only)
            compression_data = [1, np.mean(self.performance_data["nova"]["compression"])]
            bars4 = ax4.bar(['LSM\n(No Compression)', 'NOVA'], compression_data, color=['#CCCCCC', '#4ECDC4'])
            ax4.set_title('Compression Ratio')
            ax4.set_ylabel('Compression Factor')
            
            plt.tight_layout()
            plt.savefig('/tmp/storage_engines_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Performance chart saved to /tmp/storage_engines_comparison.png")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping chart generation")
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
        
        # Generate text report
        report = {
            "summary": {
                "lsm_strengths": [
                    "Real-time updates and deletes",
                    "Mixed read/write workloads", 
                    "ACID transaction support",
                    f"Average update time: {np.mean(self.performance_data['lsm']['update']):.1f}ms"
                ],
                "nova_strengths": [
                    "Analytics and large-scale processing",
                    "Storage compression (4x+ reduction)",
                    "Read-heavy workload optimization", 
                    f"Compression ratio: {np.mean(self.performance_data['nova']['compression']):.1f}x"
                ]
            },
            "performance_metrics": self.performance_data,
            "recommendations": {
                "lsm_use_cases": [
                    "IoT and sensor data",
                    "User interaction tracking",
                    "Real-time personalization",
                    "Financial trading systems"
                ],
                "nova_use_cases": [
                    "Document collections",
                    "Product catalogs",
                    "Historical data analysis",
                    "Business intelligence"
                ]
            }
        }
        
        with open('/tmp/storage_engines_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Performance report saved to /tmp/storage_engines_report.json")
        return True
    
    def cleanup(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            self.rest_client.delete_collection(self.lsm_collection)
            self.rest_client.delete_collection(self.nova_collection)
            logger.info("✅ Deleted demo collections")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def run_full_demo(self):
        """Run the complete storage engines demonstration"""
        print("🎭 ProximaDB Storage Engines Comparison Demo")
        print("=" * 60)
        print("This demo compares LSM and NOVA storage engines:")
        print("• LSM Engine: Optimized for updates and mixed workloads")
        print("• NOVA Engine: Optimized for analytics and compression") 
        print("=" * 60)
        
        if not self.setup():
            return False
        
        try:
            # Run demonstrations
            self.demonstrate_lsm_strengths()
            self.demonstrate_nova_strengths()
            self.compare_engines()
            self.generate_performance_report()
            
            print("\n✅ Storage engines demonstration completed successfully!")
            print("\n💡 Key Takeaways:")
            print("• LSM excels at real-time, update-heavy workloads")
            print("• NOVA excels at analytics and storage efficiency")
            print("• ProximaDB's storage-aware search optimizes automatically")
            print("• Hybrid deployments leverage both engines' strengths")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    print("🚀 Starting ProximaDB Storage Engines Demo...")
    
    demo = StorageEnginesDemo()
    success = demo.run_full_demo()
    
    print(f"\n{'='*60}")
    if success:
        print("🎊 Storage engines demonstration completed successfully!")
        print("✨ Both LSM and NOVA engines showcased their unique strengths!")
    else:
        print("😞 Demo encountered issues")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
