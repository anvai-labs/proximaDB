#!/usr/bin/env python3
"""
ProximaDB Demo Setup Script - Uses Pre-generated Data

This script loads pre-generated demo data into ProximaDB collections.
The data must be generated first using: python generate_datasets_only.py

Usage:
    python demo_setup.py [--skip-existing]
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add path utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.path_utils import setup_demo_environment

# Setup environment
env_info = setup_demo_environment()
from proximadb import ProximaDBClient, Protocol, ClientConfig
from proximadb import (
    CollectionConfig,
    DistanceMetric,
    StorageEngine,
    IndexType,
    VectorRecord,
)

# Configuration
PROXIMADB_REST_URL = os.getenv("PROXIMADB_URL", "http://localhost:5678")
PROXIMADB_GRPC_URL = os.getenv("PROXIMADB_GRPC_URL", "http://localhost:5679")
PRE_DIR = Path("pre")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemoSetup:
    """Setup demo collections using pre-generated data"""
    
    def __init__(self, skip_existing: bool = False):
        self.skip_existing = skip_existing
        self.client = self._create_client()
        self.collections_created = []
        
    def _create_client(self) -> ProximaDBClient:
        """Create ProximaDB client with optimal settings"""
        config = ClientConfig(
            url=PROXIMADB_REST_URL,
            grpc_url=PROXIMADB_GRPC_URL,
        )
        # Use gRPC for better performance
        return ProximaDBClient(protocol=Protocol.GRPC, config=config)
    
    def _load_json_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from pre-generated JSON file"""
        filepath = PRE_DIR / filename
        if not filepath.exists():
            logger.error(f"Pre-generated data not found: {filepath}")
            logger.error("Please run: python generate_datasets_only.py")
            raise FileNotFoundError(f"Missing {filepath}")
        
        logger.info(f"Loading {filename}...")
        with open(filepath) as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} items from {filename}")
        return data
    
    def _create_collection(self, name: str, config: CollectionConfig) -> bool:
        """Create a collection with the given configuration"""
        try:
            # Check if exists
            if self.skip_existing:
                try:
                    existing = self.client.get_collection(name)
                    logger.warning(f"Collection '{name}' already exists, skipping...")
                    return False
                except:
                    pass  # Collection doesn't exist, proceed
            
            # Create collection
            logger.info(f"Creating collection: {name}")
            collection = self.client.create_collection(name=name, config=config)
            self.collections_created.append(name)
            logger.info(f"✅ Collection '{name}' created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            return False
    
    def _batch_insert(self, collection_name: str, data: List[Dict[str, Any]], 
                     batch_size: int = 100):
        """Insert vectors in batches"""
        logger.info(f"Inserting {len(data)} vectors into '{collection_name}'...")
        
        start_time = time.time()
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            vectors = []
            
            for item in batch:
                # Extract vector and metadata
                vector_record = VectorRecord(
                    id=item["id"],
                    vector=item["vector"],
                    metadata={k: v for k, v in item.items() if k not in ["id", "vector"]}
                )
                vectors.append(vector_record)
            
            try:
                self.client.insert_vectors(collection_name, records=vectors)
                total_inserted += len(vectors)
                
                # Progress update every 10 batches
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  Progress: {total_inserted}/{len(data)} vectors")
                    
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Inserted {total_inserted} vectors in {elapsed:.2f}s")
    
    def setup_ecommerce_collection(self):
        """Setup e-commerce demo collection"""
        logger.info("\n🛍️  Setting up E-commerce Collection...")
        
        # Load data
        data = self._load_json_data("ecommerce_data.json")
        
        # Create collection
        config = CollectionConfig(
            name="ecommerce_demo",
            dimension=768,
            distance_metric=DistanceMetric.COSINE,
            storage_engine=StorageEngine.NOVA,
            primary_indexing_algorithm=IndexType.HNSW,
            filterable_metadata_fields=[
                "category", "brand", "price", "rating", "in_stock", "tags", "created_at"
            ],
            description="E-commerce product catalog with NOVA storage engine",
            tags=["demo", "ecommerce", "bert", "nova"]
        )
        
        if self._create_collection("ecommerce_demo", config):
            self._batch_insert("ecommerce_demo", data)
    
    def setup_sec_edgar_collection(self):
        """Setup SEC EDGAR filings collection"""
        logger.info("\n📄 Setting up SEC EDGAR Collection...")
        
        # Load data
        data = self._load_json_data("sec_edgar_data.json")
        
        # Create collection
        config = CollectionConfig(
            name="sec_edgar_large_filings",
            dimension=768,
            distance_metric=DistanceMetric.COSINE,
            storage_engine=StorageEngine.NOVA,
            primary_indexing_algorithm=IndexType.HNSW,
            filterable_metadata_fields=[
                "company", "ticker", "filing_type", "section", "fiscal_year"
            ],
            description="SEC EDGAR filing documents from S&P 50 companies",
            tags=["demo", "sec", "financial", "nova"]
        )
        
        if self._create_collection("sec_edgar_large_filings", config):
            # Use larger batches for SEC data
            self._batch_insert("sec_edgar_large_filings", data, batch_size=200)
    
    def setup_knowledge_base_collection(self):
        """Setup knowledge base collection"""
        logger.info("\n📚 Setting up Knowledge Base Collection...")
        
        # Load data
        data = self._load_json_data("knowledge_base_data.json")
        
        # Create collection
        config = CollectionConfig(
            name="knowledge_base",
            dimension=768,
            distance_metric=DistanceMetric.COSINE,
            storage_engine=StorageEngine.SST,
            primary_indexing_algorithm=IndexType.HNSW,
            filterable_metadata_fields=[
                "document_type", "source", "chunk_index", "document_id", "language", "confidence_score"
            ],
            description="Technical knowledge base with SST storage engine",
            tags=["demo", "rag", "knowledge", "sst"]
        )
        
        if self._create_collection("knowledge_base", config):
            self._batch_insert("knowledge_base", data, batch_size=50)
    
    def setup(self):
        """Run complete setup"""
        logger.info("Starting ProximaDB demo setup...")
        logger.info(f"Using pre-generated data from: {PRE_DIR.absolute()}")
        
        # Check if pre-generated data exists
        if not PRE_DIR.exists():
            logger.error(f"Pre-generated data directory not found: {PRE_DIR}")
            logger.error("Please run: python generate_datasets_only.py")
            sys.exit(1)
        
        # Setup collections
        self.setup_ecommerce_collection()
        self.setup_sec_edgar_collection()
        self.setup_knowledge_base_collection()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print setup summary"""
        logger.info("\n" + "=" * 80)
        logger.info("✅ ProximaDB Demo Setup Complete!")
        logger.info("=" * 80)
        
        if self.collections_created:
            logger.info(f"\nCollections created: {len(self.collections_created)}")
            for collection in self.collections_created:
                logger.info(f"  - {collection}")
        
        logger.info("\nDemo data loaded:")
        logger.info("  - E-commerce: 1,200 products")
        logger.info("  - SEC EDGAR: 13,320 document chunks from S&P 50")
        logger.info("  - Knowledge Base: 256 technical articles")
        
        logger.info("\nNext steps:")
        logger.info("1. Run feature demos: python feature_showcase_consolidated.py")
        logger.info("2. Run benchmarks: python benchmarks/performance_suite.py")
        logger.info("3. Access the web UI: http://localhost:8080")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ProximaDB Demo Setup with Pre-generated Data")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip creation of existing collections"
    )
    
    args = parser.parse_args()
    
    # Run setup
    setup = DemoSetup(skip_existing=args.skip_existing)
    setup.setup()


if __name__ == "__main__":
    main()