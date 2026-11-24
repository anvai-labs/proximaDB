#!/usr/bin/env python3
"""
ProximaDB E-commerce Demo with Optimized Metadata Storage
=========================================================

This demo showcases:
- Smart metadata separation for the columnar engine
- Real BERT embeddings for semantic product search
- High-performance gRPC for bulk operations
- Flexible search: REST, gRPC, and SQL options
- Rich metadata filtering with proper indexing

Prerequisites:
    pip install sentence-transformers proximadb-client

Usage:
    python ecommerce_demo.py
"""

import time
import json
import requests
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from proximadb import connect_grpc, connect_rest, Protocol
from proximadb import CollectionConfig, DistanceMetric, VectorRecord, StorageEngine

# Comprehensive real product dataset for semantic search
REAL_PRODUCTS = [
    # Electronics - Laptops
    {"id": "laptop_001", "name": "Dell XPS 13 Plus", "category": "Electronics", "subcategory": "Laptops",
     "description": "Ultra-slim laptop with 12th Gen Intel Core i7, 13.4-inch 4K+ touchscreen, 16GB LPDDR5 RAM, 512GB SSD. Features edge-to-edge InfinityEdge display, premium aluminum construction, and exceptional battery life for professionals.",
     "brand": "Dell", "price": 1649.99, "rating": 4.6, "in_stock": True, "tags": ["premium", "business", "4K", "touchscreen"]},
    
    {"id": "laptop_002", "name": "MacBook Air M2", "category": "Electronics", "subcategory": "Laptops", 
     "description": "Revolutionary MacBook Air with Apple M2 chip, 13.6-inch Liquid Retina display, 8-core CPU, 8-core GPU, 16-core Neural Engine, 8GB unified memory, 256GB SSD. Silent fanless design with up to 18 hours battery life.",
     "brand": "Apple", "price": 1199.99, "rating": 4.8, "in_stock": True, "tags": ["M2", "fanless", "premium", "mac"]},
    
    {"id": "laptop_003", "name": "ASUS ROG Strix G15", "category": "Electronics", "subcategory": "Laptops",
     "description": "Gaming laptop powered by AMD Ryzen 9 5900HX, NVIDIA GeForce RTX 3070, 15.6-inch 300Hz IPS display, 16GB DDR4 RAM, 1TB NVMe SSD. Advanced cooling system and RGB backlit keyboard for serious gamers.",
     "brand": "ASUS", "price": 1599.99, "rating": 4.5, "in_stock": True, "tags": ["gaming", "RGB", "high-refresh", "RTX"]},
    
    # Electronics - Smartphones
    {"id": "phone_001", "name": "iPhone 14 Pro Max", "category": "Electronics", "subcategory": "Smartphones",
     "description": "Pro camera system with 48MP Main camera, Dynamic Island, A16 Bionic chip, 6.7-inch Super Retina XDR display with ProMotion, 5G connectivity, Face ID, and Ceramic Shield front. Available in Space Black, Silver, Gold, Deep Purple.",
     "brand": "Apple", "price": 1099.99, "rating": 4.7, "in_stock": True, "tags": ["48MP", "Dynamic Island", "Pro", "5G"]},
    
    {"id": "phone_002", "name": "Samsung Galaxy S23 Ultra", "category": "Electronics", "subcategory": "Smartphones",
     "description": "Premium Android smartphone with 200MP camera, built-in S Pen, 6.8-inch Dynamic AMOLED 2X display, Snapdragon 8 Gen 2, 12GB RAM, 256GB storage. Features advanced night photography and 100x Space Zoom.",
     "brand": "Samsung", "price": 1199.99, "rating": 4.6, "in_stock": True, "tags": ["200MP", "S Pen", "Android", "zoom"]},
    
    {"id": "phone_003", "name": "Google Pixel 7 Pro", "category": "Electronics", "subcategory": "Smartphones",
     "description": "Google's flagship with advanced computational photography, Google Tensor G2 chip, 6.7-inch LTPO OLED display, 12GB RAM, 128GB storage. Features Magic Eraser, Real Tone, and 5 years of security updates.",
     "brand": "Google", "price": 899.99, "rating": 4.5, "in_stock": True, "tags": ["computational photography", "Tensor", "Magic Eraser", "Android"]},
    
    # Fashion - Athletic Wear
    {"id": "shoes_001", "name": "Nike Air Jordan 1 Retro High", "category": "Fashion", "subcategory": "Sneakers",
     "description": "Iconic basketball sneaker with premium leather upper, Air-Sole unit cushioning, rubber outsole with pivot circle. Classic colorway in Chicago Bulls red, black, and white. Timeless style meets modern comfort.",
     "brand": "Nike", "price": 170.00, "rating": 4.8, "in_stock": True, "tags": ["basketball", "retro", "leather", "classic"]},
    
    {"id": "shoes_002", "name": "Adidas Ultraboost 22", "category": "Fashion", "subcategory": "Running Shoes",
     "description": "High-performance running shoes with BOOST midsole, Primeknit+ upper, Continental rubber outsole, and Torsion Spring for energy return. Designed for long-distance running with superior comfort and responsiveness.",
     "brand": "Adidas", "price": 190.00, "rating": 4.6, "in_stock": True, "tags": ["running", "BOOST", "comfort", "long-distance"]},
    
    # Home & Kitchen
    {"id": "coffee_001", "name": "Breville Barista Express Espresso Machine", "category": "Home", "subcategory": "Coffee",
     "description": "Professional-grade espresso machine with built-in conical burr grinder, 15-bar Italian pump, precise temperature control, milk steam wand, and stainless steel construction. Perfect for coffee enthusiasts.",
     "brand": "Breville", "price": 699.95, "rating": 4.4, "in_stock": True, "tags": ["espresso", "grinder", "professional", "barista"]},
    
    {"id": "vacuum_001", "name": "Dyson V15 Detect Cordless Vacuum", "category": "Home", "subcategory": "Cleaning",
     "description": "Advanced cordless vacuum with laser dust detection, intelligent suction adjustment, 60-minute runtime, HEPA filtration, and lightweight design. Transforms to handheld for versatile cleaning.",
     "brand": "Dyson", "price": 749.99, "rating": 4.5, "in_stock": True, "tags": ["cordless", "laser", "HEPA", "lightweight"]},
    
    # Sports & Outdoors
    {"id": "bike_001", "name": "Trek Domane SL 5 Road Bike", "category": "Sports", "subcategory": "Cycling",
     "description": "High-performance carbon road bike with Shimano 105 groupset, endurance geometry, IsoSpeed decoupler for comfort, tubeless-ready wheels, and aerodynamic frame design for serious cyclists.",
     "brand": "Trek", "price": 2999.99, "rating": 4.7, "in_stock": True, "tags": ["carbon", "road", "Shimano", "endurance"]},
    
    {"id": "tent_001", "name": "REI Co-op Half Dome 4 Plus Tent", "category": "Sports", "subcategory": "Camping",
     "description": "Spacious 4-person tent with easy setup, excellent ventilation, rainfly coverage, gear loft, and color-coded clips. Perfect for family camping with ample headroom and vestibule space.",
     "brand": "REI", "price": 349.00, "rating": 4.6, "in_stock": True, "tags": ["4-person", "family", "camping", "spacious"]},
    
    # Beauty & Personal Care
    {"id": "skincare_001", "name": "The Ordinary Niacinamide 10% + Zinc 1%", "category": "Beauty", "subcategory": "Skincare",
     "description": "High-strength vitamin B3 serum for reducing appearance of blemishes, regulating sebum production, and improving skin texture. Water-based formula suitable for all skin types.",
     "brand": "The Ordinary", "price": 7.90, "rating": 4.3, "in_stock": True, "tags": ["vitamin B3", "serum", "blemishes", "affordable"]},
    
    {"id": "perfume_001", "name": "Chanel No. 5 Eau de Parfum", "category": "Beauty", "subcategory": "Fragrance",
     "description": "Iconic feminine fragrance with floral aldehydic composition, featuring ylang-ylang, rose, iris, and sandalwood. Timeless elegance in classic bottle design since 1921.",
     "brand": "Chanel", "price": 165.00, "rating": 4.8, "in_stock": True, "tags": ["iconic", "floral", "luxury", "classic"]},
    
    # Books & Media
    {"id": "book_001", "name": "Atomic Habits by James Clear", "category": "Books", "subcategory": "Self-Help",
     "description": "Practical guide for building good habits and breaking bad ones through proven strategies, scientific research, and real-world examples. Transform your life with tiny changes that deliver remarkable results.",
     "brand": "Avery", "price": 16.99, "rating": 4.8, "in_stock": True, "tags": ["habits", "self-improvement", "bestseller", "practical"]},
    
    # Gaming
    {"id": "console_001", "name": "PlayStation 5", "category": "Gaming", "subcategory": "Consoles",
     "description": "Next-gen gaming console with custom SSD for ultra-fast loading, ray tracing, 4K gaming at 120fps, 3D audio, and haptic feedback controller. Experience gaming like never before.",
     "brand": "Sony", "price": 499.99, "rating": 4.7, "in_stock": False, "tags": ["next-gen", "4K", "ray-tracing", "gaming"]},
]

class EcommerceDemo:
    def __init__(self):
        self.collection_id = "ecommerce_products"
        self.embedding_model = None
        self.grpc_client = None
        self.rest_client = None
        self.api_base = "http://localhost:5678"
        self.grpc_url = "http://localhost:5679"
        
    def initialize_embedding_model(self):
        """Initialize BERT model for product embeddings"""
        print("\n🤖 Initializing all-mpnet-base-v2 BERT model...")
        try:
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
    
    def initialize_clients(self):
        """Initialize ProximaDB clients"""
        print("\n🔌 Connecting to ProximaDB...")
        
        # gRPC client for high performance
        try:
            self.grpc_client = connect_grpc(self.grpc_url)
            print(f"✅ gRPC client connected to {self.grpc_url}")
        except Exception as e:
            print(f"⚠️  gRPC connection failed: {e}")
            
        # REST client as fallback
        try:
            self.rest_client = connect_rest(self.api_base)
            print(f"✅ REST client connected to {self.api_base}")
        except Exception as e:
            print(f"⚠️  REST connection failed: {e}")
            
        if not self.grpc_client and not self.rest_client:
            raise Exception("Failed to connect to ProximaDB")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for product texts"""
        return self.embedding_model.encode(texts, convert_to_numpy=True)
    
    def setup_collection(self):
        """Create collection with optimized metadata configuration"""
        print(f"\n📦 Setting up collection: {self.collection_id}")
        
        # Define filterable fields for columnar optimization
        filterable_fields = [
            # Core search fields
            "text", "chunk_index", "source_type",
            # High cardinality product fields
            "product_id", "name", "category", "subcategory",
            "brand", "price", "rating", "in_stock",
            # Product attributes
            "color", "size", "material",
            # Search flags
            "is_featured", "has_discount", "is_new"
        ]
        
        # Try REST first for collection creation
        if self.rest_client:
            try:
                config = CollectionConfig(
                    name=self.collection_id,
                    dimension=768,  # all-mpnet-base-v2 dimension
                    distance_metric=DistanceMetric.COSINE,
                    storage_engine=StorageEngine.NOVA,  # Columnar storage
                    filterable_metadata_fields=filterable_fields  # Specify indexed fields
                )
                
                result = self.rest_client.create_collection(self.collection_id, config)
                print(f"✅ Collection created via REST with {len(filterable_fields)} filterable fields")
                print(f"   Engine: NOVA (columnar storage)")
                print(f"   Filterable: {', '.join(filterable_fields[:5])}...")
                return
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Collection already exists")
                    return
                else:
                    print(f"⚠️  REST collection creation failed: {e}")
        
        # Fallback to gRPC
        if self.grpc_client:
            try:
                config = CollectionConfig(
                    name=self.collection_id,
                    dimension=768,
                    distance_metric=DistanceMetric.COSINE,
                    storage_engine=StorageEngine.NOVA,
                    filterable_metadata_fields=filterable_fields
                )
                
                result = self.grpc_client.create_collection(self.collection_id, config)
                print(f"✅ Collection created via gRPC")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Collection already exists")
                else:
                    print(f"gRPC collection creation failed: {e}")
                    raise
    
    def prepare_product_vectors(self, products: List[Dict]) -> List[VectorRecord]:
        """Prepare product vectors with optimized metadata separation"""
        try:
            from proximadb.chunking import prepare_vector_records
        except ImportError:
            # Fallback if chunking module not available
            from proximadb import prepare_vector_records
        
        # Generate embeddings
        descriptions = [f"{p['name']} {p['description']}" for p in products]
        embeddings = self.create_embeddings(descriptions)
        
        # Simulate embedding service response format
        chunks = []
        for i, product in enumerate(products):
            chunks.append({
                "id": product["id"],
                "text": f"{product['name']} - {product['description']}",
                "embedding": embeddings[i].tolist()
            })
        
        embedding_response = {
            "chunks": chunks,
            "model": "all-mpnet-base-v2",
            "dimension": 768
        }
        
        # Define custom enrichment for products
        def enrich_product(chunk, index):
            product = products[index]
            return {
                # Derived search flags
                "is_featured": product["rating"] >= 4.7,
                "has_discount": product["price"] < 100,
                "is_new": "2023" in product["description"] or "2024" in product["description"],
                # Price ranges for filtering
                "price_range": (
                    "budget" if product["price"] < 100 else
                    "mid-range" if product["price"] < 500 else
                    "premium"
                )
            }
        
        # Convert to vector records with metadata separation
        return prepare_vector_records(
            embedding_response,
            source_id="product_catalog",
            source_type="product",
            source_metadata={
                # Filterable (high cardinality, search targets)
                "name": product["name"],
                "category": product["category"],
                "subcategory": product["subcategory"],
                "brand": product["brand"],
                "price": product["price"],
                "rating": product["rating"],
                "in_stock": product["in_stock"],
                
                # Non-filterable (low cardinality, repetitive)
                "currency": "USD",
                "store_id": "STORE_001",
                "catalog_version": "2024.01",
                "import_source": "demo_data"
            } if i < len(products) else {},
            chunk_metadata_fn=enrich_product,
            filterable_fields=[
                "name", "category", "subcategory", "brand",
                "price", "rating", "in_stock", "is_featured",
                "has_discount", "is_new", "price_range"
            ]
        )
    
    def insert_products_grpc(self):
        """High-performance product insertion via gRPC"""
        if not self.grpc_client:
            print("⚠️  gRPC not available, skipping bulk insert")
            return
            
        print(f"\n⚡ Inserting {len(REAL_PRODUCTS)} products via gRPC...")
        start_time = time.time()
        
        # Prepare vectors with optimized metadata
        vectors = self.prepare_product_vectors(REAL_PRODUCTS)
        
        try:
            result = self.grpc_client.insert_vectors(self.collection_id, vectors)
            elapsed = time.time() - start_time
            rate = len(vectors) / elapsed
            print(f"✅ Inserted {len(vectors)} products in {elapsed:.2f}s ({rate:.0f} products/sec)")
            print(f"   Metadata fields optimized for NOVA columnar storage")
            
        except Exception as e:
            print(f"❌ gRPC insert failed: {e}")
            print("   Falling back to REST insertion...")
            self.insert_products_rest()
    
    def insert_products_rest(self):
        """Fallback product insertion via REST"""
        if not self.rest_client:
            print("No REST client available")
            return
            
        print(f"\n🌐 Inserting products via REST API...")
        
        # Prepare all vectors
        vectors = self.prepare_product_vectors(REAL_PRODUCTS)
        
        # Batch insert for better performance
        batch_size = 5
        total_inserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                result = self.rest_client.insert_vectors(self.collection_id, batch)
                total_inserted += len(batch)
                print(f"   Batch {i//batch_size + 1}: {len(batch)} products")
                
            except Exception as e:
                print(f"   ❌ REST batch insert failed: {e}")
        
        print(f"✅ Total inserted via REST: {total_inserted} products")
    
    def search_grpc(self, query: str, top_k: int = 5, filters: Dict = None) -> Dict:
        """Semantic search via gRPC with metadata filtering"""
        if not self.grpc_client:
            return {"error": "gRPC not available"}
            
        print(f"\n⚡ gRPC Search: '{query}'")
        if filters:
            print(f"   Filters: {filters}")
        start_time = time.time()
        
        try:
            query_embedding = self.create_embeddings([query])[0]
            search_result = self.grpc_client.search(
                collection_id=self.collection_id,
                vector=query_embedding.tolist(),
                k=top_k,
                filter=filters,
                include_metadata=True
            )
            
            elapsed = time.time() - start_time
            print(f"✅ gRPC search completed in {elapsed*1000:.1f}ms")
            
            # Handle the SearchResult object properly
            if hasattr(search_result, 'results'):
                return {"results": search_result.results, "latency_ms": elapsed*1000, "method": "gRPC"}
            else:
                return {"results": search_result, "latency_ms": elapsed*1000, "method": "gRPC"}
            
        except Exception as e:
            print(f"❌ gRPC search failed: {e}")
            return {"error": str(e)}
    
    def search_rest(self, query: str, top_k: int = 5, filters: Dict = None) -> Dict:
        """Semantic search via REST API"""
        print(f"\n🌐 REST Search: '{query}'")
        if filters:
            print(f"   Filters: {filters}")
        start_time = time.time()
        
        try:
            query_embedding = self.create_embeddings([query])[0]
            
            response = requests.post(
                f"{self.api_base}/api/v1/vector/search",
                json={
                    "collection_id": self.collection_id,
                    "queries": [{
                        "vector": query_embedding.tolist(),
                        "id": None,
                        "metadata_filter": filters
                    }],
                    "top_k": top_k,
                    "distance_metric_override": "cosine",
                    "include_fields": {
                        "vector": False,
                        "metadata": True,
                        "score": True,
                        "rank": True
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()["results"][0]["results"]
                print(f"✅ REST search completed in {elapsed*1000:.1f}ms")
                return {"results": results, "latency_ms": elapsed*1000, "method": "REST"}
            else:
                print(f"❌ REST search failed: {response.status_code} - {response.text}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ REST search error: {e}")
            return {"error": str(e)}
    
    def search_sql(self, query: str, filters: Dict = None) -> Dict:
        """SQL-based semantic search with rich filtering"""
        print(f"\n🔍 SQL Search: '{query}'")
        start_time = time.time()
        
        query_embedding = self.create_embeddings([query])[0]
        vector_str = json.dumps(query_embedding.tolist())
        
        # Build SQL with optional filters
        sql = f"""
        SELECT id, name, brand, category, price, rating, in_stock
        FROM {self.collection_id}
        """
        
        conditions = []
        if filters:
            for key, value in filters.items():
                if key == "price":
                    if isinstance(value, dict):
                        if "$gte" in value:
                            conditions.append(f"price >= {value['$gte']}")
                        if "$lte" in value:
                            conditions.append(f"price <= {value['$lte']}")
                    else:
                        conditions.append(f"price = {value}")
                elif key == "in_stock":
                    conditions.append(f"in_stock = {str(value).lower()}")
                else:
                    conditions.append(f"{key} = '{value}'")
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += f"""
        ORDER BY VECTOR_SIMILARITY(vector, {vector_str}, 'cosine')
        LIMIT 5
        """
        
        try:
            response = requests.post(
                f"{self.api_base}/api/v1/sql",
                json={"query": sql},
                headers={"Content-Type": "application/json"}
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()["rows"]
                print(f"✅ SQL search completed in {elapsed*1000:.1f}ms")
                return {"results": results, "latency_ms": elapsed*1000, "method": "SQL", "query": sql}
            else:
                print(f"❌ SQL search failed: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            print(f"❌ SQL search error: {e}")
            return {"error": str(e)}
    
    def format_results(self, search_result: Dict):
        """Format and display search results"""
        if "error" in search_result:
            print(f"\n❌ Search failed: {search_result['error']}")
            return
            
        results = search_result.get("results", [])
        method = search_result.get("method", "Unknown")
        latency = search_result.get("latency_ms", 0)
        
        print(f"\n📊 Results ({method} - {latency:.1f}ms):")
        print("=" * 80)
        
        if not results:
            print("No results found")
            return
            
        for i, result in enumerate(results[:5], 1):
            # Handle different result formats
            if isinstance(result, dict):
                if method == "SQL":
                    # SQL returns flat structure
                    product_id = result.get("id", "")
                    name = result.get("name", "")
                    brand = result.get("brand", "")
                    price = result.get("price", 0)
                    rating = result.get("rating", 0)
                    score = result.get("_score", 0)
                else:
                    # REST/gRPC return nested structure
                    product_id = result.get("id", "")
                    metadata = result.get("metadata", {})
                    if isinstance(metadata, list):
                        # Convert list format to dict
                        metadata_dict = {}
                        for item in metadata:
                            metadata_dict[item["key"]] = item["value"]
                        metadata = metadata_dict
                    
                    name = metadata.get("name", "")
                    brand = metadata.get("brand", "")
                    price = float(metadata.get("price", 0))
                    rating = float(metadata.get("rating", 0))
                    score = result.get("score", 0)
            else:
                # Handle object format (gRPC)
                product_id = getattr(result, "id", "")
                metadata = getattr(result, "metadata", {})
                name = metadata.get("name", "")
                brand = metadata.get("brand", "")
                price = float(metadata.get("price", 0))
                rating = float(metadata.get("rating", 0))
                score = getattr(result, "score", 0)
            
            print(f"\n{i}. {name}")
            print(f"   Brand: {brand} | Price: ${price:.2f} | Rating: ⭐ {rating}")
            print(f"   ID: {product_id} | Score: {score:.3f}")
    
    def demonstrate_search_methods(self):
        """Compare different search methods"""
        print("\n" + "="*80)
        print("🔍 SEMANTIC PRODUCT SEARCH COMPARISON")
        print("="*80)
        
        # Test queries
        queries = [
            "lightweight laptop for business travel",
            "gaming laptop with RTX graphics",
            "running shoes for long distance",
            "coffee machine with grinder"
        ]
        
        for query in queries:
            print(f"\n\n{'='*80}")
            print(f"QUERY: '{query}'")
            print(f"{'='*80}")
            
            # 1. gRPC Search (fastest)
            grpc_result = self.search_grpc(query)
            self.format_results(grpc_result)
            
            # 2. REST Search (compatible)
            rest_result = self.search_rest(query)
            self.format_results(rest_result)
            
            # 3. SQL Search (flexible)
            sql_result = self.search_sql(query)
            self.format_results(sql_result)
    
    def demonstrate_filtered_search(self):
        """Demonstrate metadata filtering capabilities"""
        print("\n" + "="*80)
        print("🎯 FILTERED SEARCH DEMONSTRATION")
        print("="*80)
        
        # Example: Search for laptops under $1500
        print("\n📱 Query: 'laptop' with price < $1500")
        filters = {
            "category": "Electronics",
            "price": {"$lte": 1500},
            "in_stock": True
        }
        
        result = self.search_grpc("laptop for work", filters=filters)
        self.format_results(result)
        
        # Example: High-rated products only
        print("\n⭐ Query: 'premium quality' with rating >= 4.6")
        filters = {
            "rating": {"$gte": 4.6}
        }
        
        result = self.search_grpc("premium quality", filters=filters)
        self.format_results(result)
    
    def run_demo(self):
        """Run the complete e-commerce demo"""
        print("\n🛒 ProximaDB E-commerce Demo")
        print("="*50)
        
        # Initialize
        self.initialize_embedding_model()
        self.initialize_clients()
        
        # Setup collection
        self.setup_collection()
        
        # Insert products
        if self.grpc_client:
            self.insert_products_grpc()
        else:
            self.insert_products_rest()
        
        # Wait for indexing
        print("\n⏳ Waiting for indexing...")
        time.sleep(2)
        
        # Demonstrate search capabilities
        self.demonstrate_search_methods()
        self.demonstrate_filtered_search()
        
        print("\n\n✅ E-commerce demo completed!")
        print("\n📊 Key Insights:")
        print("1. gRPC provides 2-3x faster performance for bulk operations")
        print("2. NOVA engine with filterable metadata enables efficient queries")
        print("3. SQL interface allows complex filtering with natural syntax")
        print("4. Metadata separation optimizes storage and query performance")
        
        print("\n🎯 Next Steps:")
        print("1. Run the web UI: python serve_ui.py")
        print("2. Try the SQL console for ad-hoc queries")
        print("3. Experiment with different embedding models")
        print("4. Monitor performance with larger datasets")

if __name__ == "__main__":
    demo = EcommerceDemo()
    demo.run_demo()