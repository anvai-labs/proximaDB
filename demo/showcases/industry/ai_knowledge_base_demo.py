#!/usr/bin/env python3
"""
STATUS: 🚧 Requires Demo Server - Embedding Service Not Available
SDK Version: v1.0
Server Version: v0.2.0+
Test Result: SKIP - Requires demo server with embedding/LLM services

AI Knowledge Base Demo for ProximaDB

An intelligent knowledge base system that combines ProximaDB's vector search with
LLM-powered responses. This demo showcases how to build production-ready AI applications
with semantic search, document understanding, and natural language Q&A.

Features:
- Document ingestion with intelligent chunking
- Semantic search using BERT embeddings
- LLM-powered answer generation (Flan-T5)
- High-performance gRPC protocol
- Interactive Q&A interface

Requirements:
- ProximaDB server running (localhost:5678)
- Demo server with embedding/LLM services (localhost:8080)

NOTE: This demo requires a separate demo server that provides:
  - /api/embeddings/chunk endpoint
  - /api/embeddings/embed endpoint
  - /api/embeddings/info endpoint
  - LLM service for answer generation

For basic RAG functionality without the demo server, see:
  demo/quickstart/basic_demo.py (shows vector insertion/search)

Usage:
    # Start demo server first (if available)
    docker compose up -d

    # Then run demo
    python ai_knowledge_base_demo.py
"""

import requests
import json
import time
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add demo root to path for utils import
demo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, demo_root)

# Try to import LLM service, but make it optional with error handling
try:
    from utils.llm_service import get_llm_service, LLMConfig
    HAS_LLM_SERVICE = True
except (ImportError, ModuleNotFoundError):
    HAS_LLM_SERVICE = False
    print("=" * 70)
    print("⚠️  WARNING: LLM Service Not Available")
    print("=" * 70)
    print("\n📋 This demo requires utils/llm_service module")
    print("   Expected location: demo/utils/llm_service.py")
    print("\n💡 This demo requires a separate embedding/LLM server")
    print("   For basic vector operations, see: demo/quickstart/basic_demo.py\n")
    print("=" * 70)

# Import ProximaDB SDK for proper API usage
from proximadb import ProximaDBClient, Protocol, CollectionConfig, DistanceMetric, StorageEngine, VectorRecord

# Configuration
PROXIMADB_URL = "http://localhost:5678"
PROXIMADB_GRPC_URL = "http://localhost:5679"
DEMO_SERVER_URL = "http://localhost:8080"
COLLECTION_NAME = "ai_knowledge_base"

# Sample documents for RAG system
SAMPLE_DOCUMENTS = {
    "machine_learning.txt": """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:

1. Supervised Learning: The algorithm learns from labeled training data, helping predict outcomes for unforeseen data. Common algorithms include Linear Regression, Decision Trees, Random Forests, and Support Vector Machines.

2. Unsupervised Learning: The algorithm finds hidden patterns in unlabeled data. It includes clustering (K-means, DBSCAN) and dimensionality reduction (PCA, t-SNE).

3. Reinforcement Learning: The algorithm learns to make decisions by taking actions in an environment to maximize cumulative reward. Applications include game playing, robotics, and autonomous vehicles.

Key Concepts:
- Feature Engineering: The process of selecting and transforming variables for your model
- Cross-Validation: A technique to assess model performance on unseen data
- Overfitting: When a model performs well on training data but poorly on new data
- Hyperparameter Tuning: Optimizing the configuration settings of algorithms

Machine learning powers many modern applications including recommendation systems, fraud detection, natural language processing, and computer vision.
""",
    
    "vector_databases.txt": """
Vector Databases: The Foundation of AI Applications

Vector databases are specialized systems designed to store, index, and query high-dimensional vector embeddings. They have become essential infrastructure for modern AI applications, particularly those involving semantic search, recommendation systems, and retrieval-augmented generation (RAG).

Why Vector Databases Matter:

Traditional databases excel at exact matches and structured queries but struggle with semantic similarity. Vector databases solve this by:
- Converting data into mathematical representations (embeddings)
- Enabling similarity search based on meaning rather than keywords
- Supporting real-time updates and queries at scale

Key Features:
1. Distance Metrics: Support for various similarity measures (cosine, euclidean, dot product)
2. Indexing Algorithms: HNSW, IVF, LSH for efficient nearest neighbor search
3. Scalability: Handling billions of vectors with sub-second query times
4. Integration: APIs for popular ML frameworks and embedding models

Use Cases:
- Semantic Search: Finding relevant documents based on meaning
- Recommendation Systems: Suggesting similar items based on user preferences
- RAG Systems: Enhancing LLMs with external knowledge
- Image/Video Search: Finding visually similar content
- Anomaly Detection: Identifying outliers in high-dimensional data

Popular vector databases include ProximaDB, Pinecone, Weaviate, Qdrant, and Milvus. Each offers unique features for different use cases and scale requirements.
""",
    
    "rag_architecture.txt": """
Retrieval-Augmented Generation (RAG) Architecture

RAG combines the power of large language models with external knowledge retrieval to provide accurate, up-to-date, and verifiable responses. This architecture has become the standard for building AI systems that need access to specific domain knowledge.

Core Components:

1. Document Processing Pipeline:
   - Ingestion: Loading documents from various sources
   - Chunking: Breaking documents into optimal-sized segments
   - Embedding: Converting chunks into vector representations
   - Storage: Storing vectors and metadata in a vector database

2. Retrieval System:
   - Query Processing: Converting user questions to embeddings
   - Similarity Search: Finding relevant document chunks
   - Ranking: Ordering results by relevance
   - Context Assembly: Combining retrieved chunks

3. Generation Layer:
   - Prompt Engineering: Crafting effective prompts with context
   - LLM Integration: Using models like GPT, Claude, or Llama
   - Response Generation: Creating answers based on retrieved context
   - Citation Management: Linking responses to source documents

Best Practices:
- Chunk Size: Balance between context and relevance (typically 200-1000 tokens)
- Overlap: Include overlap between chunks to preserve context
- Metadata: Store source, timestamp, and other relevant information
- Hybrid Search: Combine vector similarity with keyword matching
- Evaluation: Measure retrieval quality and generation accuracy

Common Challenges and Solutions:
- Hallucination: Mitigated by grounding responses in retrieved documents
- Context Limits: Solved by intelligent chunk selection and summarization
- Update Frequency: Addressed with incremental indexing strategies
- Query Understanding: Enhanced with query expansion and reformulation
""",
    
    "embeddings_guide.txt": """
Understanding Embeddings: The Bridge Between Text and Vectors

Embeddings are dense vector representations of data that capture semantic meaning in a mathematical format. They serve as the foundation for modern NLP applications and vector search systems.

What Are Embeddings?

Embeddings transform discrete data (words, sentences, documents) into continuous vector spaces where:
- Similar items are close together
- Dissimilar items are far apart
- Mathematical operations preserve semantic relationships

Popular Embedding Models:

1. BERT-based Models:
   - all-MiniLM-L6-v2: Fast, 384 dimensions, good for general use
   - all-mpnet-base-v2: Higher quality, 768 dimensions
   - Specialized models for specific domains (legal, medical, scientific)

2. OpenAI Embeddings:
   - text-embedding-3-small: Cost-effective, 1536 dimensions
   - text-embedding-3-large: High quality, 3072 dimensions

3. Open Source Alternatives:
   - Sentence-BERT variants
   - Instructor embeddings
   - E5 models from Microsoft

Key Considerations:

Dimensionality: Higher dimensions capture more information but require more storage and computation. Common dimensions range from 384 to 4096.

Distance Metrics:
- Cosine Similarity: Best for normalized embeddings, measures angle
- Euclidean Distance: Measures straight-line distance, sensitive to magnitude
- Dot Product: Fast computation, works well with certain models

Quality Factors:
- Training Data: Models trained on diverse, high-quality data perform better
- Fine-tuning: Domain-specific fine-tuning improves performance
- Context Window: Longer context allows better understanding

Practical Tips:
- Normalize embeddings for consistent similarity scores
- Batch processing for efficiency
- Cache embeddings to avoid recomputation
- Monitor embedding drift over time
- Test different models for your specific use case
"""
}

class AIKnowledgeBase:
    def __init__(self):
        self.proximadb_url = PROXIMADB_URL
        self.demo_server_url = DEMO_SERVER_URL
        self.collection_name = COLLECTION_NAME

        # Initialize ProximaDB client with gRPC for better performance
        self.client = ProximaDBClient(
            url=PROXIMADB_URL,
            grpc_url=PROXIMADB_GRPC_URL,
            protocol=Protocol.GRPC  # Use gRPC for faster performance
        )

        # Initialize LLM service (optional, local model for reliability)
        self.llm_service = None
        if HAS_LLM_SERVICE:
            try:
                llm_config = LLMConfig(
                    model_type="flan-t5-small",  # Use local model
                    api_provider="huggingface",
                    max_length=200,
                    temperature=0.7
                )
                self.llm_service = get_llm_service(llm_config)
            except Exception as e:
                print(f"⚠️  LLM service initialization failed: {e}")
                print("   Continuing without LLM - will use retrieved context only")
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}\n")
    
    def check_services(self) -> bool:
        """Check if all required services are running"""
        print("🔍 Checking services...")
        
        # Check ProximaDB
        try:
            response = requests.get(f"{self.proximadb_url}/health")
            if response.status_code == 200:
                print("✅ ProximaDB is running")
            else:
                print("❌ ProximaDB is not healthy")
                return False
        except:
            print("❌ Cannot connect to ProximaDB at", self.proximadb_url)
            return False
        
        # Check Demo Server
        try:
            response = requests.get(f"{self.demo_server_url}/api/embeddings/info")
            if response.status_code == 200:
                print("✅ Embedding service is running")
                return True
            else:
                print("❌ Embedding service is not healthy")
                return False
        except:
            print("❌ Cannot connect to demo server at", self.demo_server_url)
            return False
    
    def create_collection(self):
        """Create the RAG collection with optimized metadata configuration"""
        self.print_section("Creating RAG Collection")
        
        # Delete if exists
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Define filterable fields for knowledge base optimization
        filterable_fields = [
            # Core search fields
            "text", "chunk_index", "source_type",
            # High cardinality document fields
            "filename", "category", "topic", "subtopic",
            "document_type", "section", "difficulty",
            # Content attributes
            "has_code", "has_examples", "has_diagrams",
            "content_type", "technical_level",
            # Search flags
            "is_definition", "is_tutorial", "is_reference"
        ]
        
        # Create new collection using SDK
        try:
            config = CollectionConfig(
                name=self.collection_name,
                dimension=768,  # all-mpnet-base-v2 dimension
                distance_metric=DistanceMetric.COSINE,
                storage_engine=StorageEngine.NOVA,  # Optimized for metadata filtering
                filterable_metadata_fields=filterable_fields  # Specify indexed fields
            )
            
            collection = self.client.create_collection(
                name=self.collection_name,
                config=config
            )
            
            print(f"✅ Created collection: {self.collection_name}")
            print(f"   - Dimension: 768 (BERT all-mpnet-base-v2)")
            print(f"   - Engine: NOVA (columnar storage)")
            print(f"   - Distance: Cosine similarity")
            print(f"   - Protocol: gRPC (high performance)")
            print(f"   - Filterable fields: {len(filterable_fields)} configured")
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            sys.exit(1)
    
    def ingest_documents(self):
        """Ingest sample documents using the unified endpoint with optimized metadata"""
        self.print_section("Ingesting Knowledge Base Documents")
        
        total_chunks = 0
        
        # Define custom metadata enrichment for knowledge base
        def enrich_knowledge_chunk(chunk, index):
            text = chunk["text"].lower()
            
            # Detect content characteristics
            return {
                # Content attributes
                "has_code": "```" in chunk["text"] or "import" in text or "class" in text,
                "has_examples": "example" in text or "for instance" in text,
                "has_diagrams": "diagram" in text or "figure" in text or "illustration" in text,
                
                # Content type detection
                "content_type": (
                    "code" if "```" in chunk["text"] else
                    "definition" if text.startswith(("what is", "what are", "definition")) else
                    "explanation" if "because" in text or "therefore" in text else
                    "general"
                ),
                
                # Technical level
                "technical_level": (
                    "advanced" if any(term in text for term in ["algorithm", "optimization", "hyperparameter"]) else
                    "intermediate" if any(term in text for term in ["function", "model", "training"]) else
                    "beginner"
                ),
                
                # Flags for search
                "is_definition": text.startswith(("what is", "what are", "definition")),
                "is_tutorial": "step" in text or "how to" in text,
                "is_reference": "key concepts" in text or "overview" in text,
                
                # Section identification
                "section": (
                    "introduction" if index == 0 else
                    "conclusion" if "summary" in text or "conclusion" in text else
                    "body"
                )
            }
        
        # Topic mapping for documents
        doc_topics = {
            "machine_learning.txt": ("Machine Learning", "Fundamentals"),
            "vector_databases.txt": ("Vector Databases", "Infrastructure"),
            "rag_architecture.txt": ("RAG Systems", "Architecture"),
            "embeddings_guide.txt": ("Embeddings", "Technical Guide")
        }
        
        for filename, content in SAMPLE_DOCUMENTS.items():
            print(f"\n📄 Processing: {filename}")
            print(f"   Document size: {len(content)} characters")
            
            # Extract topic info
            topic, subtopic = doc_topics.get(filename, ("General", "Knowledge"))
            
            # Determine document type and difficulty
            doc_type = (
                "guide" if "guide" in filename else
                "architecture" if "architecture" in filename else
                "fundamentals" if "fundamentals" in content[:100].lower() else
                "tutorial"
            )
            
            difficulty = (
                "advanced" if "architecture" in filename else
                "intermediate" if "embeddings" in filename else
                "beginner"
            )
            
            # First chunk the document
            chunk_response = requests.post(
                f"{self.demo_server_url}/api/embeddings/chunk",
                json={
                    "text": content,
                    "strategy": "semantic",
                    "chunk_size": 400,
                    "overlap": 100,
                    "model": "all-mpnet-base-v2"
                }
            )
            
            if chunk_response.status_code != 200:
                print(f"❌ Failed to chunk document: {chunk_response.text}")
                continue
            
            chunks_data = chunk_response.json()
            chunks = chunks_data.get("chunks", [])
            
            # Use SDK helper to convert to VectorRecords with metadata separation
            from proximadb.chunking import prepare_vector_records
            
            vectors_to_insert = prepare_vector_records(
                chunks_data,
                source_id=filename.replace(".txt", ""),
                source_type="knowledge_base",
                source_metadata={
                    # Filterable (high cardinality, search targets)
                    "filename": filename,
                    "category": "AI/ML",
                    "topic": topic,
                    "subtopic": subtopic,
                    "document_type": doc_type,
                    "difficulty": difficulty,
                    
                    # Non-filterable (low cardinality, metadata)
                    "language": "en",
                    "version": "1.0",
                    "collection": self.collection_name,
                    "demo": "ai_knowledge_base"
                },
                chunk_metadata_fn=enrich_knowledge_chunk,
                filterable_fields=[
                    "filename", "category", "topic", "subtopic",
                    "document_type", "difficulty", "has_code",
                    "has_examples", "has_diagrams", "content_type",
                    "technical_level", "is_definition", "is_tutorial",
                    "is_reference", "section"
                ]
            )
            
            # Use SDK to insert vectors
            try:
                insert_response = self.client.insert_vectors(
                    collection_id=self.collection_name,
                    records=vectors_to_insert
                )
                
                print(f"   ✅ Successfully ingested")
                print(f"   📊 Chunks created: {len(chunks)}")
                print(f"   🔤 Chunking strategy: semantic")
                print(f"   🏷️  Topic: {topic} > {subtopic}")
                print(f"   📚 Type: {doc_type} ({difficulty})")
                total_chunks += len(chunks)
            except Exception as e:
                print(f"   ❌ Failed to ingest: {e}")
        
        print(f"\n📊 Total chunks in knowledge base: {total_chunks}")
        print(f"   💾 Metadata optimized for NOVA columnar storage")
        return total_chunks
    
    def perform_rag_query(self, query: str, top_k: int = 3):
        """Perform a RAG-style query"""
        print(f"\n🔍 Query: '{query}'")
        print("─" * 50)
        
        # Step 1: Convert query to embedding
        embed_response = requests.post(
            f"{self.demo_server_url}/api/embeddings/embed",
            json={"text": query, "model": "all-mpnet-base-v2"}
        )
        
        if embed_response.status_code != 200:
            print("❌ Failed to generate query embedding")
            return
        
        query_vector = embed_response.json()["embedding"]
        
        # Step 2: Search for relevant chunks using SDK
        try:
            results = self.client.search(
                collection_id=self.collection_name,
                vector=query_vector,
                k=top_k,
                include_metadata=True,
                include_vector=True  # Include vectors for similarity verification
            )
            
            # SDK returns results directly as a list
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return
        
        if not results:
            print("❌ No relevant documents found")
            return
        
        # Step 3: Display retrieved context
        print(f"\n📚 Retrieved {len(results)} relevant chunks:")
        print("─" * 50)
        
        context_parts = []
        for i, result in enumerate(results):
            # SDK returns SearchResult objects with attributes
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            score = result.score if hasattr(result, 'score') else 0
            text = metadata.get("text", "") if metadata else ""
            source = metadata.get("source", "unknown") if metadata else "unknown"
            
            print(f"\n🔖 Chunk {i+1} (Score: {score:.3f}, Source: {source})")
            print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
            
            context_parts.append(text)
        
        # Step 4: Generate RAG response using LLM
        print("\n💡 RAG Response Generation:")
        print("─" * 50)
        
        # Prepare context for LLM
        context_data = []
        for i, result in enumerate(results):
            # SDK returns SearchResult objects
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            text = metadata.get("text", "") if metadata else ""
            source = metadata.get("source", "unknown") if metadata else "unknown"
            score = result.score if hasattr(result, 'score') else 0
            
            context_data.append({
                "content": text,
                "source": source,
                "score": score
            })
        
        # Generate response using LLM service (if available)
        if self.llm_service is not None:
            print("🤖 Generating response using LLM...")
            try:
                response = self.llm_service.generate_response(
                    query=query,
                    context=context_data
                )

                # Display the generated response
                print(f"\n📝 Answer:")
                print("─" * 50)
                print(response)

                # Show model info
                model_info = self.llm_service.get_model_info()
                print(f"\n🔧 Generated using: {model_info['model_name']} ({model_info['provider']})")

            except Exception as e:
                print(f"⚠️  LLM generation failed: {e}")
                print("📝 Fallback response based on retrieved context:")
                print("─" * 50)
                print(f"Based on the search results, here are relevant excerpts about '{query}':")
                for i, text in enumerate(context_parts[:2]):
                    print(f"\n{i+1}. {text[:200]}...")
                print("\n[Note: Install transformers for local LLM or ensure internet connectivity for API access]")
        else:
            # No LLM service - show retrieved context
            print("📝 Response based on retrieved context:")
            print("─" * 50)
            print(f"Based on the search results, here are relevant excerpts about '{query}':")
            for i, text in enumerate(context_parts[:2]):
                print(f"\n{i+1}. {text[:200]}...")
            print("\n[Note: LLM service not available - showing raw context]")
    
    def demonstrate_filtered_search(self):
        """Demonstrate metadata filtering capabilities"""
        self.print_section("Filtered Search Examples")
        
        # Example 1: Search for beginner content only
        print("\n📚 Example 1: Beginner-friendly content about machine learning")
        filters = {
            "difficulty": "beginner",
            "category": "AI/ML"
        }
        self.perform_filtered_search("What is machine learning?", filters)
        
        # Example 2: Search for content with code examples
        print("\n\n💻 Example 2: Technical content with code examples")
        filters = {
            "has_code": True,
            "technical_level": {"$in": ["intermediate", "advanced"]}
        }
        self.perform_filtered_search("How to implement embeddings?", filters)
        
        # Example 3: Search for architecture guides
        print("\n\n🏗️ Example 3: Architecture documentation")
        filters = {
            "document_type": "architecture",
            "topic": "RAG Systems"
        }
        self.perform_filtered_search("RAG components", filters)
    
    def perform_filtered_search(self, query: str, filters: Dict):
        """Perform a search with metadata filters"""
        print(f"🔍 Query: '{query}'")
        print(f"🎯 Filters: {filters}")
        print("─" * 50)
        
        # Convert query to embedding
        embed_response = requests.post(
            f"{self.demo_server_url}/api/embeddings/embed",
            json={"text": query, "model": "all-mpnet-base-v2"}
        )
        
        if embed_response.status_code != 200:
            print("❌ Failed to generate query embedding")
            return
        
        query_vector = embed_response.json()["embedding"]
        
        # Search with filters
        try:
            results = self.client.search(
                collection_id=self.collection_name,
                vector=query_vector,
                k=3,
                filter=filters,
                include_metadata=True
            )
            
            if not results:
                print("❌ No results matching filters")
                return
            
            print(f"\n📚 Found {len(results)} filtered results:")
            for i, result in enumerate(results):
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                score = result.score if hasattr(result, 'score') else 0
                
                print(f"\n🔖 Result {i+1} (Score: {score:.3f})")
                print(f"   Topic: {metadata.get('topic', 'N/A')} > {metadata.get('subtopic', 'N/A')}")
                print(f"   Type: {metadata.get('document_type', 'N/A')} | Level: {metadata.get('technical_level', 'N/A')}")
                print(f"   Text: {metadata.get('text', '')[:150]}...")
                
        except Exception as e:
            print(f"❌ Filtered search failed: {e}")
    
    def interactive_mode(self):
        """Run interactive query mode"""
        self.print_section("Interactive RAG Query Mode")
        
        print("Enter your questions about machine learning, vector databases, or RAG.")
        print("Type 'quit' to exit.\n")
        
        while True:
            query = input("🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            self.perform_rag_query(query)
    
    def run_demo(self):
        """Run the complete AI Knowledge Base demo"""
        print("🤖 ProximaDB AI Knowledge Base Demo")
        print("=" * 60)
        print("\nBuilding an intelligent knowledge base with:")
        print("✨ Smart document processing & semantic chunking")
        print("🔍 Lightning-fast vector search (gRPC)")
        print("🧠 AI-powered answers using Flan-T5 LLM")
        print("📚 Real-time document understanding")
        print("💬 Natural language Q&A interface")
        
        # Check services
        if not self.check_services():
            print("\n❌ Please ensure ProximaDB and the demo server are running:")
            print("   docker compose up -d")
            sys.exit(1)
        
        # Create collection
        self.create_collection()
        
        # Ingest documents
        chunk_count = self.ingest_documents()
        
        # Wait for indexing
        print("\n⏳ Waiting for indexing...")
        time.sleep(2)
        
        # Run example queries
        self.print_section("Example RAG Queries")
        
        example_queries = [
            "What are the different types of machine learning?",
            "How do vector databases enable semantic search?",
            "What are the key components of a RAG architecture?",
            "What factors affect embedding quality?"
        ]
        
        for query in example_queries:
            self.perform_rag_query(query, top_k=2)
            print("\n" + "="*60)
        
        # Demonstrate filtered search
        self.demonstrate_filtered_search()
        
        # Interactive mode
        try:
            self.interactive_mode()
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
        
        print("\n🎉 RAG demo completed!")
        print("\n📝 Key Features Demonstrated:")
        print("✅ Smart metadata separation for NOVA optimization")
        print("✅ Rich content analysis (code detection, difficulty levels)")
        print("✅ Filtered search by topic, difficulty, and content type")
        print("✅ LLM-powered answer generation with context")
        print("✅ High-performance gRPC protocol")
        
        print("\n🚀 Next steps:")
        print("1. Add more documents to expand the knowledge base")
        print("2. Experiment with different metadata filters")
        print("3. Fine-tune chunk sizes for your use case")
        print("4. Implement hybrid search (vector + keyword)")
        print("5. Add user feedback to improve relevance")

if __name__ == "__main__":
    demo = AIKnowledgeBase()
    demo.run_demo()