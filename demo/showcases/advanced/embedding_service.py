#!/usr/bin/env python3
"""
Real-World BERT Embedding Service for ProximaDB Web UI

This service provides:
- Real BERT embeddings using sentence-transformers
- Text chunking with ProximaDB SDK strategies
- Consistent seed for reproducible embeddings
- Both server-side and client-side embedding support
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import asyncio
import logging

# Add demo root to path for utils import
demo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, demo_root)

# Try to import path utilities (optional)
try:
    from utils.path_utils import setup_demo_environment, get_embedding_cache_dir
    HAS_PATH_UTILS = True
except (ImportError, ModuleNotFoundError):
    HAS_PATH_UTILS = False
    print("⚠️  Path utilities not available - using defaults")
    # Provide fallback implementations
    def setup_demo_environment():
        return {"sdk_path": None, "cache_dir": "/tmp/proximadb_cache"}
    def get_embedding_cache_dir():
        return "/tmp/proximadb_cache"

# Setup environment (adds SDK to path and sets up caches)
env_info = setup_demo_environment()

from sentence_transformers import SentenceTransformer
import torch

# Configure logging with file output and debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/embedding_service_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import ProximaDB SDK chunking (required)
from proximadb.chunking import (
    TextChunker, ChunkingConfig, ChunkingStrategy,
    chunk_by_sentences, chunk_sliding_window
)

# Import ProximaDB client for direct ingestion
try:
    from proximadb import ProximaDBClient, Protocol
    from proximadb import ClientConfig, CompressionConfig
    PROXIMADB_CLIENT_AVAILABLE = True
except ImportError:
    PROXIMADB_CLIENT_AVAILABLE = False
    logger.warning("ProximaDB client not available for direct ingestion")

class ProximaDBEmbeddingService:
    """
    Comprehensive embedding service with real BERT embeddings and text chunking
    """
    
    # Consistent model configurations with reproducible seeds
    MODELS = {
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Fast, lightweight, good quality",
            "use_case": "general_purpose"
        },
        "all-mpnet-base-v2": {
            "dimension": 768, 
            "description": "Best quality, slower",
            "use_case": "high_accuracy"
        },
        "all-MiniLM-L12-v2": {
            "dimension": 384,
            "description": "Balanced speed/quality",
            "use_case": "balanced"
        }
    }
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        cache_dir: Optional[str] = None,
        seed: int = 42,
        device: Optional[str] = None,
        proximadb_client=None
    ):
        """
        Initialize the embedding service
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache embeddings  
            seed: Random seed for reproducible embeddings
            device: Device to run model on ('cpu', 'cuda', 'auto')
            proximadb_client: ProximaDB client for direct ingestion (optional)
        """
        self.model_name = model_name
        self.seed = seed
        # Use absolute path for cache directory in Docker environment
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Check for Docker environment
            if os.path.exists('/app/demo/embedding_cache'):
                self.cache_dir = Path('/app/demo/embedding_cache')
            elif os.path.exists('demo/embedding_cache'):
                self.cache_dir = Path('demo/embedding_cache')
            else:
                self.cache_dir = Path('./embedding_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize model (required - no fallbacks)
        logger.info(f"🤖 Loading BERT model: {model_name}")
        try:
            # Use provided cache_dir or get from environment
            if not cache_dir:
                cache_dir = str(get_embedding_cache_dir())
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded with {self.dimension} dimensions on {self.model.device}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load BERT model {model_name}: {e}")
            raise RuntimeError(f"BERT model {model_name} is required but could not be loaded. Ensure sentence-transformers is installed.")
        
        # Initialize chunker (required - ProximaDB SDK)
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=512,
            chunk_overlap=128,
            min_chunk_size=50,
            preserve_sentences=True
        )
        self.chunker = TextChunker(config)
        logger.info("✅ ProximaDB chunker initialized")
        
        # ProximaDB client for direct ingestion
        self.proximadb_client = proximadb_client
        if proximadb_client:
            logger.info("✅ ProximaDB client configured for direct ingestion")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "seed": self.seed,
            "available": True,  # Always true - no fallbacks
            "device": str(self.model.device),
            "description": self.MODELS.get(self.model_name, {}).get("description", "Unknown model")
        }
    
    def _get_cache_key(self, text: str, include_chunks: bool = False) -> str:
        """Generate cache key for text"""
        content = f"{self.model_name}_{self.seed}_{text}_{include_chunks}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        
        text = text.strip()
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
        
        # Generate real BERT embedding
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text, 
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error generating BERT embedding: {e}")
            raise RuntimeError(f"Failed to generate BERT embedding for text: {e}")
        
        # Ensure consistent dtype
        embedding = embedding.astype(np.float32)
        
        # Cache the result
        if use_cache:
            try:
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.npy"
                np.save(cache_path, embedding)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Use batch processing for efficiency
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error in batch BERT embedding: {e}")
            raise RuntimeError(f"Failed to generate batch BERT embeddings: {e}")
        
        return [emb.astype(np.float32) for emb in embeddings]
    
    def chunk_and_embed(
        self,
        text: str,
        strategy: str = "sliding_window",
        chunk_size: int = 512,
        overlap: int = 128,
        document_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and generate embeddings for each chunk
        
        Args:
            text: Input text to chunk and embed
            strategy: Chunking strategy ("sliding_window", "sentence", "paragraph", "semantic")
            chunk_size: Size of each chunk
            overlap: Overlap between chunks (for sliding window)
            document_id: ID for the source document
            metadata: Additional metadata
            
        Returns:
            List of dicts with chunk text, embedding, and metadata
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Use ProximaDB SDK chunking (required)
        try:
            logger.debug(f"Chunking text with strategy={strategy}, chunk_size={chunk_size}, overlap={overlap}")
            logger.debug(f"Text length: {len(text)}, document_id={document_id}")
            
            # Update chunker config
            self.chunker.config.strategy = ChunkingStrategy(strategy.lower())
            self.chunker.config.chunk_size = chunk_size
            self.chunker.config.chunk_overlap = overlap
            
            logger.debug(f"Chunker config updated: {self.chunker.config}")
            chunks = self.chunker.chunk_text(text, document_id, metadata)
            logger.debug(f"Generated {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i}: text_length={len(chunk.text) if hasattr(chunk, 'text') else 'N/A'}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error with ProximaDB chunking: {e}", exc_info=True)
            raise RuntimeError(f"ProximaDB chunking failed: {e}")
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(chunk_texts)
        
        # Combine chunks with embeddings (ProximaDB TextChunk objects only)
        result = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": embedding.tolist(),
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.dimension
                }
            }
            result.append(chunk_data)
        
        return result
    
    def search_similar_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to query using cosine similarity
        
        Args:
            query: Search query text
            chunks: List of chunk dicts with embeddings
            top_k: Number of top results to return
            
        Returns:
            List of most similar chunks with similarity scores
        """
        if not chunks:
            return []
        
        # Get query embedding
        query_embedding = self.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"], dtype=np.float32)
            similarity = np.dot(query_embedding, chunk_embedding)
            similarities.append({
                **chunk,
                "similarity_score": float(similarity)
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]
    
    async def embed_and_ingest_text(
        self,
        text: str,
        collection_id: str,
        vector_id: str = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Single-step: text → embedding → ProximaDB insert
        
        Args:
            text: Text to embed and insert
            collection_id: Target ProximaDB collection
            vector_id: Optional vector ID (auto-generated if None)
            additional_metadata: Additional metadata to include
            
        Returns:
            Result dict with insertion details
        """
        if not self.proximadb_client:
            raise RuntimeError("ProximaDB client not configured. Cannot perform direct ingestion.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        # Generate embedding
        embedding = self.embed_text(text)
        
        # Prepare metadata with text and embedding info
        metadata = {
            "text": text,
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content_type": "single_text",
            **(additional_metadata or {})
        }
        
        # Generate vector ID if not provided
        if not vector_id:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
            vector_id = f"embed_{text_hash}_{int(time.time())}"
        
        # Direct ProximaDB insert
        try:
            result = await self.proximadb_client.insert_vector(
                collection_id=collection_id,
                vector_id=vector_id,
                vector=embedding.tolist(),
                metadata=metadata
            )
            
            return {
                "success": True,
                "vector_id": vector_id,
                "collection_id": collection_id,
                "embedding_dimension": len(embedding),
                "metadata_keys": list(metadata.keys()),
                "text_length": len(text),
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to insert vector into ProximaDB: {e}")
            raise RuntimeError(f"ProximaDB insertion failed: {e}")
    
    async def chunk_embed_and_ingest(
        self,
        text: str,
        collection_id: str,
        document_id: str = None,
        strategy: str = "sliding_window",
        chunk_size: int = 512,
        overlap: int = 128,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Single-step: document → chunks → embeddings → ProximaDB batch insert
        
        Args:
            text: Document text to chunk, embed, and insert
            collection_id: Target ProximaDB collection
            document_id: Document identifier (auto-generated if None)
            strategy: Chunking strategy
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            additional_metadata: Additional metadata for all chunks
            
        Returns:
            Result dict with batch insertion details
        """
        if not self.proximadb_client:
            raise RuntimeError("ProximaDB client not configured. Cannot perform direct ingestion.")
        
        if not text or not text.strip():
            logger.error(f"Text is empty or whitespace only")
            raise ValueError("Text cannot be empty")
        
        logger.debug(f"chunk_embed_and_ingest called with text_length={len(text)}, collection_id={collection_id}")
        
        # Generate document ID if not provided
        if not document_id:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
            document_id = f"doc_{text_hash}_{int(time.time())}"
        
        logger.debug(f"Document ID: {document_id}")
        
        # Generate chunks with embeddings
        base_metadata = {
            "document_id": document_id,
            "content_type": "document_chunk",
            "chunking_strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            **(additional_metadata or {})
        }
        
        logger.debug(f"Calling chunk_and_embed with strategy={strategy}, chunk_size={chunk_size}, overlap={overlap}")
        chunks = self.chunk_and_embed(
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            document_id=document_id,
            metadata=base_metadata
        )
        
        logger.debug(f"chunk_and_embed returned {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks were generated from the text")
            raise ValueError("No chunks were generated from the text")
        
        # Prepare vector records for ProximaDB batch insert
        vector_records = []
        for chunk in chunks:
            vector_records.append({
                "id": chunk["id"],
                "vector": chunk["embedding"],
                "metadata": chunk["metadata"]
            })
        
        # Batch insert via ProximaDB
        try:
            batch_result = await self.proximadb_client.insert_vectors(
                collection_id=collection_id,
                records=vector_records
            )
            
            return {
                "success": True,
                "collection_id": collection_id,
                "document_id": document_id,
                "chunks_inserted": len(vector_records),
                "vector_ids": [r["id"] for r in vector_records],
                "chunking_strategy": strategy,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_text_length": len(text),
                "model_used": self.model_name,
                "embedding_dimension": self.dimension
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to batch insert vectors into ProximaDB: {e}")
            raise RuntimeError(f"ProximaDB batch insertion failed: {e}")

# Global embedding service instance
_embedding_service: Optional[ProximaDBEmbeddingService] = None

def get_embedding_service(
    model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
    proximadb_client=None
) -> ProximaDBEmbeddingService:
    """Get or create global embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None or _embedding_service.model_name != model_name:
        # Use appropriate cache directory based on environment
        cache_dir = os.getenv('EMBEDDING_CACHE_DIR', '/app/demo/embedding_cache')
        
        # Handle both container and local paths
        if not os.path.exists(cache_dir):
            # Try alternative paths
            if os.path.exists('/app/demo/embedding_cache'):
                cache_dir = '/app/demo/embedding_cache'
            elif os.path.exists('demo/embedding_cache'):
                cache_dir = 'demo/embedding_cache'
            elif os.path.exists('./embedding_cache'):
                cache_dir = './embedding_cache'
            else:
                # Create the directory
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                except Exception as e:
                    # Fallback to temp directory
                    cache_dir = '/tmp/embedding_cache'
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.warning(f"Using temporary cache directory: {cache_dir}")
        
        _embedding_service = ProximaDBEmbeddingService(
            model_name=model_name,
            seed=seed,
            cache_dir=cache_dir,
            proximadb_client=proximadb_client
        )
    
    return _embedding_service

# API endpoints for Web UI
async def embed_text_endpoint(text: str, model: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """API endpoint for single text embedding"""
    service = get_embedding_service(model)
    embedding = service.embed_text(text)
    
    return {
        "success": True,
        "embedding": embedding.tolist(),
        "dimension": service.dimension,
        "model": service.model_name,
        "text_length": len(text)
    }

async def chunk_and_embed_endpoint(
    text: str,
    strategy: str = "sliding_window",
    chunk_size: int = 512,
    overlap: int = 128,
    model: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """API endpoint for chunking and embedding"""
    service = get_embedding_service(model)
    chunks = service.chunk_and_embed(
        text=text,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    return {
        "success": True,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "model": service.model_name,
        "dimension": service.dimension,
        "chunking_strategy": strategy
    }

if __name__ == "__main__":
    # Test the embedding service
    service = get_embedding_service()
    
    # Test text
    test_text = """
    ProximaDB is a high-performance vector database designed for modern AI applications.
    It supports multiple storage engines including NOVA for analytics and SST for write optimization.
    The database provides both REST and gRPC APIs with comprehensive query capabilities.
    Advanced features include SIMD acceleration, GPU support, and sophisticated indexing algorithms.
    """
    
    print("🧪 Testing ProximaDB Embedding Service")
    print(f"📊 Model: {service.get_model_info()}")
    
    # Test single embedding
    embedding = service.embed_text("Hello, ProximaDB!")
    print(f"✅ Single embedding: {embedding.shape} {embedding.dtype}")
    
    # Test chunking and embedding
    chunks = service.chunk_and_embed(test_text, strategy="sliding_window", chunk_size=100, overlap=20)
    print(f"✅ Chunked into {len(chunks)} pieces")
    
    # Test similarity search
    results = service.search_similar_chunks("vector database performance", chunks, top_k=2)
    print(f"✅ Found {len(results)} similar chunks")
    
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['similarity_score']:.3f} | Text: {result['text'][:60]}...")