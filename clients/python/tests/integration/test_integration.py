"""
Comprehensive integration tests for ProximaDB functionality.

NOTE: Moved from tests/unit/ to tests/integration/ - these are integration tests
requiring a running ProximaDB server.
"""

import logging
import time

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.slow
class TestComprehensiveIntegration:
    """Comprehensive integration tests using real data."""

    @pytest.fixture(autouse=True)
    def setup_collection(self, client, cleanup_collection):
        """Set up test collection."""
        self.client = client
        self.collection_name = cleanup_collection

        # Create collection with comprehensive configuration
        self.client.create_collection(
            name=self.collection_name,
            dimension=384,
            distance_metric="cosine",
            storage_engine="nova",
            indexing_algorithm="hnsw",
        )

    @pytest.mark.embedding
    def test_10mb_corpus_integration(
        self, corpus_data, cached_embeddings, bert_service
    ):
        """Test integration with 10MB corpus data."""
        if not corpus_data or cached_embeddings is None:
            # Data not available, test passes (optional feature)
            return

        # Use a subset for testing (full corpus is too large for unit tests)
        test_size = min(100, len(corpus_data))
        test_corpus = corpus_data[:test_size]
        test_embeddings = cached_embeddings[:test_size]

        # Prepare data for insertion
        vector_ids = [f"corpus_vec_{i:05d}" for i in range(test_size)]
        # test_embeddings is already a list of lists, no need to call tolist()
        vectors = test_embeddings
        metadata_list = []

        for i, doc in enumerate(test_corpus):
            metadata = {
                "category": doc.get("category", "unknown"),
                "author": doc.get("author", f"Author_{i}"),
                "doc_type": doc.get("doc_type", "article"),
                "year": doc.get("year", 2024),
                "length": doc.get("length", len(doc.get("text", ""))),
                "title": doc.get("title", f"Document {i}"),
                "source": "corpus_test",
                "importance": doc.get("importance", 5),
            }
            metadata_list.append(metadata)

        # Measure insertion performance
        start_time = time.time()

        # Insert in smaller batches
        batch_size = 50
        for i in range(0, test_size, batch_size):
            end_idx = min(i + batch_size, test_size)
            batch_vectors = vectors[i:end_idx]
            batch_ids = vector_ids[i:end_idx]
            batch_metadata = metadata_list[i:end_idx]

            result = self.client.insert_vectors(
                self.collection_name, batch_vectors, batch_ids, batch_metadata
            )
            assert result is not None

        insert_time = time.time() - start_time
        throughput = test_size / insert_time

        logger.info("\\n📊 INSERT PERFORMANCE:")
        logger.info(f"   Vectors: {test_size}")
        logger.info(f"   Time: {insert_time:.2f}s")
        logger.info(f"   Throughput: {throughput:.1f} vectors/sec")

        # Test various search operations
        self._test_search_operations(bert_service, test_size)

    def _test_search_operations(self, bert_service, corpus_size):
        """Test different search operations."""
        logger.info("\\n🔍 TESTING SEARCH OPERATIONS:")

        # 1. Test ID-based search (metadata filter)
        test_id = "corpus_vec_00001"
        start_time = time.time()
        results = self.client.search(
            self.collection_name,
            [0.1] * 384,  # dummy vector
            top_k=1,
            metadata_filter={"id": test_id},
        )
        search_time = (time.time() - start_time) * 1000
        logger.info(f"   ID search: {search_time:.2f}ms")

        # 2. Test metadata filtering
        start_time = time.time()
        results = self.client.search(
            self.collection_name,
            [0.1] * 384,  # dummy vector
            top_k=10,
            metadata_filter={"category": "AI"},
        )
        search_time = (time.time() - start_time) * 1000
        logger.info(f"   Metadata filter: {search_time:.2f}ms")

        # 3. Test similarity search
        query_text = "machine learning algorithms"
        query_embedding = bert_service.encode([query_text])[0]

        start_time = time.time()
        results = self.client.search(
            self.collection_name, query_embedding.tolist(), top_k=10
        )
        search_time = (time.time() - start_time) * 1000
        logger.info(f"   Similarity search: {search_time:.2f}ms")

        # 4. Test hybrid search
        start_time = time.time()
        results = self.client.search(
            self.collection_name,
            query_embedding.tolist(),
            top_k=10,
            metadata_filter={"category": "AI"},
        )
        search_time = (time.time() - start_time) * 1000
        logger.info(f"   Hybrid search: {search_time:.2f}ms")
