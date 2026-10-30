"""
Tests for real embedding providers (with actual model downloads)

These tests require sentence-transformers to be installed and will download
models from HuggingFace on first run. Models are cached in ~/.cache/huggingface/

To skip these tests (e.g., in CI without models):
    pytest -m "not requires_models"

To run only these tests:
    pytest -m "requires_models"

NOTE: These tests are skipped by default as they require large model downloads
and sentence-transformers installation. Enable with --run-slow flag or remove skip.
"""

import pytest

pytest.skip(
    "Tests require sentence-transformers and model downloads. Use --run-slow to enable.",
    allow_module_level=True,
)

import numpy as np

try:
    import sentence_transformers

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from proximadb_sdk.embedding_providers.core import ProviderConfig
from proximadb_sdk.embedding_providers.providers.local.bge import BGEProvider
from proximadb_sdk.embedding_providers.providers.local.e5 import E5Provider
from proximadb_sdk.embedding_providers.providers.local.gte_qwen import GTEQwenProvider
from proximadb_sdk.embedding_providers.providers.local.sentence_transformer import (
    SentenceTransformerProvider,
)
from proximadb_sdk.embedding_providers.providers.local.sfr import SFRProvider

# Mark all tests in this module as requiring models
pytestmark = pytest.mark.requires_models

# Skip entire module if sentence-transformers not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    pytestmark = pytest.mark.skip(reason="sentence-transformers not installed")


# Test fixtures
@pytest.fixture
def sample_texts():
    """Sample texts for embedding tests"""
    return [
        "This is a test sentence about machine learning.",
        "Python is a popular programming language.",
        "Vector databases store high-dimensional embeddings.",
    ]


@pytest.fixture
def sample_documents():
    """Sample documents with metadata"""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "category": "tech",
        },
        {
            "text": "Python supports multiple programming paradigms.",
            "category": "programming",
        },
        {"text": "Embeddings capture semantic meaning of text.", "category": "nlp"},
    ]


# ============================================================================
# GTE-Qwen Provider Tests (#1 MTEB Multilingual)
# ============================================================================


class TestGTEQwenProvider:
    """Test suite for GTE-Qwen embedding provider"""

    @pytest.fixture
    def gte_qwen_config(self):
        """Configuration for gte-Qwen 1.5B model"""
        return ProviderConfig(
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            dimension=1536,
            batch_size=16,
            normalize=True,
            # Note: trust_remote_code is automatically set to False for compatibility
        )

    def test_initialization(self, gte_qwen_config):
        """Test provider initialization"""
        provider = GTEQwenProvider(gte_qwen_config)
        assert provider.is_available()
        assert provider.get_dimension() == 1536

    def test_embed_texts(self, gte_qwen_config, sample_texts):
        """Test basic text embedding"""
        provider = GTEQwenProvider(gte_qwen_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 1536)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embed_query(self, gte_qwen_config):
        """Test query embedding with instruction"""
        provider = GTEQwenProvider(gte_qwen_config)
        query_emb = provider.embed_query("What is machine learning?")

        assert query_emb.shape == (1536,)
        assert np.isclose(np.linalg.norm(query_emb), 1.0, atol=1e-5)

    def test_embed_documents(self, gte_qwen_config, sample_documents):
        """Test document embedding"""
        provider = GTEQwenProvider(gte_qwen_config)
        doc_embs = provider.embed_documents(sample_documents)

        assert doc_embs.shape == (len(sample_documents), 1536)

    def test_multilingual_support(self, gte_qwen_config):
        """Test multilingual embedding (gte-Qwen supports 100+ languages)"""
        provider = GTEQwenProvider(gte_qwen_config)

        multilingual_texts = [
            "Hello world",  # English
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "你好世界",  # Chinese
            "こんにちは世界",  # Japanese
        ]

        embeddings = provider.embed_texts(multilingual_texts)
        assert embeddings.shape == (5, 1536)

    def test_model_info(self, gte_qwen_config):
        """Test model information retrieval"""
        provider = GTEQwenProvider(gte_qwen_config)
        info = provider.get_model_info()

        assert info["model_name"] == "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        assert info["dimension"] == 1536
        assert info["provider"] == "gte-qwen"
        assert info["available"] is True


# ============================================================================
# SFR Provider Tests (Top English Accuracy)
# ============================================================================


class TestSFRProvider:
    """Test suite for SFR embedding provider"""

    @pytest.fixture
    def sfr_config(self):
        """Configuration for SFR model"""
        return ProviderConfig(
            model_name="Salesforce/SFR-Embedding-2_R",
            dimension=4096,
            batch_size=16,
            normalize=True,
        )

    @pytest.mark.slow
    def test_initialization(self, sfr_config):
        """Test provider initialization (marked slow due to large model)"""
        provider = SFRProvider(sfr_config)
        assert provider.is_available()
        assert provider.get_dimension() == 4096

    @pytest.mark.slow
    def test_embed_texts(self, sfr_config, sample_texts):
        """Test basic text embedding"""
        provider = SFRProvider(sfr_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 4096)

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.slow
    def test_query_vs_document_embeddings(self, sfr_config):
        """Test that query and document embeddings differ (instruction effect)"""
        provider = SFRProvider(sfr_config)

        text = "machine learning algorithms"

        # Query embedding (with instruction)
        query_emb = provider.embed_query(text)

        # Document embedding (without instruction)
        doc_emb = provider.embed_texts([text], is_query=False)[0]

        # They should be different due to instruction prefix
        similarity = np.dot(query_emb, doc_emb)
        assert similarity < 0.99  # Not identical


# ============================================================================
# BGE Provider Tests (Best Retrieval)
# ============================================================================


class TestBGEProvider:
    """Test suite for BGE embedding provider"""

    @pytest.fixture
    def bge_large_config(self):
        """Configuration for BGE large model"""
        return ProviderConfig(
            model_name="BAAI/bge-large-en-v1.5",
            dimension=1024,
            batch_size=32,
            normalize=True,
        )

    @pytest.fixture
    def bge_small_config(self):
        """Configuration for BGE small model (faster)"""
        return ProviderConfig(
            model_name="BAAI/bge-small-en-v1.5",
            dimension=384,
            batch_size=64,
            normalize=True,
        )

    def test_initialization_small(self, bge_small_config):
        """Test provider initialization with small model"""
        provider = BGEProvider(bge_small_config)
        assert provider.is_available()
        assert provider.get_dimension() == 384

    def test_embed_texts_small(self, bge_small_config, sample_texts):
        """Test basic text embedding with small model"""
        provider = BGEProvider(bge_small_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 384)

    def test_embed_query_with_instruction(self, bge_small_config):
        """Test query embedding with BGE instruction prefix"""
        provider = BGEProvider(bge_small_config)

        query = "best machine learning algorithms"
        query_emb = provider.embed_query(query)

        assert query_emb.shape == (384,)
        assert np.isclose(np.linalg.norm(query_emb), 1.0, atol=1e-5)

    def test_embed_documents(self, bge_small_config, sample_documents):
        """Test document embedding (no instruction)"""
        provider = BGEProvider(bge_small_config)
        doc_embs = provider.embed_documents(sample_documents)

        assert doc_embs.shape == (len(sample_documents), 384)

    def test_batch_processing(self, bge_small_config):
        """Test efficient batch processing"""
        provider = BGEProvider(bge_small_config)

        # Large batch
        texts = [f"This is test sentence number {i}" for i in range(100)]
        embeddings = provider.embed_texts(texts)

        assert embeddings.shape == (100, 384)

    @pytest.mark.slow
    def test_large_model(self, bge_large_config, sample_texts):
        """Test BGE large model (1024 dims)"""
        provider = BGEProvider(bge_large_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 1024)


# ============================================================================
# E5 Provider Tests (General Purpose)
# ============================================================================


class TestE5Provider:
    """Test suite for E5 embedding provider"""

    @pytest.fixture
    def e5_base_config(self):
        """Configuration for E5 base model"""
        return ProviderConfig(
            model_name="intfloat/e5-base-v2",
            dimension=768,
            batch_size=32,
            normalize=True,
        )

    def test_initialization(self, e5_base_config):
        """Test provider initialization"""
        provider = E5Provider(e5_base_config)
        assert provider.is_available()
        assert provider.get_dimension() == 768

    def test_embed_query_with_prefix(self, e5_base_config):
        """Test query embedding with 'query: ' prefix"""
        provider = E5Provider(e5_base_config)

        query = "python programming tutorial"
        query_emb = provider.embed_query(query)

        assert query_emb.shape == (768,)
        assert np.isclose(np.linalg.norm(query_emb), 1.0, atol=1e-5)

    def test_embed_passages(self, e5_base_config):
        """Test passage embedding with 'passage: ' prefix"""
        provider = E5Provider(e5_base_config)

        passages = [
            "Python is a high-level programming language",
            "It supports multiple programming paradigms",
        ]

        passage_embs = provider.embed_passages(passages)
        assert passage_embs.shape == (2, 768)

    def test_query_passage_difference(self, e5_base_config):
        """Test that query and passage prefixes create different embeddings"""
        provider = E5Provider(e5_base_config)

        text = "machine learning algorithms"

        query_emb = provider.embed_query(text)
        passage_emb = provider.embed_passages([text])[0]

        # Should be different due to different prefixes
        similarity = np.dot(query_emb, passage_emb)
        assert similarity < 0.99

    def test_embed_documents(self, e5_base_config, sample_documents):
        """Test document embedding"""
        provider = E5Provider(e5_base_config)
        doc_embs = provider.embed_documents(sample_documents)

        assert doc_embs.shape == (len(sample_documents), 768)


# ============================================================================
# Sentence-Transformers Provider Tests (Most Versatile)
# ============================================================================


class TestSentenceTransformerProvider:
    """Test suite for Sentence-Transformers provider"""

    @pytest.fixture
    def minilm_config(self):
        """Configuration for MiniLM model (fastest)"""
        return ProviderConfig(
            model_name="all-MiniLM-L6-v2", dimension=384, batch_size=64, normalize=True
        )

    @pytest.fixture
    def mpnet_config(self):
        """Configuration for MPNet model (better quality)"""
        return ProviderConfig(
            model_name="all-mpnet-base-v2", dimension=768, batch_size=32, normalize=True
        )

    def test_initialization_minilm(self, minilm_config):
        """Test provider initialization with MiniLM"""
        provider = SentenceTransformerProvider(minilm_config)
        assert provider.is_available()
        assert provider.get_dimension() == 384

    def test_embed_texts_minilm(self, minilm_config, sample_texts):
        """Test text embedding with MiniLM"""
        provider = SentenceTransformerProvider(minilm_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 384)

    def test_embed_documents(self, minilm_config, sample_documents):
        """Test document embedding"""
        provider = SentenceTransformerProvider(minilm_config)
        doc_embs = provider.embed_documents(sample_documents)

        assert doc_embs.shape == (len(sample_documents), 384)

    def test_high_throughput(self, minilm_config):
        """Test high throughput with large batch"""
        provider = SentenceTransformerProvider(minilm_config)

        # 500 texts
        texts = [f"Sample text number {i}" for i in range(500)]
        embeddings = provider.embed_texts(texts)

        assert embeddings.shape == (500, 384)

    def test_mpnet_model(self, mpnet_config, sample_texts):
        """Test MPNet model (higher quality)"""
        provider = SentenceTransformerProvider(mpnet_config)
        embeddings = provider.embed_texts(sample_texts)

        assert embeddings.shape == (len(sample_texts), 768)


# ============================================================================
# Cross-Provider Comparison Tests
# ============================================================================


class TestCrossProviderComparison:
    """Tests comparing different providers"""

    def test_semantic_similarity_preserved(self):
        """Test that semantic similarity is preserved across providers"""
        # Use small/fast models for comparison
        bge_config = ProviderConfig(
            model_name="BAAI/bge-small-en-v1.5", dimension=384, normalize=True
        )

        minilm_config = ProviderConfig(
            model_name="all-MiniLM-L6-v2", dimension=384, normalize=True
        )

        similar_texts = [
            "machine learning algorithms",
            "artificial intelligence methods",  # Semantically similar
        ]

        different_text = "apple pie recipe"  # Semantically different

        # BGE embeddings
        bge_provider = BGEProvider(bge_config)
        bge_embs = bge_provider.embed_texts(similar_texts + [different_text])

        # Sentence-Transformers embeddings
        st_provider = SentenceTransformerProvider(minilm_config)
        st_embs = st_provider.embed_texts(similar_texts + [different_text])

        # For both providers, similar texts should be more similar than different texts
        for embs in [bge_embs, st_embs]:
            similar_sim = np.dot(embs[0], embs[1])
            different_sim1 = np.dot(embs[0], embs[2])
            different_sim2 = np.dot(embs[1], embs[2])

            assert similar_sim > different_sim1
            assert similar_sim > different_sim2

    def test_dimension_consistency(self):
        """Test that each provider returns consistent dimensions"""
        configs = [
            (
                GTEQwenProvider,
                {
                    "model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                    "dimension": 1536,
                },
            ),
            (BGEProvider, {"model_name": "BAAI/bge-small-en-v1.5", "dimension": 384}),
            (E5Provider, {"model_name": "intfloat/e5-base-v2", "dimension": 768}),
            (
                SentenceTransformerProvider,
                {"model_name": "all-MiniLM-L6-v2", "dimension": 384},
            ),
        ]

        test_texts = ["test sentence one", "test sentence two"]

        for provider_class, config_dict in configs:
            config = ProviderConfig(**config_dict, normalize=True)
            provider = provider_class(config)

            embeddings = provider.embed_texts(test_texts)
            assert embeddings.shape[0] == len(test_texts)
            assert embeddings.shape[1] == config_dict["dimension"]


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def fast_provider(self):
        """Fast provider for edge case testing"""
        config = ProviderConfig(
            model_name="all-MiniLM-L6-v2", dimension=384, normalize=True
        )
        return SentenceTransformerProvider(config)

    def test_empty_input(self, fast_provider):
        """Test embedding empty list"""
        embeddings = fast_provider.embed_texts([])
        assert embeddings.shape == (0,)

    def test_single_text(self, fast_provider):
        """Test embedding single text"""
        embedding = fast_provider.embed_text("single text")
        assert embedding.shape == (384,)

    def test_very_long_text(self, fast_provider):
        """Test embedding very long text (will be truncated)"""
        long_text = " ".join(["word"] * 1000)  # Very long text
        embedding = fast_provider.embed_text(long_text)
        assert embedding.shape == (384,)

    def test_special_characters(self, fast_provider):
        """Test embedding text with special characters"""
        special_texts = [
            "Hello! How are you?",
            "Price: $99.99",
            "Email: test@example.com",
            "Code: if (x > 0) { return true; }",
        ]
        embeddings = fast_provider.embed_texts(special_texts)
        assert embeddings.shape == (4, 384)

    def test_unicode_text(self, fast_provider):
        """Test embedding unicode text"""
        unicode_texts = ["Hello 世界", "Café résumé", "Emoji: 😀 🚀 ✨"]
        embeddings = fast_provider.embed_texts(unicode_texts)
        assert embeddings.shape == (3, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "requires_models"])
