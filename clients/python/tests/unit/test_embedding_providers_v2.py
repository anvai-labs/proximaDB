"""
Tests for optimized embedding providers v2 architecture

Tests the new core infrastructure (base, config, registry, cache, mixins)
and the refactored gte-Qwen provider.
"""

import sys
import types

import numpy as np
import pytest

# Core components
from proximadb_sdk.embedding_providers.core import (
    BaseEmbeddingProvider,
    ModelCache,
    ModelMetadata,
    ProviderConfig,
    ProviderRegistry,
)

# Mixins
from proximadb_sdk.embedding_providers.mixins import NormalizationMixin

# Import the new provider
from proximadb_sdk.embedding_providers.providers.local.gte_qwen import GTEQwenProvider


# ---------------------------------------------------------------------------
# Offline fake for sentence-transformers.
#
# GTEQwenProvider speaks SentenceTransformerMixin, which loads its model lazily
# via ``from sentence_transformers import SentenceTransformer``. Without a stub
# this would download a real 1.5B-parameter model. Sibling offline test files
# also install their own (incompatible) fakes into ``sys.modules`` for the whole
# pytest session, so the TestGTEQwenProvider autouse fixture re-pins THIS fake
# per-test to keep the provider offline and deterministic.
# ---------------------------------------------------------------------------
class _FakeSentenceTransformer:
    """Deterministic fake SentenceTransformer (returns unit-normalized rows)."""

    def __init__(
        self,
        model_name,
        device=None,
        trust_remote_code=False,
        cache_folder=None,
        backend=None,
        prompts=None,
        **kwargs,
    ):
        self.model_name = model_name
        # GTE-Qwen2-1.5B-instruct is 1536-d (the provider default); other models
        # in the catalog pick up their real dimension via the config metadata, so
        # the fake only has to produce SOME embedding of the expected width.
        self._dim = 1536

    def _embed(self, texts):
        n = len(texts)
        emb = np.arange(n * self._dim, dtype=np.float32).reshape(n, self._dim)
        # Unit-normalize each row so NormalizationMixin.check_normalized passes.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        **kwargs,
    ):
        return self._embed(texts)


def _install_st_stub():
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer
    return mod


class TestModelMetadata:
    """Test ModelMetadata dataclass"""

    def test_basic_metadata(self):
        """Test basic model metadata creation"""
        metadata = ModelMetadata(name="test-model", dimension=768, max_length=512)

        assert metadata.name == "test-model"
        assert metadata.dimension == 768
        assert metadata.max_length == 512
        assert metadata.provider_type == "sentence-transformer"  # default
        assert metadata.requires_instruction is False  # default

    def test_metadata_with_instruction(self):
        """Test metadata with instruction support"""
        metadata = ModelMetadata(
            name="instructed-model",
            dimension=1024,
            requires_instruction=True,
            instruction_template="Query: {query}",
        )

        assert metadata.requires_instruction is True
        assert metadata.instruction_template == "Query: {query}"

    def test_metadata_str(self):
        """Test string representation"""
        metadata = ModelMetadata(
            name="model-1", dimension=384, mteb_score=65.5, languages="multilingual"
        )

        str_repr = str(metadata)
        assert "model-1" in str_repr
        assert "384D" in str_repr
        assert "65.5" in str_repr


class TestProviderConfig:
    """Test ProviderConfig"""

    def test_basic_config(self):
        """Test basic configuration"""
        model = ModelMetadata(name="test", dimension=768)
        config = ProviderConfig(model=model)

        assert config.model.name == "test"
        assert config.batch_size == 32  # default
        assert config.normalize is True  # default

    def test_config_merge(self):
        """Test configuration merging"""
        model = ModelMetadata(name="test", dimension=768)
        config1 = ProviderConfig(model=model, batch_size=32)

        config2 = config1.merge(batch_size=64, normalize=False)

        # Original unchanged
        assert config1.batch_size == 32
        # New config updated
        assert config2.batch_size == 64
        assert config2.normalize is False

    def test_config_extra_merge(self):
        """Test merging extra params"""
        model = ModelMetadata(name="test", dimension=768)
        config1 = ProviderConfig(model=model, extra={"key1": "value1"})

        config2 = config1.merge(extra={"key2": "value2"})

        # Both keys should be present
        assert config2.extra["key1"] == "value1"
        assert config2.extra["key2"] == "value2"


class TestProviderRegistry:
    """Test ProviderRegistry"""

    @classmethod
    def setup_class(cls):
        """Ensure provider is imported/registered before tests"""
        # Import triggers registration via decorator

    def test_registry_decorator(self):
        """Test provider registration via decorator"""
        # Save current registry state
        saved_providers = ProviderRegistry._providers.copy()
        saved_metadata = ProviderRegistry._metadata.copy()
        saved_aliases = ProviderRegistry._aliases.copy()

        try:
            # Clear registry for test
            ProviderRegistry.clear()

            test_models = {"model-1": ModelMetadata(name="model-1", dimension=384)}

            @ProviderRegistry.register(
                name="test-provider",
                models=test_models,
                aliases=["alias1"],
                description="Test provider",
            )
            class TestProvider(BaseEmbeddingProvider):
                def default_config(self):
                    return ProviderConfig(model=test_models["model-1"])

                def _load_model(self):
                    return None

                def embed(self, texts):
                    return np.zeros((len(texts), 384))

            # Check registration
            assert "test-provider" in ProviderRegistry.list_providers()
            provider_class = ProviderRegistry.get_provider("test-provider")
            assert provider_class == TestProvider

            # Check alias
            provider_via_alias = ProviderRegistry.get_provider("alias1")
            assert provider_via_alias == TestProvider
        finally:
            # Restore registry state
            ProviderRegistry._providers = saved_providers
            ProviderRegistry._metadata = saved_metadata
            ProviderRegistry._aliases = saved_aliases

    def test_get_models(self):
        """Test getting models for a provider"""
        # Re-import to ensure registration after previous test cleared registry

        models = ProviderRegistry.get_models("gte-qwen")
        assert len(models) > 0
        assert "Alibaba-NLP/gte-Qwen2-1.5B-instruct" in models

    def test_get_provider_info(self):
        """Test getting provider info"""
        # Re-import to ensure registration after previous test cleared registry

        info = ProviderRegistry.get_provider_info("gte-qwen")
        assert info["name"] == "gte-qwen"
        assert "models" in info
        assert len(info["models"]) > 0


class TestModelCache:
    """Test ModelCache"""

    def test_cache_singleton(self):
        """Test that cache is a singleton"""
        cache1 = ModelCache()
        cache2 = ModelCache()
        assert cache1 is cache2

    def test_get_or_load(self):
        """Test cache get_or_load"""
        cache = ModelCache()
        cache.clear()  # Start fresh

        load_count = [0]

        def loader():
            load_count[0] += 1
            return {"data": "model"}

        # First call loads
        model1 = cache.get_or_load("test-key", loader)
        assert load_count[0] == 1
        assert model1["data"] == "model"

        # Second call uses cache
        model2 = cache.get_or_load("test-key", loader)
        assert load_count[0] == 1  # Not incremented
        assert model1 is model2  # Same instance

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = ModelCache()
        cache.clear()

        cache.get_or_load("key1", lambda: "model1")
        cache.get_or_load("key2", lambda: "model2")

        assert cache.size() == 2

        # Clear specific key
        cache.clear("key1")
        assert cache.size() == 1

        # Clear all
        cache.clear()
        assert cache.size() == 0

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = ModelCache()
        cache.clear()
        cache.reset_stats()

        # First call - miss
        cache.get_or_load("key1", lambda: "model")

        # Second call - hit
        cache.get_or_load("key1", lambda: "model")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["loads"] == 1


class TestNormalizationMixin:
    """Test NormalizationMixin"""

    def test_normalize_1d(self):
        """Test normalizing 1D array"""
        vec = np.array([3.0, 4.0])
        normalized = NormalizationMixin.normalize_embeddings(vec)

        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0)

    def test_normalize_2d(self):
        """Test normalizing 2D array"""
        vecs = np.array([[3.0, 4.0], [1.0, 0.0]])
        normalized = NormalizationMixin.normalize_embeddings(vecs)

        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, [1.0, 1.0])

    def test_check_normalized(self):
        """Test checking if embeddings are normalized"""
        vecs = np.array([[0.6, 0.8], [1.0, 0.0]])
        assert NormalizationMixin.check_normalized(vecs)

        not_normalized = np.array([[3.0, 4.0]])
        assert not NormalizationMixin.check_normalized(not_normalized)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        vec3 = np.array([1.0, 0.0])

        sim12 = NormalizationMixin.get_cosine_similarity(vec1, vec2)
        assert np.isclose(sim12, 0.0)  # Orthogonal

        sim13 = NormalizationMixin.get_cosine_similarity(vec1, vec3)
        assert np.isclose(sim13, 1.0)  # Parallel


# Tests that require actual models (marked as requires_models)
pytestmark_models = pytest.mark.requires_models


@pytestmark_models
class TestGTEQwenProvider:
    """Test gte-Qwen Provider"""

    @pytest.fixture(autouse=True)
    def _offline_sentence_transformers(self, monkeypatch):
        """Pin the offline fake SentenceTransformer into sys.modules for each
        test and start from an empty ModelCache. Without this, the provider's
        lazy ``_load_model`` would either download the real 1.5B model or pick
        up an incompatible fake installed by a sibling offline test file."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", _install_st_stub())
        ModelCache().clear()
        ModelCache().reset_stats()
        yield
        ModelCache().clear()

    def test_initialization(self):
        """Test provider initialization"""
        provider = GTEQwenProvider()
        assert provider is not None
        assert provider.config.model.dimension == 1536  # Default model

    def test_default_config(self):
        """Test default configuration"""
        provider = GTEQwenProvider()
        config = provider.config

        assert config.model.name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        assert config.model.dimension == 1536
        assert config.batch_size == 16
        assert config.normalize is True

    def test_embed_query(self):
        """Test query embedding with instruction"""
        provider = GTEQwenProvider()
        query_emb = provider.embed_query("What is machine learning?")

        assert query_emb.shape == (1536,)
        assert NormalizationMixin.check_normalized(query_emb)

    def test_embed_passages(self):
        """Test passage embedding without instruction"""
        provider = GTEQwenProvider()
        passages = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
        ]

        passage_embs = provider.embed_passages(passages)

        assert passage_embs.shape == (2, 1536)
        assert NormalizationMixin.check_normalized(passage_embs)

    def test_embed_batch(self):
        """Test batch embedding"""
        provider = GTEQwenProvider()
        texts = ["text1", "text2", "text3", "text4"]

        embeddings = provider.embed(texts)

        assert embeddings.shape == (4, 1536)
        assert NormalizationMixin.check_normalized(embeddings)

    def test_model_caching(self):
        """Test that models are cached across instances"""
        cache = ModelCache()

        # Clear cache to start fresh for this test
        cache.clear()
        cache.reset_stats()

        initial_size = cache.size()
        assert initial_size == 0  # Cache is empty

        # Create first provider
        provider1 = GTEQwenProvider()
        provider1.ensure_initialized()

        size_after_1 = cache.size()
        assert size_after_1 == 1  # One model loaded

        stats = cache.stats()
        assert stats["loads"] == 1  # First load

        # Create second provider (should reuse model)
        provider2 = GTEQwenProvider()
        provider2.ensure_initialized()

        size_after_2 = cache.size()
        assert size_after_2 == 1  # Still only one model

        stats = cache.stats()
        assert stats["loads"] == 1  # No additional loads
        assert stats["hits"] >= 1  # At least one cache hit

    def test_context_manager(self):
        """Test provider as context manager"""
        with GTEQwenProvider() as provider:
            emb = provider.embed_query("test")
            assert emb.shape == (1536,)

        # After context, model should be cleaned up
        assert not provider._initialized

    def test_backward_compatibility_alias(self):
        """Test that GTEQwenProvider works correctly"""
        provider = GTEQwenProvider()
        assert isinstance(provider, GTEQwenProvider)
        assert provider.config.model.dimension == 1536
