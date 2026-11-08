"""
Offline unit tests for proximadb_sdk.chunking.

Strategy
--------
* FULLY OFFLINE: no network, no server, no model downloads, no embedded DB.
* The chunking strategies themselves are pure Python (no heavy deps at import
  time), so we can call the real TextChunker / strategy helpers directly.
* The ONLY blocking risk is proximadb_sdk.resource_pool.ResourcePool, which
  spins up a ThreadPoolExecutor + maintenance thread on construction. We never
  construct the real one: a trivial in-memory FakeResourcePool is monkeypatched
  over proximadb_sdk.chunking.ResourcePool for every test that exercises the
  ChunkerPool / PooledChunkerContext paths. Embedders are plain mock callables.
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Defensive stubs for heavy/optional deps in case any transitive import wants
# them. The chunking strategies used here are pure-python, but stubbing these
# before import keeps the module load offline and fast under all conditions.
# ---------------------------------------------------------------------------
for _name in ("torch", "sentence_transformers"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

from proximadb_sdk import chunking  # noqa: E402
from proximadb_sdk.chunking import (  # noqa: E402
    ChunkerFactory,
    ChunkerPool,
    ChunkingConfig,
    ChunkingStrategy,
    PooledChunkerContext,
    TextChunk,
    TextChunker,
    chunk_and_embed_records,
    chunk_and_embed_text,
    chunk_by_paragraphs,
    chunk_by_sentences,
    chunk_sliding_window,
    cleanup_chunker_pool,
    create_chunker,
    create_enhanced_semantic_chunker,
    create_records,
    create_vector_records,
    get_chunker_pool_stats,
    prepare_records,
    prepare_vector_records,
)


# ---------------------------------------------------------------------------
# Fake resource pool — trivial, no threads, no sleeps, no blocking.
# Matches the surface ChunkerPool.get_stats expects: acquire / release /
# get_metrics / health_check.
# ---------------------------------------------------------------------------
class FakeResourcePool:
    def __init__(self, factory=None, max_size=50, **kwargs):
        self.factory = factory
        self.max_size = max_size
        self._created = 0
        self._acquisitions = 0
        self._in_use = 0
        self._available = 0

    def acquire(self, *a, **k):
        self._acquisitions += 1
        res = self.factory.create() if self.factory else TextChunker()
        self._created += 1
        self._in_use += 1
        return res

    def release(self, resource, *a, **k):
        if self._in_use > 0:
            self._in_use -= 1
        self._available += 1

    def get_metrics(self):
        return {
            "total_acquisitions": self._acquisitions,
            "active_resources": self._in_use,
            "available_resources": self._available,
            "total_created": self._created,
        }

    def health_check(self):
        return "healthy"

    def shutdown(self):
        pass


@pytest.fixture(autouse=True)
def patch_resource_pool(monkeypatch):
    """Replace the real ResourcePool everywhere chunking references it."""
    monkeypatch.setattr(chunking, "ResourcePool", FakeResourcePool)
    yield


@pytest.fixture
def fresh_pool(monkeypatch):
    """A ChunkerPool that is NOT the leaked global singleton."""
    pool = ChunkerPool(max_pool_size=4)
    # Point the module global at our fresh pool so context-mgr/global helpers
    # exercise our fake too.
    monkeypatch.setattr(chunking, "_global_chunker_pool", pool)
    return pool


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class SimpleEmbedder:
    """Embedder exposing only embed_texts (no metadata variant)."""

    def __init__(self, dim=3):
        self.dim = dim
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]


class MetadataEmbedder:
    """Embedder exposing embed_texts_with_metadata."""

    def __init__(self, dim=2):
        self.dim = dim

    def embed_texts_with_metadata(self, texts):
        vecs = [[0.5] * self.dim for _ in texts]
        return vecs, {"model": "fake-embedder", "dim": self.dim}


class NumpyLikeVec:
    """Mimics an ndarray with .tolist()."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class TolistEmbedder:
    def embed_texts(self, texts):
        return NumpyLikeVec([[1.0, 2.0] for _ in texts])


# ===========================================================================
# TextChunker / strategy helpers
# ===========================================================================
SAMPLE = (
    "The quick brown fox. It jumped over the lazy dog. "
    "Then it ran away quickly. The end of the story arrived."
)


def test_text_chunker_default_config():
    chunker = TextChunker()
    assert chunker.config.strategy == ChunkingStrategy.SLIDING_WINDOW
    assert chunker._strategy is not None


def test_chunk_text_sentence_strategy():
    chunker = TextChunker(
        ChunkingConfig(strategy=ChunkingStrategy.SENTENCE, chunk_size=40)
    )
    chunks = chunker.chunk_text(SAMPLE, source_id="docA")
    assert chunks
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.text for c in chunks)


def test_chunk_text_default_source_id_and_reinit():
    chunker = TextChunker(ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH))
    chunker._strategy = None  # force the re-init branch
    chunks = chunker.chunk_text("First para.\n\nSecond para.")
    assert chunker._strategy is not None
    assert chunks


def test_chunk_text_with_metadata_passthrough():
    chunker = TextChunker(ChunkingConfig(strategy=ChunkingStrategy.SENTENCE))
    chunks = chunker.chunk_text("Hello there. General Kenobi.", "d", {"tag": "x"})
    assert chunks
    # base metadata threads into chunk metadata
    assert any(c.metadata.get("tag") == "x" for c in chunks)


def test_add_context_to_chunks_short_circuits():
    chunker = TextChunker()
    assert chunker.add_context_to_chunks([]) == []
    one = [TextChunk(text="solo", start_pos=0, end_pos=4, chunk_id="c0")]
    assert chunker.add_context_to_chunks(one) == one


def test_add_context_to_chunks_adds_prev_and_next():
    chunker = TextChunker()
    chunks = [
        TextChunk(text="A" * 80, start_pos=0, end_pos=80, chunk_id="c0"),
        TextChunk(text="B" * 80, start_pos=80, end_pos=160, chunk_id="c1"),
        TextChunk(text="C" * 80, start_pos=160, end_pos=240, chunk_id="c2"),
    ]
    out = chunker.add_context_to_chunks(chunks, context_size=10)
    assert len(out) == 3
    assert "next_context" in out[0].metadata
    assert out[0].metadata["has_context"] is True
    assert "prev_context" in out[1].metadata
    assert "next_context" in out[1].metadata
    assert "prev_context" in out[2].metadata
    # context truncated to context_size
    assert len(out[0].metadata["next_context"]) == 10


def test_add_context_short_texts_not_truncated():
    chunker = TextChunker()
    chunks = [
        TextChunk(text="hi", start_pos=0, end_pos=2, chunk_id="c0"),
        TextChunk(text="yo", start_pos=2, end_pos=4, chunk_id="c1"),
    ]
    out = chunker.add_context_to_chunks(chunks, context_size=50)
    assert out[0].metadata["next_context"] == "yo"
    assert out[1].metadata["prev_context"] == "hi"


def test_chunk_by_sentences_helper():
    chunks = chunk_by_sentences(SAMPLE, chunk_size=30, document_id="sent-doc")
    assert chunks
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_by_sentences_default_doc_id():
    chunks = chunk_by_sentences("One. Two. Three.")
    assert chunks


def test_chunk_by_paragraphs_helper():
    text = "Para one line.\n\nPara two line.\n\nPara three line."
    chunks = chunk_by_paragraphs(text, max_size=50, document_id="para-doc")
    assert chunks


def test_chunk_by_paragraphs_default_doc_id():
    chunks = chunk_by_paragraphs("Alpha.\n\nBeta.")
    assert chunks


def test_chunk_sliding_window_helper():
    chunks = chunk_sliding_window(
        "x" * 500, window_size=100, overlap=20, document_id="sw-doc"
    )
    assert len(chunks) >= 2


def test_chunk_sliding_window_default_doc_id():
    chunks = chunk_sliding_window("y" * 300)
    assert chunks


# ===========================================================================
# create_chunker / create_enhanced_semantic_chunker
# ===========================================================================
def test_create_chunker_from_none():
    assert isinstance(create_chunker(), TextChunker)


def test_create_chunker_from_config():
    cfg = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
    ch = create_chunker(cfg)
    assert ch.config is cfg


def test_create_chunker_from_string():
    ch = create_chunker("sentence", chunk_size=64)
    assert ch.config.strategy == ChunkingStrategy.SENTENCE
    assert ch.config.chunk_size == 64


def test_create_chunker_from_enum():
    ch = create_chunker(ChunkingStrategy.SLIDING_WINDOW, chunk_size=128)
    assert ch.config.strategy == ChunkingStrategy.SLIDING_WINDOW


def test_create_enhanced_semantic_chunker_defaults():
    # enable_caching is accepted but no longer forwarded as an invalid kwarg
    # (previously a TypeError on every call); returns a semantic chunker.
    chunker = create_enhanced_semantic_chunker()
    assert isinstance(chunker, TextChunker)
    assert chunker.config.strategy == ChunkingStrategy.SEMANTIC


def test_create_enhanced_semantic_chunker_caching_kw():
    chunker = create_enhanced_semantic_chunker(enable_caching=False)
    assert isinstance(chunker, TextChunker)
    assert chunker.config.strategy == ChunkingStrategy.SEMANTIC


# ===========================================================================
# create_records / create_vector_records
# ===========================================================================
def _two_chunks():
    return [
        TextChunk(text="short text", start_pos=0, end_pos=10, chunk_id="docX_0"),
        TextChunk(
            text="L" * 150,
            start_pos=10,
            end_pos=160,
            chunk_id="docX_1",
            metadata={"source_id": "explicit"},
        ),
    ]


def test_create_records_basic_shape():
    chunks = _two_chunks()
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    recs = create_records(chunks, embeddings)
    assert len(recs) == 2
    r0 = recs[0]
    assert r0["id"] == "docX_0"
    assert r0["vector"] == [0.1, 0.2]
    # TD-REC-1: no top-level source; the text rides in text_fields.
    assert "source" not in r0
    assert r0["text_fields"][0]["content"] == "short text"
    assert r0["props"]["embedding_dimension"] == 2
    # short text -> no ellipsis
    assert r0["props"]["text_preview"] == "short text"
    # long text -> truncated preview
    assert recs[1]["props"]["text_preview"].endswith("...")
    # explicit source_id preserved
    assert recs[1]["props"]["source_id"] == "explicit"


def test_create_records_source_id_from_chunk_id():
    chunks = [TextChunk(text="t", start_pos=0, end_pos=1, chunk_id="abc_5")]
    recs = create_records(chunks, [[1.0]])
    assert recs[0]["props"]["source_id"] == "abc"


def test_create_records_with_source_type_and_metadata():
    chunks = _two_chunks()
    recs = create_records(
        chunks,
        [[0.1, 0.2], [0.3, 0.4]],
        collection_metadata={"tenant": "t1"},
        filterable_fields=["tenant"],
        source_type="documentation",
        source_metadata={"author": "alice"},
    )
    assert recs[0]["props"]["source_type"] == "documentation"
    assert recs[0]["props"]["author"] == "alice"
    assert recs[0]["props"]["tenant"] == "t1"


def test_create_records_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        create_records(_two_chunks(), [[0.1, 0.2]])


def test_create_vector_records_wrapper():
    chunks = _two_chunks()
    vrecs = create_vector_records(chunks, [[0.1, 0.2], [0.3, 0.4]])
    assert len(vrecs) == 2
    assert vrecs[0].id == "docX_0"
    assert vrecs[0].vector == [0.1, 0.2]
    # TD-REC-1: the wrapper keeps the record-level source attribute; the
    # WIRE payload carries the text in text_fields (asserted elsewhere).
    assert vrecs[0].source == "short text"


# ===========================================================================
# chunk_and_embed_records / chunk_and_embed_text
# ===========================================================================
def test_chunk_and_embed_records_simple_embedder(fresh_pool):
    embedder = SimpleEmbedder(dim=3)
    recs = chunk_and_embed_records(
        SAMPLE,
        source_id="doc99",
        embedding_provider=embedder,
        chunking_config=ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
    )
    assert recs
    assert embedder.calls  # embed_texts was invoked
    assert all("vector" in r for r in recs)


def test_chunk_and_embed_records_metadata_embedder_merges_config(fresh_pool):
    embedder = MetadataEmbedder(dim=2)
    pc = {"existing": "value"}
    recs = chunk_and_embed_records(
        "Sentence one. Sentence two.",
        source_id="docM",
        embedding_provider=embedder,
        chunking_config=ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
        processing_config=pc,
    )
    assert recs
    # processing_config got merged with embedding metadata in place
    assert pc["model"] == "fake-embedder"
    assert pc["existing"] == "value"


def test_chunk_and_embed_records_metadata_embedder_no_processing_config(fresh_pool):
    embedder = MetadataEmbedder(dim=2)
    recs = chunk_and_embed_records(
        "Only one sentence here.",
        source_id="docN",
        embedding_provider=embedder,
        chunking_config=ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
    )
    assert recs


def test_chunk_and_embed_records_tolist_embeddings(fresh_pool):
    embedder = TolistEmbedder()
    recs = chunk_and_embed_records(
        "Alpha beta gamma. Delta epsilon.",
        source_id="docT",
        embedding_provider=embedder,
        chunking_config=ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
    )
    assert recs
    assert recs[0]["vector"] == [1.0, 2.0]


def test_chunk_and_embed_text_wrapper(fresh_pool):
    embedder = SimpleEmbedder(dim=3)
    vrecs = chunk_and_embed_text(
        "First sentence. Second sentence.",
        source_id="docW",
        embedding_provider=embedder,
        chunking_config=ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
    )
    assert vrecs
    from proximadb_sdk.models import VectorRecord

    assert all(isinstance(v, VectorRecord) for v in vrecs)


def test_chunk_and_embed_records_default_config(fresh_pool):
    embedder = SimpleEmbedder(dim=2)
    recs = chunk_and_embed_records(
        "x" * 200, source_id="docDefault", embedding_provider=embedder
    )
    assert recs


# ===========================================================================
# prepare_records / prepare_vector_records
# ===========================================================================
def test_prepare_records_basic():
    response = {
        "chunks": [
            {"id": "c1", "text": "Hello", "embedding": [0.1, 0.2]},
            {"id": "c2", "text": "World", "embedding": [0.3, 0.4]},
        ],
        "model": "all-mpnet-base-v2",
        "chunking_strategy": "sentence",
        "chunk_size": 512,
        "overlap": 50,
        "dimension": 2,
    }
    recs = prepare_records(response, source_id="doc1", source_type="test")
    assert len(recs) == 2
    p0 = recs[0]["props"]
    assert p0["text"] == "Hello"
    assert p0["chunk_index"] == 0
    assert p0["source_type"] == "test"
    assert p0["source_id"] == "doc1"
    assert p0["embedding_model"] == "all-mpnet-base-v2"
    assert p0["chunk_strategy"] == "sentence"
    assert p0["chunk_size"] == 512
    assert p0["chunk_overlap"] == 50
    assert p0["embedding_dimension"] == 2
    assert "created_at" in p0 and "indexed_at" in p0


def test_prepare_records_missing_chunks_raises():
    with pytest.raises(ValueError, match="No chunks"):
        prepare_records({"chunks": []}, source_id="d")
    with pytest.raises(ValueError, match="No chunks"):
        prepare_records({}, source_id="d")


def test_prepare_records_missing_embedding_raises():
    response = {"chunks": [{"id": "c1", "text": "x"}]}
    with pytest.raises(ValueError, match="missing embedding"):
        prepare_records(response, source_id="d")


def test_prepare_records_default_chunk_id_and_text():
    response = {"chunks": [{"embedding": [0.1]}]}
    recs = prepare_records(response, source_id="d")
    assert recs[0]["id"] == "chunk_0"
    assert recs[0]["props"]["text"] == ""


def test_prepare_records_source_metadata_filterable_vs_prefixed():
    response = {"chunks": [{"id": "c1", "text": "p", "embedding": [0.1]}]}
    recs = prepare_records(
        response,
        source_id="PROD-1",
        source_metadata={"category": "Electronics", "internal_note": "secret"},
        filterable_fields=["category"],
    )
    props = recs[0]["props"]
    assert props["category"] == "Electronics"  # filterable -> top level
    assert props["source_internal_note"] == "secret"  # not filterable -> prefixed


def test_prepare_records_custom_metadata_fn():
    response = {"chunks": [{"id": "c1", "text": "abc123", "embedding": [0.1]}]}

    def enrich(chunk, idx):
        return {
            "section": f"part_{idx}",
            "has_numbers": any(ch.isdigit() for ch in chunk["text"]),
        }

    recs = prepare_records(
        response,
        source_id="doc",
        chunk_metadata_fn=enrich,
        filterable_fields=["section"],
    )
    props = recs[0]["props"]
    assert props["section"] == "part_0"  # filterable
    assert props["custom_has_numbers"] is True  # prefixed


def test_prepare_records_custom_metadata_fn_exception_is_logged(caplog):
    response = {"chunks": [{"id": "c1", "text": "x", "embedding": [0.1]}]}

    def boom(chunk, idx):
        raise RuntimeError("nope")

    # should not raise — exception is caught & logged
    recs = prepare_records(response, source_id="doc", chunk_metadata_fn=boom)
    assert len(recs) == 1


def test_prepare_records_preserve_embedding_metadata():
    response = {
        "chunks": [
            {
                "id": "c1",
                "text": "x",
                "embedding": [0.1],
                "confidence": 0.9,
                "language": "en",
            }
        ]
    }
    recs = prepare_records(
        response,
        source_id="doc",
        preserve_embedding_metadata=True,
        filterable_fields=["confidence"],
    )
    props = recs[0]["props"]
    assert props["confidence"] == 0.9  # filterable
    assert props["chunk_language"] == "en"  # prefixed


def test_prepare_vector_records_wrapper():
    response = {"chunks": [{"id": "c1", "text": "hi", "embedding": [0.1, 0.2]}]}
    vrecs = prepare_vector_records(response, source_id="doc")
    assert len(vrecs) == 1
    assert vrecs[0].id == "c1"
    assert vrecs[0].vector == [0.1, 0.2]


# ===========================================================================
# ChunkerFactory
# ===========================================================================
def test_chunker_factory_lifecycle():
    cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
    factory = ChunkerFactory(cfg)
    chunker = factory.create()
    assert isinstance(chunker, TextChunker)
    assert factory.validate(chunker) is True
    # reset / dispose / destroy are no-ops but must be callable
    factory.reset(chunker)
    factory.dispose(chunker)
    factory.destroy(chunker)
    # invalid when strategy missing
    chunker._strategy = None
    assert factory.validate(chunker) is False


# ===========================================================================
# ChunkerPool (uses FakeResourcePool via autouse patch)
# ===========================================================================
def test_chunker_pool_get_and_return():
    pool = ChunkerPool(max_pool_size=3)
    cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
    chunker = pool.get_chunker(cfg)
    assert isinstance(chunker, TextChunker)
    pool.return_chunker(chunker, cfg)
    # second call for same config reuses the same underlying pool object
    pool.get_chunker(cfg)
    assert len(pool._pools) == 1


def test_chunker_pool_key_distinct_per_config():
    pool = ChunkerPool()
    cfg1 = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE, chunk_size=100)
    cfg2 = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH, chunk_size=200)
    pool.get_chunker(cfg1)
    pool.get_chunker(cfg2)
    assert len(pool._pools) == 2


def test_chunker_pool_stats_empty():
    pool = ChunkerPool()
    stats = pool.get_stats()
    assert stats["active_pools"] == 0
    assert stats["total_requests"] == 0
    assert stats["hit_rate_percent"] == 0


def test_chunker_pool_stats_with_activity():
    pool = ChunkerPool()
    cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
    c = pool.get_chunker(cfg)
    pool.return_chunker(c, cfg)
    pool.get_chunker(cfg)
    stats = pool.get_stats()
    assert stats["active_pools"] == 1
    assert stats["total_requests"] >= 1
    assert "pool_stats" in stats
    key = next(iter(stats["pool_stats"]))
    assert stats["pool_stats"][key]["health"] == "healthy"


def test_chunker_pool_cleanup_unused_pools_noop():
    pool = ChunkerPool()
    # documented no-op
    assert pool.cleanup_unused_pools() is None
    assert pool.cleanup_unused_pools(max_idle_time=1.0) is None


def test_chunker_pool_singleton():
    a = ChunkerPool.get_instance()
    b = ChunkerPool.get_instance()
    assert a is b


# ===========================================================================
# PooledChunkerContext
# ===========================================================================
def test_pooled_chunker_context_uses_given_pool():
    pool = ChunkerPool(max_pool_size=2)
    cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
    with PooledChunkerContext(cfg, pool=pool) as chunker:
        assert isinstance(chunker, TextChunker)
        chunks = chunker.chunk_text("One sentence. Two sentence.", "ctx")
        assert chunks
    assert len(pool._pools) == 1


def test_pooled_chunker_context_default_global_pool(fresh_pool):
    cfg = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
    with PooledChunkerContext(cfg) as chunker:
        assert isinstance(chunker, TextChunker)
    assert len(fresh_pool._pools) == 1


def test_pooled_chunker_context_exit_without_chunker():
    cfg = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
    ctx = PooledChunkerContext(cfg)
    # __exit__ with chunker still None must be safe
    ctx.chunker = None
    assert ctx.__exit__(None, None, None) is None


# ===========================================================================
# Module-level pool helpers
# ===========================================================================
def test_get_chunker_pool_stats_global(fresh_pool):
    stats = get_chunker_pool_stats()
    assert "active_pools" in stats
    assert "total_requests" in stats


def test_cleanup_chunker_pool_global(fresh_pool):
    # no-op, just must not raise
    assert cleanup_chunker_pool() is None
