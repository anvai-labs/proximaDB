"""Offline unit tests for genuine incremental streaming in the chunking pipeline.

Covers the ``supports_streaming`` capability flag and the ``chunk_stream``
contract added in the chunking true-streaming work (TD-126):

* Equivalence: for the four streamable strategies (fixed-size, sliding-window,
  sentence, paragraph), streaming a piece-split source yields the SAME chunks
  (text / offsets / id, including sliding-window overlap) as the batch
  ``chunk()`` of the joined input.
* Bounded memory: a streamable strategy fed a generator yields its first chunk
  BEFORE the generator is fully consumed (true streaming, not materialize-all).
* Honest fallback: non-streamable strategies (recursive / structural-semantic /
  semantic-embedding / code) report ``supports_streaming is False`` and their
  ``chunk_stream`` materializes the input but still yields chunks incrementally
  and equivalently to batch.
* Backward compatibility: ``process_stream(str)`` keeps working for every
  strategy.

Fully offline: pure CPU text processing, no network, no embedding model.
"""

import asyncio

import pytest

from proximadb_sdk.chunking_strategies.base import ChunkingConfig, ChunkingStrategy
from proximadb_sdk.chunking_strategies.factory import ChunkingStrategyFactory
from proximadb_sdk.chunking_strategies.pipeline import ChunkingPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TEXTS = {
    "lorem": "The quick brown fox jumps over the lazy dog. " * 20,
    "sentences": (
        "Hello world. This is a test. Dr. Smith went home. "
        "Another sentence here! And a question? Final one."
    ),
    "paras": (
        "First paragraph here with content.\n\n"
        "Second paragraph also here.\n\n"
        "Third one is a bit longer than the others to vary.\n\n"
        "- item a\n- item b\n- item c"
    ),
    "short": "abc",
    "unicode": "これはテストです。次の文も。English mixes in. Done.",
    "abbrev": "I met Mr. Smith today. He works at Inc. Corp. nearby. Cool.",
}

# (strategy, config-kwargs) for the four streamable strategies, several configs.
STREAMABLE = {
    "fixed_size": (
        ChunkingStrategy.FIXED_SIZE,
        dict(chunk_size=40, chunk_overlap=0, min_chunk_size=5),
    ),
    "sliding_a": (
        ChunkingStrategy.SLIDING_WINDOW,
        dict(chunk_size=40, chunk_overlap=10, min_chunk_size=5),
    ),
    "sliding_b": (
        ChunkingStrategy.SLIDING_WINDOW,
        dict(chunk_size=30, chunk_overlap=7, min_chunk_size=1),
    ),
    "sliding_high_overlap": (
        ChunkingStrategy.SLIDING_WINDOW,
        dict(chunk_size=12, chunk_overlap=11, min_chunk_size=1),
    ),
    "sentence_a": (
        ChunkingStrategy.SENTENCE,
        dict(chunk_size=50, chunk_overlap=0, min_chunk_size=5),
    ),
    "sentence_b": (
        ChunkingStrategy.SENTENCE,
        dict(chunk_size=30, chunk_overlap=0, min_chunk_size=1),
    ),
    "paragraph_a": (
        ChunkingStrategy.PARAGRAPH,
        dict(chunk_size=80, chunk_overlap=0, min_chunk_size=5, max_chunk_size=120),
    ),
    "paragraph_b": (
        ChunkingStrategy.PARAGRAPH,
        dict(chunk_size=40, chunk_overlap=0, min_chunk_size=1, max_chunk_size=50),
    ),
}

# Awkward piece splittings (1-char, mid-word, single huge piece, fibonacci, ...).
PIECE_PLANS = [
    [7],
    [1, 1, 1, 1, 1],
    [3, 13, 2, 40],
    [100],
    [13, 9, 21],
    [2],
    [1, 2, 3, 5, 8, 13],
]


def _make(strategy_key):
    strat, cfg = STREAMABLE[strategy_key]
    config = ChunkingConfig(strategy=strat, **cfg)
    return ChunkingStrategyFactory.create_strategy(strat, config)


def _split_pieces(text, sizes):
    out = []
    i = 0
    for s in sizes:
        out.append(text[i : i + s])
        i += s
    if i < len(text):
        out.append(text[i:])
    return out


def _key(chunks):
    """The streaming-stable identity of a chunk (excludes total_chunks, which is
    inherently global and cannot be known without consuming the whole input)."""
    return [(c.text, c.start_pos, c.end_pos, c.chunk_id) for c in chunks]


# ---------------------------------------------------------------------------
# Capability flag
# ---------------------------------------------------------------------------


def test_streamable_strategies_advertise_support():
    for key in ("fixed_size", "sliding_a", "sentence_a", "paragraph_a"):
        assert _make(key).supports_streaming is True, key


def test_nonstreamable_strategies_do_not_advertise_support():
    from proximadb_sdk.chunking_strategies.code import CodeChunkingStrategy
    from proximadb_sdk.chunking_strategies.recursive import RecursiveStrategy
    from proximadb_sdk.chunking_strategies.semantic import SemanticStrategy
    from proximadb_sdk.chunking_strategies.semantic_embedding import (
        SemanticEmbeddingStrategy,
    )

    assert RecursiveStrategy.supports_streaming is False
    assert SemanticStrategy.supports_streaming is False
    assert SemanticEmbeddingStrategy.supports_streaming is False
    assert CodeChunkingStrategy.supports_streaming is False


# ---------------------------------------------------------------------------
# Equivalence: chunk_stream(pieces) == chunk("".join(pieces))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_key", list(STREAMABLE))
@pytest.mark.parametrize("text_name", list(TEXTS))
def test_str_source_matches_batch(strategy_key, text_name):
    s = _make(strategy_key)
    text = TEXTS[text_name]
    assert _key(list(s.chunk_stream(text, "doc"))) == _key(s.chunk(text, "doc"))


@pytest.mark.parametrize("strategy_key", list(STREAMABLE))
@pytest.mark.parametrize("text_name", list(TEXTS))
@pytest.mark.parametrize("plan", PIECE_PLANS)
def test_piece_source_matches_batch(strategy_key, text_name, plan):
    s = _make(strategy_key)
    text = TEXTS[text_name]
    pieces = _split_pieces(text, plan)
    streamed = list(s.chunk_stream(iter(pieces), "doc"))
    assert _key(streamed) == _key(s.chunk(text, "doc"))


def test_sliding_window_overlap_preserved_across_piece_boundary():
    """The tricky bit: overlap text shared by consecutive windows must survive
    being split across input pieces."""
    s = _make("sliding_a")
    text = TEXTS["lorem"]
    batch = s.chunk(text, "doc")
    # Pieces that deliberately split inside the overlap region.
    pieces = _split_pieces(text, [35, 4, 6, 1, 1, 1])
    streamed = list(s.chunk_stream(iter(pieces), "doc"))
    assert _key(streamed) == _key(batch)
    # Sanity: there really is overlap to preserve.
    assert any(c.metadata.get("has_overlap") for c in batch)


def test_total_chunks_sentinel_in_stream():
    """total_chunks is the one inherently-global field; streaming leaves it -1."""
    s = _make("fixed_size")
    streamed = list(s.chunk_stream(TEXTS["lorem"], "doc"))
    assert streamed
    assert all(c.metadata["total_chunks"] == -1 for c in streamed)
    # ...while batch back-fills the real count.
    batch = s.chunk(TEXTS["lorem"], "doc")
    assert all(c.metadata["total_chunks"] == len(batch) for c in batch)


# ---------------------------------------------------------------------------
# Bounded memory / genuine streaming
# ---------------------------------------------------------------------------


def test_first_chunk_emitted_before_generator_exhausted():
    s = _make("sliding_a")
    state = {"advanced": 0}
    n_pieces = 5000

    def gen():
        for i in range(n_pieces):
            state["advanced"] = i + 1
            yield "The quick brown fox jumps over the lazy dog. "

    it = s.chunk_stream(gen(), "doc")
    first = next(it)
    assert first.chunk_id == "doc_chunk_0"
    # True streaming: the first chunk is produced long before the whole
    # generator is drained.
    assert state["advanced"] < n_pieces
    # In fact only a couple of pieces are needed to fill the first window.
    assert state["advanced"] <= 3


@pytest.mark.parametrize(
    "strategy_key", ["fixed_size", "sliding_a", "sentence_a", "paragraph_a"]
)
def test_each_streamable_strategy_yields_before_exhaustion(strategy_key):
    s = _make(strategy_key)
    state = {"advanced": 0}
    n_pieces = 2000
    # A piece that contains a complete sentence + paragraph break so every
    # strategy can find a boundary early.
    piece = "This is a complete sentence here.\n\n"

    def gen():
        for i in range(n_pieces):
            state["advanced"] = i + 1
            yield piece

    first = next(s.chunk_stream(gen(), "doc"))
    assert first is not None
    assert state["advanced"] < n_pieces, strategy_key


# ---------------------------------------------------------------------------
# Honest fallback for non-streamable strategies
# ---------------------------------------------------------------------------

_FALLBACK_TEXT = (
    "First paragraph here.\n\n"
    "Second paragraph here with more.\n\n"
    "Third block of content here too."
)


@pytest.mark.parametrize(
    "strategy",
    [ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC],
)
def test_nonstreamable_fallback_matches_batch(strategy):
    config = ChunkingConfig(
        strategy=strategy, chunk_size=50, chunk_overlap=5, min_chunk_size=5
    )
    s = ChunkingStrategyFactory.create_strategy(strategy, config)
    batch = s.chunk(_FALLBACK_TEXT, "doc")
    # Default chunk_stream materializes then yields from chunk(): same result.
    via_str = list(s.chunk_stream(_FALLBACK_TEXT, "doc"))
    via_iter = list(
        s.chunk_stream(
            iter([_FALLBACK_TEXT[:20], _FALLBACK_TEXT[20:50], _FALLBACK_TEXT[50:]]),
            "doc",
        )
    )
    assert [c.text for c in via_str] == [c.text for c in batch]
    assert [c.text for c in via_iter] == [c.text for c in batch]


def test_pipeline_fallback_yields_correct_chunks_for_nonstreamable():
    cfg = PipelineConfig(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=5,
        ),
    )
    p = ChunkingPipeline(cfg)
    batch = p.chunker.chunk(_FALLBACK_TEXT, "doc")
    streamed = list(
        p.process_stream(iter([_FALLBACK_TEXT[:25], _FALLBACK_TEXT[25:]]), "doc")
    )
    assert [c.text for c in streamed] == [c.text for c in batch]


# ---------------------------------------------------------------------------
# Pipeline backward compatibility + iterable sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.FIXED_SIZE,
        ChunkingStrategy.SLIDING_WINDOW,
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.PARAGRAPH,
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SEMANTIC,
    ],
)
def test_process_stream_str_backward_compatible(strategy):
    cfg = PipelineConfig(
        chunking_strategy=strategy,
        chunking_config=ChunkingConfig(
            strategy=strategy, chunk_size=40, chunk_overlap=5, min_chunk_size=3
        ),
    )
    p = ChunkingPipeline(cfg)
    chunks = list(p.process_stream(TEXTS["lorem"], "doc"))
    assert all(c.text for c in chunks)


def test_process_stream_iterable_source_streamable():
    cfg = PipelineConfig(
        chunking_strategy=ChunkingStrategy.FIXED_SIZE,
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=30,
            chunk_overlap=0,
            min_chunk_size=3,
        ),
    )
    p = ChunkingPipeline(cfg)
    text = TEXTS["lorem"]
    via_str = [c.text for c in p.process_stream(text, "doc")]
    via_iter = [
        c.text for c in p.process_stream(iter(_split_pieces(text, [7, 11, 3])), "doc")
    ]
    assert via_iter == via_str


def test_process_stream_async_iterable_source():
    cfg = PipelineConfig(
        chunking_strategy=ChunkingStrategy.SENTENCE,
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=40,
            chunk_overlap=0,
            min_chunk_size=3,
        ),
    )
    p = ChunkingPipeline(cfg)
    text = TEXTS["sentences"]

    async def collect_iter():
        return [
            c.text
            async for c in p.process_stream_async(
                iter(_split_pieces(text, [9, 5, 21])), "doc"
            )
        ]

    async def collect_str():
        return [c.text async for c in p.process_stream_async(text, "doc")]

    assert asyncio.run(collect_iter()) == asyncio.run(collect_str())
