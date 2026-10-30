"""Chunking conformance sweep — the TD-CHUNK-1 baseline, landed RED-as-recorded.

This is the specification ADR-091 says the ecosystem never had, made enforceable.
It deliberately does **not** fix anything: TD-CHUNK-1's acceptance is the suite
plus the recorded baseline, and TD-CHUNK-2 onward turns the recorded violations
green.

Why a recorded baseline instead of ``xfail``
--------------------------------------------
The baseline is a **bidirectional ratchet**. Each (strategy, corpus) case asserts
its violation set *exactly*:

* a new violation fails -> regressions are caught;
* a **fixed** violation also fails, until the baseline is updated -> nobody
  repairs a defect without recording that they did, and nobody can quietly let a
  fix rot back.

``xfail(strict=True)`` gives the second property but scatters the state across 60
markers. A single reviewable mapping is the state of truth, and it is generated
from a real sweep (``format_baseline``) rather than hand-asserted — so it records
what the code *does*, not what anyone believes it does. That distinction is the
whole lesson of the audit that produced ADR-091: the pre-existing suite had 426
passing tests and caught none of these defects.

Baseline after the correctness slice: **70 cases, 70 clean, 0 violations,
0 errors** — seven strategies x ten corpus entries — every swept strategy now satisfies every invariant, so BASELINE is
empty and ``VIOLATION_CEILING = 0`` makes the aggregate ratchet an ABSOLUTE
assertion rather than a shrinking allowance. A single new violation anywhere
fails the build.

(Initial recording against develop @482075bc1 was 22 clean / 67 violations.)
(Initial recording against develop @482075bc1 was 22 clean / 67 violations.)

Note on running locally
-----------------------
The repo's conftests rely on the *installed* package. If your editable install
points at a different worktree you will silently test that one, so run this with
``PYTHONPATH=clients/python/src`` to pin the tree under test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import pytest

from proximadb_sdk.chunking import ChunkerPool, TextChunker
from proximadb_sdk.chunking_strategies import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkingStrategyFactory,
    ChunkingStrategyInterface,
    ResolvedSizing,
    SizingPolicy,
    config_kwargs,
)
from proximadb_sdk.chunking_strategies.base import (
    OFFSET_CONTRACT_EXACT,
    OFFSET_CONTRACT_LEGACY,
    TextChunk,
)
from proximadb_sdk.chunking_strategies.boundaries import (
    Boundary,
    BoundaryKind,
    BoundarySource,
    CompositeBoundarySource,
    HeadingBoundarySource,
    StrategyBoundarySource,
    annotate_heading_paths,
    merge_boundaries,
)
from proximadb_sdk.chunking_strategies.conformance import (
    BASIS_BYTE,
    BASIS_CHAR,
    CHAR_MEASURE,
    ChunkerAdapter,
    Invariant,
    Measure,
    TokenMeasure,
    by_name,
    check_config_safety,
    check_exactness,
    check_non_empty,
    check_totality,
    compute_trace,
    diff_digests,
    evaluate,
    evaluate_all,
    format_baseline,
    load_golden,
    standard_corpus,
    sweep_digests,
)
from proximadb_sdk.chunking_strategies.conformance.evals import (
    STANDARD_CASES,
    run_eval,
    score_case,
)
from proximadb_sdk.chunking_strategies.conformance.invariants import _wall_clock_budget
from proximadb_sdk.chunking_strategies.contracts import (
    ChunkContextRenderer,
    ResolvedInputContract,
    TokenBudget,
)
from proximadb_sdk.chunking_strategies.dedup import deduplicate
from proximadb_sdk.chunking_strategies.sizing import Absolute, Fraction
from proximadb_sdk.chunking_strategies.spans import TextSlicer
from proximadb_sdk.chunking_strategies.structure import (
    MARKDOWN_HEADING,
    HeadingOutline,
    protected_spans,
)
from proximadb_sdk.chunking_strategies.token_budget import TokenBudgetStrategy

BUDGET = 2048
DEFAULTS = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "min_chunk_size": 100,
    "max_chunk_size": BUDGET,
}

SWEPT_STRATEGIES = (
    ChunkingStrategy.SLIDING_WINDOW,
    ChunkingStrategy.FIXED_SIZE,
    ChunkingStrategy.SENTENCE,
    ChunkingStrategy.PARAGRAPH,
    ChunkingStrategy.SEMANTIC,
    ChunkingStrategy.RECURSIVE,
    ChunkingStrategy.SEMANTIC_EMBEDDING,
)
# CODE is excluded deliberately, not by oversight: it publishes UTF-8 BYTE offsets
# through the same field the text strategies fill with CHARACTER offsets, so it
# cannot be held to this suite's exactness check until TD-CG2 settles the code
# path's offset basis. It is the last unswept strategy.


def _deterministic_provider(texts: list[str]) -> list[list[float]]:
    """A pure, seed-free stand-in for an embedding model.

    SEMANTIC_EMBEDDING needs a provider to reach its real code path. Hashing the
    text keeps the corpus determinism rule intact (same input -> same vectors ->
    same breakpoints) without downloading a model or touching the network, and
    ``md5`` is used only because it is stable across processes, unlike ``hash()``.
    """
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(text[:64].encode("utf-8")).digest()
        vectors.append([byte / 255.0 for byte in digest[:8]])
    return vectors


def _v(*names: str) -> frozenset[Invariant]:
    return frozenset(Invariant(n) for n in names)


#: Recorded state of today's code. Generated, not hand-written. Now EMPTY: no
#: swept strategy violates any invariant. Keep the mapping (rather than deleting
#: it) so a regression is recorded here as data when it appears.
#: Strategies absent from this mapping are clean on all ten entries:
#: ``sliding_window`` (always was — the only one whose offsets survived the
#: audit) and now every other swept strategy. ``recursive`` needed no edit of its
#: own: it composes paragraph's spans, so it became correct the moment its
#: parent's text was a verbatim slice.
BASELINE: dict[tuple[str, str], frozenset[Invariant]] = {}

#: Aggregate ratchet. Clean cases may only increase; violations may only fall.
#: At 70/0 both are pinned: the sweep is exhaustive over every swept strategy.
CLEAN_FLOOR = 70
VIOLATION_CEILING = 0


def _adapter(strategy: ChunkingStrategy) -> ChunkerAdapter:
    config = ChunkingConfig(
        strategy=strategy, embedding_provider=_deterministic_provider, **DEFAULTS
    )
    strat = ChunkingStrategyFactory.create_strategy(strategy, config)

    stream = None
    if getattr(strat, "supports_streaming", False):

        def stream(pieces):  # type: ignore[misc]
            return list(strat.chunk_stream(pieces, source_id="doc"))

    def hostile():
        """A legal-but-hostile config: overlap close to size, tiny minimum."""
        hostile_config = ChunkingConfig(
            strategy=strategy,
            chunk_size=100,
            chunk_overlap=90,
            min_chunk_size=10,
            max_chunk_size=200,
            embedding_provider=_deterministic_provider,
        )
        hostile_strategy = ChunkingStrategyFactory.create_strategy(
            strategy, hostile_config
        )
        return hostile_strategy.chunk("word " * 4000, "doc")

    # IDEMPOTENCE does not apply to a distribution-relative criterion.
    # SEMANTIC_EMBEDDING breaks where a gap exceeds the Nth PERCENTILE of the
    # document's own distance distribution, so re-chunking one chunk recomputes
    # that percentile over a smaller distribution and can legitimately find a
    # boundary that was not in the whole document's top 5%. Demanding a fixed
    # point would be demanding the strategy stop being percentile-based. It is
    # SKIPPED rather than baselined, so the result is recorded as unmeasured
    # instead of as a passing check.
    absolute_criterion = strategy is not ChunkingStrategy.SEMANTIC_EMBEDDING

    return ChunkerAdapter(
        name=strategy.value,
        chunk=lambda text: strat.chunk(text, "doc"),
        budget=BUDGET,
        chunk_stream=stream,
        rechunk=(lambda text: strat.chunk(text, "doc")) if absolute_criterion else None,
        hostile=hostile,
    )


CASES = [
    pytest.param(strategy, entry, id=f"{strategy.value}-{entry.name}")
    for strategy in SWEPT_STRATEGIES
    for entry in standard_corpus()
]


@pytest.mark.parametrize("strategy,entry", CASES)
def test_conformance_matches_recorded_baseline(strategy, entry):
    """Each case's violation set must equal the recorded baseline, exactly."""
    result = evaluate(_adapter(strategy), entry)
    assert (
        result.error is None
    ), f"{result.chunker} raised on {entry.name}: {result.error}"

    expected = BASELINE.get((strategy.value, entry.name), frozenset())
    actual = result.violated

    if actual != expected:
        fixed = sorted(i.value for i in expected - actual)
        broke = sorted(i.value for i in actual - expected)
        raise AssertionError(
            "conformance baseline drift for "
            f"{strategy.value} x {entry.name}\n"
            f"  newly VIOLATED (regression): {broke or 'none'}\n"
            f"  no longer violated (fixed):  {fixed or 'none'}\n"
            f"  -> if this is a fix, update BASELINE and lower "
            f"VIOLATION_CEILING / raise CLEAN_FLOOR in the same commit.\n\n"
            f"{result.render()}\n\n"
            f"corpus entry earned its place by catching: {entry.caught}"
        )


def test_aggregate_ratchet_does_not_regress():
    """The single number a reviewer can watch: violations may only fall."""
    results = [
        evaluate(_adapter(strategy), entry)
        for strategy in SWEPT_STRATEGIES
        for entry in standard_corpus()
    ]
    clean = sum(1 for r in results if not r.violated and r.error is None)
    violations = sum(len(r.violations) for r in results)
    errors = [r.render() for r in results if r.error]

    assert not errors, "chunking raised on: " + "; ".join(errors)
    assert (
        clean >= CLEAN_FLOOR
    ), f"clean cases fell to {clean}, floor is {CLEAN_FLOOR} — a regression"
    assert (
        violations <= VIOLATION_CEILING
    ), f"violations rose to {violations}, ceiling is {VIOLATION_CEILING}"


def test_sliding_window_is_clean_across_the_whole_corpus():
    """The one strategy that already satisfies the specification.

    Pinned separately so that "at least one implementation is correct" cannot
    silently stop being true — it is the reference the others are migrating to.
    """
    adapter = _adapter(ChunkingStrategy.SLIDING_WINDOW)
    failures = []
    for entry in standard_corpus():
        result = evaluate(adapter, entry)
        if result.violated or result.error:
            failures.append(result.render())
    assert not failures, "sliding_window regressed:\n" + "\n".join(failures)


class TestTrace:
    """The trace is new production code; it needs its own teeth."""

    def test_detects_total_coverage(self):
        source = "alpha beta gamma"

        class C:
            text = source
            start_pos = 0
            end_pos = len(source)
            metadata = {"chunking_strategy": "unit"}

        trace = compute_trace(source, [C()])
        assert trace.is_total
        assert trace.units_covered == trace.units_in
        assert trace.spans_dropped == 0
        assert trace.boundary_sources == {"unit": 1}

    def test_detects_a_dropped_span(self):
        source = "keep this DROPPED keep that"

        class C:
            text = "keep this"
            start_pos = 0
            end_pos = 9
            metadata: dict = {}

        trace = compute_trace(source, [C()])
        assert not trace.is_total
        assert trace.spans_dropped == 1
        assert trace.units_covered < trace.units_in
        assert "DROPPED" in trace.gaps[0].preview

    def test_counts_duplication_from_overlap(self):
        source = "abcdefghij"

        def chunk(start, end):
            obj = type("C", (), {})()
            obj.text = source[start:end]
            obj.start_pos = start
            obj.end_pos = end
            obj.metadata = {}
            return obj

        trace = compute_trace(source, [chunk(0, 6), chunk(4, 10)])
        assert trace.is_total
        assert trace.units_duplicated == 2
        assert trace.duplication_ratio == pytest.approx(0.2)

    def test_empty_input_is_total(self):
        trace = compute_trace("", [])
        assert trace.is_total
        assert trace.coverage_ratio == 1.0


class TestMeasureEquivalence:
    """Injecting the character measure explicitly must change nothing.

    A LIVE oracle, not a snapshot: both constructions exist at once, so this
    stays true forever rather than going stale. It is the property that makes
    the whole decoupling safe to build on — if the default path and an
    explicitly-injected CharMeasure can disagree, then the measure seam is not
    actually neutral and every later phase is built on sand.
    """

    @staticmethod
    def _chunks(strategy, entry, **config_extra):
        config = ChunkingConfig(
            strategy=strategy,
            embedding_provider=_deterministic_provider,
            **DEFAULTS,
            **config_extra,
        )
        strat = ChunkingStrategyFactory.create_strategy(strategy, config)
        return [
            (c.text, c.start_pos, c.end_pos) for c in strat.chunk(entry.text, "doc")
        ]

    @pytest.mark.parametrize("strategy,entry", CASES)
    def test_explicit_char_measure_matches_the_default(self, strategy, entry):
        default = self._chunks(strategy, entry)
        explicit = self._chunks(strategy, entry, measure=CHAR_MEASURE)
        assert explicit == default, (
            f"{strategy.value} x {entry.name}: injecting CharMeasure changed output, "
            "so the measure seam is not neutral"
        )

    def test_char_measure_satisfies_the_protocol(self):
        assert isinstance(CHAR_MEASURE, Measure)
        assert CHAR_MEASURE.is_additive is True
        # No grid: a character measure permits a cut at any position. This is
        # distinct from "cannot decompose", which must raise instead.
        assert CHAR_MEASURE.unit_spans("abc") is None

    def test_char_measure_is_exact_arithmetic(self):
        text = "hello world"
        assert CHAR_MEASURE.size(text, 2, 7) == 5
        assert CHAR_MEASURE.advance(text, 2, 5) == 7

        # It must never materialise the source — that is what keeps the default
        # path allocation-free. A slicer that explodes proves it is not called.
        def exploding_slicer(_a, _b):
            raise AssertionError("CharMeasure must not materialise the source")

        assert CHAR_MEASURE.size(exploding_slicer, 2, 7) == 5
        assert CHAR_MEASURE.advance(exploding_slicer, 2, 5) == 7


class WordTokenCounter:
    """Fast-tokenizer-shaped double: words, plus two special tokens on render.

    Mirrors ``tests/unit/test_token_budget_chunking.py``'s counter deliberately.
    The ``+ 2`` is the whole point: ``count(text) != len(content_offsets(text))``
    is the real, by-design shape of every rendered tokenizer, and a measure
    design that quietly assumes they are equal is wrong on real models.
    """

    name = "words"
    fingerprint = "fingerprint:words"
    advertised_limit = 512

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text)) + 2

    def content_offsets(self, text: str):
        return tuple(m.span() for m in re.finditer(r"\S+", text))


class BlindCounter:
    """Counts, but cannot say where its units are (``content_offsets -> None``).

    A real case: some tokenizer APIs expose a count and no offset mapping.
    """

    name = "blind"
    fingerprint = "fingerprint:blind"
    advertised_limit = 512

    def count(self, text: str) -> int:
        return len(text.split())

    def content_offsets(self, text: str):
        return None


class NonMonotoneCounter:
    """Returns offsets out of order — the silent-corruption case."""

    name = "non-monotone"
    fingerprint = "fingerprint:non-monotone"
    advertised_limit = 512

    def count(self, text: str) -> int:
        return 3

    def content_offsets(self, text: str):
        return ((0, 5), (12, 18), (6, 11))


class OverCountingMeasure:
    """A deliberately NON-additive measure: every span costs 3 extra units.

    Stands in for rendered-token overhead that no span owns (role prefix,
    ``[CLS]``/``[SEP]``). ``advance`` reports the additive answer, so it
    systematically overshoots the real budget — exactly the trap ``_fit_end``
    exists to catch.
    """

    name = "overcounting"
    is_additive = False
    needs_document = False

    def size(self, source, start, end):
        if end <= start:
            return 0
        return (end - start) + 3

    def advance(self, source, start, units):
        return start + units

    def unit_spans(self, text):
        return None


class TestTokenMeasure:
    """The token measure's own contract, and how it fails."""

    def test_satisfies_the_protocol(self):
        measure = TokenMeasure(WordTokenCounter())
        assert isinstance(measure, Measure)
        assert measure.name == "token:words"
        assert measure.needs_document is True

    def test_counts_only_wholly_contained_tokens(self):
        text = "alpha beta gamma delta"
        measure = TokenMeasure(WordTokenCounter())
        assert measure.size(text, 0, len(text)) == 4
        # "alpha" spans [0,5) and "beta" [6,10); a cut at 8 splits "beta", which
        # is then inside neither half. Crediting it to both would let overlap
        # inflate the billable chunk count.
        assert measure.size(text, 0, 8) == 1
        assert measure.size(text, 8, len(text)) == 2

    def test_advance_and_size_agree(self):
        text = "alpha beta gamma delta epsilon zeta"
        measure = TokenMeasure(WordTokenCounter())
        for units in (1, 2, 3):
            end = measure.advance(text, 0, units)
            assert measure.size(text, 0, end) == units

    def test_advance_runs_to_the_end_when_fewer_units_remain(self):
        # Otherwise the tail (trailing punctuation, whitespace) is dropped from
        # coverage and TOTALITY fails on every document.
        text = "alpha beta.  "
        measure = TokenMeasure(WordTokenCounter())
        assert measure.advance(text, 0, 99) == len(text)

    def test_grid_is_the_tokenizer_grid(self):
        measure = TokenMeasure(WordTokenCounter())
        assert measure.unit_spans("ab cd") == ((0, 2), (3, 5))

    def test_a_counter_without_offsets_fails_loudly(self):
        # It can count but cannot say where its units begin, so it cannot cut
        # text. Degrading to "no grid" would silently mis-cut instead.
        measure = TokenMeasure(BlindCounter())
        with pytest.raises(ValueError, match="cannot provide source offsets"):
            measure.size("alpha beta", 0, 10)

    def test_non_monotone_offsets_fail_loudly(self):
        # bisect does not raise on unordered input — it returns a wrong index,
        # so without this check the chunks come out mis-cut with no error.
        measure = TokenMeasure(NonMonotoneCounter())
        with pytest.raises(ValueError, match="non-monotone"):
            measure.size("alpha  gamma  beta", 0, 18)

    def test_a_windowed_source_fails_loudly(self):
        measure = TokenMeasure(WordTokenCounter())
        with pytest.raises(TypeError, match="needs the whole document"):
            measure.size(lambda a, b: "fragment"[a:b], 0, 8)

    def test_a_text_slicer_yields_its_document(self):
        # The batch grouping loops pass a slicer, so a token measure is only
        # usable there because TextSlicer carries the document with it.
        text = "alpha beta gamma"
        measure = TokenMeasure(WordTokenCounter())
        assert measure.size(TextSlicer(text), 0, len(text)) == 3

    def test_cache_does_not_leak_between_documents(self):
        measure = TokenMeasure(WordTokenCounter())
        first, second = "a b c d e", "x y"
        assert measure.size(first, 0, len(first)) == 5
        assert measure.size(second, 0, len(second)) == 2
        assert measure.size(first, 0, len(first)) == 5

    def test_tokenizes_each_document_once(self):
        calls: list[int] = []

        class CountingCounter(WordTokenCounter):
            def content_offsets(self, text: str):
                calls.append(len(text))
                return super().content_offsets(text)

        text = "alpha beta gamma delta"
        measure = TokenMeasure(CountingCounter())
        for _ in range(20):
            measure.size(text, 0, len(text))
        assert calls == [len(text)], "the per-document grid must be cached"


#: Module scope, not class scope: a comprehension inside a class body cannot
#: see names bound in that body (only the outermost iterable is evaluated there).
TOKEN_MEASURED_STRATEGIES = [
    ChunkingStrategy.FIXED_SIZE,
    ChunkingStrategy.SLIDING_WINDOW,
    ChunkingStrategy.SENTENCE,
    ChunkingStrategy.PARAGRAPH,
]
#: "cjk_emoji" is the pathological case for a whitespace tokenizer and is here
#: on purpose: CJK has no spaces, so the whole document counts as a handful of
#: units and every window asks for more than remain. That is the tail path.
TOKEN_MEASURED_ENTRIES = ["prose", "header_dense_markdown", "cjk_emoji"]


class TestChunkingUnderATokenMeasure:
    """The invariants must hold when the measure is not characters.

    A representative subset rather than the full cross-product: the sweep is
    already the slow part of this file, and the point here is that the seam
    works in a second measure, not to re-derive per-corpus behaviour.
    """

    @staticmethod
    def _adapter(strategy, budget=40):
        counter = WordTokenCounter()
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=budget,
            chunk_overlap=5,
            min_chunk_size=1,
            max_chunk_size=budget * 2,
            measure=TokenMeasure(counter),
        )
        strat = ChunkingStrategyFactory.create_strategy(strategy, config)
        return ChunkerAdapter(
            name=f"{strategy.value}+token",
            chunk=lambda text: strat.chunk(text, "doc"),
            budget=budget * 2,
            rechunk=lambda text: strat.chunk(text, "doc"),
            token_counter=lambda text: len(re.findall(r"\S+", text)),
            measure=config.measure,
        )

    @pytest.mark.parametrize(
        "strategy,entry_name",
        [
            pytest.param(s, e, id=f"{s.value}-{e}")
            for s in TOKEN_MEASURED_STRATEGIES
            for e in TOKEN_MEASURED_ENTRIES
        ],
    )
    def test_invariants_hold_under_a_token_measure(self, strategy, entry_name):
        entry = by_name(entry_name)
        result = evaluate(self._adapter(strategy), entry)
        assert result.error is None, result.render()
        assert not result.violations, result.render()

    def test_the_budget_is_actually_counted_in_tokens(self):
        # The real check that the measure is load-bearing: a token budget of 8
        # must produce chunks of ~8 WORDS, not 8 characters. If the measure were
        # ignored, every chunk would be a fragment of a single word.
        text = " ".join(f"w{i}" for i in range(200))
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=8,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=16,
            measure=TokenMeasure(WordTokenCounter()),
        )
        strat = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        chunks = strat.chunk(text, "doc")
        assert len(chunks) == 25, [c.text for c in chunks[:3]]
        for chunk in chunks:
            words = len(re.findall(r"\S+", chunk.text))
            assert words <= 8, f"{words} words in {chunk.text!r} exceeds the budget"

    def test_streaming_refuses_a_whole_document_measure(self):
        # Streaming holds a bounded buffer, so a token measure would silently
        # measure a FRAGMENT as if it were the document. Refusing is the only
        # honest answer.
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=8,
            min_chunk_size=1,
            measure=TokenMeasure(WordTokenCounter()),
        )
        strat = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        with pytest.raises(ValueError, match="needs the whole document"):
            list(strat.chunk_stream(["alpha beta gamma"], "doc"))


class TestNonAdditiveMeasure:
    """The crux: a measure whose size exceeds the sum of its parts.

    ``advance`` gives the additive answer and is therefore WRONG for such a
    measure — it overshoots by exactly the per-span overhead. Nothing else in
    the pipeline would notice: the chunks look fine, the offsets are exact, and
    every structural invariant passes. Only a size check catches it, which is
    why ``_fit_end`` verifies rather than trusts.
    """

    @staticmethod
    def _strategy(strategy, budget):
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=budget,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=budget * 4,
            measure=OverCountingMeasure(),
        )
        return ChunkingStrategyFactory.create_strategy(strategy, config)

    @pytest.mark.parametrize(
        "strategy",
        [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.SLIDING_WINDOW],
    )
    def test_emitted_chunks_respect_the_declared_budget(self, strategy):
        budget = 20
        measure = OverCountingMeasure()
        text = "abcdefghij " * 60
        chunks = self._strategy(strategy, budget).chunk(text, "doc")
        assert chunks
        for chunk in chunks:
            size = measure.size(text, chunk.start_pos, chunk.end_pos)
            assert size <= budget, (
                f"chunk of {size} units exceeds budget {budget}: _fit_end did "
                "not verify the candidate advance() proposed"
            )

    def test_the_naive_advance_really_would_overflow(self):
        # A positive control. Without this, the assertion above could pass
        # because the measure is harmless rather than because _fit_end works.
        measure = OverCountingMeasure()
        assert measure.size("x" * 100, 0, measure.advance("x" * 100, 0, 20)) == 23

    def test_coverage_is_still_total(self):
        text = "abcdefghij " * 60
        chunks = self._strategy(ChunkingStrategy.FIXED_SIZE, 20).chunk(text, "doc")
        violation = check_totality(text, chunks)
        assert violation is None, violation


class TestConfigPropagation:
    """A configured field must survive every rebuild, and split the pool.

    Two hand-written forwarding lists used to decide which fields survived, and
    both dropped anything added after they were written. That failure is
    silent by construction -- a dropped field is indistinguishable from an unset
    one -- so it is only catchable by asserting the field's *effect* downstream,
    which is what these do.
    """

    @staticmethod
    def _token_config(strategy=ChunkingStrategy.FIXED_SIZE, **extra):
        return ChunkingConfig(
            strategy=strategy,
            chunk_size=8,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=16,
            measure=TokenMeasure(WordTokenCounter()),
            **extra,
        )

    def test_config_kwargs_covers_every_field_but_strategy(self):
        # The whole point of deriving instead of listing: this cannot rot.
        expected = {f.name for f in dataclasses.fields(ChunkingConfig)} - {"strategy"}
        assert set(config_kwargs(ChunkingConfig())) == expected

    def test_text_chunker_honours_a_configured_measure(self):
        # The end-to-end statement of the bug: TextChunker rebuilds its strategy
        # from the config, and the rebuild used to drop unknown fields. If the
        # measure is lost, a budget of 8 silently means 8 CHARACTERS.
        text = " ".join(f"word{i}" for i in range(60))
        chunks = TextChunker(self._token_config()).chunk_text(text, "doc")
        assert chunks
        counts = [len(re.findall(r"\S+", chunk.text)) for chunk in chunks]

        # The upper bound alone is NOT sufficient, and asserting only it is how
        # this test first passed while the measure was being dropped: 8-CHARACTER
        # chunks also hold at most 8 words, vacuously. The load-bearing claim is
        # that the chunks are token-sized, i.e. FEW and FULL -- 60 words at 8
        # words each is ~8 chunks, where character sizing yields ~50.
        assert len(chunks) <= 10, (
            f"{len(chunks)} chunks for 60 words at a budget of 8: these are "
            "character-sized, so the measure was dropped during the rebuild"
        )
        assert max(counts) > 1, f"chunks hold {max(counts)} word(s), not ~8"
        assert max(counts) <= 8, f"{max(counts)} words exceeds the budget of 8"

    def test_pool_key_separates_configs_that_differ_only_by_measure(self):
        # Sharing a pool entry here means the second caller silently receives a
        # chunker built for the first caller's measure.
        pool = ChunkerPool()
        char_only = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=8,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=16,
        )
        assert pool._get_pool_key(char_only) != pool._get_pool_key(self._token_config())

    def test_pool_key_is_stable_for_equivalent_configs(self):
        # The other direction: over-splitting on every construction would make
        # the pool useless, so identity must come from VALUES, not object ids.
        pool = ChunkerPool()
        assert pool._get_pool_key(ChunkingConfig()) == pool._get_pool_key(
            ChunkingConfig()
        )
        assert pool._get_pool_key(self._token_config()) == pool._get_pool_key(
            self._token_config()
        )

    def test_pool_key_separates_every_sizing_field(self):
        pool = ChunkerPool()
        base = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE)
        baseline = pool._get_pool_key(base)
        for field_name, value in (
            ("chunk_size", 999),
            ("chunk_overlap", 7),
            ("min_chunk_size", 3),
            ("max_chunk_size", 4096),
            ("buffer_size", 9),
            ("breakpoint_percentile_threshold", 50.0),
            ("add_context", True),
            ("preserve_sentences", False),
        ):
            other = ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE, **{field_name: value}
            )
            assert pool._get_pool_key(other) != baseline, (
                f"{field_name} does not participate in the pool key, so two "
                "configurations differing only by it would share a chunker"
            )

    def test_pool_key_separates_distinct_injected_providers(self):
        # A provider is a callable, not JSON, and must still split the pool.
        pool = ChunkerPool()

        def provider_a(texts):
            return _deterministic_provider(texts)

        def provider_b(texts):
            return _deterministic_provider(texts)

        keys = {
            pool._get_pool_key(
                ChunkingConfig(
                    strategy=ChunkingStrategy.SEMANTIC_EMBEDDING,
                    embedding_provider=provider,
                )
            )
            for provider in (provider_a, provider_b)
        }
        assert len(keys) == 2


class TestSizingPolicy:
    """A declarative budget must reduce to exactly the absolute one it names.

    The equivalence is the whole safety argument: if `Fraction(0.10)` of 512 is
    not `Absolute(51)`, then the two dialects are two behaviours and the
    declarative front door has forked the system instead of unifying it.
    """

    @staticmethod
    def _absolute(window, overlap, minimum, maximum):
        return ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=window,
            chunk_overlap=overlap,
            min_chunk_size=minimum,
            max_chunk_size=maximum,
        )

    @staticmethod
    def _sized(config):
        return (
            config.chunk_size,
            config.chunk_overlap,
            config.min_chunk_size,
            config.max_chunk_size,
        )

    def test_a_fraction_resolves_to_the_absolute_it_names(self):
        declarative = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            sizing=SizingPolicy(
                window=Absolute(512),
                overlap=Fraction(0.10),
                minimum=Absolute(100),
                maximum=Absolute(2048),
            ),
        )
        assert self._sized(declarative) == self._sized(
            self._absolute(512, 51, 100, 2048)
        )

    def test_both_dialects_chunk_identically(self):
        # Equivalence at the config level is necessary but not sufficient; the
        # claim that matters is that the OUTPUT is the same.
        text = "word " * 500
        declarative = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            sizing=SizingPolicy(
                window=Absolute(512),
                overlap=Fraction(0.10),
                minimum=Fraction(0.20),
                maximum=Absolute(2048),
            ),
        )
        absolute = self._absolute(512, 51, 102, 2048)
        made = [
            [
                (c.text, c.start_pos, c.end_pos)
                for c in ChunkingStrategyFactory.create_strategy(
                    ChunkingStrategy.FIXED_SIZE, config
                ).chunk(text, "doc")
            ]
            for config in (declarative, absolute)
        ]
        assert made[0] == made[1]

    def test_fraction_boundaries(self):
        policy = SizingPolicy(window=Absolute(100), overlap=Fraction(0.0))
        assert policy.resolve().overlap == 0
        # Just under 1.0 must still leave forward progress, or the loop hangs.
        resolved = SizingPolicy(window=Absolute(100), overlap=Fraction(0.999)).resolve()
        assert resolved.overlap < resolved.window
        assert resolved.step >= 1

    def test_a_fraction_at_or_above_one_is_rejected(self):
        # A step of zero is non-termination, not a slow chunker.
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\)"):
            Fraction(1.0)
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\)"):
            Fraction(-0.1)

    def test_window_must_be_absolute(self):
        # It is the referent every Fraction resolves against.
        with pytest.raises(TypeError, match="window must be Absolute"):
            SizingPolicy(window=Fraction(0.5)).resolve()

    def test_resolved_sizing_owns_the_invariants(self):
        with pytest.raises(ValueError, match="window must be positive"):
            ResolvedSizing(window=0, overlap=0, minimum=0, maximum=10)
        with pytest.raises(ValueError, match="less than window"):
            ResolvedSizing(window=10, overlap=10, minimum=0, maximum=10)
        with pytest.raises(ValueError, match="exceeds window"):
            ResolvedSizing(window=10, overlap=0, minimum=11, maximum=10)
        with pytest.raises(ValueError, match="below window"):
            ResolvedSizing(window=10, overlap=0, minimum=0, maximum=9)

    def test_sizing_and_token_budget_together_are_rejected(self):
        # Two spellings of one budget with no correct precedence between them,
        # so ambiguity fails closed rather than resolving arbitrarily.
        with pytest.raises(ValueError, match="not\\s+both"):
            ChunkingConfig(
                sizing=SizingPolicy(window=Absolute(100)),
                token_budget=object(),
            )

    def test_a_policy_carries_its_measure(self):
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            sizing=SizingPolicy(
                window=Absolute(8),
                measure=TokenMeasure(WordTokenCounter()),
            ),
        )
        assert config.measure is not None
        assert config.measure.name == "token:words"

    def test_omitting_sizing_changes_nothing(self):
        # The legacy path must stay byte-identical, which is what makes the new
        # dialect additive rather than a migration.
        assert self._sized(ChunkingConfig()) == (512, 50, 100, 2048)


class TestTraceHonesty:
    """A trace must report sizes in the unit it was actually measured in.

    `min/max_chunk_units` were hardcoded `len()` while documented as the cap
    check. Under a token measure that makes the comparison a reader would
    naturally make -- "max 512 against my 512 budget" -- meaningless, and
    nothing fails, because a plausible number is still a number. This is the
    quiet half of the measure work: the code got a second unit, and every
    readout that assumed one had to be told.
    """

    def test_the_default_trace_reports_characters(self):
        trace = compute_trace("alpha beta gamma", [])
        assert trace.size_unit == "char"

    def test_a_measured_trace_reports_the_measure_unit(self):
        text = " ".join(f"word{i}" for i in range(40))
        measure = TokenMeasure(WordTokenCounter())
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=8,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=16,
            measure=measure,
        )
        strat = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        chunks = strat.chunk(text, "doc")
        trace = compute_trace(text, chunks, measure=measure)

        assert trace.size_unit == "token:words"
        # The load-bearing assertion: the reported max must be comparable to the
        # budget of 8. Measured in characters it would be ~50 and would read as
        # a gross overflow of a budget it in fact respects.
        assert trace.max_chunk_units <= 8, trace.summary()
        assert "token:words" in trace.summary()

        untold = compute_trace(text, chunks)
        assert untold.max_chunk_units > 8, (
            "positive control: without the measure the same chunks report a "
            "character size, which is exactly the misreading being fixed"
        )

    def test_size_warnings_are_scoped_to_the_measure_they_fit(self):
        # 100 and 10 000 are character intuitions. A 100-token chunk is
        # ordinary; a 10 000-token one exceeds most model contexts. Emitting
        # character advice about a token budget trains readers to ignore
        # warnings, which is worse than silence.
        from proximadb_sdk.chunking_strategies.parser_utils import ConfigValidator

        chars = ConfigValidator.validate_chunk_size(50)
        assert chars.warnings and "chars" in chars.warnings[0]

        tokens = ConfigValidator.validate_chunk_size(50, measure_name="token:words")
        assert not tokens.warnings

        # Ordering errors are measure-independent and must still fire.
        bad = ConfigValidator.validate_chunk_size(
            5, min_chunk_size=10, measure_name="token:words"
        )
        assert not bad.valid

        # And the whole-config entry point must FORWARD the measure, or the
        # scoping is dead code everywhere it actually runs.
        via_config = ConfigValidator.validate_config(
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=8,
                chunk_overlap=0,
                min_chunk_size=1,
                max_chunk_size=16,
                measure=TokenMeasure(WordTokenCounter()),
            )
        )
        assert not any("chars" in w for w in via_config.warnings), via_config.warnings


class TestDocumentedNonInvariants:
    """Things that are deliberately NOT guaranteed, recorded so they survive.

    An unwritten non-invariant is indistinguishable from a bug, and the next
    reader "fixes" it. These record the reasoning instead.
    """

    def test_token_measured_chunks_are_not_char_contiguous(self):
        # A token grid excludes inter-token whitespace, so consecutive chunks
        # can leave a gap in CHARACTER space. That is correct, not a leak: the
        # gaps are whitespace, no content is lost, and forcing contiguity would
        # mean assigning whitespace arbitrarily to one side of every cut.
        #
        # This is precisely why the suite measures TOTALITY over non-whitespace
        # units. Asserting char-contiguity instead would fail every token-
        # measured chunking for a reason that is not a defect.
        text = "alpha beta   gamma delta   epsilon zeta eta theta"
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=3,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=6,
            measure=TokenMeasure(WordTokenCounter()),
        )
        chunks = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        ).chunk(text, "doc")
        assert len(chunks) > 1

        gaps = [text[a.end_pos : b.start_pos] for a, b in zip(chunks, chunks[1:])]
        assert any(gaps), "the premise of this test has changed: no gaps at all"
        assert all(
            not gap.strip() for gap in gaps
        ), f"a gap carried CONTENT, which IS a bug: {gaps!r}"
        assert check_totality(text, chunks) is None

    def test_measure_count_need_not_equal_its_own_unit_count(self):
        # count() measures RENDERED text (special tokens); content_offsets()
        # excludes them. They disagree BY DESIGN, and code that assumes
        # otherwise breaks on every real tokenizer -- it is the reason
        # TokenBudgetStrategy binary-searches instead of computing start+target.
        counter = WordTokenCounter()
        text = "alpha beta gamma"
        assert counter.count(text) != len(counter.content_offsets(text))
        assert counter.count(text) == len(counter.content_offsets(text)) + 2


class TestCapPostCondition:
    """`max_chunk_size` must hold for chunks a strategy has ALREADY emitted.

    Per-strategy discipline was sufficient while there were seven strategies
    maintained together. TD-CHUNK-2 makes boundary sources plural and
    composable, so the number of places that could forget is about to grow --
    and a forgotten cap is invisible in production (the provider truncates
    silently). Hence a structural guarantee rather than a convention.
    """

    class _OverCapStrategy(ChunkingStrategyInterface):
        """A deliberately non-compliant source: one chunk, whole document."""

        _offset_contract = OFFSET_CONTRACT_EXACT

        def chunk(self, text, source_id, base_metadata=None):
            return [
                TextChunk(
                    text=text,
                    start_pos=0,
                    end_pos=len(text),
                    chunk_id=f"{source_id}_chunk_0",
                    metadata={"source_id": source_id, "total_chunks": 1},
                )
            ]

    class _LegacyOverCapStrategy(_OverCapStrategy):
        """Same violation, but its offsets do not index the source."""

        _offset_contract = OFFSET_CONTRACT_LEGACY

    @staticmethod
    def _config(cap):
        return ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=cap,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=cap,
        )

    def test_an_over_cap_chunk_is_split(self):
        text = "word " * 400
        cap = 100
        chunks = self._OverCapStrategy(self._config(cap)).chunk(text, "doc")

        assert len(chunks) > 1, "the guard did not engage"
        for chunk in chunks:
            assert chunk.end_pos - chunk.start_pos <= cap, chunk.text[:40]
            assert chunk.metadata.get("cap_enforced") is True

    def test_splitting_preserves_content_and_exactness(self):
        # Splitting must not become its own data-loss bug.
        text = "word " * 400
        chunks = self._OverCapStrategy(self._config(100)).chunk(text, "doc")
        assert check_totality(text, chunks) is None
        assert check_exactness(text, chunks) is None
        assert "".join(c.text for c in chunks) == text

    def test_ids_are_restated_so_none_collide(self):
        # Ids encode POSITION; leaving them after a count change emits duplicate
        # ids -- the exact collision this program already found in anvaiops.
        text = "word " * 400
        chunks = self._OverCapStrategy(self._config(100)).chunk(text, "doc")
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert ids[0] == "doc_chunk_0" and ids[1] == "doc_chunk_1"
        assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)

    def test_a_compliant_strategy_is_returned_untouched(self):
        # The guard must be a genuine no-op on the common path -- same objects,
        # not merely equal ones -- or it would rewrite every chunk in the SDK.
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, **DEFAULTS)
        strat = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        text = by_name("prose").text
        first = strat.chunk(text, "doc")
        assert all(not c.metadata.get("cap_enforced") for c in first)

    def test_legacy_offsets_are_left_alone(self):
        # Re-cutting means indexing the source with the chunk's own offsets. A
        # legacy strategy's offsets do not index the source, so splitting on
        # them would produce confidently wrong text. Refusing is the honest act.
        text = "word " * 400
        chunks = self._LegacyOverCapStrategy(self._config(100)).chunk(text, "doc")
        assert len(chunks) == 1
        assert not chunks[0].metadata.get("cap_enforced")

    def test_the_guard_does_not_narrow_the_signature_it_guards(self):
        """A guard that changes the API it guards is worse than no guard.

        `CodeChunkingStrategy.chunk` takes `metadata=` where the base class
        takes `base_metadata=` -- a real interface divergence in this codebase.
        A wrapper that restates the common signature silently rejects the
        divergent one with a TypeError, which is how this first shipped.
        """

        class DivergentSignature(ChunkingStrategyInterface):
            _offset_contract = OFFSET_CONTRACT_EXACT

            def chunk(self, text, source_id, metadata=None, *, extra=None):
                return [
                    TextChunk(
                        text=text,
                        start_pos=0,
                        end_pos=len(text),
                        chunk_id=f"{source_id}_chunk_0",
                        metadata={"seen": metadata, "extra": extra},
                    )
                ]

        strategy = DivergentSignature(self._config(10_000))
        chunks = strategy.chunk("abc", "doc", {"a": 1}, extra="x")
        assert chunks[0].metadata["seen"] == {"a": 1}
        assert chunks[0].metadata["extra"] == "x"

    def test_the_guard_is_measure_aware(self):
        # A char-arithmetic cut would mis-split under a token measure. The cap
        # here is 8 WORDS, so a character reading would produce ~50 chunks.
        text = " ".join(f"word{i}" for i in range(80))
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=8,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=8,
            measure=TokenMeasure(WordTokenCounter()),
        )
        chunks = self._OverCapStrategy(config).chunk(text, "doc")
        assert len(chunks) <= 12, f"{len(chunks)} chunks: cut in characters"
        for chunk in chunks:
            assert len(re.findall(r"\S+", chunk.text)) <= 8
        assert check_totality(text, chunks) is None


class TestBoundarySources:
    """Proposing cuts, separately from choosing them (ADR-091 D2).

    The claim under test is not that the abstraction exists, but that it buys
    something: several kinds of structure informing ONE segmentation, which is
    impossible while a strategy both proposes and segments.
    """

    @staticmethod
    def _strategy(kind):
        config = ChunkingConfig(strategy=kind, **DEFAULTS)
        return ChunkingStrategyFactory.create_strategy(kind, config)

    def test_a_strategy_becomes_a_source_without_being_rewritten(self):
        # The compatibility facade: every strategy already answers
        # preferred_boundaries, so it is already a source.
        strategy = self._strategy(ChunkingStrategy.SENTENCE)
        source = StrategyBoundarySource(strategy, kind=BoundaryKind.SENTENCE)
        assert isinstance(source, BoundarySource)

        text = by_name("prose").text
        found = source.boundaries(text)
        assert found
        assert [b.end for b in found] == sorted(b.end for b in found)
        assert all(0 < b.end <= len(text) for b in found)
        assert all(b.kind is BoundaryKind.SENTENCE for b in found)

    def test_a_composite_is_the_union_of_its_sources(self):
        text = by_name("header_dense_markdown").text
        sentences = StrategyBoundarySource(
            self._strategy(ChunkingStrategy.SENTENCE), kind=BoundaryKind.SENTENCE
        )
        paragraphs = StrategyBoundarySource(
            self._strategy(ChunkingStrategy.PARAGRAPH), kind=BoundaryKind.PARAGRAPH
        )
        composite = CompositeBoundarySource(sentences, paragraphs)

        merged = {b.end for b in composite.boundaries(text)}
        individually = {b.end for b in sentences.boundaries(text)} | {
            b.end for b in paragraphs.boundaries(text)
        }
        assert merged == individually

    def test_nested_structures_compose_to_nothing(self):
        """Composition pays only when sources propose positions others cannot.

        Measured, not assumed: on the header-dense corpus entry, paragraph ends
        are a STRICT SUBSET of sentence ends (80 of 81), because a paragraph
        break is also a sentence break. So composing SENTENCE with PARAGRAPH
        adds exactly nothing.

        Recorded because the natural first demo of a composite is "sentences
        plus paragraphs", and it would look like the abstraction works while
        proving nothing. The sources worth composing are the ones whose
        positions are structurally independent -- a heading's START is not a
        sentence END, which is why the heading case below does move the cuts.
        """
        text = by_name("header_dense_markdown").text
        sentence_ends = {
            b.end
            for b in StrategyBoundarySource(
                self._strategy(ChunkingStrategy.SENTENCE)
            ).boundaries(text)
        }
        paragraph_ends = {
            b.end
            for b in StrategyBoundarySource(
                self._strategy(ChunkingStrategy.PARAGRAPH)
            ).boundaries(text)
        }
        assert paragraph_ends < sentence_ends

    def test_merging_keeps_the_strongest_meaning_at_a_shared_offset(self):
        # Sources agreeing is the COMMON case (a heading is also a paragraph
        # break). Keeping the strongest preserves the informative meaning;
        # keeping whichever came first would make segmentation depend on
        # argument order.
        weak = [Boundary(end=10, kind=BoundaryKind.SENTENCE, meaning={"who": "weak"})]
        strong = [
            Boundary(
                end=10,
                kind=BoundaryKind.HEADING,
                meaning={"who": "strong", "path": "A"},
            )
        ]
        assert merge_boundaries([weak, strong])[0].meaning["who"] == "strong"
        # Order must not matter.
        assert merge_boundaries([strong, weak])[0].meaning["who"] == "strong"

    def test_merge_is_ordered_and_deduplicated(self):
        merged = merge_boundaries(
            [
                [Boundary(end=30), Boundary(end=10)],
                [Boundary(end=10), Boundary(end=20)],
            ]
        )
        assert [b.end for b in merged] == [10, 20, 30]

    def test_a_source_need_not_be_total_or_respect_a_budget(self):
        # The whole point of the split: a source author knows about markdown,
        # not about token budgets. Proposing two cuts in a long document is a
        # legal source, and the segmenter still owns coverage.
        class SparseSource:
            name = "sparse"

            def boundaries(self, text, *, source_id="doc", base_metadata=None):
                return (
                    Boundary(end=len(text) // 2, kind=BoundaryKind.SECTION),
                    Boundary(end=len(text), kind=BoundaryKind.DOCUMENT),
                )

        assert isinstance(SparseSource(), BoundarySource)


class TestSegmenterConsumesSources:
    """`TokenBudgetStrategy` is the segmenter; it must take candidates from any source."""

    @staticmethod
    def _contract():
        counter = WordTokenCounter()
        return ResolvedInputContract(
            model_id="m",
            model_revision="r",
            counter=counter,
            effective_context_limit=512,
        )

    def _segmenter(self, source=None, target=12):
        base = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.SENTENCE,
            ChunkingConfig(
                strategy=ChunkingStrategy.SENTENCE,
                chunk_size=10_000,
                min_chunk_size=1,
                max_chunk_size=100_000,
            ),
        )
        return TokenBudgetStrategy(
            base,
            TokenBudget(target_tokens=target, overlap_tokens=0, min_content_tokens=1),
            self._contract(),
            boundary_source=source,
        )

    def test_the_single_strategy_path_is_the_degenerate_general_one(self):
        # Passing no source must be identical to adapting the wrapped strategy,
        # or the compat path is a second implementation waiting to diverge.
        text = by_name("prose").text
        implicit = self._segmenter().chunk(text, "doc")
        explicit = self._segmenter(
            StrategyBoundarySource(
                ChunkingStrategyFactory.create_strategy(
                    ChunkingStrategy.SENTENCE,
                    ChunkingConfig(
                        strategy=ChunkingStrategy.SENTENCE,
                        chunk_size=10_000,
                        min_chunk_size=1,
                        max_chunk_size=100_000,
                    ),
                )
            )
        ).chunk(text, "doc")
        assert [(c.text, c.start_pos, c.end_pos) for c in implicit] == [
            (c.text, c.start_pos, c.end_pos) for c in explicit
        ]

    def test_a_composite_source_changes_where_cuts_land(self):
        """The load-bearing claim: composing sources is not decorative.

        Uses `prose`, where the two structures genuinely diverge (522 sentence
        ends against 61 paragraph ends). On `header_dense_markdown` they nearly
        coincide, so the same test there would pass vacuously -- see
        `test_nested_structures_compose_to_nothing`.
        """
        text = by_name("prose").text

        def strat(kind):
            return ChunkingStrategyFactory.create_strategy(
                kind,
                ChunkingConfig(
                    strategy=kind,
                    chunk_size=10_000,
                    min_chunk_size=1,
                    max_chunk_size=100_000,
                ),
            )

        only_paragraph = StrategyBoundarySource(
            strat(ChunkingStrategy.PARAGRAPH), kind=BoundaryKind.PARAGRAPH
        )
        composite = CompositeBoundarySource(
            only_paragraph,
            StrategyBoundarySource(
                strat(ChunkingStrategy.SENTENCE), kind=BoundaryKind.SENTENCE
            ),
        )
        one = self._segmenter(only_paragraph).chunk(text, "doc")
        many = self._segmenter(composite).chunk(text, "doc")

        assert [c.end_pos for c in one] != [c.end_pos for c in many]
        # Coverage must survive the extra freedom -- more candidates must never
        # mean less document.
        for chunks in (one, many):
            assert check_totality(text, chunks) is None
            assert check_exactness(text, chunks) is None

    def test_the_segmenter_resolves_candidates_at_token_granularity(self):
        """Candidates closer together than one token are indistinguishable.

        Measured while building this: on `header_dense_markdown` a heading START
        at offset 49 and a paragraph END at 47 map to the SAME token index, so
        adding the heading source moved no cut at all. That is correct -- the
        segmenter cuts on the token grid, and there is no cut position between
        two adjacent tokens -- but it is deeply counter-intuitive when a source
        is visibly contributing offsets and nothing changes.

        Recorded so the next person debugging "my boundary source does nothing"
        checks token distance before suspecting the plumbing. Note the MEANING
        still propagates, because it attaches to the token index either way.
        """
        text = by_name("header_dense_markdown").text

        class NearMiss:
            name = "near_miss"

            def boundaries(self, text, *, source_id="doc", base_metadata=None):
                # Two characters off a real paragraph end: a different offset,
                # the same token.
                return (Boundary(end=49, kind=BoundaryKind.HEADING),)

        paragraph = StrategyBoundarySource(
            ChunkingStrategyFactory.create_strategy(
                ChunkingStrategy.PARAGRAPH,
                ChunkingConfig(
                    strategy=ChunkingStrategy.PARAGRAPH,
                    chunk_size=10_000,
                    min_chunk_size=1,
                    max_chunk_size=100_000,
                ),
            ),
            kind=BoundaryKind.PARAGRAPH,
        )
        alone = self._segmenter(paragraph).chunk(text, "doc")
        with_near_miss = self._segmenter(
            CompositeBoundarySource(paragraph, NearMiss())
        ).chunk(text, "doc")
        assert [c.end_pos for c in alone] == [c.end_pos for c in with_near_miss]

    def test_boundary_meaning_reaches_the_emitted_chunk(self):
        # A cut position alone is lossy; `meaning` is the part retrieval wants,
        # and every implementation in the ADR-091 census threw it away.
        text = by_name("header_dense_markdown").text

        class Headingish:
            name = "headingish"

            def boundaries(self, text, *, source_id="doc", base_metadata=None):
                return tuple(
                    Boundary(
                        end=m.start(),
                        kind=BoundaryKind.HEADING,
                        meaning={"path": m.group(0).strip()},
                    )
                    for m in re.finditer(r"^#{1,6} .+$", text, re.MULTILINE)
                    if m.start() > 0
                ) + (Boundary(end=len(text), kind=BoundaryKind.DOCUMENT),)

        chunks = self._segmenter(Headingish(), target=40).chunk(text, "doc")
        carried = [c.metadata.get("boundary_meaning") for c in chunks]
        assert any(m and "path" in m for m in carried), carried
        assert any(
            c.metadata.get("boundary_kind") == BoundaryKind.HEADING.value
            for c in chunks
        )


NESTED_DOC = """# Install

intro text here

## Docker

docker text here

### Compose

compose text here

## Native

native text here

# Usage

usage text here
"""


class TestPropagatedChunkContext:
    """Repeated context is model input, while spans remain source-exact."""

    @staticmethod
    def _segmenter(target=8):
        base = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE,
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=10_000,
                min_chunk_size=1,
                max_chunk_size=100_000,
            ),
        )
        return TokenBudgetStrategy(
            base,
            TokenBudget(target_tokens=target, overlap_tokens=1),
            ResolvedInputContract(
                model_id="m",
                model_revision="r",
                counter=WordTokenCounter(),
                effective_context_limit=32,
            ),
            context_renderer=ChunkContextRenderer(
                metadata_key="title",
                template="Title: {context}\n\n{text}",
            ),
        )

    def test_context_is_counted_on_every_chunk_without_corrupting_offsets(self):
        text = "one two three four five six seven"

        chunks = self._segmenter().chunk(text, "doc", {"title": "Useful heading"})

        assert len(chunks) > 1
        assert all(chunk.model_input_text is not None for chunk in chunks)
        assert all(
            chunk.model_input_text.startswith("Title: Useful heading\n\n")
            for chunk in chunks
        )
        assert all(
            WordTokenCounter().count(chunk.model_input_text) <= 8 for chunk in chunks
        )
        assert all(
            text[chunk.start_pos : chunk.end_pos] == chunk.text for chunk in chunks
        )
        assert all(chunk.input_text == chunk.model_input_text for chunk in chunks)
        assert check_totality(text, chunks) is None

    def test_context_that_leaves_no_source_token_capacity_fails_closed(self):
        with pytest.raises(ValueError, match="context.*no room"):
            self._segmenter(target=4).chunk(
                "body tokens", "doc", {"title": "far too many title words"}
            )


class TestHeadingOutline:
    """The hierarchy, which is the part every implementation threw away.

    Heading *detection* was never the gap -- `semantic.py` already found
    markdown and HTML headings and already excluded a `#` inside a fenced code
    block. What it discarded was the ancestor chain: it recorded "level 2,
    titled Docker" but not "Installation > Docker". Level and title alone do not
    identify a section (half a dozen documents have a "Configuration" heading);
    the path does.
    """

    def test_ancestors_nest_and_pop(self):
        outline = HeadingOutline.build(NESTED_DOC)
        assert [h.title for h in outline.headings] == [
            "Install",
            "Docker",
            "Compose",
            "Native",
            "Usage",
        ]
        assert outline.paths == [
            ("Install",),
            ("Install", "Docker"),
            ("Install", "Docker", "Compose"),
            # Back up a level: a sibling must not inherit its predecessor's
            # children, and must not become its own ancestor.
            ("Install", "Native"),
            ("Usage",),
        ]

    def test_a_level_jump_does_not_invent_ancestors(self):
        # `#` then `###` skips level 2. The honest path names what is actually
        # there rather than fabricating an intermediate section.
        outline = HeadingOutline.build("# A\n\ntext\n\n### C\n\ntext\n")
        assert outline.paths == [("A",), ("A", "C")]

    def test_path_at_resolves_any_offset(self):
        outline = HeadingOutline.build(NESTED_DOC)
        offset = NESTED_DOC.index("compose text here")
        assert outline.path_at(offset) == ("Install", "Docker", "Compose")

    def test_content_before_the_first_heading_has_no_path(self):
        # Inventing "Introduction" would put a title in the index that the
        # document never contained.
        text = "preamble\n\n# First\n\nbody\n"
        outline = HeadingOutline.build(text)
        assert outline.path_at(0) == ()
        assert outline.meaning_at(0) == {}

    def test_a_hash_inside_a_fenced_block_is_not_a_heading(self):
        text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\nbody\n"
        outline = HeadingOutline.build(text)
        assert [h.title for h in outline.headings] == ["Real"]

    def test_html_headings_with_attributes_are_found(self):
        # A bare `<h2>` pattern never matched real markup, so HTML documents
        # were never split by heading at all.
        outline = HeadingOutline.build('<h1>Top</h1>\n<h2 class="x">Sub</h2>\n')
        assert [h.title for h in outline.headings] == ["Top", "Sub"]
        assert outline.paths[-1] == ("Top", "Sub")

    def test_documents_without_headings_are_not_an_error(self):
        outline = HeadingOutline.build("just prose, no structure at all")
        assert outline.headings == []
        assert outline.path_at(5) == ()


class TestHeadingSourceAndAnnotation:
    def test_the_source_carries_the_path_of_the_section_it_closes(self):
        # The segmenter attaches a boundary's meaning to the chunk that ENDS
        # there, so a heading boundary must describe the section being closed,
        # not the one beginning.
        found = HeadingBoundarySource().boundaries(NESTED_DOC)
        assert isinstance(HeadingBoundarySource(), BoundarySource)
        by_end = {b.end: b for b in found}
        docker_start = NESTED_DOC.index("## Docker")
        assert by_end[docker_start].meaning["heading_path"] == ["Install"]
        compose_start = NESTED_DOC.index("### Compose")
        assert by_end[compose_start].meaning["heading_path"] == ["Install", "Docker"]

    def test_annotation_labels_every_chunk_not_just_lucky_ones(self):
        """The reason this is a post-pass and not only a source.

        A source can describe only chunks that happen to END on one of its
        proposals. A chunk that ended mid-section because the budget ran out
        gets nothing -- and most chunks are that kind. Labelling by LOOKUP works
        for every chunk under every strategy, including ones with no structural
        awareness at all.
        """
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=40,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=80,
        )
        strategy = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        chunks = strategy.chunk(NESTED_DOC, "doc")
        assert len(chunks) > 3

        before = [c.metadata.get("heading_path") for c in chunks]
        assert not any(before), "precondition: fixed_size knows nothing of headings"

        annotate_heading_paths(NESTED_DOC, chunks)
        labelled = [c.metadata.get("heading_path") for c in chunks]
        assert all(labelled), "every chunk after the first heading must be labelled"
        assert ["Install", "Docker", "Compose"] in labelled

    def test_annotation_is_a_no_op_without_headings(self):
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, **DEFAULTS)
        strategy = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        text = "prose with no headings at all. " * 40
        chunks = strategy.chunk(text, "doc")
        annotate_heading_paths(text, chunks)
        assert not any(c.metadata.get("heading_path") for c in chunks)

    def test_annotation_adds_no_vectors(self):
        # Co-design: metadata on an existing chunk. No extra chunks means no KEU
        # and no KSU delta -- which is why TD-CHUNK-3 sequences this first as
        # "strictly free quality".
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=40,
            chunk_overlap=0,
            min_chunk_size=1,
            max_chunk_size=80,
        )
        strategy = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.FIXED_SIZE, config
        )
        chunks = strategy.chunk(NESTED_DOC, "doc")
        before = [(c.text, c.start_pos, c.end_pos) for c in chunks]
        annotate_heading_paths(NESTED_DOC, chunks)
        after = [(c.text, c.start_pos, c.end_pos) for c in chunks]
        assert before == after, "annotation must not move or add a single chunk"

    def test_semantic_and_the_shared_module_agree(self):
        # One implementation, asserted rather than assumed: semantic.py now
        # shares structure.py instead of owning a private copy, and a private
        # copy drifting back is exactly what this program exists to prevent.
        text = by_name("header_dense_markdown").text
        strategy = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.SEMANTIC,
            ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC, **DEFAULTS),
        )
        assert strategy._protected_spans(text) == protected_spans(text)
        assert strategy.markdown_header_pattern is MARKDOWN_HEADING


#: Module scope, not class scope: a comprehension in a class body cannot see
#: names bound in that same body (only the outermost iterable is evaluated
#: there). Same trap as TOKEN_MEASURED_STRATEGIES above.
BOILERPLATE_LINE = (
    "Copyright 2026 Example Corp. All rights reserved worldwide everywhere."
)
BOILERPLATE_DOC = "".join(
    f"Unique body paragraph number {i} with its own distinct wording written.\n\n"
    f"{BOILERPLATE_LINE}\n\n"
    for i in range(6)
)


class TestIngestCarriesHeadingPaths:
    """The capability must reach the product, not merely exist.

    The ADR-091 census's central finding was capabilities built and never
    called -- the Rust chunker with zero callers, the SDP pipeline behind a flag
    nothing sets. Shipping a heading annotator that no ingest path invokes would
    have reproduced that exactly, so this pins the wiring, not just the feature.

    `TextDocumentProcessor` is the only production text-ingest caller in the
    SDK: every .txt/.md/.rst/.adoc goes through it, and anvaiops's connector SDK
    delegates here.
    """

    DOC = (
        "# Install\n\nSome intro prose long enough to matter.\n\n"
        "## Docker\n\nDocker instructions continue for a while here.\n\n"
        "### Compose\n\nCompose details, nested two levels deep.\n\n"
        "# Usage\n\nUsage text back at the top level.\n"
    )

    @staticmethod
    def _processor(**overrides):
        from proximadb_sdk.document_processor import (
            ProcessorConfig,
            TextDocumentProcessor,
        )

        settings = {
            "chunk_size": 60,
            "chunk_overlap": 0,
            "min_chunk_size": 1,
            "max_chunk_size": 200,
        }
        settings.update(overrides)
        return TextDocumentProcessor(ProcessorConfig(**settings))

    def test_every_chunk_reaches_ingest_with_its_path(self):
        chunks = self._processor().chunk(self.DOC, "readme.md")
        paths = [c.metadata.get("heading_path") for c in chunks]
        assert all(paths), paths
        assert ["Install", "Docker", "Compose"] in paths
        assert ["Usage"] in paths

    def test_preserve_structure_false_disables_it(self):
        chunks = self._processor(preserve_structure=False).chunk(self.DOC, "readme.md")
        assert not any(c.metadata.get("heading_path") for c in chunks)

    def test_annotation_changes_no_chunk(self):
        # The co-design claim at the product boundary: additive metadata only,
        # so no KEU (embedding) and no KSU (storage) delta.
        on = self._processor().chunk(self.DOC, "readme.md")
        off = self._processor(preserve_structure=False).chunk(self.DOC, "readme.md")
        assert [(c.text, c.start_pos, c.end_pos) for c in on] == [
            (c.text, c.start_pos, c.end_pos) for c in off
        ]

    @staticmethod
    def _dedup_processor(**overrides):
        from proximadb_sdk.document_processor import (
            ProcessorConfig,
            TextDocumentProcessor,
        )

        settings = {
            "chunk_size": 80,
            "chunk_overlap": 0,
            "min_chunk_size": 1,
            "max_chunk_size": 160,
        }
        settings.update(overrides)
        return TextDocumentProcessor(ProcessorConfig(**settings))

    def test_dedup_is_off_by_default(self):
        """Default OFF, unlike heading annotation, and the difference matters.

        Annotation only ADDS metadata; dedup REMOVES chunks, so enabling it
        changes what a tenant has stored. That is a product decision, and the
        repo's rule for anything in that class is to ship default-off until
        baked.
        """
        chunks = self._dedup_processor().chunk(BOILERPLATE_DOC, "doc.md")
        assert not any(c.metadata.get("duplicates_absorbed") for c in chunks)
        assert len(chunks) == 12

    def test_dedup_engages_when_opted_in(self):
        plain = self._dedup_processor().chunk(BOILERPLATE_DOC, "doc.md")
        deduped = self._dedup_processor(deduplicate_chunks=True).chunk(
            BOILERPLATE_DOC, "doc.md"
        )
        assert len(deduped) < len(plain)
        absorbed = [
            c.metadata.get("duplicates_absorbed")
            for c in deduped
            if c.metadata.get("duplicates_absorbed")
        ]
        assert absorbed, "removal must be recorded, never silent"

    def test_opting_in_keeps_every_distinct_body(self):
        # The safety property at the product boundary: only the repeated
        # boilerplate may go, never a body paragraph.
        deduped = self._dedup_processor(deduplicate_chunks=True).chunk(
            BOILERPLATE_DOC, "doc.md"
        )
        text = " ".join(c.text for c in deduped)
        for i in range(6):
            assert f"number {i}" in text, f"body paragraph {i} was dropped"

    def test_dedup_runs_after_annotation_so_survivors_keep_their_path(self):
        # Order matters: a representative that loses its heading path would
        # trade a cost saving for a retrieval regression.
        document = "# Guide\n\n" + BOILERPLATE_DOC
        deduped = self._dedup_processor(
            deduplicate_chunks=True, preserve_structure=True
        ).chunk(document, "doc.md")
        assert all(c.metadata.get("heading_path") for c in deduped)

    def test_a_document_without_headings_is_untouched(self):
        plain = "Just prose with no structure whatsoever. " * 20
        chunks = self._processor().chunk(plain, "notes.txt")
        assert chunks
        assert not any(c.metadata.get("heading_path") for c in chunks)


EVAL_BUDGET = {
    "chunk_size": 200,
    "chunk_overlap": 20,
    "min_chunk_size": 1,
    "max_chunk_size": 400,
}

#: Recorded floors, GENERATED from a real sweep, never hand-picked. Each is the
#: measured score at the commit that introduced it, so the gate is a ratchet:
#: a drop fails, and an improvement must be recorded deliberately (which is what
#: keeps a raised bar from silently rotting back down).
#:
#: Scored at a TIGHT budget on purpose. At the default 512 every answer in the
#: corpus fits comfortably and every chunker scores 1.00 containment -- a
#: saturated metric measures nothing. The regime where chunking decides anything
#: is the one where answers are comparable in size to the budget.
#: Written as EXACT fractions of the 8 cases, not as the rounded values the
#: report prints. Transcribing a displayed "0.38" into a threshold sets the bar
#: at 0.38 when the measurement is 3/8 = 0.375, so the gate fails against the
#: very run that produced it. Fractions cannot drift that way.
EVAL_BASELINE: dict[str, dict[str, float]] = {
    "sliding_window": {"containment": 3 / 4, "structure": 3 / 4, "attributable": 0},
    "fixed_size": {"containment": 3 / 4, "structure": 3 / 4, "attributable": 3 / 8},
    "sentence": {"containment": 7 / 8, "structure": 7 / 8, "attributable": 3 / 8},
    "paragraph": {"containment": 3 / 4, "structure": 7 / 8, "attributable": 3 / 8},
    "semantic": {"containment": 1, "structure": 1, "attributable": 5 / 8},
    "recursive": {"containment": 3 / 4, "structure": 7 / 8, "attributable": 3 / 8},
}


def _eval_chunker(strategy):
    config = ChunkingConfig(
        strategy=strategy,
        embedding_provider=_deterministic_provider,
        **EVAL_BUDGET,
    )
    built = ChunkingStrategyFactory.create_strategy(strategy, config)

    def chunk(text):
        chunks = built.chunk(text, "doc")
        annotate_heading_paths(text, chunks)
        return chunks

    return chunk


class TestChunkingEvals:
    """The non-deterministic half: boundary QUALITY, scored against a rubric.

    Tests cover what has a right answer. Whether a cut lands in a *good* place
    does not, so it is gated on thresholds rather than asserted -- the
    tests-vs-evals split the repo mandates for any ranked or generated surface.
    """

    @pytest.mark.parametrize(
        "strategy",
        [
            pytest.param(s, id=s.value)
            for s in SWEPT_STRATEGIES
            if s.value in EVAL_BASELINE
        ],
    )
    def test_quality_does_not_regress(self, strategy):
        report = run_eval(strategy.value, _eval_chunker(strategy))
        floor = EVAL_BASELINE[strategy.value]
        assert report.containment >= floor["containment"] - 1e-9, report.render()
        assert report.structural_integrity >= floor["structure"] - 1e-9, report.render()
        assert report.attributability >= floor["attributable"] - 1e-9, report.render()

    def test_the_rubric_discriminates(self):
        """A rubric where everything scores the same is not measuring anything.

        This is the guard on the eval itself. The first version of this suite
        scored 1.00 containment for every strategy, because the corpus answers
        were all far smaller than the budget -- it looked like a clean pass and
        told us nothing. If a future change flattens the scores again, that is a
        problem with the eval, not a triumph of the chunkers.
        """
        scores = {
            s.value: run_eval(s.value, _eval_chunker(s)).containment
            for s in SWEPT_STRATEGIES
            if s.value in EVAL_BASELINE
        }
        assert len(set(scores.values())) > 1, f"rubric is saturated: {scores}"

    def test_output_cannot_be_gamed_by_one_huge_chunk(self):
        """Containment alone is gameable: emit the whole document as one chunk.

        That scores a perfect 1.00 and is useless -- the embedding is about the
        entire document, so nothing is retrievable. Density is what catches it,
        which is why the rubric scores both.
        """

        class WholeDocument(ChunkingStrategyInterface):
            _offset_contract = OFFSET_CONTRACT_EXACT

            def chunk(self, text, source_id, base_metadata=None):
                return [
                    TextChunk(
                        text=text,
                        start_pos=0,
                        end_pos=len(text),
                        chunk_id=f"{source_id}_chunk_0",
                        metadata={},
                    )
                ]

        cheat = WholeDocument(
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=10**9,
                min_chunk_size=1,
                max_chunk_size=10**9,
            )
        )
        report = run_eval("whole_document", lambda t: cheat.chunk(t, "doc"))
        assert report.containment == 1.0, "precondition: the cheat does contain all"
        best = max(
            run_eval(s.value, _eval_chunker(s)).mean_density
            for s in SWEPT_STRATEGIES
            if s.value in EVAL_BASELINE
        )
        assert report.mean_density < best, (
            "density must punish the degenerate chunker that wins containment "
            "by emitting the whole document"
        )

    def test_the_gate_rejects_a_structurally_blind_chunker(self):
        """Teeth: a chunker that ignores structure must fail the recorded floor.

        Written as a permanent test rather than a one-off sabotage, because the
        thing worth pinning is that the FLOOR mechanism fires -- a rubric whose
        thresholds cannot fail is a dashboard, not a gate.

        Cuts blindly every 50 characters, which slices fences and table rows
        apart. It respects the offset contract, so this is a quality failure and
        not a correctness one: the invariant suite would pass it.
        """

        class BlindCutter(ChunkingStrategyInterface):
            _offset_contract = OFFSET_CONTRACT_EXACT

            def chunk(self, text, source_id, base_metadata=None):
                return [
                    TextChunk(
                        text=text[i : i + 50],
                        start_pos=i,
                        end_pos=min(i + 50, len(text)),
                        chunk_id=f"{source_id}_chunk_{i // 50}",
                        metadata={},
                    )
                    for i in range(0, len(text), 50)
                ]

        blind = BlindCutter(
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=50,
                chunk_overlap=0,
                min_chunk_size=1,
                max_chunk_size=50,
            )
        )
        report = run_eval("blind", lambda t: blind.chunk(t, "doc"))
        best = EVAL_BASELINE["semantic"]
        assert report.structural_integrity < best["structure"]
        assert report.containment < best["containment"]

    def test_every_case_locates_its_answer(self):
        # A needle that no longer matches would silently score 0 forever. Fail
        # loudly instead, naming the case.
        for case in STANDARD_CASES:
            start, end = case.span()
            assert end > start
            assert by_name(case.corpus).text[start:end] == case.needle

    def test_cases_state_what_they_test(self):
        # An eval nobody can read is an eval nobody trusts.
        for case in STANDARD_CASES:
            assert case.question.strip()
            assert case.tests.strip(), f"{case.name} does not say what it tests"


class TestDeduplication:
    """The only item in TD-CHUNK-3 that REDUCES cost (KEU once, KSU forever).

    Every safety property here exists because the asymmetry is severe: keeping a
    duplicate costs a little money, while dropping a distinct chunk removes
    content that no downstream layer can detect is missing.
    """

    @staticmethod
    def _chunk(text, index, start=0):
        return TextChunk(
            text=text,
            start_pos=start,
            end_pos=start + len(text),
            chunk_id=f"doc_chunk_{index}",
            metadata={},
        )

    def test_exact_duplicates_collapse_to_one(self):
        boiler = "Copyright 2026 Example Corp. All rights reserved worldwide."
        chunks = [
            self._chunk(boiler, 0, 0),
            self._chunk("Genuinely distinct content about vector indexes.", 1, 100),
            self._chunk(boiler, 2, 200),
            self._chunk(boiler, 3, 300),
        ]
        result = deduplicate(chunks)
        assert len(result.kept) == 2
        assert result.removed_count == 2
        assert result.reduction == 0.5

    def test_distinct_content_is_never_removed(self):
        # The property that matters most. Dropping distinct content is
        # unrecoverable and invisible.
        chunks = [
            self._chunk(
                f"Section {i} discusses a completely different topic "
                f"with its own vocabulary and specifics {i}.",
                i,
                i * 100,
            )
            for i in range(12)
        ]
        result = deduplicate(chunks)
        assert result.removed_count == 0
        assert len(result.kept) == 12

    def test_first_occurrence_is_the_representative(self):
        boiler = "Shared boilerplate footer text repeated across the document."
        chunks = [self._chunk(boiler, i, i * 100) for i in range(3)]
        result = deduplicate(chunks)
        assert [c.chunk_id for c in result.kept] == ["doc_chunk_0"]

    def test_removal_is_recorded_never_silent(self):
        # A removal that leaves no trace is indistinguishable from data loss.
        boiler = "Shared boilerplate footer text repeated across the document."
        chunks = [self._chunk(boiler, i, i * 100) for i in range(3)]
        result = deduplicate(chunks)
        keeper = result.kept[0]
        assert keeper.metadata["duplicates_absorbed"] == 2
        assert keeper.metadata["duplicate_spans"] == [(100, 160), (200, 260)]

    def test_similarity_does_not_chain_transitively(self):
        """A~B and B~C must not collapse A, B and C together.

        Chaining is how a dedup pass quietly eats a document: each step looks
        like a small merge, and the composition chews through content that is
        not similar at all. Only comparisons against a surviving representative
        may remove anything.
        """
        base = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        drifted = "beta gamma delta epsilon zeta eta theta iota kappa lambda"
        far = "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi"
        chunks = [
            self._chunk(base, 0, 0),
            self._chunk(drifted, 1, 100),
            self._chunk(far, 2, 200),
        ]
        result = deduplicate(chunks, threshold=0.3)
        kept = {c.text for c in result.kept}
        assert far in kept, "a dissimilar chunk was chained away"

    def test_the_result_is_deterministic(self):
        """Same input, same decision -- across processes.

        Python salts `hash()` for str per process, so a dedup built on it would
        make the chunk count -- and therefore the bill -- differ between two runs
        on identical input.
        """
        texts = ["repeated boilerplate line here"] * 3 + [
            "unique line %d" % i for i in range(5)
        ]
        chunks = [self._chunk(t, i, i * 100) for i, t in enumerate(texts)]
        first = deduplicate(list(chunks))
        second = deduplicate([self._chunk(t, i, i * 100) for i, t in enumerate(texts)])
        assert [c.chunk_id for c in first.kept] == [c.chunk_id for c in second.kept]

    def test_stable_hash_survives_a_different_hash_seed(self):
        """The determinism guard, tested the only way it can be.

        `PYTHONHASHSEED` salts `hash()` per PROCESS, so an in-process assertion
        cannot see the bug at all -- the value is stable within one run and
        differs between runs. This spawns two interpreters with different seeds
        and requires the same answer, and confirms the built-in really does
        differ under them (otherwise the test would pass on a platform where
        there is nothing to defend against).
        """
        import os
        import subprocess
        import sys

        program = (
            "from proximadb_sdk.chunking_strategies.dedup import _stable_hash;"
            "print(_stable_hash('boilerplate'), hash('boilerplate'))"
        )
        outputs = []
        for seed in ("0", "1"):
            environment = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
            completed = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                env=environment,
                check=True,
            )
            outputs.append(completed.stdout.split())

        assert outputs[0][0] == outputs[1][0], "_stable_hash changed with the seed"
        assert outputs[0][1] != outputs[1][1], (
            "built-in hash() did not vary, so this platform cannot demonstrate "
            "the hazard and the guard is untested"
        )

    @pytest.mark.parametrize(
        "strategy", [ChunkingStrategy.PARAGRAPH, ChunkingStrategy.SEMANTIC]
    )
    def test_recall_ratchet_no_answer_is_lost(self, strategy):
        """Dedup must never reduce the number of retrievable answers.

        Stated as a DELTA, not an absolute. A strategy's own containment
        limitations are not dedup's fault, and an absolute floor would blame it
        for them -- which is what the first version of this measurement did.
        """
        config = ChunkingConfig(
            strategy=strategy,
            embedding_provider=_deterministic_provider,
            **EVAL_BUDGET,
        )
        built = ChunkingStrategyFactory.create_strategy(strategy, config)
        for case in STANDARD_CASES:
            text = by_name(case.corpus).text
            chunks = list(built.chunk(text, "doc"))
            before = score_case(case, chunks).contained
            kept = deduplicate(list(chunks)).kept
            after = any(case.needle in c.text for c in kept)
            assert (
                after >= before
            ), f"{strategy.value}/{case.name}: dedup lost the answer"

    def test_reduction_is_real_on_the_corpus(self):
        # The cost claim, measured rather than asserted. Most of the saving is
        # EXACT duplicates: loosening the threshold from 1.0 to 0.4 adds only a
        # few points while taking on all the risk of dropping distinct content,
        # which is why the default stays conservative.
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            embedding_provider=_deterministic_provider,
            **EVAL_BUDGET,
        )
        built = ChunkingStrategyFactory.create_strategy(
            ChunkingStrategy.PARAGRAPH, config
        )
        total = kept = 0
        for entry in standard_corpus():
            chunks = list(built.chunk(entry.text, "doc"))
            total += len(chunks)
            kept += len(deduplicate(list(chunks)).kept)
        assert total > 0
        reduction = 1 - kept / total
        assert reduction > 0.15, f"expected a real saving, got {reduction:.1%}"


class TestGoldenOutput:
    """Behaviour-preservation oracle for refactors.

    The invariants prove chunking is CORRECT; they cannot prove a refactor was
    NEUTRAL. A change that moves a boundary while keeping every invariant
    satisfied is invisible to them — which is exactly the risk when a mechanical
    change is spread across dozens of call sites.

    Recorded at the completion of the correctness slice (0 violations, 70/70
    clean). A deliberate behaviour change regenerates this file in the same
    commit, exactly like BASELINE; it is generated, never hand-edited.
    """

    GOLDEN_PATH = Path(__file__).parent / "golden_chunk_output.json"

    def _actual(self) -> dict[str, str]:
        adapters = [_adapter(strategy) for strategy in SWEPT_STRATEGIES]
        return sweep_digests(adapters, standard_corpus())

    def test_output_is_unchanged(self):
        expected = load_golden(self.GOLDEN_PATH.read_text())
        actual = self._actual()
        changed, missing, added = diff_digests(expected, actual)

        if changed or missing or added:
            raise AssertionError(
                "chunk output changed against the recorded golden snapshot\n"
                f"  changed: {changed or 'none'}\n"
                f"  missing (case no longer produced): {missing or 'none'}\n"
                f"  added (new case): {added or 'none'}\n\n"
                "If this change was INTENTIONAL, regenerate the golden file in "
                "the same commit and say in the message what moved and why. If "
                "it was not, a refactor altered behaviour it was meant to "
                "preserve — that is the whole reason this oracle exists."
            )

    def test_golden_covers_the_whole_sweep(self):
        """The snapshot must not silently shrink."""
        expected = load_golden(self.GOLDEN_PATH.read_text())
        assert len(expected) == len(SWEPT_STRATEGIES) * len(standard_corpus())

    def test_no_case_records_an_error(self):
        """Every swept strategy produces output on every corpus entry."""
        expected = load_golden(self.GOLDEN_PATH.read_text())
        errored = sorted(k for k, v in expected.items() if v.startswith("ERROR"))
        assert not errored, f"golden records failures: {errored}"


class TestRunner:
    """The runner is the surface a foreign consumer drives, so it is tested as one.

    victor-rag and the anvaiops connector SDK are meant to run this bed against
    their own chunkers (ADR-091 D4); these tests pin the contract they will code
    against, using a deliberately trivial chunker rather than a ProximaDB one.
    """

    @staticmethod
    def _whole_document_chunker(text):
        obj = type("C", (), {})()
        obj.text = text
        obj.start_pos = 0
        obj.end_pos = len(text)
        obj.metadata = {"chunking_strategy": "whole"}
        return [obj]

    def _adapter(self, **overrides):
        base = {
            "name": "whole",
            "chunk": self._whole_document_chunker,
            "budget": 10**9,
        }
        base.update(overrides)
        return ChunkerAdapter(**base)

    def test_a_trivially_correct_chunker_is_clean(self):
        entry = by_name("prose")
        result = evaluate(self._adapter(), entry)
        assert result.error is None
        assert result.violated == frozenset()
        assert result.trace is not None and result.trace.is_total

    def test_unmeasured_invariants_are_reported_as_skipped_not_passed(self):
        """A green result must never be mistaken for full coverage."""
        result = evaluate(self._adapter(), by_name("prose"))
        assert set(result.skipped) == {
            Invariant.STREAM_EQUIVALENCE,
            Invariant.IDEMPOTENCE,
            Invariant.CONFIG_SAFETY,
        }
        assert "skipped" in result.render()

    def test_a_raising_chunker_is_recorded_not_propagated(self):
        """One throwing case must not hide the other 69."""

        def boom(_text):
            raise ValueError("synthetic failure")

        result = evaluate(self._adapter(chunk=boom), by_name("prose"))
        assert result.error is not None
        assert "ValueError: synthetic failure" in result.error
        assert "ERROR" in result.render()

    def test_budget_violation_is_detected(self):
        result = evaluate(self._adapter(budget=10), by_name("prose"))
        assert Invariant.CAP in result.violated

    def test_evaluate_all_sweeps_in_stable_order(self):
        entries = standard_corpus()[:3]
        results = evaluate_all([self._adapter()], entries)
        assert [r.corpus for r in results] == [e.name for e in entries]

    def test_format_baseline_omits_clean_cases(self):
        """The emitted baseline records only what actually violates."""
        rendered = format_baseline(evaluate_all([self._adapter()], standard_corpus()))
        assert rendered.startswith("BASELINE")
        assert "prose" not in rendered  # clean under a whole-document chunker

    def test_format_baseline_records_violations(self):
        rendered = format_baseline(
            evaluate_all([self._adapter(budget=10)], standard_corpus()[:1])
        )
        assert "Invariant.CAP" in rendered


class TestInvariantMechanics:
    """The checks themselves need teeth — especially the hang detector."""

    def test_config_safety_actually_catches_a_hang(self):
        """CONFIG_SAFETY is worthless if it cannot fire.

        victor-rag's ``_chunk_text`` infinite-loops on a legal config; this proves
        the detector would catch that rather than hanging CI forever.
        """
        if not _wall_clock_budget.supported:
            pytest.skip("SIGALRM unavailable on this platform")

        def spin():
            while True:
                pass

        violation = check_config_safety(spin, seconds=1)
        assert violation is not None
        assert violation.invariant is Invariant.CONFIG_SAFETY
        assert "did not terminate" in violation.detail

    def test_config_safety_tolerates_a_raise(self):
        """Raising on a hostile config is acceptable; hanging is not."""

        def boom():
            raise ValueError("rejected")

        assert check_config_safety(boom, seconds=1) is None

    def test_exactness_understands_a_byte_basis(self):
        """Byte offsets on non-ASCII must validate under BASIS_BYTE, not char."""
        source = "café résumé"
        encoded = source.encode("utf-8")
        chunk = type("C", (), {})()
        chunk.text = "café"
        chunk.start_pos = 0
        chunk.end_pos = len("café".encode())

        assert check_exactness(source, [chunk], basis=BASIS_BYTE) is None
        # The same span read as characters is wrong — which is the whole reason
        # ADR-091 axiom 2 demands the basis be declared.
        assert check_exactness(source, [chunk], basis=BASIS_CHAR) is not None
        assert len(encoded) != len(source)

    def test_non_empty_ignores_whitespace_only_input(self):
        assert check_non_empty("   \n\t ", []) is None
        assert check_non_empty("content", []) is not None


class TestCorpus:
    def test_is_deterministic(self):
        """Two builds must be byte-identical (repo determinism mandate)."""
        first = {e.name: e.text for e in standard_corpus()}
        second = {e.name: e.text for e in standard_corpus()}
        assert first == second

    def test_every_entry_records_the_defect_it_caught(self):
        """Admission rule: no decorative corpus entries."""
        for entry in standard_corpus():
            assert entry.text, f"{entry.name} is empty"
            assert (
                len(entry.caught) > 30
            ), f"{entry.name} must record which defect earned its place"

    def test_names_are_unique(self):
        names = [e.name for e in standard_corpus()]
        assert len(names) == len(set(names))
