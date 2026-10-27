"""Eval gates for TD-CHUNK-3 items 4 and 5, and the honest limit of the rubric.

Per the evals mandate, a ranked surface needs a rubric rather than an
assertion. The finding that shapes this file is that the rubric can gate one of
these two capabilities and **structurally cannot gate the other** -- and saying
so is more useful than inventing a proxy that would gate it badly.

``conformance/evals.py`` is deliberately model-independent: every metric scores
what the chunker decided, so a red gate is always attributable to chunking.
That is exactly why it can see small-to-big (which changes which spans are
returned) and cannot see context enrichment (which changes only the text sent
to a model, leaving every span identical). A rubric that could see enrichment
would be a rubric whose failures could not be attributed -- the property the
harness was built to avoid.
"""

from __future__ import annotations

import pytest

from proximadb_sdk.chunking_strategies.base import ChunkingConfig, ChunkingStrategy
from proximadb_sdk.chunking_strategies.capabilities import (
    ContextEnrichmentPass,
    HeadingPathPass,
    ParentLinkagePass,
)
from proximadb_sdk.chunking_strategies.conformance.corpus import by_name
from proximadb_sdk.chunking_strategies.conformance.evals import STANDARD_CASES, run_eval
from proximadb_sdk.chunking_strategies.passes import (
    CHUNK_TYPE_PARENT,
    PARENT_ID_KEY,
    run_passes,
)
from proximadb_sdk.chunking_strategies.recursive import RecursiveStrategy

#: Tight on purpose. At the default 512 every corpus answer fits in one chunk
#: and every configuration scores 1.00, so the rubric measures nothing --
#: recorded during TD-CHUNK-3 item 2 and guarded here by construction.
BUDGET = 220


def _partition(text: str, passes):
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=BUDGET,
        chunk_overlap=0,
        min_chunk_size=1,
        max_chunk_size=BUDGET * 4,
    )
    chunks = RecursiveStrategy(config).chunk(text, "d")
    return run_passes(text, list(chunks), passes)


def _is_parent(chunk) -> bool:
    return (getattr(chunk, "metadata", None) or {}).get(
        "chunk_type"
    ) == CHUNK_TYPE_PARENT


def _children(window: int):
    def chunk(text):
        result = _partition(
            text, [HeadingPathPass(), ParentLinkagePass(parent_window=window)]
        )
        return [c for c in result.chunks if not _is_parent(c)]

    return chunk


def _returned(window: int):
    """What a retriever actually hands back: the parent if linked, else the child.

    Scoring the parent set ALONE reads 0.38 and looks like a regression. It is
    not: parents are deliberately not a total partition, because a lone child
    gets none -- a parent identical to its only child is pure storage. So the
    returned unit is the union, and measuring anything else measures an
    artefact of the construction.
    """

    def chunk(text):
        result = _partition(
            text, [HeadingPathPass(), ParentLinkagePass(parent_window=window)]
        )
        parents = {c.chunk_id: c for c in result.chunks if _is_parent(c)}
        out, seen = [], set()
        for child in result.chunks:
            if _is_parent(child):
                continue
            parent_id = (child.metadata or {}).get(PARENT_ID_KEY)
            if parent_id and parent_id in parents:
                if parent_id not in seen:
                    seen.add(parent_id)
                    out.append(parents[parent_id])
            else:
                out.append(child)
        return out

    return chunk


# --- item 5: small-to-big IS gateable, and these are the gates --------------


def test_small_to_big_returns_more_complete_answers_than_it_embeds():
    """The capability's entire claim, as one inequality.

    Children stay sharp (what is embedded), the returned unit gets complete
    (what is read). If these ever converge, small-to-big is costing KSU for
    nothing and should be turned off.
    """
    children = run_eval("children", _children(BUDGET * 2))
    returned = run_eval("returned", _returned(BUDGET * 2))
    assert returned.containment > children.containment
    assert (children.containment, returned.containment) == (0.75, 0.875)


def test_a_bigger_parent_window_costs_the_same_and_buys_less():
    """Why the default is 2x the child budget and not a round number.

    KSU is flat across windows -- the parent layer is a second copy of the
    covered text either way -- so a larger window is not a cost/benefit trade
    at all. It is strictly worse: same storage, lower density.
    """
    tight = run_eval("tight", _returned(BUDGET * 2))
    loose = run_eval("loose", _returned(BUDGET * 8))
    assert tight.containment == loose.containment
    assert tight.mean_density > loose.mean_density


def test_parent_layer_ksu_is_bounded_and_recorded():
    """~1.74x, measured. Quoted so enabling this is a decision, not a hope."""
    base = extra = 0
    for case in STANDARD_CASES:
        text = by_name(case.corpus).text
        result = _partition(
            text, [HeadingPathPass(), ParentLinkagePass(parent_window=BUDGET * 2)]
        )
        base += result.span_chars
        extra += sum(len(c.text) for c in result.chunks if _is_parent(c))
    multiplier = 1 + extra / base
    assert 1.6 < multiplier < 1.9, multiplier


# --- item 4: enrichment is NOT gateable here, and that is the finding -------


def test_enrichment_is_invisible_to_this_rubric_by_construction():
    """Not a gap to fill with a proxy metric -- a boundary to state.

    Enrichment declares face=EMBEDDED, so every span, every text and the chunk
    count are identical before and after. Every metric in this rubric scores
    exactly those, so it MUST report no change. A rubric that did react would
    have to score a model's output, and then a red gate could no longer be
    attributed to chunking -- which is the property the harness exists to
    protect. Enrichment's gate belongs in the ANN/recall harnesses, where the
    model is the thing under test.
    """
    plain = run_eval(
        "plain",
        lambda text: [
            c for c in _partition(text, [HeadingPathPass()]).chunks if not _is_parent(c)
        ],
    )
    enriched = run_eval(
        "enriched",
        lambda text: [
            c
            for c in _partition(
                text, [HeadingPathPass(), ContextEnrichmentPass()]
            ).chunks
            if not _is_parent(c)
        ],
    )
    assert enriched.containment == plain.containment
    assert enriched.mean_density == pytest.approx(plain.mean_density)
    assert enriched.structural_integrity == plain.structural_integrity


def test_enrichment_keu_tax_is_measured_per_document_not_assumed():
    """What CAN be gated here: the cost side of the ratio.

    Recall per KEU spent needs both halves. This harness owns the denominator
    and says so, rather than reporting a recall number it cannot attribute.
    """
    taxes = []
    for case in STANDARD_CASES:
        text = by_name(case.corpus).text
        result = _partition(text, [HeadingPathPass(), ContextEnrichmentPass()])
        taxes.append(result.enrichment_tax)
    worst = max(taxes)
    assert worst > 1.0, "enrichment that costs nothing enriched nothing"
    # Structural context is short by construction; a tax above ~1.2 would mean
    # the heading paths have grown into something that should be truncated.
    assert worst < 1.2, worst
