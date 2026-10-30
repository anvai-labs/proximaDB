"""Chunking conformance kit — the executable form of ADR-091's specification.

Shipped inside the package rather than kept in ``tests/`` on purpose: ADR-091 D4
makes the invariant suite and the corpus *part of the contract*, and the whole
point is that every consumer runs the same bed. anvaiops's connector SDK already
calls ``proximadb_sdk.chunking.TextChunker`` directly, and victor-rag holds an
independent implementation that reached the same defect — neither can be held to a
specification it cannot import.

Usage::

    from proximadb_sdk.chunking_strategies.conformance import (
        Invariant, standard_corpus, check_totality, compute_trace,
    )

    for entry in standard_corpus():
        chunks = my_chunker(entry.text)
        if (v := check_totality(entry.text, chunks)) is not None:
            print(entry.name, v)

Nothing here imports a chunking strategy, a tokenizer, or any heavy dependency —
the checks are structural and operate on anything exposing ``text``,
``start_pos`` and ``end_pos``, so the kit stays usable from a foreign codebase.

Tracked by TD-CHUNK-1; the defects it measures are ADR-091's evidence base.
"""

from ..measures import CHAR_MEASURE, CharMeasure, Measure, TokenMeasure
from .corpus import CorpusEntry, by_name, scale_corpus, standard_corpus
from .evals import STANDARD_CASES, CaseScore, EvalCase, EvalReport, run_eval, score_case
from .golden import (
    DIGEST_RECIPE_VERSION,
    case_digest,
    case_key,
    diff_digests,
    load_golden,
    render_golden,
    sweep_digests,
)
from .invariants import (
    ALL_INVARIANTS,
    Invariant,
    Violation,
    check_cap,
    check_config_safety,
    check_exactness,
    check_idempotence,
    check_no_containment,
    check_non_empty,
    check_stream_equivalence,
    check_totality,
)
from .runner import ChunkerAdapter, Evaluation, evaluate, evaluate_all, format_baseline
from .trace import BASIS_BYTE, BASIS_CHAR, ChunkTrace, Gap, compute_trace

__all__ = [
    # Measures (re-exported so a consumer needs one import)
    "Measure",
    "CharMeasure",
    "CHAR_MEASURE",
    "TokenMeasure",
    # Evals (boundary QUALITY, rubric-scored)
    "EvalCase",
    "CaseScore",
    "EvalReport",
    "STANDARD_CASES",
    "run_eval",
    "score_case",
    # Corpus
    "CorpusEntry",
    "standard_corpus",
    "scale_corpus",
    "by_name",
    # Invariants
    "Invariant",
    "Violation",
    "ALL_INVARIANTS",
    "check_totality",
    "check_exactness",
    "check_cap",
    "check_non_empty",
    "check_no_containment",
    "check_stream_equivalence",
    "check_config_safety",
    "check_idempotence",
    # Golden output oracle
    "case_digest",
    "case_key",
    "sweep_digests",
    "render_golden",
    "load_golden",
    "diff_digests",
    "DIGEST_RECIPE_VERSION",
    # Runner
    "ChunkerAdapter",
    "Evaluation",
    "evaluate",
    "evaluate_all",
    "format_baseline",
    # Trace
    "ChunkTrace",
    "Gap",
    "compute_trace",
    "BASIS_CHAR",
    "BASIS_BYTE",
]
