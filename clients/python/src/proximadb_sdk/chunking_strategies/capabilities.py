"""The four post-partition capabilities of TD-CHUNK-3, as :mod:`passes` passes.

Each one is a thin adapter over logic that lives (or now lives) in its own
module, so the capability and its transport into the pipeline stay separable:

============================  =========  ===========================  =========
Pass                          Face       Implementation               Cost
============================  =========  ===========================  =========
:class:`HeadingPathPass`      METADATA   ``boundaries.annotate_...``  free
:class:`DedupPass`            SELECTION  ``dedup.deduplicate``        KEU,KSU v
:class:`ContextEnrichment...` EMBEDDED   here                          KEU ^
:class:`ParentLinkagePass`    RETRIEVED  here                          KSU ^
============================  =========  ===========================  =========

The first two already shipped as boolean flags inlined in
``TextDocumentProcessor.chunk``; they are retrofitted rather than rewritten, so
the retrofit is provably behaviour-neutral against the golden snapshot, and the
two new capabilities land on a seam that already has two working occupants
instead of being its first and only users.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .passes import (
    CHUNK_TYPE_PARENT,
    EMBEDDED_TEXT_KEY,
    PARENT_ID_KEY,
    ChunkEdge,
    Face,
    PassResult,
)


@dataclass
class HeadingPathPass:
    """Label each chunk with the heading path it lives under (TD-CHUNK-3 item 1).

    Pure metadata, so no KEU or KSU delta -- strictly free retrieval quality,
    which is why it is the one capability that defaults ON.
    """

    respect_code_blocks: bool = True
    name: str = "heading_path"
    face: Face = Face.METADATA

    def apply(self, document: str, chunks: list[Any]) -> PassResult:
        from .boundaries import annotate_heading_paths

        annotate_heading_paths(
            document, chunks, respect_code_blocks=self.respect_code_blocks
        )
        labelled = sum(
            1
            for c in chunks
            if (getattr(c, "metadata", None) or {}).get("heading_path")
        )
        return PassResult(
            chunks=chunks, stats={"labelled": labelled, "total": len(chunks)}
        )


@dataclass
class DedupPass:
    """Drop near-duplicate chunks before they are embedded (TD-CHUNK-3 item 2).

    The only capability that REDUCES cost. Lexical by necessity, not by
    convenience: to skip paying KEU for a chunk you must decide before
    embedding it, so an embedding-based detector has already spent the money it
    exists to save.
    """

    threshold: float = 0.9
    shingle_size: int = 5
    name: str = "dedup"
    face: Face = Face.SELECTION

    def apply(self, document: str, chunks: list[Any]) -> PassResult:
        from .dedup import deduplicate

        result = deduplicate(
            list(chunks), threshold=self.threshold, shingle_size=self.shingle_size
        )
        return PassResult(
            chunks=list(result.kept),
            stats={
                "before": len(chunks),
                "after": len(result.kept),
                "removed": len(chunks) - len(result.kept),
            },
        )


# ---------------------------------------------------------------------------
# TD-CHUNK-3 item 4 — context enrichment
# ---------------------------------------------------------------------------


def structural_context(chunk: Any, *, source_title: str | None = None) -> str:
    """The context line for a chunk, from metadata that already exists.

    Deliberately **structural and deterministic**, not generated. Generated
    per-chunk context is the well-known form of this idea and costs one model
    call per chunk at ingest -- which has to be justified against the retrieval
    it buys, on a corpus, before it is anyone's default. This version costs
    nothing to produce and consumes exactly what :class:`HeadingPathPass`
    already computed, which is why the two compose: METADATA runs before
    EMBEDDED precisely so this can read what that wrote.

    A generated variant is a different ``context_fn``, not a different pass.
    """
    parts: list[str] = []
    if source_title:
        parts.append(source_title)
    metadata = getattr(chunk, "metadata", None) or {}
    # `heading_path` is a LIST of ancestor titles, and deliberately so: the
    # annotator stores the structure and leaves rendering to whoever consumes
    # it, so a metadata filter can match one ancestor without string surgery.
    # This pass is a consumer, so joining is its job, not the annotator's.
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, str):
        heading_path = [heading_path] if heading_path else []
    if isinstance(heading_path, (list, tuple)):
        parts.extend(str(part) for part in heading_path if part)

    # Drop a part that repeats its predecessor. A document whose H1 IS its
    # title is the common case, and it produced "Guide > Guide" -- paying KEU
    # for a word already there. Only CONSECUTIVE repeats are dropped: a real
    # "Setup > Windows > Setup" is a distinct section and must survive.
    deduped: list[str] = []
    for part in parts:
        if not deduped or deduped[-1].casefold() != part.casefold():
            deduped.append(part)
    return " > ".join(deduped)


@dataclass
class ContextEnrichmentPass:
    """Prefix each chunk's EMBEDDED text with where it sits (TD-CHUNK-3 item 4).

    The span is untouched, so offsets keep describing the document exactly and
    a caller still gets the original text back. Only the vector changes.

    **This is the one capability that costs KEU**, which is why it is the only
    one whose stats include the multiplier it applied. TD-CHUNK-3 requires it
    to be gated on recall-per-KEU rather than recall alone; the pipeline's
    ``enrichment_tax`` is the denominator of that ratio, reported per document
    rather than assumed from the template.

    Default OFF, per the ship-default-OFF-until-baked rule: it changes what is
    stored for every chunk in the collection.
    """

    source_title: str | None = None
    separator: str = "\n\n"
    #: Skip enrichment when the context would exceed this fraction of the chunk.
    #: A short chunk under a long heading path is mostly context, and a vector
    #: dominated by a section title retrieves the section, not the chunk.
    max_context_ratio: float = 0.5
    name: str = "context_enrichment"
    face: Face = Face.EMBEDDED

    def apply(self, document: str, chunks: list[Any]) -> PassResult:
        enriched = 0
        skipped_ratio = 0
        added = 0
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            context = structural_context(chunk, source_title=self.source_title)
            if not context:
                continue
            body = getattr(chunk, "text", "")
            prefix = context + self.separator
            if body and len(prefix) > self.max_context_ratio * len(body):
                skipped_ratio += 1
                continue
            metadata[EMBEDDED_TEXT_KEY] = prefix + body
            enriched += 1
            added += len(prefix)
        return PassResult(
            chunks=chunks,
            stats={
                "enriched": enriched,
                "skipped_context_ratio": skipped_ratio,
                "added_chars": added,
            },
        )


# ---------------------------------------------------------------------------
# TD-CHUNK-3 item 5 — small-to-big / parent-document retrieval
# ---------------------------------------------------------------------------


@dataclass
class ParentLinkagePass:
    """Group children into parent spans and link them (TD-CHUNK-3 item 5).

    Small-to-big: embed precise child chunks so retrieval is sharp, then return
    the surrounding parent so generation has room. The child is what matches;
    the parent is what is read.

    **The parent is emitted once and referenced, never copied onto its
    children.** Copying is the obvious implementation and it multiplies KSU by
    the fan-out -- ~4x here -- which is the same cost mistake this program has
    now found in three separate places. Children carry ``parent_id``; the
    parent is a chunk in its own right, marked ``chunk_type="parent"`` so a
    caller can store it without embedding it, and the relationship is emitted
    as a :class:`~.passes.ChunkEdge` for whatever the caller uses to hold edges
    (an ORION edge in the server -- the linkage this database can offer that a
    pure vector store cannot).

    Parents are formed by merging **consecutive** children up to
    ``parent_window`` characters, never across a change of heading path. That
    guarantees the two properties small-to-big actually needs -- a parent
    contains its children's spans exactly, and a parent is itself a span of the
    document -- without a second chunking pass, and it reuses the heading
    metadata rather than re-deriving structure.
    """

    parent_window: int = 2048
    relation: str = "parent_of"
    respect_heading_path: bool = True
    name: str = "parent_linkage"
    face: Face = Face.RETRIEVED

    def _groups(self, chunks: Sequence[Any]) -> list[list[Any]]:
        groups: list[list[Any]] = []
        current: list[Any] = []
        current_path: Any = None
        for chunk in chunks:
            if (getattr(chunk, "metadata", None) or {}).get(
                "chunk_type"
            ) == CHUNK_TYPE_PARENT:
                continue
            path = (
                (getattr(chunk, "metadata", None) or {}).get("heading_path")
                if self.respect_heading_path
                else None
            )
            end = int(getattr(chunk, "end_pos", 0))
            if current:
                span = end - int(getattr(current[0], "start_pos", 0))
                # A section change ends a parent even under budget: a parent
                # spanning two sections answers as neither.
                if span > self.parent_window or path != current_path:
                    groups.append(current)
                    current = []
            if not current:
                current_path = path
            current.append(chunk)
        if current:
            groups.append(current)
        return groups

    def apply(self, document: str, chunks: list[Any]) -> PassResult:
        from .base import OFFSET_BASIS_CHAR, OFFSET_CONTRACT_EXACT, TextChunk

        parents: list[Any] = []
        edges: list[ChunkEdge] = []
        for index, group in enumerate(self._groups(chunks)):
            # A parent identical to its only child is pure overhead: it doubles
            # storage to return text the child already carries.
            if len(group) < 2:
                continue
            start = int(getattr(group[0], "start_pos", 0))
            end = int(getattr(group[-1], "end_pos", 0))
            source_id = (getattr(group[0], "metadata", None) or {}).get(
                "source_id", "doc"
            )
            parent_id = f"{source_id}_parent_{index}"
            parent = TextChunk(
                text=document[start:end],
                start_pos=start,
                end_pos=end,
                chunk_id=parent_id,
                metadata={
                    "chunk_type": CHUNK_TYPE_PARENT,
                    "source_id": source_id,
                    "child_count": len(group),
                    "offset_basis": OFFSET_BASIS_CHAR,
                    "offset_contract": OFFSET_CONTRACT_EXACT,
                    **(
                        {"heading_path": group[0].metadata["heading_path"]}
                        if (getattr(group[0], "metadata", None) or {}).get(
                            "heading_path"
                        )
                        else {}
                    ),
                },
            )
            parents.append(parent)
            for child in group:
                metadata = getattr(child, "metadata", None)
                if isinstance(metadata, dict):
                    metadata[PARENT_ID_KEY] = parent_id
                edges.append(
                    ChunkEdge(
                        source_id=parent_id,
                        target_id=str(getattr(child, "chunk_id", "")),
                        relation=self.relation,
                    )
                )

        return PassResult(
            chunks=list(chunks) + parents,
            edges=tuple(edges),
            stats={
                "parents": len(parents),
                "linked_children": len(edges),
                "parent_chars": sum(len(getattr(p, "text", "")) for p in parents),
            },
        )


@dataclass
class PassPipeline:
    """The passes a configuration asks for, assembled once.

    Exists so the resolution lives in one place instead of in every caller's
    ``chunk()``. ``TextDocumentProcessor`` builds one from its
    ``ProcessorConfig``; anvaiops or a test can build one directly.
    """

    passes: list[Any] = field(default_factory=list)

    @classmethod
    def from_processor_config(
        cls, config: Any, *, source_title: str | None = None
    ) -> PassPipeline:
        selected: list[Any] = []
        if getattr(config, "preserve_structure", False):
            selected.append(HeadingPathPass())
        if getattr(config, "deduplicate_chunks", False):
            selected.append(
                DedupPass(threshold=getattr(config, "deduplicate_threshold", 0.9))
            )
        if getattr(config, "enrich_context", False):
            selected.append(ContextEnrichmentPass(source_title=source_title))
        if getattr(config, "link_parent_chunks", False):
            window = getattr(config, "parent_window", None)
            if not window:
                # Twice the child budget: measured to capture the whole
                # containment gain, at a KSU cost that does not fall for
                # smaller windows or rise for larger ones.
                window = 2 * int(getattr(config, "chunk_size", 512) or 512)
            selected.append(ParentLinkagePass(parent_window=window))
        return cls(passes=selected)

    def run(self, document: str, chunks: list[Any]) -> Any:
        from .passes import run_passes

        return run_passes(document, chunks, self.passes)
