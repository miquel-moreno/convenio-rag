"""Hybrid search: Spanish full-text + semantic (pgvector), fused with RRF.

Reciprocal Rank Fusion: each chunk scores sum(1 / (k + rank)) over the rankings it
appears in. It only needs the positions, not comparable scores, so a BM25-like
text rank and a cosine similarity can be combined without tuning weights.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import (
    fulltext_search,
    get_chunks_with_agreements,
    vector_search,
)
from convenio_rag.adapters.embeddings import Embedder

RRF_K = 60  # the usual constant from the original RRF paper
CANDIDATES_PER_SEARCH = 20


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    TEXT = "text"


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    agreement_id: str
    agreement: str  # short name, e.g. "Metal (IV)"
    ref: str
    title: str
    part: int
    text: str
    source_url: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # position in each search, 1-based


def rrf(rankings: Mapping[str, Sequence[int]], *, k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse several rankings of ids. Returns (id, score), best first; ties by id."""
    scores: dict[int, float] = {}
    for ranking in rankings.values():
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


async def search(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    *,
    limit: int = 5,
    mode: SearchMode = SearchMode.HYBRID,
    agreement_id: str | None = None,
    candidates: int = CANDIDATES_PER_SEARCH,
) -> list[SearchResult]:
    rankings: dict[str, list[int]] = {}
    if mode in (SearchMode.HYBRID, SearchMode.TEXT):
        hits = await fulltext_search(session, question, limit=candidates, agreement_id=agreement_id)
        rankings["text"] = [h.chunk_id for h in hits]
    if mode in (SearchMode.HYBRID, SearchMode.VECTOR):
        [vector] = await embedder.embed([question])
        hits = await vector_search(session, vector, limit=candidates, agreement_id=agreement_id)
        rankings["vector"] = [h.chunk_id for h in hits]

    fused = rrf(rankings)
    rows = await get_chunks_with_agreements(session, [chunk_id for chunk_id, _ in fused])
    results: list[SearchResult] = []
    seen: set[tuple[str, str]] = set()
    for chunk_id, score in fused:
        chunk, agreement = rows[chunk_id]
        # One result per article: the parts of a long article would fill the top otherwise.
        key = (agreement.boe_id, chunk.ref)
        if key in seen:
            continue
        seen.add(key)
        if len(results) == limit:
            break
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                agreement_id=agreement.boe_id,
                agreement=agreement.short_name,
                ref=chunk.ref,
                title=chunk.title,
                part=chunk.part,
                text=chunk.text,
                source_url=agreement.source_url,
                score=round(score, 6),
                ranks={
                    name: ranking.index(chunk_id) + 1
                    for name, ranking in rankings.items()
                    if chunk_id in ranking
                },
            )
        )
    return results
