"""GET /search: the retrieval step on its own (useful to inspect and evaluate it)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.embeddings import Embedder
from convenio_rag.api.dependencies import get_embedder, get_session
from convenio_rag.services.search import SearchMode, search

router = APIRouter(tags=["search"])


class SearchHit(BaseModel):
    agreement_id: str
    agreement: str
    ref: str
    title: str
    part: int
    text: str
    source_url: str
    score: float
    ranks: dict[str, int]


@router.get("/search")
async def search_chunks(
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    q: Annotated[str, Query(min_length=2, max_length=500, description="Question in Spanish")],
    mode: SearchMode = SearchMode.HYBRID,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    agreement: Annotated[str | None, Query(description="BOE id, e.g. BOE-A-2022-479")] = None,
) -> list[SearchHit]:
    results = await search(session, embedder, q, limit=limit, mode=mode, agreement_id=agreement)
    return [
        SearchHit(
            agreement_id=r.agreement_id,
            agreement=r.agreement,
            ref=r.ref,
            title=r.title,
            part=r.part,
            text=r.text,
            source_url=r.source_url,
            score=r.score,
            ranks=r.ranks,
        )
        for r in results
    ]
