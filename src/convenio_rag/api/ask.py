"""POST /ask: a short answer with the articles it comes from, or an honest 'not found'."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.embeddings import Embedder
from convenio_rag.adapters.llm import LLMClient, LLMError
from convenio_rag.api.dependencies import get_embedder, get_llm, get_session
from convenio_rag.core.errors import ServiceUnavailableError
from convenio_rag.services.answer import ask

router = APIRouter(tags=["ask"])


class AskIn(BaseModel):
    question: str = Field(
        min_length=3, max_length=500, examples=["¿Cuántos días de vacaciones tengo?"]
    )
    agreement: str | None = Field(
        default=None, description="Limit to one agreement (BOE id, e.g. BOE-A-2025-7766)"
    )


class CitationOut(BaseModel):
    agreement_id: str
    agreement: str
    ref: str
    title: str
    source_url: str


class AskOut(BaseModel):
    question: str
    found: bool
    answer: str
    citations: list[CitationOut]
    sources: list[CitationOut]
    model: str
    latency_ms: float


@router.post("/ask")
async def ask_question(
    body: AskIn,
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> AskOut:
    try:
        result = await ask(
            body.question,
            session=session,
            embedder=embedder,
            llm=llm,
            agreement_id=body.agreement,
        )
    except LLMError as exc:
        raise ServiceUnavailableError(f"the LLM provider is not available: {exc}") from exc
    return AskOut(
        question=result.question,
        found=result.found,
        answer=result.answer,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        sources=[CitationOut(**vars(c)) for c in result.sources],
        model=result.model,
        latency_ms=round(result.latency_ms, 1),
    )
