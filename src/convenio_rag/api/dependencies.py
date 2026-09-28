"""FastAPI dependencies. Tests override them to use SQLite and a fake LLM."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.embeddings import Embedder, FastEmbedEmbedder
from convenio_rag.adapters.llm import LLMClient, LLMError, build_llm_client
from convenio_rag.core.config import get_settings
from convenio_rag.core.errors import ServiceUnavailableError


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


def get_llm() -> LLMClient:
    # Built per request (it is cheap) so a missing key is a clear 503, not a crash at startup.
    try:
        return build_llm_client(get_settings())
    except LLMError as exc:
        raise ServiceUnavailableError(str(exc)) from exc


def get_embedder(request: Request) -> Embedder:
    # One model per process: loading it takes seconds, embedding a question milliseconds.
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        settings = get_settings()
        embedder = FastEmbedEmbedder(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
        request.app.state.embedder = embedder
    return embedder
