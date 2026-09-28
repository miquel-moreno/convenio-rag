from pathlib import Path

import httpx
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import (
    Base,
    ChunkRecord,
    fulltext_search,
    include_object,
    vector_search,
)
from convenio_rag.adapters.embeddings import FakeEmbedder
from convenio_rag.api.dependencies import get_embedder, get_session
from convenio_rag.main import create_app
from convenio_rag.services.ingest import CatalogEntry, ingest_agreement
from convenio_rag.services.search import SearchMode, search
from tests.pg.conftest import TEST_DATABASE_URL

DATA = Path(__file__).resolve().parents[2] / "data" / "boe"
CONSULTANCY = CatalogEntry("BOE-A-2025-7766", "Consultoría y TI (XIX)", "Consultoría")
METAL = CatalogEntry("BOE-A-2022-479", "Metal (IV)", "Metal")


async def load(session: AsyncSession, *entries: CatalogEntry) -> FakeEmbedder:
    embedder = FakeEmbedder()
    for entry in entries:
        xml = (DATA / f"{entry.boe_id}.xml").read_bytes()
        await ingest_agreement(session, entry, xml, embedder=embedder)
    return embedder


def sync_url() -> str:
    return TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://")


def test_migrations_create_exactly_the_orm_schema(migrated: None) -> None:
    engine = create_engine(sync_url())
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"include_object": include_object})
        diff = compare_metadata(context, Base.metadata)
        indexes = {i["name"] for i in inspect(conn).get_indexes("chunks")}
    engine.dispose()

    assert diff == [], "the ORM models changed without a migration"
    assert {"ix_chunks_fts", "ix_chunks_embedding"} <= indexes


async def test_fulltext_search_uses_spanish_stemming(pg_session: AsyncSession) -> None:
    await load(pg_session, CONSULTANCY)

    # "vacación" (singular) must match "vacaciones".
    hits = await fulltext_search(pg_session, "¿cuántos días de vacación tengo?", limit=5)

    refs = [
        await pg_session.scalar(select(ChunkRecord.ref).where(ChunkRecord.id == h.chunk_id))
        for h in hits
    ]
    assert "Artículo 21" in refs
    assert hits == sorted(hits, key=lambda h: -h.score)


async def test_fulltext_search_ignores_questions_without_words(pg_session: AsyncSession) -> None:
    await load(pg_session, CONSULTANCY)

    assert await fulltext_search(pg_session, "¿?!", limit=5) == []


async def test_vector_search_returns_nearest_chunks(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session, CONSULTANCY)
    [vector] = await embedder.embed(["Vacaciones veintitrés días laborables"])

    hits = await vector_search(pg_session, vector, limit=3)

    assert len(hits) == 3
    assert hits[0].score >= hits[-1].score


@pytest.mark.parametrize("mode", list(SearchMode))
async def test_search_modes_return_one_result_per_article(
    pg_session: AsyncSession, mode: SearchMode
) -> None:
    embedder = await load(pg_session, CONSULTANCY)

    results = await search(
        pg_session, embedder, "vacaciones anuales retribuidas", limit=5, mode=mode
    )

    assert 0 < len(results) <= 5
    keys = [(r.agreement_id, r.ref) for r in results]
    assert len(keys) == len(set(keys)), "the same article appears twice"
    assert results[0].ref == "Artículo 21"
    assert all(set(r.ranks) <= {"text", "vector"} for r in results)


async def test_hybrid_search_combines_both_rankings(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session, CONSULTANCY)

    [top, *_] = await search(pg_session, embedder, "vacaciones anuales retribuidas")

    assert set(top.ranks) == {"text", "vector"}
    assert top.source_url == "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-7766"


async def test_search_can_be_limited_to_one_agreement(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session, CONSULTANCY, METAL)

    results = await search(
        pg_session, embedder, "jornada de trabajo", limit=10, agreement_id="BOE-A-2022-479"
    )

    assert results and {r.agreement_id for r in results} == {"BOE-A-2022-479"}


async def test_search_endpoint(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session, CONSULTANCY)
    app = create_app()
    app.dependency_overrides[get_session] = lambda: pg_session
    app.dependency_overrides[get_embedder] = lambda: embedder

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        ok = await client.get("/search", params={"q": "vacaciones anuales", "limit": 3})
        bad = await client.get("/search", params={"q": "x"})

    assert ok.status_code == 200
    body = ok.json()
    assert len(body) == 3 and body[0]["ref"] == "Artículo 21"
    assert body[0]["agreement"] == "Consultoría y TI (XIX)"
    assert bad.status_code == 422  # question too short
