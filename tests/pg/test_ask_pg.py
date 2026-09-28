import json
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.embeddings import FakeEmbedder
from convenio_rag.adapters.llm import FakeLLMClient, LLMError
from convenio_rag.api.dependencies import get_embedder, get_llm, get_session
from convenio_rag.main import create_app
from convenio_rag.services.answer import NOT_FOUND, ask
from convenio_rag.services.ingest import CatalogEntry, ingest_agreement

DATA = Path(__file__).resolve().parents[2] / "data" / "boe"
CONSULTANCY = CatalogEntry("BOE-A-2025-7766", "Consultoría y TI (XIX)", "Consultoría")
QUESTION = "¿Cuántos días de vacaciones anuales retribuidas tengo?"


async def load(session: AsyncSession) -> FakeEmbedder:
    embedder = FakeEmbedder()
    xml = (DATA / f"{CONSULTANCY.boe_id}.xml").read_bytes()
    await ingest_agreement(session, CONSULTANCY, xml, embedder=embedder)
    return embedder


def answer(found: bool, text: str, citations: list[int]) -> str:
    return json.dumps({"found": found, "answer": text, "citations": citations})


async def test_answer_cites_the_retrieved_article(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)
    llm = FakeLLMClient([answer(True, "Tienes 23 días laborables.", [1])])

    result = await ask(QUESTION, session=pg_session, embedder=embedder, llm=llm)

    assert result.found and result.answer == "Tienes 23 días laborables."
    assert [c.ref for c in result.citations] == ["Artículo 21"]
    assert result.citations[0] == result.sources[0]
    assert len(result.sources) == 5
    # The LLM received the full article text, with the source numbered.
    prompt = llm.calls[0][-1].content
    assert "[1] Consultoría y TI (XIX) · Artículo 21. Vacaciones" in prompt
    assert "veintitrés (23) días laborables" in prompt


async def test_invented_citation_means_no_answer(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)
    llm = FakeLLMClient([answer(True, "Tienes 30 días.", [9])])

    result = await ask(QUESTION, session=pg_session, embedder=embedder, llm=llm)

    assert not result.found
    assert result.answer == NOT_FOUND
    assert result.citations == [] and result.dropped_citations == [9]


async def test_invalid_output_is_retried_once(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)
    llm = FakeLLMClient(["not json", answer(True, "23 días.", [1])])

    result = await ask(QUESTION, session=pg_session, embedder=embedder, llm=llm)

    assert result.found and len(llm.calls) == 2
    assert (result.input_tokens, result.output_tokens) == (200, 100)


async def test_two_invalid_outputs_give_not_found(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)

    result = await ask(
        QUESTION, session=pg_session, embedder=embedder, llm=FakeLLMClient(["x", "y"])
    )

    assert not result.found and result.answer == NOT_FOUND


async def test_ask_endpoint(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)
    app = create_app()
    app.dependency_overrides[get_session] = lambda: pg_session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: FakeLLMClient(
        [answer(True, "Tienes 23 días laborables.", [1])]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        ok = await client.post("/ask", json={"question": QUESTION})
        short = await client.post("/ask", json={"question": "¿?"})

    assert ok.status_code == 200
    body = ok.json()
    assert body["found"] is True
    assert body["citations"][0]["ref"] == "Artículo 21"
    assert body["citations"][0]["source_url"].startswith("https://www.boe.es/")
    assert short.status_code == 422


async def test_ask_endpoint_returns_503_when_the_llm_is_down(pg_session: AsyncSession) -> None:
    embedder = await load(pg_session)

    class DownLLM:
        async def chat(self, *args: object, **kwargs: object) -> None:
            raise LLMError("connection refused")

    app = create_app()
    app.dependency_overrides[get_session] = lambda: pg_session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: DownLLM()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
