from pathlib import Path

import httpx
import pytest
from scripts import ingest as ingest_script
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from convenio_rag.adapters.boe import BoeError, fetch_boe_xml
from convenio_rag.adapters.db import AgreementRecord, Base, list_chunks
from convenio_rag.core.config import get_settings
from convenio_rag.services.ingest import CatalogEntry, ingest_agreement, load_catalog

DATA = Path(__file__).resolve().parents[2] / "data" / "boe"
CONSULTANCY_XML = (DATA / "BOE-A-2025-7766.xml").read_bytes()
ENTRY = CatalogEntry("BOE-A-2025-7766", "Consultoría y TI (XIX)", "Consultoría", "nota")


# --- BOE download --------------------------------------------------------------


async def test_fetch_returns_the_xml_of_the_disposition() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=CONSULTANCY_XML)

    xml = await fetch_boe_xml("BOE-A-2025-7766", transport=httpx.MockTransport(handler))

    assert xml == CONSULTANCY_XML
    assert seen == ["https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-7766"]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(404), httpx.Response(200, content=b"<html>Documento no encontrado</html>")],
)
async def test_fetch_raises_when_the_boe_does_not_return_it(response: httpx.Response) -> None:
    with pytest.raises(BoeError):
        await fetch_boe_xml("BOE-A-2025-7766", transport=httpx.MockTransport(lambda _: response))


async def test_fetch_raises_on_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(BoeError):
        await fetch_boe_xml("BOE-A-2025-7766", transport=httpx.MockTransport(handler))


async def test_fetch_rejects_ids_that_are_not_boe_ids() -> None:
    with pytest.raises(ValueError):
        await fetch_boe_xml("../etc/passwd")


# --- ingestion -----------------------------------------------------------------


def test_catalog_lists_the_two_agreements() -> None:
    catalog = load_catalog(DATA / "catalog.json")

    assert [e.boe_id for e in catalog] == ["BOE-A-2025-7766", "BOE-A-2022-479"]
    assert all((DATA / f"{e.boe_id}.xml").exists() for e in catalog)


async def test_ingest_stores_the_agreement_and_its_chunks(session: AsyncSession) -> None:
    count = await ingest_agreement(session, ENTRY, CONSULTANCY_XML)

    agreement = await session.get(AgreementRecord, "BOE-A-2025-7766")
    chunks = await list_chunks(session, "BOE-A-2025-7766")
    assert agreement is not None and agreement.short_name == "Consultoría y TI (XIX)"
    assert agreement.source_url == "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-7766"
    assert len(chunks) == count > 100
    assert chunks[0].position == 0


async def test_ingesting_twice_does_not_duplicate(session: AsyncSession) -> None:
    first = await ingest_agreement(session, ENTRY, CONSULTANCY_XML)
    second = await ingest_agreement(session, ENTRY, CONSULTANCY_XML)

    assert first == second == len(await list_chunks(session, "BOE-A-2025-7766"))


async def test_ingest_refuses_an_xml_that_does_not_match_the_catalog(
    session: AsyncSession,
) -> None:
    wrong = CatalogEntry("BOE-A-2022-479", "Metal (IV)", "Metal")

    with pytest.raises(ValueError, match="catalog expects"):
        await ingest_agreement(session, wrong, CONSULTANCY_XML)


async def test_ingest_script_loads_the_catalog_from_saved_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "boe"
    data_dir.mkdir()
    (data_dir / "catalog.json").write_text(
        '[{"boe_id": "BOE-A-2025-7766", "short_name": "C", "sector": "S"}]', encoding="utf-8"
    )
    (data_dir / "BOE-A-2025-7766.xml").write_bytes(CONSULTANCY_XML)
    url = f"sqlite+aiosqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        counts = await ingest_script.run(refresh=False, data_dir=data_dir)
    finally:
        get_settings.cache_clear()

    assert list(counts) == ["BOE-A-2025-7766"] and counts["BOE-A-2025-7766"] > 100

    # A second start of the container with --if-empty does not load again.
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        again = await ingest_script.run(refresh=False, data_dir=data_dir, if_empty=True)
    finally:
        get_settings.cache_clear()
    assert again == {}
