"""Load the agreements of the catalog into the database (idempotent)."""

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import replace_agreement
from convenio_rag.services.agreements import parse_boe_xml


@dataclass(frozen=True)
class CatalogEntry:
    boe_id: str
    short_name: str
    sector: str
    note: str | None = None


def load_catalog(path: Path) -> list[CatalogEntry]:
    return [CatalogEntry(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


async def ingest_agreement(session: AsyncSession, entry: CatalogEntry, xml: bytes) -> int:
    """Parse the BOE XML and store it, replacing a previous load. Returns the chunk count."""
    parsed = parse_boe_xml(xml)
    if parsed.boe_id != entry.boe_id:
        raise ValueError(f"XML is {parsed.boe_id}, catalog expects {entry.boe_id}")
    return await replace_agreement(
        session, parsed, short_name=entry.short_name, sector=entry.sector, note=entry.note
    )
