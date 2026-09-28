"""Load the agreements of the catalog into the database (idempotent)."""

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import (
    embedding_text,
    list_chunks,
    replace_agreement,
    set_embeddings,
)
from convenio_rag.adapters.embeddings import Embedder
from convenio_rag.services.agreements import parse_boe_xml


@dataclass(frozen=True)
class CatalogEntry:
    boe_id: str
    short_name: str
    sector: str
    note: str | None = None


def load_catalog(path: Path) -> list[CatalogEntry]:
    return [CatalogEntry(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


EMBED_BATCH = 64


async def ingest_agreement(
    session: AsyncSession, entry: CatalogEntry, xml: bytes, *, embedder: Embedder | None = None
) -> int:
    """Parse the BOE XML and store it, replacing a previous load. Returns the chunk count.

    With an embedder, the chunks also get their embedding (for the semantic search).
    """
    parsed = parse_boe_xml(xml)
    if parsed.boe_id != entry.boe_id:
        raise ValueError(f"XML is {parsed.boe_id}, catalog expects {entry.boe_id}")
    count = await replace_agreement(
        session, parsed, short_name=entry.short_name, sector=entry.sector, note=entry.note
    )
    if embedder is not None:
        chunks = list(await list_chunks(session, entry.boe_id))
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = await embedder.embed([embedding_text(c) for c in batch])
            await set_embeddings(session, [c.id for c in batch], vectors)
    return count
