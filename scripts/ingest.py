"""Load the agreements listed in data/boe/catalog.json into the database.

    uv run python -m scripts.ingest              # uses the XML files saved in data/boe/
    uv run python -m scripts.ingest --refresh    # downloads them again from the BOE first

Running it again replaces the stored version: nothing is duplicated. It also computes
the embeddings with the local model (--no-embed to skip them).
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import func, select

from convenio_rag.adapters.boe import fetch_boe_xml
from convenio_rag.adapters.db import AgreementRecord, make_engine, make_session_factory
from convenio_rag.adapters.embeddings import Embedder, FastEmbedEmbedder
from convenio_rag.core.config import get_settings
from convenio_rag.services.ingest import ingest_agreement, load_catalog

DATA = Path(__file__).resolve().parent.parent / "data" / "boe"


async def run(
    *,
    refresh: bool,
    data_dir: Path = DATA,
    embedder: Embedder | None = None,
    if_empty: bool = False,
) -> dict[str, int]:
    engine = make_engine(get_settings().database_url)
    counts: dict[str, int] = {}
    try:
        async with make_session_factory(engine)() as session:
            stored = await session.scalar(select(func.count()).select_from(AgreementRecord))
            if if_empty and stored:
                print(f"{stored} agreements already loaded: nothing to do")
                return counts
            for entry in load_catalog(data_dir / "catalog.json"):
                path = data_dir / f"{entry.boe_id}.xml"
                if refresh or not path.exists():
                    path.write_bytes(await fetch_boe_xml(entry.boe_id))
                counts[entry.boe_id] = await ingest_agreement(
                    session, entry, path.read_bytes(), embedder=embedder
                )
                print(f"{entry.boe_id} · {entry.short_name}: {counts[entry.boe_id]} chunks")
    finally:
        await engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--refresh", action="store_true", help="download the XML again")
    parser.add_argument("--no-embed", action="store_true", help="skip the embeddings")
    parser.add_argument(
        "--if-empty", action="store_true", help="only load when no agreement is stored yet"
    )
    args = parser.parse_args()
    settings = get_settings()
    embedder = None
    if not args.no_embed:
        embedder = FastEmbedEmbedder(
            settings.embedding_model, cache_dir=settings.embedding_cache_dir
        )
    asyncio.run(run(refresh=args.refresh, embedder=embedder, if_empty=args.if_empty))


if __name__ == "__main__":
    main()
