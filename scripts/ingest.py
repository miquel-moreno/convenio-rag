"""Load the agreements listed in data/boe/catalog.json into the database.

    uv run python -m scripts.ingest              # uses the XML files saved in data/boe/
    uv run python -m scripts.ingest --refresh    # downloads them again from the BOE first

Running it again replaces the stored version: nothing is duplicated.
"""

import argparse
import asyncio
from pathlib import Path

from convenio_rag.adapters.boe import fetch_boe_xml
from convenio_rag.adapters.db import make_engine, make_session_factory
from convenio_rag.core.config import get_settings
from convenio_rag.services.ingest import ingest_agreement, load_catalog

DATA = Path(__file__).resolve().parent.parent / "data" / "boe"


async def run(*, refresh: bool, data_dir: Path = DATA) -> dict[str, int]:
    engine = make_engine(get_settings().database_url)
    counts: dict[str, int] = {}
    try:
        async with make_session_factory(engine)() as session:
            for entry in load_catalog(data_dir / "catalog.json"):
                path = data_dir / f"{entry.boe_id}.xml"
                if refresh or not path.exists():
                    path.write_bytes(await fetch_boe_xml(entry.boe_id))
                counts[entry.boe_id] = await ingest_agreement(session, entry, path.read_bytes())
                print(f"{entry.boe_id} · {entry.short_name}: {counts[entry.boe_id]} chunks")
    finally:
        await engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--refresh", action="store_true", help="download the XML again")
    args = parser.parse_args()
    asyncio.run(run(refresh=args.refresh))


if __name__ == "__main__":
    main()
