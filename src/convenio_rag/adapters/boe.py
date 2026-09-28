"""Download dispositions from the BOE open data service (official XML)."""

import re

import httpx

XML_URL = "https://www.boe.es/diario_boe/xml.php?id={boe_id}"
_BOE_ID_RE = re.compile(r"^BOE-A-\d{4}-\d{1,6}$")


class BoeError(Exception):
    """The BOE could not be reached or did not return the disposition."""


async def fetch_boe_xml(
    boe_id: str, *, timeout: float = 30.0, transport: httpx.AsyncBaseTransport | None = None
) -> bytes:
    if not _BOE_ID_RE.match(boe_id):
        raise ValueError(f"not a BOE id: {boe_id!r}")
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.get(
                XML_URL.format(boe_id=boe_id), headers={"User-Agent": "convenio-rag"}
            )
    except httpx.HTTPError as exc:
        raise BoeError(f"could not reach the BOE: {exc!r}") from exc
    if response.status_code != httpx.codes.OK or b"<documento" not in response.content[:500]:
        raise BoeError(f"BOE answered HTTP {response.status_code} for {boe_id}")
    return response.content
